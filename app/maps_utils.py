"""Navigation helpers.

Routing is handled by the project's accessibility Dijkstra graph loaded from
Firebase. Google Maps is kept only for geocoding text destinations and nearby
place search; Google Directions is not used.
"""

from __future__ import annotations

import os
import json
import re
import time
import urllib.parse
from typing import Any, Optional

import requests

from app.config import (
    firebase_api_key,
    firebase_edges_collection,
    firebase_environment_collections,
    firebase_nodes_collection,
    firebase_project_id,
    google_maps_api_key,
    local_edges_path,
    local_environment_points_path,
    local_nodes_path,
)
from app.safe_route.accessibility_dijkstra import (
    Node,
    SafetyPoint,
    explain_edge,
    find_accessible_route_from_data,
    firestore_value,
    haversine_m,
    load_nodes,
    load_safety_points,
    nearest_node,
    parse_safety_points,
)


_FIRESTORE_CACHE_TTL_SECONDS = int(os.environ.get("FIRESTORE_ROUTE_CACHE_TTL_SECONDS", "300"))
_MAX_SNAP_DISTANCE_M = float(os.environ.get("SAFE_ROUTE_MAX_SNAP_DISTANCE_M", "450"))
_WALKING_SPEED_M_PER_MIN = float(os.environ.get("SAFE_ROUTE_WALKING_SPEED_M_PER_MIN", "75"))
_route_data_cache: tuple[float, dict[str, Node], list[dict[str, Any]], list[SafetyPoint]] | None = None


def rough_origin_latlng() -> Optional[str]:
    """Fallback rough location when the Android client cannot provide GPS."""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5).json()
        loc = r.get("loc")
        if loc and "," in str(loc):
            return str(loc).strip()
    except OSError:
        pass
    return None


def _json_items(value: Any, array_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in array_keys:
            inner = value.get(key)
            if isinstance(inner, list):
                return [item for item in inner if isinstance(item, dict)]
        return [value]
    return []


def _parse_latlng(value: str) -> tuple[float, float] | None:
    if not value:
        return None
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", value)
    if not m:
        return None
    lat, lng = float(m.group(1)), float(m.group(2))
    if -90 <= lat <= 90 and -180 <= lng <= 180:
        return lat, lng
    return None


def geocode_address(address: str) -> tuple[Optional[float], Optional[float], str]:
    """Resolve a text destination to coordinates.

    This still uses Google Geocoding as a fallback because Dijkstra needs a
    destination coordinate before it can snap to the Firebase graph.
    """
    parsed = _parse_latlng(address.strip())
    if parsed:
        lat, lng = parsed
        return lat, lng, address

    key = google_maps_api_key()
    if not key:
        return None, None, address

    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "language": "zh-TW", "key": key}
    res = requests.get(url, params=params, timeout=15).json()
    if res.get("status") == "OK" and res.get("results"):
        loc = res["results"][0]["geometry"]["location"]
        formatted = res["results"][0].get("formatted_address") or address
        return loc["lat"], loc["lng"], formatted
    return None, None, address


def _firestore_documents(collection: str) -> list[dict[str, Any]]:
    project_id = firebase_project_id()
    if not project_id:
        raise RuntimeError("FIREBASE_PROJECT_ID is not configured")

    encoded_collection = urllib.parse.quote(collection, safe="/")
    url = (
        "https://firestore.googleapis.com/v1/projects/"
        f"{project_id}/databases/(default)/documents/{encoded_collection}"
    )
    params: dict[str, Any] = {"pageSize": 300}
    api_key = firebase_api_key()
    if api_key:
        params["key"] = api_key

    docs: list[dict[str, Any]] = []
    while True:
        data = requests.get(url, params=params, timeout=20).json()
        if "error" in data:
            message = data["error"].get("message", "Unknown Firestore error")
            raise RuntimeError(f"Firestore read failed for {collection}: {message}")
        docs.extend(data.get("documents", []))
        token = data.get("nextPageToken")
        if not token:
            break
        params["pageToken"] = token
    return docs


def _firestore_doc_to_plain(document: dict[str, Any]) -> dict[str, Any]:
    fields = {
        key: firestore_value(value)
        for key, value in document.get("fields", {}).items()
    }
    doc_id = document.get("name", "").split("/")[-1]
    fields.setdefault("document_id", doc_id)
    fields.setdefault("id", doc_id)
    fields.setdefault("name", doc_id)
    return fields


def _first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _coordinate_pair(item: dict[str, Any]) -> tuple[float | None, float | None]:
    lat = _first_present(item, ("lat", "latitude", "y"))
    lng = _first_present(item, ("lng", "lon", "longitude", "x"))
    if lat is None or lng is None:
        location = item.get("location") or item.get("geo") or item.get("position") or {}
        if isinstance(location, dict):
            lat = lat if lat is not None else _first_present(location, ("lat", "latitude", "y"))
            lng = lng if lng is not None else _first_present(location, ("lng", "lon", "longitude", "x"))
    if lat is None or lng is None:
        return None, None
    return float(lat), float(lng)


def _expand_collection_items(
    docs: list[dict[str, Any]],
    array_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for doc in docs:
        plain = _firestore_doc_to_plain(doc)
        expanded = False
        for key in array_keys:
            value = plain.get(key)
            if isinstance(value, list):
                for index, item in enumerate(value):
                    if not isinstance(item, dict):
                        continue
                    expanded_item = dict(item)
                    expanded_item.setdefault("parent_document_id", plain.get("document_id"))
                    expanded_item.setdefault("id", f"{plain.get('document_id')}_{index}")
                    expanded_item.setdefault("name", expanded_item.get("id"))
                    items.append(expanded_item)
                expanded = True
        if not expanded:
            items.append(plain)
    return items


def _load_firebase_nodes() -> dict[str, Node]:
    docs = _firestore_documents(firebase_nodes_collection())
    raw_nodes = _expand_collection_items(docs, ("nodes", "items", "data"))
    nodes: dict[str, Node] = {}

    for item in raw_nodes:
        node_id = str(
            _first_present(
                item,
                ("id", "node_id", "nodeId", "place_id", "placeId", "document_id", "name", "title"),
            )
            or ""
        ).strip()
        lat, lng = _coordinate_pair(item)
        if not node_id or lat is None or lng is None:
            continue
        nodes[node_id] = Node(
            id=node_id,
            name=str(_first_present(item, ("name", "title", "label", "address", "document_id")) or node_id),
            lat=lat,
            lng=lng,
        )

    if not nodes:
        raise RuntimeError(f"No valid nodes found in Firebase collection {firebase_nodes_collection()}")
    return nodes


def _normalize_edge(item: dict[str, Any]) -> dict[str, Any] | None:
    from_id = _first_present(
        item,
        ("from", "from_id", "fromId", "source_id", "sourceId", "start", "start_id", "startId"),
    )
    to_id = _first_present(
        item,
        ("to", "to_id", "toId", "target_id", "targetId", "end", "end_id", "endId"),
    )
    if not from_id or not to_id:
        return None

    normalized = dict(item)
    normalized["from"] = str(from_id)
    normalized["to"] = str(to_id)
    normalized["name"] = str(normalized.get("name") or normalized.get("title") or "safe route segment")
    bidirectional = normalized.get("bidirectional", True)
    if isinstance(bidirectional, str):
        bidirectional = bidirectional.strip().lower() not in {"0", "false", "no", "off"}
    normalized["bidirectional"] = bool(bidirectional)
    normalized["accessibility"] = normalized.get("accessibility") or {}
    normalized["source"] = normalized.get("source") or {}
    return normalized


def _load_firebase_edges() -> list[dict[str, Any]]:
    docs = _firestore_documents(firebase_edges_collection())
    raw_edges = _expand_collection_items(docs, ("edges", "items", "data"))
    edges = [edge for item in raw_edges if (edge := _normalize_edge(item)) is not None]
    if not edges:
        raise RuntimeError(f"No valid edges found in Firebase collection {firebase_edges_collection()}")
    return edges


def _load_firebase_environment_points() -> list[SafetyPoint]:
    raw_points: list[dict[str, Any]] = []
    nodes_collection = firebase_nodes_collection()
    for collection in firebase_environment_collections():
        if collection == nodes_collection:
            continue
        try:
            docs = _firestore_documents(collection)
        except RuntimeError:
            continue
        raw_points.extend(_expand_collection_items(docs, ("points", "items", "data")))
    return parse_safety_points(raw_points)


def _load_local_route_data() -> tuple[dict[str, Node], list[dict[str, Any]], list[SafetyPoint]] | None:
    nodes_path = local_nodes_path()
    edges_path = local_edges_path()
    if not nodes_path.exists() or not edges_path.exists():
        return None

    nodes = load_nodes(nodes_path)
    raw_edges = _json_items(json.loads(edges_path.read_text(encoding="utf-8")), ("edges", "items", "data"))

    environment_points: list[SafetyPoint] = []
    environment_path = local_environment_points_path()
    if environment_path.exists():
        environment_points = load_safety_points(environment_path)

    return nodes, raw_edges, environment_points


def _load_route_data() -> tuple[dict[str, Node], list[dict[str, Any]], list[SafetyPoint]]:
    global _route_data_cache
    now = time.time()
    if _route_data_cache and now - _route_data_cache[0] < _FIRESTORE_CACHE_TTL_SECONDS:
        _, nodes, edges, environment_points = _route_data_cache
        return nodes, edges, environment_points

    local_route_data = _load_local_route_data()
    if local_route_data is not None:
        nodes, edges, environment_points = local_route_data
    else:
        nodes = _load_firebase_nodes()
        edges = _load_firebase_edges()
        environment_points = _load_firebase_environment_points()
    _route_data_cache = (now, nodes, edges, environment_points)
    return nodes, edges, environment_points


def _find_node_by_destination_text(nodes: dict[str, Node], destination: str) -> Node | None:
    query = destination.strip().lower()
    if not query:
        return None
    if query in nodes:
        return nodes[query]

    exact = [node for node in nodes.values() if node.name.strip().lower() == query]
    if exact:
        return exact[0]

    contains = [
        node for node in nodes.values()
        if query in node.name.strip().lower() or node.name.strip().lower() in query
    ]
    return min(contains, key=lambda node: len(node.name)) if contains else None


def _resolve_destination(
    nodes: dict[str, Node],
    destination: str,
) -> tuple[float | None, float | None, str]:
    parsed = _parse_latlng(destination)
    if parsed:
        return parsed[0], parsed[1], destination

    matched_node = _find_node_by_destination_text(nodes, destination)
    if matched_node:
        return matched_node.lat, matched_node.lng, matched_node.name

    return geocode_address(destination)


def _format_distance(distance_m: float) -> str:
    if distance_m >= 1000:
        return f"{distance_m / 1000:.1f} 公里"
    return f"{distance_m:.0f} 公尺"


def _safe_route_step_text(edge_name: str, reasons: list[str]) -> str:
    useful_reasons = []
    for reason in reasons:
        if "has sidewalk" in reason:
            useful_reasons.append("此路段有人行空間")
        elif "near help point" in reason:
            useful_reasons.append("附近有人流或可求助地點")
        elif "lighting: good" in reason:
            useful_reasons.append("照明較佳")
    suffix = f"，{useful_reasons[0]}" if useful_reasons else ""
    return f"沿著 {edge_name} 前進{suffix}"


def get_directions_stepwise(
    origin: str,
    destination: str,
    mode: str = "walking",
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    del mode
    origin_latlng = _parse_latlng(origin)
    if not origin_latlng:
        return None, ["目前沒有有效 GPS 位置，無法規劃安全路線。"]

    try:
        nodes, raw_edges, environment_points = _load_route_data()
        dest_lat, dest_lng, dest_name = _resolve_destination(nodes, destination)
        if dest_lat is None or dest_lng is None:
            return None, [f"找不到目的地「{destination}」的座標，無法規劃安全路線。"]

        start_node = nearest_node(nodes, origin_latlng[0], origin_latlng[1])
        end_node = nearest_node(nodes, dest_lat, dest_lng)
        start_snap_m = haversine_m(origin_latlng[0], origin_latlng[1], start_node.lat, start_node.lng)
        end_snap_m = haversine_m(dest_lat, dest_lng, end_node.lat, end_node.lng)
        if start_snap_m > _MAX_SNAP_DISTANCE_M:
            return None, ["目前位置超出安全路網範圍，暫時無法規劃安全路線。"]
        if end_snap_m > _MAX_SNAP_DISTANCE_M:
            return None, [f"目的地「{destination}」超出安全路網範圍，暫時無法規劃安全路線。"]

        total_cost, route = find_accessible_route_from_data(
            nodes,
            raw_edges,
            start_node.id,
            end_node.id,
            environment_points=environment_points,
        )
        if not route:
            return None, [f"找不到從 {start_node.name} 到 {end_node.name} 的安全路線。"]
    except Exception as exc:  # noqa: BLE001
        return None, [f"安全路線規劃失敗：{exc}"]

    total_distance = sum(edge.distance_m for edge in route)
    duration_min = max(1, round(total_distance / _WALKING_SPEED_M_PER_MIN))
    leg = {
        "start_address": start_node.name,
        "end_address": dest_name or end_node.name,
        "distance": {"text": _format_distance(total_distance)},
        "duration": {"text": f"{duration_min} 分鐘"},
        "safe_route_cost": round(total_cost, 1),
    }

    steps_out: list[dict[str, Any]] = []
    for edge in route:
        to_node = nodes[edge.to_id]
        steps_out.append(
            {
                "text": _safe_route_step_text(edge.name, explain_edge(edge)),
                "distance": _format_distance(edge.distance_m),
                "end_lat": to_node.lat,
                "end_lng": to_node.lng,
                "risk_breakdown": edge.risk_breakdown,
            }
        )
    return leg, steps_out


def get_directions_summary(origin: str, destination: str, mode: str = "walking") -> str:
    leg, steps = get_directions_stepwise(origin, destination, mode=mode)
    if not leg:
        first = steps[0] if steps else None
        if isinstance(first, dict):
            return str(first.get("text") or "無法規劃安全路線。")
        return str(first or "無法規劃安全路線。")
    return (
        f"安全路線從 {leg['start_address']} 到 {leg['end_address']}，"
        f"預估 {leg['duration']['text']}，距離 {leg['distance']['text']}。"
    )


def find_nearby_places(keyword: str, lat: float, lng: float, radius: int = 3000) -> tuple[Optional[str], str]:
    key = google_maps_api_key()
    if not key:
        return None, "尚未設定 GOOGLE_MAPS_API_KEY，無法搜尋附近地點。"

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": key,
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": keyword,
        "language": "zh-TW",
    }
    data = requests.get(url, params=params, timeout=15).json()
    results = data.get("results", [])
    if not results:
        return None, f"附近找不到「{keyword}」。"

    nearest = results[0]
    name = nearest["name"]
    address = nearest.get("vicinity", "地址不明")
    return name, f"附近找到 {name}，位置在 {address}。"
