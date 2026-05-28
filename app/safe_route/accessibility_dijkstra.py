from __future__ import annotations

import heapq
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    lat: float
    lng: float


@dataclass(frozen=True)
class Edge:
    from_id: str
    to_id: str
    name: str
    distance_m: float
    cost: float
    accessibility: dict[str, Any]
    source: dict[str, Any]
    bearing: float
    risk_breakdown: dict[str, float]


@dataclass(frozen=True)
class SafetyPoint:
    lat: float
    lng: float
    kind: str
    name: str = ""
    weight: float = 1.0


EARTH_RADIUS_M = 6_371_000
SAFETY_WEIGHT = 0.70
ENVIRONMENT_WEIGHT = 0.20
NAVIGATION_WEIGHT = 0.10
TURN_THRESHOLD_DEG = 35.0
SHARP_TURN_THRESHOLD_DEG = 70.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)

    a = (
        math.sin(d_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lng / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_M * c


def bearing_deg(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lng = math.radians(lng2 - lng1)
    y = math.sin(d_lng) * math.cos(lat2_rad)
    x = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(d_lng)
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def as_text(value: Any) -> str:
    if isinstance(value, list):
        return ";".join(str(item).lower() for item in value)
    if value is None:
        return ""
    text = str(value).lower()
    return "" if text == "nan" else text


def is_help_point(point: SafetyPoint) -> bool:
    return point.kind in {
        "convenience_store",
        "police",
        "fire_station",
        "security",
        "health_center",
        "office",
        "help",
    }


def is_people_flow_point(point: SafetyPoint) -> bool:
    return point.kind in {
        "bus_stop",
        "restaurant",
        "canteen",
        "library",
        "admin",
        "classroom",
        "crowd",
        "people_flow",
    }


def closest_distance_m(lat: float, lng: float, points: list[SafetyPoint]) -> float | None:
    if not points:
        return None
    return min(haversine_m(lat, lng, point.lat, point.lng) for point in points)


def distance_bonus(distance_m: float | None, close_m: float, far_m: float, max_bonus: float) -> float:
    if distance_m is None:
        return 0.0
    if distance_m <= close_m:
        return max_bonus
    if distance_m >= far_m:
        return 0.0
    ratio = (far_m - distance_m) / (far_m - close_m)
    return max_bonus * ratio


def safety_risk(accessibility: dict[str, Any]) -> float:
    risk = 0.0

    if accessibility.get("has_sidewalk"):
        risk -= 60
    else:
        risk += 650

    crossing_penalty = {
        "low": 0,
        "medium": 180,
        "high": 520,
    }
    risk += crossing_penalty.get(accessibility.get("crossing_risk"), 240)

    lighting_penalty = {
        "good": 0,
        "normal": 60,
        "poor": 220,
    }
    risk += lighting_penalty.get(accessibility.get("lighting"), 90)

    if accessibility.get("has_audio_signal"):
        risk -= 50

    if accessibility.get("has_tactile_paving"):
        risk -= 60

    return max(risk, 0.0)


def static_navigation_risk(
    accessibility: dict[str, Any],
    source: dict[str, Any],
    degree_to: int,
) -> float:
    risk = 0.0
    highway = as_text(source.get("highway"))
    footway = as_text(source.get("footway"))
    service = as_text(source.get("service"))

    if accessibility.get("crossing_risk") == "high" or any(
        item in highway for item in ["primary", "secondary", "tertiary", "trunk"]
    ):
        risk += 260

    if "service" in highway or "parking" in service:
        risk += 180

    if any(item in highway for item in ["path", "track"]) and not accessibility.get("has_tactile_paving"):
        risk += 150

    if "footway" in highway and footway not in {"sidewalk", "crossing"}:
        risk += 80

    if degree_to <= 1:
        risk += 160

    return risk


def environment_risk(
    from_node: Node,
    to_node: Node,
    environment_points: list[SafetyPoint] | None,
) -> float:
    if not environment_points:
        return 120.0

    mid_lat = (from_node.lat + to_node.lat) / 2
    mid_lng = (from_node.lng + to_node.lng) / 2
    help_points = [point for point in environment_points if is_help_point(point)]
    people_points = [point for point in environment_points if is_people_flow_point(point)]
    isolated_points = [point for point in environment_points if point.kind in {"isolated", "remote"}]

    risk = 220.0
    risk -= distance_bonus(closest_distance_m(mid_lat, mid_lng, help_points), 60, 350, 90)
    risk -= distance_bonus(closest_distance_m(mid_lat, mid_lng, people_points), 80, 420, 80)
    risk += distance_bonus(closest_distance_m(mid_lat, mid_lng, isolated_points), 0, 250, 120)
    return max(risk, 0.0)


def weighted_cost(
    distance_m: float,
    safety: float,
    environment: float,
    navigation: float,
) -> tuple[float, dict[str, float]]:
    weighted_risk = (
        SAFETY_WEIGHT * safety
        + ENVIRONMENT_WEIGHT * environment
        + NAVIGATION_WEIGHT * navigation
    )
    total = max(distance_m + weighted_risk, 1.0)
    return total, {
        "distance": round(distance_m, 3),
        "safety": round(safety, 3),
        "environment": round(environment, 3),
        "navigation": round(navigation, 3),
        "weighted_risk": round(weighted_risk, 3),
        "total": round(total, 3),
    }


def turn_navigation_penalty(prev_edge: Edge | None, edge: Edge, consecutive_turns: int) -> tuple[float, int]:
    if prev_edge is None:
        return 0.0, 0

    turn_angle = angle_diff_deg(prev_edge.bearing, edge.bearing)
    if turn_angle < TURN_THRESHOLD_DEG:
        return 0.0, 0

    penalty = 90.0
    if turn_angle >= SHARP_TURN_THRESHOLD_DEG:
        penalty += 80.0

    next_consecutive_turns = min(consecutive_turns + 1, 3)
    if next_consecutive_turns >= 2:
        penalty += 140.0 * (next_consecutive_turns - 1)

    return penalty, next_consecutive_turns


def load_nodes(path: Path) -> dict[str, Node]:
    raw_nodes = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: Node(**{key: item[key] for key in ["id", "name", "lat", "lng"]}) for item in raw_nodes}


def load_safety_points(path: Path) -> list[SafetyPoint]:
    raw_points = json.loads(path.read_text(encoding="utf-8"))
    return parse_safety_points(raw_points)


def parse_safety_points(raw_points: list[dict[str, Any]]) -> list[SafetyPoint]:
    points: list[SafetyPoint] = []
    for item in raw_points:
        lat = item.get("lat", item.get("latitude"))
        lng = item.get("lng", item.get("lon", item.get("longitude")))
        if lat is None or lng is None:
            location = item.get("location") or {}
            lat = location.get("lat", location.get("latitude"))
            lng = location.get("lng", location.get("lon", location.get("longitude")))
        if lat is None or lng is None:
            continue
        points.append(
            SafetyPoint(
                lat=float(lat),
                lng=float(lng),
                kind=str(item.get("kind", item.get("type", "help"))),
                name=str(item.get("name", "")),
                weight=float(item.get("weight", 1.0)),
            )
        )
    return points


def firestore_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "integerValue" in value:
        return int(value["integerValue"])
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "booleanValue" in value:
        return bool(value["booleanValue"])
    if "nullValue" in value:
        return None
    if "geoPointValue" in value:
        geo = value["geoPointValue"]
        return {"latitude": geo.get("latitude"), "longitude": geo.get("longitude")}
    if "mapValue" in value:
        return {
            key: firestore_value(field_value)
            for key, field_value in value["mapValue"].get("fields", {}).items()
        }
    if "arrayValue" in value:
        return [firestore_value(item) for item in value["arrayValue"].get("values", [])]
    return None


def firestore_document_to_point(document: dict[str, Any], collection_name: str) -> dict[str, Any]:
    fields = {
        key: firestore_value(value)
        for key, value in document.get("fields", {}).items()
    }
    doc_id = document.get("name", "").split("/")[-1]
    fields.setdefault("id", doc_id)
    fields.setdefault("name", fields.get("title", doc_id))
    fields.setdefault("kind", fields.get("type", collection_name))
    return fields


def load_firebase_safety_points(
    firebase_config_path: Path,
    collections: list[str] | None = None,
) -> list[SafetyPoint]:
    config = json.loads(firebase_config_path.read_text(encoding="utf-8"))
    project_id = config.get("projectId")
    api_key = config.get("apiKey")
    if not project_id:
        raise ValueError("Firebase config is missing projectId")

    collections = collections or ["safe_places", "nuk"]
    raw_points: list[dict[str, Any]] = []
    for collection_name in collections:
        encoded_collection = urllib.parse.quote(collection_name, safe="")
        url = (
            "https://firestore.googleapis.com/v1/projects/"
            f"{project_id}/databases/(default)/documents/{encoded_collection}"
        )
        if api_key:
            url += "?" + urllib.parse.urlencode({"key": api_key})

        with urllib.request.urlopen(url, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))

        for document in payload.get("documents", []):
            raw_points.append(firestore_document_to_point(document, collection_name))

    return parse_safety_points(raw_points)


def load_graph(
    nodes: dict[str, Node],
    edges_path: Path,
    environment_points: list[SafetyPoint] | None = None,
) -> dict[str, list[Edge]]:
    raw_edges = json.loads(edges_path.read_text(encoding="utf-8"))
    return load_graph_from_edges(nodes, raw_edges, environment_points)


def load_graph_from_edges(
    nodes: dict[str, Node],
    raw_edges: list[dict[str, Any]],
    environment_points: list[SafetyPoint] | None = None,
) -> dict[str, list[Edge]]:
    graph: dict[str, list[Edge]] = {node_id: [] for node_id in nodes}
    degree: dict[str, int] = {node_id: 0 for node_id in nodes}

    for item in raw_edges:
        if item["from"] in degree and item["to"] in degree:
            degree[item["from"]] += 1
            degree[item["to"]] += 1

    for item in raw_edges:
        from_node = nodes[item["from"]]
        to_node = nodes[item["to"]]
        distance_m = haversine_m(from_node.lat, from_node.lng, to_node.lat, to_node.lng)
        accessibility = item.get("accessibility", {})
        source = item.get("source", {})

        safety = safety_risk(accessibility)
        env = environment_risk(from_node, to_node, environment_points)
        nav = static_navigation_risk(accessibility, source, degree.get(item["to"], 0))
        cost, breakdown = weighted_cost(distance_m, safety, env, nav)

        edge = Edge(
            from_id=item["from"],
            to_id=item["to"],
            name=item.get("name", "unknown route segment"),
            distance_m=distance_m,
            cost=cost,
            accessibility=accessibility,
            source=source,
            bearing=bearing_deg(from_node.lat, from_node.lng, to_node.lat, to_node.lng),
            risk_breakdown=breakdown,
        )
        graph[edge.from_id].append(edge)

        if item.get("bidirectional", True):
            reverse_nav = static_navigation_risk(accessibility, source, degree.get(item["from"], 0))
            reverse_cost, reverse_breakdown = weighted_cost(distance_m, safety, env, reverse_nav)
            reverse_edge = Edge(
                from_id=edge.to_id,
                to_id=edge.from_id,
                name=edge.name,
                distance_m=edge.distance_m,
                cost=reverse_cost,
                accessibility=edge.accessibility,
                source=edge.source,
                bearing=bearing_deg(to_node.lat, to_node.lng, from_node.lat, from_node.lng),
                risk_breakdown=reverse_breakdown,
            )
            graph[reverse_edge.from_id].append(reverse_edge)

    return graph


def dijkstra(graph: dict[str, list[Edge]], start_id: str, end_id: str) -> tuple[float, list[Edge]]:
    start_state = (start_id, -1, 0)
    queue: list[tuple[float, int, tuple[str, int, int]]] = [(0.0, 0, start_state)]
    distances = {start_state: 0.0}
    previous: dict[tuple[str, int, int], tuple[tuple[str, int, int], Edge]] = {}
    best_end_state: tuple[str, int, int] | None = None
    sequence = 0

    while queue:
        current_cost, _, state = heapq.heappop(queue)
        current_id, _, consecutive_turns = state
        if current_cost > distances.get(state, float("inf")):
            continue

        if current_id == end_id:
            best_end_state = state
            break

        prev_edge = previous[state][1] if state in previous else None

        for edge in graph[current_id]:
            turn_penalty, next_turns = turn_navigation_penalty(prev_edge, edge, consecutive_turns)
            bearing_bucket = int(edge.bearing // 15)
            next_state = (edge.to_id, bearing_bucket, next_turns)
            next_cost = current_cost + edge.cost + NAVIGATION_WEIGHT * turn_penalty
            if next_cost < distances.get(next_state, float("inf")):
                distances[next_state] = next_cost
                previous[next_state] = (state, edge)
                sequence += 1
                heapq.heappush(queue, (next_cost, sequence, next_state))

    if best_end_state is None:
        raise ValueError(f"No route found from {start_id} to {end_id}")

    route: list[Edge] = []
    current_state = best_end_state
    while current_state != start_state:
        previous_state, edge = previous[current_state]
        route.append(edge)
        current_state = previous_state

    route.reverse()
    return distances[best_end_state], route


def nearest_node(nodes: dict[str, Node], lat: float, lng: float) -> Node:
    return min(
        nodes.values(),
        key=lambda node: haversine_m(lat, lng, node.lat, node.lng),
    )


def explain_edge(edge: Edge) -> list[str]:
    a = edge.accessibility
    source = edge.source
    reasons: list[str] = []

    reasons.append("has sidewalk" if a.get("has_sidewalk") else "no sidewalk, high safety penalty")
    reasons.append(f"crossing risk: {a.get('crossing_risk', 'unknown')}")
    reasons.append(f"lighting: {a.get('lighting', 'unknown')}")

    highway = as_text(source.get("highway"))
    if "service" in highway:
        reasons.append("passes service road or parking/vehicle area")
    if any(item in highway for item in ["path", "track"]):
        reasons.append("campus path or trail; markings may be unclear")
    if edge.risk_breakdown.get("navigation", 0) >= 180:
        reasons.append("higher navigation difficulty")
    if edge.risk_breakdown.get("environment", 0) <= 80:
        reasons.append("near help point or high foot traffic area")

    return reasons


def route_to_geojson(nodes: dict[str, Node], route: list[Edge]) -> dict[str, Any]:
    features: list[dict[str, Any]] = []

    for edge in route:
        from_node = nodes[edge.from_id]
        to_node = nodes[edge.to_id]
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "name": edge.name,
                    "from": from_node.name,
                    "to": to_node.name,
                    "distance_m": round(edge.distance_m, 1),
                    "cost": round(edge.cost, 1),
                    "risk_breakdown": edge.risk_breakdown,
                    "reasons": explain_edge(edge),
                    "source": edge.source,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [from_node.lng, from_node.lat],
                        [to_node.lng, to_node.lat],
                    ],
                },
            }
        )

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def find_accessible_route(
    nodes_path: Path,
    edges_path: Path,
    start_id: str,
    end_id: str,
    environment_points_path: Path | None = None,
    environment_points: list[SafetyPoint] | None = None,
) -> tuple[dict[str, Node], float, list[Edge]]:
    nodes = load_nodes(nodes_path)
    loaded_environment_points = environment_points or (
        load_safety_points(environment_points_path)
        if environment_points_path is not None and environment_points_path.exists()
        else None
    )
    graph = load_graph(nodes, edges_path, loaded_environment_points)
    total_cost, route = dijkstra(graph, start_id, end_id)
    return nodes, total_cost, route


def find_accessible_route_from_data(
    nodes: dict[str, Node],
    raw_edges: list[dict[str, Any]],
    start_id: str,
    end_id: str,
    environment_points: list[SafetyPoint] | None = None,
) -> tuple[float, list[Edge]]:
    graph = load_graph_from_edges(nodes, raw_edges, environment_points)
    return dijkstra(graph, start_id, end_id)
