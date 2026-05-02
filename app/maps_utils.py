"""
Google Maps Directions / Geocoding（與 Pi/navigation_utils.py 對齊概念）。
需環境變數 GOOGLE_MAPS_API_KEY。
"""

from __future__ import annotations

import re
from typing import Any, Optional

import requests

from app.config import google_maps_api_key


def rough_origin_latlng() -> Optional[str]:
    """無手機 GPS 時的備援：用出口 IP 粗估「緯度,經度」（精度有限，僅供開發）。"""
    try:
        r = requests.get("https://ipinfo.io/json", timeout=5).json()
        loc = r.get("loc")
        if loc and "," in str(loc):
            return str(loc).strip()
    except OSError:
        pass
    return None


def geocode_address(address: str) -> tuple[Optional[float], Optional[float], str]:
    key = google_maps_api_key()
    if not key:
        return None, None, address
    if re.match(r"^-?\d+\.?\d*,-?\d+\.?\d*$", address.strip()):
        lat, lng = map(float, address.split(","))
        return lat, lng, address
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": address, "language": "zh-TW", "key": key}
    res = requests.get(url, params=params, timeout=15).json()
    if res.get("status") == "OK" and res.get("results"):
        loc = res["results"][0]["geometry"]["location"]
        formatted = res["results"][0]["formatted_address"]
        return loc["lat"], loc["lng"], formatted
    return None, None, address


def get_directions_stepwise(
    origin: str,
    destination: str,
    mode: str = "walking",
) -> tuple[Optional[dict[str, Any]], list[dict[str, Any]]]:
    key = google_maps_api_key()
    if not key:
        return None, ["尚未設定 GOOGLE_MAPS_API_KEY，無法查詢路線。"]

    dest_lat, dest_lng, dest_fmt = geocode_address(destination)
    if dest_lat and dest_lng:
        dest_param = f"{dest_lat},{dest_lng}"
    elif dest_fmt:
        dest_param = dest_fmt
    else:
        dest_param = destination

    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": origin,
        "destination": dest_param,
        "mode": mode,
        "language": "zh-TW",
        "key": key,
    }
    res = requests.get(url, params=params, timeout=20).json()
    status = res.get("status")
    if status != "OK":
        msg = "抱歉，找不到可行的路線。"
        if status == "ZERO_RESULTS":
            msg = f"抱歉，找不到從起點到「{destination}」的路線。"
        elif status == "REQUEST_DENIED":
            msg = "Google Maps API 請求被拒絕，請檢查 GOOGLE_MAPS_API_KEY 與已啟用的 API（Directions、Geocoding）。"
        return None, [msg]

    leg = res["routes"][0]["legs"][0]
    steps_out: list[dict[str, Any]] = []
    for step in leg["steps"]:
        instruction = re.sub(r"<[^>]+>", "", step["html_instructions"])
        end_loc = step["end_location"]
        steps_out.append(
            {
                "text": instruction,
                "distance": step["distance"]["text"],
                "end_lat": end_loc["lat"],
                "end_lng": end_loc["lng"],
            }
        )
    return leg, steps_out


def get_directions_summary(origin: str, destination: str, mode: str = "walking") -> str:
    key = google_maps_api_key()
    if not key:
        return "尚未設定 GOOGLE_MAPS_API_KEY，無法查詢路線。"
    leg, steps = get_directions_stepwise(origin, destination, mode=mode)
    if not leg:
        return steps[0] if steps else "無法取得路線。"
    return (
        f"從 {leg['start_address']} 到 {leg['end_address']} "
        f"約 {leg['duration']['text']}，距離 {leg['distance']['text']}。"
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
        return None, f"附近沒有找到關於「{keyword}」的地點。"

    nearest = results[0]
    name = nearest["name"]
    address = nearest.get("vicinity", "無地址資訊")
    return name, f"離你最近的是「{name}」，約在 {address}。"
