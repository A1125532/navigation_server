"""
導航／搜尋／聊天邏輯，對齊 Pi 專案 ai_server + chat_utils + navigation_utils 概念。
需 OPENAI_API_KEY（意圖與聊天）；導航與附近搜尋另需 GOOGLE_MAPS_API_KEY。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from openai import OpenAI

from app.config import (
    google_maps_api_key,
    openai_api_key,
    openai_chat_model,
    yolo_enabled,
    yolo_model_path,
)
from app.maps_utils import (
    find_nearby_places,
    get_directions_stepwise,
    get_directions_summary,
    rough_origin_latlng,
)

INTENT_SYSTEM_PROMPT = """你是一個智慧語音助理，請根據使用者語句判斷意圖。
請只輸出 JSON 格式，包含以下三種 intent：
1️⃣ "navigate"：使用者想去某個地點，請回傳 "destination"
2️⃣ "search"：使用者想找附近的東西，請回傳 "keyword"
3️⃣ "chat"：一般對話
---
範例：
{"intent": "navigate", "destination": "台北101"}
{"intent": "search", "keyword": "餐廳"}
{"intent": "chat"}"""

CHAT_SYSTEM_PROMPT = "你是一個語音導航助理，請用簡短清楚的繁體中文回答。"


def _clean(s: str) -> str:
    return (
        s.replace("！", "")
        .replace("!", "")
        .replace("。", "")
        .replace(".", "")
        .strip()
    )


def _extract_payload(payload: dict[str, Any]) -> tuple[str, Optional[str], Optional[str], Any]:
    text = str(payload.get("text") or "").strip()
    dest = payload.get("current_destination") or payload.get("currentDestination")
    if dest is not None:
        dest = str(dest).strip() or None
    origin = payload.get("origin") or payload.get("Origin")
    if origin is not None:
        origin = str(origin).strip() or None
    hist = payload.get("conversation_history") or payload.get("conversationHistory")
    return text, dest, origin, hist


def _analyze_intent(client: OpenAI, user_input: str) -> dict[str, Any]:
    r = client.chat.completions.create(
        model=openai_chat_model(),
        messages=[
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ],
        temperature=0,
    )
    raw = (r.choices[0].message.content or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "chat"}


def _resolve_origin(origin_from_phone: Optional[str]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """優先用手機傳的 lat,lng；否則用 IP 粗估（僅備援，精度有限）。"""
    if origin_from_phone and "," in origin_from_phone:
        parts = origin_from_phone.split(",")
        if len(parts) == 2:
            try:
                lat, lng = float(parts[0].strip()), float(parts[1].strip())
                if -90 <= lat <= 90 and -180 <= lng <= 180:
                    return lat, lng, f"{lat},{lng}"
            except ValueError:
                pass
    rough = rough_origin_latlng()
    if rough:
        try:
            a, b = rough.split(",")
            return float(a), float(b), rough
        except ValueError:
            pass
    return None, None, None


def handle_ai(payload: dict[str, Any]) -> dict[str, Any]:
    text, current_dest, origin_phone, conv_hist = _extract_payload(payload)
    if not text:
        return {"reply": "辨識不清或環境過於嘈雜，請再說一次。", "intent": "chat"}

    clean = _clean(text)
    bye_kw = ["結束導航", "结束导航", "再見", "再见", "結束", "结束", "掰掰"]
    if any(kw in clean for kw in bye_kw):
        return {"reply": "好的，已結束導航，下次再見！", "intent": "chat"}

    key = openai_api_key()
    if not key:
        demo = (
            "（未設定 OPENAI_API_KEY）\n"
            f"已收到：「{text}」。"
            + (f"\n目前目的地欄位：{current_dest}。" if current_dest else "")
            + "\n請在 navigation-server/.env 填入 OPENAI_API_KEY 與 GOOGLE_MAPS_API_KEY 後重啟伺服器。"
        )
        intent = "navigate" if current_dest or "帶我去" in text else "chat"
        return {"reply": demo, "intent": intent}

    client = OpenAI(api_key=key)
    maps_ok = bool(google_maps_api_key())

    # 導航中：問剩餘時間／距離
    remaining_kw = [
        "還有多久",
        "多久到",
        "多遠",
        "距離",
        "還有多遠",
        "要走多久",
        "要多久",
        "幾分鐘",
    ]
    if current_dest and any(kw in clean for kw in remaining_kw):
        lat, lng, origin_str = _resolve_origin(origin_phone)
        if lat is None:
            return {
                "reply": "無法取得你的位置（請在手機傳 origin=緯度,經度，或檢查伺服器網路以使用 IP 粗估）。",
                "intent": "navigate",
            }
        if maps_ok:
            summary = get_directions_summary(origin_str or f"{lat},{lng}", current_dest, mode="walking")
            return {"reply": summary, "intent": "navigate", "origin_used": origin_str or f"{lat},{lng}"}
        return {
            "reply": "已記錄你的問題，但尚未設定 GOOGLE_MAPS_API_KEY，無法計算距離與時間。",
            "intent": "navigate",
        }

    intent_data = _analyze_intent(client, clean)
    intent = str(intent_data.get("intent", "chat")).lower()

    # 導航
    if intent == "navigate":
        destination = str(intent_data.get("destination", "") or "").strip()
        if not destination and current_dest:
            destination = current_dest.strip()
        if not destination and "帶我去" in text:
            destination = text.split("帶我去", 1)[-1].strip() or destination
        if not destination:
            return {
                "reply": "請告訴我目的地，例如「帶我去高雄火車站」。",
                "intent": "navigate",
            }

        lat, lng, origin_str = _resolve_origin(origin_phone)
        if lat is None or not maps_ok:
            msg = (
                "無法取得起點座標或尚未設定 Google Maps API。"
                "請在 .env 設定 GOOGLE_MAPS_API_KEY，並讓手機在呼叫 /ai 時帶入 origin（「緯度,經度」）。"
            )
            if key:
                r2 = client.chat.completions.create(
                    model=openai_chat_model(),
                    messages=[
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"使用者想去「{destination}」，但目前沒有精確 GPS。請簡短說明如何查路或到達後再問，不要捏造轉彎。",
                        },
                    ],
                    temperature=0.4,
                    max_tokens=400,
                )
                msg = (r2.choices[0].message.content or msg).strip()
            return {"reply": msg, "intent": "navigate"}

        origin = origin_str or f"{lat},{lng}"
        leg, steps = get_directions_stepwise(origin, destination, mode="walking")
        if not leg or not steps:
            err = steps[0] if steps else "無法取得路線。"
            return {"reply": err, "intent": "navigate", "origin_used": origin}

        parts: list[str] = [
            f"開始導航到「{destination}」。從 {leg['start_address']} 出發，全程約 {leg['distance']['text']}。"
        ]
        for i, step in enumerate(steps[:2], start=1):
            parts.append(f"第 {i} 步：{step['text']}，約 {step['distance']}。")
        if len(steps) > 2:
            parts.append("更多轉彎請再問「下一步」或繼續對話。")
        return {"reply": "".join(parts), "intent": "navigate", "origin_used": origin}

    # 搜尋附近
    if intent == "search":
        keyword = str(intent_data.get("keyword", "") or "").strip()
        if not keyword:
            return {"reply": "請告訴我你想找什麼，例如「附近有咖啡廳嗎」。", "intent": "search"}
        lat, lng, origin_str = _resolve_origin(origin_phone)
        if lat is None or not maps_ok:
            return {
                "reply": "需要你的位置與 Google Maps API Key 才能搜尋附近。請設定 GOOGLE_MAPS_API_KEY 並傳 origin。",
                "intent": "search",
            }
        _name, info = find_nearby_places(keyword, lat, lng)
        return {"reply": info, "intent": "search", "origin_used": origin_str or f"{lat},{lng}"}

    # 一般聊天
    system_content = CHAT_SYSTEM_PROMPT
    if current_dest:
        system_content += (
            f"\n\n（使用者正在導航中，目的地：「{current_dest}」。請簡短回應並記住情境。）"
        )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_content}]
    if isinstance(conv_hist, list):
        for item in conv_hist:
            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(
                    {"role": str(item["role"]), "content": str(item["content"])}
                )
    messages.append({"role": "user", "content": text})
    r = client.chat.completions.create(
        model=openai_chat_model(),
        messages=messages,
        temperature=0.4,
        max_tokens=500,
    )
    reply = (r.choices[0].message.content or "").strip() or "（無回覆）"
    return {"reply": reply, "intent": "chat"}


_yolo_model = None
_yolo_load_error: str | None = None


def _get_yolo_model():
    global _yolo_model, _yolo_load_error
    if _yolo_model is not None:
        return _yolo_model
    if _yolo_load_error is not None:
        raise RuntimeError(_yolo_load_error)
    try:
        from ultralytics import YOLO

        _yolo_model = YOLO(yolo_model_path())
        return _yolo_model
    except Exception as e:  # noqa: BLE001
        _yolo_load_error = str(e)
        raise


def warmup_yolo_model() -> dict[str, Any]:
    """啟動時預載模型（雲端部署建議在 lifespan 呼叫）。"""
    if not yolo_enabled():
        return {"enabled": False, "loaded": False}
    try:
        _get_yolo_model()
        return {"enabled": True, "loaded": True, "model": yolo_model_path()}
    except Exception as e:  # noqa: BLE001
        return {"enabled": True, "loaded": False, "error": str(e), "model": yolo_model_path()}


def yolo_status() -> dict[str, Any]:
    if not yolo_enabled():
        return {"enabled": False, "loaded": False}
    return {
        "enabled": True,
        "loaded": _yolo_model is not None,
        "model": yolo_model_path(),
        "error": _yolo_load_error,
    }


def run_yolo_inference(image_bytes: bytes) -> list[dict[str, Any]]:
    """將 JPEG 位元組解碼後執行 YOLOv8n，回傳偵測框列表。"""
    if not yolo_enabled():
        return []

    import cv2
    import numpy as np

    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    # 縮小影像以加快雲端 CPU 推理，降低 WebSocket 逾時斷線
    h, w = frame.shape[:2]
    max_side = 480
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        frame = cv2.resize(
            frame,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )

    model = _get_yolo_model()
    results = model(frame, verbose=False, imgsz=320)
    detections: list[dict[str, Any]] = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append(
                {
                    "label": label,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    return detections
