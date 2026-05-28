"""
與 Android PiNavApi 對齊的導航後端（參考 d:\\Pi）。
  POST /transcribe  multipart file -> { "text", "error" }
  POST /ai          JSON -> { "reply", "intent", "origin_used?" }

設定：複製 config.example.env 為 .env 或 config.env（勿提交），填入兩把金鑰。

執行：uvicorn app.main:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import asyncio
import io
import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import Body, FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import app.config  # noqa: F401 — 載入 .env
from app.ai_logic import handle_ai, run_yolo_inference, warmup_yolo_model, yolo_status
from app.config import openai_api_key, yolo_enabled
from app.transcribe_sanity import is_unreliable_transcription

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """雲端啟動時預載 YOLO，避免第一幀 WebSocket 才下載模型。"""
    if yolo_enabled():
        info = await asyncio.to_thread(warmup_yolo_model)
        logger.info("YOLO warmup: %s", info)
    yield


app = FastAPI(title="Navigation Pi Backend", version="0.3.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, Any]:
    """瀏覽器開首頁時可看見服務與常用路徑（避免只看到 Not Found）。"""
    return {
        "service": "Navigation Pi Backend",
        "version": app.version,
        "paths": {
            "docs": "/docs",
            "health": "GET /health",
            "config_status": "GET /config-status",
            "transcribe": "POST /transcribe",
            "ai": "POST /ai",
            "video_ws": "WS /ws/video",
        },
    }


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)) -> dict[str, Optional[str]]:
    data = await file.read()
    if len(data) < 200:
        return {"text": "", "error": "audio_too_short"}

    key = openai_api_key()
    if not key:
        return {
            "text": "",
            "error": "在 .env 設定 OPENAI_API_KEY 以啟用 Whisper 語音辨識（見 config.example.env）。",
        }

    def _run() -> dict[str, Optional[str]]:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=key)
            bio = io.BytesIO(data)
            bio.name = file.filename or "audio.wav"
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
                language="zh",
                prompt="語音導航：使用者說出目的地或導航指令，例如帶我去台中車站、開始導航。",
            )
            t = (getattr(result, "text", None) or "").strip()
            if t and is_unreliable_transcription(t):
                return {"text": "", "error": "unreliable_transcription"}
            return {"text": t, "error": None if t else "empty_transcription"}
        except Exception as e:  # noqa: BLE001
            return {"text": "", "error": str(e)}

    return await asyncio.to_thread(_run)


@app.post("/ai")
async def ai(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    return await asyncio.to_thread(handle_ai, payload)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/config-status")
async def config_status() -> dict[str, Any]:
    """不洩漏金鑰，只回報是否已設定（方便本機檢查）。"""
    from app.config import firebase_edges_collection, firebase_nodes_collection, firebase_project_id, google_maps_api_key

    return {
        "openai_configured": bool(openai_api_key()),
        "google_maps_configured": bool(google_maps_api_key()),
        "firebase_route_configured": bool(firebase_project_id()),
        "firebase_nodes_collection": firebase_nodes_collection(),
        "firebase_edges_collection": firebase_edges_collection(),
        "yolo": yolo_status(),
        "video_ws": "wss://<your-host>/ws/video" if yolo_enabled() else None,
        "env_file_hint": "navigation-server/.env（可從 config.example.env 複製）",
    }


@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket) -> None:
    """接收眼鏡 Live Video 送來的 JPEG 幀，回傳 YOLO 偵測結果（與導航同一雲端主機）。"""
    await websocket.accept()
    if not yolo_enabled():
        await websocket.send_json({"objects": [], "error": "yolo_disabled"})
        await websocket.close()
        return
    infer_timeout = 45.0
    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(run_yolo_inference, data),
                    timeout=infer_timeout,
                )
                await websocket.send_json(results)
            except asyncio.TimeoutError:
                logger.warning("YOLO inference timeout (%.0fs)", infer_timeout)
                await websocket.send_json(
                    {"objects": [], "error": "inference_timeout"}
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("YOLO inference failed")
                await websocket.send_json({"objects": [], "error": str(e)})
    except WebSocketDisconnect:
        pass
