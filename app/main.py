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
from typing import Any, Optional

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import app.config  # noqa: F401 — 載入 .env
from app.ai_logic import handle_ai
from app.config import openai_api_key

app = FastAPI(title="Navigation Pi Backend", version="0.2.0")

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
            )
            t = (getattr(result, "text", None) or "").strip()
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
    from app.config import google_maps_api_key

    return {
        "openai_configured": bool(openai_api_key()),
        "google_maps_configured": bool(google_maps_api_key()),
        "env_file_hint": "navigation-server/.env（可從 config.example.env 複製）",
    }
