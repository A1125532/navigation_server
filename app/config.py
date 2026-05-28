"""
從專案根目錄（navigation-server/）的 .env 讀取設定；勿把 .env 提交到 Git。
參考舊專案 Pi：OPENAI_API_KEY、GOOGLE_MAPS_API_KEY。
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv

    # 本機：可擇一使用 config.env 或 .env；若兩者皆有，.env 覆寫同名變數。
    _cfg = _ROOT / "config.env"
    if _cfg.is_file():
        load_dotenv(_cfg, override=False)
    load_dotenv(_ROOT / ".env", override=True)
except ImportError:
    pass


def openai_api_key() -> str | None:
    k = (os.environ.get("OPENAI_API_KEY") or "").strip()
    return k or None


def google_maps_api_key() -> str | None:
    k = (os.environ.get("GOOGLE_MAPS_API_KEY") or "").strip()
    return k or None


def firebase_project_id() -> str | None:
    k = (os.environ.get("FIREBASE_PROJECT_ID") or "").strip()
    return k or None


def firebase_api_key() -> str | None:
    k = (os.environ.get("FIREBASE_API_KEY") or "").strip()
    return k or None


def firebase_nodes_collection() -> str:
    return (os.environ.get("FIREBASE_NODES_COLLECTION") or "safe_places").strip()


def firebase_edges_collection() -> str:
    return (os.environ.get("FIREBASE_EDGES_COLLECTION") or "nuk_edges_new").strip()


def firebase_environment_collections() -> list[str]:
    raw = (os.environ.get("FIREBASE_ENVIRONMENT_COLLECTIONS") or "safe_places,nuk").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def local_nodes_path() -> Path:
    raw = (os.environ.get("LOCAL_NODES_PATH") or "data/nuk_nodes_new.json").strip()
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def local_edges_path() -> Path:
    raw = (os.environ.get("LOCAL_EDGES_PATH") or "data/nuk_edges_new.json").strip()
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def local_environment_points_path() -> Path:
    raw = (os.environ.get("LOCAL_ENVIRONMENT_POINTS_PATH") or "data/environment_points.json").strip()
    path = Path(raw)
    return path if path.is_absolute() else _ROOT / path


def openai_chat_model() -> str:
    return (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()


def yolo_enabled() -> bool:
    """是否啟用 YOLO 即時辨識（WS /ws/video）。雲端與本機皆預設開啟。"""
    v = (os.environ.get("YOLO_ENABLED") or "true").strip().lower()
    return v not in ("0", "false", "no", "off")


def yolo_model_path() -> str:
    return (os.environ.get("YOLO_MODEL") or "yolov8n.pt").strip()
