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


def openai_chat_model() -> str:
    return (os.environ.get("OPENAI_CHAT_MODEL") or "gpt-4o-mini").strip()
