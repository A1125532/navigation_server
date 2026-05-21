"""過濾 Whisper 常見幻覺，與 Android TranscriptionSanity 對齊。"""

from __future__ import annotations

import re

_PHRASE_BLOCKLIST = (
    "subscribe",
    "subscription",
    "subscri",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "like and subscribe",
    "click subscribe",
    "amara.org",
    "subtitle",
    "subtitles",
    "caption",
    "captions",
    "transcribed by",
    "www.",
    "http://",
    "https://",
)

_REGEX_BLOCKLIST = (
    re.compile(r"订[阅閱]"),
    re.compile(r"訂[阅閱]"),
    re.compile(r"感[谢謝].*收[看視]"),
    re.compile(r"謝[谢].*觀看"),
    re.compile(r"謝[谢].*收看"),
    re.compile(r"请[请請].*订[阅閱]"),
    re.compile(r"請.*訂[阅閱]"),
    re.compile(r"字幕"),
    re.compile(r"點點欄目"),
    re.compile(r"明镜"),
    re.compile(r"明鏡"),
    re.compile(r"mbc", re.I),
    re.compile(r"südtirol", re.I),
)

_SHORT_NAV_ALLOWLIST = frozenset({"ok", "go", "gps", "ai"})


def _count_cjk(s: str) -> int:
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff" or "\u3400" <= ch <= "\u4dbf")


def is_unreliable_transcription(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    lower = t.lower()
    if any(p in lower for p in _PHRASE_BLOCKLIST):
        return True
    if any(r.search(t) for r in _REGEX_BLOCKLIST):
        return True
    if "字幕" in t or "subtitle" in lower:
        return True
    cjk = _count_cjk(t)
    letters = sum(1 for ch in t if ch.isalpha())
    if letters >= 6 and cjk == 0 and lower not in _SHORT_NAV_ALLOWLIST:
        return True
    if len(t) <= 2 and cjk == 0 and letters > 0:
        return True
    return False
