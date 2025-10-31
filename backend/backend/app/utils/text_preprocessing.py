from __future__ import annotations
import re
from typing import List

DELIM_PATTERN = re.compile(r"[\n,;/]|\band\b|\bor\b", re.IGNORECASE)
WS_PATTERN = re.compile(r"\s+")

def split_text_to_phrases(text: str) -> List[str]:
    if not text:
        return []
    parts = DELIM_PATTERN.split(text)
    phrases: List[str] = []
    for p in parts:
        p = WS_PATTERN.sub(" ", p).strip().strip("-._")
        if p:
            phrases.append(p)
    return phrases

def normalize_phrase(p: str) -> str:
    return WS_PATTERN.sub(" ", p.strip().lower())
