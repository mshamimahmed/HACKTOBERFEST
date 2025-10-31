from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Any, List

from fastapi import APIRouter, Query, HTTPException

router = APIRouter()

BASE = "https://dph.illinois.gov"
# Known model JSON endpoints discovered from tabs on the A–Z page
MODEL_ENDPOINTS = [
    "/content/soi/idph/en/topics-services/diseases-and-conditions/diseases-a-z-list/jcr:content/responsivegrid/container/container/container/container/tabs/item_1623812502092/list.model.json",
    "/content/soi/idph/en/topics-services/diseases-and-conditions/diseases-a-z-list/jcr:content/responsivegrid/container/container/container/container/tabs/item_1623813770520/list_copy.model.json",
    "/content/soi/idph/en/topics-services/diseases-and-conditions/diseases-a-z-list/jcr:content/responsivegrid/container/container/container/container/tabs/item_1623813783052/list_copy.model.json",
    "/content/soi/idph/en/topics-services/diseases-and-conditions/diseases-a-z-list/jcr:content/responsivegrid/container/container/container/container/tabs/item_1623813792102/list_copy.model.json",
]

HEADERS = {"User-Agent": "SymptomRepurposeBot/1.0 (contact: dev@example.com)"}
TIMEOUT = 20
CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_PATH = CACHE_DIR / "illinois_index.json"
ANALYZER_PATH = CACHE_DIR / "illinois_dph.json"


def _fetch_json(url: str) -> Dict[str, Any] | List[Any]:
    """
    GET a JSON payload from the given URL with default headers/timeouts.
    Returns parsed JSON (dict or list). Raises HTTP errors on failure.
    """
    try:
        import requests  # type: ignore
    except Exception:
        raise HTTPException(status_code=500, detail="'requests' package not installed. Install it to use /illinois endpoints.")
    resp = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize a list item from the IDPH model JSON to a compact dict
    containing `name`, `title`, `description`, and absolute `url`.
    """
    title = (item.get("title") or item.get("name") or "").strip()
    desc = (item.get("description") or "").strip()
    url = item.get("url") or ""
    if url and url.startswith("/"):
        url = BASE + url
    name = (item.get("name") or title).strip()
    return {
        "name": name,
        "title": title,
        "description": desc,
        "url": url,
    }


def _gather_all() -> List[Dict[str, str]]:
    """
    Fetch all configured model.json endpoints and collect unique disease entries
    from their `listItems`. Deduplicates by lowercased `name`.
    """
    all_items: List[Dict[str, str]] = []
    seen = set()
    for path in MODEL_ENDPOINTS:
        try:
            data = _fetch_json(BASE + path)
        except Exception:
            continue
        # The payloads have listItems: [...] with title/description/url
        items = data.get("listItems") if isinstance(data, dict) else None
        if not isinstance(items, list):
            continue
        for it in items:
            norm = _normalize_item(it)
            key = norm["name"].lower()
            if key in seen:
                continue
            seen.add(key)
            all_items.append(norm)
    return all_items


@router.post("/sync")
async def sync() -> Dict[str, Any]:
    """
    Pull the Diseases A–Z lists from configured IDPH model endpoints and cache
    into `app/data/illinois_*`. Returns counts and cache paths.
    """
    items = _gather_all()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Save index list
    CACHE_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    # Save analyzer-compatible JSON (list is fine; analyzer accepts list or dict)
    ANALYZER_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False))
    return {"count": len(items), "cache": str(CACHE_PATH), "analyzer_cache": str(ANALYZER_PATH)}


def _load_cache() -> List[Dict[str, str]]:
    """
    Load the cached Illinois entries from disk, tolerating both list and dict
    formats, and return a normalized list variant.
    """
    if not CACHE_PATH.exists():
        return []
    try:
        data = json.loads(CACHE_PATH.read_text())
        if isinstance(data, list):
            return [
                {
                    "name": (d.get("name") or "").strip(),
                    "title": (d.get("title") or "").strip(),
                    "description": (d.get("description") or "").strip(),
                    "url": (d.get("url") or "").strip(),
                }
                for d in data
                if isinstance(d, dict)
            ]
    except Exception:
        return []
    return []


@router.get("/search")
async def search(q: str = Query(..., min_length=1), limit: int = Query(20, ge=1, le=100)) -> Dict[str, Any]:
    """
    Case-insensitive substring search across name/title/description of cached
    Illinois entries. Returns up to `limit` items.
    """
    items = _load_cache()
    if not items:
        return {"count": 0, "results": [], "message": "Cache empty. POST /illinois/sync first."}
    ql = q.strip().lower()
    # Simple substring match in name/title/description
    results: List[Dict[str, str]] = []
    for it in items:
        blob = f"{it['name']} {it['title']} {it['description']}".lower()
        if ql in blob:
            results.append(it)
            if len(results) >= limit:
                break
    return {"count": len(results), "results": results}
