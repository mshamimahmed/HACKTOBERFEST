from __future__ import annotations
import json
from typing import List, Dict
from pathlib import Path

import requests

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

A_Z_URL = "https://dph.illinois.gov/topics-services/diseases-and-conditions/diseases-a-z-list.html"
HEADERS = {"User-Agent": "SymptomRepurposeBot/1.0 (contact: dev@example.com)"}


def fetch_html(url: str, timeout: int = 20) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def extract_disease_links(index_html: str) -> List[Dict[str, str]]:
    if not BeautifulSoup:
        raise RuntimeError("beautifulsoup4 is required to scrape Illinois DPH site")
    soup = BeautifulSoup(index_html, "html.parser")
    links: List[Dict[str, str]] = []
    # The page has an A-Z listing with anchors; collect anchors under main content
    for a in soup.select('a'):
        href = a.get('href') or ''
        text = a.get_text(strip=True)
        if not href or not text:
            continue
        if href.startswith('/topics-services/diseases-and-conditions/') and href.endswith('.html'):
            links.append({"name": text, "url": f"https://dph.illinois.gov{href}"})
    # Deduplicate by name
    seen = set()
    uniq: List[Dict[str, str]] = []
    for item in links:
        key = item["name"].lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def parse_disease_page(html: str) -> str:
    if not BeautifulSoup:
        raise RuntimeError("beautifulsoup4 is required to scrape Illinois DPH site")
    soup = BeautifulSoup(html, "html.parser")
    # Collect paragraphs within main content
    main = soup.find('main') or soup
    paras: List[str] = []
    for p in main.find_all('p'):
        txt = p.get_text(' ', strip=True)
        if txt:
            paras.append(txt)
    text = ' '.join(paras)
    return text[:2000]


def scrape_to_json(out_path: Path, limit: int | None = None) -> Dict[str, Dict[str, str]]:
    index_html = fetch_html(A_Z_URL)
    links = extract_disease_links(index_html)
    if limit:
        links = links[:limit]
    data: Dict[str, Dict[str, str]] = {}
    for item in links:
        try:
            html = fetch_html(item['url'])
            desc = parse_disease_page(html)
        except Exception:
            desc = ""
        name = item['name'].strip()
        data[name] = {"name": name, "description": desc, "source": item['url']}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data
