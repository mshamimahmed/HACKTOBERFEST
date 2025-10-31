from __future__ import annotations
import csv
import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from app.services.semantic_service import normalize_and_expand
from app.services.embedding_service import embed_text

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "diseases.csv"
TREAT_OVR_PATH = DATA_DIR / "treatments_overrides.csv"
CACHE_EMB_PATH = DATA_DIR / "diseases_embeds.json"

_DB: List[Dict[str, Any]] = []
_EMB: Dict[str, List[float]] = {}


def _pick(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return str(value)


def _parse_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    # Try common Kaggle headers and fallbacks
    disease = _pick(row.get("Disease") or row.get("disease") or row.get("name"), f"disease-{idx}")
    symptoms = _pick(row.get("Symptoms") or row.get("symptoms") or row.get("symptom"))
    treatments = _pick(row.get("Treatments") or row.get("treatments") or row.get("therapy") or "")

    # Normalize symptoms (some datasets separate by pipe)
    if isinstance(symptoms, str):
        symptoms_text = symptoms.replace("|", ", ")
    else:
        symptoms_text = _pick(symptoms)

    # Parse treatments into a lightweight list of drug names/classes
    existing_drugs: List[Dict[str, Any]] = []
    if treatments:
        # split on commas/semicolons and trim
        import re
        parts = [p.strip() for p in re.split(r"[,;]", treatments) if p and p.strip()]
        for j, name in enumerate(parts):
            existing_drugs.append({
                "drug_id": f"csv-{idx}-{j}",
                "drug_name": name,
            })

    return {
        "disease_id": f"csv-{idx}",
        "disease_name": disease.strip(),
        "symptoms_text": symptoms_text.strip(),
        "existing_drugs": existing_drugs,
        "notes": "imported from CSV",
    }


def load_csv_if_exists() -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    global _DB, _EMB
    if _DB and _EMB:
        return _DB, _EMB
    if not CSV_PATH.exists():
        return [], {}
    rows: List[Dict[str, Any]] = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            try:
                rec = _parse_row(row, i)
                rows.append(rec)
            except Exception:
                continue
    _DB = rows

    # Optional: load treatments overrides and merge by disease name
    if TREAT_OVR_PATH.exists():
        try:
            import csv as _csv
            with TREAT_OVR_PATH.open("r", encoding="utf-8") as tf:
                reader = _csv.DictReader(tf)
                # Build map: normalized disease name -> list of drug names
                ovr: Dict[str, list[str]] = {}
                for r in reader:
                    name = _pick(r.get("Disease") or r.get("disease") or "").strip()
                    tx = _pick(r.get("Treatments") or r.get("treatments") or "")
                    if not name or not tx:
                        continue
                    import re as _re
                    parts = [p.strip() for p in _re.split(r"[,;]", tx) if p and p.strip()]
                    if parts:
                        ovr[name.lower()] = parts
                # apply overrides
                for rec in _DB:
                    key = rec.get("disease_name", "").strip().lower()
                    parts = ovr.get(key)
                    if parts:
                        rec["existing_drugs"] = [{"drug_id": f"ovr-{i}", "drug_name": nm} for i, nm in enumerate(parts)]
        except Exception:
            pass

    # embeddings
    _EMB = {}
    for rec in _DB:
        norm, _ = normalize_and_expand(rec.get("symptoms_text", ""))
        _EMB[rec["disease_id"]] = embed_text(norm)
    # optional cache to disk
    try:
        CACHE_EMB_PATH.write_text(json.dumps({k: v for k, v in _EMB.items()}))
    except Exception:
        pass
    return _DB, _EMB


def get_db_and_embeddings() -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    return load_csv_if_exists()
