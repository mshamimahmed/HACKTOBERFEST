from __future__ import annotations
import uuid
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.services.embedding_service import embed_text, cosine
from app.utils.text_preprocessing import normalize_phrase
from app.services.semantic_service import normalize_and_expand, infer_hypotheses, log_query
from app.services.disease_store import get_db_and_embeddings as get_csv_store

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "diseases_demo.json"

# In-memory caches
_DB: List[Dict[str, Any]] = []
_EMB: Dict[str, List[float]] = {}

STOP = {
    "the","and","or","with","of","to","in","for","on","a","an","is","are","was","were","it","its","at","by",
    "what","this","that","these","those","from","as","than","then","be","been","being","can","may","might","will","would","should","could",
    "i","am","not","feeling","due","because","have","has","had","very","severe","mild","moderate","like"
}


def _ensure_db() -> None:
    """
    Initialize in-memory demo disease DB and embeddings. If `diseases_demo.json`
    is missing, seed it deterministically. Optionally merge CSV-based diseases
    via `disease_store` if present.
    Side effects: populates module-level `_DB` and `_EMB`.
    """
    global _DB, _EMB
    if _DB:
        return
    if not DB_PATH.exists():
        # Seed expanded demo DB covering target test cases
        demo = [
            {
                "disease_id": "d-100",
                "disease_name": "Influenza-like Illness",
                "symptoms_text": "high fever dry cough sore throat muscle aches fatigue headache chills",
                "existing_drugs": [
                    {"drug_id": "dr-osa", "drug_name": "Oseltamivir"},
                    {"drug_id": "dr-ibp", "drug_name": "Ibuprofen"}
                ],
                "notes": "Flu-like illness"
            },
            {
                "disease_id": "d-110",
                "disease_name": "Common Cold",
                "symptoms_text": "tiredness headache runny nose cough sneezing mild fever congestion",
                "existing_drugs": [
                    {"drug_id": "dr-dxm", "drug_name": "Dextromethorphan"},
                    {"drug_id": "dr-ibp", "drug_name": "Ibuprofen"}
                ],
                "notes": "Viral upper respiratory infection"
            },
            {
                "disease_id": "d-120",
                "disease_name": "Asthma",
                "symptoms_text": "shortness of breath chest tightness wheezing cough nighttime symptoms",
                "existing_drugs": [
                    {"drug_id": "dr-salb", "drug_name": "Salbutamol"},
                    {"drug_id": "dr-ics", "drug_name": "Inhaled Corticosteroids"}
                ],
                "notes": "Airway hyperresponsiveness"
            },
            {
                "disease_id": "d-130",
                "disease_name": "Varicella (Chickenpox)",
                "symptoms_text": "itching rash vesicles blister painful blisters fever malaise",
                "existing_drugs": [
                    {"drug_id": "dr-acy", "drug_name": "Acyclovir"}
                ],
                "notes": "VZV primary infection"
            },
            {
                "disease_id": "d-131",
                "disease_name": "Herpes Zoster (Shingles)",
                "symptoms_text": "painful blisters rash burning tingling dermatomal vesicular eruption",
                "existing_drugs": [
                    {"drug_id": "dr-acy", "drug_name": "Acyclovir"}
                ],
                "notes": "VZV reactivation"
            },
            {
                "disease_id": "d-140",
                "disease_name": "Arthritis",
                "symptoms_text": "joint pain swelling redness stiffness limited range of motion",
                "existing_drugs": [
                    {"drug_id": "dr-nsaid", "drug_name": "NSAIDs"}
                ],
                "notes": "Joint inflammation"
            },
            {
                "disease_id": "d-150",
                "disease_name": "Gastroenteritis",
                "symptoms_text": "stomach pain diarrhea vomiting nausea abdominal cramps dehydration",
                "existing_drugs": [
                    {"drug_id": "dr-ors", "drug_name": "Oral Rehydration Salts"}
                ],
                "notes": "Stomach flu"
            },
            {
                "disease_id": "d-160",
                "disease_name": "COVID-19",
                "symptoms_text": "loss of smell fever cough fatigue sore throat headache",
                "existing_drugs": [],
                "notes": "SARS-CoV-2 infection"
            },
            {
                "disease_id": "d-170",
                "disease_name": "Allergic Rhinitis",
                "symptoms_text": "sneezing itchy eyes nasal congestion runny nose rhinorrhea",
                "existing_drugs": [
                    {"drug_id": "dr-antiH1", "drug_name": "Antihistamines"}
                ],
                "notes": "Allergic inflammation of nasal mucosa"
            },
            {
                "disease_id": "d-180",
                "disease_name": "Stroke",
                "symptoms_text": "numbness one side slurred speech facial droop weakness sudden onset",
                "existing_drugs": [],
                "notes": "Acute neurologic deficit"
            },
            {
                "disease_id": "d-190",
                "disease_name": "Diabetes Mellitus",
                "symptoms_text": "high blood sugar thirst frequent urination weight loss fatigue",
                "existing_drugs": [
                    {"drug_id": "dr-met", "drug_name": "Metformin"}
                ],
                "notes": "Hyperglycemia due to insulin dysfunction"
            }
        ]
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        DB_PATH.write_text(json.dumps(demo, indent=2))
    _DB = json.loads(DB_PATH.read_text())
    # Precompute embeddings
    _EMB = {}
    for rec in _DB:
        text = _normalize_text(rec.get("symptoms_text", ""))
        _EMB[rec["disease_id"]] = embed_text(text)
    # Load CSV diseases (if present) and merge
    try:
        csv_db, csv_emb = get_csv_store()
        if csv_db:
            _DB.extend(csv_db)
            _EMB.update(csv_emb)
    except Exception:
        # tolerate absence or parse errors silently
        pass

def _normalize_text(text: str) -> str:
    """Normalize and expand raw text using semantic_service rules."""
    norm, _ = normalize_and_expand(text)
    return norm


@router.post("/search")
async def researcher_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Semantic disease search for researcher workflow.

    - Input: `{ text: str, threshold?: float }`.
    - Process: normalize tokens, embed text, cosine similarity against DB.
    - Output: `{ query_id, query_embedding, matches[], hypotheses[], explanation? }`.
    """
    _ensure_db()
    raw_text = payload.get("text") or ""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    normalized, toks = normalize_and_expand(raw_text)
    q_vec = embed_text(normalized)

    threshold = float(payload.get("threshold") or 0.50)
    matches: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    best_score: float = -1.0
    best_overlap_tokens: List[str] = []

    # Precompute query tokens set for name matching
    q_tokens = set(toks)
    q_lower = normalized.lower()

    for rec in _DB:
        did = rec["disease_id"]
        score_sem = cosine(q_vec, _EMB.get(did, []))

        # Name-based similarity: token overlap and substring checks
        name = (rec.get("disease_name") or "").strip()
        name_norm = _normalize_text(name)
        name_tokens = set([t for t in name_norm.split() if t and t not in STOP])
        overlap = 0.0
        if name_tokens and q_tokens:
            inter = len(q_tokens & name_tokens)
            denom = max(1, len(name_tokens))
            overlap = inter / denom
        substring_boost = 0.0
        if name_norm and name_norm in q_lower or q_lower in name_norm:
            substring_boost = 0.95
        score_name = max(overlap, substring_boost)

        # Final score is the best of semantic and name-based
        score = max(score_sem, score_name)
        if score > best_score:
            best_score = score
            best = rec
            # track current overlap with normalized symptoms
            sym_norm_tmp = _normalize_text(rec.get("symptoms_text", ""))
            best_overlap_tokens = [k for k in toks if k in sym_norm_tmp]
        # More permissive threshold if the name matches strongly
        dyn_threshold = threshold
        if score_name >= 0.8:
            dyn_threshold = min(threshold, 0.2)
        if score >= dyn_threshold:
            # simple snippet: take intersection tokens
            sym = rec.get("symptoms_text", "")
            sym_norm = _normalize_text(sym)
            matched = [k for k in toks if k in sym_norm]
            if matched:
                snippet = ", ".join(matched[:6])
            elif score_name >= 0.8:
                snippet = f"name match: {name}"
            else:
                snippet = sym[:60]
            matches.append({
                "disease_id": did,
                "disease_name": rec.get("disease_name"),
                "match_score": round(float(score), 4),
                "matched_symptom_snippet": snippet,
                "existing_drugs": rec.get("existing_drugs", []),
                "notes": rec.get("notes", "")
            })
    # sort desc
    matches.sort(key=lambda m: m["match_score"], reverse=True)
    # If no matches, include best low-confidence candidate only if token overlap exists
    if not matches and best is not None and len(best_overlap_tokens) > 0:
        sym = best.get("symptoms_text", "")
        sym_norm = _normalize_text(sym)
        matched = [k for k in toks if k in sym_norm]
        snippet = ", ".join(matched[:6]) if matched else sym[:60]
        matches.append({
            "disease_id": best["disease_id"],
            "disease_name": best.get("disease_name"),
            "match_score": round(float(best_score), 4),
            "matched_symptom_snippet": snippet,
            "existing_drugs": best.get("existing_drugs", []),
            "notes": (best.get("notes", "") + " | Possible related condition — confidence low.").strip()
        })

    # Hypotheses inference if no strong matches or to augment context
    hyps = infer_hypotheses(normalized, toks)
    explanation = None
    if not matches and hyps:
        explanation = "No direct disease match at threshold. Providing physiological hypotheses based on semantic patterns."
    elif not matches and not hyps:
        explanation = "No direct disease match. Providing supportive wellness-oriented suggestions based on reported context."
    # Log ambiguous / unknown inputs
    if not matches:
        log_query({
            "type": "ambiguous_input",
            "text": raw_text,
            "normalized": normalized,
            "tokens": toks,
            "hypotheses": [h.get("hypothesis_id") for h in hyps],
        })

    return {
        "query_id": str(uuid.uuid4()),
        "query_embedding": q_vec,  # optional but included
        "matches": matches,
        "hypotheses": hyps,
        "explanation": explanation,
    }


@router.post("/search_symptoms")
async def researcher_search_symptoms(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Alias of `/search` using the same pipeline."""
    return await researcher_search(payload)


@router.post("/generate-composition")
async def researcher_generate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a suggested composition for a given `disease_id` or hypothesis id.
    If disease exists in DB, use simple rules + existing drugs; otherwise map
    hypothesis to supportive compounds.
    """
    _ensure_db()
    disease_id = payload.get("disease_id")
    if not disease_id:
        raise HTTPException(status_code=400, detail="disease_id is required")
    selected_symptoms: Optional[str] = payload.get("selected_symptoms")

    rec = next((r for r in _DB if r["disease_id"] == disease_id), None)
    # If not a known disease, allow hypothesis-driven supportive composition
    if not rec and isinstance(disease_id, str) and disease_id.startswith("hypothesis:"):
        # Map hypothesis to supportive compounds deterministically
        hyp_to_support = {
            "hypothesis:fatique_sleep": [
                {"ingredient": "Melatonin", "amount": "1-3 mg", "rationale": "sleep onset support", "confidence": 0.7},
                {"ingredient": "Magnesium", "amount": "200-400 mg", "rationale": "sleep quality, relaxation", "confidence": 0.65},
                {"ingredient": "Vitamin B12", "amount": "500 mcg", "rationale": "fatigue support", "confidence": 0.6},
            ],
            "hypothesis:postprandial_fatigue": [
                {"ingredient": "Chromium", "amount": "200 mcg", "rationale": "glucose regulation support", "confidence": 0.65},
                {"ingredient": "Berberine", "amount": "500 mg", "rationale": "glycemic control support", "confidence": 0.6},
                {"ingredient": "Fiber", "amount": "5-10 g", "rationale": "slows glucose absorption", "confidence": 0.6},
            ],
            "hypothesis:arrhythmia": [
                {"ingredient": "Magnesium", "amount": "200-400 mg", "rationale": "supports cardiac rhythm", "confidence": 0.6},
                {"ingredient": "Omega-3", "amount": "1 g", "rationale": "cardiovascular support", "confidence": 0.55},
                {"ingredient": "Electrolytes", "amount": "As needed", "rationale": "electrolyte balance", "confidence": 0.55},
            ],
            "hypothesis:orthostatic": [
                {"ingredient": "Oral Rehydration Salts", "amount": "As directed", "rationale": "volume expansion", "confidence": 0.7},
                {"ingredient": "Electrolytes", "amount": "As needed", "rationale": "BP support", "confidence": 0.6},
            ],
            "hypothesis:mood_motivation": [
                {"ingredient": "Omega-3", "amount": "1 g", "rationale": "mood support", "confidence": 0.6},
                {"ingredient": "B-Complex", "amount": "As per RDA", "rationale": "neurotransmitter co-factors", "confidence": 0.6},
                {"ingredient": "Rhodiola", "amount": "200-400 mg", "rationale": "stress adaptation", "confidence": 0.55},
            ],
            "hypothesis:pruritus": [
                {"ingredient": "Antihistamines", "amount": "Per label", "rationale": "histamine-mediated itch relief", "confidence": 0.6},
                {"ingredient": "Calamine", "amount": "Topical", "rationale": "soothes irritated skin", "confidence": 0.55},
                {"ingredient": "Hydrocortisone (topical)", "amount": "1% cream", "rationale": "reduces local inflammation", "confidence": 0.55},
            ],
        }
        comps = []
        options = hyp_to_support.get(disease_id, [])
        for opt in options:
            comps.append({
                "ingredient": opt["ingredient"],
                "from_drugs": [],
                "suggested_amount": opt["amount"],
                "rationale": opt["rationale"],
                "confidence": opt["confidence"],
            })
        return {
            "disease_id": disease_id,
            "composition": comps,
            "notes": "supportive composition generated from hypothesis mapping"
        }
    if not rec:
        raise HTTPException(status_code=404, detail="disease not found")

    # Deterministic mock composition from existing drugs
    comps: List[Dict[str, Any]] = []
    base = rec.get("existing_drugs") or []
    # Simple rules
    rules = [
        ("fever", {"ingredient": "Paracetamol", "amount": "500 mg", "rationale": "reduces fever and pain"}),
        ("headache", {"ingredient": "Sumatriptan", "amount": "50 mg", "rationale": "abortive for migraine"}),
        ("congestion", {"ingredient": "Pseudoephedrine", "amount": "60 mg", "rationale": "reduces nasal congestion"}),
        ("nausea", {"ingredient": "Ondansetron", "amount": "4 mg", "rationale": "reduces nausea/vomiting"}),
        ("diarrhea", {"ingredient": "Oral Rehydration Salts", "amount": "As directed", "rationale": "rehydration"}),
    ]
    sym_text = (selected_symptoms or rec.get("symptoms_text", "")).lower()
    used = set()
    for key, spec in rules:
        if key in sym_text and spec["ingredient"] not in used:
            used.add(spec["ingredient"])
            comps.append({
                "ingredient": spec["ingredient"],
                "from_drugs": [d["drug_name"] for d in base] if base else [],
                "suggested_amount": spec["amount"],
                "rationale": spec["rationale"],
                "confidence": 0.85 if key in ("fever","headache") else 0.65
            })
    if not comps:
        # generic fallback from existing drugs
        for d in base[:2]:
            comps.append({
                "ingredient": d["drug_name"],
                "from_drugs": [d["drug_name"]],
                "suggested_amount": "Refer label",
                "rationale": "derived from existing drugs",
                "confidence": 0.6
            })

    return {
        "disease_id": disease_id,
        "composition": comps,
        "notes": "composition generated from known drugs' ingredients and symptom mapping"
    }
