from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from pathlib import Path
from app.services.analyzer_service import get_analyzer
try:
    from rapidfuzz import process, fuzz  # type: ignore
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False

router = APIRouter()

class Demographics(BaseModel):
    age: Optional[int] = None
    sex: Optional[str] = None

class DifferentialReq(BaseModel):
    symptoms: List[str]
    demographics: Optional[Demographics] = None

@router.post('/differential')
async def differential(req: DifferentialReq):
    """
    Compute a simple differential diagnosis from a list of symptom strings.

    - Input: `DifferentialReq` with `symptoms` and optional `demographics`.
    - Process: normalize symptoms (synonyms, heuristics, optional fuzzy match) and
      score overlap against a small rule-based disease keyword table.
    - Output: dict with `differentials` (top diseases + confidence),
      `suggestedDrugs` (flat list for the top disease), and `suggestedByDisease`
      (grouped suggestions for each returned disease).
    """
    # Simple rule-based scorer for demo purposes.
    # Map diseases to indicative symptom keywords.
    disease_rules = {
        ("icd11:1A00", "Fever"): {"fever", "pyrexia", "temperature", "chills", "sweats"},
        ("icd11:CA40", "Cough-related infection"): {"cough", "sputum", "productive cough", "cold", "bronchitis"},
        ("icd11:CA01", "Upper respiratory infection"): {"sore throat", "runny nose", "rhinorrhea", "congestion", "sneezing"},
        ("icd11:5A11", "Migraine"): {"headache", "migraine", "photophobia", "nausea"},
        ("icd11:2C21", "Gastroenteritis"): {"diarrhea", "vomiting", "nausea", "stomach pain", "abdominal pain", "abdominal cramps"},
        ("icd11:DA11", "Gastroesophageal reflux disease"): {"heartburn", "acid reflux", "stomach burning", "regurgitation", "sour taste"},
        ("icd11:DA9Z", "Gastritis"): {"stomach burning", "epigastric pain", "abdominal pain", "nausea"},
        ("icd11:FA00", "Knee pain"): {"knee pain", "knee swelling", "joint pain", "arthralgia"},
        ("icd11:EB01", "Androgenetic alopecia"): {"hair loss", "thinning hair", "receding hairline", "male pattern baldness", "female pattern hair loss"},
        ("icd11:EB02", "Alopecia areata"): {"hair loss", "patchy hair loss", "bald patch", "autoimmune hair loss"},
        ("icd11:1C13", "Measles"): {"measles", "fever", "cough", "coryza", "conjunctivitis", "koplik spots", "rash"},
    }

    # Simple medication suggestions mapped by disease label
    drug_suggestions = {
        "Fever": [
            {"drugId": "DB00316", "name": "Acetaminophen", "class": "Antipyretic", "rationale": "Guideline-based for fever", "contraindications": ["Severe liver disease"]}
        ],
        "Cough-related infection": [
            {"drugId": "DB01393", "name": "Dextromethorphan", "class": "Antitussive", "rationale": "Symptomatic cough relief", "contraindications": ["MAOI use"]}
        ],
        "Upper respiratory infection": [
            {"drugId": "DB00878", "name": "Ibuprofen", "class": "NSAID", "rationale": "Pain/fever relief", "contraindications": ["GI ulcer", "CKD"]}
        ],
        "Migraine": [
            {"drugId": "DB00945", "name": "Sumatriptan", "class": "Triptan", "rationale": "Acute migraine management", "contraindications": ["CAD", "Uncontrolled HTN"]}
        ],
        "Gastroenteritis": [
            {"drugId": "DB00836", "name": "Oral rehydration salts", "class": "Electrolyte solution", "rationale": "First-line rehydration", "contraindications": []}
        ],
        "Gastroesophageal reflux disease": [
            {"drugId": "DB00338", "name": "Omeprazole", "class": "PPI", "rationale": "Acid suppression for heartburn", "contraindications": ["Hypersensitivity"]},
            {"drugId": "DB01374", "name": "Aluminum hydroxide/Magnesium hydroxide", "class": "Antacid", "rationale": "Symptom relief", "contraindications": ["Severe renal impairment"]}
        ],
        "Gastritis": [
            {"drugId": "DB00338", "name": "Omeprazole", "class": "PPI", "rationale": "Acid suppression", "contraindications": ["Hypersensitivity"]}
        ],
        "Knee pain": [
            {"drugId": "DB01050", "name": "Topical Diclofenac", "class": "NSAID topical", "rationale": "Localized musculoskeletal pain", "contraindications": ["NSAID allergy", "Open wounds"]},
            {"drugId": "DB01009", "name": "Ibuprofen", "class": "NSAID", "rationale": "Analgesia", "contraindications": ["GI ulcer", "CKD"]}
        ],
        "Androgenetic alopecia": [
            {"drugId": "DB00350", "name": "Minoxidil (topical)", "class": "Vasodilator", "rationale": "Promotes hair growth", "contraindications": ["Scalp irritation"]},
            {"drugId": "DB01216", "name": "Finasteride", "class": "5-alpha-reductase inhibitor", "rationale": "Male pattern hair loss", "contraindications": ["Pregnancy", "Women of childbearing potential"]}
        ],
        "Alopecia areata": [
            {"drugId": "DB00835", "name": "Topical corticosteroids", "class": "Corticosteroid", "rationale": "First-line immune suppression", "contraindications": ["Skin infection"]}
        ],
        "Measles": [
            {"drugId": "DB00316", "name": "Acetaminophen", "class": "Antipyretic", "rationale": "Fever control and comfort", "contraindications": ["Severe liver disease"]},
            {"drugId": "N/A", "name": "Hydration and rest", "class": "Supportive care", "rationale": "Standard supportive management", "contraindications": []}
        ],
    }

    # Normalize inputs
    raw_syms = [s.strip().lower() for s in (req.symptoms or []) if isinstance(s, str)]
    # Pre-tokenize: split phrases like "eye pain and back side head is paining"
    # into smaller symptom chunks using common delimiters and connector words.
    import re
    tokens: List[str] = []
    for s in raw_syms:
        parts = re.split(r"[;,/]|\b(and|with|along with|plus)\b", s)
        for p in parts:
            if not p or p.strip() in {"and", "with", "along with", "plus"}:
                continue
            p = p.strip()
            if p:
                tokens.append(p)
    raw_syms = tokens

    # Small synonym map (expandable)
    synonym_map = {
        "head pain": "headache",
        "cephalgia": "headache",
        "eye pain": "headache",
        "stomach pain": "abdominal pain",
        "stomach burning": "heartburn",
        "burning stomach": "heartburn",
        "acid reflux": "heartburn",
        "acidity": "heartburn",
        "heart burn": "heartburn",
        "tummy ache": "abdominal pain",
        "runny nose": "rhinorrhea",
        "blocked nose": "congestion",
        "high temperature": "fever",
        "raised temperature": "fever",
        "vomit": "vomiting",
        "keen pain": "knee pain",
        "hair fall": "hair loss",
        "hairfall": "hair loss",
        "hair shedding": "hair loss",
        "thinning hair": "hair loss",
        "bald patch": "patchy hair loss",
        "rubeola": "measles",
    }

    # Build keyword universe for fuzzy matching
    keyword_universe = set()
    for kw_set in disease_rules.values():
        keyword_universe.update(kw_set)
    keyword_list = list(keyword_universe)

    normalized = []
    for s in raw_syms:
        # heuristic patterns
        if ("head" in s and ("pain" in s or "paining" in s or "ache" in s)):
            normalized.append("headache")
            continue
        if ("eye" in s and ("pain" in s or "strain" in s)):
            normalized.append("headache")
            continue
        if ("hair" in s) and ("fall" in s or "loss" in s or "thinning" in s or "shedding" in s):
            normalized.append("hair loss")
            continue
        # exact synonym mapping
        if s in synonym_map:
            normalized.append(synonym_map[s])
            continue
        # substring synonym mapping (e.g., "keen pain with" contains "keen pain")
        matched_sub = None
        for key, canon in synonym_map.items():
            if key in s:
                matched_sub = canon
                break
        if matched_sub:
            normalized.append(matched_sub)
            continue
        # fuzzy map to closest known keyword if available
        if HAVE_FUZZ and keyword_list:
            match = process.extractOne(s, keyword_list, scorer=fuzz.QRatio)
            if match and match[1] >= 75:  # slightly lower similarity threshold
                normalized.append(match[0])
                continue
        # default: keep as-is
        normalized.append(s)

    syms = set(normalized)
    if not syms:
        return {"differentials": [], "suggestedDrugs": []}

    # Score diseases by overlap
    scored = []
    for (d_id, label), kw in list(disease_rules.items()):
        hits = len(syms & kw)
        if hits:
            scored.append({"diseaseId": d_id, "label": label, "score": float(hits) / len(kw)})

    # If nothing matched, return a generic low-confidence entry
    if not scored:
        differentials = [{"diseaseId": "icd11:XA00", "label": "Non-specific presentation", "confidence": 0.2}]
        suggested = []
        return {"differentials": differentials, "suggestedDrugs": suggested}

    # Normalize scores to confidences
    max_score = max(x["score"] for x in scored) or 1.0
    differentials = [
        {"diseaseId": x["diseaseId"], "label": x["label"], "confidence": round(x["score"] / max_score, 3)}
        for x in sorted(scored, key=lambda y: y["score"], reverse=True)[:5]
    ]
    # Build grouped suggestions for each differential
    suggested_by_disease = []
    for d in differentials:
        label = d["label"]
        suggested_by_disease.append({
            "diseaseId": d["diseaseId"],
            "label": label,
            "drugs": drug_suggestions.get(label, [])
        })

    # Keep backward-compatible flat list using the top disease
    top_label = differentials[0]["label"]
    suggested_flat = drug_suggestions.get(top_label, [])

    return {"differentials": differentials, "suggestedDrugs": suggested_flat, "suggestedByDisease": suggested_by_disease}


class AnalyzeReq(BaseModel):
    text: str

@router.post('/analyze')
async def analyze(req: AnalyzeReq) -> Dict[str, Any]:
    """
    Analyze free-text symptom description using the shared `SymptomAnalyzer`.

    - Input: `{ text: str }` free-text.
    - Output: list of per-phrase analyses with normalized symptom,
      possible diseases, and next action hints.
    """
    analyzer = get_analyzer()
    results = analyzer.analyze_text(req.text)
    return {"results": results}


# ---- Array-based search over Illinois cache ----

class SearchArrayReq(BaseModel):
    text: str
    limit: Optional[int] = 25

_IL_CACHE_PATH = Path(__file__).resolve().parent.parent / 'data' / 'illinois_dph.json'
_IL_CACHE: List[Dict[str, Any]] | None = None


def _ensure_il_cache() -> List[Dict[str, Any]]:
    """
    Load and cache the Illinois DPH disease list from app/data/illinois_dph.json.
    Accepts both dict and list formats for backward compatibility.
    Returns the cached list version.
    """
    global _IL_CACHE
    if _IL_CACHE is not None:
        return _IL_CACHE
    if not _IL_CACHE_PATH.exists():
        _IL_CACHE = []
        return _IL_CACHE
    try:
        data = __import__('json').loads(_IL_CACHE_PATH.read_text())
        if isinstance(data, dict):
            # convert dict to list
            _IL_CACHE = [
                {"name": k, "title": v.get("title") or k, "description": v.get("description") or "", "url": v.get("url") or v.get("source") or ""}
                for k, v in data.items() if isinstance(v, dict)
            ]
        elif isinstance(data, list):
            _IL_CACHE = [
                {
                    "name": (d.get("name") or "").strip(),
                    "title": (d.get("title") or d.get("name") or "").strip(),
                    "description": (d.get("description") or "").strip(),
                    "url": (d.get("url") or d.get("source") or "").strip(),
                }
                for d in data if isinstance(d, dict)
            ]
        else:
            _IL_CACHE = []
    except Exception:
        _IL_CACHE = []
    return _IL_CACHE


def _tokenize(text: str) -> List[str]:
    """
    Lightweight tokenizer: lowercase, split on non-alphanumerics, remove common
    stopwords, and de-duplicate while keeping order. Used for KB search.
    """
    import re
    stop = {"the","and","or","with","of","to","in","for","on","a","an","is","are","was","were","it","its","at","by","what","this","that","these","those","from","as","than","then","be","been","being","can","may","might","will","would","should","could","pain","stomach","disease","cancer","fever","infection","illness","symptoms","i","am","not","feeling","good","due","because"}
    toks = [t for t in re.split(r"[^a-z0-9]+", (text or '').lower()) if t and t not in stop]
    # dedupe while keeping order
    seen = set()
    out = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


@router.post('/search-array')
async def search_array(req: SearchArrayReq) -> Dict[str, Any]:
    """
    Search the Illinois cache using token/phrase overlap and optional fuzzy
    matching. Returns ranked results with match diagnostics.

    - Input: `{ text, limit }`
    - Output: `{ count, results[], query_tokens }`
    """
    items = _ensure_il_cache()
    if not items:
        return {"count": 0, "results": [], "message": "Illinois cache missing. Call POST /illinois/sync first."}
    # simple normalization for common synonyms/typos before tokenizing
    raw = (req.text or "").lower()
    # map "keen pain" -> "knee pain"
    raw = raw.replace("keen pain", "knee pain")
    q_tokens = _tokenize(raw)
    if not q_tokens:
        return {"count": 0, "results": []}
    # build bigram phrases for exact phrase boost/filter
    phrases: List[str] = []
    for i in range(len(q_tokens)-1):
        phrases.append(f"{q_tokens[i]} {q_tokens[i+1]}")
    # if core symptom tokens are present, allow single-token matches
    core = {"cold","cough","headache","migraine","nausea","vomiting","diarrhea","sore","throat","runny","nose","congestion","sneezing","chills","fatigue"}
    core_present = any(t in core for t in q_tokens)
    query_text = raw
    results = []
    for it in items:
        blob = f"{it.get('name','')} {it.get('title','')} {it.get('description','')}".lower()
        matched = [t for t in q_tokens if t in blob]
        phrase_hit = any(p in blob for p in phrases)
        # require at least 2 token matches or a phrase match; if core symptoms present, allow 1
        required = 1 if core_present else 2
        base_overlap = (len(matched) / max(1, len(q_tokens)))
        base_score = min(1.0, base_overlap + (0.2 if phrase_hit else 0.0))
        fuzzy_score = 0.0
        if 'HAVE_FUZZ' in globals() and HAVE_FUZZ:
            try:
                # token_set_ratio captures unordered overlap; partial_ratio helps substrings
                s1 = fuzz.token_set_ratio(query_text, blob) / 100.0
                s2 = fuzz.partial_ratio(query_text, blob) / 100.0
                fuzzy_score = max(s1, s2)
            except Exception:
                fuzzy_score = 0.0
        # combined score prefers fuzzy when available
        score = 0.7 * (fuzzy_score if fuzzy_score > 0 else base_score) + 0.3 * base_score
        # gating: either threshold or classic token requirement
        threshold = 0.55 if (('HAVE_FUZZ' in globals() and HAVE_FUZZ)) else 0.0
        if not phrase_hit and len(matched) < required and score < threshold:
            continue
        match_percent = round(score * 100.0, 1)
        title = it.get("title") or it.get("name") or ""
        # known medications (limited set aligned with differential rules)
        meds = _known_meds_for(title)
        # composition fallback
        comps = _composition_suggestions(title, it.get("description") or "") if not meds else []
        results.append({
            "name": it.get("name"),
            "title": title,
            "url": it.get("url"),
            "description": (it.get("description") or "")[:240],
            "matched_tokens": matched,
            "match_percent": match_percent,
            "score": score,
            "fuzzy_score": fuzzy_score,
            "medications": meds,
            "composition_suggestions": comps,
        })
    results.sort(key=lambda r: (r["match_percent"], len(r["matched_tokens"]), len(r.get("description",""))), reverse=True)
    limit = req.limit or 25
    results = results[:limit]
    return {"count": len(results), "results": results, "query_tokens": q_tokens}


# ---- Disease Lookup by name ----

class LookupReq(BaseModel):
    name: str


def _find_in_il_cache(name: str) -> Dict[str, Any] | None:
    """
    Try to find an Illinois cache entry by name/title or substring. Returns the
    entry dict augmented with `source: 'illinois'` when found.
    """
    items = _ensure_il_cache()
    if not items:
        return None
    n = (name or '').strip().lower()
    if not n:
        return None
    # exact name match first
    for it in items:
        if it.get('name', '').strip().lower() == n:
            return {**it, 'source': 'illinois'}
    # title exact
    for it in items:
        if it.get('title', '').strip().lower() == n:
            return {**it, 'source': 'illinois'}
    # substring in title/name
    for it in items:
        blob = f"{it.get('name','')} {it.get('title','')}".lower()
        if n in blob:
            return {**it, 'source': 'illinois'}
    return None


def _composition_suggestions(name: str, description: str) -> List[Dict[str, Any]]:
    """
    Heuristic composition suggestions based on keywords in disease name/desc.
    Intended as educational placeholders, not clinical guidance.
    """
    text = f"{name} {description}".lower()
    suggestions: List[Dict[str, Any]] = []
    def add(cls: str, actives: List[str], caution: List[str] | None = None):
        suggestions.append({"class": cls, "actives": actives, "caution": caution or []})
    if any(k in text for k in ["acne", "comedone", "pimple"]):
        add("Topical acne therapy", ["Benzoyl peroxide", "Adapalene"], ["Skin irritation", "Pregnancy caution for retinoids"])
    if any(k in text for k in ["fever", "pyrexia"]):
        add("Antipyretic", ["Acetaminophen"], ["Severe liver disease"])
    if any(k in text for k in ["fungal", "tinea", "ringworm", "candid"]):
        add("Topical antifungal", ["Clotrimazole 1%", "Miconazole 2%"], ["Open wounds", "Severe irritation"])
    if any(k in text for k in ["allergy", "urticaria", "rhinitis"]):
        add("Antihistamine", ["Cetirizine", "Loratadine"], ["Sedation (some agents)"])
    if any(k in text for k in ["pain", "arthralgia", "muscle strain", "sprain"]):
        add("Analgesic/NSAID", ["Ibuprofen", "Topical diclofenac"], ["GI ulcer", "CKD", "NSAID allergy"])
    if any(k in text for k in ["diarrhea", "gastroenteritis"]):
        add("Rehydration / Antidiarrheal", ["Oral rehydration salts", "Loperamide (no fever/bloody stools)"])
    return suggestions


def _known_meds_for(label: str) -> List[Dict[str, Any]]:
    """
    Return known medication suggestions for a few supported disease labels.
    Used by KB search results to surface typical symptom-relief agents.
    """
    l = (label or '').strip().lower()
    table: Dict[str, List[Dict[str, Any]]] = {
        "fever": [
            {"drugId": "DB00316", "name": "Acetaminophen", "class": "Antipyretic", "rationale": "Guideline-based for fever", "contraindications": ["Severe liver disease"]}
        ],
        "cough-related infection": [
            {"drugId": "DB01393", "name": "Dextromethorphan", "class": "Antitussive", "rationale": "Symptomatic cough relief", "contraindications": ["MAOI use"]}
        ],
        "upper respiratory infection": [
            {"drugId": "DB01393", "name": "Dextromethorphan", "class": "Antitussive", "rationale": "URI cough symptomatic relief", "contraindications": ["MAOI use"]}
        ],
        "migraine": [
            {"drugId": "DB00945", "name": "Sumatriptan", "class": "Triptan", "rationale": "Abortive therapy", "contraindications": ["Uncontrolled HTN", "CAD"]}
        ],
        "gastroenteritis": [
            {"drugId": "N/A", "name": "Oral rehydration salts", "class": "Rehydration", "rationale": "Prevent dehydration", "contraindications": []}
        ],
        "gastroesophageal reflux disease": [
            {"drugId": "DB00338", "name": "Omeprazole", "class": "PPI", "rationale": "Acid suppression for heartburn", "contraindications": ["Hypersensitivity"]},
            {"drugId": "N/A", "name": "Aluminum hydroxide/Magnesium hydroxide", "class": "Antacid", "rationale": "Symptom relief", "contraindications": ["Severe renal impairment"]}
        ],
        "gastritis": [
            {"drugId": "DB00338", "name": "Omeprazole", "class": "PPI", "rationale": "Acid suppression", "contraindications": ["Hypersensitivity"]}
        ],
        "knee pain": [
            {"drugId": "DB01050", "name": "Topical Diclofenac", "class": "NSAID topical", "rationale": "Localized musculoskeletal pain", "contraindications": ["NSAID allergy", "Open wounds"]},
            {"drugId": "DB01009", "name": "Ibuprofen", "class": "NSAID", "rationale": "Analgesia", "contraindications": ["GI ulcer", "CKD"]}
        ],
        "androgenetic alopecia": [
            {"drugId": "DB00350", "name": "Minoxidil (topical)", "class": "Vasodilator", "rationale": "Promotes hair growth", "contraindications": ["Scalp irritation"]},
            {"drugId": "DB01216", "name": "Finasteride", "class": "5-alpha-reductase inhibitor", "rationale": "Male pattern hair loss", "contraindications": ["Pregnancy", "Women of childbearing potential"]}
        ],
        "alopecia areata": [
            {"drugId": "DB00835", "name": "Topical corticosteroids", "class": "Corticosteroid", "rationale": "First-line immune suppression", "contraindications": ["Skin infection"]}
        ],
        "measles": [
            {"drugId": "DB00316", "name": "Acetaminophen", "class": "Antipyretic", "rationale": "Fever control and comfort", "contraindications": ["Severe liver disease"]},
            {"drugId": "N/A", "name": "Hydration and rest", "class": "Supportive care", "rationale": "Standard supportive management", "contraindications": []}
        ],
    }
    # exact key or simple match by startswith to catch capitalization and variants
    for k, v in table.items():
        if l == k or l.startswith(k):
            return v
    return []


@router.post('/lookup')
async def lookup(req: LookupReq) -> Dict[str, Any]:
    """
    Lookup a disease definition from the Illinois cache by exact/substring
    match. If not found, fall back to a best-effort semantic/overlap match and
    return composition suggestions.
    """
    entry = _find_in_il_cache(req.name)
    if not entry:
        # Fallback: treat input as a free-text symptom sentence and find best Illinois match
        items = _ensure_il_cache()
        if not items:
            return {"found": False, "message": "Not found in Illinois cache"}
        raw = (req.name or "").lower()
        q_tokens = _tokenize(raw)
        if not q_tokens:
            return {"found": False, "message": "Not found in Illinois cache"}
        phrases: List[str] = []
        for i in range(len(q_tokens)-1):
            phrases.append(f"{q_tokens[i]} {q_tokens[i+1]}")
        best = None
        best_score = 0.0
        for it in items:
            blob = f"{it.get('name','')} {it.get('title','')} {it.get('description','')}".lower()
            matched = [t for t in q_tokens if t in blob]
            phrase_hit = any(p in blob for p in phrases)
            base_overlap = (len(matched) / max(1, len(q_tokens)))
            base_score = min(1.0, base_overlap + (0.2 if phrase_hit else 0.0))
            fuzzy_score = 0.0
            if 'HAVE_FUZZ' in globals() and HAVE_FUZZ:
                try:
                    s1 = fuzz.token_set_ratio(raw, blob) / 100.0
                    s2 = fuzz.partial_ratio(raw, blob) / 100.0
                    fuzzy_score = max(s1, s2)
                except Exception:
                    fuzzy_score = 0.0
            score = 0.7 * (fuzzy_score if fuzzy_score > 0 else base_score) + 0.3 * base_score
            if score > best_score:
                best_score = score
                best = it
        # Require a modest threshold to avoid bad guesses
        if best is None or best_score < 0.5:
            return {"found": False, "message": "Not found in Illinois cache"}
        entry = {**best, 'source': 'illinois'}
    title = entry.get('title') or entry.get('name') or req.name
    desc = entry.get('description') or ''
    url = entry.get('url') or ''

    # known meds from our rule table if names match
    known: List[Dict[str, Any]] = []
    # Build a reverse map label->drugs using existing drug_suggestions from differential rules
    try:
        # reuse the locally defined drug_suggestions mapping
        pass
    except Exception:
        pass
    # We cannot access drug_suggestions here easily without refactor; provide composition suggestions
    compositions = _composition_suggestions(title.lower(), desc)

    return {
        "found": True,
        "name": entry.get('name'),
        "title": title,
        "description": desc,
        "url": url,
        "source": entry.get('source', 'illinois'),
        "medications": known,
        "composition_suggestions": compositions
    }
