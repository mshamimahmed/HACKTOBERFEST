from __future__ import annotations
import re
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from pathlib import Path

# Lightweight lemmatizer and synonym expansion without heavy deps
# Suffix-based lemmatization for common English forms
_SUFFIX_RULES: List[Tuple[str, str]] = [
    (r"ing$", ""),
    (r"ies$", "y"),
    (r"ed$", ""),
    (r"s$", ""),
]

# Optional NLTK WordNet support
try:
    from nltk.corpus import wordnet as wn  # type: ignore
    HAVE_WN = True
except Exception:
    HAVE_WN = False

# Domain synonyms -> canonical concepts
_SYNONYMS: Dict[str, str] = {
    # fatigue/sleep
    "tired": "fatigue",
    "tiredness": "fatigue",
    "exhausted": "fatigue",
    "exhaustion": "fatigue",
    "sleepy": "somnolence",
    "sleepiness": "somnolence",
    "lack of sleep": "sleep deprivation",
    "no sleep": "sleep deprivation",
    "insomnia": "sleep deprivation",
    # cognition/mood
    "memory issues": "memory impairment",
    "forgetting": "memory impairment",
    "forgetfulness": "memory impairment",
    "can't focus": "inattention",
    "cannot focus": "inattention",
    "unmotivated": "low motivation",
    "no motivation": "low motivation",
    "poor mood": "low mood",
    "sad": "low mood",
    # cardiology
    "palpitations": "irregular heartbeat",
    "skipping beats": "irregular heartbeat",
    "heart skipping": "irregular heartbeat",
    # dizziness / BP
    "dizzy": "dizziness",
    "lightheaded": "dizziness",
    "standing up": "postural change",
    "stand up": "postural change",
    # dermatology
    "itchiness": "pruritus",
    "itchy": "pruritus",
    "hives": "urticaria",
    "skin rash": "dermatitis",
    "rash": "dermatitis",
}

# Medical multi-word phrases to preserve (used by normalize_and_expand)
_PHRASES: List[str] = [
    "shortness of breath",
    "chest tightness",
    "runny nose",
    "nasal congestion",
    "sore throat",
    "muscle aches",
    "muscle pain",
    "loss of smell",
    "frequent urination",
    "high blood sugar",
    "slurred speech",
    "skin rash",
    "contact dermatitis",
    "atopic dermatitis",
    "urticaria",
]

# Pattern rules map tokens to hypotheses
_PATTERNS: List[Dict[str, Any]] = [
    {
        "id": "hypothesis:fatique_sleep",
        "name": "Sleep Deprivation / Fatigue",
        "triggers": ["sleep deprivation", "fatigue", "somnolence", "long hours", "overtime"],
        "reasoning": "Reported lack of sleep or prolonged work can disrupt circadian rhythm leading to fatigue, impaired attention, and mood changes.",
        "support": ["Vitamin B12", "Magnesium", "Melatonin"],
    },
    {
        "id": "hypothesis:postprandial_fatigue",
        "name": "Postprandial Fatigue / Glucose Regulation",
        "triggers": ["after meal", "after meals", "post meal", "post-meal", "postprandial", "heavy meals", "after eating"],
        "reasoning": "Post-meal somnolence may relate to postprandial glucose and insulin dynamics and parasympathetic predominance.",
        "support": ["Chromium", "Berberine", "Fiber"],
    },
    {
        "id": "hypothesis:arrhythmia",
        "name": "Arrhythmia Risk / Palpitations",
        "triggers": ["irregular heartbeat", "palpitation", "skipping beat", "skipping beats", "heart racing"],
        "reasoning": "Irregular heartbeat can reflect ectopy or arrhythmia; contributors include anxiety, stimulants, electrolyte imbalance.",
        "support": ["Electrolytes", "Magnesium", "Omega-3"],
    },
    {
        "id": "hypothesis:orthostatic",
        "name": "Orthostatic Hypotension / Dehydration",
        "triggers": ["dizziness", "lightheaded", "postural change", "standing up"],
        "reasoning": "Dizziness when standing quickly suggests reduced cerebral perfusion due to low blood pressure or dehydration.",
        "support": ["Oral Rehydration Salts", "Electrolytes"],
    },
    {
        "id": "hypothesis:mood_motivation",
        "name": "Low Mood / Stress / Neurotransmitter Changes",
        "triggers": ["low mood", "low motivation", "inattention", "memory impairment", "stress"],
        "reasoning": "Psychological stress can alter monoamine neurotransmitters contributing to low mood, decreased motivation, and cognitive issues.",
        "support": ["Omega-3", "B-Complex", "Rhodiola"],
    },
    {
        "id": "hypothesis:pruritus",
        "name": "Dermatologic Irritation / Pruritus",
        "triggers": ["pruritus", "itchiness", "itchy", "urticaria", "hives", "dermatitis", "skin rash"],
        "reasoning": "Itching and rash suggest cutaneous irritation, allergy, or urticaria; histamine pathways often implicated.",
        "support": ["Antihistamines", "Calamine", "Hydrocortisone (topical)"],
    },
]

_STOP = {
    "the","and","or","with","of","to","in","for","on","a","an","is","are","was","were","it","its","at","by",
    "what","this","that","these","those","from","as","than","then","be","been","being","can","may","might","will","would","should","could",
    "i","am","not","feeling","due","because","have","has","had","very","severe","mild","moderate","like"
}

LOG_DIR = Path(__file__).resolve().parent.parent / "data"
LOG_PATH = LOG_DIR / "researcher_logs.jsonl"


def lemmatize_token(tok: str) -> str:
    t = tok
    for pat, repl in _SUFFIX_RULES:
        t = re.sub(pat, repl, t)
    return t


def normalize_and_expand(text: str) -> Tuple[str, List[str]]:
    t = (text or "").lower()
    # preserve known phrases by underscoring
    for p in _PHRASES:
        t = t.replace(p, p.replace(" ", "_"))
    t = re.sub(r"[^a-z0-9_\s]", " ", t)
    raw = [w for w in t.split() if w and w not in _STOP]
    joined = " ".join(raw)
    # expand domain synonyms at phrase-level
    for k, v in _SYNONYMS.items():
        if k.replace(" ", "_") in joined:
            joined = joined.replace(k.replace(" ", "_"), v.replace(" ", "_"))
        elif k in joined:
            joined = joined.replace(k, v)
    # tokens with basic lemmatization
    toks = [lemmatize_token(w) for w in joined.split()]
    # optional WordNet lemma and synonym expansion
    if HAVE_WN:
        expanded: List[str] = []
        for w in toks:
            lemma = w
            # try wordnet lemma via first synset lemma
            try:
                syns = wn.synsets(w)
                if syns:
                    lemma = syns[0].lemmas()[0].name().lower().replace(' ', '_')
            except Exception:
                pass
            expanded.append(lemma)
            # add a few synonyms (up to 2 to limit noise)
            try:
                syns = wn.synsets(w)
                seen = set()
                for s in syns[:2]:
                    for l in s.lemmas()[:2]:
                        name = l.name().lower().replace(' ', '_')
                        if name != w and name not in seen:
                            seen.add(name)
                            expanded.append(name)
            except Exception:
                pass
        toks = expanded
    # revert underscores to spaces in the normalized text, but keep tokens underscored to indicate phrases
    norm_text = " ".join(toks).replace("_", " ")
    return norm_text, toks


def infer_hypotheses(norm_text: str, toks: List[str]) -> List[Dict[str, Any]]:
    joined = " ".join(toks)
    hyps: List[Dict[str, Any]] = []
    for pat in _PATTERNS:
        hit = False
        for trig in pat["triggers"]:
            if trig in norm_text or trig in joined:
                hit = True
                break
        if hit:
            hyps.append({
                "hypothesis_id": pat["id"],
                "title": pat["name"],
                "reasoning": pat["reasoning"],
                "support": pat["support"],
                "score": 0.75,
            })
    return hyps


def log_query(payload: Dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        payload = {**payload, "ts": datetime.utcnow().isoformat() + "Z"}
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # best-effort logging
        pass
