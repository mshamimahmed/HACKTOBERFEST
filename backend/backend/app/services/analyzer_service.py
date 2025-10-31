from __future__ import annotations
import json
from typing import Dict, List, Any, Tuple
from pathlib import Path

from app.utils.text_preprocessing import split_text_to_phrases, normalize_phrase
from app.utils.similarity_utils import best_match, hybrid_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "symptom_disease_map.json"

class SymptomAnalyzer:
    def __init__(self, data_path: Path = DATA_PATH) -> None:
        self.data_path = data_path
        self.data: Dict[str, Any] = {}
        self.vocab: List[str] = []
        self.synonyms: Dict[str, str] = {}
        self.body_systems: Dict[str, str] = {}
        # Illinois DPH cache
        self.illinois_path: Path = Path(__file__).resolve().parent.parent / "data" / "illinois_dph.json"
        self.illinois: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.data_path.exists():
            self.data = {}
            self.vocab = []
            self.synonyms = {}
            self.body_systems = {}
            return
        self.data = json.loads(self.data_path.read_text())
        self.vocab = list(self.data.keys())
        syn: Dict[str, str] = {}
        bodies: Dict[str, str] = {}
        for canonical, entry in self.data.items():
            for syno in entry.get("synonyms", []):
                syn[normalize_phrase(syno)] = canonical
            if entry.get("body_system"):
                bodies[canonical] = entry["body_system"]
        self.synonyms = syn
        self.body_systems = bodies
        # Load Illinois dataset if available
        try:
            if self.illinois_path.exists():
                raw = json.loads(self.illinois_path.read_text())
                # raw can be dict[name] = {...} or list
                if isinstance(raw, dict):
                    self.illinois = [
                        {"name": k, "description": (v.get("description") or ""), "source": v.get("source")}
                        for k, v in raw.items()
                    ]
                elif isinstance(raw, list):
                    self.illinois = raw
        except Exception:
            self.illinois = []

    def analyze_text(self, text: str) -> List[Dict[str, Any]]:
        phrases = [normalize_phrase(p) for p in split_text_to_phrases(text)]
        results: List[Dict[str, Any]] = []
        for phrase in phrases:
            canonical = self.synonyms.get(phrase)
            score = 1.0 if canonical else 0.0
            if not canonical:
                # Find best canonical match from vocabulary
                if self.vocab:
                    # Use weights: 0.3 fuzzy, 0.7 semantic
                    best, best_score = best_match(phrase, self.vocab, w_fuzzy=0.3, w_sem=0.7)
                    canonical = best
                    score = best_score
            entry = self.data.get(canonical or "", {})
            diseases = entry.get("diseases", [])
            body_system = entry.get("body_system") or self.body_systems.get(canonical or "", "unknown")

            # Research fallback threshold at 0.6
            if not canonical or score < 0.6 or not diseases:
                results.append({
                    "symptom": phrase,
                    "normalized": canonical or phrase,
                    "body_system": body_system,
                    "possible_diseases": [],
                    "next_action": "Go to Research",
                    "research_link": f"/research?symptom={phrase}"
                })
                continue

            # Scale disease confidences using symptom match score and priors
            enriched = []
            for d in diseases:
                prior = float(d.get("prior", 0.5))
                conf = max(0.0, min(1.0, 0.5 * score + 0.5 * prior))
                enriched.append({
                    "name": d.get("name", "Unknown"),
                    "confidence": round(conf, 3),
                    "suggestions": d.get("suggestions", [])
                })
            enriched.sort(key=lambda x: x["confidence"], reverse=True)

            # External disease candidates from Illinois DPH dataset (if present)
            external_candidates: List[Dict[str, Any]] = []
            if self.illinois:
                scored_ext: List[Tuple[float, Dict[str, Any]]] = []
                for item in self.illinois:
                    blob = f"{item.get('name','')} {item.get('description','')}".strip()
                    if not blob:
                        continue
                    score_ext = hybrid_similarity(phrase, blob, w_fuzzy=0.3, w_sem=0.7)
                    scored_ext.append((score_ext, item))
                scored_ext.sort(key=lambda t: t[0], reverse=True)
                for s, item in scored_ext[:5]:
                    if s < 0.6:
                        continue
                    external_candidates.append({
                        "name": item.get("name"),
                        "confidence": round(s, 3),
                        "suggestions": [],
                        "source": item.get("source")
                    })

            results.append({
                "symptom": phrase,
                "normalized": canonical,
                "body_system": body_system,
                "possible_diseases": enriched,
                "external_candidates": external_candidates,
                "next_action": "Suggest medication",
                "research_link": f"/research?symptom={phrase}"
            })
        return results

# Singleton analyzer instance
_analyzer: SymptomAnalyzer | None = None

def get_analyzer() -> SymptomAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SymptomAnalyzer()
    return _analyzer
