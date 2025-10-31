from __future__ import annotations
from typing import Dict, List, Tuple

# Optional imports
try:
    from rapidfuzz import process, fuzz  # type: ignore
    HAVE_FUZZ = True
except Exception:
    HAVE_FUZZ = False

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
    HAVE_ST = True
except Exception:
    HAVE_ST = False

_model = None
_embeddings_cache: Dict[str, List[float]] = {}


def get_model() -> object | None:
    global _model
    if _model is not None:
        return _model
    if HAVE_ST:
        # Select model via env var if provided
        import os
        model_name = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
        _model = SentenceTransformer(model_name)
    return _model


def embed(texts: List[str]) -> List[List[float]]:
    model = get_model()
    if not model:
        # Fallback: no embeddings
        return [[0.0] for _ in texts]
    reps = model.encode(texts, normalize_embeddings=True)
    return reps.tolist() if hasattr(reps, 'tolist') else reps


def cached_embedding(text: str) -> List[float]:
    if text in _embeddings_cache:
        return _embeddings_cache[text]
    vec = embed([text])[0]
    _embeddings_cache[text] = vec
    return vec


def semantic_similarity(a: str, b: str) -> float:
    if not HAVE_ST:
        return 0.0
    va = cached_embedding(a)
    vb = cached_embedding(b)
    # cosine similarity for normalized embeddings equals dot product
    if len(va) != len(vb) or len(va) == 0:
        return 0.0
    return float(sum(x*y for x, y in zip(va, vb)))


def fuzzy_similarity(a: str, b: str) -> float:
    if HAVE_FUZZ:
        return float(process.extractOne(a, [b], scorer=fuzz.QRatio)[1]) / 100.0
    return 1.0 if a == b else 0.0


def hybrid_similarity(a: str, b: str, w_fuzzy: float = 0.4, w_sem: float = 0.6) -> float:
    f = fuzzy_similarity(a, b)
    s = semantic_similarity(a, b)
    return w_fuzzy * f + w_sem * s


def best_match(candidate: str, vocabulary: List[str], w_fuzzy: float = 0.3, w_sem: float = 0.7) -> Tuple[str, float]:
    best_term = ""
    best_score = -1.0
    for term in vocabulary:
        score = hybrid_similarity(candidate, term, w_fuzzy=w_fuzzy, w_sem=w_sem)
        if score > best_score:
            best_score = score
            best_term = term
    return best_term, best_score
