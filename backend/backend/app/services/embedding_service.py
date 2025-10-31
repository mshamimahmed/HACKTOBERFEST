from __future__ import annotations
import math
from typing import List, Dict, Any, Tuple

# Optional semantic model
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    HAVE_ST = True
except Exception:  # pragma: no cover
    HAVE_ST = False

_model = None


def get_model() -> object | None:
    global _model
    if _model is not None:
        return _model
    if HAVE_ST:
        import os
        name = os.getenv("ST_MODEL", "all-MiniLM-L6-v2")
        try:
            _model = SentenceTransformer(name)
            return _model
        except Exception:
            return None
    return None


def embed_text(text: str) -> List[float]:
    model = get_model()
    if model:
        vec = model.encode([text], normalize_embeddings=True)
        v = vec[0]
        return v.tolist() if hasattr(v, 'tolist') else list(map(float, v))
    # Fallback: simple hashed bag-of-words embedding
    # Deterministic 256-dim vector with token hashing
    import re
    toks = [t for t in re.split(r"[^a-z0-9]+", (text or '').lower()) if t]
    dim = 256
    vec = [0.0] * dim
    for t in toks:
        h = (hash(t) % dim)
        vec[h] += 1.0
    # L2 normalize
    norm = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i]*b[i] for i in range(n))
    na = math.sqrt(sum(a[i]*a[i] for i in range(n))) or 1.0
    nb = math.sqrt(sum(b[i]*b[i] for i in range(n))) or 1.0
    return float(dot / (na * nb))
