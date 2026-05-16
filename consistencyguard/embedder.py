import re
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def normalize_text(text: str) -> str:
    """
    Normalize before embedding to prevent whitespace/casing artifacts
    from causing spurious cache misses.
    - Strip leading/trailing whitespace
    - Collapse internal runs of whitespace to single space
    - Lowercase (MiniLM is case-insensitive, but explicit is safer)
    """
    if not text or not isinstance(text, str):
        raise ValueError(f"embed() requires a non-empty string, got {type(text)!r}")
    return re.sub(r"\s+", " ", text).strip().lower()


def embed(text: str) -> list[float]:
    """Embed a single string. Return as plain list for JSON storage."""
    normalized = normalize_text(text)
    model = get_model()
    vec = model.encode(normalized, normalize_embeddings=True)
    return vec.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two normalized embedding vectors."""
    va = np.array(a)
    vb = np.array(b)
    # Already normalized, so dot product = cosine similarity
    return float(np.dot(va, vb))
