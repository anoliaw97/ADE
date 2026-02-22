import sys
import os

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Local embedding model (replaces openai text-embedding-ada-002)
# Lazily loaded and cached.
# ---------------------------------------------------------------------------

_st_model = None


def _get_model() -> SentenceTransformer:
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
    return _st_model


def _embed(text: str) -> np.ndarray:
    """Return a 1-D numpy embedding for *text*."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def compute_cosine_similarity(extractedtext, filetext) -> float:
    """Compute cosine similarity between two text strings using local embeddings."""

    if isinstance(filetext, dict):
        filetext = json.dumps(filetext)
    if isinstance(extractedtext, dict):
        extractedtext = json.dumps(extractedtext)

    emb1 = _embed(str(filetext)).reshape(1, -1)
    emb2 = _embed(str(extractedtext)).reshape(1, -1)

    return float(cosine_similarity(emb1, emb2)[0][0])
