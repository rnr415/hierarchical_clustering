"""Vec2GC package

This package was moved to a src-layout for better packaging.
"""

__version__ = "0.1.0"

# Avoid importing heavy optional dependencies at import time so the package can be
# imported even when optional deps (e.g., `networkit`, `sentence-transformers`) are
# not installed. Consumers that need the full functionality should install the
# `full` extra (e.g., `pip install vec2gc[full]`).

try:  # pragma: no cover - behavior depends on optional deps
    from .vec2gc import HierarchicalSequentialClustering, generate_sample_embeddings
except Exception:
    HierarchicalSequentialClustering = None
    generate_sample_embeddings = None

try:  # pragma: no cover - behavior depends on optional deps
    from .embeddings import create_sentence_embeddings
except Exception:
    create_sentence_embeddings = None

__all__ = [
    "HierarchicalSequentialClustering",
    "generate_sample_embeddings",
    "create_sentence_embeddings",
    "__version__",
]
