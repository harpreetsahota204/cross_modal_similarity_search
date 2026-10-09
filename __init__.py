"""
Cross-modal retrieval: search images, video and audio with text, samples or
files, using EmbeddingGemma 2 vectors in LanceDB similarity indexes.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from .operators import BuildIndex, Search
from .panel import CrossModalRetrievalPanel


def register(p):
    p.register(BuildIndex)
    p.register(Search)
    p.register(CrossModalRetrievalPanel)
