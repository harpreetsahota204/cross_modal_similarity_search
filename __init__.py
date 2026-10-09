"""
Cross-modal retrieval: search images, video and audio with text, samples or
files, using EmbeddingGemma 2 vectors in LanceDB similarity indexes.

Module layout:

-   ``engine.py``: embedding, LanceDB indexes and search; no FiftyOne App code
-   ``operators.py``: the ``build_index`` and ``search`` operators
-   ``panel.py``: the Cross-Modal Search panel's Python half; its React half
    is in ``js/src``

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from fiftyone.plugins import PluginContext

from .operators import BuildIndex, Search
from .panel import CrossModalRetrievalPanel


def register(p: PluginContext) -> None:
    """Registers the plugin's operators and panel with FiftyOne.

    Args:
        p: the plugin context FiftyOne passes in when loading the plugin
    """
    p.register(BuildIndex)
    p.register(Search)
    p.register(CrossModalRetrievalPanel)
