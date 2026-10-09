"""
Cross-modal retrieval with EmbeddingGemma 2 and LanceDB.

Every index is a native ``fiftyone.brain`` similarity index with the
``lancedb`` backend. Text, images, video and audio share one embedding space,
so a vector from any of them can query an index of any other.

The module has four parts:

-   **Model**: loading the shared EmbeddingGemma 2 model and embedding text
    or media files with it
-   **Indexes**: finding, describing and repairing the dataset's
    EmbeddingGemma 2 LanceDB indexes
-   **Queries**: :class:`TextQuery`, :class:`VectorQuery` and
    :class:`MediaQuery`, which turn what the user searched with into a
    vector for each index
-   **Search** and **Building**: running queries against LanceDB, and
    embedding samples into new indexes

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from __future__ import annotations

import contextlib
import logging
import os
import re
import threading
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, Optional, Protocol, TypedDict

import numpy as np

import fiftyone.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fomm
import fiftyone.zoo as foz

logger = logging.getLogger(__name__)

MODEL_NAME = "google/embeddinggemma-2"
ZOO_SOURCE = "https://github.com/harpreetsahota204/EmbeddingGemma2"

# The lancedb backend's own default, used when the brain config sets no URI
DEFAULT_LANCEDB_URI = "/tmp/lancedb"

# What an index's vectors can be made from. "video+audio" is one vector from
# frames and soundtrack together, not two vectors
MODALITIES = ("image", "video", "audio", "video+audio")

# Matryoshka sizes: the first N numbers of a 768-d vector are a usable
# embedding on their own
EMBEDDING_DIMS = (768, 512, 256, 128)
MAX_DIM = 768

# The model's prompt for search queries; it's prepended to query text
DEFAULT_PROMPT = "SearchQuery"

# Upper bound on results per index, so one request can't load the whole table
MAX_K = 100

# Recognized by extension first, because FiftyOne's mime-type lookup doesn't
# classify some of these (e.g. .weba, .opus) as media
AUDIO_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg",
    ".opus", ".wav", ".weba", ".wma",
}  # fmt: skip

# Browser recordings (MediaRecorder) are written as a stream, without the
# duration that frame sampling needs, so they are remuxed into these
# containers, which the model's decoder also accepts
RECORDING_EXTENSIONS = {".webm": ".webm", ".weba": ".ogg"}

# Output extension -> PyAV container format name
_CONTAINER_FORMATS = {".webm": "webm", ".ogg": "ogg"}

# What each kind of media can be embedded as
QUERY_MODALITIES = {
    "image": ("image",),
    "video": ("video", "audio", "video+audio"),
    "audio": ("audio",),
    "mcap": ("video", "audio", "video+audio"),
}

# FiftyOne brain metric name -> LanceDB distance type
_LANCE_METRICS = {"cosine": "cosine", "euclidean": "l2"}

# One model with every encoder serves all modalities. It is shared by the
# App server's request threads, and its settings are mutable, so all use of
# it happens under the lock
_model: Any = None
_model_lock = threading.RLock()


###############################################################################
# Types
###############################################################################


class IndexInfo(TypedDict):
    """What the plugin knows about one index, from its brain run config.

    Built by :func:`list_indexes`.
    """

    key: str  # brain key
    modality: str  # one of MODALITIES
    field: str  # sample field that stores the vectors
    dim: int  # vector size: one of EMBEDDING_DIMS
    prompt_name: Optional[str]  # prompt used to embed text queries
    settings: dict[str, Any]  # fps / max_frames / max_audio_seconds, if set


class Hit(TypedDict, total=False):
    """One search result, as sent to the panel. Built by :func:`describe_hits`."""

    id: str
    score: float  # cosine similarity, -1 to 1, higher is closer
    filepath: str
    media: Optional[str]  # "image", "video", "audio", "mcap" or None
    label: Optional[str]  # caption from label_field, if one was given
    group_id: str  # grouped datasets only
    slice: str  # grouped datasets only


class Section(TypedDict):
    """The results of one index in a search. Built by :func:`search`."""

    key: str
    modality: str
    hits: list[Hit]
    error: Optional[str]  # None unless searching this index failed


class Query(Protocol):
    """What :func:`search` needs from a query. Implemented by
    :class:`TextQuery`, :class:`VectorQuery` and :class:`MediaQuery`.
    """

    def describe(self) -> str:
        """A short description of the query, for labels and messages."""

    def vector_for(self, index: IndexInfo) -> np.ndarray:
        """The query vector to search the given index with."""

    def excludes(self, index: IndexInfo) -> Iterable[str]:
        """Sample IDs to leave out of the given index's results."""


def lancedb_uri() -> str:
    """The LanceDB URI the brain uses: the ``lancedb`` backend's ``uri`` in
    the brain config (``FIFTYONE_BRAIN_SIMILARITY_LANCEDB_URI``), else the
    backend's default.

    The URI is not saved with the index, so the App's own similarity search
    reloads indexes from here too.

    Returns:
        a local path or cloud storage URI
    """
    params = fob.brain_config.similarity_backends.get("lancedb", {})
    return params.get("uri") or DEFAULT_LANCEDB_URI


###############################################################################
# Model
###############################################################################


def get_model() -> Any:
    """Returns the shared EmbeddingGemma 2 model, loading it on first use.

    The model is loaded with every encoder (``media_type="video"`` with
    vision and audio), so one instance can embed any modality.

    Returns:
        the EmbeddingGemma 2 zoo model

    Raises:
        ValueError: if the model's zoo source hasn't been registered
    """
    global _model
    with _model_lock:
        if _model is None:
            if MODEL_NAME not in foz.list_zoo_models():
                raise ValueError(
                    "EmbeddingGemma 2 is not in your model zoo. Register it "
                    "with fiftyone.zoo.register_zoo_model_source('%s')"
                    % ZOO_SOURCE
                )

            _model = foz.load_zoo_model(
                MODEL_NAME, media_type="video", modalities=["vision", "audio"]
            )

        return _model


def is_model_loaded() -> bool:
    """Whether the model is already in memory, so the panel can warn that
    the first search will be slow.
    """
    return _model is not None


def has_gpu() -> bool:
    """Whether torch can use a CUDA or Apple Silicon GPU."""
    import torch  # deferred: torch is slow to import

    return torch.cuda.is_available() or torch.backends.mps.is_available()


def slugify(name: str) -> str:
    """Makes a string usable in field names and brain keys.

    Args:
        name: any string

    Returns:
        the string lowercased, with runs of non-word characters replaced by
        ``_``
    """
    return re.sub(r"\W+", "_", name).strip("_").lower()


@contextlib.contextmanager
def _model_settings(model: Any, settings: dict[str, Any]) -> Iterator[None]:
    """Temporarily sets model properties such as ``fps``, restoring the
    previous values on exit.

    The caller must hold ``_model_lock``, since other threads share the model.
    """
    previous = {k: getattr(model, k) for k in settings}
    for k, v in settings.items():
        setattr(model, k, v)
    try:
        yield
    finally:
        for k, v in previous.items():
            setattr(model, k, v)


def fit_dim(vector: Sequence[float] | np.ndarray, dim: int) -> np.ndarray:
    """Truncates a vector to a Matryoshka dimension and re-normalizes it.

    Args:
        vector: a vector of at least ``dim`` values
        dim: the size to keep, e.g. 256

    Returns:
        a ``(dim,)`` float32 unit vector
    """
    vector = np.asarray(vector, dtype=np.float32)[:dim]
    # cosine similarity assumes unit vectors; truncation shortens them
    return vector / np.linalg.norm(vector)


def embed_text(text: str, prompt_name: Optional[str] = DEFAULT_PROMPT) -> np.ndarray:
    """Embeds a text query.

    Args:
        text: the query text
        prompt_name (DEFAULT_PROMPT): the model prompt to prefix it with, or
            None for no prompt

    Returns:
        a ``(768,)`` unit vector
    """
    model = get_model()
    with _model_lock:
        return model.embed_text([text], prompt_name=prompt_name)[0]


def embed_file(
    path: str, modality: str, settings: Optional[dict[str, Any]] = None
) -> np.ndarray:
    """Embeds a media file.

    Args:
        path: an image, video, audio or ``.mcap`` filepath
        modality: one of :data:`MODALITIES`
        settings (None): model settings to apply while embedding, e.g.
            ``{"fps": 2}``

    Returns:
        a ``(768,)`` unit vector
    """
    model = get_model()
    with _model_lock, _model_settings(model, settings or {}):
        if modality == "image":
            return model.embed_images([path])[0]

        if modality == "video":
            # frames only: sampled per the model's fps / max_frames
            return model.embed_frames([model.load_frames(path)])[0]

        if modality == "audio":
            # the soundtrack of a video, or an audio file
            return model.embed_audio_file(path)

        # "video+audio": frames and soundtrack as one input, one vector
        return model.embed_video_audio(path)


def media_kind(filepath: str) -> Optional[str]:
    """Classifies a file by what the model can read from it.

    Args:
        filepath: a filepath or filename

    Returns:
        ``"image"``, ``"video"``, ``"audio"``, ``"mcap"``, or None if the
        file isn't media the model supports
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext in AUDIO_EXTENSIONS:
        return "audio"

    media_type = fomm.get_media_type(filepath)
    if media_type in ("image", "video"):
        return media_type

    if media_type == "multimodal" and ext == ".mcap":
        return "mcap"

    return None


###############################################################################
# Indexes
###############################################################################


# Model settings an index can be built with, and the modalities they change
SETTINGS_MODALITIES = {
    "fps": ("video", "video+audio"),
    "max_frames": ("video", "video+audio"),
    "max_audio_seconds": ("audio", "video+audio"),
}


def query_settings(index: IndexInfo | dict, modality: str) -> dict[str, Any]:
    """The settings of an index that apply to embedding a query as the given
    modality, so the query is decoded the way the index's samples were.

    Args:
        index: an :class:`IndexInfo`, or ``{}`` for the model's defaults
        modality: the modality the query is embedded as

    Returns:
        a dict of model settings, e.g. ``{"fps": 2.0}``; empty when the
        index used the defaults
    """
    return {
        k: v
        for k, v in (index.get("settings") or {}).items()
        if modality in SETTINGS_MODALITIES[k]
    }


def index_modality(model_kwargs: dict[str, Any]) -> str:
    """The modality an index's vectors came from, per its ``model_kwargs``.

    Args:
        model_kwargs: the ``model_kwargs`` of the index's brain config

    Returns:
        one of :data:`MODALITIES`
    """
    media_type = model_kwargs.get("media_type", "image")
    if media_type == "video" and "audio" in (model_kwargs.get("modalities") or ()):
        return "video+audio"

    return media_type


def build_model_kwargs(
    modality: str, dim: int = MAX_DIM, settings: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """The ``model_kwargs`` stored on an index, so the App can reload the
    model for text search and :func:`index_modality` can tell the modality.

    Args:
        modality: one of :data:`MODALITIES`
        dim (MAX_DIM): the index's vector size
        settings (None): the fps / max_frames / max_audio_seconds the samples
            were embedded with

    Returns:
        a dict of ``load_zoo_model()`` keyword arguments
    """
    kwargs: dict[str, Any] = {
        "media_type": "video" if modality == "video+audio" else modality
    }
    if modality == "video+audio":
        kwargs["modalities"] = ["vision", "audio"]

    if dim != MAX_DIM:
        kwargs["embedding_dim"] = dim

    kwargs.update(settings or {})
    return kwargs


def flat(dataset: Any) -> Any:
    """A view with every sample of the dataset, across all group slices.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`

    Returns:
        the dataset, or for grouped datasets a view of all its slices
    """
    if dataset.media_type == "group":
        return dataset.select_group_slices(_allow_mixed=True)

    return dataset


def list_indexes(dataset: Any) -> list[IndexInfo]:
    """Finds the EmbeddingGemma 2 LanceDB indexes on a dataset.

    Indexes built in Python count too, if they were created with
    ``model="google/embeddinggemma-2"`` and ``backend="lancedb"``.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`

    Returns:
        an :class:`IndexInfo` per index
    """
    indexes: list[IndexInfo] = []
    for key in dataset.list_brain_runs(type=fob.Similarity, method="lancedb"):
        config = dataset.get_brain_info(key).config

        # other models' indexes, and patch (object-level) indexes, aren't ours
        if config.model != MODEL_NAME or config.patches_field is not None:
            continue

        kwargs = config.model_kwargs or {}
        indexes.append(
            {
                "key": key,
                "modality": index_modality(kwargs),
                "field": config.embeddings_field,
                "dim": kwargs.get("embedding_dim", MAX_DIM),
                "prompt_name": kwargs.get("prompt_name", DEFAULT_PROMPT),
                "settings": {
                    k: kwargs[k] for k in SETTINGS_MODALITIES if k in kwargs
                },
            }
        )

    return indexes


def load_index(dataset: Any, brain_key: str) -> Any:
    """Loads a LanceDB similarity index from the brain's LanceDB URI.

    Not cached: a cached index keeps its table handle after the table's
    files are deleted, and would neither notice nor see a rebuild.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        brain_key: the index's brain key

    Returns:
        a :class:`fiftyone.brain.internal.core.lancedb.LanceDBSimilarityIndex`
    """
    return dataset.load_brain_results(
        brain_key, uri=lancedb_uri(), cache=False
    )


def describe_index(dataset: Any, index: IndexInfo) -> dict[str, Any]:
    """Adds what the panel shows about an index: size, table health and
    group slices.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        index: an :class:`IndexInfo`

    Returns:
        the :class:`IndexInfo` plus ``table`` (LanceDB table name), ``count``
        (vectors in the table), ``ok``, ``error`` and ``slices`` (group slices
        the index covers, or None)
    """
    info: dict[str, Any] = dict(index)
    try:
        results = load_index(dataset, index["key"])
        info["table"] = results.config.table_name
        info["count"] = results.total_index_size
        # the brain run can outlive its table, e.g. when /tmp is cleared
        info["ok"] = results.table is not None
        info["error"] = None if info["ok"] else "LanceDB table not found"
    except Exception as e:
        logger.warning("Failed to load index '%s': %s", index["key"], e)
        info.update(table=None, count=0, ok=False, error=str(e))

    info["slices"] = None
    if dataset.media_type == "group":
        info["slices"] = dataset.load_brain_view(index["key"]).distinct(
            "%s.name" % dataset.group_field
        )

    return info


def rebuild_table(dataset: Any, brain_key: str) -> int:
    """Recreates an index's LanceDB table from its embeddings field, e.g.
    after the URI's directory was cleared. Nothing is re-embedded.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        brain_key: the index's brain key

    Returns:
        the number of vectors written

    Raises:
        ValueError: if the key isn't one of our indexes, or its field is empty
    """
    field = next(
        (i["field"] for i in list_indexes(dataset) if i["key"] == brain_key),
        None,
    )
    if field is None:
        raise ValueError(
            "'%s' is not an EmbeddingGemma 2 LanceDB index on this dataset"
            % brain_key
        )

    results = load_index(dataset, brain_key)
    if results.table is not None:
        # drop the partial table so the rebuild doesn't duplicate rows
        results.cleanup()

    view = dataset.load_brain_view(brain_key).exists(field)
    ids, vectors = view.values(["id", field])
    if not ids:
        raise ValueError("Field '%s' has no embeddings" % field)

    results.add_to_index(np.stack(vectors), np.array(ids))
    return len(ids)


###############################################################################
# Queries
###############################################################################


class TextQuery(object):
    """A text query, embedded with each target index's prompt.

    Embeds lazily, once per distinct prompt among the indexes searched.

    Args:
        text: the query text
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self._vectors: dict[Optional[str], np.ndarray] = {}  # prompt -> vector

    def describe(self) -> str:
        """Returns ``'text "<the query>"'``."""
        return 'text "%s"' % self.text

    def vector_for(self, index: IndexInfo) -> np.ndarray:
        """Returns the text's vector for the given index.

        Args:
            index: an :class:`IndexInfo`

        Returns:
            a unit vector of the index's size
        """
        prompt_name = index["prompt_name"]
        if prompt_name not in self._vectors:
            self._vectors[prompt_name] = embed_text(self.text, prompt_name)

        return fit_dim(self._vectors[prompt_name], index["dim"])

    def excludes(self, index: IndexInfo) -> Iterable[str]:
        """Text matches no sample, so nothing is excluded."""
        return ()


class VectorQuery(object):
    """A query by media: the mean of one or more unit vectors.

    Args:
        vector: the query vector
        label: what the vector represents, for messages
        source_field (None): the embeddings field the vector was read from.
            Results from indexes on that field leave out ``exclude_ids``,
            which would only find themselves
        exclude_ids (()): the IDs of the query samples
    """

    def __init__(
        self,
        vector: Sequence[float] | np.ndarray,
        label: str,
        source_field: Optional[str] = None,
        exclude_ids: Iterable[str] = (),
    ) -> None:
        self.vector = np.asarray(vector, dtype=np.float32)
        self.label = label
        self.source_field = source_field
        self.exclude_ids = set(exclude_ids)

    def describe(self) -> str:
        """Returns the query's label."""
        return self.label

    def vector_for(self, index: IndexInfo) -> np.ndarray:
        """Returns the vector, cut to the index's size.

        Args:
            index: an :class:`IndexInfo`

        Returns:
            a unit vector of the index's size

        Raises:
            ValueError: if the vector is smaller than the index's vectors,
                e.g. read from a 256-d index to search a 768-d one
        """
        if index["dim"] > len(self.vector):
            raise ValueError(
                "The query is %dd but this index is %dd"
                % (len(self.vector), index["dim"])
            )

        return fit_dim(self.vector, index["dim"])

    def excludes(self, index: IndexInfo) -> Iterable[str]:
        """Leaves the query samples out of indexes built on the field the
        query vector came from, where they'd trivially rank first.
        """
        if self.source_field and index["field"] == self.source_field:
            return self.exclude_ids

        return ()


class MediaQuery(object):
    """A query by media files, embedded once per distinct set of model
    settings among the target indexes, since an index built with e.g.
    ``fps=2`` must be queried with frames sampled at 2 fps.

    Embeds eagerly, so the files only need to exist while this is built.

    Args:
        paths: the media filepaths, averaged into one query
        modality: the modality to embed them as
        label: what the query represents, for messages
        indexes: the indexes the query will search
    """

    def __init__(
        self,
        paths: Sequence[str],
        modality: str,
        label: str,
        indexes: Sequence[IndexInfo],
    ) -> None:
        self.label = label
        self._modality = modality
        # settings key (see _settings_key) -> averaged query vector
        self._vectors: dict[tuple, np.ndarray] = {}

        # with no indexes, embed once with the model's defaults
        for index in indexes or [{}]:
            settings = query_settings(index, modality)
            key = _settings_key(settings)
            if key not in self._vectors:
                self._vectors[key] = _mean(
                    [embed_file(p, modality, settings=settings) for p in paths]
                )

    def describe(self) -> str:
        """Returns the query's label."""
        return self.label

    def vector_for(self, index: IndexInfo) -> np.ndarray:
        """Returns the vector embedded with the index's settings.

        Args:
            index: an :class:`IndexInfo`

        Returns:
            a unit vector of the index's size

        Raises:
            ValueError: if the index wasn't among those given at construction
                and its settings differ from all of them
        """
        key = _settings_key(query_settings(index, self._modality))
        if key not in self._vectors:
            raise ValueError(
                "The query wasn't embedded for index '%s'" % index["key"]
            )

        return fit_dim(self._vectors[key], index["dim"])

    def excludes(self, index: IndexInfo) -> Iterable[str]:
        """Files from outside the dataset match no sample, so nothing is
        excluded.
        """
        return ()


def _settings_key(settings: dict[str, Any]) -> tuple:
    """A hashable, order-independent key for a settings dict."""
    return tuple(sorted(settings.items()))


def _mean(vectors: Sequence[Sequence[float] | np.ndarray]) -> np.ndarray:
    """Averages unit vectors and re-normalizes the result.

    Args:
        vectors: one or more vectors of the same size

    Returns:
        a unit vector of that size
    """
    return fit_dim(np.mean(np.stack(vectors), axis=0), len(vectors[0]))


def describe_selection(
    dataset: Any, sample_ids: Sequence[str], indexes: Sequence[IndexInfo]
) -> Optional[dict[str, Any]]:
    """Describes the given samples as a query: their media kind, what they
    can be embedded as, and which of those are already stored in an index.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        sample_ids: the selected sample IDs
        indexes: the dataset's :class:`IndexInfo` list

    Returns:
        None if nothing is selected, else a dict with ``ids``, ``count``,
        ``kind`` (media kind, or None if mixed or unsupported),
        ``modalities`` (what the selection can be embedded as), ``stored``
        (modality -> brain key of an index that already has every selected
        sample's vector) and ``error`` (why the selection can't be used, or
        None)
    """
    if not sample_ids:
        return None

    samples = flat(dataset).select(sample_ids)
    kinds = {media_kind(p) for p in samples.values("filepath")}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    modalities = list(QUERY_MODALITIES.get(kind, ()))

    stored: dict[str, str] = {}
    for index in indexes:
        m = index["modality"]
        if (
            m in modalities
            and m not in stored
            and samples.exists(index["field"]).count() == len(sample_ids)
        ):
            stored[m] = index["key"]

    error = None
    if len(kinds) > 1:
        error = "The selection mixes media types; select one kind"
    elif kind is None:
        error = "This sample's media can't be embedded"

    return {
        "ids": list(sample_ids),
        "count": len(sample_ids),
        "kind": kind,
        "modalities": modalities,
        "stored": stored,
        "error": error,
    }


def sample_query(
    dataset: Any,
    sample_ids: Sequence[str],
    modality: str,
    indexes: Sequence[IndexInfo],
) -> VectorQuery | MediaQuery:
    """Builds a query from samples of the dataset, as the given modality.

    Uses the vectors stored in an index of that modality when every sample
    has one, otherwise embeds the samples' media.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        sample_ids: the query sample IDs; several are averaged into one query
        modality: the modality to search with
        indexes: the indexes the query will search

    Returns:
        a :class:`VectorQuery` (stored vectors) or :class:`MediaQuery`
        (freshly embedded)
    """
    samples = flat(dataset).select(sample_ids)
    if len(sample_ids) > 1:
        noun = "%d selected samples" % len(sample_ids)
    else:
        noun = os.path.basename(samples.first().filepath)

    label = "%s of %s" % (modality, noun)

    for index in indexes:
        if index["modality"] != modality:
            continue

        # reuse stored vectors only if every sample has one, so the average
        # isn't silently missing samples
        field = index["field"]
        if samples.exists(field).count() == len(sample_ids):
            return VectorQuery(
                _mean(samples.values(field)),
                label,
                source_field=field,
                exclude_ids=sample_ids,
            )

    return MediaQuery(samples.values("filepath"), modality, label, indexes)


def remux_recording(src: str, dst: str) -> None:
    """Copies the audio and video packets of a browser recording into a new
    WebM or Ogg file (per ``dst``'s extension), which records the duration
    and seek index the recording lacks. Nothing is re-encoded.

    Args:
        src: the recording, as uploaded
        dst: the output path, ending in ``.webm`` or ``.ogg``

    Raises:
        ValueError: if the recording has no audio or video stream
    """
    import av  # deferred: only needed for browser recordings

    fmt = _CONTAINER_FORMATS[os.path.splitext(dst)[1].lower()]
    with av.open(src) as inp, av.open(dst, "w", format=fmt) as out:
        # input stream index -> matching output stream
        streams = {
            s.index: out.add_stream_from_template(s)
            for s in inp.streams
            if s.type in ("video", "audio")
        }
        if not streams:
            raise ValueError("The recording has no audio or video")

        for packet in inp.demux(*[inp.streams[i] for i in streams]):
            # demux() ends each stream with an empty flush packet
            if packet.dts is None:
                continue

            packet.stream = streams[packet.stream.index]
            out.mux(packet)


def file_query(
    path: str,
    modality: str,
    indexes: Sequence[IndexInfo],
    name: Optional[str] = None,
) -> MediaQuery:
    """Builds a query from a media file that is not in the dataset, embedded
    with the settings of each of the given indexes.

    Args:
        path: the media filepath
        modality: the modality to embed it as
        indexes: the indexes the query will search
        name (None): the name to show for the file; defaults to its basename

    Returns:
        a :class:`MediaQuery`
    """
    return MediaQuery(
        [path],
        modality,
        "%s of %s" % (modality, name or os.path.basename(path)),
        indexes,
    )


###############################################################################
# Search
###############################################################################


def _score(distance: float, metric: str) -> float:
    """Converts a LanceDB distance into cosine similarity (higher is closer).

    Args:
        distance: LanceDB's ``_distance`` for a result
        metric: the index's brain metric, ``"cosine"`` or ``"euclidean"``

    Returns:
        the cosine similarity, -1 to 1
    """
    # cosine distance is 1 - cos; LanceDB's l2 is squared, so for unit
    # vectors it's 2 - 2 cos
    if metric == "euclidean":
        return 1.0 - distance / 2.0

    return 1.0 - distance


def search_index(
    dataset: Any, index: IndexInfo, query: Query, k: int
) -> list[tuple[str, float]]:
    """Searches one index.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        index: the :class:`IndexInfo` to search
        query: a :class:`Query`
        k: the number of results

    Returns:
        up to ``k`` ``(sample_id, score)`` tuples, best first

    Raises:
        ValueError: if the index's LanceDB table is missing
    """
    results = load_index(dataset, index["key"])
    if results.table is None:
        raise ValueError("LanceDB table not found; rebuild it")

    metric = results.config.metric
    exclude = query.excludes(index)
    rows = (
        results.table.search(query.vector_for(index))
        .distance_type(_LANCE_METRICS[metric])
        # over-fetch so k remain after the excluded samples are dropped
        .limit(k + len(exclude))
        .select(["id"])
        .to_list()
    )

    hits = [
        (r["id"], _score(r["_distance"], metric))
        for r in rows
        if r["id"] not in exclude
    ]
    return hits[:k]


# label type -> path to its label strings, relative to the field
_LABEL_PATHS = {
    fol.Classification: ".label",
    fol.Classifications: ".classifications.label",
    fol.Detections: ".detections.label",
}


def label_fields(dataset: Any) -> list[str]:
    """Top-level fields whose values make a good caption for a result.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`

    Returns:
        the names of label fields of a supported type, and string fields
    """
    names = []
    schema = flat(dataset).get_field_schema()
    for name, field in schema.items():
        if isinstance(field, fof.EmbeddedDocumentField):
            if field.document_type in _LABEL_PATHS:
                names.append(name)
        elif isinstance(field, fof.StringField) and name != "filepath":
            names.append(name)

    return names


def _label_path(dataset: Any, label_field: str) -> str:
    """The path to read caption strings from, e.g. ``"gt.label"`` for a
    :class:`fiftyone.core.labels.Classification` field ``"gt"``.
    """
    field = flat(dataset).get_field(label_field)
    if isinstance(field, fof.EmbeddedDocumentField):
        return label_field + _LABEL_PATHS[field.document_type]

    return label_field


def _caption(value: Any) -> Optional[str]:
    """Turns a label value into a caption string.

    Args:
        value: a string, a list of strings (e.g. detection labels), or None

    Returns:
        the caption, with list values de-duplicated and comma-joined, or None
    """
    if isinstance(value, list):
        # dict.fromkeys de-duplicates while keeping the order
        unique = dict.fromkeys(str(v) for v in value if v is not None)
        return ", ".join(unique) or None

    return None if value is None else str(value)


def describe_hits(
    dataset: Any,
    index: IndexInfo,
    hits: Sequence[tuple[str, float]],
    label_field: Optional[str] = None,
) -> list[Hit]:
    """Adds media and captions to search hits.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        index: the :class:`IndexInfo` the hits came from
        hits: ``(sample_id, score)`` tuples, best first
        label_field (None): a field to caption the hits with

    Returns:
        a :class:`Hit` per hit still in the dataset, in the same order
    """
    if not hits:
        return []

    ids = [h[0] for h in hits]
    # the brain view covers every group slice the index holds
    samples = dataset.load_brain_view(index["key"]).select(ids)

    paths = ["id", "filepath"]
    if label_field:
        paths.append(_label_path(dataset, label_field))
    if dataset.media_type == "group":
        paths += ["%s.id" % dataset.group_field, "%s.name" % dataset.group_field]

    # sample ID -> (id, filepath, [label], [group id, slice name])
    rows = {v[0]: v for v in zip(*samples.values(paths), strict=True)}

    out: list[Hit] = []
    for sample_id, score in hits:
        row = rows.get(sample_id)
        if row is None:
            # in the LanceDB table but deleted from the dataset
            continue

        hit: Hit = {
            "id": sample_id,
            "score": round(float(score), 4),
            "filepath": row[1],
            "media": media_kind(row[1]),
            "label": _caption(row[2]) if label_field else None,
        }
        if dataset.media_type == "group":
            hit["group_id"], hit["slice"] = row[-2], row[-1]

        out.append(hit)

    return out


def search(
    dataset: Any,
    query: Query,
    brain_keys: Optional[Sequence[str]] = None,
    k: int = 12,
    label_field: Optional[str] = None,
) -> list[Section]:
    """Runs a query against EmbeddingGemma 2 LanceDB indexes.

    One failing index doesn't fail the search: its section carries the
    error, and the other indexes still return results.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        query: a :class:`TextQuery`, :class:`VectorQuery` or
            :class:`MediaQuery`
        brain_keys (None): the indexes to search. By default, all of them
        k (12): results per index, at most :data:`MAX_K`
        label_field (None): a field to caption the results with

    Returns:
        a :class:`Section` per index searched
    """
    k = max(1, min(int(k), MAX_K))
    indexes = list_indexes(dataset)
    if brain_keys is not None:
        indexes = [i for i in indexes if i["key"] in brain_keys]

    sections: list[Section] = []
    for index in indexes:
        section: Section = {
            "key": index["key"],
            "modality": index["modality"],
            "hits": [],
            "error": None,
        }
        try:
            hits = search_index(dataset, index, query, k)
            section["hits"] = describe_hits(dataset, index, hits, label_field)
        except Exception as e:
            logger.warning("Search of '%s' failed: %s", index["key"], e)
            section["error"] = str(e)

        sections.append(section)

    return sections


###############################################################################
# Building
###############################################################################


def default_modalities(view: Any) -> list[str]:
    """The modalities the samples in a view can be embedded as.

    Args:
        view: a :class:`fiftyone.core.collections.SampleCollection`

    Returns:
        a list of :data:`MODALITIES` values, in a stable order
    """
    if view.media_type == "image":
        return ["image"]

    if view.media_type == "video":
        return ["video", "audio", "video+audio"]

    # mixed or grouped media: look at a sample of filepaths
    kinds = {media_kind(p) for p in view.take(20, seed=51).values("filepath")}
    modalities: list[str] = []
    for kind in ("image", "video", "audio", "mcap"):
        if kind in kinds:
            for m in QUERY_MODALITIES[kind]:
                if m not in modalities:
                    modalities.append(m)

    return modalities


def embed_samples(
    view: Any,
    field: str,
    modality: str,
    dim: int = MAX_DIM,
    settings: Optional[dict[str, Any]] = None,
    overwrite: bool = False,
) -> Iterator[tuple[int, int, int]]:
    """Embeds the media of each sample into a vector field, in batches.

    Samples whose media fails to decode (e.g. a video with no audio track
    embedded as audio) are logged and left empty.

    Args:
        view: a :class:`fiftyone.core.collections.SampleCollection`
        field: the vector field to write, created if needed
        modality: one of :data:`MODALITIES`
        dim (MAX_DIM): the vector size to store
        settings (None): fps / max_frames / max_audio_seconds to embed with
        overwrite (False): whether to re-embed samples that already have a
            vector in ``field``

    Yields:
        ``(num_done, num_total, num_failed)`` after each batch
    """
    dataset = view._dataset
    if not dataset.has_sample_field(field):
        dataset.add_sample_field(field, fof.VectorField)

    todo = view if overwrite else view.exists(field, False)
    ids, paths = todo.values(["id", "filepath"])
    total = len(ids)

    # batches only set how often progress is reported and values are saved;
    # each file is embedded on its own. Video is slow, so report more often
    batch_size = 16 if modality == "image" else 4

    failed = 0
    for start in range(0, total, batch_size):
        batch = zip(
            ids[start : start + batch_size],
            paths[start : start + batch_size],
            strict=True,
        )
        values: dict[str, np.ndarray] = {}
        for sample_id, path in batch:
            try:
                vector = embed_file(path, modality, settings=settings)
                values[sample_id] = fit_dim(vector, dim)
            except Exception as e:
                logger.warning("Failed to embed %s: %s", path, e)
                failed += 1

        if values:
            view.set_values(field, values, key_field="id")

        yield min(start + batch_size, total), total, failed


def create_index(
    view: Any,
    field: str,
    brain_key: str,
    modality: str,
    dim: int = MAX_DIM,
    settings: Optional[dict[str, Any]] = None,
) -> Any:
    """Creates a LanceDB similarity index from an embeddings field.

    The model is recorded on the index, so the App's text search works with
    it, without loading the model here: ``compute_similarity()`` would load
    a copy of the model just to read whether it supports prompts.

    Args:
        view: the samples to index; those without a vector are skipped
        field: the vector field to index
        brain_key: the brain key; an existing index with it is replaced
        modality: one of :data:`MODALITIES`, recorded on the index
        dim (MAX_DIM): the vector size, recorded on the index
        settings (None): the embedding settings, recorded on the index

    Returns:
        a :class:`fiftyone.brain.similarity.SimilarityIndex`
    """
    dataset = view._dataset
    if brain_key in dataset.list_brain_runs():
        delete_index(dataset, brain_key)

    results = fob.compute_similarity(
        view.exists(field),
        embeddings=field,
        backend="lancedb",
        brain_key=brain_key,
        metric="cosine",
        uri=lancedb_uri(),
    )

    # recorded after the fact; see the docstring
    results.config.model = MODEL_NAME
    results.config.model_kwargs = build_model_kwargs(modality, dim, settings)
    results.config.supports_prompts = True
    results.save_config()
    return results


def delete_index(dataset: Any, brain_key: str) -> None:
    """Deletes a brain run and its LanceDB table.

    The brain run is deleted even if the table can't be dropped, so a
    broken index can always be removed.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        brain_key: the index's brain key
    """
    try:
        load_index(dataset, brain_key).cleanup()
    except Exception as e:
        logger.warning("Failed to drop table of '%s': %s", brain_key, e)

    dataset.delete_brain_run(brain_key)
