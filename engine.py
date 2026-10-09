"""
Cross-modal retrieval with EmbeddingGemma 2 and LanceDB.

Every index is a native ``fiftyone.brain`` similarity index with the
``lancedb`` backend. Text, images, video and audio share one embedding space,
so a vector from any of them can query an index of any other.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import contextlib
import logging
import os
import re
import threading

import numpy as np

import fiftyone.brain as fob
import fiftyone.core.fields as fof
import fiftyone.core.labels as fol
import fiftyone.core.media as fomm
import fiftyone.zoo as foz

logger = logging.getLogger(__name__)

MODEL_NAME = "google/embeddinggemma-2"
ZOO_SOURCE = "https://github.com/harpreetsahota204/EmbeddingGemma2"
DEFAULT_LANCEDB_URI = "/tmp/lancedb"

MODALITIES = ("image", "video", "audio", "video+audio")
EMBEDDING_DIMS = (768, 512, 256, 128)
MAX_DIM = 768
DEFAULT_PROMPT = "SearchQuery"
MAX_K = 100

AUDIO_EXTENSIONS = {
    ".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".oga", ".ogg",
    ".opus", ".wav", ".weba", ".wma",
}  # fmt: skip

# Browser recordings (MediaRecorder) are written as a stream, without the
# duration that frame sampling needs, so they are remuxed into these
# containers, which the model's decoder also accepts
RECORDING_EXTENSIONS = {".webm": ".webm", ".weba": ".ogg"}

_CONTAINER_FORMATS = {".webm": "webm", ".ogg": "ogg"}

# What each kind of media can be embedded as
QUERY_MODALITIES = {
    "image": ("image",),
    "video": ("video", "audio", "video+audio"),
    "audio": ("audio",),
    "mcap": ("video", "audio", "video+audio"),
}

_LANCE_METRICS = {"cosine": "cosine", "euclidean": "l2"}

# One model with every encoder serves all modalities. It is shared by the
# App server's request threads, and its settings are mutable, so all use of
# it happens under the lock
_model = None
_model_lock = threading.RLock()


def lancedb_uri():
    """The LanceDB URI the brain uses: the ``lancedb`` backend's ``uri`` in
    the brain config (``FIFTYONE_BRAIN_SIMILARITY_LANCEDB_URI``), else the
    backend's default.

    The URI is not saved with the index, so the App's own similarity search
    reloads indexes from here too.
    """
    params = fob.brain_config.similarity_backends.get("lancedb", {})
    return params.get("uri") or DEFAULT_LANCEDB_URI


###############################################################################
# Model
###############################################################################


def get_model():
    """Returns the shared EmbeddingGemma 2 model, loading it on first use."""
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


def is_model_loaded():
    return _model is not None


def has_gpu():
    import torch

    return torch.cuda.is_available() or torch.backends.mps.is_available()


def slugify(name):
    """Makes a string usable in field names and brain keys."""
    return re.sub(r"\W+", "_", name).strip("_").lower()


@contextlib.contextmanager
def _model_settings(model, settings):
    previous = {k: getattr(model, k) for k in settings}
    for k, v in settings.items():
        setattr(model, k, v)
    try:
        yield
    finally:
        for k, v in previous.items():
            setattr(model, k, v)


def fit_dim(vector, dim):
    """Truncates a vector to a Matryoshka dimension and re-normalizes it."""
    vector = np.asarray(vector, dtype=np.float32)[:dim]
    return vector / np.linalg.norm(vector)


def embed_text(text, prompt_name=DEFAULT_PROMPT):
    """Embeds a text query as a ``(768,)`` unit vector."""
    model = get_model()
    with _model_lock:
        return model.embed_text([text], prompt_name=prompt_name)[0]


def embed_file(path, modality, settings=None):
    """Embeds a media file as a ``(768,)`` unit vector.

    Args:
        path: an image, video, audio or ``.mcap`` filepath
        modality: one of :data:`MODALITIES`
        settings (None): model settings to apply, e.g. ``{"fps": 2}``
    """
    model = get_model()
    with _model_lock, _model_settings(model, settings or {}):
        if modality == "image":
            return model.embed_images([path])[0]

        if modality == "video":
            return model.embed_frames([model.load_frames(path)])[0]

        if modality == "audio":
            return model.embed_audio_file(path)

        return model.embed_video_audio(path)


def media_kind(filepath):
    """Returns ``"image"``, ``"video"``, ``"audio"``, ``"mcap"`` or None."""
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


def query_settings(index, modality):
    """The settings of an index that apply to embedding a query as the given
    modality, so the query is decoded the way the index's samples were.
    """
    return {
        k: v
        for k, v in (index.get("settings") or {}).items()
        if modality in SETTINGS_MODALITIES[k]
    }


def index_modality(model_kwargs):
    """The modality an index's vectors came from, per its ``model_kwargs``."""
    media_type = model_kwargs.get("media_type", "image")
    if media_type == "video" and "audio" in (model_kwargs.get("modalities") or ()):
        return "video+audio"

    return media_type


def build_model_kwargs(modality, dim=MAX_DIM, settings=None):
    """The ``model_kwargs`` stored on an index, so the App can reload the
    model for text search and :func:`index_modality` can tell the modality.
    """
    kwargs = {"media_type": "video" if modality == "video+audio" else modality}
    if modality == "video+audio":
        kwargs["modalities"] = ["vision", "audio"]

    if dim != MAX_DIM:
        kwargs["embedding_dim"] = dim

    kwargs.update(settings or {})
    return kwargs


def flat(dataset):
    """A view with every sample of the dataset, across all group slices."""
    if dataset.media_type == "group":
        return dataset.select_group_slices(_allow_mixed=True)

    return dataset


def list_indexes(dataset):
    """Returns a dict per EmbeddingGemma 2 LanceDB index on the dataset.

    Indexes built in Python count too, if they were created with
    ``model="google/embeddinggemma-2"`` and ``backend="lancedb"``.
    """
    indexes = []
    for key in dataset.list_brain_runs(type=fob.Similarity, method="lancedb"):
        config = dataset.get_brain_info(key).config
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


def load_index(dataset, brain_key):
    """Loads a LanceDB similarity index from the brain's LanceDB URI.

    Not cached: a cached index keeps its table handle after the table's
    files are deleted, and would neither notice nor see a rebuild.
    """
    return dataset.load_brain_results(
        brain_key, uri=lancedb_uri(), cache=False
    )


def describe_index(dataset, index):
    """Adds what the panel shows about an index: size, table health and
    group slices.
    """
    info = dict(index)
    try:
        results = load_index(dataset, index["key"])
        info["table"] = results.config.table_name
        info["count"] = results.total_index_size
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


def rebuild_table(dataset, brain_key):
    """Recreates an index's LanceDB table from its embeddings field, e.g.
    after the URI's directory was cleared.

    Returns:
        the number of vectors written
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
    """A text query, embedded with each target index's prompt."""

    def __init__(self, text):
        self.text = text
        self._vectors = {}

    def describe(self):
        return 'text "%s"' % self.text

    def vector_for(self, index):
        prompt_name = index["prompt_name"]
        if prompt_name not in self._vectors:
            self._vectors[prompt_name] = embed_text(self.text, prompt_name)

        return fit_dim(self._vectors[prompt_name], index["dim"])

    def excludes(self, index):
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

    def __init__(self, vector, label, source_field=None, exclude_ids=()):
        self.vector = np.asarray(vector, dtype=np.float32)
        self.label = label
        self.source_field = source_field
        self.exclude_ids = set(exclude_ids)

    def describe(self):
        return self.label

    def vector_for(self, index):
        if index["dim"] > len(self.vector):
            raise ValueError(
                "The query is %dd but this index is %dd"
                % (len(self.vector), index["dim"])
            )

        return fit_dim(self.vector, index["dim"])

    def excludes(self, index):
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

    def __init__(self, paths, modality, label, indexes):
        self.label = label
        self._modality = modality
        self._vectors = {}
        for index in indexes or [{}]:
            settings = query_settings(index, modality)
            key = _settings_key(settings)
            if key not in self._vectors:
                self._vectors[key] = _mean(
                    [embed_file(p, modality, settings=settings) for p in paths]
                )

    def describe(self):
        return self.label

    def vector_for(self, index):
        key = _settings_key(query_settings(index, self._modality))
        if key not in self._vectors:
            raise ValueError(
                "The query wasn't embedded for index '%s'" % index["key"]
            )

        return fit_dim(self._vectors[key], index["dim"])

    def excludes(self, index):
        return ()


def _settings_key(settings):
    return tuple(sorted(settings.items()))


def _mean(vectors):
    return fit_dim(np.mean(np.stack(vectors), axis=0), len(vectors[0]))


def describe_selection(dataset, sample_ids, indexes):
    """Describes the given samples as a query: their media kind, what they
    can be embedded as, and which of those are already stored in an index.
    """
    if not sample_ids:
        return None

    samples = flat(dataset).select(sample_ids)
    kinds = {media_kind(p) for p in samples.values("filepath")}
    kind = next(iter(kinds)) if len(kinds) == 1 else None
    modalities = list(QUERY_MODALITIES.get(kind, ()))

    stored = {}
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


def sample_query(dataset, sample_ids, modality, indexes):
    """Builds a query from samples of the dataset, as the given modality.

    Uses the vectors stored in an index of that modality when every sample
    has one, otherwise embeds the samples' media.
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

        field = index["field"]
        if samples.exists(field).count() == len(sample_ids):
            return VectorQuery(
                _mean(samples.values(field)),
                label,
                source_field=field,
                exclude_ids=sample_ids,
            )

    return MediaQuery(samples.values("filepath"), modality, label, indexes)


def remux_recording(src, dst):
    """Copies the audio and video packets of a browser recording into a new
    WebM or Ogg file (per ``dst``'s extension), which records the duration
    and seek index the recording lacks. Nothing is re-encoded.
    """
    import av

    fmt = _CONTAINER_FORMATS[os.path.splitext(dst)[1].lower()]
    with av.open(src) as inp, av.open(dst, "w", format=fmt) as out:
        streams = {
            s.index: out.add_stream_from_template(s)
            for s in inp.streams
            if s.type in ("video", "audio")
        }
        if not streams:
            raise ValueError("The recording has no audio or video")

        for packet in inp.demux(*[inp.streams[i] for i in streams]):
            if packet.dts is None:
                continue

            packet.stream = streams[packet.stream.index]
            out.mux(packet)


def file_query(path, modality, indexes, name=None):
    """Builds a query from a media file that is not in the dataset, embedded
    with the settings of each of the given indexes.
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


def _score(distance, metric):
    # cosine distance is 1 - cos; LanceDB's l2 is squared, so for unit
    # vectors it's 2 - 2 cos
    if metric == "euclidean":
        return 1.0 - distance / 2.0

    return 1.0 - distance


def search_index(dataset, index, query, k):
    """Returns the ``[(sample_id, score), ...]`` nearest to the query."""
    results = load_index(dataset, index["key"])
    if results.table is None:
        raise ValueError("LanceDB table not found; rebuild it")

    metric = results.config.metric
    exclude = query.excludes(index)
    rows = (
        results.table.search(query.vector_for(index))
        .distance_type(_LANCE_METRICS[metric])
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


def label_fields(dataset):
    """Top-level fields whose values make a good caption for a result."""
    names = []
    schema = flat(dataset).get_field_schema()
    for name, field in schema.items():
        if isinstance(field, fof.EmbeddedDocumentField):
            if field.document_type in _LABEL_PATHS:
                names.append(name)
        elif isinstance(field, fof.StringField) and name != "filepath":
            names.append(name)

    return names


def _label_path(dataset, label_field):
    field = flat(dataset).get_field(label_field)
    if isinstance(field, fof.EmbeddedDocumentField):
        return label_field + _LABEL_PATHS[field.document_type]

    return label_field


def _caption(value):
    if isinstance(value, list):
        unique = dict.fromkeys(str(v) for v in value if v is not None)
        return ", ".join(unique) or None

    return None if value is None else str(value)


def describe_hits(dataset, index, hits, label_field=None):
    """Adds media and captions to search hits."""
    if not hits:
        return []

    ids = [h[0] for h in hits]
    samples = dataset.load_brain_view(index["key"]).select(ids)

    paths = ["id", "filepath"]
    if label_field:
        paths.append(_label_path(dataset, label_field))
    if dataset.media_type == "group":
        paths += ["%s.id" % dataset.group_field, "%s.name" % dataset.group_field]

    rows = {v[0]: v for v in zip(*samples.values(paths), strict=True)}

    out = []
    for sample_id, score in hits:
        row = rows.get(sample_id)
        if row is None:
            # in the LanceDB table but deleted from the dataset
            continue

        hit = {
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


def search(dataset, query, brain_keys=None, k=12, label_field=None):
    """Runs a query against EmbeddingGemma 2 LanceDB indexes.

    Args:
        dataset: a :class:`fiftyone.core.dataset.Dataset`
        query: a :class:`TextQuery`, :class:`VectorQuery` or
            :class:`MediaQuery`
        brain_keys (None): the indexes to search. By default, all of them
        k (12): results per index, at most :data:`MAX_K`
        label_field (None): a field to caption the results with

    Returns:
        a list with a dict per index: ``key``, ``modality``, ``hits``, and
        ``error`` (None unless that index failed)
    """
    k = max(1, min(int(k), MAX_K))
    indexes = list_indexes(dataset)
    if brain_keys is not None:
        indexes = [i for i in indexes if i["key"] in brain_keys]

    sections = []
    for index in indexes:
        section = {
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


def default_modalities(view):
    """The modalities the samples in a view can be embedded as."""
    if view.media_type == "image":
        return ["image"]

    if view.media_type == "video":
        return ["video", "audio", "video+audio"]

    kinds = {media_kind(p) for p in view.take(20, seed=51).values("filepath")}
    modalities = []
    for kind in ("image", "video", "audio", "mcap"):
        if kind in kinds:
            for m in QUERY_MODALITIES[kind]:
                if m not in modalities:
                    modalities.append(m)

    return modalities


def embed_samples(
    view, field, modality, dim=MAX_DIM, settings=None, overwrite=False
):
    """Embeds the media of each sample into a vector field, in batches.

    Samples whose media fails to decode (e.g. a video with no audio track
    embedded as audio) are logged and left empty.

    Yields:
        ``(num_done, num_total, num_failed)`` after each batch
    """
    dataset = view._dataset
    if not dataset.has_sample_field(field):
        dataset.add_sample_field(field, fof.VectorField)

    todo = view if overwrite else view.exists(field, False)
    ids, paths = todo.values(["id", "filepath"])
    total = len(ids)
    batch_size = 16 if modality == "image" else 4

    failed = 0
    for start in range(0, total, batch_size):
        batch = zip(
            ids[start : start + batch_size],
            paths[start : start + batch_size],
            strict=True,
        )
        values = {}
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


def create_index(view, field, brain_key, modality, dim=MAX_DIM, settings=None):
    """Creates a LanceDB similarity index from an embeddings field.

    The model is recorded on the index, so the App's text search works with
    it, without loading the model here: ``compute_similarity()`` would load
    a copy of the model just to read whether it supports prompts.

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

    results.config.model = MODEL_NAME
    results.config.model_kwargs = build_model_kwargs(modality, dim, settings)
    results.config.supports_prompts = True
    results.save_config()
    return results


def delete_index(dataset, brain_key):
    """Deletes a brain run and its LanceDB table."""
    try:
        load_index(dataset, brain_key).cleanup()
    except Exception as e:
        logger.warning("Failed to drop table of '%s': %s", brain_key, e)

    dataset.delete_brain_run(brain_key)
