"""
Cross-modal retrieval panel.

Laid out like FiftyOne's Similarity Search panel: a list of saved searches,
a New Search form and a list of indexes. Python embeds queries and searches
LanceDB; results open in the sample grid. The ``CrossModalRetrievalView``
React component (``js/src``) draws the pages.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import base64
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone

import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import engine

logger = logging.getLogger(__name__)

PLUGIN_URI = "@harpreetsahota/cross-modal-retrieval"
STORE_NAME = "cross_modal_retrieval"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_RUN_NAME = 80


class CrossModalRetrievalPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="cross_modal_retrieval",
            label="Cross-Modal Search",
            icon="travel_explore",
            surfaces="grid",
            help_markdown=(
                "Search images, video and audio with text, selected samples "
                "or a file. EmbeddingGemma 2 puts every modality in one "
                "space, and each index lives in LanceDB"
            ),
        )

    # -- Lifecycle --

    def on_load(self, ctx):
        ctx.panel.set_state("applied", None)
        self._refresh(ctx)

    def on_change_dataset(self, ctx):
        ctx.panel.set_state("applied", None)
        self._refresh(ctx)

    def on_change_selected(self, ctx):
        self._send_selection(ctx, engine.list_indexes(ctx.dataset))

    # -- Methods exposed to the frontend --

    def refresh(self, ctx):
        self._refresh(ctx)

    def search(self, ctx):
        """Runs a search, saves it, and shows its first results in the grid."""
        params = ctx.params
        run = {
            "run_id": str(uuid.uuid4()),
            "run_name": (params.get("run_name") or "").strip()[:MAX_RUN_NAME],
            "query_type": params.get("mode", "text"),
            "query": _stored_query(params),
            "query_label": None,
            "modality": params.get("modality"),
            "brain_keys": params.get("brain_keys") or [],
            "k": params.get("k", 12),
            "status": "completed",
            "error": None,
            "sections": [],
            "result_count": 0,
            "creation_time": datetime.now(timezone.utc).isoformat(),
        }

        try:
            query = self._query(ctx)
            run["query_label"] = query.describe()
            sections = engine.search(
                ctx.dataset,
                query,
                brain_keys=run["brain_keys"] or None,
                k=run["k"],
            )
            run["sections"] = [_section(s) for s in sections]
        except Exception as e:
            logger.warning("Search failed: %s", e)
            run.update(status="failed", error=str(e))

        found = [s for s in run["sections"] if s["ids"]]
        errors = [s["error"] for s in run["sections"] if s["error"]]
        if run["status"] == "completed" and not found and errors:
            run.update(status="failed", error=errors[0])

        run["result_count"] = sum(len(s["ids"]) for s in run["sections"])
        if not run["run_name"]:
            run["run_name"] = _default_name(run)

        history = _History(ctx)
        history.add(run)
        ctx.panel.set_data("model_loaded", engine.is_model_loaded())

        if found:
            self._apply(ctx, run, found[0]["key"])
        elif run["status"] == "failed":
            ctx.ops.notify(run["error"], variant="error")
        else:
            ctx.ops.notify("No matches", variant="warning")

        ctx.panel.set_data("runs", history.list())

    def apply_run(self, ctx):
        """Shows a saved search's results for one of its indexes in the
        grid."""
        run = _History(ctx).get(ctx.params.get("run_id"))
        if not run:
            ctx.ops.notify("Search not found", variant="error")
            return

        self._apply(ctx, run, ctx.params.get("key"))

    def delete_run(self, ctx):
        run_id = ctx.params.get("run_id")
        history = _History(ctx)
        history.delete(run_id)

        applied = ctx.panel.get_state("applied") or {}
        if applied.get("run_id") == run_id:
            ctx.panel.set_state("applied", None)

        ctx.panel.set_data("runs", history.list())

    def clear_view(self, ctx):
        ctx.ops.clear_view()
        ctx.panel.set_state("applied", None)

    def build_index(self, ctx):
        ctx.prompt(PLUGIN_URI + "/build_index", on_success=self.refresh)

    def rebuild_table(self, ctx):
        key = ctx.params.get("key")
        try:
            count = engine.rebuild_table(ctx.dataset, key)
        except Exception as e:
            ctx.ops.notify("Rebuild failed: %s" % e, variant="error")
            return

        ctx.ops.notify(
            "Rebuilt '%s' from %d stored vectors" % (key, count),
            variant="success",
        )
        self._refresh(ctx)

    # -- Render --

    def render(self, ctx):
        return types.Property(
            types.Object(),
            view=types.View(
                component="CrossModalRetrievalView",
                composite_view=True,
                refresh=self.refresh,
                search=self.search,
                apply_run=self.apply_run,
                delete_run=self.delete_run,
                clear_view=self.clear_view,
                build_index=self.build_index,
                rebuild_table=self.rebuild_table,
            ),
        )

    # -- Helpers --

    def _refresh(self, ctx):
        dataset = ctx.dataset
        indexes = engine.list_indexes(dataset)
        ctx.panel.set_data(
            "status",
            {
                "uri": engine.lancedb_uri(),
                "indexes": [engine.describe_index(dataset, i) for i in indexes],
            },
        )
        ctx.panel.set_data("runs", _History(ctx).list())
        ctx.panel.set_data("model_loaded", engine.is_model_loaded())
        self._send_selection(ctx, indexes)

    def _send_selection(self, ctx, indexes):
        ctx.panel.set_data(
            "selection",
            engine.describe_selection(ctx.dataset, ctx.selected, indexes),
        )

    def _apply(self, ctx, run, key=None):
        sections = [s for s in run.get("sections", []) if s["ids"]]
        section = next((s for s in sections if s["key"] == key), None)
        section = section or (sections[0] if sections else None)
        if section is None:
            ctx.ops.notify("This search has no results", variant="warning")
            return

        try:
            view = ctx.dataset.load_brain_view(section["key"]).select(
                section["ids"], ordered=True
            )
        except Exception as e:
            ctx.ops.notify("Failed to show results: %s" % e, variant="error")
            return

        ctx.ops.clear_selected_samples()
        ctx.ops.set_view(view)
        ctx.panel.set_state(
            "applied", {"run_id": run["run_id"], "key": section["key"]}
        )

    def _query(self, ctx):
        params = ctx.params
        mode = params.get("mode", "text")
        if mode == "text":
            text = (params.get("text") or "").strip()
            if not text:
                raise ValueError("Type something to search for")

            return engine.TextQuery(text)

        brain_keys = params.get("brain_keys")
        indexes = [
            i
            for i in engine.list_indexes(ctx.dataset)
            if not brain_keys or i["key"] in brain_keys
        ]

        if mode == "samples":
            sample_ids = [str(i) for i in params.get("sample_ids") or []]
            if not sample_ids:
                raise ValueError("Select samples in the grid first")

            return engine.sample_query(
                ctx.dataset, sample_ids, params["modality"], indexes
            )

        if mode == "file":
            return _file_query(
                params.get("file") or {}, params["modality"], indexes
            )

        raise ValueError("Unknown query mode '%s'" % mode)


class _History(object):
    """The dataset's saved searches, in the plugin's execution store."""

    _PREFIX = "run:"

    def __init__(self, ctx):
        self._store = ctx.store(STORE_NAME)

    def add(self, run):
        self._store.set(self._PREFIX + run["run_id"], run)

    def get(self, run_id):
        return self._store.get(self._PREFIX + run_id) if run_id else None

    def delete(self, run_id):
        if run_id:
            self._store.delete(self._PREFIX + run_id)

    def list(self):
        """Newest first, without result IDs, which only applying needs."""
        runs = []
        for key in self._store.list_keys():
            if not key.startswith(self._PREFIX):
                continue

            run = self._store.get(key)
            if not run:
                continue

            run = dict(run)
            run["sections"] = [
                {
                    "key": s["key"],
                    "modality": s["modality"],
                    "count": len(s["ids"]),
                    "top_score": s["scores"][0] if s["scores"] else None,
                    "error": s["error"],
                }
                for s in run.get("sections", [])
            ]
            runs.append(run)

        runs.sort(key=lambda r: r.get("creation_time", ""), reverse=True)
        return runs


def _section(section):
    return {
        "key": section["key"],
        "modality": section["modality"],
        "ids": [h["id"] for h in section["hits"]],
        "scores": [h["score"] for h in section["hits"]],
        "error": section["error"],
    }


def _stored_query(params):
    """What a search was asked with, as far as a clone can reuse it."""
    mode = params.get("mode", "text")
    if mode == "text":
        return (params.get("text") or "").strip()

    if mode == "samples":
        return [str(i) for i in params.get("sample_ids") or []]

    return (params.get("file") or {}).get("name")


def _default_name(run):
    if run["query_type"] == "text":
        name = run["query"]
    else:
        name = run["query_label"] or run["query"] or "Search"

    name = str(name)
    if len(name) > MAX_RUN_NAME:
        name = name[: MAX_RUN_NAME - 1] + "…"

    return name


def _file_query(upload, modality, indexes):
    name = upload.get("name") or "upload"
    data = upload.get("data") or ""
    if data.startswith("data:"):
        # MIME parameters can hold commas, e.g. "video/webm;codecs=vp8,opus"
        data = data.split(";base64,", 1)[-1]

    content = base64.b64decode(data)
    if not content:
        raise ValueError("Choose a file to search with")
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Files up to 50 MB are supported")

    suffix = os.path.splitext(name)[1].lower()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "query" + suffix)
        with open(path, "wb") as f:
            f.write(content)

        if suffix in engine.RECORDING_EXTENSIONS:
            fixed = os.path.join(
                tmp, "fixed" + engine.RECORDING_EXTENSIONS[suffix]
            )
            engine.remux_recording(path, fixed)
            path = fixed

        return engine.file_query(path, modality, indexes, name=name)
