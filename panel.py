"""
Cross-modal retrieval panel.

Laid out like FiftyOne's Similarity Search panel: a list of saved searches,
a New Search form and a list of indexes. Python embeds queries and searches
LanceDB; results open in the sample grid. The ``CrossModalRetrievalView``
React component (``js/src``) draws the pages.

How the two halves talk:

-   **Python -> React**: ``ctx.panel.set_data()`` sends ``status`` (indexes),
    ``runs`` (saved searches), ``selection`` and ``model_loaded``, which the
    React view reads as props
-   **React -> Python**: the view calls the panel methods passed to
    :class:`fiftyone.operators.types.View` in :meth:`render` (``search``,
    ``apply_run``, ...) with ``ctx.params``

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, TypedDict

import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators.executor import ExecutionContext

from . import engine

logger = logging.getLogger(__name__)

PLUGIN_URI = "@harpreetsahota/cross-modal-retrieval"

# Name of the execution store that holds saved searches (per dataset)
STORE_NAME = "cross_modal_retrieval"

# Matches the frontend's upload limit; uploads arrive base64-encoded in one
# request, so this also bounds the request size
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

MAX_RUN_NAME = 80


class StoredSection(TypedDict):
    """One index's results, as saved with a search.

    Only IDs and scores are kept: captions and filepaths are looked up again
    when the results are shown, so they stay current.
    """

    key: str  # brain key
    modality: str
    ids: list[str]  # sample IDs, best first
    scores: list[float]  # cosine similarity per ID
    error: Optional[str]


class Run(TypedDict):
    """A saved search, as stored in the execution store."""

    run_id: str  # uuid4
    run_name: str  # user-given, or derived from the query
    query_type: str  # "text", "samples" or "file"
    query: Any  # what a clone needs: the text, sample IDs or file name
    query_label: Optional[str]  # e.g. 'audio of clip.mp4'
    modality: Optional[str]  # what the query was embedded as (not for text)
    brain_keys: list[str]  # indexes searched; [] means all
    k: int  # results per index
    status: str  # "completed" or "failed"
    error: Optional[str]
    sections: list[StoredSection]
    result_count: int  # total hits across indexes
    creation_time: str  # ISO 8601, UTC


class CrossModalRetrievalPanel(foo.Panel):
    """The Cross-Modal Search panel."""

    @property
    def config(self) -> foo.PanelConfig:
        """The panel's name, label and icon; ``surfaces="grid"`` puts it in
        the grid's `+` menu.
        """
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

    def on_load(self, ctx: ExecutionContext) -> None:
        """Sends the initial data when the panel opens."""
        # "applied" marks which search's results are in the grid
        ctx.panel.set_state("applied", None)
        self._refresh(ctx)

    def on_change_dataset(self, ctx: ExecutionContext) -> None:
        """Reloads everything for the newly loaded dataset."""
        ctx.panel.set_state("applied", None)
        self._refresh(ctx)

    def on_change_selected(self, ctx: ExecutionContext) -> None:
        """Updates the Samples query type when the grid selection changes."""
        self._send_selection(ctx, engine.list_indexes(ctx.dataset))

    # -- Methods exposed to the frontend --

    def refresh(self, ctx: ExecutionContext) -> None:
        """Re-sends indexes, saved searches and the selection."""
        self._refresh(ctx)

    def search(self, ctx: ExecutionContext) -> None:
        """Runs a search, saves it, and shows its first results in the grid.

        A failed search is saved too, with its error, so it can be cloned
        and fixed.

        Params:
            mode: ``"text"``, ``"samples"`` or ``"file"``
            text: the query text (text mode)
            sample_ids: the query sample IDs (samples mode)
            file: ``{"name", "data"}`` with a base64 data URL (file mode)
            modality: what to embed the query as (samples and file modes)
            brain_keys: the indexes to search; empty means all
            k: results per index
            run_name: optional name for the saved search
        """
        params = ctx.params
        run: Run = {
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
                # [] would match no index; None means all of them
                brain_keys=run["brain_keys"] or None,
                k=run["k"],
            )
            run["sections"] = [_section(s) for s in sections]
        except Exception as e:
            logger.warning("Search failed: %s", e)
            run.update(status="failed", error=str(e))

        # a search where every index errored counts as failed; one where
        # some index worked but nothing matched is a completed empty search
        found = [s for s in run["sections"] if s["ids"]]
        errors = [s["error"] for s in run["sections"] if s["error"]]
        if run["status"] == "completed" and not found and errors:
            run.update(status="failed", error=errors[0])

        run["result_count"] = sum(len(s["ids"]) for s in run["sections"])
        if not run["run_name"]:
            run["run_name"] = _default_name(run)

        history = _History(ctx)
        history.add(run)
        # the first search loads the model; tell the form it's warm now
        ctx.panel.set_data("model_loaded", engine.is_model_loaded())

        if found:
            self._apply(ctx, run, found[0]["key"])
        elif run["status"] == "failed":
            ctx.ops.notify(run["error"], variant="error")
        else:
            ctx.ops.notify("No matches", variant="warning")

        ctx.panel.set_data("runs", history.list())

    def apply_run(self, ctx: ExecutionContext) -> None:
        """Shows a saved search's results for one of its indexes in the
        grid.

        Params:
            run_id: the saved search
            key (None): the index whose results to show; defaults to the
                first index with results
        """
        run = _History(ctx).get(ctx.params.get("run_id"))
        if not run:
            ctx.ops.notify("Search not found", variant="error")
            return

        self._apply(ctx, run, ctx.params.get("key"))

    def delete_run(self, ctx: ExecutionContext) -> None:
        """Deletes a saved search.

        Params:
            run_id: the saved search
        """
        run_id = ctx.params.get("run_id")
        history = _History(ctx)
        history.delete(run_id)

        # the grid keeps its view, but no search is marked as shown anymore
        applied = ctx.panel.get_state("applied") or {}
        if applied.get("run_id") == run_id:
            ctx.panel.set_state("applied", None)

        ctx.panel.set_data("runs", history.list())

    def clear_view(self, ctx: ExecutionContext) -> None:
        """Resets the grid to the whole dataset."""
        ctx.ops.clear_view()
        ctx.panel.set_state("applied", None)

    def build_index(self, ctx: ExecutionContext) -> None:
        """Opens the ``build_index`` operator's form; refreshes when done."""
        ctx.prompt(PLUGIN_URI + "/build_index", on_success=self.refresh)

    def rebuild_table(self, ctx: ExecutionContext) -> None:
        """Recreates an index's missing LanceDB table from stored vectors.

        Params:
            key: the index's brain key
        """
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

    def render(self, ctx: ExecutionContext) -> types.Property:
        """Hands the whole panel to the React view, with the methods it may
        call.
        """
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

    def _refresh(self, ctx: ExecutionContext) -> None:
        """Sends everything the React view shows."""
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

    def _send_selection(
        self, ctx: ExecutionContext, indexes: list[engine.IndexInfo]
    ) -> None:
        """Sends what the grid selection can be searched as."""
        ctx.panel.set_data(
            "selection",
            engine.describe_selection(ctx.dataset, ctx.selected, indexes),
        )

    def _apply(
        self, ctx: ExecutionContext, run: Run, key: Optional[str] = None
    ) -> None:
        """Loads one index's results of a search into the grid, best first.

        Args:
            ctx: the execution context
            run: the saved search
            key (None): the index to show; falls back to the first index with
                results
        """
        sections = [s for s in run.get("sections", []) if s["ids"]]
        section = next((s for s in sections if s["key"] == key), None)
        section = section or (sections[0] if sections else None)
        if section is None:
            ctx.ops.notify("This search has no results", variant="warning")
            return

        try:
            # load_brain_view() covers every group slice the index holds;
            # ordered=True keeps the ranking in the grid
            view = ctx.dataset.load_brain_view(section["key"]).select(
                section["ids"], ordered=True
            )
        except Exception as e:
            ctx.ops.notify("Failed to show results: %s" % e, variant="error")
            return

        # a stale selection would make the next Samples search confusing
        ctx.ops.clear_selected_samples()
        ctx.ops.set_view(view)
        ctx.panel.set_state(
            "applied", {"run_id": run["run_id"], "key": section["key"]}
        )

    def _query(self, ctx: ExecutionContext) -> engine.Query:
        """Builds the query described by ``ctx.params``.

        Raises:
            ValueError: if the params don't describe a usable query
        """
        params = ctx.params
        mode = params.get("mode", "text")
        if mode == "text":
            text = (params.get("text") or "").strip()
            if not text:
                raise ValueError("Type something to search for")

            # text is embedded lazily, per index prompt
            return engine.TextQuery(text)

        # media queries are embedded up front with each target index's
        # settings, so they need to know the indexes now
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
    """The dataset's saved searches, in the plugin's execution store.

    Each search is one store key, ``run:<run_id>``, so adding or deleting a
    search never rewrites the others.

    Args:
        ctx: the execution context, whose store is scoped to the dataset
    """

    _PREFIX = "run:"

    def __init__(self, ctx: ExecutionContext) -> None:
        self._store = ctx.store(STORE_NAME)

    def add(self, run: Run) -> None:
        """Saves a search."""
        self._store.set(self._PREFIX + run["run_id"], run)

    def get(self, run_id: Optional[str]) -> Optional[Run]:
        """Returns a saved search, or None if there's no such search."""
        return self._store.get(self._PREFIX + run_id) if run_id else None

    def delete(self, run_id: Optional[str]) -> None:
        """Deletes a saved search, if it exists."""
        if run_id:
            self._store.delete(self._PREFIX + run_id)

    def list(self) -> list[dict[str, Any]]:
        """Returns the saved searches for the panel's list, newest first.

        Each section is summarized as ``key``, ``modality``, ``count``,
        ``top_score`` and ``error``, without the result IDs, which only
        :meth:`CrossModalRetrievalPanel.apply_run` needs. That keeps the data
        sent to the browser small.
        """
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

        # ISO 8601 strings sort chronologically
        runs.sort(key=lambda r: r.get("creation_time", ""), reverse=True)
        return runs


def _section(section: engine.Section) -> StoredSection:
    """Reduces an engine search section to what's saved: IDs and scores."""
    return {
        "key": section["key"],
        "modality": section["modality"],
        "ids": [h["id"] for h in section["hits"]],
        "scores": [h["score"] for h in section["hits"]],
        "error": section["error"],
    }


def _stored_query(params: dict[str, Any]) -> Any:
    """What a search was asked with, as far as a clone can reuse it.

    Uploaded files aren't kept, only their names, so a cloned file search
    needs the file again.

    Args:
        params: the ``search`` params

    Returns:
        the text, the list of sample IDs, or the file name
    """
    mode = params.get("mode", "text")
    if mode == "text":
        return (params.get("text") or "").strip()

    if mode == "samples":
        return [str(i) for i in params.get("sample_ids") or []]

    return (params.get("file") or {}).get("name")


def _default_name(run: Run) -> str:
    """Names a search after its query, truncated to :data:`MAX_RUN_NAME`."""
    if run["query_type"] == "text":
        name = run["query"]
    else:
        name = run["query_label"] or run["query"] or "Search"

    name = str(name)
    if len(name) > MAX_RUN_NAME:
        name = name[: MAX_RUN_NAME - 1] + "…"

    return name


def _file_query(
    upload: dict[str, Any], modality: str, indexes: list[engine.IndexInfo]
) -> engine.MediaQuery:
    """Embeds an uploaded or recorded file as a query.

    The file is written to a temporary directory, embedded with each target
    index's settings, and deleted; it never touches the dataset.

    Args:
        upload: ``{"name": ..., "data": ...}``, where ``data`` is a base64
            data URL (or bare base64)
        modality: what to embed the file as
        indexes: the indexes the query will search

    Returns:
        an :class:`engine.MediaQuery`

    Raises:
        ValueError: if the file is empty or larger than
            :data:`MAX_UPLOAD_BYTES`
    """
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

    # the decoders pick the container by extension, so keep the original one
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

        # MediaQuery embeds eagerly, so the temp files can go after this
        return engine.file_query(path, modality, indexes, name=name)
