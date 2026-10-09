"""
Cross-modal retrieval panel.

Python embeds queries and searches LanceDB; the ``CrossModalRetrievalView``
React component (``js/src``) draws the query bar and one strip of results per
index, playing images, video and audio inline.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import base64
import logging
import os
import tempfile

import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import engine

logger = logging.getLogger(__name__)

PLUGIN_URI = "@harpreetsahota/cross-modal-retrieval"
STORE_NAME = "cross_modal_retrieval"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
GRID_TTL = 24 * 60 * 60


class CrossModalRetrievalPanel(foo.Panel):
    @property
    def config(self):
        return foo.PanelConfig(
            name="cross_modal_retrieval",
            label="Cross-Modal Retrieval",
            icon="travel_explore",
            surfaces="grid",
            help_markdown=(
                "Search images, video and audio with text, a selected sample "
                "or a file. EmbeddingGemma 2 puts every modality in one "
                "space, and each index lives in LanceDB"
            ),
        )

    # -- Lifecycle --

    def on_load(self, ctx):
        self._refresh(ctx)

    def on_change_dataset(self, ctx):
        ctx.panel.set_data("results", None)
        self._refresh(ctx)

    def on_change_selected(self, ctx):
        self._send_selection(ctx, engine.list_indexes(ctx.dataset))

    # -- Methods exposed to the frontend --

    def refresh(self, ctx):
        self._refresh(ctx)

    def search(self, ctx):
        params = ctx.params
        try:
            query = self._query(ctx)
            sections = engine.search(
                ctx.dataset,
                query,
                brain_keys=params.get("brain_keys"),
                k=params.get("k", 12),
                label_field=params.get("label_field") or None,
            )
        except Exception as e:
            logger.warning("Search failed: %s", e)
            ctx.panel.set_data("error", str(e))
            return

        ctx.panel.set_data("error", None)
        ctx.panel.set_data(
            "results", {"query": query.describe(), "sections": sections}
        )
        ctx.panel.set_data("model_loaded", True)

    def show_in_grid(self, ctx):
        """Shows the given results of an index in the grid, in order, and
        remembers the view to come back to."""
        key = ctx.params.get("key")
        ids = [str(i) for i in ctx.params.get("ids") or []]
        if not key or not ids:
            return

        store = ctx.store(STORE_NAME)
        current = store.get(_grid_key(ctx))
        base = current["base_view"] if current else ctx.view._serialize()
        store.set(
            _grid_key(ctx), {"key": key, "base_view": base}, ttl=GRID_TTL
        )

        view = ctx.dataset.load_brain_view(key).select(ids, ordered=True)
        ctx.ops.set_view(view)
        ctx.panel.set_data("grid", {"key": key, "count": len(ids)})

    def clear_grid(self, ctx):
        """Puts the grid back to the view it showed before."""
        store = ctx.store(STORE_NAME)
        current = store.get(_grid_key(ctx))
        store.delete(_grid_key(ctx))
        ctx.panel.set_data("grid", None)

        if current and current["base_view"]:
            ctx.ops.set_view(
                fo.DatasetView._build(ctx.dataset, current["base_view"])
            )
        else:
            ctx.ops.clear_view()

    def open_sample(self, ctx):
        sample_id = ctx.params.get("id")
        group_id = ctx.params.get("group_id")
        if group_id and ctx.view.media_type == "group":
            ctx.ops.set_group_slice(ctx.params.get("slice"))
            ctx.ops.open_sample(group_id=group_id)
        elif sample_id:
            ctx.ops.open_sample(id=sample_id)

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
                show_in_grid=self.show_in_grid,
                clear_grid=self.clear_grid,
                open_sample=self.open_sample,
                build_index=self.build_index,
                rebuild_table=self.rebuild_table,
            ),
        )

    # -- Helpers --

    def _refresh(self, ctx):
        dataset = ctx.dataset
        indexes = engine.list_indexes(dataset)
        label_fields = engine.label_fields(dataset)
        ctx.panel.set_data(
            "status",
            {
                "dataset": dataset.name,
                "uri": engine.lancedb_uri(),
                "indexes": [engine.describe_index(dataset, i) for i in indexes],
                "label_fields": label_fields,
                "default_label_field": _default_label_field(label_fields),
            },
        )
        ctx.panel.set_data("model_loaded", engine.is_model_loaded())
        current = ctx.store(STORE_NAME).get(_grid_key(ctx))
        ctx.panel.set_data("grid", {"key": current["key"]} if current else None)
        self._send_selection(ctx, indexes)

    def _send_selection(self, ctx, indexes):
        ctx.panel.set_data(
            "selection",
            engine.describe_selection(ctx.dataset, ctx.selected, indexes),
        )

    def _query(self, ctx):
        params = ctx.params
        mode = params.get("mode", "text")
        if mode == "text":
            text = (params.get("text") or "").strip()
            if not text:
                raise ValueError("Type something to search for")

            return engine.TextQuery(text)

        if mode == "samples":
            sample_ids = [str(i) for i in params.get("sample_ids") or []]
            if not sample_ids:
                raise ValueError("Select samples in the grid first")

            return engine.sample_query(
                ctx.dataset,
                sample_ids,
                params["modality"],
                engine.list_indexes(ctx.dataset),
            )

        if mode == "file":
            return _file_query(params.get("file") or {}, params["modality"])

        raise ValueError("Unknown query mode '%s'" % mode)


def _file_query(upload, modality):
    name = upload.get("name") or "upload"
    data = upload.get("data") or ""
    if "," in data:
        data = data.split(",", 1)[1]  # data URL

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

        return engine.file_query(path, modality, name=name)


def _default_label_field(label_fields):
    for name in ("ground_truth", "label", "caption"):
        if name in label_fields:
            return name

    return None


def _grid_key(ctx):
    return "grid:%s" % ctx.panel_id
