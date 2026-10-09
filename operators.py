"""
Cross-modal retrieval operators.

| Copyright 2017-2026, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|
"""
import fiftyone.operators as foo
import fiftyone.operators.types as types

from . import engine

_MODALITY_LABELS = {
    "image": "Image",
    "video": "Video (frames)",
    "audio": "Audio",
    "video+audio": "Video + audio (one vector)",
}

_FIELD_SUFFIXES = {
    "image": "image",
    "video": "video",
    "audio": "audio",
    "video+audio": "av",
}

# Model settings the build operator exposes:
# name -> (modalities they apply to, label, description, default)
_SETTINGS = {
    "fps": (
        ("video", "video+audio"),
        "Frames per second",
        "Video frames sampled per second",
        1.0,
    ),
    "max_frames": (
        ("video", "video+audio"),
        "Max frames",
        "Frames per video, spread over its length",
        32,
    ),
    "max_audio_seconds": (
        ("audio", "video+audio"),
        "Max audio seconds",
        "Audio longer than this is truncated",
        30.0,
    ),
}


def _build_view(ctx):
    view = ctx.target_view()
    group_slice = ctx.params.get("group_slice")
    if ctx.dataset.media_type == "group" and group_slice:
        view = view.select_group_slices(group_slice)

    return view


def _default_name(ctx, modality):
    name = "eg2_" + _FIELD_SUFFIXES[modality]
    group_slice = ctx.params.get("group_slice")
    if ctx.dataset.media_type == "group" and group_slice:
        name += "_" + engine.slugify(group_slice)

    return name


class BuildIndex(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="build_index",
            label="Cross-modal: build EmbeddingGemma 2 index",
            description=(
                "Embeds images, video, audio or video+audio with EmbeddingGemma "
                "2 and indexes the vectors in LanceDB, so text and media of "
                "any modality can search them"
            ),
            icon="travel_explore",
            dynamic=True,
            execute_as_generator=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.view_target(ctx)

        if ctx.dataset.media_type == "group":
            slices = ctx.dataset.group_slices
            inputs.enum(
                "group_slice",
                slices,
                default=ctx.group_slice or ctx.dataset.default_group_slice,
                required=True,
                label="Group slice",
                description="Each index holds the samples of one slice",
                view=types.DropdownView(),
            )

        view = _build_view(ctx)
        modalities = engine.default_modalities(view)
        if not modalities:
            inputs.view(
                "no_media",
                types.Error(
                    label="Nothing to embed",
                    description=(
                        "EmbeddingGemma 2 embeds images, video, audio and "
                        "MCAP files; these samples have none of them"
                    ),
                ),
            )
            return types.Property(inputs)

        choices = types.RadioGroup()
        for m in modalities:
            choices.add_choice(m, label=_MODALITY_LABELS[m])

        inputs.enum(
            "modality",
            choices.values(),
            default=modalities[0],
            required=True,
            label="Embed the samples as",
            view=choices,
        )
        modality = ctx.params.get("modality") or modalities[0]
        default_name = _default_name(ctx, modality)

        inputs.str(
            "embeddings_field",
            default=default_name,
            required=True,
            label="Embeddings field",
            description="Sample field that stores the vectors",
        )
        field = ctx.params.get("embeddings_field") or default_name

        total = view.count()
        existing = (
            view.exists(field).count()
            if ctx.dataset.has_sample_field(field)
            else 0
        )
        if existing:
            inputs.bool(
                "overwrite",
                default=False,
                label="Recompute existing embeddings",
                description=(
                    "%d of %d samples already have vectors in '%s'. Unticked, "
                    "only the rest are embedded" % (existing, total, field)
                ),
                view=types.CheckboxView(),
            )

        inputs.str(
            "brain_key",
            default=default_name,
            required=True,
            label="Index name (brain key)",
        )
        brain_key = ctx.params.get("brain_key") or default_name
        if not brain_key.isidentifier():
            inputs.view(
                "bad_key",
                types.Error(
                    label="Use letters, numbers and underscores only",
                ),
            )
        elif brain_key in ctx.dataset.list_brain_runs():
            inputs.view(
                "key_exists",
                types.Warning(
                    label="'%s' exists and will be replaced" % brain_key
                ),
            )

        inputs.enum(
            "embedding_dim",
            list(engine.EMBEDDING_DIMS),
            default=engine.MAX_DIM,
            label="Dimensions",
            description="Smaller is faster to search; 128 loses a lot of quality",
            view=types.DropdownView(),
        )

        for name, (used_by, label, description, default) in _SETTINGS.items():
            if modality in used_by:
                add = inputs.float if isinstance(default, float) else inputs.int
                add(name, default=default, label=label, description=description)

        overwrite = ctx.params.get("overwrite", False)
        to_embed = total if overwrite else total - existing
        inputs.view(
            "summary",
            types.Notice(
                label="%d samples to embed, %d to index. Runs on %s"
                % (to_embed, total, "GPU" if engine.has_gpu() else "CPU")
            ),
        )

        return types.Property(
            inputs, view=types.View(label="Build cross-modal index")
        )

    def execute(self, ctx):
        view = _build_view(ctx)
        modality = ctx.params["modality"]
        field = ctx.params["embeddings_field"]
        brain_key = ctx.params["brain_key"]
        dim = int(ctx.params.get("embedding_dim", engine.MAX_DIM))
        settings = {
            name: ctx.params[name]
            for name, (used_by, *_) in _SETTINGS.items()
            if modality in used_by and ctx.params.get(name) is not None
        }

        failed = 0
        for done, total, failed in engine.embed_samples(
            view,
            field,
            modality,
            dim=dim,
            settings=settings,
            overwrite=ctx.params.get("overwrite", False),
        ):
            label = "Embedded %d of %d samples" % (done, total)
            if ctx.delegated:
                ctx.set_progress(progress=done / total, label=label)
            else:
                yield ctx.trigger(
                    "set_progress", {"progress": done / total, "label": label}
                )

        results = engine.create_index(
            view, field, brain_key, modality, dim=dim, settings=settings
        )

        if not ctx.delegated:
            yield ctx.trigger("reload_dataset")

        yield {
            "brain_key": brain_key,
            "indexed": results.total_index_size,
            "failed": failed,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("brain_key", label="Index")
        outputs.int("indexed", label="Vectors indexed")
        outputs.int("failed", label="Samples that failed to embed")
        return types.Property(outputs, view=types.View(label="Index ready"))


class Search(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="search",
            label="Cross-modal: search",
            description=(
                "Searches EmbeddingGemma 2 LanceDB indexes with text or with "
                "the selected samples, across modalities, and returns the "
                "nearest samples of each index"
            ),
            icon="travel_explore",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        indexes = engine.list_indexes(ctx.dataset)
        if not indexes:
            inputs.view(
                "no_indexes",
                types.Warning(
                    label="No cross-modal indexes",
                    description="Run 'Cross-modal: build EmbeddingGemma 2 index' first",
                ),
            )
            return types.Property(inputs)

        selection = engine.describe_selection(
            ctx.dataset, ctx.selected, indexes
        )
        query_types = types.RadioGroup()
        query_types.add_choice("text", label="Text")
        if selection and not selection["error"]:
            query_types.add_choice(
                "samples", label="%d selected samples" % selection["count"]
            )

        inputs.enum(
            "query_type",
            query_types.values(),
            default="text",
            label="Query with",
            view=query_types,
        )

        if ctx.params.get("query_type", "text") == "text":
            inputs.str("text", required=True, label="Text")
        else:
            inputs.enum(
                "modality",
                selection["modalities"],
                default=selection["modalities"][0],
                label="Embed the selection as",
                view=types.DropdownView(),
            )

        index_choices = types.Choices()
        for index in indexes:
            index_choices.add_choice(
                index["key"],
                label="%s (%s)" % (index["key"], index["modality"]),
            )

        inputs.list(
            "brain_keys",
            types.String(),
            default=[i["key"] for i in indexes],
            label="Indexes to search",
            view=types.AutocompleteView(
                choices=index_choices.choices, allow_user_input=False
            ),
        )
        inputs.int(
            "k", default=12, label="Results per index", min=1, max=engine.MAX_K
        )

        return types.Property(inputs, view=types.View(label="Cross-modal search"))

    def execute(self, ctx):
        query = _query_from_params(ctx)
        sections = engine.search(
            ctx.dataset,
            query,
            brain_keys=ctx.params.get("brain_keys"),
            k=ctx.params.get("k", 12),
        )
        return {"query": query.describe(), "results": sections}

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str("query", label="Query")
        outputs.list("results", types.Object(), label="Results per index")
        return types.Property(outputs, view=types.View(label="Search results"))


def _query_from_params(ctx):
    if ctx.params.get("query_type", "text") == "text":
        text = (ctx.params.get("text") or "").strip()
        if not text:
            raise ValueError("Enter a text query")

        return engine.TextQuery(text)

    sample_ids = list(ctx.params.get("sample_ids") or ctx.selected or [])
    if not sample_ids:
        raise ValueError("Select samples to query with")

    return engine.sample_query(
        ctx.dataset,
        sample_ids,
        ctx.params["modality"],
        engine.list_indexes(ctx.dataset),
    )
