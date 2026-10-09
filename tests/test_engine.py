"""
Tests for the cross-modal retrieval engine and panel helpers.

They run on a 20-sample clone of the ``cmr-vggsound`` dataset (VGGSound clips
with ``eg2_audio`` / ``eg2_video`` vectors) and load EmbeddingGemma 2, so they
need the dataset, the zoo source and ideally a GPU::

    python -m pytest tests -q
"""
import base64
import os
import uuid

import numpy as np
import pytest

import fiftyone as fo
import fiftyone.brain as fob

import cross_modal_retrieval_plugin.engine as engine
import cross_modal_retrieval_plugin.panel as panel

SOURCE_DATASET = "cmr-vggsound"


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    if not fo.dataset_exists(SOURCE_DATASET):
        pytest.skip("needs the %s dataset" % SOURCE_DATASET)

    backends = fob.brain_config.similarity_backends.setdefault("lancedb", {})
    old_uri = backends.get("uri")
    backends["uri"] = str(tmp_path_factory.mktemp("lancedb"))

    name = "cmr-test-" + uuid.uuid4().hex[:8]
    ds = fo.load_dataset(SOURCE_DATASET).limit(20).clone(name)
    for key in ds.list_brain_runs():
        ds.delete_brain_run(key)

    engine.create_index(ds, "eg2_audio", "audio_idx", "audio")
    engine.create_index(ds, "eg2_video", "video_idx", "video")

    yield ds

    for key in ds.list_brain_runs():
        engine.delete_index(ds, key)
    ds.delete()
    if old_uri is None:
        backends.pop("uri", None)
    else:
        backends["uri"] = old_uri


# -- Pure helpers --


def test_fit_dim_truncates_and_renormalizes():
    v = np.arange(1, 9, dtype=np.float32)
    out = engine.fit_dim(v, 4)
    assert out.shape == (4,)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(out, v[:4] / np.linalg.norm(v[:4]))


def test_media_kind():
    assert engine.media_kind("a/b.mp4") == "video"
    assert engine.media_kind("a/b.WAV") == "audio"
    assert engine.media_kind("a/b.jpg") == "image"
    assert engine.media_kind("a/b.mcap") == "mcap"
    assert engine.media_kind("a/b.txt") is None


def test_slugify():
    assert engine.slugify("Left Camera/RGB") == "left_camera_rgb"


def test_build_model_kwargs_roundtrip():
    for modality in engine.MODALITIES:
        kwargs = engine.build_model_kwargs(modality, dim=256)
        assert engine.index_modality(kwargs) == modality
        assert kwargs["embedding_dim"] == 256


# -- Indexes --


def test_indexes_are_lancedb_brain_runs(dataset):
    indexes = {i["key"]: i for i in engine.list_indexes(dataset)}
    assert set(indexes) == {"audio_idx", "video_idx"}
    assert indexes["audio_idx"]["modality"] == "audio"
    assert indexes["video_idx"]["modality"] == "video"

    info = dataset.get_brain_info("audio_idx")
    assert info.config.method == "lancedb"
    assert info.config.model == engine.MODEL_NAME

    described = engine.describe_index(dataset, indexes["audio_idx"])
    assert described["ok"]
    assert described["count"] == 20


def test_rebuild_table_restores_missing_table(dataset):
    results = engine.load_index(dataset, "audio_idx")
    results.table._conn.drop_table(results.table.name)
    index = next(i for i in engine.list_indexes(dataset) if i["key"] == "audio_idx")
    assert not engine.describe_index(dataset, index)["ok"]

    assert engine.rebuild_table(dataset, "audio_idx") == 20
    assert engine.describe_index(dataset, index)["ok"]


# -- Queries --


def test_text_search_returns_a_section_per_index(dataset):
    sections = engine.search(dataset, engine.TextQuery("a dog barking"), k=5)
    assert [s["key"] for s in sections] == ["audio_idx", "video_idx"]
    for section in sections:
        assert section["error"] is None
        assert len(section["hits"]) == 5
        scores = [h["score"] for h in section["hits"]]
        assert scores == sorted(scores, reverse=True)
        assert all(-1.0 <= s <= 1.0 for s in scores)


def test_search_respects_brain_keys_and_captions(dataset):
    sections = engine.search(
        dataset,
        engine.TextQuery("music"),
        brain_keys=["video_idx"],
        k=3,
        label_field="ground_truth",
    )
    assert [s["key"] for s in sections] == ["video_idx"]
    assert all(h["label"] for h in sections[0]["hits"])


def test_sample_query_uses_stored_vectors_and_skips_itself(dataset):
    indexes = engine.list_indexes(dataset)
    sample_id = dataset.first().id
    query = engine.sample_query(dataset, [sample_id], "audio", indexes)

    stored = np.asarray(dataset.first()["eg2_audio"], dtype=np.float32)
    assert np.allclose(query.vector, stored / np.linalg.norm(stored), atol=1e-5)

    sections = {s["key"]: s for s in engine.search(dataset, query, k=20)}
    assert sample_id not in [h["id"] for h in sections["audio_idx"]["hits"]]
    # Other indexes are a different modality, so the sample is a fair result
    assert sample_id in [h["id"] for h in sections["video_idx"]["hits"]]


def test_describe_selection(dataset):
    indexes = engine.list_indexes(dataset)
    ids = dataset.take(2, seed=0).values("id")
    selection = engine.describe_selection(dataset, ids, indexes)
    assert selection["kind"] == "video"
    assert selection["count"] == 2
    assert selection["error"] is None
    assert selection["stored"] == {"audio": "audio_idx", "video": "video_idx"}
    assert engine.describe_selection(dataset, [], indexes) is None


def test_file_query_from_upload_matches_the_stored_sample(dataset):
    sample = dataset.first()
    with open(sample.filepath, "rb") as f:
        data = "data:video/mp4;base64," + base64.b64encode(f.read()).decode()

    query = panel._file_query(
        {"name": os.path.basename(sample.filepath), "data": data}, "audio"
    )
    hits = engine.search(dataset, query, brain_keys=["audio_idx"], k=1)[0]["hits"]
    assert hits[0]["id"] == sample.id
    assert hits[0]["score"] > 0.95


def test_upload_rejects_empty_file():
    with pytest.raises(ValueError):
        panel._file_query({"name": "x.wav", "data": ""}, "audio")


# -- Build --


def test_embed_samples_and_create_index(dataset):
    view = dataset.limit(3)
    progress = list(engine.embed_samples(view, "eg2_test_av", "video+audio", dim=256))
    done, total, failed = progress[-1]
    assert (done, total, failed) == (3, 3, 0)
    assert len(view.first()["eg2_test_av"]) == 256

    engine.create_index(view, "eg2_test_av", "av_idx", "video+audio", dim=256)
    index = next(i for i in engine.list_indexes(dataset) if i["key"] == "av_idx")
    assert index["dim"] == 256
    assert index["modality"] == "video+audio"

    sections = engine.search(dataset, engine.TextQuery("a guitar"), brain_keys=["av_idx"], k=3)
    assert len(sections[0]["hits"]) == 3

    engine.delete_index(dataset, "av_idx")
    assert "av_idx" not in dataset.list_brain_runs()
