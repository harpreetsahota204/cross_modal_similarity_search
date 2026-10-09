# Cross-Modal Retrieval

Search the images, video and audio in a FiftyOne dataset with words, with
other samples, or with a file you drop in. Ask for "a dog barking" and get the
clips that sound like it; pick a clip and find the videos that look like its
sound.

[EmbeddingGemma 2](https://github.com/harpreetsahota204/EmbeddingGemma2) puts
text, images, video and audio in one 768-dimensional embedding space, so a
vector from any modality can query an index of any other. Every index is a
FiftyOne similarity index with the
[LanceDB backend](https://docs.voxel51.com/api/fiftyone.brain.internal.core.lancedb.html),
which means the App's own similarity search and `sort_by_similarity()` work on
them too.

## Install

```shell
fiftyone plugins download https://github.com/harpreetsahota204/cross_modal_retrieval_plugin
fiftyone plugins requirements @harpreetsahota/cross-modal-retrieval --install
```

Then register the model once:

```python
import fiftyone.zoo as foz

foz.register_zoo_model_source("https://github.com/harpreetsahota204/EmbeddingGemma2")
```

The model is 740M parameters. It runs on CPU, but building video indexes
wants a GPU.

## Use it

The panel is laid out like FiftyOne's Similarity Search panel.

1. Open the **Cross-Modal Search** panel from the `+` next to the grid's tab
2. Click **Build Cross-Modal Index** and choose what to embed the samples as:

   | Modality | Embeds | Good for |
   |---|---|---|
   | Image | the image | photos |
   | Video | frames sampled from the video | what a clip shows |
   | Audio | the soundtrack, or an audio file | what a clip sounds like |
   | Video + audio | frames and soundtrack as one vector | both at once |

   A video dataset can have all three video indexes side by side
3. Click **New Search**, pick one or more indexes, and search by:
   - **Text**: a description. Each index embeds it with the retrieval prompt
   - **Samples**: the samples selected in the grid, as any modality their
     media supports. Several samples are averaged into one query
   - **Media**: upload an image, video, audio or MCAP file, record a clip from
     your webcam or microphone, or take a photo. Nothing is added to the
     dataset. The webcam and microphone need the App on `localhost` or https

   Media is embedded with the same settings as each index's samples, so a
   query and the dataset go through the same preprocessing

The results open in the grid in rank order, and the search is saved to the
panel's list. Click a saved search to show its results again; a search of
several indexes has a button per index. Searches can be cloned or deleted,
and the gear button lists the indexes, with **Rebuild table** for an index
whose LanceDB table is missing.

## Indexes made in Python

The panel finds any LanceDB index made with the model:

```python
import fiftyone.brain as fob

fob.compute_similarity(
    dataset,
    model="google/embeddinggemma-2",
    model_kwargs={"media_type": "video", "modalities": ["vision", "audio"]},
    embeddings="eg2_av",
    backend="lancedb",
    brain_key="eg2_av",
)
```

`model_kwargs` tells the panel the modality: `media_type` is `image`, `video`
or `audio`, and `modalities=["vision", "audio"]` on a video index means
video + audio.

## Where the vectors live

The LanceDB backend doesn't save its URI with an index, so every load uses the
brain config's `lancedb` URI, which defaults to `/tmp/lancedb`. Set it
somewhere permanent in `~/.fiftyone/brain_config.json`:

```json
{
    "similarity_backends": {
        "lancedb": {"uri": "~/fiftyone/lancedb"}
    }
}
```

The vectors are also stored on the samples, so if the table goes missing
(e.g. `/tmp` was cleared), the panel marks the index and **Rebuild**
recreates it from the field.

## Operators

| Operator | What it does |
|---|---|
| `build_index` | Embeds the target view and creates a LanceDB index. Can run delegated |
| `search` | Searches the indexes with text or the selected samples and returns the hits |

## Development

```shell
cd js && npm install && npm run build   # panel bundle -> js/dist/index.umd.js
```

Commit the rebuilt `js/dist/index.umd.js`; the App loads it as is.
