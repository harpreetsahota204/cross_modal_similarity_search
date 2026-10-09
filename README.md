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

1. Open the **Cross-Modal Retrieval** panel from the `+` next to the grid's tab
2. Click **Build an index** and choose what to embed the samples as:

   | Modality | Embeds | Good for |
   |---|---|---|
   | Image | the image | photos |
   | Video | frames sampled from the video | what a clip shows |
   | Audio | the soundtrack, or an audio file | what a clip sounds like |
   | Video + audio | frames and soundtrack as one vector | both at once |

   A video dataset can have all three video indexes side by side
3. Search with one of the three query tabs:
   - **Text**: a description. Each index embeds it with the retrieval prompt
   - **Selection**: the samples selected in the grid, as any modality their
     media supports. Several samples are averaged into one query
   - **File**: an image, video or audio file that isn't in the dataset

Results come back as one row per index, with players for the media. From a
result you can:

- **Open** it in the sample modal
- **Similar**: search every index again with that sample
- **Show in grid**: show that index's results in the grid, in rank order.
  **Restore grid** puts back the view you had

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
npm test                                # format helpers

# engine tests, on a clone of a VGGSound dataset named cmr-vggsound
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q
```
