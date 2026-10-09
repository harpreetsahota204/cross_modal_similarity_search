"""Render the README figures.

Style and palette follow lerobot_dataset_downloads/community_dataset_v3/figures.
Figures 05-07 use real vectors from the FiftyOne dataset `t1-vggsound-eg2`
(100 VGGSound clips, 62 classes). The class-name text embeddings are cached in
`.figure_data.npz`; delete it to recompute (needs the model).

Run from the repo root:  python figures/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent
CACHE = OUT / ".figure_data.npz"
DS_NAME = "t1-vggsound-eg2"
INDEXES = ("eg2_video", "eg2_audio", "eg2_av")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "none"})

THEMES = {
    "light": dict(INK="#1f2933", MUTED="#7b8794", LINE="#cbd2d9", BG="#ffffff", PANEL="#f5f7fa",
                  FO_FILL="#fff5ee", LANCE_FILL="#eef0fc", SPACE_FILL="#fbfcfd", CLUSTER="#eef1f5",
                  FO_DARK="#c2530a", LANCE="#4c5fd5"),
    # GitHub's dark background
    "dark": dict(INK="#e6edf3", MUTED="#8b949e", LINE="#3d444d", BG="#0d1117", PANEL="#161b22",
                 FO_FILL="#2a1a0f", LANCE_FILL="#171c3a", SPACE_FILL="#11161d", CLUSTER="#1c232c",
                 FO_DARK="#ff8a3d", LANCE="#8b98f5"),
}
THEME = "light"
globals().update(THEMES[THEME])
TEXT_C = "#7c5cbf"   # text
IMG = "#3e7cb1"      # image / video
AUDIO = "#2f9e78"    # audio
AV = "#e07a1f"       # video + audio
PAD = "#d64545"
FO = "#ff6d04"
MONO = "DejaVu Sans Mono"

INDEX_COLOR = {"eg2_video": IMG, "eg2_audio": AUDIO, "eg2_av": AV}
INDEX_LABEL = {"eg2_video": "Video", "eg2_audio": "Audio", "eg2_av": "Video + audio"}


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def box(ax, x, y, w, h, color, text="", fc=None, lw=1.4, fontsize=10, textcolor=None,
        style="round,pad=0.02,rounding_size=0.06", z=2, family=None, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, ec=color, fc=fc or color, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor or INK, zorder=z + 1, family=family, weight=weight)


def arrow(ax, x0, y0, x1, y1, color=None, lw=1.6, ms=14, ls="-", conn="arc3,rad=0", z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color or MUTED,
                                 lw=lw, zorder=z, linestyle=ls, connectionstyle=conn))


def label(ax, x, y, s, size=10, color=None, ha="center", va="center", weight="normal", family=None, z=5, **kw):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color or INK, weight=weight, family=family, zorder=z, **kw)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def style_axes(ax, title=""):
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(labelsize=8.5, colors=MUTED)
    ax.grid(alpha=0.2)


def use_theme(name):
    global THEME
    THEME = name
    globals().update(THEMES[name])
    plt.rcParams.update({"text.color": INK, "axes.facecolor": BG, "figure.facecolor": BG,
                         "axes.labelcolor": MUTED})


def save(fig, name):
    if THEME == "dark":
        name = name.replace(".png", "_dark.png")
    fig.savefig(OUT / name, dpi=170, bbox_inches="tight", facecolor=BG, pad_inches=0.15)
    plt.close(fig)
    print("wrote", OUT / name)


def film_strip(ax, x, y, w, h, n, color=IMG, highlight=None):
    fw = w / n
    for i in range(n):
        on = highlight is None or i in highlight
        ax.add_patch(Rectangle((x + i * fw, y), fw - 0.03, h, fc=color, ec=BG, lw=0.8,
                               alpha=0.9 if on else 0.2, zorder=2))


def waveform(ax, x, y, w, h, color=AUDIO, seed=0, lw=1.2, alpha=1.0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 160)
    env = 0.35 + 0.65 * np.abs(np.sin(t * 7 + 0.5)) * (0.6 + 0.4 * rng.random(t.size))
    sig = env * np.sin(t * 90 + rng.random() * 3)
    ax.plot(x + t * w, y + h / 2 + sig * h / 2, color=color, lw=lw, zorder=3, alpha=alpha)


def photo(ax, x, y, w, h, color=IMG):
    ax.add_patch(Rectangle((x, y), w, h, fc=BG, ec=color, lw=1.4, zorder=2))
    ax.add_patch(plt.Polygon([(x + 0.08 * w, y + 0.12 * h), (x + 0.4 * w, y + 0.65 * h), (x + 0.6 * w, y + 0.35 * h),
                              (x + 0.75 * w, y + 0.5 * h), (x + 0.92 * w, y + 0.12 * h)], fc=color, alpha=0.6, zorder=3))
    ax.add_patch(Circle((x + 0.75 * w, y + 0.78 * h), 0.09 * h, fc=AV, zorder=3))


def vector(ax, x, y, color, n=8, cell=0.16, seed=0):
    """A short row of cells standing for an embedding vector."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        ax.add_patch(Rectangle((x + i * cell, y), cell * 0.9, cell * 1.6, fc=color, ec="none",
                               alpha=0.35 + 0.65 * rng.random(), zorder=3))


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load_data():
    if CACHE.exists():
        return dict(np.load(CACHE, allow_pickle=True))

    import fiftyone as fo

    sys.path.insert(0, str(OUT.parent))
    import engine

    ds = fo.load_dataset(DS_NAME)
    labels = np.array(ds.values("ground_truth.label"))
    classes = sorted(set(labels))
    d = {"labels": labels, "classes": np.array(classes),
         "text": np.stack([engine.embed_text(c) for c in classes])}
    for f in INDEXES:
        v = np.stack(ds.values(f))
        d[f] = v / np.linalg.norm(v, axis=1, keepdims=True)
    np.savez(CACHE, **d)
    return d


def first_hit_ranks(d, field):
    sims = d["text"] @ d[field].T
    ranks = []
    for i, c in enumerate(d["classes"]):
        order = np.argsort(-sims[i])
        ranks.append(int(np.argmax(d["labels"][order] == c)) + 1)
    return np.array(ranks)


# ----------------------------------------------------------------------------------------------
# 01 shared space
# ----------------------------------------------------------------------------------------------
def fig_shared_space():
    fig, ax = canvas(13, 5.6)
    label(ax, 6.5, 5.3, "EmbeddingGemma 2 puts every modality in one 768-dimensional space",
          size=13, weight="bold")

    rows = [("Text", TEXT_C, 4.2), ("Image", IMG, 3.2), ("Video", IMG, 2.2), ("Audio", AUDIO, 1.2)]
    for name, c, y in rows:
        label(ax, 0.3, y, name, size=10, weight="bold", color=c, ha="left")
        if name == "Text":
            box(ax, 1.2, y - 0.25, 1.9, 0.5, c, "“church bells”", fc=BG, fontsize=9, textcolor=c)
        elif name == "Image":
            photo(ax, 1.6, y - 0.3, 0.9, 0.6, c)
        elif name == "Video":
            film_strip(ax, 1.2, y - 0.22, 1.9, 0.44, 7, c)
        else:
            waveform(ax, 1.2, y - 0.3, 1.9, 0.6, c, seed=5)
        arrow(ax, 3.25, y, 4.35, 2.7 + (y - 2.7) * 0.3, color=c, lw=1.4)

    box(ax, 4.3, 1.7, 2.3, 2.0, INK, fc=PANEL, lw=1.4)
    label(ax, 5.45, 3.0, "EmbeddingGemma 2", size=9.5, weight="bold")
    label(ax, 5.45, 2.55, "740M parameters", size=8.5, color=MUTED)
    label(ax, 5.45, 2.2, "one model for text,", size=8.5, color=MUTED)
    label(ax, 5.45, 1.95, "image, video, audio", size=8.5, color=MUTED)
    arrow(ax, 6.5, 2.7, 7.3, 2.7, color=INK, lw=1.6)

    ax.add_patch(FancyBboxPatch((7.4, 0.55), 5.3, 4.3, boxstyle="round,pad=0.02,rounding_size=0.1",
                                ec=LINE, fc=SPACE_FILL, lw=1.2, zorder=1))
    label(ax, 7.6, 4.6, "shared space (sketch)", size=9, color=MUTED, ha="left")
    rng = np.random.default_rng(7)
    markers = {TEXT_C: "*", IMG: "s", AUDIO: "o"}
    clusters = [(8.8, 3.6, "church bells"), (11.2, 3.5, "dog barking"), (9.7, 1.5, "rain on a window")]
    for x, y, name in clusters:
        ax.add_patch(Circle((x, y), 0.72, fc=CLUSTER, ec="none", zorder=1.5))
        for c, m in markers.items():
            n = 1 if c == TEXT_C else 3
            pts = rng.normal(0, 0.22, (n, 2))
            ax.scatter(x + pts[:, 0], y + pts[:, 1], marker=m, s=150 if m == "*" else 46, color=c,
                       ec=BG, lw=0.6, zorder=4)
        label(ax, x, y - 0.95, name, size=9, color=INK)
    for c, m, t, yy in [(TEXT_C, "*", "text", 1.75), (IMG, "s", "image / video", 1.45), (AUDIO, "o", "audio", 1.15)]:
        ax.scatter([11.3], [yy], marker=m, s=110 if m == "*" else 40, color=c, zorder=4)
        label(ax, 11.48, yy, t, size=8.5, color=MUTED, ha="left")

    label(ax, 6.5, 0.15, "Things that mean the same land close together, whatever their modality, "
          "so a vector of any kind can search an index of any other kind.", size=9.5, color=MUTED)
    save(fig, "01_shared_space.png")


# ----------------------------------------------------------------------------------------------
# 02 what each index embeds
# ----------------------------------------------------------------------------------------------
def fig_index_modalities():
    fig, ax = canvas(13, 6.4)
    label(ax, 6.5, 6.55, "One clip, three indexes: choose which part of each sample becomes its vector",
          size=13, weight="bold")

    box(ax, 0.3, 2.2, 3.1, 2.4, FO, fc=FO_FILL, lw=1.6)
    label(ax, 1.85, 4.3, "a video sample", size=10.5, weight="bold", color=FO_DARK)
    film_strip(ax, 0.55, 3.35, 2.6, 0.55, 8, IMG)
    label(ax, 1.85, 3.15, "frames", size=8.5, color=MUTED)
    waveform(ax, 0.55, 2.4, 2.6, 0.55, AUDIO, seed=1)
    label(ax, 1.85, 2.3, "soundtrack", size=8.5, color=MUTED)

    rows = [
        (1, "Video", IMG, 4.85, "frames only; the sound is ignored",
         "1 frame per second, at most 32, spread over the whole clip", "what it shows", "film"),
        (2, "Audio", AUDIO, 3.05, "soundtrack only; the picture is ignored",
         "resampled to 16 kHz mono; the first 30 seconds", "what it sounds like", "wave"),
        (3, "Video + audio", AV, 1.25, "frames and soundtrack as ONE vector",
         "same frame and audio settings, one input to the model", "both at once", "both"),
    ]
    for num, name, c, y, what, how, finds, kind in rows:
        arrow(ax, 3.45, 3.4, 4.45, y + 0.6, color=c, lw=1.6)
        box(ax, 4.5, y, 8.2, 1.25, c, fc=BG, lw=1.5)
        ax.add_patch(Rectangle((4.5, y), 0.55, 1.25, fc=c, ec="none", zorder=3))
        label(ax, 4.775, y + 0.62, str(num), size=15, weight="bold", color="white")
        label(ax, 5.25, y + 0.95, name, size=11, weight="bold", color=c, ha="left")
        label(ax, 5.25, y + 0.6, what, size=9.5, ha="left")
        label(ax, 5.25, y + 0.27, how, size=8.5, color=MUTED, ha="left")
        if kind in ("film", "both"):
            film_strip(ax, 9.75, y + (0.7 if kind == "both" else 0.42), 1.2, 0.4 if kind != "both" else 0.3, 5, IMG,
                       highlight={0, 2, 4})
        if kind in ("wave", "both"):
            waveform(ax, 9.75, y + (0.2 if kind == "both" else 0.35), 1.2, 0.45 if kind != "both" else 0.35, AUDIO,
                     seed=num)
        arrow(ax, 11.05, y + 0.62, 11.35, y + 0.62, color=c, lw=1.3, ms=10)
        vector(ax, 11.4, y + 0.5, c, n=7, cell=0.17, seed=num)
        label(ax, 12.6, y + 0.22, "finds: " + finds, size=8, color=c, ha="right")

    label(ax, 6.5, 0.6, "Image datasets get one index type: the image. A video dataset can keep all three "
          "side by side, and one search can query them together.", size=9.5, color=MUTED)
    ax.set_ylim(0.35, 6.85)
    save(fig, "02_index_modalities.png")


# ----------------------------------------------------------------------------------------------
# 03 what happens when you search
# ----------------------------------------------------------------------------------------------
def fig_search_pipeline():
    fig, ax = canvas(15, 6.0)
    label(ax, 7.5, 5.7, "What happens when you click Search", size=13, weight="bold")

    queries = [("Text", "“rain on a window”", TEXT_C, 4.3), ("Samples", "selected in the grid", IMG, 3.2),
               ("Media", "upload · webcam · mic", AUDIO, 2.1)]
    for name, sub, c, y in queries:
        box(ax, 0.2, y - 0.4, 2.3, 0.8, c, fc=BG, lw=1.5)
        label(ax, 1.35, y + 0.13, name, size=10.5, weight="bold", color=c)
        label(ax, 1.35, y - 0.17, sub, size=8.2, color=MUTED)
        arrow(ax, 2.55, y, 3.15, 3.2, color=c, lw=1.4)

    steps = [
        (3.2, "1", "Embed", ["*text", "with the SearchQuery prompt", "", "*media", "decoded with each index's",
                             "frame and audio settings", "", "*samples", "reuse their stored vectors"]),
        (6.0, "2", "Fit to the index", ["cut to the index's", "dimensions (768, 512,", "256 or 128), then", "rescaled to length 1",
                                        "", "several samples:", "averaged first"]),
        (8.8, "3", "LanceDB search", ["one table per index", "", "cosine similarity", "to every vector", "",
                                      "top k per index"]),
    ]
    for x, num, title, lines in steps:
        box(ax, x, 1.0, 2.5, 4.3, INK, fc=PANEL, lw=1.3)
        ax.add_patch(Circle((x + 0.35, 4.95), 0.22, fc=INK, zorder=4))
        label(ax, x + 0.35, 4.95, num, size=10, weight="bold", color=BG, z=6)
        label(ax, x + 0.7, 4.95, title, size=10.5, weight="bold", ha="left")
        for i, line in enumerate(lines):
            head = line.startswith("*")
            label(ax, x + 1.25, 4.4 - i * 0.36, line.lstrip("*"), size=8.6 if not head else 8.8,
                  weight="bold" if head else "normal")
        if x < 8.8:
            arrow(ax, x + 2.55, 3.2, x + 2.75, 3.2, color=INK, lw=1.6)

    # the math, small and concrete
    label(ax, 10.05, 1.45, "score = cos(q, v) = q · v", size=8.8, family=MONO, color=LANCE)

    arrow(ax, 11.35, 3.2, 11.85, 3.2, color=FO, lw=2.0, ms=16)
    box(ax, 11.9, 1.0, 2.9, 4.3, FO, fc=FO_FILL, lw=1.6)
    label(ax, 13.35, 4.95, "4  Results", size=10.5, weight="bold", color=FO_DARK)
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((12.2 + j * 0.8, 3.6 - i * 0.75), 0.7, 0.6, fc=IMG,
                                   alpha=0.85 - 0.08 * (i * 3 + j), ec=BG, zorder=3))
            label(ax, 12.25 + j * 0.8, 4.12 - i * 0.75, "#%d" % (i * 3 + j + 1), size=7, color="white", ha="left", z=6)
    label(ax, 13.35, 1.5, "loaded into the grid,", size=8.6)
    label(ax, 13.35, 1.2, "best match first", size=8.6)

    label(ax, 7.5, 0.45, "Uploads and recordings go through exactly the preprocessing the index's samples did, "
          "so query and dataset vectors are comparable.", size=9.5, color=MUTED)
    save(fig, "03_search_pipeline.png")


# ----------------------------------------------------------------------------------------------
# 04 LanceDB + FiftyOne
# ----------------------------------------------------------------------------------------------
def fig_lancedb():
    fig, ax = canvas(14, 6.2)
    label(ax, 7.0, 5.9, "Every index is an ordinary FiftyOne similarity index on the LanceDB backend",
          size=13, weight="bold")

    # dataset
    box(ax, 0.3, 1.2, 3.9, 4.0, FO, fc=FO_FILL, lw=1.6)
    label(ax, 2.25, 4.85, "FiftyOne dataset", size=11, weight="bold", color=FO_DARK)
    for i, (f, c) in enumerate([("eg2_video", IMG), ("eg2_audio", AUDIO), ("eg2_av", AV)]):
        y = 4.1 - i * 0.75
        label(ax, 0.55, y, "sample." + f, size=8.8, family=MONO, ha="left")
        vector(ax, 2.75, y - 0.13, c, n=7, cell=0.17, seed=i)
    label(ax, 2.25, 1.75, "vectors are also kept on", size=8.6, color=MUTED)
    label(ax, 2.25, 1.45, "the samples themselves", size=8.6, color=MUTED)

    # brain run
    box(ax, 5.0, 3.3, 3.8, 1.9, INK, fc=PANEL, lw=1.3)
    label(ax, 6.9, 4.85, "brain run (one per index)", size=10, weight="bold")
    label(ax, 6.9, 4.4, "compute_similarity(", size=8.4, family=MONO)
    label(ax, 6.9, 4.1, "  backend=\"lancedb\",", size=8.4, family=MONO)
    label(ax, 6.9, 3.8, "  model=\"google/embeddinggemma-2\",", size=8.0, family=MONO)
    label(ax, 6.9, 3.5, "  metric=\"cosine\")", size=8.4, family=MONO)
    arrow(ax, 4.25, 4.25, 4.95, 4.25, color=INK)

    # lancedb
    box(ax, 5.0, 0.6, 3.8, 2.3, LANCE, fc=LANCE_FILL, lw=1.5)
    label(ax, 6.9, 2.55, "LanceDB (embedded, no server)", size=10, weight="bold", color=LANCE)
    for i, (t, c) in enumerate([("eg2_video", IMG), ("eg2_audio", AUDIO), ("eg2_av", AV)]):
        y = 2.0 - i * 0.45
        ax.add_patch(Rectangle((5.35, y - 0.15), 0.3, 0.3, fc=c, ec="none", zorder=3))
        label(ax, 5.8, y, "table: " + t, size=8.6, family=MONO, ha="left")
    label(ax, 6.9, 0.8, "Lance files on disk or cloud storage", size=8.2, color=MUTED)
    arrow(ax, 6.9, 3.25, 6.9, 2.95, color=LANCE, lw=1.6)
    arrow(ax, 4.25, 1.6, 4.95, 1.6, color=MUTED, ls="--")
    label(ax, 4.6, 1.3, "Rebuild table", size=8, color=MUTED)

    # consumers
    consumers = [("this panel", "text, samples, media, webcam, mic", FO, 4.6),
                 ("the App's similarity search", "sort by similarity from the grid", IMG, 3.2),
                 ("Python", "view.sort_by_similarity(\"a dog barking\",\n  brain_key=\"eg2_av\")", AV, 1.6)]
    for name, sub, c, y in consumers:
        box(ax, 9.8, y - 0.55, 3.9, 1.1, c, fc=BG, lw=1.4)
        label(ax, 11.75, y + 0.22, name, size=10, weight="bold", color=c)
        label(ax, 11.75, y - 0.18, sub, size=7.8 if "\n" in sub else 8.4, color=MUTED,
              family=MONO if "\n" in sub else None)
        arrow(ax, 8.85, 1.75, 9.75, y, color=c, lw=1.3, conn="arc3,rad=0.1")

    label(ax, 7.0, 0.15, "Build once, search from anywhere. If the LanceDB table is lost, it is rebuilt "
          "from the sample field without re-embedding.", size=9.5, color=MUTED)
    save(fig, "04_lancedb_fiftyone.png")


# ----------------------------------------------------------------------------------------------
# 05 measured text -> clip retrieval
# ----------------------------------------------------------------------------------------------
def fig_retrieval_quality(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.suptitle("Text search on %d VGGSound clips: each of the %d class names as a query"
                 % (len(d["labels"]), len(d["classes"])), fontsize=13, weight="bold", color=INK, y=1.02)

    ks = [1, 5, 10]
    width = 0.26
    for j, f in enumerate(INDEXES):
        ranks = first_hit_ranks(d, f)
        vals = [(ranks <= k).mean() * 100 for k in ks]
        xs = np.arange(len(ks)) + (j - 1) * width
        a1.bar(xs, vals, width * 0.92, color=INDEX_COLOR[f], label=INDEX_LABEL[f])
        for x, v in zip(xs, vals):
            a1.text(x, v + 1.5, "%d%%" % round(v), ha="center", fontsize=8, color=INK)
    # chance that k random clips include one of the class, averaged over classes
    chance = [np.mean([100 * (1 - np.prod([1 - (d["labels"] == c).sum() / (len(d["labels"]) - i) for i in range(k)]))
                       for c in d["classes"]]) for k in ks]
    a1.plot(np.arange(len(ks)), chance, color=MUTED, ls="--", marker="o", ms=4, lw=1.2, label="random ranking")
    a1.set_xticks(range(len(ks)), ["right class at #1", "in the top 5", "in the top 10"])
    a1.set_ylim(0, 110)
    a1.set_ylabel("% of queries", color=MUTED, fontsize=9)
    a1.legend(fontsize=8.5, frameon=False, loc="upper left")
    style_axes(a1, "how often a clip of the queried class comes back")

    f = "eg2_av"
    sims = d["text"] @ d[f].T
    match, other = [], []
    for i, c in enumerate(d["classes"]):
        m = d["labels"] == c
        match.extend(sims[i][m])
        other.extend(sims[i][~m])
    bins = np.linspace(0.4, 0.85, 40)
    a2.hist(other, bins=bins, color=LINE, density=True, label="clips of other classes")
    a2.hist(match, bins=bins, color=AV, alpha=0.85, density=True, label="clips of the queried class")
    a2.set_xlabel("cosine similarity (text query vs clip, video + audio index)", color=MUTED, fontsize=9)
    a2.set_yticks([])
    a2.legend(fontsize=8.5, frameon=False)
    style_axes(a2, "scores: matches sit higher, but in a narrow band")

    fig.text(0.5, -0.06, "Text-to-clip scores mostly fall between 0.5 and 0.8, so read them as a ranking, not a "
             "threshold. Combining frames and sound gives the best top-1.", ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "05_retrieval_quality.png")


# ----------------------------------------------------------------------------------------------
# 06 one text query across three indexes
# ----------------------------------------------------------------------------------------------
def fig_example_search(d, query="church bell ringing", k=8):
    i = list(d["classes"]).index(query)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharex=True)
    fig.suptitle("One text query, three indexes:  “%s”" % query, fontsize=13, weight="bold", color=INK, y=1.03)
    for ax, f in zip(axes, INDEXES):
        sims = d[f] @ d["text"][i]
        order = np.argsort(-sims)[:k]
        hit = d["labels"][order] == query
        colors = [INDEX_COLOR[f] if h else LINE for h in hit]
        ax.barh(np.arange(k), sims[order], color=colors, height=0.7)
        for r, idx in enumerate(order):
            ax.text(sims[order][r] - 0.005, r, d["labels"][idx], ha="right", va="center", fontsize=8,
                    color="white" if hit[r] else INK)
        ax.set_yticks(range(k), ["#%d" % (r + 1) for r in range(k)])
        ax.invert_yaxis()
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel("cosine similarity", color=MUTED, fontsize=8.5)
        style_axes(ax, "%s index  (%d of top %d match)" % (INDEX_LABEL[f], hit.sum(), k))
    fig.text(0.5, -0.05, "Colored bars are clips labeled “%s”. Each index ranks the same clips by a different part "
             "of them." % query, ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "06_example_search.png")


# ----------------------------------------------------------------------------------------------
# 07 sound searches pictures
# ----------------------------------------------------------------------------------------------
def fig_sound_to_frames(d, k=6):
    """Query the video (frames only) index with each clip's audio vector."""
    audio, video, labels = d["eg2_audio"], d["eg2_video"], d["labels"]
    sims = audio @ video.T
    np.fill_diagonal(sims, -1)  # leave out the clip itself
    hit_at = [(labels[np.argsort(-sims[q])[:5]] == labels[q]).any() for q in range(len(labels))
              if (labels == labels[q]).sum() > 1]

    # pick the query whose sound finds the most same-class frames
    candidates = [q for q in range(len(labels)) if (labels == labels[q]).sum() >= 3]
    q = max(candidates, key=lambda q: (labels[np.argsort(-sims[q])[:k]] == labels[q]).sum())
    order = np.argsort(-sims[q])[:k]
    hit = labels[order] == labels[q]

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(13, 4.0), gridspec_kw={"width_ratios": [0.8, 1.4]})
    fig.suptitle("Sound searches pictures: a clip's audio vector against the frames-only video index",
                 fontsize=13, weight="bold", color=INK, y=1.03)
    a0.axis("off")
    a0.set_xlim(0, 4)
    a0.set_ylim(0, 3)
    waveform(a0, 0.3, 1.4, 3.4, 1.0, AUDIO, seed=4, lw=1.4)
    a0.text(2.0, 1.1, "query: the soundtrack of a\n“%s” clip" % labels[q], ha="center", va="top", fontsize=10, color=INK)
    a0.text(2.0, 0.25, "searched against video vectors,\nwhich contain no sound at all", ha="center",
            fontsize=8.5, color=MUTED)

    a1.barh(np.arange(k), sims[q][order], color=[IMG if h else LINE for h in hit], height=0.7)
    for r, idx in enumerate(order):
        a1.text(sims[q][order][r] - 0.005, r, labels[idx], ha="right", va="center", fontsize=8.5,
                color="white" if hit[r] else INK)
    a1.set_yticks(range(k), ["#%d" % (r + 1) for r in range(k)])
    a1.invert_yaxis()
    a1.set_xlim(max(0, sims[q][order].min() - 0.15), sims[q][order].max() + 0.02)
    a1.set_xlabel("cosine similarity (audio query vs video frames)", color=MUTED, fontsize=8.5)
    style_axes(a1, "top %d videos by what they show" % k)

    fig.text(0.5, -0.06, "Across all clips with a same-class partner, the audio-only query finds one by its frames "
             "in the top 5 for %d%% of clips." % round(100 * np.mean(hit_at)),
             ha="center", fontsize=9.5, color=MUTED)
    fig.tight_layout()
    save(fig, "07_sound_to_frames.png")


if __name__ == "__main__":
    data = load_data()
    for theme in THEMES:
        use_theme(theme)
        fig_shared_space()
        fig_index_modalities()
        fig_search_pipeline()
        fig_lancedb()
        fig_retrieval_quality(data)
        fig_example_search(data)
        fig_sound_to_frames(data)
