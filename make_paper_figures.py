"""Generate all DepthDiff paper figures (fig1-fig5).

Restored generator. Self-contained: reads each dataset's results CSVs,
results/figure_data.npz (per-cell Pearson + 50%-depth pred/target matrices for
the small datasets), and the processed npz (for gene names). Clustering / logFC
/ marker computations mirror evaluate.py exactly (KMeans k=8, random_state=SEED,
n_init=20; logFC & DEG use the largest cluster vs. the rest).

Typography is unified across all figures via the rcParams block below.

Outputs -> paper_figures/:
  fig1_workflow.png            schematic pipeline
  fig2_{pbmc3k,pbmc68k,pancreas}.png  per-dataset 4-panel composite over its 3
                               depths (A biological lollipop, B RMSE mean bar +
                               per-depth dots, C Pearson mean+/-SD, D matrix)
  fig3_density_{pbmc3k,pbmc68k,pancreas}.png  hexbin pred vs true, 6 methods,
                               at 25% / 50% / 75% depth respectively
  fig5_marker_heatmap.png      PBMC3K, per-cluster marker expression (50% depth)
  fig6_citeseq.png             CITE-seq break-then-restore RNA-protein recovery
fig4_logfc_scatter() is kept but not called (dropped from the paper).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from sklearn.cluster import KMeans

# Unified typography across all paper figures (font family + size hierarchy).
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,          # base / annotations
    "axes.titlesize": 13,     # panel titles
    "axes.labelsize": 12,     # axis labels
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.titleweight": "normal",
})
ANNOT_SIZE = 9               # in-bar / in-cell value labels (consistent everywhere)

ROOT = Path(__file__).resolve().parent
FIGDIR = ROOT / "paper_figures"
FIGDIR.mkdir(exist_ok=True)
SEED = 2026
DATASETS = ["PBMC3K", "PBMC68K", "PANCREAS"]
DEPTHS = [0.25, 0.5, 0.75]

PROCESSED = {
    "PBMC3K": ROOT / "PBMC3K" / "data" / "processed" / "pbmc3k_fixed_umi_pairs.npz",
    "PANCREAS": ROOT / "PANCREAS" / "data" / "processed" / "pancreas_fixed_umi_pairs.npz",
}

# (display name, source file, internal method key). MAGIC is the unsupervised
# imputation baseline kept in the main comparison; scVI/DCA/ALRA were dropped.
METHODS = [
    ("raw", "metrics.csv", "raw_low_depth"),
    ("kNN", "metrics_extra.csv", "knn_supervised"),
    ("cVAE", "metrics.csv", "cvae"),
    ("MLP", "metrics.csv", "mlp"),
    ("MAGIC", "metrics_magic.csv", "magic"),
    ("DepthDiff", "metrics.csv", "depthdiff"),
]
MAIN_METHODS = METHODS
COLORS = {
    "raw": "#8C8C8C", "kNN": "#4C72B0", "cVAE": "#DD8452",
    "MLP": "#8172B3", "MAGIC": "#17BECF", "DepthDiff": "#C44E52",
}
# biological-metric series for the lollipop: (column, label, marker, ls, color)
BIO = [
    ("deg_top100_overlap", "DEG", "o", "-", "#4C72B0"),
    ("logfc_spearman", "logFC-S", "s", "--", "#DD8452"),
    ("marker_spearman", "marker-S", "^", "-.", "#55A868"),
]
# canonical PBMC marker genes for fig5 (filtered to those present in the HVGs)
MARKERS = ["IL32", "LTB", "S100A9", "S100A8", "CD3D", "CCL5", "NKG7",
           "CD79A", "HLA-DRA", "TYROBP", "CST3", "AIF1", "GZMB"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def pearson_flat(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kmeans_labels(target):
    return KMeans(n_clusters=8, random_state=SEED, n_init=20).fit_predict(target)


def collect_aggregate():
    rows = []
    for ds in DATASETS:
        cache = {}
        for _, fname, _ in METHODS:
            p = ROOT / ds / "results" / fname
            if p.exists() and fname not in cache:
                cache[fname] = pd.read_csv(p)
        for name, fname, key in METHODS:
            df = cache.get(fname)
            if df is None:
                continue
            for d in DEPTHS:
                r = df[(df["method"] == key) & (df["depth"].round(2) == round(d, 2))]
                if r.empty:
                    continue
                r = r.iloc[0]
                rows.append({
                    "dataset": ds, "depth": d, "method": name,
                    "rmse": r["rmse"], "pearson": r["pearson"],
                    "deg_top100_overlap": r["deg_top100_overlap"],
                    "logfc_spearman": r["logfc_spearman"],
                    "marker_spearman": r["marker_spearman"],
                })
    return pd.DataFrame(rows)


def load_fig_npz(dataset):
    return np.load(ROOT / dataset / "results" / "figure_data.npz", allow_pickle=True)


# ----------------------------------------------------------------------------
# fig1: workflow schematic
# ----------------------------------------------------------------------------
def fig1_workflow():
    """4-panel architecture schematic (A data/degradation, B diffusion-style
    training, C single-step inference, D CITE-seq orthogonal validation)."""
    PASTEL = ["#F4C2C2", "#BFE3C0", "#BCD4F0", "#FCE5A0", "#F7C59F"]
    INK = "#2F3A45"
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis("off")

    # ---- helpers -----------------------------------------------------------
    def stack(cx, cy, w=1.4, h=1.0, label="", rows=4, cols=5, fs=9):
        for off in (0.16, 0.08):
            ax.add_patch(Rectangle((cx - w / 2 + off, cy - h / 2 + off), w, h,
                                   facecolor="white", edgecolor="#9aa6b0", lw=0.8, zorder=2))
        cw, ch = w / cols, h / rows
        for i in range(rows):
            for j in range(cols):
                ax.add_patch(Rectangle((cx - w / 2 + j * cw, cy - h / 2 + i * ch), cw, ch,
                                       facecolor=PASTEL[(i + j) % len(PASTEL)],
                                       edgecolor="white", lw=0.5, zorder=3))
        ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                               edgecolor="#5A6B7B", lw=1.0, zorder=4))
        if label:
            ax.text(cx, cy - h / 2 - 0.22, label, ha="center", va="top",
                    fontsize=fs, color=INK, zorder=5)

    def rbox(cx, cy, w, h, text, fc, ec, fs=9, tcol=INK, lw=1.4):
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor=fc, edgecolor=ec, lw=lw, zorder=4))
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tcol, zorder=5)

    def arrow(x1, y1, x2, y2, color="#5A6B7B", lw=2.0, ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls),
                    zorder=3)

    def op(cx, cy, sym, color):  # circle operator (+/-)
        ax.add_patch(Circle((cx, cy), 0.22, facecolor="white", edgecolor=color, lw=1.6, zorder=6))
        ax.text(cx, cy, sym, ha="center", va="center", fontsize=12, color=color,
                fontweight="bold", zorder=7)

    def gauss(cx, cy, w=0.7, h=0.4):
        xs = np.linspace(-2, 2, 40)
        ys = np.exp(-xs ** 2 / 1.2)
        ax.plot(cx + xs / 4 * w, cy - h / 2 + ys * h, color="#444", lw=1.1, zorder=5)

    def noiseblk(cx, cy, s=0.55):
        g = np.random.default_rng(0).random((6, 6))
        ax.imshow(g, cmap="Greys", extent=(cx - s, cx + s, cy - s, cy + s),
                  zorder=4, aspect="auto")
        ax.add_patch(Rectangle((cx - s, cy - s), 2 * s, 2 * s, fill=False,
                               edgecolor="#555", lw=1.0, zorder=5))

    def panel(x0, y0, x1, y1, color, label):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fill=False, edgecolor=color, lw=2.2,
                                    linestyle=(0, (7, 4)), zorder=1))
        ax.text(x0 + 0.18, y1 - 0.30, label, ha="left", va="top", fontsize=16,
                fontweight="bold", color=color, zorder=6)

    def denoiser(cx, cy, w=2.8, h=1.1):
        rbox(cx, cy + h * 0.18, w, h, "", "#EFE6D6", "#B79B6E")
        ax.text(cx, cy + h * 0.27, "Residual MLP denoiser", ha="center", va="center",
                fontsize=9, color=INK, zorder=6)
        sw, sh = w * 0.40, h * 0.34
        rbox(cx - w * 0.23, cy - h * 0.10, sw, sh, "MLP\n(hidden 1024)", "#9DC3E6", "none", fs=7.5)
        rbox(cx + w * 0.23, cy - h * 0.10, sw, sh, "t + depth\nembedding", "#FFD966", "none", fs=7.5)

    # ===================== Panel A: data & degradation =====================
    panel(0.3, 0.4, 4.15, 10.6, "#C0392B", "A")
    ax.text(2.2, 10.15, "Benchmark & low-depth simulation", ha="center", fontsize=10,
            color=INK, style="italic")
    stack(2.2, 9.2, label="High-depth scRNA-seq\ncounts (training)")
    arrow(2.2, 8.55, 2.2, 8.0)
    rbox(2.2, 7.55, 3.1, 0.8, "QC + HVG\n(train set only, 2000 genes)", "#E8EEF5", "#4C6F9C")
    arrow(2.2, 7.1, 2.2, 6.55)
    rbox(2.2, 5.95, 3.3, 1.05, "Fixed-UMI down-sampling\nd ∈ {25%, 50%, 75%}\n(gene capture + dropout)",
         "#3a3f44", "#222", fs=8.5, tcol="white")
    op(0.95, 6.5, "①", "#C0392B")
    arrow(2.2, 5.4, 2.2, 4.85)
    stack(2.2, 4.15, label="Paired (low-depth, high-depth)")
    ax.text(2.2, 3.1, "Residual training target\n$x_0 = x_{high}-x_{low}$",
            ha="center", va="center", fontsize=9, color=INK,
            bbox=dict(boxstyle="round,pad=0.3", fc="#FBEAD9", ec="#C97B40", lw=1.0))
    ax.text(2.2, 1.5, "train / val / test\nsplit by cells", ha="center", va="center",
            fontsize=8.5, color="#5A6B7B")

    # connectors A -> B and A -> C
    arrow(4.15, 5.0, 5.0, 6.9, color="#2C5FA8")          # to training
    arrow(4.15, 4.0, 5.0, 2.9, color="#E08E0B")          # to inference

    # ===================== Panel B: diffusion training =====================
    panel(4.45, 5.7, 15.7, 10.6, "#2C5FA8", "B")
    ax.text(10.0, 10.5, "Diffusion-style training", ha="center", fontsize=10,
            color=INK, style="italic")
    # forward process row
    arrow(5.4, 9.9, 9.2, 9.9, color="#2C5FA8", lw=2.2)
    ax.text(7.2, 10.12, "Forward Process (add noise to $x_0$)", ha="center",
            fontsize=8.5, color="#2C5FA8")
    stack(5.7, 9.1, w=1.1, h=0.85, label="", rows=4, cols=4)
    ax.text(8.0, 9.1, "$\\cdots$", ha="center", va="center", fontsize=14, color=INK)
    stack(9.0, 9.1, w=1.1, h=0.85, label="", rows=4, cols=4)
    ax.text(9.0, 8.5, "$x_t$ (step t)\ncosine, T=50", ha="center", va="top",
            fontsize=8, color=INK)
    gauss(6.5, 8.2)
    ax.text(6.5, 7.85, "$\\mathcal{N}(0,I)$", ha="center", fontsize=8, color=INK)
    noiseblk(7.5, 8.2, s=0.32)
    ax.text(7.5, 7.78, "Random Noise", ha="center", fontsize=7.5, color=INK)
    # denoiser block + residual loss (green inner box)
    ax.add_patch(FancyBboxPatch((8.0, 6.0), 6.6, 2.1,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                fill=False, edgecolor="#3C8A4E", lw=1.6,
                                linestyle=(0, (5, 3)), zorder=2))
    arrow(9.0, 8.6, 9.7, 7.6, color="#2C5FA8")           # x_t into denoiser
    denoiser(10.2, 7.0)
    arrow(6.2, 6.6, 8.7, 6.85, color="#5A6B7B")          # condition in
    ax.text(6.2, 6.35, "low-depth expr.\n(condition)", ha="center", va="top",
            fontsize=7.5, color=INK)
    arrow(13.2, 7.0, 12.0, 7.0, color="#5A6B7B")
    ax.text(13.5, 7.0, "Step t\ndepth d", ha="left", va="center", fontsize=8, color=INK)
    arrow(11.6, 7.0, 12.6, 7.0, color="#5A6B7B")
    op(13.0, 7.0, "−", "#C0392B")
    stack(14.0, 7.0, w=0.9, h=0.7, rows=3, cols=3)
    ax.text(13.0, 6.35, "denoising loss ($x_0$-prediction):  predicted $\\hat{x}_0$  vs  $x_0$",
            ha="center", va="top", fontsize=8, color="#3C8A4E")

    # ===================== Panel C: single-step inference ==================
    panel(4.45, 0.4, 10.55, 5.4, "#E08E0B", "C")
    ax.text(7.5, 5.05, "Single-step inference", ha="center", fontsize=10,
            color=INK, style="italic")
    noiseblk(5.4, 3.7, s=0.34)
    ax.text(5.4, 3.25, "$x_T\\sim\\mathcal{N}(0,I)$", ha="center", fontsize=8, color=INK)
    stack(5.4, 2.3, w=1.1, h=0.85, label="low-depth\nexpression")
    arrow(6.1, 3.6, 6.9, 3.0, color="#E08E0B")
    arrow(6.1, 2.3, 6.9, 2.7, color="#E08E0B")
    denoiser(8.0, 2.85)
    ax.text(8.0, 1.95, "depth d ,  t = T", ha="center", fontsize=8, color=INK)
    arrow(9.1, 2.85, 9.6, 2.85, color="#E08E0B")
    op(9.85, 2.85, "+", "#3C8A4E")
    ax.text(7.5, 0.78, "Single forward step (no reverse sampling)", ha="center",
            fontsize=9.5, color="#E08E0B", fontweight="bold")

    # output stack just inside the right edge of panel C
    stack(9.95, 4.4, w=1.0, h=0.8, rows=3, cols=4)
    ax.text(9.95, 3.88, "Enhanced\nhigh-depth expr.", ha="center", va="top",
            fontsize=8, color=INK)
    arrow(9.85, 3.12, 9.9, 3.95, color="#E08E0B")

    # ===================== Panel D: CITE-seq validation ====================
    panel(10.85, 0.4, 15.7, 5.4, "#3C8A4E", "D")
    ax.text(13.3, 5.05, "CITE-seq orthogonal validation", ha="center", fontsize=10,
            color=INK, style="italic")

    def rnaprot(cx, cy, tag):
        for k in range(4):
            ax.add_patch(Rectangle((cx + k * 0.16, cy), 0.15, 0.22,
                                   facecolor=PASTEL[k % len(PASTEL)], edgecolor="white", lw=0.4, zorder=4))
        for k in range(4):
            ax.add_patch(Rectangle((cx + 0.95 + k * 0.16, cy), 0.15, 0.22,
                                   facecolor="#7FB3D5", edgecolor="white", lw=0.4, zorder=4))
        ax.text(cx - 0.15, cy + 0.11, tag, ha="right", va="center", fontsize=8,
                fontweight="bold", color=INK)
        ax.text(cx + 0.32, cy + 0.32, "RNA", ha="center", fontsize=6.5, color=INK)
        ax.text(cx + 1.27, cy + 0.32, "protein", ha="center", fontsize=6.5, color=INK)

    rnaprot(11.6, 4.45, "a")
    rnaprot(11.6, 3.65, "b")
    rnaprot(11.6, 2.85, "c")
    arrow(12.0, 4.4, 12.0, 3.95, color="#C0392B")
    ax.text(12.15, 4.18, "① RNA degradation", ha="left", va="center", fontsize=7, color="#C0392B")
    arrow(12.0, 3.6, 12.0, 3.15, color="#3C8A4E")
    ax.text(12.15, 3.38, "② DepthDiff enhance", ha="left", va="center", fontsize=7, color="#3C8A4E")

    # mini bar chart (manual)
    vals = [0.278, 0.161, 0.297]
    cols_bar = ["#8C8C8C", "#DD8452", "#C44E52"]
    names = ["High", "Low", "Enh."]
    base, scale, bw = 0.7, 4.3, 0.45
    xb = [11.6, 12.5, 13.4]
    ax.plot([11.4, 14.0], [base, base], color="#888", lw=0.8, zorder=3)
    for x, v, c, nm in zip(xb, vals, cols_bar, names):
        ax.add_patch(Rectangle((x, base), bw, v * scale, facecolor=c, edgecolor="none", zorder=4))
        ax.text(x + bw / 2, base + v * scale + 0.05, f"{v:.2f}", ha="center", va="bottom",
                fontsize=7, color=INK)
        ax.text(x + bw / 2, base - 0.08, nm, ha="center", va="top", fontsize=7.5, color=INK)
    ax.text(14.6, base + 0.7, "Mean\nRNA–protein\nSpearman", ha="center", va="center",
            fontsize=7.5, color=INK)

    fig.savefig(FIGDIR / "fig1_workflow.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# fig2: per-dataset 4-panel performance composite
# ----------------------------------------------------------------------------
def composite(df, outpath, n_label):
    """4-panel composite for an arbitrary slice of the aggregate dataframe.

    n_label: text describing the conditions the box/SD spreads over
    (e.g. "3 depths" per dataset, or "9 conditions" for the pooled view).
    """
    names = [m[0] for m in MAIN_METHODS if m[0] in set(df["method"])]
    means = df.groupby("method").mean(numeric_only=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axA, axB = axes[0]
    axC, axD = axes[1]

    # A: lollipop
    order = list(reversed(names))
    offsets = [0.26, 0.0, -0.26]
    for i, name in enumerate(order):
        for (col, _lab, mk, ls, c), off in zip(BIO, offsets):
            v = means.loc[name, col]
            y = i + off
            axA.hlines(y, 0, v, color=c, linestyle=ls, linewidth=1.6, alpha=0.9, zorder=1)
            axA.plot(v, y, marker=mk, color=c, markersize=9,
                     markeredgecolor="white", markeredgewidth=0.8, zorder=2)
    axA.set_yticks(range(len(order))); axA.set_yticklabels(order)
    # small headroom above the top row so the one-row legend sits clear of data
    axA.set_ylim(-0.6, len(order) - 0.4 + 0.55); axA.set_xlim(0, 1.0)
    axA.set_xlabel("Score ↑")
    axA.set_title("A. Biological metrics", loc="left")
    axA.grid(axis="x", linestyle=":", alpha=0.5)
    handles = [plt.Line2D([0], [0], color=c, linestyle=ls, marker=mk, markersize=8,
                          markeredgecolor="white", label=lab)
               for col, lab, mk, ls, c in BIO]
    axA.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.4)

    # B: RMSE mean bar with per-depth points overlaid
    rmean = [df[df["method"] == n]["rmse"].mean() for n in names]
    bars = axB.bar(range(len(names)), rmean, color=[COLORS[n] for n in names], alpha=0.85)
    for i, n in enumerate(names):
        vals = df[df["method"] == n]["rmse"].values
        axB.scatter(np.full(len(vals), i), vals, color="0.25", s=16,
                    zorder=3, alpha=0.85, edgecolors="white", linewidths=0.4)
    for b, m in zip(bars, rmean):
        axB.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=ANNOT_SIZE)
    axB.set_xticks(range(len(names)))
    axB.set_xticklabels(names, rotation=30, ha="right")
    axB.set_ylabel("RMSE ↓")
    axB.set_title(f"B. RMSE (mean, {n_label} overlaid)", loc="left")
    axB.grid(axis="y", linestyle=":", alpha=0.5)

    # C: Pearson mean +/- SD
    pmean = [df[df["method"] == n]["pearson"].mean() for n in names]
    psd = [df[df["method"] == n]["pearson"].std() for n in names]
    bars = axC.bar(range(len(names)), pmean, yerr=psd, capsize=4,
                   color=[COLORS[n] for n in names], alpha=0.85,
                   error_kw=dict(ecolor="0.3", lw=1.2))
    for b, m in zip(bars, pmean):
        axC.text(b.get_x() + b.get_width() / 2, m + 0.02, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=ANNOT_SIZE)
    axC.set_xticks(range(len(names)))
    axC.set_xticklabels(names, rotation=30, ha="right")
    axC.set_ylabel("Pearson ↑"); axC.set_ylim(0, 1.05)
    axC.set_title("C. Reconstruction Pearson (mean ± SD)", loc="left")
    axC.grid(axis="y", linestyle=":", alpha=0.5)

    # D: comprehensive matrix
    cols = [("pearson", "Pearson"), ("deg_top100_overlap", "DEG"),
            ("logfc_spearman", "logFC-S"), ("marker_spearman", "marker-S")]
    mat = np.array([[means.loc[n, c] * 100 for c, _ in cols] for n in names])
    im = axD.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    axD.set_xticks(range(len(cols))); axD.set_xticklabels([lab for _, lab in cols])
    axD.set_yticks(range(len(names))); axD.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(cols)):
            axD.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center",
                     fontsize=ANNOT_SIZE, color="black")
    axD.set_title("D. Comprehensive performance matrix", loc="left")
    fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04).set_label("Score (×100)")

    fig.tight_layout()
    fig.savefig(outpath, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# fig3: pred-vs-true density (Pancreas, 50% depth)
# ----------------------------------------------------------------------------
def fig3_density(dataset, outname, pct=50):
    d = load_fig_npz(dataset)
    target = d[f"target_{pct}"]
    # all same-paradigm methods (npz key suffix, display label), matching the
    # composite figures' method set
    panels = [("raw", "Raw"), ("knn", "kNN"), ("cvae", "cVAE"), ("mlp", "MLP"),
              ("magic", "MAGIC"), ("depthdiff", "DepthDiff")]
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.7 * nrow),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    hb = None
    for ax, (key, label) in zip(axes, panels):
        pred = d[f"pred_{key}_{pct}"]
        r = pearson_flat(pred, target)
        hb = ax.hexbin(target.reshape(-1), pred.reshape(-1), gridsize=45,
                       bins="log", cmap="magma_r", mincnt=1)
        lim = [0, max(target.max(), pred.max())]
        ax.plot(lim, lim, "--", color="#1f77b4", lw=1.2)
        ax.set_title(f"{label}  (r={r:.3f})")
    for ax in axes[len(panels):]:  # hide unused cells
        ax.set_visible(False)
    # single shared axis labels instead of repeating them under every panel
    fig.supxlabel("True high-depth expression")
    fig.supylabel("Predicted expression")
    cbar = fig.colorbar(hb, ax=axes.tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("log10 count")
    fig.savefig(FIGDIR / outname, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# fig4: per-gene logFC true vs recovered (PBMC3K, 50% depth)
# ----------------------------------------------------------------------------
def fig4_logfc_scatter(dataset="PBMC3K"):
    d = load_fig_npz(dataset)
    target = d["target_50"]
    labels = kmeans_labels(target)
    group = labels == np.bincount(labels).argmax()
    true_logfc = target[group].mean(0) - target[~group].mean(0)
    panels = [("raw", "Raw"), ("mlp", "MLP"), ("depthdiff", "DepthDiff")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True, sharey=True)
    for ax, (key, label) in zip(axes, panels):
        pred = d[f"pred_{key}_50"]
        pred_logfc = pred[group].mean(0) - pred[~group].mean(0)
        r = pearson_flat(true_logfc, pred_logfc)
        ax.axhline(0, color="0.8", lw=0.8); ax.axvline(0, color="0.8", lw=0.8)
        ax.plot([-3, 3], [-3, 3], "--", color="black", lw=1)
        ax.scatter(true_logfc, pred_logfc, s=9, alpha=0.45,
                   color=COLORS["DepthDiff"] if key == "depthdiff"
                   else (COLORS["MLP"] if key == "mlp" else COLORS["raw"]),
                   edgecolors="none")
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xlabel("True high-depth gene logFC")
        ax.set_title(f"{label}  (r={r:.3f})")
    axes[0].set_ylabel("Recovered gene logFC")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig4_logfc_scatter.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# fig5: marker-gene expression heatmap per cluster (PBMC3K, 50% depth)
# ----------------------------------------------------------------------------
def fig5_marker_heatmap(dataset="PBMC3K"):
    d = load_fig_npz(dataset)
    target = d["target_50"]
    proc = np.load(PROCESSED[dataset], allow_pickle=True)
    gene_names = [str(g) for g in proc["gene_names"]]
    idx = {g: i for i, g in enumerate(gene_names)}
    markers = [m for m in MARKERS if m in idx]
    cols = [idx[m] for m in markers]
    labels = kmeans_labels(target)
    clusters = np.sort(np.unique(labels))

    panels = [(target, "High-depth"), (d["pred_raw_50"], "Raw"),
              (d["pred_mlp_50"], "MLP"), (d["pred_depthdiff_50"], "DepthDiff")]

    def heat(mat):
        return np.array([mat[labels == c][:, cols].mean(0) for c in clusters])

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    im = None
    for ax, (mat, label) in zip(axes, panels):
        im = ax.imshow(heat(mat), cmap="magma", vmin=0, vmax=7, aspect="auto")
        ax.set_xticks(range(len(markers)))
        ax.set_xticklabels(markers, rotation=90)
        ax.set_xlabel("Marker gene")
        ax.set_title(label)
    axes[0].set_yticks(range(len(clusters)))
    axes[0].set_yticklabels([f"C{c}" for c in clusters])
    axes[0].set_ylabel("Cell cluster")
    cbar = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    cbar.set_label("Mean log-norm expression")
    fig.savefig(FIGDIR / "fig5_marker_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# fig6: CITE-seq orthogonal validation (break-then-restore)
# ----------------------------------------------------------------------------
def fig6_citeseq():
    csv = ROOT / "PBMC3K" / "results" / "citeseq_validation.csv"
    df = pd.read_csv(csv).sort_values("depth")
    depths = [int(round(d * 100)) for d in df["depth"]]
    high = df["corr_high"].values
    low = df["corr_low"].values
    enh = df["corr_enhanced"].values
    rec = df["recovery_%"].values

    x = np.arange(len(depths))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8, 5))
    b_low = ax.bar(x - w, low, w, label="Low-depth RNA", color="#DD8452", alpha=0.9)
    b_enh = ax.bar(x, enh, w, label="DepthDiff enhanced", color="#C44E52", alpha=0.9)
    b_high = ax.bar(x + w, high, w, label="High-depth RNA", color="#8C8C8C", alpha=0.9)
    ax.axhline(high[0], ls="--", lw=1, color="0.4", zorder=0)

    for bars in (b_low, b_enh, b_high):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004,
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=ANNOT_SIZE)
    # recovery% above the enhanced bars
    for xi, e, r in zip(x, enh, rec):
        ax.text(xi, e + 0.022, f"recovery {r:.0f}%", ha="center", va="bottom",
                fontsize=ANNOT_SIZE, fontweight="bold", color="#C44E52")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d}%" for d in depths])
    ax.set_xlabel("RNA down-sampling depth")
    ax.set_ylabel("Mean RNA–protein Spearman (9 pairs)")
    ax.set_ylim(0, max(enh.max(), high.max()) + 0.07)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig6_citeseq.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    df = collect_aggregate()
    # NOTE: fig1 (paper_figures/fig1_workflow.png) is an externally drawn AI
    # figure placed by hand -- do NOT call fig1_workflow() here, or it would be
    # overwritten by the matplotlib version. The function is kept for reference;
    # uncomment the next line to regenerate the matplotlib schematic instead.
    # fig1_workflow()
    composite(df[df["dataset"] == "PBMC3K"], FIGDIR / "fig2_pbmc3k.png", "3 depths")
    composite(df[df["dataset"] == "PBMC68K"], FIGDIR / "fig2_pbmc68k.png", "3 depths")
    composite(df[df["dataset"] == "PANCREAS"], FIGDIR / "fig2_pancreas.png", "3 depths")
    fig3_density("PBMC3K", "fig3_density_pbmc3k.png", pct=25)
    fig3_density("PBMC68K", "fig3_density_pbmc68k.png", pct=50)
    fig3_density("PANCREAS", "fig3_density_pancreas.png", pct=75)
    fig5_marker_heatmap(dataset="PBMC3K")
    fig6_citeseq()
    print("all figures written to", FIGDIR)
    # fig4_logfc_scatter dropped from the paper (redundant with the logFC-S
    # column/series already shown in the per-dataset composites); the function
    # is kept below in case it is needed again.


if __name__ == "__main__":
    main()
