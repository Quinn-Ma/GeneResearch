"""
Week 2 · Task 2 — Visualization
================================
Inputs:
  ../../week2/task1_dge/DEG_full_results.tsv          (all contrasts, long format)
  ../../week1/task3_confounder_correction/Genelevel_VST_corrected.tsv
  ../../week1/task2_sample_qc/MetaSheet_QC.csv

Outputs:
  volcano_overview.png         — 3-panel volcano (MGS2/3/4 vs MGS1)
  volcano_MGS4_vs_MGS1.png     — publication-quality single volcano
  heatmap_top50_DEGs.png       — heatmap of top 50 DEGs (VST Z-score)
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

# ── 0. Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OUT_DIR    = SCRIPT_DIR

DEG_PATH  = os.path.join(ROOT, "week2/task1_dge/DEG_full_results.tsv")
VST_PATH  = os.path.join(ROOT, "week1/task3_confounder_correction/Genelevel_VST_corrected.tsv")
META_PATH = os.path.join(ROOT, "week1/task2_sample_qc/MetaSheet_QC.csv")

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("-- 1. Loading data --")
full_deg = pd.read_csv(DEG_PATH, sep="\t")
vst      = pd.read_csv(VST_PATH, sep="\t", index_col=0)   # genes x samples
meta     = pd.read_csv(META_PATH, index_col=0, encoding="latin-1")
meta     = meta.set_index("r_id")
meta.index = meta.index.astype(str)

print(f"  DEG table  : {len(full_deg):,} rows, {full_deg['comparison'].nunique()} contrasts")
print(f"  VST matrix : {vst.shape[0]:,} genes x {vst.shape[1]:,} samples")
print(f"  Metadata   : {len(meta):,} samples")

# Palette — shared across all plots
MGS_COLORS  = {"1": "#4e79a7", "2": "#f28e2b", "3": "#59a14f", "4": "#e15759"}
UP_COLOR    = "#d62728"   # red — upregulated
DOWN_COLOR  = "#1f77b4"   # blue — downregulated
NS_COLOR    = "#bbbbbb"   # gray — not significant

PADJ_THRESH = 0.05
FC_THRESH   = 1.0         # shown as reference lines only; not used for coloring

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A  — VOLCANO PLOTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 2. Drawing volcano plots --")

def _prepare_volcano(df, padj_thresh=0.05, max_neg_log10=50):
    """Clean one contrast's DEG table for volcano plotting."""
    d = df.copy()
    # Fill NaN padj with 1 (failed independent filtering = not significant)
    d["padj"] = d["padj"].fillna(1.0)
    d["neg_log10_padj"] = -np.log10(d["padj"].clip(lower=1e-300))
    d["neg_log10_padj"] = d["neg_log10_padj"].clip(upper=max_neg_log10)
    # Significance label
    d["color"] = NS_COLOR
    d.loc[(d["padj"] < padj_thresh) & (d["log2FoldChange"] > 0), "color"] = UP_COLOR
    d.loc[(d["padj"] < padj_thresh) & (d["log2FoldChange"] < 0), "color"] = DOWN_COLOR
    return d


def _add_volcano_labels(ax, df, n_labels=15, padj_thresh=0.05):
    """
    Annotate top n_labels genes (by padj) on a volcano axes.
    Labels are placed with staggered y-offsets to reduce overlap.
    """
    sig = df[df["padj"] < padj_thresh].copy()
    if sig.empty:
        return
    top = sig.nsmallest(n_labels, "padj").copy()
    top = top.sort_values("neg_log10_padj", ascending=False)

    # Assign label x-side
    top["label_side"] = np.where(top["log2FoldChange"] >= 0, "right", "left")

    # Compute staggered y positions (avoid label pile-up at same y)
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    label_height = y_range * 0.065   # spacing between stacked labels

    # Split by side
    for side in ("right", "left"):
        subset = top[top["label_side"] == side].reset_index(drop=True)
        if subset.empty:
            continue
        ha = "left" if side == "right" else "right"
        x_offset = x_range * 0.04 if side == "right" else -x_range * 0.04

        # Assign y-slots to avoid dense stacking
        prev_text_y = None
        for i, row in subset.iterrows():
            text_x = row["log2FoldChange"] + x_offset
            text_y = row["neg_log10_padj"]
            # Push up if too close to previous label
            if prev_text_y is not None and abs(text_y - prev_text_y) < label_height:
                text_y = prev_text_y + label_height
            label = row["gene_symbol"] if row["gene_symbol"] else row["ensembl_id"]
            ax.annotate(
                label,
                xy=(row["log2FoldChange"], row["neg_log10_padj"]),
                xytext=(text_x, text_y),
                fontsize=6.5,
                ha=ha,
                va="center",
                color="#222222",
                arrowprops=dict(
                    arrowstyle="-",
                    color="#888888",
                    lw=0.6,
                    shrinkA=0,
                    shrinkB=2,
                ),
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    fc="white",
                    ec="none",
                    alpha=0.75,
                ),
            )
            prev_text_y = text_y


def draw_single_volcano(ax, df, title, padj_thresh=0.05, n_labels=15):
    """Draw one volcano plot on the given axes."""
    d = _prepare_volcano(df, padj_thresh=padj_thresh)

    n_up   = ((d["padj"] < padj_thresh) & (d["log2FoldChange"] > 0)).sum()
    n_down = ((d["padj"] < padj_thresh) & (d["log2FoldChange"] < 0)).sum()

    # Background (NS) first, then colored on top
    ns   = d[d["color"] == NS_COLOR]
    sig  = d[d["color"] != NS_COLOR]

    ax.scatter(ns["log2FoldChange"],  ns["neg_log10_padj"],
               c=NS_COLOR, s=6,  alpha=0.4, linewidths=0, rasterized=True)
    ax.scatter(sig["log2FoldChange"], sig["neg_log10_padj"],
               c=sig["color"],   s=14, alpha=0.9, linewidths=0, rasterized=True)

    # Threshold lines
    ax.axhline(-np.log10(padj_thresh), color="#555555", lw=0.8,
               linestyle="--", label=f"padj = {padj_thresh}")
    ax.axvline(0, color="#999999", lw=0.5, linestyle="-")

    # Axis labels & title
    ax.set_xlabel("log\u2082 Fold Change", fontsize=9)
    ax.set_ylabel("-log\u2081\u2080(padj)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Gene count annotation
    y_max = d["neg_log10_padj"].max() * 1.05
    ax.set_ylim(bottom=-0.3, top=y_max + y_max * 0.20)
    ax.set_xlim(
        left=d["log2FoldChange"].min() - 0.15,
        right=d["log2FoldChange"].max() + 0.15,
    )

    up_label   = f"Up: {n_up}"
    down_label = f"Down: {n_down}"
    ax.text(0.97, 0.96, up_label,   transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color=UP_COLOR,   fontweight="bold")
    ax.text(0.03, 0.96, down_label, transform=ax.transAxes, ha="left",
            va="top", fontsize=8, color=DOWN_COLOR, fontweight="bold")

    # Labels
    _add_volcano_labels(ax, d, n_labels=n_labels, padj_thresh=padj_thresh)

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)


# ── A1. 3-panel overview (all contrasts) ──────────────────────────────────────
CONTRAST_ORDER = ["MGS2_vs_MGS1", "MGS3_vs_MGS1", "MGS4_vs_MGS1"]
CONTRAST_TITLE = {
    "MGS4_vs_MGS1": "Advanced AMD\n(MGS4 vs MGS1)",
    "MGS3_vs_MGS1": "Moderate AMD\n(MGS3 vs MGS1)",
    "MGS2_vs_MGS1": "Early AMD\n(MGS2 vs MGS1)",
}

fig_ov, axes_ov = plt.subplots(1, 3, figsize=(18, 6))
fig_ov.suptitle(
    "Differential Gene Expression — AMD Progression vs Healthy Control\n"
    "Design: ~ RIN + PMI + Age + Sex + MGS_level   |   Threshold: padj < 0.05",
    fontsize=11, y=1.01,
)

for ax, contrast in zip(axes_ov, CONTRAST_ORDER):
    sub = full_deg[full_deg["comparison"] == contrast].copy()
    draw_single_volcano(ax, sub, CONTRAST_TITLE[contrast])

plt.tight_layout()
ov_path = os.path.join(OUT_DIR, "volcano_overview.png")
fig_ov.savefig(ov_path, dpi=200, bbox_inches="tight")
plt.close(fig_ov)
print(f"  Saved -> volcano_overview.png")


# ── A2. Publication-quality single volcano: MGS4 vs MGS1 ─────────────────────
sub4 = full_deg[full_deg["comparison"] == "MGS4_vs_MGS1"].copy()

fig_v4, ax_v4 = plt.subplots(figsize=(8, 7))
draw_single_volcano(
    ax_v4, sub4,
    title="Advanced AMD vs Healthy Control\n(MGS4 vs MGS1)",
    padj_thresh=0.05,
    n_labels=20,
)
# Custom legend
legend_handles = [
    mpatches.Patch(color=UP_COLOR,   label=f"Upregulated (padj < 0.05)"),
    mpatches.Patch(color=DOWN_COLOR, label=f"Downregulated (padj < 0.05)"),
    mpatches.Patch(color=NS_COLOR,   label="Not significant"),
]
ax_v4.legend(handles=legend_handles, fontsize=8, loc="lower right",
             framealpha=0.8, edgecolor="none")

plt.tight_layout()
v4_path = os.path.join(OUT_DIR, "volcano_MGS4_vs_MGS1.png")
fig_v4.savefig(v4_path, dpi=300, bbox_inches="tight")
plt.close(fig_v4)
print(f"  Saved -> volcano_MGS4_vs_MGS1.png")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION B  — HEATMAP  (Top 50 DEGs · VST Z-score)
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 3. Drawing heatmap --")

# ── B1. Select top 50 feature genes ──────────────────────────────────────────
# Primary pool: all 40 significant DEGs from MGS4 vs MGS1 (sorted by padj)
deg4 = full_deg[
    (full_deg["comparison"] == "MGS4_vs_MGS1") & (full_deg["significant"] == True)
].sort_values("padj").copy()

# Secondary pool: top significant DEGs from MGS2 vs MGS1 not already in deg4
deg2 = full_deg[
    (full_deg["comparison"] == "MGS2_vs_MGS1") & (full_deg["significant"] == True)
].sort_values("padj").copy()

n_primary   = len(deg4)         # should be 40
n_secondary = max(0, 50 - n_primary)

secondary_ids = [
    eid for eid in deg2["ensembl_id"].tolist()
    if eid not in set(deg4["ensembl_id"])
][:n_secondary]

feature_ids    = list(deg4["ensembl_id"]) + secondary_ids
feature_labels = []
id2sym = dict(zip(full_deg["ensembl_id"], full_deg["gene_symbol"]))
for eid in feature_ids:
    sym = id2sym.get(eid, "")
    feature_labels.append(sym if sym else eid)

print(f"  Feature genes: {n_primary} from MGS4 + {len(secondary_ids)} from MGS2 = {len(feature_ids)} total")

# ── B2. Extract & Z-score VST values ─────────────────────────────────────────
avail_ids = [eid for eid in feature_ids if eid in vst.index]
print(f"  Genes found in VST matrix: {len(avail_ids)}")

vst_sub = vst.loc[avail_ids].copy()       # genes x samples
gene_labels_ordered = [feature_labels[feature_ids.index(g)] for g in avail_ids]

# Z-score each gene (row-wise)
means = vst_sub.mean(axis=1)
stds  = vst_sub.std(axis=1)
stds  = stds.replace(0, 1)   # avoid division by zero
z_mat = vst_sub.subtract(means, axis=0).divide(stds, axis=0)
z_mat = z_mat.clip(-3, 3)    # clip outliers for visual clarity

# ── B3. Sort samples: MGS1 -> MGS2 -> MGS3 -> MGS4 ──────────────────────────
meta_sub = meta.loc[meta.index.isin(z_mat.columns)].copy()
meta_sub["mgs_level"] = meta_sub["mgs_level"].astype(int)
sample_order = meta_sub.sort_values("mgs_level").index.tolist()

# Keep only samples present in both
sample_order = [s for s in sample_order if s in z_mat.columns]
z_sorted = z_mat[sample_order]

# ── B4. Column color annotation bar (MGS level) ───────────────────────────────
col_colors = pd.Series(
    [MGS_COLORS[str(meta_sub.loc[s, "mgs_level"])] for s in sample_order],
    index=sample_order,
    name="MGS Level",
)

# ── B5. Draw clustermap ───────────────────────────────────────────────────────
# Row Z-score already applied; cluster genes (rows) but NOT samples (cols)
g = sns.clustermap(
    z_sorted,
    col_cluster   = False,        # preserve MGS-ordered sample layout
    row_cluster   = True,         # cluster genes by expression pattern
    col_colors    = col_colors,
    cmap          = "RdBu_r",
    vmin          = -3,
    vmax          =  3,
    linewidths    = 0,
    xticklabels   = False,
    yticklabels   = gene_labels_ordered,
    figsize       = (16, 12),
    dendrogram_ratio = (0.15, 0.04),
    colors_ratio  = 0.015,
    cbar_pos      = (0.02, 0.82, 0.03, 0.15),
    cbar_kws      = {"label": "Z-score (VST)"},
)

# Style the heatmap
g.ax_heatmap.tick_params(axis="y", labelsize=7, length=0)
g.ax_heatmap.set_ylabel("")
g.ax_heatmap.set_xlabel("Samples (sorted by MGS level: 1 -> 4)", fontsize=9)

# Add vertical dividers between MGS groups
mgs_counts  = meta_sub.sort_values("mgs_level")["mgs_level"].value_counts().sort_index()
boundaries  = np.cumsum([mgs_counts.get(i, 0) for i in [1, 2, 3, 4]])[:-1]
for b in boundaries:
    g.ax_heatmap.axvline(b, color="white", lw=1.2, linestyle="-")

# Title
g.fig.suptitle(
    f"Top {len(avail_ids)} DEGs — VST Expression Z-score\n"
    f"Samples ordered: MGS1 (n={mgs_counts.get(1,0)}) | "
    f"MGS2 (n={mgs_counts.get(2,0)}) | "
    f"MGS3 (n={mgs_counts.get(3,0)}) | "
    f"MGS4 (n={mgs_counts.get(4,0)})",
    fontsize=10, y=1.01,
)

# MGS level legend (manual patches)
legend_patches = [
    mpatches.Patch(color=MGS_COLORS["1"], label="MGS1 — Healthy"),
    mpatches.Patch(color=MGS_COLORS["2"], label="MGS2 — Early AMD"),
    mpatches.Patch(color=MGS_COLORS["3"], label="MGS3 — Intermediate AMD"),
    mpatches.Patch(color=MGS_COLORS["4"], label="MGS4 — Advanced AMD"),
]
g.ax_col_dendrogram.legend(
    handles=legend_patches, loc="center", ncol=4,
    fontsize=8, frameon=False,
    bbox_to_anchor=(0.5, 1.3),
)

hm_path = os.path.join(OUT_DIR, "heatmap_top50_DEGs.png")
g.savefig(hm_path, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"  Saved -> heatmap_top50_DEGs.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Week 2 Task 2 — Visualization Complete ===")
print(f"  volcano_overview.png      — 3-panel (MGS2/3/4 vs MGS1)")
print(f"  volcano_MGS4_vs_MGS1.png  — publication-quality single volcano")
print(f"  heatmap_top50_DEGs.png    — top {len(avail_ids)} DEGs, VST Z-score")
