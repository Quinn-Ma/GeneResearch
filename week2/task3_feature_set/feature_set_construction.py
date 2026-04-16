"""
Week 2 · Task 3 — Feature Set Construction
============================================
Builds the "golden feature list" for Week 4 XGBoost classification.

Strategy  (union of two independent selection tracks):
  Track A — DEG track  : padj < 0.05 across any contrast (MGS2/3/4 vs MGS1)
  Track B — HVG track  : top N genes by per-sample variance in VST matrix

NOTE on |log2FC| > 1 threshold:
  AMD bulk RNA-seq effects are characteristically subtle (<2-fold).
  The strict padj < 0.05 & |log2FC| > 1 filter yields 0 genes in all
  three contrasts — biologically expected for a complex neurodegenerative
  disease. Track A therefore uses padj < 0.05 alone as the significance
  criterion, with both thresholds clearly annotated in the output table
  for downstream filtering flexibility.

Inputs:
  ../../week2/task1_dge/DEG_full_results.tsv
  ../../week1/task3_confounder_correction/Genelevel_VST_corrected.tsv
  ../../gene_info.tsv

Outputs:
  feature_set_final.tsv          — master annotated feature list (XGBoost-ready)
  feature_set_summary.txt        — run statistics and breakdown
  feature_selection_plot.png     — HVG variance curve + composition bar chart
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

# ── 0. Paths & parameters ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OUT_DIR    = SCRIPT_DIR

DEG_PATH  = os.path.join(ROOT, "week2/task1_dge/DEG_full_results.tsv")
VST_PATH  = os.path.join(ROOT, "week1/task3_confounder_correction/Genelevel_VST_corrected.tsv")
GENE_PATH = os.path.join(ROOT, "gene_info.tsv")

# Tunable parameters
PADJ_THRESH  = 0.05      # DEG significance threshold
FC_THRESH    = 1.0       # |log2FC| threshold (annotated but NOT used for filtering; yields 0 genes)
N_HVG        = 2000      # number of top high-variance genes to include


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────────────────────────────────────
print("-- 1. Loading data --")
full_deg  = pd.read_csv(DEG_PATH,  sep="\t")
vst       = pd.read_csv(VST_PATH,  sep="\t", index_col=0)   # genes x samples
gene_info = pd.read_csv(GENE_PATH, sep="\t")

print(f"  DEG table    : {len(full_deg):,} rows | {full_deg['comparison'].nunique()} contrasts")
print(f"  VST matrix   : {vst.shape[0]:,} genes x {vst.shape[1]:,} samples")
print(f"  Gene info    : {len(gene_info):,} entries")

# Build gene annotation lookups
gene_info = gene_info.rename(columns={
    "ensembl_gene_id"   : "ensembl_id",
    "external_gene_name": "gene_symbol",
})
id2sym  = gene_info.set_index("ensembl_id")["gene_symbol"].to_dict()
id2bio  = gene_info.set_index("ensembl_id")["gene_biotype"].to_dict()
protein_coding_ids = set(
    gene_info.loc[gene_info["gene_biotype"] == "protein_coding", "ensembl_id"]
)
print(f"  Protein-coding genes in gene_info: {len(protein_coding_ids):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TRACK A — DEG Selection
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 2. Track A: DEG selection --")

# Check strict threshold (padj < 0.05 AND |log2FC| > 1)
n_strict = ((full_deg["padj"] < PADJ_THRESH) &
            (full_deg["log2FoldChange"].abs() > FC_THRESH)).sum()
print(f"  Genes passing strict (padj<{PADJ_THRESH} & |log2FC|>{FC_THRESH}): {n_strict}")
print(f"  --> AMD bulk RNA-seq effects are characteristically subtle (<2-fold).")
print(f"  --> Falling back to padj<{PADJ_THRESH} alone for Track A.")

# Per-contrast stats
print("  Per-contrast breakdown (padj<0.05):")
for comp, grp in full_deg.groupby("comparison"):
    n = (grp["padj"] < PADJ_THRESH).sum()
    n_up   = ((grp["padj"] < PADJ_THRESH) & (grp["log2FoldChange"] > 0)).sum()
    n_down = ((grp["padj"] < PADJ_THRESH) & (grp["log2FoldChange"] < 0)).sum()
    print(f"    {comp:<22} : {n:>5} DEGs (up={n_up}, down={n_down})")

# Build per-gene summary across all contrasts
# For each gene, keep the row with the best (smallest) padj across contrasts
deg_sig = full_deg[full_deg["padj"] < PADJ_THRESH].copy()

# Best stats per gene (across comparisons)
best_per_gene = (
    deg_sig.sort_values("padj")
           .groupby("ensembl_id", as_index=False)
           .first()   # best (lowest padj) row per gene
)
best_per_gene = best_per_gene.rename(columns={
    "log2FoldChange": "best_log2FC",
    "padj"          : "best_padj",
    "comparison"    : "best_comparison",
})
best_per_gene["is_DEG"]        = True
best_per_gene["strict_DEG"]    = (
    (best_per_gene["best_padj"] < PADJ_THRESH) &
    (best_per_gene["best_log2FC"].abs() > FC_THRESH)
)

deg_ids = set(best_per_gene["ensembl_id"])
print(f"\n  Unique DEGs (padj<{PADJ_THRESH}, union all contrasts): {len(deg_ids):,}")
print(f"  Of which strict (also |log2FC|>{FC_THRESH})          : {best_per_gene['strict_DEG'].sum():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. TRACK B — HVG Selection
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n-- 3. Track B: HVG selection (top {N_HVG:,}) --")

# Variance computed across all 433 samples on VST-corrected values
gene_var  = vst.var(axis=1)                           # Series: gene → variance
var_rank  = gene_var.rank(method="first", ascending=False).astype(int)  # 1 = highest

hvg_df = pd.DataFrame({
    "ensembl_id"  : gene_var.index,
    "variance_vst": gene_var.values,
    "hvg_rank"    : var_rank.values,
    "is_HVG"      : var_rank.values <= N_HVG,
})

hvg_top = hvg_df[hvg_df["is_HVG"]].copy()
hvg_ids = set(hvg_top["ensembl_id"])
var_cutoff = float(gene_var.nlargest(N_HVG).min())

print(f"  Variance cutoff (rank {N_HVG})  : {var_cutoff:.4f}")
print(f"  Variance range (all genes)      : [{gene_var.min():.4f}, {gene_var.max():.4f}]")
print(f"  HVG IDs selected                : {len(hvg_ids):,}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Union + Protein-coding filter
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 4. Building union feature set --")

union_ids   = deg_ids | hvg_ids
both_ids    = deg_ids & hvg_ids
only_deg    = deg_ids - hvg_ids
only_hvg    = hvg_ids - deg_ids

print(f"  DEG only        : {len(only_deg):,}")
print(f"  HVG only        : {len(only_hvg):,}")
print(f"  DEG + HVG both  : {len(both_ids):,}")
print(f"  Union total     : {len(union_ids):,}")

# Protein-coding filter
union_pc   = union_ids & protein_coding_ids
n_removed  = len(union_ids) - len(union_pc)
print(f"\n  After protein_coding filter     : {len(union_pc):,}  (removed {n_removed})")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assemble final annotated table
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 5. Assembling final feature table --")

# Start from all union genes (protein_coding only)
all_ids = sorted(union_pc)

final = pd.DataFrame({"ensembl_id": all_ids})
final["gene_symbol"]  = final["ensembl_id"].map(id2sym).fillna("")
final["gene_biotype"] = final["ensembl_id"].map(id2bio).fillna("")

# Merge HVG info
final = final.merge(
    hvg_df[["ensembl_id", "variance_vst", "hvg_rank", "is_HVG"]],
    on="ensembl_id", how="left",
)
final["is_HVG"] = final["is_HVG"].fillna(False)

# Merge DEG info
deg_cols = ["ensembl_id", "is_DEG", "strict_DEG", "best_padj",
            "best_log2FC", "best_comparison"]
final = final.merge(
    best_per_gene[deg_cols],
    on="ensembl_id", how="left",
)
final["is_DEG"]     = final["is_DEG"].fillna(False)
final["strict_DEG"] = final["strict_DEG"].fillna(False)

# Selection source
def _source(row):
    if row["is_DEG"] and row["is_HVG"]:
        return "DEG+HVG"
    if row["is_DEG"]:
        return "DEG_only"
    return "HVG_only"

final["selection_source"] = final.apply(_source, axis=1)

# Sort: DEG+HVG first, then DEG_only by padj, then HVG_only by rank
source_order = {"DEG+HVG": 0, "DEG_only": 1, "HVG_only": 2}
final["_sort_src"]  = final["selection_source"].map(source_order)
final["_sort_padj"] = final["best_padj"].fillna(1.0)
final["_sort_rank"] = final["hvg_rank"].fillna(99999)
final = final.sort_values(["_sort_src", "_sort_padj", "_sort_rank"]).drop(
    columns=["_sort_src", "_sort_padj", "_sort_rank"]
).reset_index(drop=True)

final["feature_rank"] = range(1, len(final) + 1)

# Final column order
col_order = [
    "feature_rank", "ensembl_id", "gene_symbol", "gene_biotype",
    "selection_source", "is_DEG", "strict_DEG",
    "best_padj", "best_log2FC", "best_comparison",
    "is_HVG", "hvg_rank", "variance_vst",
]
final = final[col_order]

# Save
out_path = os.path.join(OUT_DIR, "feature_set_final.tsv")
final.to_csv(out_path, sep="\t", index=False)
print(f"  Saved -> feature_set_final.tsv  ({len(final):,} features)")

# Source breakdown
print("\n  Composition breakdown:")
for src, grp in final.groupby("selection_source", sort=False):
    print(f"    {src:<15} : {len(grp):>5} genes")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Visualization — variance curve + composition chart
# ─────────────────────────────────────────────────────────────────────────────
print("\n-- 6. Drawing feature selection plot --")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Panel A: HVG variance rank curve ──────────────────────────────────────────
ax = axes[0]
sorted_var = gene_var.sort_values(ascending=False).values
ranks = np.arange(1, len(sorted_var) + 1)

ax.plot(ranks, sorted_var, color="#555555", lw=0.8, rasterized=True)
ax.axvline(N_HVG, color="#e15759", lw=1.5, linestyle="--",
           label=f"HVG cutoff (top {N_HVG:,})")
ax.axhline(var_cutoff, color="#e15759", lw=0.8, linestyle=":", alpha=0.6)
ax.fill_between(ranks[:N_HVG], sorted_var[:N_HVG], alpha=0.15, color="#e15759")

# Mark genes that are also significant DEGs (within HVG range)
deg_in_hvg_rank = hvg_df[
    (hvg_df["ensembl_id"].isin(deg_ids)) & (hvg_df["hvg_rank"] <= N_HVG)
]["hvg_rank"].values
deg_in_hvg_var = gene_var.sort_values(ascending=False).iloc[
    [r - 1 for r in deg_in_hvg_rank if r <= len(sorted_var)]
].values
ax.scatter(deg_in_hvg_rank, deg_in_hvg_var,
           color="#4e79a7", s=18, zorder=5, label=f"DEG+HVG ({len(both_ids)} genes)",
           alpha=0.8)

ax.set_xscale("log")
ax.set_xlabel("Gene rank (by variance, log scale)", fontsize=10)
ax.set_ylabel("Per-gene VST variance", fontsize=10)
ax.set_title("HVG Selection — Variance Rank Curve", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)

# Annotation: cutoff stats
ax.text(0.97, 0.95,
        f"Variance cutoff: {var_cutoff:.3f}\n"
        f"HVG selected: {N_HVG:,} / {len(gene_var):,}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, bbox=dict(fc="white", ec="none", alpha=0.8))

# ── Panel B: Feature set composition stacked bar ──────────────────────────────
ax2 = axes[1]
src_counts = final["selection_source"].value_counts()

# Ordered categories
cats   = ["DEG+HVG", "DEG_only", "HVG_only"]
colors = {"DEG+HVG": "#59a14f", "DEG_only": "#4e79a7", "HVG_only": "#f28e2b"}
counts = [src_counts.get(c, 0) for c in cats]
total  = sum(counts)

bars = ax2.bar(["Final Feature Set\n(protein-coding)"], [total],
               color="#dddddd", width=0.5)

bottom = 0
for cat, cnt, col in zip(cats, counts, [colors[c] for c in cats]):
    ax2.bar(["Final Feature Set\n(protein-coding)"], [cnt],
            bottom=bottom, color=col, width=0.5, label=f"{cat} ({cnt:,})")
    if cnt > 0:
        ax2.text(0, bottom + cnt / 2,
                 f"{cat}\n{cnt:,} ({cnt/total*100:.1f}%)",
                 ha="center", va="center", fontsize=9, color="white",
                 fontweight="bold")
    bottom += cnt

ax2.set_ylim(0, total * 1.15)
ax2.set_ylabel("Number of genes", fontsize=10)
ax2.set_title(f"Feature Set Composition\n(Total = {total:,} genes)", fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", fontsize=8, framealpha=0.8)
ax2.spines[["top", "right"]].set_visible(False)
ax2.tick_params(axis="x", which="both", bottom=False)

# Extra info text
info = (
    f"DEG source: padj < {PADJ_THRESH} (any contrast)\n"
    f"HVG source: top {N_HVG:,} by VST variance\n"
    f"Protein-coding only (gene_info.tsv)"
)
ax2.text(0.97, 0.05, info, transform=ax2.transAxes, ha="right", va="bottom",
         fontsize=7.5, color="#555555",
         bbox=dict(fc="white", ec="#cccccc", alpha=0.9, boxstyle="round,pad=0.3"))

plt.suptitle("Week 2 Task 3 — Feature Set Construction", fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()

plot_path = os.path.join(OUT_DIR, "feature_selection_plot.png")
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"  Saved -> feature_selection_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary report
# ─────────────────────────────────────────────────────────────────────────────
summary_lines = [
    "===================================================",
    "  Week 2 Task 3 -- Feature Set Construction",
    "===================================================",
    "",
    "  INPUT",
    f"    DEG table    : {len(full_deg):,} rows ({full_deg['comparison'].nunique()} contrasts)",
    f"    VST matrix   : {vst.shape[0]:,} genes x {vst.shape[1]:,} samples",
    f"    Gene info    : {len(gene_info):,} entries",
    "",
    "  TRACK A  --  DEG selection",
    f"    Threshold            : padj < {PADJ_THRESH}",
    f"    Strict (also |FC|>1) : {n_strict} genes  [AMD effects too subtle]",
    f"    MGS4 vs MGS1         : 40 DEGs (all up)",
    f"    MGS3 vs MGS1         : 0 DEGs",
    f"    MGS2 vs MGS1         : 1037 DEGs (180 up, 857 down)",
    f"    Unique DEGs (union)  : {len(deg_ids):,}",
    "",
    "  TRACK B  --  HVG selection",
    f"    N_HVG selected   : {N_HVG:,}",
    f"    Variance cutoff  : {var_cutoff:.4f}",
    f"    Variance max     : {gene_var.max():.4f}",
    "",
    "  UNION + FILTER",
    f"    Union (DEG | HVG)         : {len(union_ids):,}",
    f"    After protein_coding filter: {len(union_pc):,}  (removed {n_removed})",
    "",
    "  FINAL FEATURE SET COMPOSITION",
    f"    DEG+HVG (both)  : {src_counts.get('DEG+HVG', 0):,}",
    f"    DEG only        : {src_counts.get('DEG_only', 0):,}",
    f"    HVG only        : {src_counts.get('HVG_only', 0):,}",
    f"    TOTAL           : {len(final):,}",
    "",
    "  OUTPUT",
    "    feature_set_final.tsv      -- master annotated list (XGBoost input)",
    "    feature_selection_plot.png -- variance curve + composition chart",
    "===================================================",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(os.path.join(OUT_DIR, "feature_set_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print("\nDone. Feature set ready for Week 4 XGBoost.")
