"""
Week 3 · Task 1 — Gene Regulatory Network Inference (GENIE3-style)
===================================================================
Algorithm: All-vs-all ExtraTreesRegressor (GENIE3 approach)
  For each of the 3,031 feature genes as TARGET:
    - Features = expression of the other 3,030 genes
    - Fit ExtraTreesRegressor on 433 samples
    - Extract feature_importances_ → importance row in the network matrix
  Result: 3,031 × 3,031 directed weighted adjacency matrix
          cell [i, j] = importance of gene j (regulator) for predicting gene i (target)

Expression input: VST-normalized, confounder-corrected matrix (Week 1)
  Note: VST values are already log-scale; no additional log1p transform needed.
  (If aak500_cpmdat.csv were used instead, log1p transform would be required.)

Feature genes: 3,031 protein-coding genes from Week 2 Task 3

Outputs:
  network_importance_matrix.tsv.gz  — full 3031×3031 importance matrix (compressed)
  network_edge_list.tsv             — all edges (regulator→target, weight), sorted desc
  network_top_edges.tsv             — top 100,000 edges for visualization
  network_hub_genes.tsv             — top 50 hub genes by total outgoing importance
  network_summary.txt               — run statistics
  network_degree_plot.png           — importance-score distribution
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from joblib import Parallel, delayed

# ── 0. Paths & parameters ──────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OUT_DIR    = SCRIPT_DIR

VST_PATH  = os.path.join(ROOT, "week1/task3_confounder_correction/Genelevel_VST_corrected.tsv")
FEAT_PATH = os.path.join(ROOT, "week2/task3_feature_set/feature_set_final.tsv")

# GENIE3 hyperparameters
N_TREES      = 100           # trees per gene model (1000 = more accurate, slower)
MAX_FEATURES = "sqrt"        # standard GENIE3 setting: sqrt(n_features)
N_JOBS       = -1            # -1 = use all 24 cores
TOP_EDGES    = 100_000       # edges to save in top-edge file


# ── 1. Load & prepare expression matrix ────────────────────────────────────────
print("-- 1. Loading data --")
t_load = time.time()

vst  = pd.read_csv(VST_PATH,  sep="\t", index_col=0)
feat = pd.read_csv(FEAT_PATH, sep="\t")

# Keep only feature genes present in VST
feature_ids = feat["ensembl_id"].tolist()
avail_ids   = [g for g in feature_ids if g in vst.index]
avail_mask  = feat["ensembl_id"].isin(set(avail_ids))

print(f"  VST matrix     : {vst.shape[0]:,} genes x {vst.shape[1]:,} samples")
print(f"  Feature genes  : {len(feature_ids):,}")
print(f"  Found in VST   : {len(avail_ids):,}")

# Expression matrix: samples x genes  (float32 saves ~half the RAM)
expr_df = vst.loc[avail_ids].T          # samples x genes
expr    = expr_df.values.astype(np.float32)
gene_ids= list(avail_ids)
id2sym  = feat.set_index("ensembl_id")["gene_symbol"].to_dict()

n_samples, n_genes = expr.shape
print(f"\n  Expression matrix shape : {n_samples} samples x {n_genes:,} genes")
print(f"  Memory (float32)        : {expr.nbytes / 1e6:.1f} MB")
print(f"  Load time               : {time.time()-t_load:.1f}s")

# Value-range sanity check (VST values should be ~5-15 for expressed genes)
print(f"  VST value range         : [{expr.min():.2f}, {expr.max():.2f}]")
print(f"  (Already log-scale; no log1p transform needed)")


# ── 2. GENIE3 parallel worker ──────────────────────────────────────────────────
def _fit_gene(target_idx, expr_shared, n_estimators, max_features):
    """
    Fit one ExtraTreesRegressor for target gene `target_idx`.
    Returns (target_idx, importance_array) where importance_array[j] is the
    importance of gene j for predicting gene target_idx.
    """
    # Build X (all other genes) and y (target gene)
    feature_mask = np.ones(expr_shared.shape[1], dtype=bool)
    feature_mask[target_idx] = False
    X = expr_shared[:, feature_mask]
    y = expr_shared[:, target_idx]

    et = ExtraTreesRegressor(
        n_estimators  = n_estimators,
        max_features  = max_features,
        random_state  = 42 + target_idx,
        n_jobs        = 1,            # inner parallelism OFF (joblib handles outer)
    )
    et.fit(X, y)

    # Map importances back to the full gene index space
    imp = np.zeros(expr_shared.shape[1], dtype=np.float32)
    imp[feature_mask] = et.feature_importances_.astype(np.float32)
    return target_idx, imp


# ── 3. Run all-vs-all network inference ───────────────────────────────────────
print(f"\n-- 2. Running GENIE3 ({n_genes:,} genes x {N_TREES} trees x {n_genes:,} predictors) --")
print(f"  Parallelism : {N_JOBS} (all cores)")

t_net = time.time()

results = Parallel(n_jobs=N_JOBS, prefer="threads", verbose=5)(
    delayed(_fit_gene)(i, expr, N_TREES, MAX_FEATURES)
    for i in range(n_genes)
)

elapsed_net = time.time() - t_net
print(f"\n  Network inference complete in {elapsed_net:.1f}s ({elapsed_net/60:.2f} min)")

# ── 4. Assemble importance matrix ─────────────────────────────────────────────
print("\n-- 3. Assembling importance matrix --")
# importance_matrix[i, j] = importance of gene j (regulator) for predicting gene i (target)
importance_matrix = np.zeros((n_genes, n_genes), dtype=np.float32)
for idx, imp in results:
    importance_matrix[idx] = imp

# Set diagonal to 0 (gene cannot predict itself, already done, but explicit)
np.fill_diagonal(importance_matrix, 0.0)

print(f"  Matrix shape  : {importance_matrix.shape}")
print(f"  Non-zero edges: {(importance_matrix > 0).sum():,}")
print(f"  Max importance: {importance_matrix.max():.6f}")
print(f"  Mean (non-zero): {importance_matrix[importance_matrix > 0].mean():.6f}")


# ── 5. Save full importance matrix (compressed TSV) ──────────────────────────
print("\n-- 4. Saving importance matrix --")
t_save = time.time()
imp_df = pd.DataFrame(importance_matrix, index=gene_ids, columns=gene_ids)
imp_df.index.name   = "target_gene"
imp_df.columns.name = "regulator_gene"

mat_path = os.path.join(OUT_DIR, "network_importance_matrix.tsv.gz")
imp_df.to_csv(mat_path, sep="\t", compression="gzip")
print(f"  Saved -> network_importance_matrix.tsv.gz  ({os.path.getsize(mat_path)/1e6:.1f} MB)")


# ── 6. Build edge list ─────────────────────────────────────────────────────────
print("\n-- 5. Building edge list --")
# Extract all non-zero edges: (regulator_idx, target_idx, weight)
rows, cols = np.where(importance_matrix > 0)
weights    = importance_matrix[rows, cols]

# Sort by importance descending
order = np.argsort(-weights)
rows, cols, weights = rows[order], cols[order], weights[order]

edges = pd.DataFrame({
    "regulator_ensembl" : [gene_ids[c] for c in cols],
    "regulator_symbol"  : [id2sym.get(gene_ids[c], "") for c in cols],
    "target_ensembl"    : [gene_ids[r] for r in rows],
    "target_symbol"     : [id2sym.get(gene_ids[r], "") for r in rows],
    "importance_score"  : weights,
})

print(f"  Total edges  : {len(edges):,}")
print(f"  Score range  : [{edges['importance_score'].min():.6f}, "
      f"{edges['importance_score'].max():.6f}]")

# Save full edge list
edge_path = os.path.join(OUT_DIR, "network_edge_list.tsv")
edges.to_csv(edge_path, sep="\t", index=False)
print(f"  Saved -> network_edge_list.tsv  ({os.path.getsize(edge_path)/1e6:.1f} MB)")

# Save top-N edge list
top_path = os.path.join(OUT_DIR, "network_top_edges.tsv")
edges.head(TOP_EDGES).to_csv(top_path, sep="\t", index=False)
print(f"  Saved -> network_top_edges.tsv  (top {TOP_EDGES:,} edges)")


# ── 7. Hub gene analysis ──────────────────────────────────────────────────────
print("\n-- 6. Hub gene analysis --")
# Out-strength: sum of importances from gene j as REGULATOR
# (sum over each column j of the importance matrix)
out_strength = importance_matrix.sum(axis=0)   # shape (n_genes,)
in_strength  = importance_matrix.sum(axis=1)   # shape (n_genes,)

hub_df = pd.DataFrame({
    "ensembl_id"          : gene_ids,
    "gene_symbol"         : [id2sym.get(g, "") for g in gene_ids],
    "out_strength"        : out_strength,    # how strongly this gene regulates others
    "in_strength"         : in_strength,     # how strongly this gene is regulated
    "out_degree"          : (importance_matrix > 0).sum(axis=0),  # # targets
    "in_degree"           : (importance_matrix > 0).sum(axis=1),  # # regulators
}).sort_values("out_strength", ascending=False).reset_index(drop=True)

hub_df["out_rank"] = range(1, len(hub_df) + 1)

print("  Top 15 hub genes (by outgoing regulatory strength):")
print("  {:>4}  {:>12}  {:>14}  {:>12}  {:>10}".format(
    "Rank", "Gene", "Out-strength", "In-strength", "Out-degree"))
for _, row in hub_df.head(15).iterrows():
    print("  {:>4}  {:>12}  {:>14.4f}  {:>12.4f}  {:>10}".format(
        int(row["out_rank"]), row["gene_symbol"],
        row["out_strength"], row["in_strength"],
        int(row["out_degree"])))

hub_path = os.path.join(OUT_DIR, "network_hub_genes.tsv")
hub_df.to_csv(hub_path, sep="\t", index=False)
print(f"\n  Saved -> network_hub_genes.tsv  ({len(hub_df):,} genes ranked by out-strength)")


# ── 8. Visualization ──────────────────────────────────────────────────────────
print("\n-- 7. Drawing network statistics plots --")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Week 3 Task 1 — Gene Regulatory Network Statistics", fontsize=12, fontweight="bold")

# Panel A: Importance score distribution (log scale)
ax = axes[0]
pos_scores = weights[weights > 0]
ax.hist(np.log10(pos_scores + 1e-10), bins=80, color="#4e79a7", alpha=0.8, edgecolor="none")
# Mark top 100K cutoff
if len(pos_scores) > TOP_EDGES:
    top_cutoff = np.sort(pos_scores)[::-1][TOP_EDGES - 1]
    ax.axvline(np.log10(top_cutoff), color="#e15759", lw=1.5, linestyle="--",
               label=f"Top {TOP_EDGES//1000}K cutoff")
    ax.legend(fontsize=8)
ax.set_xlabel("log10(importance score)", fontsize=9)
ax.set_ylabel("Number of edges", fontsize=9)
ax.set_title("Edge Importance Distribution", fontsize=10, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)

# Panel B: Out-strength distribution (top regulators)
ax2 = axes[1]
top_n = 30
top_hubs = hub_df.head(top_n)
colors_bar = ["#e15759" if i < 10 else "#4e79a7" for i in range(top_n)]
ax2.barh(range(top_n - 1, -1, -1),
         top_hubs["out_strength"].values,
         color=colors_bar, alpha=0.85)
ax2.set_yticks(range(top_n - 1, -1, -1))
ax2.set_yticklabels(top_hubs["gene_symbol"].values, fontsize=7)
ax2.set_xlabel("Total out-strength (sum of importances)", fontsize=9)
ax2.set_title(f"Top {top_n} Hub Genes\n(Regulatory Hubs by Out-strength)", fontsize=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)

# Panel C: In-strength vs Out-strength scatter (log-log)
ax3 = axes[2]
ax3.scatter(
    np.log10(hub_df["out_strength"] + 1e-8),
    np.log10(hub_df["in_strength"]  + 1e-8),
    s=6, alpha=0.4, color="#555555", linewidths=0, rasterized=True,
)
# Highlight top 20 hubs
top20 = hub_df.head(20)
ax3.scatter(
    np.log10(top20["out_strength"] + 1e-8),
    np.log10(top20["in_strength"]  + 1e-8),
    s=30, color="#e15759", zorder=5, label="Top 20 hubs",
)
for _, row in top20.head(10).iterrows():
    ax3.annotate(
        row["gene_symbol"],
        (np.log10(row["out_strength"] + 1e-8),
         np.log10(row["in_strength"] + 1e-8)),
        fontsize=6, color="#222222",
        xytext=(3, 3), textcoords="offset points",
    )
ax3.set_xlabel("log10(Out-strength)", fontsize=9)
ax3.set_ylabel("log10(In-strength)",  fontsize=9)
ax3.set_title("Regulatory vs Regulated Strength\n(Hub gene scatter)", fontsize=10, fontweight="bold")
ax3.legend(fontsize=8)
ax3.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "network_degree_plot.png")
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"  Saved -> network_degree_plot.png")


# ── 9. Summary report ─────────────────────────────────────────────────────────
summary_lines = [
    "===================================================",
    "  Week 3 Task 1 -- Gene Regulatory Network",
    "===================================================",
    "",
    "  ALGORITHM     : GENIE3-style ExtraTreesRegressor",
    f"  n_estimators  : {N_TREES}",
    f"  max_features  : {MAX_FEATURES}",
    f"  n_jobs        : {N_JOBS} (all cores)",
    f"  Runtime       : {elapsed_net:.1f}s ({elapsed_net/60:.2f} min)",
    "",
    "  INPUT",
    f"    Expression  : VST-corrected, {n_samples} samples x {n_genes:,} genes",
    f"    Note        : VST is log-scale; no log1p needed",
    "",
    "  NETWORK STATISTICS",
    f"    Genes (nodes)       : {n_genes:,}",
    f"    Total edges         : {len(edges):,}",
    f"    Non-zero edges      : {(importance_matrix > 0).sum():,}",
    f"    Max edge weight     : {importance_matrix.max():.6f}",
    f"    Mean edge weight    : {importance_matrix[importance_matrix>0].mean():.6f}",
    "",
    "  TOP 10 HUB GENES (by out-strength)",
]
for _, row in hub_df.head(10).iterrows():
    summary_lines.append(
        f"    {int(row['out_rank']):>2}. {row['gene_symbol']:<12} "
        f"out={row['out_strength']:.4f}  in={row['in_strength']:.4f}"
    )
summary_lines += [
    "",
    "  OUTPUTS",
    "    network_importance_matrix.tsv.gz  -- full 3031x3031 matrix",
    "    network_edge_list.tsv             -- all edges (regulator->target, weight)",
    f"    network_top_edges.tsv             -- top {TOP_EDGES:,} edges",
    "    network_hub_genes.tsv             -- hub gene rankings",
    "    network_degree_plot.png           -- statistics plots",
    "===================================================",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

with open(os.path.join(OUT_DIR, "network_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")

print(f"\nTotal script time: {time.time()-t_load:.1f}s")
