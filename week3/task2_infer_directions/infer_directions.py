"""
Week 3 · Task 2 — Infer Directed Edges
========================================
For each gene pair {A, B} in the 3,031-gene network:
  forward  A→B score = importance_matrix[B, A]   (A is regulator, B is target)
  reverse  B→A score = importance_matrix[A, B]   (B is regulator, A is target)
  asymmetry        = (forward - reverse) / (forward + reverse + eps)
  direction_conf   = |asymmetry|   [0 = symmetric co-expression; 1 = purely unidirectional]

Only pairs above the p95 edge-weight threshold are kept.
Dominant direction = whichever side has the larger importance score.

Input:
  ../task1_build_network/network_importance_matrix.npy
  ../task1_build_network/network_gene_ids.json

Outputs:
  directed_edge_list.tsv       — top 50K directed edges
  direction_confidence_plot.png
  direction_summary.txt
"""

import os, sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR  = os.path.join(SCRIPT_DIR, "../task1_build_network")
FEAT_PATH  = os.path.join(SCRIPT_DIR, "../../week2/task3_feature_set/feature_set_final.tsv")

EDGE_PCTILE  = 95
TOP_DIRECTED = 50_000

# ── 1. Load ────────────────────────────────────────────────────────────────────
print("-- 1. Loading importance matrix --")
t0 = time.time()
imp = np.load(os.path.join(TASK1_DIR, "network_importance_matrix.npy"))
with open(os.path.join(TASK1_DIR, "network_gene_ids.json")) as f:
    gene_ids = json.load(f)

feat   = pd.read_csv(FEAT_PATH, sep="\t")
id2sym = feat.set_index("ensembl_id")["gene_symbol"].to_dict()
n_genes = len(gene_ids)
print(f"  Matrix: {imp.shape}  loaded in {time.time()-t0:.2f}s")

# ── 2. Vectorised direction inference ─────────────────────────────────────────
print("\n-- 2. Inferring directed edges (all pairs) --")
t1 = time.time()

I, J = np.triu_indices(n_genes, k=1)          # all unique pairs i < j
fwd  = imp[J, I].astype(np.float64)           # A→B : gene i predicts gene j
rev  = imp[I, J].astype(np.float64)           # B→A : gene j predicts gene i

asymmetry  = (fwd - rev) / (fwd + rev + 1e-12)
confidence = np.abs(asymmetry)
max_score  = np.maximum(fwd, rev)

# Significance filter
thresh   = np.percentile(max_score, EDGE_PCTILE)
sig_mask = max_score >= thresh

I_s, J_s   = I[sig_mask], J[sig_mask]
fwd_s      = fwd[sig_mask]
rev_s      = rev[sig_mask]
asym_s     = asymmetry[sig_mask]
conf_s     = confidence[sig_mask]

# Assign dominant direction
reg_idx = np.where(asym_s >= 0, I_s, J_s)   # upstream gene index
tgt_idx = np.where(asym_s >= 0, J_s, I_s)   # downstream gene index

clearly_dir  = (conf_s > 0.5).sum()
bidirectional = len(conf_s) - clearly_dir

print(f"  Significance threshold (p{EDGE_PCTILE}) : {thresh:.6f}")
print(f"  Significant pairs       : {sig_mask.sum():,}")
print(f"  Clearly directed (>0.5) : {clearly_dir:,}  ({clearly_dir/len(conf_s)*100:.1f}%)")
print(f"  Bidirectional (<=0.5)   : {bidirectional:,}  ({bidirectional/len(conf_s)*100:.1f}%)")
print(f"  Inference time          : {time.time()-t1:.2f}s")

# ── 3. Build & save edge table ─────────────────────────────────────────────────
print("\n-- 3. Building directed edge table --")
dir_df = pd.DataFrame({
    "regulator_ensembl"  : [gene_ids[k] for k in reg_idx],
    "regulator_symbol"   : [id2sym.get(gene_ids[k], "") for k in reg_idx],
    "target_ensembl"     : [gene_ids[k] for k in tgt_idx],
    "target_symbol"      : [id2sym.get(gene_ids[k], "") for k in tgt_idx],
    "forward_score"      : fwd_s,
    "reverse_score"      : rev_s,
    "asymmetry"          : asym_s,
    "direction_confidence": conf_s,
    "dominant_score"     : np.maximum(fwd_s, rev_s),
}).sort_values("dominant_score", ascending=False).reset_index(drop=True)

out_edge = os.path.join(SCRIPT_DIR, "directed_edge_list.tsv")
dir_df.head(TOP_DIRECTED).to_csv(out_edge, sep="\t", index=False)
print(f"  Saved -> directed_edge_list.tsv  ({min(TOP_DIRECTED, len(dir_df)):,} edges)")

# ── 4. Plot ───────────────────────────────────────────────────────────────────
print("\n-- 4. Drawing direction confidence plots --")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Week 3 Task 2 — Directed Edge Inference", fontsize=12, fontweight="bold")

# Panel A: confidence histogram
ax = axes[0]
ax.hist(conf_s, bins=60, color="#4e79a7", alpha=0.85, edgecolor="none")
ax.axvline(0.5, color="#e15759", lw=2, linestyle="--",
           label=f"conf=0.5  ({clearly_dir:,} pairs, {clearly_dir/len(conf_s)*100:.1f}%)")
ax.set_xlabel("Direction Confidence  |fwd-rev| / (fwd+rev)", fontsize=10)
ax.set_ylabel("Number of gene pairs", fontsize=10)
ax.set_title("Directionality Confidence Distribution\n"
             "(1 = purely unidirectional; 0 = symmetric co-expression)", fontsize=10, fontweight="bold")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# Panel B: asymmetry distribution coloured by zone
ax2 = axes[1]
bins = np.linspace(-1, 1, 80)
# Colour each bin
counts, bin_edges = np.histogram(asym_s, bins=bins)
for i, cnt in enumerate(counts):
    mid = (bin_edges[i] + bin_edges[i+1]) / 2
    col = "#e15759" if abs(mid) > 0.5 else "#aec7e8"
    ax2.bar(bin_edges[i], cnt, width=bin_edges[i+1]-bin_edges[i],
            color=col, align="edge", edgecolor="none", alpha=0.9)
ax2.axvline(0,    color="black", lw=1)
ax2.axvline( 0.5, color="#e15759", lw=1.5, linestyle="--", alpha=0.7)
ax2.axvline(-0.5, color="#e15759", lw=1.5, linestyle="--", alpha=0.7)

n_fwd  = (asym_s >  0.5).sum()
n_rev  = (asym_s < -0.5).sum()
n_bidi = len(asym_s) - n_fwd - n_rev
ax2.text( 0.78, 0.90, f"A->B\n{n_fwd:,}\n({n_fwd/len(asym_s)*100:.0f}%)",
          transform=ax2.transAxes, ha="center", fontsize=9, color="#c00000")
ax2.text( 0.22, 0.90, f"B->A\n{n_rev:,}\n({n_rev/len(asym_s)*100:.0f}%)",
          transform=ax2.transAxes, ha="center", fontsize=9, color="#c00000")
ax2.text( 0.50, 0.82, f"Co-expr\n{n_bidi:,}\n({n_bidi/len(asym_s)*100:.0f}%)",
          transform=ax2.transAxes, ha="center", fontsize=9, color="#666666")
ax2.set_xlabel("Asymmetry score  (fwd - rev) / (fwd + rev)", fontsize=10)
ax2.set_ylabel("Number of gene pairs", fontsize=10)
ax2.set_title("Asymmetry Distribution\n"
              "(red bars = clearly directional; blue = co-regulated)", fontsize=10, fontweight="bold")
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "direction_confidence_plot.png"), dpi=200, bbox_inches="tight")
plt.close("all")
print("  Saved -> direction_confidence_plot.png")

# ── 5. Summary ────────────────────────────────────────────────────────────────
lines = [
    "===================================================",
    "  Week 3 Task 2 -- Infer Directed Edges",
    "===================================================",
    f"  Total gene pairs analysed : {len(I):,}",
    f"  Significance threshold    : p{EDGE_PCTILE} = {thresh:.6f}",
    f"  Significant pairs         : {sig_mask.sum():,}",
    f"  Clearly directed (>0.5)   : {clearly_dir:,} ({clearly_dir/len(conf_s)*100:.1f}%)",
    f"  Bidirectional (<=0.5)     : {bidirectional:,} ({bidirectional/len(conf_s)*100:.1f}%)",
    f"  A->B directed             : {n_fwd:,} ({n_fwd/len(asym_s)*100:.1f}%)",
    f"  B->A directed             : {n_rev:,} ({n_rev/len(asym_s)*100:.1f}%)",
    "",
    "  Top 5 most confident directed edges:",
]
for _, r in dir_df.head(5).iterrows():
    lines.append(f"    {r['regulator_symbol']:<10} -> {r['target_symbol']:<10} "
                 f"conf={r['direction_confidence']:.4f}  score={r['dominant_score']:.5f}")
lines += [
    "",
    "  Outputs:",
    "    directed_edge_list.tsv        -- top 50K directed edges",
    "    direction_confidence_plot.png -- confidence distribution plots",
    "===================================================",
]
txt = "\n".join(lines)
print("\n" + txt)
with open(os.path.join(SCRIPT_DIR, "direction_summary.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(f"\nDone in {time.time()-t0:.1f}s")
