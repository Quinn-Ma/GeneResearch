"""
Week 3 · Task 1 — Steps 2 & 3
================================
Step 2: Infer Directed Edges
    For every gene pair {A, B}, compare:
        A→B strength = importance_matrix[B, A]  (A as regulator predicting B)
        B→A strength = importance_matrix[A, B]  (B as regulator predicting A)
    asymmetry = (forward - reverse) / (forward + reverse)
    Dominant direction = whichever score is larger.
    High |asymmetry| = confident direction. Low |asymmetry| = co-regulated pair.

Step 3: Rank Regulators (GRNBoost2-style)
    Four metrics per gene as regulator:
      out_strength    — aggregate regulatory influence (sum of importances as regulator)
      specificity     — 1 - Shannon entropy of importance distribution (1 = few strong targets)
      directionality  — out_strength / (out_strength + in_strength)  (1 = pure upstream)
      nonlinearity    — ratio of ET importance to |Pearson correlation| (captures non-linear effects
                        invisible to WGCNA / correlation methods)

Outputs:
  directed_edge_list.tsv       — directed edges with asymmetry & confidence scores
  regulator_rankings.tsv       — composite regulator ranking table
  nonlinear_regulators.tsv     — top non-linear regulators (high ET, low Pearson)
  step2_step3_plot.png         — 4-panel figure
"""

import os, sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── 0. Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
OUT_DIR    = SCRIPT_DIR

IMP_NPY   = os.path.join(OUT_DIR, "network_importance_matrix.npy")
IDS_JSON  = os.path.join(OUT_DIR, "network_gene_ids.json")
FEAT_PATH = os.path.join(ROOT, "week2/task3_feature_set/feature_set_final.tsv")
VST_PATH  = os.path.join(ROOT, "week1/task3_confounder_correction/Genelevel_VST_corrected.tsv")

EDGE_PCTILE  = 95    # percentile threshold for "significant" edges in direction analysis
TOP_DIRECTED = 50_000   # directed edges to save


# ── 1. Load importance matrix & metadata ─────────────────────────────────────
print("-- 1. Loading importance matrix --")
t0 = time.time()

imp = np.load(IMP_NPY)           # shape (n_genes, n_genes): imp[target, regulator]
with open(IDS_JSON) as f:
    gene_ids = json.load(f)      # regulator order = column order = row order

feat = pd.read_csv(FEAT_PATH, sep="\t")
id2sym = feat.set_index("ensembl_id")["gene_symbol"].to_dict()

n_genes = len(gene_ids)
print(f"  Matrix shape  : {imp.shape}  ({n_genes} genes)")
print(f"  Load time     : {time.time()-t0:.2f}s")


# ── 2. Pre-compute per-gene statistics ────────────────────────────────────────
print("\n-- 2. Computing gene-level statistics --")

out_strength = imp.sum(axis=0)   # gene j as REGULATOR  (sum along rows = over all targets)
in_strength  = imp.sum(axis=1)   # gene i as TARGET      (sum along columns = over all regulators)

# Note: sklearn feature_importances_ sum to 1 per model →
#   each row of imp sums to 1 → in_strength[i] == 1.0 for all i
# → in_strength carries no discrimination; use directionality ratio as effective metric

# Directionality: how much more does this gene REGULATE vs BEING regulated?
# We use out_strength normalised by number of targets (expected ~3030)
directionality = out_strength / (out_strength + in_strength + 1e-10)

# Specificity (Shannon entropy of importance distribution)
# low entropy → few dominant targets (specific regulator)
# high entropy → diffuse importance (broad, weak regulator)
col_sum = out_strength + 1e-30
p_matrix = imp / col_sum                          # (n_genes, n_genes) — prob distribution over targets
log_p = np.where(p_matrix > 0, np.log(p_matrix), 0.0)
entropy = -np.sum(p_matrix * log_p, axis=0)      # per-regulator
max_entropy = np.log(n_genes - 1)
specificity = 1.0 - entropy / max_entropy         # 1 = very specific, 0 = diffuse

print(f"  out_strength range  : [{out_strength.min():.4f}, {out_strength.max():.4f}]")
print(f"  directionality mean : {directionality.mean():.4f}")
print(f"  specificity range   : [{specificity.min():.4f}, {specificity.max():.4f}]")


# ── 3. Non-linearity score (ET importance vs Pearson correlation) ──────────────
print("\n-- 3. Computing non-linearity scores --")
t_nl = time.time()

# Load VST expression for the feature genes
vst = pd.read_csv(VST_PATH, sep="\t", index_col=0)
expr = vst.loc[gene_ids].T.values.astype(np.float32)  # (n_samples, n_genes)
print(f"  Expression loaded: {expr.shape}")

# Pearson correlation matrix — numpy vectorized
corr_matrix = np.corrcoef(expr.T).astype(np.float32)  # (n_genes, n_genes)
np.fill_diagonal(corr_matrix, 0.0)

# For each gene j as regulator:
#   avg_et_imp[j]   = mean of imp[:, j]   (mean importance across all targets)
#   avg_pearson[j]  = mean |corr_matrix[:, j]| (mean |correlation| with all others)
#   nonlinearity[j] = avg_et_imp[j] / (avg_pearson[j] + 1e-6)
avg_et_imp  = imp.mean(axis=0)                    # (n_genes,)
avg_pearson = np.abs(corr_matrix).mean(axis=0)    # (n_genes,)
nonlinearity = avg_et_imp / (avg_pearson + 1e-6)

print(f"  Pearson corr mean (abs): {avg_pearson.mean():.4f}")
print(f"  Non-linearity range    : [{nonlinearity.min():.4f}, {nonlinearity.max():.4f}]")
print(f"  Correlation time       : {time.time()-t_nl:.1f}s")


# ── 4. Build regulator ranking table ──────────────────────────────────────────
print("\n-- 4. Building regulator ranking --")

# Composite score: rank sum of (out_strength, specificity, nonlinearity)
reg_df = pd.DataFrame({
    "ensembl_id"    : gene_ids,
    "gene_symbol"   : [id2sym.get(g, "") for g in gene_ids],
    "out_strength"  : out_strength,
    "specificity"   : specificity,
    "directionality": directionality,
    "avg_et_imp"    : avg_et_imp,
    "avg_pearson"   : avg_pearson,
    "nonlinearity"  : nonlinearity,
})

# Composite rank: sum of percentile ranks across the three metrics
for col in ["out_strength", "specificity", "nonlinearity"]:
    reg_df[f"{col}_pctile"] = reg_df[col].rank(pct=True)

reg_df["composite_score"] = (
    reg_df["out_strength_pctile"] +
    reg_df["specificity_pctile"]  +
    reg_df["nonlinearity_pctile"]
) / 3.0

reg_df = reg_df.sort_values("out_strength", ascending=False).reset_index(drop=True)
reg_df["out_rank"] = range(1, len(reg_df) + 1)

reg_path = os.path.join(OUT_DIR, "regulator_rankings.tsv")
reg_df.to_csv(reg_path, sep="\t", index=False)
print(f"  Saved -> regulator_rankings.tsv")

print("\n  Top 15 regulators (out_strength + nonlinearity):")
print("  {:>4}  {:>12}  {:>11}  {:>11}  {:>12}  {:>10}".format(
    "Rank", "Gene", "Out-Strength", "Specificity", "Nonlinearity", "Composite"))
for _, row in reg_df.head(15).iterrows():
    print("  {:>4}  {:>12}  {:>11.4f}  {:>11.4f}  {:>12.4f}  {:>10.4f}".format(
        int(row["out_rank"]), row["gene_symbol"],
        row["out_strength"], row["specificity"],
        row["nonlinearity"], row["composite_score"]))

# Top non-linear regulators
nl_df = reg_df.sort_values("nonlinearity", ascending=False).head(100)
nl_path = os.path.join(OUT_DIR, "nonlinear_regulators.tsv")
nl_df.to_csv(nl_path, sep="\t", index=False)
print(f"\n  Top non-linear regulators (high ET, low Pearson):")
for _, row in nl_df.head(10).iterrows():
    print(f"    {row['gene_symbol']:<12}  ET_imp={row['avg_et_imp']:.5f}  "
          f"|Pearson|={row['avg_pearson']:.4f}  NL={row['nonlinearity']:.4f}")
print(f"  Saved -> nonlinear_regulators.tsv")


# ── 5. Step 2: Directed edge inference (vectorised) ───────────────────────────
print("\n-- 5. Inferring directed edges --")
t_dir = time.time()

# All unique pairs (i, j) where i < j
I, J = np.triu_indices(n_genes, k=1)        # shape: (n_pairs,)

# i→j strength: gene i as regulator predicting gene j (target)
#   = imp[target=j, regulator=i] = imp[J, I]
fwd = imp[J, I].astype(np.float64)          # A→B: i regulates j

# j→i strength: gene j as regulator predicting gene i (target)
#   = imp[target=i, regulator=j] = imp[I, J]
rev = imp[I, J].astype(np.float64)          # B→A: j regulates i

total     = fwd + rev + 1e-12
asymmetry = (fwd - rev) / total             # >0: i→j dominant; <0: j→i dominant
confidence= np.abs(asymmetry)
max_score = np.maximum(fwd, rev)

# Keep only pairs above significance threshold
thresh = np.percentile(max_score, EDGE_PCTILE)
sig_mask = max_score >= thresh
print(f"  Significance threshold (p{EDGE_PCTILE}): {thresh:.6f}")
print(f"  Significant pairs  : {sig_mask.sum():,} / {len(I):,}")

I_sig, J_sig = I[sig_mask], J[sig_mask]
fwd_sig = fwd[sig_mask]
rev_sig = rev[sig_mask]
asym_sig= asymmetry[sig_mask]
conf_sig= confidence[sig_mask]

# Determine dominant direction
# asymmetry > 0 → i→j,  asymmetry < 0 → j→i
reg_idx  = np.where(asym_sig >= 0, I_sig, J_sig)   # upstream / regulator index
tgt_idx  = np.where(asym_sig >= 0, J_sig, I_sig)   # downstream / target index
net_score= np.maximum(fwd_sig, rev_sig)             # magnitude of dominant edge

# Build directed edge dataframe
dir_df = pd.DataFrame({
    "regulator_ensembl" : [gene_ids[k] for k in reg_idx],
    "regulator_symbol"  : [id2sym.get(gene_ids[k], "") for k in reg_idx],
    "target_ensembl"    : [gene_ids[k] for k in tgt_idx],
    "target_symbol"     : [id2sym.get(gene_ids[k], "") for k in tgt_idx],
    "forward_score"     : fwd_sig,
    "reverse_score"     : rev_sig,
    "asymmetry"         : asym_sig,
    "direction_confidence": conf_sig,
    "dominant_score"    : net_score,
}).sort_values("dominant_score", ascending=False).reset_index(drop=True)

# Summary: confidence distribution
q_lo = np.percentile(conf_sig, 25)
q_hi = np.percentile(conf_sig, 75)
clearly_directional = (conf_sig > 0.5).sum()
print(f"  Direction confidence: Q25={q_lo:.3f}  Q75={q_hi:.3f}")
print(f"  Clearly directional (conf>0.5): {clearly_directional:,} "
      f"({clearly_directional/len(conf_sig)*100:.1f}%)")

dir_path = os.path.join(OUT_DIR, "directed_edge_list.tsv")
dir_df.head(TOP_DIRECTED).to_csv(dir_path, sep="\t", index=False)
print(f"  Saved -> directed_edge_list.tsv  (top {min(TOP_DIRECTED, len(dir_df)):,} edges)")
print(f"  Direction inference time: {time.time()-t_dir:.1f}s")


# ── 6. Visualization ──────────────────────────────────────────────────────────
print("\n-- 6. Drawing plots --")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Week 3 Task 1 — Steps 2 & 3: Direction Inference & Regulator Ranking",
             fontsize=13, fontweight="bold")

# ── Panel A: Direction confidence histogram ────────────────────────────────────
ax = axes[0, 0]
bins = np.linspace(0, 1, 60)
ax.hist(conf_sig, bins=bins, color="#4e79a7", alpha=0.85, edgecolor="none")
ax.axvline(0.5, color="#e15759", lw=2, linestyle="--",
           label=f"Confidence=0.5 ({clearly_directional:,} pairs, "
                 f"{clearly_directional/len(conf_sig)*100:.1f}%)")
ax.axvline(q_lo, color="#999999", lw=1, linestyle=":", label=f"Q25={q_lo:.3f}")
ax.axvline(q_hi, color="#555555", lw=1, linestyle=":", label=f"Q75={q_hi:.3f}")
ax.set_xlabel("Direction Confidence  |asymmetry| = |fwd-rev|/(fwd+rev)", fontsize=9)
ax.set_ylabel("Number of gene pairs", fontsize=9)
ax.set_title("Step 2 — Edge Directionality Confidence\n"
             "(confidence=1: purely unidirectional; 0: symmetric co-expression)", fontsize=10, fontweight="bold")
ax.legend(fontsize=8, framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)

# ── Panel B: Top regulators bar chart ──────────────────────────────────────────
ax2 = axes[0, 1]
top_n = 25
top_regs = reg_df.head(top_n)
# Colour by composite score (darker = higher composite)
norm = mcolors.Normalize(vmin=top_regs["composite_score"].min(),
                         vmax=top_regs["composite_score"].max())
cmap = plt.cm.YlOrRd
bar_colors = [cmap(norm(v)) for v in top_regs["composite_score"]]

ax2.barh(range(top_n - 1, -1, -1),
         top_regs["out_strength"].values,
         color=bar_colors, alpha=0.9)
ax2.set_yticks(range(top_n - 1, -1, -1))
ax2.set_yticklabels(top_regs["gene_symbol"].values, fontsize=8)
ax2.set_xlabel("Out-strength (aggregate regulatory influence)", fontsize=9)
ax2.set_title(f"Step 3 — Top {top_n} Regulatory Hubs\n(colour = composite score: out_strength + specificity + nonlinearity)",
              fontsize=10, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax2, label="Composite score", shrink=0.7)
ax2.spines[["top", "right"]].set_visible(False)

# ── Panel C: Non-linearity scatter (ET importance vs |Pearson|) ───────────────
ax3 = axes[1, 0]
ax3.scatter(
    avg_pearson, avg_et_imp,
    s=5, alpha=0.3, color="#888888", linewidths=0, rasterized=True,
    label="All 3,031 genes",
)

# Highlight top non-linear genes
top_nl = reg_df.sort_values("nonlinearity", ascending=False).head(20)
ax3.scatter(
    top_nl["avg_pearson"], top_nl["avg_et_imp"],
    s=50, color="#e15759", zorder=5, label="Top 20 non-linear regulators",
)
# Highlight top out_strength genes
top_os = reg_df.head(20)
ax3.scatter(
    top_os["avg_pearson"], top_os["avg_et_imp"],
    s=40, color="#4e79a7", marker="D", zorder=4, label="Top 20 hub genes (out-strength)",
)
for _, row in top_nl.head(8).iterrows():
    ax3.annotate(row["gene_symbol"],
                 (row["avg_pearson"], row["avg_et_imp"]),
                 fontsize=6.5, color="#c00000",
                 xytext=(4, 2), textcoords="offset points")

# Reference diagonal: ET importance = |Pearson| (linear relationship line)
x_ref = np.linspace(avg_pearson.min(), avg_pearson.max(), 100)
# Scale to match ET axis
ratio_med = np.median(avg_et_imp) / np.median(avg_pearson)
ax3.plot(x_ref, x_ref * ratio_med, color="#333333", lw=1, linestyle="--",
         alpha=0.5, label="Linear expectation")

ax3.set_xlabel("Average |Pearson correlation| with other genes", fontsize=9)
ax3.set_ylabel("Average ExtraTrees importance score", fontsize=9)
ax3.set_title("Step 3 — Non-linear Co-regulation Detection\n"
              "(above line: ET captures signal invisible to Pearson/WGCNA)",
              fontsize=10, fontweight="bold")
ax3.legend(fontsize=7.5, framealpha=0.8)
ax3.spines[["top", "right"]].set_visible(False)

# ── Panel D: Asymmetry distribution + bidirectionality breakdown ──────────────
ax4 = axes[1, 1]
# Raw asymmetry (−1 to +1): positive = A→B direction by our pair labelling
ax4.hist(asym_sig, bins=80, color="#59a14f", alpha=0.8, edgecolor="none")
ax4.axvline(0,   color="black", lw=1.2)
ax4.axvline( 0.5, color="#e15759", lw=1.5, linestyle="--", alpha=0.8)
ax4.axvline(-0.5, color="#e15759", lw=1.5, linestyle="--", alpha=0.8,
            label="Confidence threshold = 0.5")

# Annotate zones
total_sig = len(asym_sig)
n_fwd  = (asym_sig > 0.5).sum()
n_rev  = (asym_sig < -0.5).sum()
n_bidi = total_sig - n_fwd - n_rev
ax4.text(0.75, 0.92, f"Directed\nA->B\n{n_fwd:,}\n({n_fwd/total_sig*100:.1f}%)",
         transform=ax4.transAxes, ha="center", fontsize=8, color="#e15759")
ax4.text(0.25, 0.92, f"Directed\nB->A\n{n_rev:,}\n({n_rev/total_sig*100:.1f}%)",
         transform=ax4.transAxes, ha="center", fontsize=8, color="#e15759")
ax4.text(0.50, 0.85, f"Co-regulated\n(bidirectional)\n{n_bidi:,}\n({n_bidi/total_sig*100:.1f}%)",
         transform=ax4.transAxes, ha="center", fontsize=8, color="#666666")

ax4.set_xlabel("Asymmetry score  (fwd - rev) / (fwd + rev)", fontsize=9)
ax4.set_ylabel("Number of gene pairs", fontsize=9)
ax4.set_title("Step 2 — Asymmetry Distribution\n(edge pair direction balance)",
              fontsize=10, fontweight="bold")
ax4.legend(fontsize=8)
ax4.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plot_path = os.path.join(OUT_DIR, "step2_step3_plot.png")
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
plt.close("all")
print(f"  Saved -> step2_step3_plot.png")


# ── 7. Summary ────────────────────────────────────────────────────────────────
summary_lines = [
    "===================================================",
    "  Week 3 Task 1 Steps 2 & 3 -- Direction & Ranking",
    "===================================================",
    "",
    "  STEP 2 — DIRECTION INFERENCE",
    f"    Gene pairs analysed      : {len(I):,}",
    f"    Significance threshold   : p{EDGE_PCTILE} = {thresh:.6f}",
    f"    Significant pairs        : {sig_mask.sum():,}",
    f"    Clearly directed (>0.5)  : {clearly_directional:,}  "
    f"({clearly_directional/len(conf_sig)*100:.1f}%)",
    f"    Bidirectional (<=0.5)    : {len(conf_sig)-clearly_directional:,}  "
    f"({(len(conf_sig)-clearly_directional)/len(conf_sig)*100:.1f}%)",
    f"    Directed A->B            : {n_fwd:,}  ({n_fwd/total_sig*100:.1f}%)",
    f"    Directed B->A (reversed) : {n_rev:,}  ({n_rev/total_sig*100:.1f}%)",
    "",
    "  STEP 3 — REGULATOR RANKING (top 10 by out_strength)",
]
for _, row in reg_df.head(10).iterrows():
    summary_lines.append(
        f"    {int(row['out_rank']):>2}. {row['gene_symbol']:<12} "
        f"out={row['out_strength']:.4f}  "
        f"spec={row['specificity']:.3f}  "
        f"NL={row['nonlinearity']:.4f}"
    )
summary_lines += [
    "",
    "  TOP 5 NON-LINEAR REGULATORS (WGCNA-invisible)",
]
for _, row in nl_df.head(5).iterrows():
    summary_lines.append(
        f"    {row['gene_symbol']:<12}  ET={row['avg_et_imp']:.5f}  "
        f"|r|={row['avg_pearson']:.4f}  NL={row['nonlinearity']:.4f}"
    )
summary_lines += [
    "",
    "  OUTPUTS",
    "    directed_edge_list.tsv    -- top 50K directed edges with confidence",
    "    regulator_rankings.tsv   -- full regulator table (3031 genes)",
    "    nonlinear_regulators.tsv -- top 100 non-linear regulators",
    "    step2_step3_plot.png     -- 4-panel figure",
    "===================================================",
]
summary_text = "\n".join(summary_lines)
print("\n" + summary_text)
with open(os.path.join(OUT_DIR, "step2_step3_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print(f"\nDone in {time.time()-t0:.1f}s total.")
