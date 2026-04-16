"""
Week 3 · Task 3 — Rank Regulators (GRNBoost2-style)
=====================================================
Four metrics per gene as regulator:
  out_strength   — aggregate regulatory influence = sum of importance scores
                   as a predictor across all 3030 target genes
  specificity    — 1 - Shannon entropy of importance distribution
                   (1 = few dominant targets; 0 = diffuse broad regulator)
  directionality — out_strength / (out_strength + in_strength)
                   (closer to 1 = pure upstream regulator)
  nonlinearity   — avg_ET_importance / avg_|Pearson_correlation|
                   captures regulators whose effect is NON-LINEAR
                   (threshold / interaction effects invisible to WGCNA)

Input:
  ../task1_build_network/network_importance_matrix.npy
  ../task1_build_network/network_gene_ids.json
  ../../week1/task3_confounder_correction/Genelevel_VST_corrected.tsv
  ../../week2/task3_feature_set/feature_set_final.tsv

Outputs:
  regulator_rankings.tsv       — all 3031 genes ranked by out_strength
  nonlinear_regulators.tsv     — top 100 non-linear regulators
  regulator_ranking_plot.png   — hub bar chart + non-linearity scatter
  ranking_summary.txt
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

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TASK1_DIR  = os.path.join(SCRIPT_DIR, "../task1_build_network")
ROOT       = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
FEAT_PATH  = os.path.join(ROOT, "week2/task3_feature_set/feature_set_final.tsv")
VST_PATH   = os.path.join(ROOT, "week1/task3_confounder_correction/Genelevel_VST_corrected.tsv")

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

# ── 2. Core regulatory metrics ────────────────────────────────────────────────
print("\n-- 2. Computing regulatory metrics --")

# Out-strength: sum over all targets (axis=0 = sum down rows = over targets)
out_strength = imp.sum(axis=0)
in_strength  = imp.sum(axis=1)   # always ~1.0 due to sklearn normalization

# Directionality
directionality = out_strength / (out_strength + in_strength + 1e-10)

# Specificity (Shannon entropy of regulator's importance distribution)
col_sum  = out_strength + 1e-30
p_matrix = imp / col_sum                          # prob distribution over targets
log_p    = np.where(p_matrix > 0, np.log(p_matrix), 0.0)
entropy  = -np.sum(p_matrix * log_p, axis=0)     # per regulator
specificity = 1.0 - entropy / np.log(n_genes - 1) # normalised [0,1]

print(f"  out_strength range  : [{out_strength.min():.4f}, {out_strength.max():.4f}]")
print(f"  directionality mean : {directionality.mean():.4f}")
print(f"  specificity range   : [{specificity.min():.4f}, {specificity.max():.4f}]")

# ── 3. Non-linearity score (ET importance vs Pearson correlation) ──────────────
print("\n-- 3. Computing non-linearity scores --")
t_nl = time.time()

vst  = pd.read_csv(VST_PATH, sep="\t", index_col=0)
expr = vst.loc[gene_ids].T.values.astype(np.float32)
print(f"  Expression loaded: {expr.shape}")

corr_matrix = np.corrcoef(expr.T).astype(np.float32)
np.fill_diagonal(corr_matrix, 0.0)

avg_et_imp  = imp.mean(axis=0)
avg_pearson = np.abs(corr_matrix).mean(axis=0)
nonlinearity = avg_et_imp / (avg_pearson + 1e-6)

print(f"  Mean |Pearson| : {avg_pearson.mean():.4f}")
print(f"  NL range       : [{nonlinearity.min():.5f}, {nonlinearity.max():.5f}]")
print(f"  Computed in    : {time.time()-t_nl:.1f}s")

# ── 4. Build ranking table ─────────────────────────────────────────────────────
print("\n-- 4. Building ranking table --")

# Composite score = average percentile rank across 3 metrics
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
for col in ["out_strength", "specificity", "nonlinearity"]:
    reg_df[f"{col}_pctile"] = reg_df[col].rank(pct=True)
reg_df["composite_score"] = (
    reg_df["out_strength_pctile"] +
    reg_df["specificity_pctile"]  +
    reg_df["nonlinearity_pctile"]
) / 3.0

reg_df = reg_df.sort_values("out_strength", ascending=False).reset_index(drop=True)
reg_df["out_rank"] = range(1, len(reg_df) + 1)

reg_path = os.path.join(SCRIPT_DIR, "regulator_rankings.tsv")
reg_df.to_csv(reg_path, sep="\t", index=False)
print(f"  Saved -> regulator_rankings.tsv  ({len(reg_df):,} genes)")

nl_df = reg_df.sort_values("nonlinearity", ascending=False).head(100)
nl_path = os.path.join(SCRIPT_DIR, "nonlinear_regulators.tsv")
nl_df.to_csv(nl_path, sep="\t", index=False)
print(f"  Saved -> nonlinear_regulators.tsv  (top 100)")

print("\n  Top 15 regulators (by out_strength):")
print("  {:>4}  {:>12}  {:>12}  {:>11}  {:>12}  {:>10}".format(
    "Rank","Gene","Out-Strength","Specificity","Nonlinearity","Composite"))
for _, r in reg_df.head(15).iterrows():
    print("  {:>4}  {:>12}  {:>12.4f}  {:>11.4f}  {:>12.4f}  {:>10.4f}".format(
        int(r["out_rank"]), r["gene_symbol"],
        r["out_strength"], r["specificity"], r["nonlinearity"], r["composite_score"]))

print("\n  Top 10 non-linear regulators (WGCNA-invisible):")
for _, r in nl_df.head(10).iterrows():
    print(f"    {r['gene_symbol']:<12}  ET={r['avg_et_imp']:.5f}  "
          f"|r|={r['avg_pearson']:.4f}  NL={r['nonlinearity']:.4f}")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
print("\n-- 5. Drawing plots --")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Week 3 Task 3 — Regulator Ranking (GRNBoost2-style)",
             fontsize=12, fontweight="bold")

# Panel A: Hub bar chart coloured by composite score
ax = axes[0]
top_n  = 30
top_r  = reg_df.head(top_n)
norm   = mcolors.Normalize(vmin=top_r["composite_score"].min(),
                            vmax=top_r["composite_score"].max())
cmap   = plt.cm.YlOrRd
colors = [cmap(norm(v)) for v in top_r["composite_score"]]

ax.barh(range(top_n - 1, -1, -1), top_r["out_strength"].values,
        color=colors, alpha=0.9)
ax.set_yticks(range(top_n - 1, -1, -1))
ax.set_yticklabels(top_r["gene_symbol"].values, fontsize=8)
ax.set_xlabel("Out-strength (aggregate regulatory influence)", fontsize=10)
ax.set_title(f"Top {top_n} Regulatory Hubs\n"
             f"(colour = composite score: out_strength + specificity + nonlinearity)",
             fontsize=10, fontweight="bold")
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Composite score", shrink=0.7)
ax.spines[["top", "right"]].set_visible(False)

# Panel B: Non-linearity scatter  — ET importance vs |Pearson|
ax2 = axes[1]
ax2.scatter(avg_pearson, avg_et_imp,
            s=5, alpha=0.3, color="#888888", linewidths=0, rasterized=True,
            label="All 3,031 genes")

# Mark top non-linear genes
top20_nl = nl_df.head(20)
ax2.scatter(top20_nl["avg_pearson"], top20_nl["avg_et_imp"],
            s=60, color="#e15759", zorder=5, label="Top 20 non-linear regulators")

# Mark top hub genes
top20_hub = reg_df.head(20)
ax2.scatter(top20_hub["avg_pearson"], top20_hub["avg_et_imp"],
            s=45, color="#4e79a7", marker="D", zorder=4, label="Top 20 hub genes")

# Linear reference line
x_ref = np.linspace(avg_pearson.min(), avg_pearson.max(), 100)
scale = np.median(avg_et_imp) / np.median(avg_pearson)
ax2.plot(x_ref, x_ref * scale, color="#333333", lw=1, linestyle="--",
         alpha=0.5, label="Linear expectation")

for _, r in top20_nl.head(8).iterrows():
    ax2.annotate(r["gene_symbol"],
                 (r["avg_pearson"], r["avg_et_imp"]),
                 fontsize=6.5, color="#c00000",
                 xytext=(4, 2), textcoords="offset points")

ax2.set_xlabel("Average |Pearson correlation| with other genes", fontsize=10)
ax2.set_ylabel("Average ExtraTrees importance score", fontsize=10)
ax2.set_title("Non-linear Co-regulation Detection\n"
              "(above line: ET captures signal invisible to Pearson / WGCNA)",
              fontsize=10, fontweight="bold")
ax2.legend(fontsize=8, framealpha=0.8)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "regulator_ranking_plot.png"), dpi=200, bbox_inches="tight")
plt.close("all")
print("  Saved -> regulator_ranking_plot.png")

# ── 6. Summary ────────────────────────────────────────────────────────────────
lines = [
    "===================================================",
    "  Week 3 Task 3 -- Regulator Ranking",
    "===================================================",
    f"  Genes ranked    : {len(reg_df):,}",
    f"  Metrics         : out_strength, specificity, directionality, nonlinearity",
    f"  Composite score : avg percentile of out_strength + specificity + nonlinearity",
    "",
    "  Top 10 hubs (out_strength):",
]
for _, r in reg_df.head(10).iterrows():
    lines.append(f"    {int(r['out_rank']):>2}. {r['gene_symbol']:<12} "
                 f"out={r['out_strength']:.4f}  spec={r['specificity']:.3f}  "
                 f"NL={r['nonlinearity']:.4f}")
lines += [
    "",
    "  Top 5 non-linear regulators (WGCNA-invisible):",
]
for _, r in nl_df.head(5).iterrows():
    lines.append(f"    {r['gene_symbol']:<12}  ET={r['avg_et_imp']:.5f}  "
                 f"|r|={r['avg_pearson']:.4f}  NL={r['nonlinearity']:.4f}")
lines += [
    "",
    "  Outputs:",
    "    regulator_rankings.tsv      -- 3031-gene ranked table",
    "    nonlinear_regulators.tsv    -- top 100 non-linear regulators",
    "    regulator_ranking_plot.png  -- hub bars + NL scatter",
    "===================================================",
]
txt = "\n".join(lines)
print("\n" + txt)
with open(os.path.join(SCRIPT_DIR, "ranking_summary.txt"), "w", encoding="utf-8") as f:
    f.write(txt + "\n")
print(f"\nDone in {time.time()-t0:.1f}s")
