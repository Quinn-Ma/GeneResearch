import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 1. Load MetaSheet ───────────────────────────────────────────────────────
meta = pd.read_csv("MetaSheet.csv", index_col=0, encoding="latin-1")
print(f"MetaSheet loaded          : {meta.shape[0]} samples")

# ── 2. RIN filter: keep RIN > 5 ─────────────────────────────────────────────
n_before = len(meta)
meta = meta[meta["rin"] > 5]
print(f"After RIN > 5 filter      : {len(meta)} samples  "
      f"(removed {n_before - len(meta)} with RIN ≤ 5)")

# ── 3. Drop samples missing mgs_level or genotype_id ────────────────────────
missing_mgs = meta["mgs_level"].isna().sum()
missing_geno = meta["genotype_id"].isna().sum()
meta = meta.dropna(subset=["mgs_level", "genotype_id"])
print(f"After missing-label filter: {len(meta)} samples  "
      f"(removed {missing_mgs} missing mgs_level, {missing_geno} missing genotype_id)")

# ── 4. Load filtered counts (from Task 1) ───────────────────────────────────
counts = pd.read_csv("Genelevel_filtered_counts.tsv", sep="\t", index_col=0)
print(f"\nFiltered counts matrix    : {counts.shape[0]} genes × {counts.shape[1]} samples")

# Align: keep only samples present in both meta and counts
common_samples = list(set(meta["r_id"].astype(str)) & set(counts.columns))
counts = counts[common_samples]
meta_aligned = meta[meta["r_id"].astype(str).isin(common_samples)].copy()
print(f"Samples after alignment   : {len(common_samples)}")

# ── 5. PCA for outlier detection ─────────────────────────────────────────────
# log1p transform then standardise
log_counts = np.log1p(counts.T.values)   # shape: (samples, genes)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(log_counts)

pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X_scaled)
explained = pca.explained_variance_ratio_ * 100

print(f"\nPCA variance explained    : PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")

# ── 6. Outlier detection: ±3 SD on PC1 and PC2 ──────────────────────────────
pc1, pc2 = pcs[:, 0], pcs[:, 1]
sample_ids = list(counts.columns)

outlier_mask = (
    (np.abs(pc1 - pc1.mean()) > 3 * pc1.std()) |
    (np.abs(pc2 - pc2.mean()) > 3 * pc2.std())
)
outlier_ids = [sid for sid, flag in zip(sample_ids, outlier_mask) if flag]
print(f"PCA outliers (±3 SD)      : {len(outlier_ids)}  →  {outlier_ids}")

# ── 7. Plot PCA (before outlier removal) ────────────────────────────────────
mgs_map = meta_aligned.set_index(meta_aligned["r_id"].astype(str))["mgs_level"].to_dict()
colors = {1: "#4e79a7", 2: "#f28e2b", 3: "#59a14f", 4: "#e15759"}
point_colors = [colors.get(mgs_map.get(sid, 1), "grey") for sid in sample_ids]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: before removal
ax = axes[0]
for i, (x, y, c, sid) in enumerate(zip(pc1, pc2, point_colors, sample_ids)):
    ax.scatter(x, y, c=c, s=18, alpha=0.7, edgecolors="none")
    if outlier_mask[i]:
        ax.scatter(x, y, s=90, facecolors="none", edgecolors="red", linewidths=1.5)
        ax.annotate(sid, (x, y), fontsize=6, color="red",
                    xytext=(4, 4), textcoords="offset points")
ax.set_xlabel(f"PC1 ({explained[0]:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]:.1f}%)")
ax.set_title("PCA – before outlier removal")
patches = [mpatches.Patch(color=v, label=f"MGS {k}") for k, v in colors.items()]
patches.append(mpatches.Patch(color="none", label="○ = outlier",
                               linewidth=1.5, edgecolor="red"))
ax.legend(handles=patches, fontsize=8)

# ── 8. Remove outliers ────────────────────────────────────────────────────────
counts_clean = counts.drop(columns=outlier_ids)
meta_clean = meta_aligned[~meta_aligned["r_id"].astype(str).isin(outlier_ids)].copy()
print(f"\nAfter PCA outlier removal : {counts_clean.shape[1]} samples remain")

# Right: after removal
log_counts2 = np.log1p(counts_clean.T.values)
X2 = scaler.fit_transform(log_counts2)
pcs2 = pca.fit_transform(X2)
ev2 = pca.explained_variance_ratio_ * 100
sample_ids2 = list(counts_clean.columns)
mgs_colors2 = [colors.get(mgs_map.get(sid, 1), "grey") for sid in sample_ids2]

ax2 = axes[1]
ax2.scatter(pcs2[:, 0], pcs2[:, 1], c=mgs_colors2, s=18, alpha=0.7, edgecolors="none")
ax2.set_xlabel(f"PC1 ({ev2[0]:.1f}%)")
ax2.set_ylabel(f"PC2 ({ev2[1]:.1f}%)")
ax2.set_title("PCA – after outlier removal")
ax2.legend(handles=patches[:-1], fontsize=8)

plt.tight_layout()
plt.savefig("PCA_sample_QC.png", dpi=150)
print("PCA plot saved → PCA_sample_QC.png")

# ── 9. Save outputs ──────────────────────────────────────────────────────────
counts_clean.to_csv("Genelevel_QC_counts.tsv", sep="\t")
meta_clean.to_csv("MetaSheet_QC.csv")
print(f"\nFinal matrix saved → Genelevel_QC_counts.tsv"
      f"  ({counts_clean.shape[0]} genes × {counts_clean.shape[1]} samples)")
print(f"Final metadata saved → MetaSheet_QC.csv  ({len(meta_clean)} samples)")

# ── 10. Summary ──────────────────────────────────────────────────────────────
print("\n══════════ Sample QC Summary ══════════")
print(f"  Original samples        : {n_before}")
print(f"  Removed (RIN ≤ 5)       : {n_before - meta.shape[0] - missing_mgs - missing_geno}")
print(f"  Removed (miss mgs_level): {missing_mgs}")
print(f"  Removed (miss genotype) : {missing_geno}")
print(f"  Removed (PCA outliers)  : {len(outlier_ids)}")
print(f"  Final usable samples    : {counts_clean.shape[1]}")
