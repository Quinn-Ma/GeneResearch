import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pydeseq2.dds import DeseqDataSet

# ── 1. Load QC-passed counts (genes × samples) → transpose to samples × genes ──
print("Loading data...")
counts_raw = pd.read_csv("Genelevel_QC_counts.tsv", sep="\t", index_col=0)
counts_t = counts_raw.T                    # samples × genes
counts_t.index.name = "sample_id"
print(f"Counts matrix             : {counts_t.shape[0]} samples × {counts_t.shape[1]} genes")

# ── 2. Load QC-passed metadata ───────────────────────────────────────────────
meta = pd.read_csv("MetaSheet_QC.csv", index_col=0, encoding="latin-1")
meta = meta.set_index("r_id")             # use r_id as row index
meta.index.name = "sample_id"

# ── 3. Extract & clean covariates ────────────────────────────────────────────
cov_cols = {
    "rin"                    : "rin",
    "postmortem_interval_hrs": "pmi",
    "age"                    : "age",
    "sex"                    : "sex",
    "mgs_level"              : "mgs_level",
}
meta_cov = meta[list(cov_cols.keys())].rename(columns=cov_cols).copy()

# sex: encode as string factor (pydeseq2 needs string categories)
meta_cov["sex"] = meta_cov["sex"].astype(str)

# mgs_level: string factor so DESeq2 treats it as categorical
meta_cov["mgs_level"] = meta_cov["mgs_level"].astype(str)

# ── 4. Align samples between counts & metadata ───────────────────────────────
common = counts_t.index.intersection(meta_cov.index)
counts_aligned = counts_t.loc[common].astype(int)
meta_aligned   = meta_cov.loc[common]
print(f"Samples after alignment   : {len(common)}")

# Sanity-check: no negative counts
assert (counts_aligned >= 0).all().all(), "Negative counts detected!"

# ── 5. Build DeseqDataSet with full design ───────────────────────────────────
#   ~ rin + pmi + age + sex + mgs_level
#   rin / pmi / age → continuous   |   sex / mgs_level → categorical
print("\nBuilding DESeq2 dataset...")
dds = DeseqDataSet(
    counts=counts_aligned,
    metadata=meta_aligned,
    design="~ rin + pmi + age + sex + mgs_level",
    continuous_factors=["rin", "pmi", "age"],
    ref_level=[("sex", "F"), ("mgs_level", "1")],
    refit_cooks=True,
    quiet=False,
)

# ── 6. Run DESeq2 (estimate size factors + dispersions + GLM fit) ────────────
print("\nRunning DESeq2 pipeline...")
dds.deseq2()

# ── 7. VST normalisation with full design (strips confounder signal) ──────────
print("\nApplying VST normalisation (use_design=True)...")
dds.vst(use_design=True)

# Extract VST matrix from AnnData: shape (samples × genes) → transpose to genes × samples
vst_array  = dds.layers["vst_counts"]             # samples × genes numpy array
vst_df     = pd.DataFrame(
    vst_array,
    index=counts_aligned.index,
    columns=counts_aligned.columns,
).T                                                # genes × samples

print(f"\nVST matrix shape          : {vst_df.shape[0]} genes × {vst_df.shape[1]} samples")

# ── 8. Save VST matrix ────────────────────────────────────────────────────────
out_path = "Genelevel_VST_corrected.tsv"
vst_df.to_csv(out_path, sep="\t")
print(f"VST matrix saved → {out_path}")

# ── 9. PCA before vs after correction (visual sanity-check) ──────────────────
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

colors = {"1": "#4e79a7", "2": "#f28e2b", "3": "#59a14f", "4": "#e15759"}
mgs_color = {sid: colors[mg]
             for sid, mg in meta_aligned["mgs_level"].items()}

def quick_pca(mat_samples_genes, title, ax):
    """mat_samples_genes : samples × genes numpy array"""
    X = StandardScaler().fit_transform(mat_samples_genes)
    pcs = PCA(n_components=2, random_state=42).fit_transform(X)
    ev  = PCA(n_components=2, random_state=42).fit(X).explained_variance_ratio_ * 100
    c   = [mgs_color.get(s, "grey") for s in counts_aligned.index]
    ax.scatter(pcs[:, 0], pcs[:, 1], c=c, s=16, alpha=0.7, edgecolors="none")
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before correction: log1p of raw integer counts
raw_log = np.log1p(counts_aligned.values.astype(float))
quick_pca(raw_log, "Before confounder correction\n(log1p raw counts)", axes[0])

# After correction: VST values (samples × genes)
quick_pca(vst_array, "After confounder correction\n(VST-normalised)", axes[1])

patches = [mpatches.Patch(color=v, label=f"MGS {k}") for k, v in colors.items()]
for ax in axes:
    ax.legend(handles=patches, fontsize=9)

plt.tight_layout()
plt.savefig("PCA_confounder_correction.png", dpi=150)
print("PCA plot saved → PCA_confounder_correction.png")

# ── 10. Summary ──────────────────────────────────────────────────────────────
print("\n══════════ Confounder Correction Summary ══════════")
print(f"  Design formula          : ~ rin + pmi + age + sex + mgs_level")
print(f"  Continuous covariates   : rin, pmi (postmortem_interval_hrs), age")
print(f"  Categorical covariates  : sex (ref=F), mgs_level (ref=1)")
print(f"  Samples                 : {vst_df.shape[1]}")
print(f"  Genes                   : {vst_df.shape[0]}")
print(f"  Output                  : {out_path}  (float, VST scale)")
