"""
Week 2 · Task 1 — Differential Gene Expression Analysis (DESeq2 via PyDESeq2)
=============================================================================
Inputs  (from Week 1):
  ../../week1/task2_sample_qc/Genelevel_QC_counts.tsv  — integer counts, QC-passed
  ../../week1/task2_sample_qc/MetaSheet_QC.csv          — sample metadata
  ../../gene_info.tsv                                    — Ensembl → gene symbol

Design formula: ~ RIN + PMI + Age + Sex + MGS_level
  · Continuous covariates : rin, postmortem_interval_hrs, age
  · Categorical factors   : sex (ref = F), mgs_level (ref = 1)

Contrasts (all vs MGS1):
  MGS4 vs MGS1  —  advanced AMD vs healthy control
  MGS3 vs MGS1
  MGS2 vs MGS1

Outputs:
  DEG_full_results.tsv       — all genes, all contrasts (wide format)
  DEG_MGS4_vs_MGS1.tsv       — significant DEGs for the primary contrast
  DEG_MGS3_vs_MGS1.tsv
  DEG_MGS2_vs_MGS1.tsv
  dge_summary.txt            — run statistics
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# ── 0. Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
COUNTS_PATH = os.path.join(ROOT, "week1/task2_sample_qc/Genelevel_QC_counts.tsv")
META_PATH   = os.path.join(ROOT, "week1/task2_sample_qc/MetaSheet_QC.csv")
GENE_PATH   = os.path.join(ROOT, "gene_info.tsv")
OUT_DIR     = SCRIPT_DIR

# ── 1. Load raw integer counts (genes × samples) ─────────────────────────────
print("── 1. Loading data ──────────────────────────────────────────────────")
counts_raw = pd.read_csv(COUNTS_PATH, sep="\t", index_col=0)
print(f"  Counts matrix (raw)  : {counts_raw.shape[0]:,} genes × {counts_raw.shape[1]:,} samples")

# ── 2. Load metadata ──────────────────────────────────────────────────────────
meta_raw = pd.read_csv(META_PATH, index_col=0, encoding="latin-1")
print(f"  Metadata samples     : {len(meta_raw):,}")

# ── 3. Align samples ──────────────────────────────────────────────────────────
print("\n── 2. Aligning samples ──────────────────────────────────────────────")
# counts columns  = r_id strings (e.g. "298_3")
# metadata r_id column is the shared key
meta_raw["r_id"] = meta_raw["r_id"].astype(str)
common = sorted(set(counts_raw.columns) & set(meta_raw["r_id"]))
print(f"  Common samples       : {len(common):,}")

counts_aligned = counts_raw[common].copy()          # genes × samples
meta_aligned   = meta_raw.set_index("r_id").loc[common].copy()  # samples × meta

# PyDESeq2 expects  counts  : samples × genes
counts_T = counts_aligned.T.copy()
assert list(counts_T.index) == list(meta_aligned.index), \
    "Sample order mismatch between counts and metadata!"

# ── 4. Prepare covariate columns ──────────────────────────────────────────────
print("\n── 3. Preparing design covariates ───────────────────────────────────")
covariates = ["rin", "postmortem_interval_hrs", "age", "sex", "mgs_level"]
meta_design = meta_aligned[covariates].copy()

# Encode categorical factors as strings
meta_design["mgs_level"] = meta_design["mgs_level"].astype(str)
meta_design["sex"]        = meta_design["sex"].astype(str)

# Drop samples with any missing covariate
n_before = len(meta_design)
meta_design = meta_design.dropna(subset=covariates)
counts_T    = counts_T.loc[meta_design.index]
n_after = len(meta_design)
if n_before > n_after:
    print(f"  Dropped {n_before - n_after} samples with missing covariates")

print(f"  Final design matrix  : {n_after} samples × {len(covariates)} covariates")
print("  MGS distribution:")
for lvl, cnt in meta_design["mgs_level"].value_counts().sort_index().items():
    print(f"    MGS{lvl} : {cnt}")

# ── 5. Build DeseqDataSet ─────────────────────────────────────────────────────
print("\n── 4. Building DeseqDataSet ──────────────────────────────────────────")
dds = DeseqDataSet(
    counts          = counts_T,
    metadata        = meta_design,
    design_factors  = covariates,
    continuous_factors = ["rin", "postmortem_interval_hrs", "age"],
    ref_level       = [["mgs_level", "1"], ["sex", "F"]],
    refit_cooks     = True,
    quiet           = False,
)

# ── 6. Run DESeq2 (estimate dispersions + fit GLM) ───────────────────────────
print("\n── 5. Running DESeq2 (dispersion estimation + GLM fitting) ──────────")
dds.deseq2()
print("  DESeq2 fitting complete.")

# ── 7. Load gene annotation ───────────────────────────────────────────────────
gene_info = pd.read_csv(GENE_PATH, sep="\t", usecols=["ensembl_gene_id",
                                                        "external_gene_name",
                                                        "gene_biotype"])
gene_info = gene_info.rename(columns={
    "ensembl_gene_id"   : "ensembl_id",
    "external_gene_name": "gene_symbol",
})
id2sym = gene_info.set_index("ensembl_id")["gene_symbol"].to_dict()
id2bio = gene_info.set_index("ensembl_id")["gene_biotype"].to_dict()

# ── 8. Extract results for each contrast ─────────────────────────────────────
print("\n── 6. Extracting contrast results ────────────────────────────────────")

CONTRASTS = [
    ("mgs_level", "4", "1"),
    ("mgs_level", "3", "1"),
    ("mgs_level", "2", "1"),
]

all_results = {}   # keyed by comparison label

for factor, test, ref in CONTRASTS:
    label = f"MGS{test}_vs_MGS{ref}"
    print(f"\n  Running: {label}")

    stat = DeseqStats(
        dds,
        contrast     = [factor, test, ref],
        alpha        = 0.05,
        cooks_filter = True,
        independent_filter = True,
    )
    stat.summary()

    res = stat.results_df.copy()
    res.index.name = "ensembl_id"
    res = res.reset_index()

    # Annotate gene symbols
    res["gene_symbol"]  = res["ensembl_id"].map(id2sym).fillna("")
    res["gene_biotype"] = res["ensembl_id"].map(id2bio).fillna("")
    res["comparison"]   = label

    # Reorder columns for readability
    res = res[["ensembl_id", "gene_symbol", "gene_biotype", "comparison",
               "baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj"]]

    # Classify significance
    # Primary threshold: padj < 0.05  (AMD effects are typically < 2-fold)
    # Strict  threshold: padj < 0.05 AND |log2FC| >= 1  (annotated separately)
    res["significant"]        = res["padj"] < 0.05
    res["significant_strict"] = (res["padj"] < 0.05) & (res["log2FoldChange"].abs() >= 1)

    n_sig        = res["significant"].sum()
    n_sig_strict = res["significant_strict"].sum()
    n_up   = ((res["significant"]) & (res["log2FoldChange"] > 0)).sum()
    n_down = ((res["significant"]) & (res["log2FoldChange"] < 0)).sum()
    n_total = len(res)

    print(f"    Total genes tested    : {n_total:,}")
    print(f"    Significant (padj<0.05)          : {n_sig:,}  (up={n_up}  down={n_down})")
    print(f"    Significant strict (+|log2FC|>=1): {n_sig_strict:,}")

    all_results[label] = res

    # Save per-contrast significant DEG table (padj < 0.05)
    sig_path = os.path.join(OUT_DIR, f"DEG_{label}.tsv")
    res_sig  = res[res["significant"]].sort_values("padj")
    res_sig.to_csv(sig_path, sep="\t", index=False)
    print(f"    Saved → {os.path.basename(sig_path)}  ({len(res_sig)} genes)")

# ── 9. Build combined full-results table ─────────────────────────────────────
print("\n── 7. Building combined results table ────────────────────────────────")
full = pd.concat(all_results.values(), ignore_index=True)

full_path = os.path.join(OUT_DIR, "DEG_full_results.tsv")
full.to_csv(full_path, sep="\t", index=False)
print(f"  Saved → DEG_full_results.tsv  ({len(full):,} rows)")

# ── 10. Summary report ────────────────────────────────────────────────────────
summary_lines = [
    "═══════════════════════════════════════════════════════════",
    "  Week 2 Task 1 — DGE Analysis Summary",
    "═══════════════════════════════════════════════════════════",
    f"  Samples analysed  : {n_after}",
    f"  Genes tested      : {counts_T.shape[1]:,}",
    f"  Design formula    : ~ RIN + PMI + Age + Sex + MGS_level",
    f"  Reference group   : MGS1",
    "",
    "  Significance threshold  : padj < 0.05 (primary); padj<0.05 & |log2FC|>=1 (strict)",
    "",
]

for label, res in all_results.items():
    n_sig        = res["significant"].sum()
    n_sig_strict = res["significant_strict"].sum()
    n_up   = ((res["significant"]) & (res["log2FoldChange"] > 0)).sum()
    n_down = ((res["significant"]) & (res["log2FoldChange"] < 0)).sum()
    summary_lines += [
        f"  {label}",
        f"    padj<0.05          : {n_sig:,}  (up={n_up}  down={n_down})",
        f"    padj<0.05+|FC|>=1  : {n_sig_strict:,}",
    ]

summary_lines += [
    "",
    "  Outputs:",
    "    DEG_full_results.tsv       — all genes, all contrasts",
    "    DEG_MGS4_vs_MGS1.tsv       — primary contrast significant DEGs",
    "    DEG_MGS3_vs_MGS1.tsv",
    "    DEG_MGS2_vs_MGS1.tsv",
    "═══════════════════════════════════════════════════════════",
]

summary_text = "\n".join(summary_lines)
print("\n" + summary_text)

summary_path = os.path.join(OUT_DIR, "dge_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_text + "\n")
print(f"\nSummary saved → dge_summary.txt")
