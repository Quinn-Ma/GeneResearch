import pandas as pd

# ── 1. Load gene annotation & keep protein_coding only ──────────────────────
gene_info = pd.read_csv("gene_info.tsv", sep="\t")
protein_coding_ids = set(
    gene_info.loc[gene_info["gene_biotype"] == "protein_coding", "ensembl_gene_id"]
)
print(f"protein_coding genes in gene_info : {len(protein_coding_ids):,}")

# ── 2. Load raw expected counts matrix ──────────────────────────────────────
counts = pd.read_csv(
    "Genelevel_expectedcounts_matrix.tsv",
    sep="\t",
    index_col=0,
)
print(f"Genes in counts matrix (raw)       : {counts.shape[0]:,}")
print(f"Samples in counts matrix           : {counts.shape[1]:,}")

# ── 3. Step A – filter to protein_coding genes ──────────────────────────────
keep_ids = protein_coding_ids & set(counts.index)
counts_pc = counts.loc[counts.index.isin(keep_ids)].copy()
print(f"Genes after protein_coding filter  : {counts_pc.shape[0]:,}")

# ── 4. Step B – apply round() to convert decimal counts to integers ─────────
counts_pc = counts_pc.round(0).astype(int)

# ── 5. Step C – remove low-expression genes (>80 % samples with count < 10) ─
n_samples = counts_pc.shape[1]
low_expr_frac = (counts_pc < 10).sum(axis=1) / n_samples
counts_filtered = counts_pc.loc[low_expr_frac <= 0.80].copy()
print(f"Genes after low-expression filter  : {counts_filtered.shape[0]:,}")
print(f"  (removed {counts_pc.shape[0] - counts_filtered.shape[0]:,} genes"
      f" with >80 % samples having count < 10)")

# ── 6. Save result ───────────────────────────────────────────────────────────
out_path = "Genelevel_filtered_counts.tsv"
counts_filtered.to_csv(out_path, sep="\t")
print(f"\nSaved → {out_path}  ({counts_filtered.shape[0]:,} genes × {counts_filtered.shape[1]:,} samples)")
