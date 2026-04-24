"""
Week 4 Task 5 — Pathway Enrichment Analysis
  Input : top-50 LIME genes (most frequent biomarkers across patients)
  Method: Enrichr API via gseapy (KEGG 2021 + GO Biological Process 2021)
  Goal  : Show AMD-relevant pathways: Complement, Lipid metabolism, Immune
"""

import pandas as pd
import numpy as np
import os, textwrap
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import gseapy as gp

BASE = '/mnt/c/Users/LuckyQinzhen/generesearch'
D3   = f'{BASE}/week4/task3_lime'
D2   = f'{BASE}/week4/task2_tabnet'
OUT  = f'{BASE}/week4/task4_pathway'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load top-50 LIME genes ─────────────────────────────────────────────────
print("Loading top-50 LIME genes …")
lime_genes = pd.read_csv(f'{D3}/top50_lime_genes.tsv', sep='\t')
gene_list  = lime_genes['gene_symbol'].tolist()
print(f"  Input gene list: {gene_list[:10]} … ({len(gene_list)} total)")

# Also load TabNet top-50 for comparison
tabnet_fi = pd.read_csv(f'{D2}/feature_importances.tsv', sep='\t')
tabnet_top50 = tabnet_fi.head(50)['gene_symbol'].tolist()

# ── 2. Enrichr enrichment (LIME top-50) ──────────────────────────────────────
GENE_SETS = {
    'KEGG_2021_Human'            : 'KEGG',
    'GO_Biological_Process_2021' : 'GO_BP',
}

results = {}
for gs_name, label in GENE_SETS.items():
    print(f"\nRunning Enrichr: {gs_name} …")
    try:
        enr = gp.enrichr(
            gene_list  = gene_list,
            gene_sets  = [gs_name],
            organism   = 'human',
            outdir     = f'{OUT}/{label}',
            cutoff     = 0.2,      # relaxed; we filter below
            no_plot    = True,
        )
        df = enr.results.copy()
        df = df.sort_values('Adjusted P-value').reset_index(drop=True)
        df.to_csv(f'{OUT}/{label}_results.tsv', sep='\t', index=False)
        results[label] = df
        sig = df[df['Adjusted P-value'] < 0.05]
        print(f"  Significant terms (padj<0.05): {len(sig)}")
        if len(sig) == 0:
            sig = df.head(10)   # show top-10 if nothing passes strict cutoff
            print(f"  (relaxed: showing top-10 by p-value)")
        print(sig[['Term','Overlap','P-value','Adjusted P-value']].head(10).to_string())
    except Exception as e:
        print(f"  WARNING: {gs_name} failed — {e}")
        results[label] = pd.DataFrame()

# ── 3. Combined barplot ───────────────────────────────────────────────────────
print("\nPlotting enrichment results …")

def plot_enrichment(df, title, out_path, top_n=20, p_col='Adjusted P-value'):
    if df.empty:
        print(f"  No results for {title}, skipping plot.")
        return
    df_plot = df.copy()
    df_plot['-log10(padj)'] = -np.log10(df_plot[p_col].clip(1e-10))
    # Highlight AMD-relevant keywords
    amd_kws = ['complement', 'lipid', 'immune', 'inflammation', 'oxidative',
               'fatty acid', 'cholesterol', 'phagocytosis', 'apoptosis',
               'macular', 'retina', 'photoreceptor', 'autophagy', 'lysosome']
    df_plot['is_amd'] = df_plot['Term'].str.lower().apply(
        lambda t: any(k in t for k in amd_kws))
    top = df_plot.head(top_n)
    colors = ['#e15759' if v else '#4e79a7' for v in top['is_amd']]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    bars = ax.barh(range(len(top)), top['-log10(padj)'][::-1].values,
                   color=colors[::-1], edgecolor='white')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top['Term'][::-1].values, fontsize=8)
    ax.axvline(-np.log10(0.05), color='gray', ls='--', lw=1, label='padj=0.05')
    ax.set_xlabel('-log10(Adjusted P-value)', fontsize=11)
    ax.set_title(f'{title}\n(Red = AMD-relevant pathway)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    # Add gene overlap annotation
    for i, (_, row) in enumerate(top.iloc[::-1].iterrows()):
        ax.text(0.05, i, row['Overlap'], va='center', fontsize=6, color='white',
                fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(out_path)}")

plot_enrichment(results.get('KEGG', pd.DataFrame()),
                'KEGG Pathway Enrichment (LIME Top-50 Genes)',
                f'{OUT}/kegg_barplot.png')
plot_enrichment(results.get('GO_BP', pd.DataFrame()),
                'GO Biological Process Enrichment (LIME Top-50 Genes)',
                f'{OUT}/go_bp_barplot.png')

# ── 4. LIME vs TabNet gene overlap ────────────────────────────────────────────
lime_set   = set(gene_list)
tabnet_set = set(tabnet_top50)
overlap    = lime_set & tabnet_set
unique_lime   = lime_set - tabnet_set
unique_tabnet = tabnet_set - lime_set

print(f"\n  LIME top-50 ∩ TabNet top-50 : {len(overlap)} genes")
print(f"  LIME only  : {len(unique_lime)}")
print(f"  TabNet only: {len(unique_tabnet)}")
print(f"  Shared genes: {sorted(overlap)[:10]} …")

overlap_df = pd.DataFrame({
    'gene_symbol'    : sorted(overlap),
    'in_lime_top50'  : True,
    'in_tabnet_top50': True,
})
overlap_df.to_csv(f'{OUT}/lime_tabnet_overlap.tsv', sep='\t', index=False)

# ── 5. AMD pathway summary ────────────────────────────────────────────────────
def get_amd_hits(df, label):
    if df.empty:
        return "  (no results)"
    amd_kws = ['complement', 'lipid', 'immune', 'inflamm', 'oxidative',
               'fatty acid', 'cholesterol', 'phagocytos', 'macular',
               'retina', 'autophagy', 'lysosome', 'apoptos']
    hits = df[df['Term'].str.lower().apply(lambda t: any(k in t for k in amd_kws))]
    if hits.empty:
        return f"  None directly matching AMD keywords (top term: {df.iloc[0]['Term']})"
    lines = []
    for _, r in hits.head(5).iterrows():
        lines.append(f"  {r['Term'][:60]:<62}  padj={r['Adjusted P-value']:.3e}")
    return '\n'.join(lines)

kegg_amd = get_amd_hits(results.get('KEGG', pd.DataFrame()), 'KEGG')
go_amd   = get_amd_hits(results.get('GO_BP', pd.DataFrame()), 'GO_BP')

kegg_n  = len(results.get('KEGG',  pd.DataFrame()))
go_n    = len(results.get('GO_BP', pd.DataFrame()))
kegg_sig = len(results.get('KEGG', pd.DataFrame()).query('`Adjusted P-value` < 0.05')) if kegg_n else 0
go_sig   = len(results.get('GO_BP', pd.DataFrame()).query('`Adjusted P-value` < 0.05')) if go_n else 0

summary = textwrap.dedent(f"""\
===================================================
  Week 4 Task 5 -- Pathway Enrichment Analysis
===================================================

  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Input      : Top-50 LIME genes (most frequent patient biomarkers)
  Databases  : KEGG 2021 Human, GO Biological Process 2021

  KEGG RESULTS
    Total terms returned  : {kegg_n}
    Significant (padj<0.05): {kegg_sig}
    AMD-relevant hits:
{kegg_amd}

  GO BIOLOGICAL PROCESS RESULTS
    Total terms returned  : {go_n}
    Significant (padj<0.05): {go_sig}
    AMD-relevant hits:
{go_amd}

  GENE COVERAGE
    LIME top-50 ∩ TabNet top-50 : {len(overlap)} genes
    (convergence = high-confidence AMD biomarkers)
    Shared: {', '.join(sorted(overlap)[:15])}

  BIOLOGICAL INTERPRETATION
    If the enrichment points to Complement, Lipid metabolism,
    or Immune pathways, this validates that the TabNet+LIME
    pipeline has re-discovered the known AMD pathological
    mechanisms from raw expression data — completely de novo.

    Any novel lipid/immune sub-pathways not in the literature
    represent new hypotheses for follow-up wet-lab validation.

  OUTPUTS
    kegg_barplot.png        -- KEGG top-20 enrichment (AMD terms in red)
    go_bp_barplot.png       -- GO BP top-20 enrichment
    KEGG/kegg_results.tsv   -- full KEGG table
    GO_BP/go_results.tsv    -- full GO BP table
    lime_tabnet_overlap.tsv -- genes in both LIME and TabNet top-50
===================================================
""")
print(summary)
with open(f'{OUT}/pathway_summary.txt', 'w') as f:
    f.write(summary)

print("Task 5 (Pathway Analysis) complete.")
print("\n" + "="*55)
print("  WEEK 4 PIPELINE COMPLETE")
print("  TabNet + LIME + Pathway → AMD Precision Medicine")
print("="*55)
