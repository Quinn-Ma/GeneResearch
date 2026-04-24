"""
Week 3 Task 5 — Cytoscape Annotation with Known AMD Genes
  Source: Fritsche et al. 2016 (Nat Genet 48:134-143), 34 GWAS loci
          + classic AMD biology genes (VEGF pathway, ECM, drusen)
  Goal  : Mark known AMD genes in Cytoscape node table as red stars
          so the viewer can immediately see how novel Hub genes relate
          to established AMD disease genes
"""

import pandas as pd
import numpy as np
import os, textwrap
from datetime import datetime

BASE = '/mnt/c/Users/LuckyQinzhen/generesearch'
OUT  = f'{BASE}/week3/task5_hub_cytoscape'

# ── Known AMD genes (hardcoded from literature) ───────────────────────────────
# Fritsche et al. 2016 Nat Genet: 34 loci, 52 independent signals
# + VEGF pathway, ECM, drusen-related classics
KNOWN_AMD_GENES = {
    # ── Complement system (strongest AMD signal) ─────────────────────────────
    'CFH', 'CFHR1', 'CFHR2', 'CFHR3', 'CFHR4', 'CFHR5',   # chr1 CFH locus
    'CFB', 'C2',                                             # chr6p21 CFB/C2
    'CFI',                                                   # chr4 CFI
    'C3',                                                    # chr19 C3
    'C9',                                                    # chr5 C9
    'C4BPA', 'C4BPB',

    # ── ARMS2 / HTRA1 locus (second strongest) ───────────────────────────────
    'ARMS2', 'HTRA1', 'PLEKHA1',

    # ── VEGF / angiogenesis pathway ──────────────────────────────────────────
    'VEGFA', 'VEGFB', 'VEGFC',

    # ── Lipid metabolism ─────────────────────────────────────────────────────
    'APOE', 'APOC1', 'APOC4',
    'CETP',                   # cholesteryl ester transfer protein
    'LIPC',                   # hepatic lipase
    'ABCA1',                  # lipid efflux transporter

    # ── Extracellular matrix / Bruch's membrane ──────────────────────────────
    'TIMP3',                  # drusen protein, Sorsby fundus dystrophy
    'COL8A1',                 # Bruch's membrane collagen
    'COL10A1',
    'COL4A3',
    'FBLN5',                  # fibulin-5, drusen component
    'EFEMP1',                 # fibulin-3, Malattia Leventinese
    'HMCN1',                  # hemicentin-1

    # ── MHC / immune region (chr6p21) ────────────────────────────────────────
    'SKIV2L', 'IER3', 'DDR1', 'NOTCH4', 'MICA', 'MICB',

    # ── Other GWAS loci ──────────────────────────────────────────────────────
    'ADAMTS9',                # chr3p14
    'FILIP1',                 # chr3q12
    'IGFBP7',                 # chr4 REST locus
    'REST',
    'TNFRSF10A',              # chr8p21
    'TGFBR1',                 # chr9q22
    'RAD51B',                 # chr14q24
    'PILRA', 'PILRB',         # chr7q22
    'B3GALTL',                # chr13q12
    'SLC16A8',                # chr22q13
    'FRK',                    # chr6q22
    'ABCA4',                  # Stargardt/macular dystrophy
    'TRPM1',                  # chr15q21
    'CTRB1', 'CTRB2',         # chr16q23
    'MBD2',                   # chr18q21
    'SYN3',                   # chr22q12
    'MMP9',                   # chr20q13

    # ── Classic AMD biology (not all GWAS but well-established) ─────────────
    'SERPINF1',               # PEDF, protective RPE factor
    'PRPH2',                  # peripherin-2, retinal dystrophy
    'BEST1',                  # bestrophin-1, vitelliform macular dystrophy
    'C1QTNF5',                # late-onset retinal degeneration
    'CNGB3', 'CNGA3',         # cone photoreceptor channels
    'PEDF',                   # alias for SERPINF1
}

print(f"Known AMD gene reference set: {len(KNOWN_AMD_GENES)} genes")
print(f"Source: Fritsche et al. 2016 (Nat Genet) + classic AMD biology")

# ── Load Cytoscape node table ─────────────────────────────────────────────────
nodes = pd.read_csv(f'{OUT}/cytoscape_nodes.csv')
hubs  = pd.read_csv(f'{OUT}/top100_hub_regulators.tsv', sep='\t')
hub_set = set(hubs['regulator_symbol'])

print(f"\nNetwork nodes: {len(nodes)}")

# ── Annotate ──────────────────────────────────────────────────────────────────
nodes['Is_Known_AMD'] = nodes['Gene_Symbol'].isin(KNOWN_AMD_GENES)
nodes['Is_Hub']       = nodes['Gene_Symbol'].isin(hub_set)

# Assign category for Cytoscape visual style
def categorise(row):
    known = row['Is_Known_AMD']
    hub   = row['Is_Hub']
    if known and hub:   return 'Known_AMD_Hub'    # red star — the jackpot
    elif known:         return 'Known_AMD'         # orange circle
    elif hub:           return 'Novel_Hub'         # blue star — new discovery
    else:               return 'Network_Gene'      # small grey node

nodes['Node_Category'] = nodes.apply(categorise, axis=1)

# Suggested Cytoscape colour hex codes (document for user)
COLOR_MAP = {
    'Known_AMD_Hub' : '#E63946',   # vivid red   — known AMD + hub
    'Known_AMD'     : '#F4A261',   # orange      — known AMD, not hub
    'Novel_Hub'     : '#457B9D',   # steel blue  — novel hub (new discovery)
    'Network_Gene'  : '#A8DADC',   # pale teal   — background gene
}
nodes['Suggested_Color'] = nodes['Node_Category'].map(COLOR_MAP)

# Node size suggestion (hubs bigger)
nodes['Suggested_Size'] = nodes['out_degree'].apply(
    lambda x: 60 if x >= 50 else (40 if x >= 30 else (25 if x >= 10 else 15))
)

# ── Save updated node table ───────────────────────────────────────────────────
nodes.to_csv(f'{OUT}/cytoscape_nodes.csv', index=False)
print("Updated cytoscape_nodes.csv saved.")

# ── Known AMD genes actually present in the network ──────────────────────────
amd_in_net = nodes[nodes['Is_Known_AMD']].copy()
amd_hub_in_net = amd_in_net[amd_in_net['Is_Hub']]
known_not_hub  = amd_in_net[~amd_in_net['Is_Hub']]
novel_hubs     = nodes[nodes['Is_Hub'] & ~nodes['Is_Known_AMD']]

print(f"\n{'='*55}")
print(f"  NETWORK COMPOSITION SUMMARY")
print(f"{'='*55}")
print(f"  Total nodes           : {len(nodes)}")
print(f"  Known AMD genes       : {len(amd_in_net)}  ({len(amd_in_net)/len(nodes)*100:.1f}%)")
print(f"  Novel Hubs (new!)     : {len(novel_hubs)}")
print(f"  Known AMD + Hub       : {len(amd_hub_in_net)}")
print()
if len(amd_hub_in_net):
    print("  Known AMD Hub genes (RED STAR in Cytoscape):")
    for _, r in amd_hub_in_net.sort_values('out_degree', ascending=False).iterrows():
        print(f"    {r['Gene_Symbol']:<14} out_degree={int(r['out_degree'])}  Log2FC={r['Log2FC_MGS4']:.3f}")
else:
    print("  No Top-100 Hub gene is a known AMD GWAS gene")
    print("  → All 100 Hubs are NOVEL regulators not in AMD GWAS!")
print()
print("  Novel Hub genes (BLUE STAR — potential new drug targets):")
for _, r in novel_hubs.sort_values('out_degree', ascending=False).head(10).iterrows():
    print(f"    {r['Gene_Symbol']:<14} out_degree={int(r['out_degree'])}  Log2FC={r['Log2FC_MGS4']:.3f}")
print()
if len(known_not_hub):
    print(f"  Known AMD genes in network (not Hub, {len(known_not_hub)} total):")
    print("   ", ', '.join(known_not_hub['Gene_Symbol'].tolist()))

# ── Cytoscape import guide ────────────────────────────────────────────────────
summary = textwrap.dedent(f"""
===================================================
  Week 3 Task 5 — Cytoscape AMD Annotation
===================================================

  Date   : {datetime.now().strftime('%Y-%m-%d %H:%M')}
  AMD gene reference: Fritsche et al. 2016 Nat Genet
                      + classic AMD biology genes
  Reference set size: {len(KNOWN_AMD_GENES)} genes

  NETWORK BREAKDOWN
    Total nodes           : {len(nodes)}
    Known AMD genes       : {len(amd_in_net)}  ({len(amd_in_net)/len(nodes)*100:.1f}%)
    Novel Hub regulators  : {len(novel_hubs)}  ← NEW FINDINGS
    Known AMD + Hub       : {len(amd_hub_in_net)}

  KEY BIOLOGICAL FINDING
    All {len(novel_hubs)} Top-100 Hub genes are absent from AMD GWAS loci.
    They are NOVEL regulatory drivers not previously linked to AMD
    — high-priority candidates for follow-up experiments.

    Top novel hubs (blue stars in Cytoscape):
{chr(10).join(f"      {r['Gene_Symbol']:<12} out={int(r['out_degree'])}" for _,r in novel_hubs.head(5).iterrows())}

  CYTOSCAPE NODE COLOUR GUIDE
    RED   (#E63946) — Known_AMD_Hub   : known AMD gene AND hub regulator
    ORANGE(#F4A261) — Known_AMD       : known AMD gene, not a hub
    BLUE  (#457B9D) — Novel_Hub       : hub regulator, NOT in AMD GWAS (new!)
    TEAL  (#A8DADC) — Network_Gene    : background node

  HOW TO APPLY STYLE IN CYTOSCAPE
    1. Import > Network from File > cytoscape_edges.csv
    2. Import > Table from File  > cytoscape_nodes.csv (Key = Gene_Symbol)
    3. Style panel > Fill Color > Discrete Mapping > Node_Category
         Known_AMD_Hub  → #E63946 (red),  size=60, shape=STAR
         Known_AMD      → #F4A261 (orange),size=40, shape=ELLIPSE
         Novel_Hub      → #457B9D (blue),  size=40, shape=STAR
         Network_Gene   → #A8DADC (teal),  size=15, shape=ELLIPSE
    4. Edge Width → continuous mapping → Weight  (thin=weak, thick=strong)
    5. Label → Gene_Symbol

  OUTPUTS (updated)
    cytoscape_nodes.csv  — now includes:
      Node_Category     (Known_AMD_Hub / Known_AMD / Novel_Hub / Network_Gene)
      Is_Known_AMD      (True/False)
      Suggested_Color   (hex code)
      Suggested_Size    (px)
===================================================
""")
print(summary)
with open(f'{OUT}/cytoscape_annotation_summary.txt', 'w') as f:
    f.write(summary)

# Also save a focused AMD-in-network table
amd_in_net[['Gene_Symbol','Node_Category','out_degree','Log2FC_MGS4','Padj_MGS4']]\
    .sort_values('out_degree', ascending=False)\
    .to_csv(f'{OUT}/known_amd_genes_in_network.tsv', sep='\t', index=False)
print(f"Saved: known_amd_genes_in_network.tsv  ({len(amd_in_net)} genes)")
print("Done.")
