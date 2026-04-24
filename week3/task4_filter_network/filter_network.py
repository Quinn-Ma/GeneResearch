"""
Week 3 Task 4 & 5 — Filter Network + Hub Identification + Cytoscape Export
"""

import pandas as pd
import numpy as np
import os, textwrap
from datetime import datetime

BASE = '/mnt/c/Users/LuckyQinzhen/generesearch'
T4_OUT  = f'{BASE}/week3/task4_filter_network'
T5_OUT  = f'{BASE}/week3/task5_hub_cytoscape'
os.makedirs(T4_OUT, exist_ok=True)
os.makedirs(T5_OUT, exist_ok=True)

# ─── Load inputs ────────────────────────────────────────────────────────────
print("Loading directed edge list …")
edges = pd.read_csv(f'{BASE}/week3/task2_infer_directions/directed_edge_list.tsv', sep='\t')
print(f"  Raw directed edges : {len(edges):,}")

deg_raw = pd.read_csv(f'{BASE}/week2/task1_dge/DEG_full_results.tsv', sep='\t')

# ────────────────────────────────────────────────────────────────────────────
# TASK 4 — Strategy B: top-10 regulators per target (by dominant_score)
# ────────────────────────────────────────────────────────────────────────────
print("\n[Task 4] Applying Strategy B filter (top-10 regulators per target) …")

filtered = (
    edges
    .sort_values('dominant_score', ascending=False)
    .groupby('target_ensembl', group_keys=False)
    .head(10)
    .reset_index(drop=True)
)
print(f"  Edges after filter : {len(filtered):,}")
print(f"  Unique regulators  : {filtered['regulator_ensembl'].nunique():,}")
print(f"  Unique targets     : {filtered['target_ensembl'].nunique():,}")

filtered.to_csv(f'{T4_OUT}/filtered_edge_list.tsv', sep='\t', index=False)

# Weight distribution
wt = filtered['dominant_score']
t4_summary = textwrap.dedent(f"""\
===================================================
  Week 3 Task 4 -- Filtered Edge List
===================================================

  STRATEGY      : B — top-10 regulators per target
  Source        : directed_edge_list.tsv (50 K edges)
  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}

  INPUT EDGES   : {len(edges):,}
  OUTPUT EDGES  : {len(filtered):,}  ({len(filtered)/len(edges)*100:.1f}% retained)

  WEIGHT STATS
    Min   : {wt.min():.6f}
    Max   : {wt.max():.6f}
    Mean  : {wt.mean():.6f}
    P25   : {wt.quantile(0.25):.6f}
    P50   : {wt.quantile(0.50):.6f}
    P75   : {wt.quantile(0.75):.6f}

  GRAPH NODES
    Unique regulators : {filtered['regulator_ensembl'].nunique():,}
    Unique targets    : {filtered['target_ensembl'].nunique():,}
    Total unique nodes: {pd.concat([filtered['regulator_ensembl'], filtered['target_ensembl']]).nunique():,}

  OUTPUTS
    filtered_edge_list.tsv  -- {len(filtered):,} high-confidence edges
===================================================
""")
print(t4_summary)
with open(f'{T4_OUT}/filter_summary.txt', 'w') as f:
    f.write(t4_summary)

# ────────────────────────────────────────────────────────────────────────────
# TASK 5a — Hub Regulators (out-degree in filtered network)
# ────────────────────────────────────────────────────────────────────────────
print("[Task 5a] Ranking hub regulators …")

hub_df = (
    filtered
    .groupby(['regulator_ensembl', 'regulator_symbol'])
    .agg(
        out_degree   = ('target_ensembl', 'nunique'),
        total_weight = ('dominant_score', 'sum'),
        mean_weight  = ('dominant_score', 'mean'),
    )
    .reset_index()
    .sort_values('out_degree', ascending=False)
    .reset_index(drop=True)
)
hub_df.insert(0, 'hub_rank', hub_df.index + 1)
top100 = hub_df.head(100).copy()

top100.to_csv(f'{T5_OUT}/top100_hub_regulators.tsv', sep='\t', index=False)
print(f"  Top hub: {top100.iloc[0]['regulator_symbol']}  out_degree={top100.iloc[0]['out_degree']}")

# ────────────────────────────────────────────────────────────────────────────
# TASK 5b — DEG Log2FC annotation (MGS4_vs_MGS1 = most severe comparison)
# ────────────────────────────────────────────────────────────────────────────
print("[Task 5b] Annotating nodes with DEG Log2FC (MGS4_vs_MGS1) …")

deg4 = (
    deg_raw[deg_raw['comparison'] == 'MGS4_vs_MGS1']
    [['ensembl_id', 'log2FoldChange', 'padj', 'significant']]
    .rename(columns={
        'log2FoldChange': 'Log2FC_MGS4',
        'padj'          : 'Padj_MGS4',
        'significant'   : 'Is_DEG_MGS4',
    })
)

# ────────────────────────────────────────────────────────────────────────────
# TASK 5c — Build Node table
# ────────────────────────────────────────────────────────────────────────────
print("[Task 5c] Building node attribute table …")

hub_set = set(top100['regulator_ensembl'])

reg_nodes = filtered[['regulator_ensembl', 'regulator_symbol']].rename(
    columns={'regulator_ensembl': 'Ensembl_ID', 'regulator_symbol': 'Gene_Symbol'})
tgt_nodes = filtered[['target_ensembl', 'target_symbol']].rename(
    columns={'target_ensembl': 'Ensembl_ID', 'target_symbol': 'Gene_Symbol'})
node_base = pd.concat([reg_nodes, tgt_nodes]).drop_duplicates('Ensembl_ID').reset_index(drop=True)

# Attach out-degree
node_base = node_base.merge(
    hub_df[['regulator_ensembl', 'out_degree']].rename(columns={'regulator_ensembl': 'Ensembl_ID'}),
    on='Ensembl_ID', how='left'
)
node_base['out_degree'] = node_base['out_degree'].fillna(0).astype(int)
node_base['Is_Hub']     = node_base['Ensembl_ID'].isin(hub_set)

# Attach DEG info
node_base = node_base.merge(deg4.rename(columns={'ensembl_id': 'Ensembl_ID'}),
                             on='Ensembl_ID', how='left')
node_base['Log2FC_MGS4']  = node_base['Log2FC_MGS4'].fillna(0.0)
# Use padj<0.1 (lenient) for Is_DEG flag; Log2FC still annotated for all nodes
node_base['Is_DEG_MGS4'] = (
    node_base['Is_DEG_MGS4'].fillna(False).astype(bool) |
    (node_base['Padj_MGS4'].fillna(1.0) < 0.1)
)
node_base['Padj_MGS4']    = node_base['Padj_MGS4'].fillna(1.0)

# Sort hubs first
node_csv = node_base.sort_values(['Is_Hub', 'out_degree'], ascending=[False, False]).reset_index(drop=True)

# ────────────────────────────────────────────────────────────────────────────
# TASK 5d — Build Edge table for Cytoscape
# ────────────────────────────────────────────────────────────────────────────
print("[Task 5d] Building Cytoscape edge table …")

edge_csv = filtered[['regulator_symbol', 'target_symbol', 'dominant_score', 'direction_confidence']].copy()
edge_csv.columns = ['Source_Node', 'Target_Node', 'Weight', 'Direction_Confidence']

# ────────────────────────────────────────────────────────────────────────────
# Save Cytoscape files
# ────────────────────────────────────────────────────────────────────────────
edge_csv.to_csv(f'{T5_OUT}/cytoscape_edges.csv', index=False)
node_csv.to_csv(f'{T5_OUT}/cytoscape_nodes.csv', index=False)

# ────────────────────────────────────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────────────────────────────────────
hub_deg_nodes = node_csv[node_csv['Is_Hub'] & node_csv['Is_DEG_MGS4']]
top5_txt = "\n".join(
    f"    {r['hub_rank']:>3}. {r['regulator_symbol']:<14} out={r['out_degree']}"
    for _, r in top100.head(10).iterrows()
)

t5_summary = textwrap.dedent(f"""\
===================================================
  Week 3 Task 5 -- Hub Regulators & Cytoscape Export
===================================================

  Date : {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Input: filtered_edge_list.tsv  ({len(filtered):,} edges)

  HUB ANALYSIS (out-degree in filtered network)
    Total regulators  : {len(hub_df):,}
    Max out-degree    : {hub_df['out_degree'].max()}  ({hub_df.iloc[0]['regulator_symbol']})
    Top-100 threshold : out_degree >= {top100.iloc[99]['out_degree']}

  TOP 10 HUB REGULATORS
{top5_txt}

  DEG OVERLAP (MGS4_vs_MGS1)
    Hub genes that are also DEGs : {len(hub_deg_nodes)}
    {', '.join(hub_deg_nodes['Gene_Symbol'].head(15).tolist())}

  CYTOSCAPE EXPORT
    cytoscape_edges.csv  -- {len(edge_csv):,} edges
      Columns: Source_Node | Target_Node | Weight | Direction_Confidence
    cytoscape_nodes.csv  -- {len(node_csv):,} nodes
      Columns: Gene_Symbol | Ensembl_ID | Out_Degree | Is_Hub | Log2FC_MGS4 | Padj_MGS4 | Is_DEG_MGS4

  CYTOSCAPE IMPORT GUIDE
    1. File > Import > Network from File > cytoscape_edges.csv
       Source col: Source_Node  |  Target col: Target_Node
       Edge attr : Weight, Direction_Confidence
    2. File > Import > Table from File > cytoscape_nodes.csv
       Key col: Gene_Symbol
    3. Style > Edge width   -> map to Weight (continuous)
    4. Style > Node size    -> map to Out_Degree (continuous)
    5. Style > Node fill    -> map to Log2FC_MGS4 (diverging: blue-white-red)
    6. Style > Node border  -> Is_Hub=True  -> thick gold border

  OUTPUTS (task5_hub_cytoscape/)
    top100_hub_regulators.tsv
    cytoscape_edges.csv
    cytoscape_nodes.csv
    hub_summary.txt
===================================================
""")
print(t5_summary)
with open(f'{T5_OUT}/hub_summary.txt', 'w') as f:
    f.write(t5_summary)

print("Done.")
