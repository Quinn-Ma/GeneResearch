"""
Week 4 Task 1 & 2 — Data Preparation for TabNet
  Input : VST-corrected expression, MetaSheet_QC, feature_set_final
  Output: aligned X/y arrays + scaler for downstream tasks
"""

import pandas as pd
import numpy as np
import os, pickle, textwrap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = '/mnt/c/Users/LuckyQinzhen/generesearch'
OUT  = f'{BASE}/week4/task1_data_prep'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load raw inputs ───────────────────────────────────────────────────────
print("Loading data …")
vst      = pd.read_csv(f'{BASE}/week1/task3_confounder_correction/Genelevel_VST_corrected.tsv',
                       sep='\t', index_col=0)
meta_raw = pd.read_csv(f'{BASE}/week1/task2_sample_qc/MetaSheet_QC.csv',
                       encoding='latin1', index_col=0)
feat_df  = pd.read_csv(f'{BASE}/week2/task3_feature_set/feature_set_final.tsv', sep='\t')

print(f"  VST matrix     : {vst.shape[0]:,} genes × {vst.shape[1]:,} samples")
print(f"  MetaSheet      : {len(meta_raw):,} samples")
print(f"  Feature genes  : {len(feat_df):,}")

# ── 2. Align samples via r_id ────────────────────────────────────────────────
meta_raw['r_id'] = meta_raw['r_id'].astype(str)
common   = sorted(set(vst.columns) & set(meta_raw['r_id']))
meta_al  = meta_raw.set_index('r_id').loc[common]
print(f"\n  Aligned samples: {len(common)}")

# ── 3. Subset to feature genes ───────────────────────────────────────────────
avail_genes = [g for g in feat_df['ensembl_id'] if g in vst.index]
print(f"  Feature genes found in VST: {len(avail_genes):,} / {len(feat_df):,}")

expr = vst.loc[avail_genes, common].T   # samples × genes  (433 × 3031)

# Gene symbol lookup
sym_map      = feat_df.set_index('ensembl_id')['gene_symbol'].to_dict()
feature_syms = np.array([sym_map.get(g, g) for g in avail_genes])

# ── 4. Labels: MGS 1-4  →  0-3 ──────────────────────────────────────────────
y           = (meta_al['mgs_level'].values - 1).astype(int)   # 0,1,2,3
sample_ids  = np.array(common)
class_names = ['MGS1', 'MGS2', 'MGS3', 'MGS4']

print("\n  Class distribution (0-indexed):")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    class {u} ({class_names[u]}): {c} samples")

# ── 5. Stratified 80/20 split ────────────────────────────────────────────────
(X_tr_raw, X_te_raw,
 y_train,  y_test,
 sid_tr,   sid_te)  = train_test_split(
    expr.values, y, sample_ids,
    test_size=0.2, random_state=42, stratify=y
)
print(f"\n  Train: {len(y_train)} samples  |  Test: {len(y_test)} samples")

# ── 6. StandardScaler (fit on train only) ───────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_tr_raw).astype(np.float32)
X_test  = scaler.transform(X_te_raw).astype(np.float32)

# ── 7. Save all artefacts ────────────────────────────────────────────────────
np.save(f'{OUT}/X_train.npy',       X_train)
np.save(f'{OUT}/X_test.npy',        X_test)
np.save(f'{OUT}/y_train.npy',       y_train)
np.save(f'{OUT}/y_test.npy',        y_test)
np.save(f'{OUT}/feature_syms.npy',  feature_syms)
np.save(f'{OUT}/train_ids.npy',     sid_tr)
np.save(f'{OUT}/test_ids.npy',      sid_te)
with open(f'{OUT}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# ── 8. Summary ───────────────────────────────────────────────────────────────
tr_dist = dict(zip(*np.unique(y_train, return_counts=True)))
te_dist = dict(zip(*np.unique(y_test,  return_counts=True)))

summary = textwrap.dedent(f"""\
===================================================
  Week 4 Task 1 & 2 -- Data Preparation
===================================================

  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}

  INPUT
    Expression  : Genelevel_VST_corrected.tsv  ({vst.shape[0]:,} genes × {vst.shape[1]:,} samples)
    MetaSheet   : MetaSheet_QC.csv  ({len(meta_raw):,} samples)
    Features    : feature_set_final.tsv  ({len(feat_df):,} genes)

  ALIGNED DATASET
    Samples     : {len(common)} (intersection via r_id key)
    Features    : {len(avail_genes):,} genes  ({len(avail_genes)/len(feat_df)*100:.1f}% of feature set)
    Target      : mgs_level (4 classes, 0-indexed)

  CLASS DISTRIBUTION
    MGS1 (0)    : {int(np.sum(y==0))} total  ({int(tr_dist.get(0,0))} train / {int(te_dist.get(0,0))} test)
    MGS2 (1)    : {int(np.sum(y==1))} total  ({int(tr_dist.get(1,0))} train / {int(te_dist.get(1,0))} test)
    MGS3 (2)    : {int(np.sum(y==2))} total  ({int(tr_dist.get(2,0))} train / {int(te_dist.get(2,0))} test)
    MGS4 (3)    : {int(np.sum(y==3))} total  ({int(tr_dist.get(3,0))} train / {int(te_dist.get(3,0))} test)
    NOTE: Class imbalance (MGS4 smallest) → TabNet will use inverse-frequency weights

  SPLIT
    Strategy    : Stratified train/test split  (80% / 20%, seed=42)
    Train       : {X_train.shape[0]} samples × {X_train.shape[1]:,} features
    Test        : {X_test.shape[0]} samples × {X_test.shape[1]:,} features

  PREPROCESSING
    Scaling     : StandardScaler (fit on train, transform both)
    Note        : VST is log-scale — no log1p needed

  OUTPUTS (task1_data_prep/)
    X_train.npy, X_test.npy      -- scaled expression arrays (float32)
    y_train.npy, y_test.npy      -- integer class labels (0-3)
    feature_syms.npy             -- gene symbol array (3031,)
    train_ids.npy, test_ids.npy  -- r_id strings for traceability
    scaler.pkl                   -- fitted StandardScaler
===================================================
""")
print(summary)
with open(f'{OUT}/data_summary.txt', 'w') as f:
    f.write(summary)

print("Task 1 & 2 complete.")
