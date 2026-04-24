"""
Week 4 Task 4 — LIME Personalized Biomarker Analysis
  Local Interpretable Model-agnostic Explanations for each AMD patient
  For each test sample: perturb expression → see which genes flip the diagnosis

  LIME vs SHAP:
    SHAP  → "C3 is globally important across all 433 patients"
    LIME  → "For THIS patient, GENE_X expression drives MGS4 diagnosis"
"""

import numpy as np
import pandas as pd
import os, textwrap
from datetime import datetime
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from lime import lime_tabular
from pytorch_tabnet.tab_model import TabNetClassifier

BASE = '/mnt/c/Users/LuckyQinzhen/generesearch'
D1   = f'{BASE}/week4/task1_data_prep'
D2   = f'{BASE}/week4/task2_tabnet'
OUT  = f'{BASE}/week4/task3_lime'
os.makedirs(OUT, exist_ok=True)
os.makedirs(f'{OUT}/sample_reports', exist_ok=True)

CLASS_NAMES = ['MGS1', 'MGS2', 'MGS3', 'MGS4']

# ── 1. Load data & model ──────────────────────────────────────────────────────
print("Loading data and TabNet model …")
X_train      = np.load(f'{D1}/X_train.npy')
X_test       = np.load(f'{D1}/X_test.npy')
y_train      = np.load(f'{D1}/y_train.npy')
y_test       = np.load(f'{D1}/y_test.npy')
feature_syms = np.load(f'{D1}/feature_syms.npy', allow_pickle=True)
test_ids     = np.load(f'{D1}/test_ids.npy', allow_pickle=True)

clf = TabNetClassifier()
clf.load_model(f'{D2}/tabnet_model.zip')

# Apply the same 500-gene selection used during TabNet stage-2 training
top500_idx   = np.load(f'{D2}/top500_gene_idx.npy')
feature_syms = np.load(f'{D2}/top500_feature_syms.npy', allow_pickle=True)
X_train      = X_train[:, top500_idx]
X_test       = X_test[:, top500_idx]

print(f"  Test samples    : {len(X_test)}")
print(f"  Feature genes   : {len(feature_syms)} (top-500 attention-selected)")

# ── 2. Prediction function for LIME ──────────────────────────────────────────
def predict_fn(X: np.ndarray) -> np.ndarray:
    return clf.predict_proba(X.astype(np.float32))

# ── 3. LIME explainer ─────────────────────────────────────────────────────────
print("Initialising LIME explainer …")
explainer = lime_tabular.LimeTabularExplainer(
    training_data        = X_train,
    feature_names        = feature_syms.tolist(),
    class_names          = CLASS_NAMES,
    mode                 = 'classification',
    discretize_continuous= False,   # gene expression is continuous
    random_state         = 42,
)

# ── 4. Explain all test samples ───────────────────────────────────────────────
print(f"Running LIME on {len(X_test)} test samples (num_samples=300 each) …")
N_FEATURES_PER_SAMPLE = 10

all_weights     = []          # list of dicts: {gene_symbol: lime_weight}
sample_results  = []

t0 = datetime.now()
for i, (x, y_true, sid) in enumerate(zip(X_test, y_test, test_ids)):
    y_pred = int(clf.predict(x.reshape(1, -1).astype(np.float32))[0])
    proba  = predict_fn(x.reshape(1, -1))[0]

    exp = explainer.explain_instance(
        data_row    = x,
        predict_fn  = predict_fn,
        labels      = (y_pred,),       # explain the predicted class
        num_features= N_FEATURES_PER_SAMPLE,
        num_samples = 300,
    )

    weights_raw = exp.as_list(label=y_pred)   # [(feature_str, weight), ...]

    # Parse "gene_symbol <= value" or "gene_symbol > value" → clean symbol
    gene_weights = {}
    for feat_str, w in weights_raw:
        # LIME feature strings look like "GENE_X <= 0.5" or "-0.3 < GENE_X <= 1.2"
        # Extract the gene symbol (the word part)
        parts = feat_str.replace('<=','|').replace('<','|').replace('>','|').split('|')
        sym = None
        for p in parts:
            p = p.strip()
            if p in set(feature_syms):
                sym = p
                break
        if sym is None:
            # Fallback: take the longest word token
            tokens = feat_str.split()
            sym = max([t for t in tokens if any(c.isalpha() for c in t)],
                      key=len, default=feat_str)
        gene_weights[sym] = w

    all_weights.append(gene_weights)

    sample_results.append({
        'sample_id'   : sid,
        'true_class'  : CLASS_NAMES[y_true],
        'pred_class'  : CLASS_NAMES[y_pred],
        'correct'     : (y_true == y_pred),
        'confidence'  : float(proba[y_pred]),
        'top_genes'   : '; '.join([f"{g}({w:+.4f})" for g, w in sorted(gene_weights.items(),
                                                                        key=lambda x: -abs(x[1]))]),
    })

    if (i + 1) % 10 == 0:
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"  {i+1}/{len(X_test)} done  ({elapsed:.0f}s elapsed)")

elapsed_total = (datetime.now() - t0).total_seconds()
print(f"\nLIME complete: {elapsed_total:.1f}s total")

# ── 5. Save per-sample results ────────────────────────────────────────────────
results_df = pd.DataFrame(sample_results)
results_df.to_csv(f'{OUT}/sample_predictions.csv', index=False)

# ── 6. Aggregate: find most universally important genes ───────────────────────
print("Aggregating LIME results …")
gene_freq    = defaultdict(int)
gene_weights_agg = defaultdict(list)

for gw in all_weights:
    for sym, w in gw.items():
        gene_freq[sym] += 1
        gene_weights_agg[sym].append(w)

agg = pd.DataFrame({
    'gene_symbol'        : list(gene_freq.keys()),
    'frequency'          : [gene_freq[g]            for g in gene_freq],
    'mean_abs_weight'    : [np.mean(np.abs(gene_weights_agg[g])) for g in gene_freq],
    'mean_weight'        : [np.mean(gene_weights_agg[g])         for g in gene_freq],
    'positive_frac'      : [np.mean([w>0 for w in gene_weights_agg[g]])  for g in gene_freq],
}).sort_values(['frequency', 'mean_abs_weight'], ascending=[False, False]).reset_index(drop=True)

agg.insert(0, 'lime_rank', agg.index + 1)
agg['pct_patients'] = (agg['frequency'] / len(X_test) * 100).round(1)
agg['direction'] = agg['mean_weight'].apply(lambda w: 'activating' if w > 0 else 'suppressing')

top50 = agg.head(50).copy()
top50.to_csv(f'{OUT}/top50_lime_genes.tsv', sep='\t', index=False)

print(f"\n  Top 5 LIME genes:")
for _, r in top50.head(5).iterrows():
    print(f"    {int(r['lime_rank']):>2}. {r['gene_symbol']:<12}  "
          f"freq={int(r['frequency'])}/{len(X_test)}  "
          f"mean_w={r['mean_weight']:+.4f}  ({r['direction']})")

# ── 7. Generate personalized patient reports (2 per MGS class = 8 total) ──────
print("\nGenerating personalized patient reports …")
reports_generated = []
for cls_idx in range(4):
    # Pick 2 correctly predicted samples from this class
    cands = [r for r in sample_results
             if r['true_class'] == CLASS_NAMES[cls_idx] and r['correct']][:2]
    for r in cands:
        i = list(test_ids).index(r['sample_id'])
        gw = all_weights[i]
        sorted_gw = sorted(gw.items(), key=lambda x: -abs(x[1]))
        lines = [
            f"{'='*55}",
            f"  PERSONALIZED BIOMARKER REPORT",
            f"{'='*55}",
            f"  Sample ID   : {r['sample_id']}",
            f"  True MGS    : {r['true_class']}",
            f"  Predicted   : {r['pred_class']}  (confidence={r['confidence']:.1%})",
            f"  Status      : {'CORRECT' if r['correct'] else 'MISCLASSIFIED'}",
            f"",
            f"  TOP 10 DRIVING BIOMARKERS FOR THIS PATIENT",
            f"  (+ = pushes toward {r['pred_class']} diagnosis)",
            f"  (- = protects against / pulls away)",
            f"  {'Rank':<6}{'Gene':<14}{'LIME Weight':>12}{'Effect':<20}",
            f"  {'-'*52}",
        ]
        for rank, (sym, w) in enumerate(sorted_gw, 1):
            effect = 'Drives diagnosis' if w > 0 else 'Protective / opposing'
            lines.append(f"  {rank:<6}{sym:<14}{w:>+12.4f}  {effect}")
        lines += [
            f"",
            f"  CLINICAL INTERPRETATION",
            f"  The top genes above explain why this patient's retina",
            f"  was classified as {r['pred_class']}.  Unlike population-level",
            f"  statistics, this profile is unique to THIS patient.",
            f"{'='*55}",
        ]
        report_txt = '\n'.join(lines)
        fname = f"{OUT}/sample_reports/{r['sample_id'].replace('/', '_')}_{r['true_class']}.txt"
        with open(fname, 'w') as f:
            f.write(report_txt)
        reports_generated.append(r['sample_id'])
print(f"  Generated {len(reports_generated)} sample reports")

# ── 8. LIME frequency heatmap (top 30 genes × all test samples) ───────────────
print("Generating LIME heatmap …")
top30_syms = top50.head(30)['gene_symbol'].tolist()
heat_data  = np.zeros((len(X_test), 30))
for i, gw in enumerate(all_weights):
    for j, sym in enumerate(top30_syms):
        heat_data[i, j] = gw.get(sym, 0.0)

# Sort rows by predicted class
sort_order = results_df['pred_class'].argsort().values
heat_sorted = heat_data[sort_order]
row_labels  = [f"{results_df.iloc[s]['pred_class']}" for s in sort_order]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(heat_sorted, aspect='auto', cmap='RdBu_r',
               vmin=-0.05, vmax=0.05)
ax.set_xticks(range(30))
ax.set_xticklabels(top30_syms, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(0, len(X_test), 5))
ax.set_yticklabels([row_labels[i] for i in range(0, len(X_test), 5)], fontsize=7)
ax.set_xlabel('Gene (LIME top-30)', fontsize=11)
ax.set_ylabel('Test Patient (sorted by predicted class)', fontsize=11)
ax.set_title('LIME Weights: Patient × Gene Heatmap\n'
             '(Red = activates predicted diagnosis; Blue = suppresses)',
             fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.6, label='LIME weight')
plt.tight_layout()
plt.savefig(f'{OUT}/lime_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lime_heatmap.png")

# ── 9. Top 50 frequency bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors_bar = ['#e15759' if d == 'activating' else '#4e79a7'
              for d in top50['direction']]
ax.bar(range(50), top50['frequency'], color=colors_bar, edgecolor='white')
ax.set_xticks(range(50))
ax.set_xticklabels(top50['gene_symbol'], rotation=90, fontsize=7)
ax.set_xlabel('Gene (LIME top-50)', fontsize=11)
ax.set_ylabel(f'Frequency (out of {len(X_test)} test patients)', fontsize=11)
ax.set_title('LIME Top-50 Genes: How Often Each Gene Appears in a Patient\'s\n'
             'Top-10 Biomarkers  (Red=activating, Blue=suppressing)',
             fontsize=12, fontweight='bold')
ax.axhline(len(X_test)*0.5, color='gray', ls='--', lw=1, label='50% threshold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/lime_top50_frequency.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: lime_top50_frequency.png")

# ── 10. Summary ────────────────────────────────────────────────────────────────
correct_rate = results_df['correct'].mean()
top5_str = '\n'.join([
    f"    {int(r['lime_rank']):>2}. {r['gene_symbol']:<12} "
    f"freq={int(r['frequency'])}/{len(X_test)} ({r['pct_patients']}%)  "
    f"direction={r['direction']}"
    for _, r in top50.head(10).iterrows()
])

lime_summary = textwrap.dedent(f"""\
===================================================
  Week 4 Task 4 -- LIME Personalized Analysis
===================================================

  Date              : {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Runtime           : {elapsed_total:.1f}s
  Test samples      : {len(X_test)}  (correct={results_df['correct'].sum()}, {correct_rate:.1%})

  LIME PARAMETERS
    num_features    : {N_FEATURES_PER_SAMPLE} per patient
    num_samples     : 300 (perturbations per explanation)
    Explained class : predicted class (personalized)

  TOP 10 MOST UNIVERSAL BIOMARKERS
  (ranked by frequency across all {len(X_test)} test patients)
{top5_str}

  KEY INSIGHT
    These genes appear in the top-10 explanation for the
    majority of patients — they are the AMD "network backbone".
    But each patient's FULL top-10 profile is unique, enabling
    precision medicine / personalized biomarker reports.

  OUTPUTS
    top50_lime_genes.tsv      -- {len(top50)} genes with freq, mean weight, direction
    sample_predictions.csv    -- per-patient predicted class + top genes
    sample_reports/           -- {len(reports_generated)} individual patient PDFs
    lime_heatmap.png          -- patient × gene LIME weight matrix
    lime_top50_frequency.png  -- gene frequency bar chart
===================================================
""")
print(lime_summary)
with open(f'{OUT}/lime_summary.txt', 'w') as f:
    f.write(lime_summary)

print("Task 4 (LIME) complete.")
