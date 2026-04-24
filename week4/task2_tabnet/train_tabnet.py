"""
Week 4 Task 3 — TabNet Deep Learning Classifier
  Sequential Attention Mechanism for AMD MGS grade prediction
  Input : prepared X/y arrays from task1_data_prep
  Output: trained model, confusion matrix, feature importances
"""

import numpy as np
import pandas as pd
import os, textwrap
from datetime import datetime

import torch
from pytorch_tabnet.tab_model import TabNetClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

BASE  = '/mnt/c/Users/LuckyQinzhen/generesearch'
D1    = f'{BASE}/week4/task1_data_prep'
OUT   = f'{BASE}/week4/task2_tabnet'
os.makedirs(OUT, exist_ok=True)

# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading prepared data …")
X_train      = np.load(f'{D1}/X_train.npy')
X_test       = np.load(f'{D1}/X_test.npy')
y_train      = np.load(f'{D1}/y_train.npy')
y_test       = np.load(f'{D1}/y_test.npy')
feature_syms = np.load(f'{D1}/feature_syms.npy', allow_pickle=True)
test_ids     = np.load(f'{D1}/test_ids.npy', allow_pickle=True)

CLASS_NAMES = ['MGS1', 'MGS2', 'MGS3', 'MGS4']
N_CLASSES   = 4
device      = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}  |  Device: {device}")

# ── 2. TabNet model ───────────────────────────────────────────────────────────
# n_d / n_a: embedding width (32 for small dataset ~350 samples)
# n_steps  : number of sequential attention steps (each step = one "look")
# gamma    : sparsity control (higher = reuse features more)
# lambda_sparse: L1 penalty on attention masks
clf = TabNetClassifier(
    n_d          = 32,
    n_a          = 32,
    n_steps      = 4,
    gamma        = 1.5,
    n_independent= 2,
    n_shared     = 2,
    lambda_sparse= 1e-4,
    momentum     = 0.02,
    clip_value   = 2.0,
    seed         = 42,
    device_name  = device,
    verbose      = 10,
)

# ── 3. Two-stage training ─────────────────────────────────────────────────────
# Stage 1: quick run on all 3031 features → get attention-based feature ranking
# Stage 2: re-train on top-500 attention genes → better signal-to-noise ratio
# (3031 features / 346 samples = 9:1 ratio → needs dimensionality reduction)
from sklearn.model_selection import train_test_split

print("\n[Stage 1] Quick run on all 3031 features to rank genes by attention …")
t0 = datetime.now()
X_tr1, X_val1, y_tr1, y_val1 = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)
clf.fit(
    X_tr1, y_tr1,
    eval_set           = [(X_val1, y_val1)],
    eval_name          = ['val'],
    eval_metric        = ['balanced_accuracy'],
    max_epochs         = 100,
    patience           = 20,
    batch_size         = 256,
    virtual_batch_size = 64,
    weights            = 1,
    drop_last          = False,
)
fi_stage1 = clf.feature_importances_
top500_idx = np.argsort(fi_stage1)[::-1][:500]
print(f"  Stage-1 done in {(datetime.now()-t0).total_seconds():.1f}s")
print(f"  Top gene after stage-1: {feature_syms[top500_idx[0]]}")

# Stage 2: re-train on top-500 selected genes
print("\n[Stage 2] Re-training on top-500 attention-selected genes …")
X_tr2  = X_tr1[:, top500_idx]
X_val2 = X_val1[:, top500_idx]
X_train_s = X_train[:, top500_idx]
X_test_s  = X_test[:, top500_idx]

clf2 = TabNetClassifier(
    n_d           = 32,
    n_a           = 32,
    n_steps       = 4,
    gamma         = 1.3,
    n_independent = 2,
    n_shared      = 2,
    lambda_sparse = 1e-3,
    momentum      = 0.02,
    clip_value    = 2.0,
    seed          = 42,
    device_name   = device,
    verbose       = 10,
)
clf2.fit(
    X_tr2, y_tr1,
    eval_set           = [(X_val2, y_val1)],
    eval_name          = ['val'],
    eval_metric        = ['balanced_accuracy'],
    max_epochs         = 500,
    patience           = 50,
    batch_size         = 128,
    virtual_batch_size = 32,
    weights            = 1,
    drop_last          = False,
)
runtime = (datetime.now() - t0).total_seconds()
print(f"  Total training time: {runtime:.1f}s")

# Use stage-2 model as the final model
clf        = clf2
X_test     = X_test_s
X_train    = X_train_s
feature_syms_s = feature_syms[top500_idx]

# Save the selected gene index for LIME
np.save(f'{OUT}/top500_gene_idx.npy', top500_idx)
np.save(f'{OUT}/top500_feature_syms.npy', feature_syms_s)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
print("\nEvaluating …")
y_pred      = clf.predict(X_test)
y_proba     = clf.predict_proba(X_test)

acc      = accuracy_score(y_test, y_pred)
bal_acc  = balanced_accuracy_score(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

# Macro OVR AUC
y_bin    = label_binarize(y_test, classes=[0,1,2,3])
auc_macro = roc_auc_score(y_bin, y_proba, multi_class='ovr', average='macro')

print(f"  Accuracy         : {acc:.4f}")
print(f"  Balanced Accuracy: {bal_acc:.4f}")
print(f"  Macro AUC        : {auc_macro:.4f}")
print("\n" + report)

# ── 5. Confusion matrix plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0])
axes[0].set_title('Confusion Matrix', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

# Normalized confusion matrix
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Confusion Matrix (Row-Normalized)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

plt.suptitle(f'TabNet AMD MGS Classifier  |  Balanced Acc={bal_acc:.3f}  AUC={auc_macro:.3f}',
             fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUT}/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: confusion_matrix.png")

# ── 6. ROC curves (one-vs-rest) ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
for i, (cls, col) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
    auc_i = roc_auc_score(y_bin[:, i], y_proba[:, i])
    ax.plot(fpr, tpr, color=col, lw=2, label=f'{cls} (AUC={auc_i:.3f})')
ax.plot([0,1],[0,1],'k--', lw=1)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate', fontsize=11)
ax.set_title('ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: roc_curves.png")

# ── 7. Feature importances ─────────────────────────────────────────────────────
print("\nExtracting feature importances …")
fi = clf.feature_importances_
fi_df = pd.DataFrame({
    'gene_symbol'       : feature_syms_s,
    'tabnet_importance' : fi,
}).sort_values('tabnet_importance', ascending=False).reset_index(drop=True)
fi_df.insert(0, 'rank', fi_df.index + 1)

fi_df.to_csv(f'{OUT}/feature_importances.tsv', sep='\t', index=False)

# Plot top 30
fig, ax = plt.subplots(figsize=(9, 7))
top30 = fi_df.head(30)
ax.barh(top30['gene_symbol'][::-1], top30['tabnet_importance'][::-1],
        color='#e15759', edgecolor='white')
ax.set_xlabel('TabNet Attention Importance', fontsize=11)
ax.set_title('Top 30 Feature Genes (TabNet Attention Masks)', fontsize=12, fontweight='bold')
ax.tick_params(axis='y', labelsize=8)
plt.tight_layout()
plt.savefig(f'{OUT}/feature_importance_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: feature_importance_plot.png")

# ── 8. Save model ─────────────────────────────────────────────────────────────
clf.save_model(f'{OUT}/tabnet_model')
print("  Saved: tabnet_model.zip")

# ── 9. Save test predictions for LIME ────────────────────────────────────────
pred_df = pd.DataFrame({
    'sample_id'   : test_ids,
    'true_label'  : y_test,
    'pred_label'  : y_pred,
    'prob_MGS1'   : y_proba[:, 0],
    'prob_MGS2'   : y_proba[:, 1],
    'prob_MGS3'   : y_proba[:, 2],
    'prob_MGS4'   : y_proba[:, 3],
})
pred_df.to_csv(f'{OUT}/test_predictions.csv', index=False)

# ── 10. Summary ────────────────────────────────────────────────────────────────
top5_genes = ', '.join(fi_df.head(5)['gene_symbol'].tolist())
summary = textwrap.dedent(f"""\
===================================================
  Week 4 Task 3 -- TabNet Classifier
===================================================

  Date          : {datetime.now().strftime('%Y-%m-%d %H:%M')}
  Device        : {device.upper()}
  Runtime       : {runtime:.1f}s

  ARCHITECTURE (2-stage training)
    Stage 1     : All 3031 features, quick run → attention-based gene ranking
    Stage 2     : Top-500 attention genes, full training
    n_d / n_a   : 32 (feature embedding width)
    n_steps     : 4  (sequential attention rounds)
    gamma       : 1.3
    lambda_sparse: 1e-3
    batch_size  : 128 (virtual: 32)
    Class weights: inverse-frequency  (handles MGS4 imbalance)

  PERFORMANCE
    Accuracy         : {acc:.4f}  ({acc*100:.1f}%)
    Balanced Accuracy: {bal_acc:.4f}
    Macro AUC (OVR)  : {auc_macro:.4f}

  PER-CLASS METRICS
{report}

  TOP 5 GENES (TabNet attention)
    {top5_genes}

  OUTPUTS
    tabnet_model.zip            -- saved model (reload with clf.load_model)
    confusion_matrix.png        -- raw + normalized confusion matrix
    roc_curves.png              -- one-vs-rest ROC per MGS class
    feature_importances.tsv     -- {len(fi_df):,} genes ranked by attention
    feature_importance_plot.png -- top-30 bar chart
    test_predictions.csv        -- per-sample predictions + probabilities
===================================================
""")
print(summary)
with open(f'{OUT}/tabnet_summary.txt', 'w') as f:
    f.write(summary)

print("Task 3 (TabNet) complete.")
