"""
06b_nn_weighted.py — NN with repo-based sample weighting
=========================================================
Same 2-layer NN as 06_nn.py, but adds per-sample weights
so that underrepresented repos (langchain) get more
influence during training.

Combines TWO reweighting strategies:
  1. pos_weight: class imbalance (YES vs NO)
  2. sample_weights: repo imbalance (pytorch vs langchain)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
MODELS_DIR   = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────
X_train_full = pd.read_parquet(PROCESSED / "X_train.parquet")
X_test       = pd.read_parquet(PROCESSED / "X_test.parquet")
y_train_full = pd.read_parquet(PROCESSED / "y_train.parquet").squeeze()
y_test       = pd.read_parquet(PROCESSED / "y_test.parquet").squeeze()

# ── Same val split (seed=42) ────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── Compute repo-based sample weights ───────────────────────────────
REPO_COLS = [c for c in X_train.columns if c.startswith("repo_")]

print("\nRepo distribution in training set:")
repo_counts = {}
for col in REPO_COLS:
    name = col.replace("repo_", "")
    count = (X_train[col] == 1).sum()
    repo_counts[col] = count
    print(f"  {name:30s}  {count:,} rows")

# Weight = total_rows / (n_repos * rows_in_this_repo)
# This makes each repo contribute equally to the total loss.
n_repos = len(REPO_COLS)
total_rows = len(X_train)

sample_weights = np.ones(total_rows, dtype=np.float32)
for col in REPO_COLS:
    mask = X_train[col].values == 1
    weight = total_rows / (n_repos * repo_counts[col])
    sample_weights[mask] = weight
    name = col.replace("repo_", "")
    print(f"  {name:30s}  weight = {weight:.2f}")

sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32)

# ── Convert to tensors ──────────────────────────────────────────────
def to_tensors(X, y):
    return (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1),
    )

X_tr_t, y_tr_t = to_tensors(X_train, y_train)
X_vl_t, y_vl_t = to_tensors(X_val, y_val)
X_te_t, y_te_t = to_tensors(X_test, y_test)

# ── DataLoader with indexed access for sample weights ────────────────
# We need row indices to look up the correct sample weight per batch
train_dataset = TensorDataset(X_tr_t, y_tr_t, sample_weights_t)
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

# ── Network (same architecture as 06_nn.py) ─────────────────────────
class IssueTriage_NN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)

n_features = X_train.shape[1]
model = IssueTriage_NN(n_features)
print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss and optimizer ───────────────────────────────────────────────
# pos_weight handles class imbalance (YES vs NO)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
print(f"pos_weight (class) = {pos_weight.item():.2f}")

# We use reduction='none' so we can multiply by sample_weights manually
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For validation loss (no sample weighting — we want honest val metrics)
val_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ── Training loop ────────────────────────────────────────────────────
EPOCHS = 100
PATIENCE = 10

best_val_loss = float("inf")
patience_counter = 0
best_state = None

print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch, w_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)

        # Per-sample loss × repo weight, then mean
        per_sample_loss = criterion(logits, y_batch)
        weighted_loss = (per_sample_loss * w_batch.unsqueeze(1)).mean()

        weighted_loss.backward()
        optimizer.step()
        train_loss += weighted_loss.item() * len(X_batch)
    train_loss /= len(X_tr_t)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_vl_t)
        val_loss = val_criterion(val_logits, y_vl_t).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1

    if epoch % 10 == 0 or patience_counter == PATIENCE:
        print(f"  Epoch {epoch:3d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  patience={patience_counter}/{PATIENCE}")

    if patience_counter == PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}. Best val_loss={best_val_loss:.4f}")
        break
else:
    print(f"\nCompleted all {EPOCHS} epochs. Best val_loss={best_val_loss:.4f}")

# ── Load best weights and save ───────────────────────────────────────
model.load_state_dict(best_state)
model_path = MODELS_DIR / "nn_weighted_model.pt"
torch.save(best_state, model_path)
print(f"Model saved → {model_path}\n")

# ── Predict on test set ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits = model(X_te_t)
    test_probs  = torch.sigmoid(test_logits).squeeze()
    y_pred_test = (test_probs >= 0.5).int().numpy()

y_test_np = y_test.values

# ── Evaluation ───────────────────────────────────────────────────────
REPO_COLS_TEST  = [c for c in X_test.columns if c.startswith("repo_")]
REPO_NAMES_TEST = [c.replace("repo_", "") for c in REPO_COLS_TEST]

def per_repo_f1(y_true, y_pred, X):
    results = {}
    for col, name in zip(REPO_COLS_TEST, REPO_NAMES_TEST):
        mask = X[col].values == 1
        if mask.sum() == 0:
            continue
        f1 = f1_score(y_true[mask], y_pred[mask], zero_division=0)
        n  = mask.sum()
        results[name] = (f1, n)
    return results

acc  = accuracy_score(y_test_np, y_pred_test)
prec = precision_score(y_test_np, y_pred_test, zero_division=0)
rec  = recall_score(y_test_np, y_pred_test, zero_division=0)
f1   = f1_score(y_test_np, y_pred_test, zero_division=0)
repo_f1s = per_repo_f1(y_test_np, y_pred_test, X_test)

print("─" * 60)
print("  Neural Network (repo-weighted) — Test set")
print("─" * 60)
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 (pos):  {f1:.4f}")
for repo, (rf1, n) in repo_f1s.items():
    print(f"    └─ {repo:30s} F1 = {rf1:.4f}  (n={n})")
print()

# ── Comparison ───────────────────────────────────────────────────────
print("=" * 60)
print("FULL COMPARISON (test set, positive-class F1)")
print("=" * 60)
print(f"  {'Model':<30s} {'Overall':>8}  {'transform':>10}  {'langchain':>10}  {'pytorch':>8}")
print(f"  {'─'*30} {'─'*8}  {'─'*10}  {'─'*10}  {'─'*8}")
print(f"  {'Brick':<30s} {'0.0000':>8}  {'0.0000':>10}  {'0.0000':>10}  {'0.0000':>8}")
print(f"  {'Coin':<30s} {'0.3737':>8}  {'0.4083':>10}  {'0.2857':>10}  {'0.3758':>8}")
print(f"  {'Logistic Regression':<30s} {'0.5122':>8}  {'0.5913':>10}  {'0.0000':>10}  {'0.5032':>8}")
print(f"  {'XGBoost':<30s} {'0.5200':>8}  {'0.5846':>10}  {'0.3103':>10}  {'0.5117':>8}")
print(f"  {'NN (unweighted)':<30s} {'0.5500':>8}  {'0.6082':>10}  {'0.2444':>10}  {'0.5450':>8}")

lc_f1 = repo_f1s.get("langchain-ai/langchain", (0.0, 0))[0]
tf_f1 = repo_f1s.get("huggingface/transformers", (0.0, 0))[0]
pt_f1 = repo_f1s.get("pytorch/pytorch", (0.0, 0))[0]
print(f"  {'NN (repo-weighted)':<30s} {f1:>8.4f}  {tf_f1:>10.4f}  {lc_f1:>10.4f}  {pt_f1:>8.4f}")
print()

if lc_f1 > 0.2857:
    print(f"→ Langchain F1 = {lc_f1:.4f} — beats the coin (0.2857). Improvement sprint worked.")
else:
    print(f"→ Langchain F1 = {lc_f1:.4f} — still below the coin (0.2857). Likely a data ceiling.")