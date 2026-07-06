"""
06_nn.py — 2-layer neural network for issue-triage
===================================================
A simple feedforward NN (18 → 64 → 32 → 1) to compare
against XGBoost on the same data/metrics.

Key design choices:
  - Same train/val/test split as 05_xgboost.py (seed=42)
  - pos_weight in BCEWithLogitsLoss for class imbalance
  - Early stopping on validation loss
  - Per-repo F1 breakdown (DL-012)

Inputs:  data/processed/{X_train, X_test, y_train, y_test}.parquet
Outputs: models/nn_model.pt, printed metrics
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

# ── Same val split as XGBoost (seed=42) for fair comparison ──────────
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2, random_state=42, stratify=y_train_full
)
print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

# ── Convert to PyTorch tensors ───────────────────────────────────────
def to_tensors(X, y):
    return (
        torch.tensor(X.values, dtype=torch.float32),
        torch.tensor(y.values, dtype=torch.float32).unsqueeze(1),
    )

X_tr_t, y_tr_t = to_tensors(X_train, y_train)
X_vl_t, y_vl_t = to_tensors(X_val, y_val)
X_te_t, y_te_t = to_tensors(X_test, y_test)

train_loader = DataLoader(
    TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True
)

# ── Define the network ───────────────────────────────────────────────
class IssueTriage_NN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),   # Layer 1: 18 → 64
            nn.ReLU(),
            nn.Dropout(0.3),             # Randomly zero 30% of neurons (overfitting brake)
            nn.Linear(64, 32),           # Layer 2: 64 → 32
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),            # Output: 32 → 1 (raw logit, sigmoid in loss)
        )

    def forward(self, x):
        return self.net(x)

n_features = X_train.shape[1]
model = IssueTriage_NN(n_features)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Loss and optimizer ───────────────────────────────────────────────
# pos_weight: same idea as scale_pos_weight in XGBoost
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
print(f"pos_weight = {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ── Training loop with early stopping ────────────────────────────────
EPOCHS = 100
PATIENCE = 10        # stop if val loss doesn't improve for 10 epochs

best_val_loss = float("inf")
patience_counter = 0
best_state = None

print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...\n")

for epoch in range(1, EPOCHS + 1):
    # ── Train phase ──────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(X_batch)
    train_loss /= len(X_tr_t)

    # ── Val phase ────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        val_logits = model(X_vl_t)
        val_loss = criterion(val_logits, y_vl_t).item()

    # ── Early stopping check ─────────────────────────────────────
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
model_path = MODELS_DIR / "nn_model.pt"
torch.save(best_state, model_path)
print(f"Model saved → {model_path}\n")

# ── Predict on test set ──────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits = model(X_te_t)
    test_probs  = torch.sigmoid(test_logits).squeeze()
    y_pred_test = (test_probs >= 0.5).int().numpy()

y_test_np = y_test.values

# ── Evaluation helpers ───────────────────────────────────────────────
REPO_COLS  = [c for c in X_test.columns if c.startswith("repo_")]
REPO_NAMES = [c.replace("repo_", "") for c in REPO_COLS]

def per_repo_f1(y_true, y_pred, X):
    results = {}
    for col, name in zip(REPO_COLS, REPO_NAMES):
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
print("  Neural Network — Test set")
print("─" * 60)
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1 (pos):  {f1:.4f}")
for repo, (rf1, n) in repo_f1s.items():
    print(f"    └─ {repo:30s} F1 = {rf1:.4f}  (n={n})")
print()

# ── Head-to-head comparison ──────────────────────────────────────────
print("=" * 60)
print("HEAD-TO-HEAD (test set, positive-class F1)")
print("=" * 60)
print(f"  Brick (majority):       0.0000")
print(f"  Coin  (stratified):     0.3737")
print(f"  Logistic Regression:    0.5122")
print(f"  XGBoost:                0.5200")
print(f"  Neural Network:         {f1:.4f}")
print()
if f1 > 0.5200:
    print("→ NN beats XGBoost. Consider NN as primary model.")
elif f1 > 0.5122:
    print("→ NN beats LR but not XGBoost. XGBoost remains primary.")
else:
    print("→ NN does not beat LR. XGBoost remains primary.")