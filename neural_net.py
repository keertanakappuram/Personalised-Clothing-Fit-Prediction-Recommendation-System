"""
neural_net.py — PyTorch Neural Network for Clothing Fit Prediction
===================================================================
Feedforward neural network trained on engineered features from the
Rent The Runway dataset to predict clothing fit (Small / Fit / Large).

Architecture:
  Input (engineered features) → BatchNorm → Dense(256) → ReLU → Dropout
  → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Dense(3) → Softmax

Handles class imbalance via weighted cross-entropy loss.

Usage:
    python neural_net.py --data renttherunway_final_data.json.gz
"""

import argparse
import json
import gzip
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, f1_score, roc_auc_score, confusion_matrix
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BATCH_SIZE   = 512
EPOCHS       = 100
PATIENCE     = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT      = 0.3
HIDDEN_DIMS  = [256, 128, 64]
LABEL_NAMES  = ['small', 'fit', 'large']

# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    print(f"Loading data from {path}...")
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    df = pd.DataFrame(data)
    print(f"Loaded {len(df):,} records")
    return df

def parse_height(h):
    try:
        if pd.isna(h): return np.nan
        h = str(h).strip().replace('"', '').replace("'", "'")
        if "'" in h:
            parts = h.split("'")
            return int(parts[0]) * 12 + (int(parts[1].strip()) if parts[1].strip() else 0)
    except: return np.nan

def parse_weight(w):
    try:
        if pd.isna(w): return np.nan
        return int(str(w).lower().replace('lbs', '').strip())
    except: return np.nan

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing...")
    df = df.copy()

    # Body measurements
    df['height_inches'] = df['height'].apply(parse_height)
    df['weight_lbs']    = df['weight'].apply(parse_weight)
    df['bmi']           = 703 * (df['weight_lbs'] / (df['height_inches'] ** 2))
    df['age']           = pd.to_numeric(df['age'], errors='coerce')
    df.loc[(df['age'] < 18) | (df['age'] > 80), 'age'] = np.nan

    for col in ['height_inches', 'weight_lbs', 'bmi', 'age']:
        df[col] = df[col].fillna(df[col].median())

    # Target
    df['fit'] = df['fit'].str.lower().str.strip()
    df = df[df['fit'].isin(['small', 'fit', 'large'])].copy()

    # Size numeric
    size_map = {'xxs': 0, 'xs': 1, 's': 2, 'sm': 2, 'm': 3, 'md': 3,
                'l': 4, 'lg': 4, 'xl': 5, 'xxl': 6, '1x': 5, '2x': 6, '3x': 7}
    df['size_numeric'] = df['size'].str.lower().str.strip().map(size_map)
    df['size_numeric'] = df['size_numeric'].fillna(df['size_numeric'].median())

    # BMI outliers
    df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)].copy()

    # Categorical
    df['body type']  = df['body type'].fillna('unknown')
    df['rented for'] = df['rented for'].fillna('unknown')

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")
    df = df.copy()

    # Size relative to BMI peers
    df['bmi_bucket'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 35, 100],
                               labels=['underweight', 'normal', 'overweight', 'obese', 'extreme'])
    bucket_avg = df.groupby('bmi_bucket')['size_numeric'].transform('mean')
    df['size_relative_to_peers'] = df['size_numeric'] - bucket_avg
    df['size_percentile'] = df.groupby('bmi_bucket')['size_numeric'].transform(
        lambda x: x.rank(pct=True)
    )

    # Rating
    df['rating_numeric'] = pd.to_numeric(df['rating'], errors='coerce').fillna(5.0)
    df['is_high_rating'] = (df['rating_numeric'] >= 9).astype(int)

    # Bust size
    df['bust_band_size'] = df['bust size'].str.extract(r'(\d+)').astype(float)
    df['bust_band_size'] = df['bust_band_size'].fillna(df['bust_band_size'].median())
    cup_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'DD': 5, 'DDD': 6, 'E': 5, 'F': 6}
    df['bust_cup_numeric'] = df['bust size'].str.extract(r'([A-Z]+)').iloc[:, 0].map(cup_map).fillna(3)

    # Text keywords
    def extract_signals(text):
        if pd.isna(text): return 0, 0, 0
        t = str(text).lower()
        small   = int(any(w in t for w in ['too small', 'tight', 'snug', 'ran small', 'size up']))
        large   = int(any(w in t for w in ['too big', 'large', 'loose', 'baggy', 'ran large']))
        perfect = int(any(w in t for w in ['perfect', 'true to size', 'fit perfectly', 'just right']))
        return small, large, perfect

    df[['text_small', 'text_large', 'text_perfect']] = df['review_text'].apply(
        lambda x: pd.Series(extract_signals(x))
    )

    # BMI interactions
    df['bmi_size_interaction'] = df['bmi'] * df['size_numeric']
    df['size_per_bmi']         = df['size_numeric'] / (df['bmi'] + 1e-9)

    return df

def add_history_features(train_df, test_df):
    print("Adding user/item history features...")

    user_feat = train_df.groupby('user_id').agg(
        user_small_rate=('fit', lambda x: (x == 'small').mean()),
        user_large_rate=('fit', lambda x: (x == 'large').mean()),
        user_fit_rate=('fit',   lambda x: (x == 'fit').mean()),
        user_avg_size=('size_numeric', 'mean'),
        user_total_rentals=('fit', 'size')
    ).reset_index()

    item_feat = train_df.groupby('item_id').agg(
        item_small_rate=('fit', lambda x: (x == 'small').mean()),
        item_large_rate=('fit', lambda x: (x == 'large').mean()),
        item_total_rentals=('fit', 'size')
    ).reset_index()
    item_feat['item_runs_small'] = (item_feat['item_small_rate'] > 0.20).astype(int)
    item_feat['item_runs_large'] = (item_feat['item_large_rate'] > 0.20).astype(int)

    for df_ in [train_df, test_df]:
        df_.merge(user_feat, on='user_id', how='left')
        df_.merge(item_feat, on='item_id', how='left')

    train_df = train_df.merge(user_feat, on='user_id', how='left')
    train_df = train_df.merge(item_feat, on='item_id', how='left')
    test_df  = test_df.merge(user_feat,  on='user_id', how='left')
    test_df  = test_df.merge(item_feat,  on='item_id', how='left')

    history_cols = ['user_small_rate', 'user_large_rate', 'user_fit_rate',
                    'user_avg_size', 'user_total_rentals',
                    'item_small_rate', 'item_large_rate', 'item_total_rentals',
                    'item_runs_small', 'item_runs_large']

    train_df['is_cold_user'] = train_df['user_small_rate'].isna().astype(int)
    test_df['is_cold_user']  = test_df['user_small_rate'].isna().astype(int)
    train_df['is_cold_item'] = train_df['item_small_rate'].isna().astype(int)
    test_df['is_cold_item']  = test_df['item_small_rate'].isna().astype(int)

    for df_ in [train_df, test_df]:
        df_[history_cols] = df_[history_cols].fillna(0)

    return train_df, test_df

# ─────────────────────────────────────────────
# FEATURE MATRIX
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'age', 'bmi', 'height_inches', 'weight_lbs', 'size_numeric',
    'size_relative_to_peers', 'size_percentile', 'bmi_size_interaction', 'size_per_bmi',
    'user_small_rate', 'user_large_rate', 'user_fit_rate',
    'user_avg_size', 'user_total_rentals',
    'item_small_rate', 'item_large_rate', 'item_total_rentals',
    'item_runs_small', 'item_runs_large',
    'rating_numeric', 'bust_band_size', 'bust_cup_numeric',
    'is_high_rating', 'is_cold_user', 'is_cold_item',
    'text_small', 'text_large', 'text_perfect'
]

# ─────────────────────────────────────────────
# PYTORCH MODEL
# ─────────────────────────────────────────────
class FitPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, n_classes: int, dropout: float):
        super().__init__()
        layers = [nn.BatchNorm1d(input_dim)]
        prev_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h
        layers.append(nn.Linear(prev_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ─────────────────────────────────────────────
# RANKING METRICS
# ─────────────────────────────────────────────
def precision_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / k if k > 0 else 0.0

def recall_at_k(recommended, relevant, k):
    return len(set(recommended[:k]) & set(relevant)) / len(relevant) if relevant else 0.0

def ndcg_at_k(recommended, relevant, k):
    rel_set = set(relevant)
    dcg  = sum(1.0 / np.log2(i + 2) for i, x in enumerate(recommended[:k]) if x in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_ranking(model, X_test, y_test, user_ids, item_ids, k=10, n_users=500):
    """
    Evaluate recommendation ranking quality with Recall@K and NDCG@K.
    For each user, rank their test items by predicted 'fit' probability
    and measure whether truly-fit items appear at the top.
    """
    print(f"\nEvaluating ranking metrics (K={k}, {n_users} sampled users)...")
    model.eval()
    device = next(model.parameters()).device

    X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    # fit_prob = probability of class 1 (fit)
    fit_label  = LABEL_NAMES.index('fit')
    fit_probs  = probs[:, fit_label]

    # Group by user
    test_df_eval = pd.DataFrame({
        'user_id':  user_ids,
        'item_id':  item_ids,
        'y_true':   y_test,
        'fit_prob': fit_probs
    })

    unique_users = test_df_eval['user_id'].unique()
    if len(unique_users) > n_users:
        np.random.seed(42)
        unique_users = np.random.choice(unique_users, n_users, replace=False)

    p_list, r_list, n_list = [], [], []
    for uid in unique_users:
        user_data = test_df_eval[test_df_eval['user_id'] == uid].copy()
        if len(user_data) < 2:
            continue
        # Rank by fit probability descending
        ranked     = user_data.sort_values('fit_prob', ascending=False)['item_id'].tolist()
        relevant   = user_data[user_data['y_true'] == fit_label]['item_id'].tolist()
        if not relevant:
            continue
        p_list.append(precision_at_k(ranked, relevant, k))
        r_list.append(recall_at_k(ranked, relevant, k))
        n_list.append(ndcg_at_k(ranked, relevant, k))

    return {
        f'Precision@{k}': np.mean(p_list),
        f'Recall@{k}':    np.mean(r_list),
        f'NDCG@{k}':      np.mean(n_list),
        'Users evaluated': len(p_list)
    }

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(model, train_loader, val_loader, device, epochs, patience, lr, weight_decay, class_weights):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))

    best_loss, patience_ctr, best_state = float('inf'), 0, None
    train_losses, val_losses = [], []

    print(f"Training for up to {epochs} epochs...")
    start = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"Training done in {time.time()-start:.1f}s ✅")
    return train_losses, val_losses

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main(data_path: str):
    print("=" * 60)
    print("  Clothing Fit Prediction — PyTorch Neural Network")
    print("=" * 60)

    # Load + preprocess
    df = load_data(data_path)
    df = preprocess(df)
    df = engineer_features(df)

    # Encode labels
    le = LabelEncoder()
    le.fit(LABEL_NAMES)
    df['fit_encoded'] = le.transform(df['fit'])

    # Train/test split
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=df['fit_encoded']
    )
    train_df = df.loc[train_idx].copy()
    test_df  = df.loc[test_idx].copy()

    # Add history features
    train_df, test_df = add_history_features(train_df, test_df)

    # Build feature matrices
    feat_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    X_train = train_df[feat_cols].fillna(0).values.astype(np.float32)
    X_test  = test_df[feat_cols].fillna(0).values.astype(np.float32)
    y_train = train_df['fit_encoded'].values
    y_test  = test_df['fit_encoded'].values

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Class weights for imbalanced data
    class_counts  = np.bincount(y_train)
    class_weights = len(y_train) / (len(class_counts) * class_counts)
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    # Weighted sampler for balanced batches
    sample_weights = torch.tensor([class_weights[y] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # DataLoaders
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test,  dtype=torch.float32)
    y_te_t = torch.tensor(y_test,  dtype=torch.long)

    # Val split from train
    val_size = int(0.1 * len(X_tr_t))
    train_dataset = TensorDataset(X_tr_t[:-val_size], y_tr_t[:-val_size])
    val_dataset   = TensorDataset(X_tr_t[-val_size:], y_tr_t[-val_size:])
    test_dataset  = TensorDataset(X_te_t, y_te_t)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    model = FitPredictor(
        input_dim=len(feat_cols),
        hidden_dims=HIDDEN_DIMS,
        n_classes=3,
        dropout=DROPOUT
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_losses, val_losses = train(
        model, train_loader, val_loader, device,
        EPOCHS, PATIENCE, LR, WEIGHT_DECAY, class_weights
    )

    # Evaluate classification
    model.eval()
    all_preds, all_probs = [], []
    with torch.no_grad():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_probs.extend(probs)

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    macro_f1 = f1_score(y_test, all_preds, average='macro')
    roc_auc  = roc_auc_score(y_test, all_probs, multi_class='ovr')

    print("\n" + "=" * 60)
    print("  CLASSIFICATION RESULTS")
    print("=" * 60)
    print(f"  Macro F1     : {macro_f1:.4f}")
    print(f"  ROC-AUC (OVR): {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, all_preds, target_names=LABEL_NAMES))
    print("=" * 60)

    # Evaluate ranking
    ranking_results = evaluate_ranking(
        model, X_test, y_test,
        user_ids=test_df['user_id'].values,
        item_ids=test_df['item_id'].values,
        k=10
    )
    print("\n" + "=" * 60)
    print("  RANKING RESULTS")
    print("=" * 60)
    for metric, value in ranking_results.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print("=" * 60)

    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'scaler': scaler,
        'feature_cols': feat_cols,
        'label_encoder': le,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT
    }, 'models/fit_predictor_nn.pt')
    print("\nModel saved to models/fit_predictor_nn.pt ✅")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='renttherunway_final_data.json.gz')
    args = parser.parse_args()
    main(args.data)
