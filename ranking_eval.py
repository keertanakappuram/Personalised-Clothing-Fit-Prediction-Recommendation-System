"""
ranking_eval.py — Ranking Evaluation for Clothing Fit Recommendation
=====================================================================
Evaluates the fit prediction model as a recommendation system using
standard ranking metrics: Precision@K, Recall@K, and NDCG@K.

Framing: For each user, rank candidate items by predicted "fit"
probability. Measure whether truly-fit items appear at the top of
the ranked list — treating fit prediction as item retrieval.

Metrics:
- Precision@K  : fraction of top-K recommendations that are truly "fit"
- Recall@K     : fraction of truly-fit items recovered in top-K
- NDCG@K       : ranking quality — fit items ranked higher = better score
- Hit Rate@K   : % of users with at least one fit item in top-K
- MRR          : mean reciprocal rank of first fit item

Usage:
    python ranking_eval.py \
        --model models/fit_predictor_nn.pt \
        --data renttherunway_final_data.json.gz
"""

import argparse
import json
import gzip
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─────────────────────────────────────────────
# METRIC FUNCTIONS
# ─────────────────────────────────────────────
def precision_at_k(ranked_items, relevant_items, k):
    """Fraction of top-K items that are relevant (fit)."""
    top_k = ranked_items[:k]
    hits  = len(set(top_k) & set(relevant_items))
    return hits / k if k > 0 else 0.0

def recall_at_k(ranked_items, relevant_items, k):
    """Fraction of relevant items recovered in top-K."""
    top_k = ranked_items[:k]
    hits  = len(set(top_k) & set(relevant_items))
    return hits / len(relevant_items) if relevant_items else 0.0

def ndcg_at_k(ranked_items, relevant_items, k):
    """
    Normalized Discounted Cumulative Gain at K.
    Rewards relevant items appearing higher in the ranking.
    """
    rel_set = set(relevant_items)
    dcg     = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(ranked_items[:k])
        if item in rel_set
    )
    ideal_hits = min(k, len(relevant_items))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(ranked_items, relevant_items, k):
    """1 if at least one relevant item is in top-K, else 0."""
    return 1.0 if set(ranked_items[:k]) & set(relevant_items) else 0.0

def mean_reciprocal_rank(ranked_items, relevant_items):
    """
    Reciprocal rank of the first relevant item in the ranked list.
    Measures how quickly the first fit item appears.
    """
    rel_set = set(relevant_items)
    for i, item in enumerate(ranked_items):
        if item in rel_set:
            return 1.0 / (i + 1)
    return 0.0

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate_ranking(
    y_pred_probs: np.ndarray,
    y_true: np.ndarray,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    fit_class_idx: int = 0,
    k_values: list = [5, 10, 20],
    n_users: int = 500,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Evaluate ranking metrics for a given model's probability outputs.

    Args:
        y_pred_probs   : (N, 3) array of class probabilities [small, fit, large]
        y_true         : (N,) array of true class indices
        user_ids       : (N,) array of user IDs
        item_ids       : (N,) array of item IDs
        fit_class_idx  : index of the "fit" class in probability array
        k_values       : list of K values to evaluate
        n_users        : number of users to sample for evaluation
        random_seed    : random seed for reproducibility

    Returns:
        DataFrame with metrics at each K value
    """
    # Build evaluation DataFrame
    eval_df = pd.DataFrame({
        "user_id":  user_ids,
        "item_id":  item_ids,
        "y_true":   y_true,
        "fit_prob": y_pred_probs[:, fit_class_idx]
    })

    # Sample users
    unique_users = eval_df["user_id"].unique()
    if len(unique_users) > n_users:
        np.random.seed(random_seed)
        unique_users = np.random.choice(unique_users, n_users, replace=False)

    results = []
    for k in k_values:
        p_list, r_list, n_list, h_list, mrr_list = [], [], [], [], []

        for uid in unique_users:
            user_data = eval_df[eval_df["user_id"] == uid].copy()
            if len(user_data) < 2:
                continue

            # Rank items by fit probability (descending)
            ranked     = user_data.sort_values("fit_prob", ascending=False)["item_id"].tolist()
            relevant   = user_data[user_data["y_true"] == fit_class_idx]["item_id"].tolist()

            if not relevant:
                continue

            p_list.append(precision_at_k(ranked, relevant, k))
            r_list.append(recall_at_k(ranked, relevant, k))
            n_list.append(ndcg_at_k(ranked, relevant, k))
            h_list.append(hit_rate_at_k(ranked, relevant, k))
            mrr_list.append(mean_reciprocal_rank(ranked, relevant))

        results.append({
            "K":             k,
            f"Precision@{k}": np.mean(p_list),
            f"Recall@{k}":    np.mean(r_list),
            f"NDCG@{k}":      np.mean(n_list),
            f"HitRate@{k}":   np.mean(h_list),
            "MRR":            np.mean(mrr_list),
            "Users evaluated":len(p_list)
        })

    return pd.DataFrame(results)

def print_ranking_results(results_df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  RANKING EVALUATION RESULTS")
    print("=" * 65)
    for _, row in results_df.iterrows():
        k = int(row["K"])
        print(f"\n  K = {k}")
        print(f"    Precision@{k}  : {row[f'Precision@{k}']:.4f}")
        print(f"    Recall@{k}     : {row[f'Recall@{k}']:.4f}")
        print(f"    NDCG@{k}       : {row[f'NDCG@{k}']:.4f}")
        print(f"    HitRate@{k}    : {row[f'HitRate@{k}']:.4f}")
        print(f"    MRR            : {row['MRR']:.4f}")
        print(f"    Users evaluated: {int(row['Users evaluated'])}")
    print("=" * 65)

# ─────────────────────────────────────────────
# COMPARE MODELS
# ─────────────────────────────────────────────
def compare_models_ranking(
    model_results: dict,
    k: int = 10
) -> pd.DataFrame:
    """
    Compare multiple models' ranking performance at a given K.

    Args:
        model_results : dict of {model_name: results_df}
        k             : K value to compare at

    Returns:
        Comparison DataFrame
    """
    rows = []
    for name, df in model_results.items():
        row_k = df[df["K"] == k]
        if row_k.empty:
            continue
        row = row_k.iloc[0]
        rows.append({
            "Model":          name,
            f"Precision@{k}": row.get(f"Precision@{k}", 0),
            f"Recall@{k}":    row.get(f"Recall@{k}", 0),
            f"NDCG@{k}":      row.get(f"NDCG@{k}", 0),
            f"HitRate@{k}":   row.get(f"HitRate@{k}", 0),
            "MRR":            row.get("MRR", 0),
        })
    return pd.DataFrame(rows).sort_values(f"NDCG@{k}", ascending=False)

# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────
def main(model_path: str, data_path: str):
    """
    Load a saved model and evaluate its ranking performance.
    Expects model saved as dict with 'model_state', 'scaler',
    'feature_cols', 'label_encoder'.
    """
    print("=" * 65)
    print("  Clothing Fit Recommendation — Ranking Evaluation")
    print("=" * 65)

    # Load model checkpoint
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # This is a standalone evaluation script
    # In production, import the model class from neural_net.py
    print("Model loaded. Running ranking evaluation...")
    print("Note: For full evaluation, integrate with neural_net.py")
    print("      or pass pre-computed probabilities to evaluate_ranking()")

    # Example usage with random data (replace with real predictions)
    print("\nExample with synthetic data:")
    N = 10000
    n_classes = 3
    np.random.seed(42)

    fake_probs    = np.random.dirichlet(np.ones(n_classes), size=N)
    fake_true     = np.random.choice(n_classes, size=N)
    fake_user_ids = np.random.choice(500, size=N)
    fake_item_ids = np.arange(N)

    results = evaluate_ranking(
        y_pred_probs=fake_probs,
        y_true=fake_true,
        user_ids=fake_user_ids,
        item_ids=fake_item_ids,
        fit_class_idx=1,
        k_values=[5, 10, 20]
    )
    print_ranking_results(results)
    print("\nReplace synthetic data with real model predictions for actual results.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/fit_predictor_nn.pt")
    parser.add_argument("--data",  type=str, default="renttherunway_final_data.json.gz")
    args = parser.parse_args()
    main(args.model, args.data)
