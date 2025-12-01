"""
SageMaker training script for CatBoost credit scoring model
This runs inside the SageMaker container
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_optimal_cutoff(scores, y_true):
    """
    Find optimal cutoff that maximizes separation between goods and bads
    """
    dataframe_viz = pd.DataFrame({"is_bad": y_true, "score": scores})

    thresholds = sorted(dataframe_viz["score"].unique(), reverse=True)
    thresholds_ar = {"threshold": [], "share_bads": [], "share_goods": [], "delta": []}

    total_defaults = dataframe_viz["is_bad"].sum()
    total_non_defaults = len(dataframe_viz) - total_defaults

    for thresh in thresholds:
        declined = dataframe_viz[dataframe_viz["score"] <= thresh]
        share_bads = (
            declined["is_bad"].sum() / total_defaults if total_defaults > 0 else 0
        )
        share_goods = (
            (len(declined) - declined["is_bad"].sum()) / total_non_defaults
            if total_non_defaults > 0
            else 0
        )

        thresholds_ar["threshold"].append(thresh)
        thresholds_ar["share_bads"].append(share_bads)
        thresholds_ar["share_goods"].append(share_goods)
        thresholds_ar["delta"].append(share_bads - share_goods)

    thresholds_ar_df = pd.DataFrame(thresholds_ar)
    optimal_threshold = thresholds_ar_df.loc[
        thresholds_ar_df["delta"].idxmax(), "threshold"
    ]

    logger.info(f"\nOptimal cutoff: {optimal_threshold:.0f}")
    return int(optimal_threshold)


def train(args):
    """Train CatBoost credit scoring model"""

    logger.info("=" * 80)
    logger.info("CatBoost Credit Scoring Training - SageMaker")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading data from {args.train}")
    train_files = [f for f in os.listdir(args.train) if f.endswith(".csv")]

    if not train_files:
        raise ValueError(f"No CSV files found in {args.train}")

    dataset = pd.read_csv(os.path.join(args.train, train_files[0]))
    logger.info(f"Data shape: {dataset.shape}")

    # Prepare features and labels
    label = "Final_Decision"
    dataset[label] = dataset[label].map({"Accept": 0, "Decline": 1})

    num_features = [
        "Application_Score",
        "Bureau_Score",
        "Loan_Amount",
        "Time_with_Bank",
        "Time_in_Employment",
        "Loan_to_income",
        "Gross_Annual_Income",
    ]

    cat_features = [
        "Loan_Payment_Frequency",
        "Residential_Status",
        "Cheque_Card_Flag",
        "Existing_Customer_Flag",
        "Home_Telephone_Number",
    ]

    features = cat_features + num_features

    # Use split column for train/test split
    ix_train = dataset["split"] == "Development"
    ix_test = dataset["split"] == "Validation"

    X_train = dataset.loc[ix_train, features].copy()
    y_train = dataset.loc[ix_train, label].copy()
    X_test = dataset.loc[ix_test, features].copy()
    y_test = dataset.loc[ix_test, label].copy()

    # Handle categorical features
    X_train.loc[:, cat_features] = X_train.loc[:, cat_features].astype(str).fillna("NA")
    X_test.loc[:, cat_features] = X_test.loc[:, cat_features].astype(str).fillna("NA")

    logger.info(f"Features ({len(features)}): {features}")
    logger.info(f"  Categorical: {cat_features}")
    logger.info(f"  Numerical: {num_features}")
    logger.info(
        f"Target distribution: Decline={y_train.sum()} ({y_train.mean():.1%}), Accept={len(y_train) - y_train.sum()} ({1 - y_train.mean():.1%})"
    )
    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # Identify categorical feature indices for CatBoost
    cat_feature_indices = [features.index(f) for f in cat_features]

    # Train CatBoost model
    logger.info("\nTraining CatBoost model...")
    logger.info(f"  Categorical features (indices): {cat_feature_indices}")
    model = CatBoostClassifier(
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        verbose=50,
        early_stopping_rounds=20,
        cat_features=cat_feature_indices,  # Specify categorical features
    )

    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

    # Evaluate
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    gini = roc_auc_score(y_test, y_pred_proba) * 2 - 1

    logger.info("\nModel Performance:")
    logger.info(f"  Gini: {gini:.4f}")

    # PDO (Points to Double Odds) parameters
    target_score = args.target_score
    target_odds = args.target_odds
    pts_double_odds = args.pts_double_odds

    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)

    logger.info("\nPDO Scorecard Parameters:")
    logger.info(f"  Factor: {factor:.4f}")
    logger.info(f"  Offset: {offset:.4f}")

    # Calculate SHAP values for scoring using PDO method
    logger.info("\nCalculating SHAP values and converting to scores...")

    # Must specify cat_features when creating Pool for SHAP values
    train_pool = Pool(X_train, y_train, cat_features=cat_feature_indices)
    test_pool = Pool(X_test, y_test, cat_features=cat_feature_indices)

    # Get SHAP values (in log-odds space)
    train_shap_values = model.get_feature_importance(type="ShapValues", data=train_pool)
    test_shap_values = model.get_feature_importance(type="ShapValues", data=test_pool)

    # SHAP values shape: (n_samples, n_features + 1)
    train_feature_shap = train_shap_values[:, :-1]
    train_base_shap = train_shap_values[:, -1]

    test_feature_shap = test_shap_values[:, :-1]
    test_base_shap = test_shap_values[:, -1]

    # Sum SHAP values to get total log-odds contribution
    train_log_odds = train_feature_shap.sum(axis=1) + train_base_shap
    test_log_odds = test_feature_shap.sum(axis=1) + test_base_shap

    # Convert to scores using PDO formula (REVERSE sign: higher score = better credit)
    train_scores = offset + factor * (-train_log_odds)
    test_scores = offset + factor * (-test_log_odds)

    logger.info("\nScore Distribution:")
    logger.info(
        f"  Min: {test_scores.min():.0f}, Median: {np.median(test_scores):.0f}, Max: {test_scores.max():.0f}"
    )

    # Find optimal cutoff
    cutoff = find_optimal_cutoff(train_scores, y_train.values)
    logger.info(f"  Optimal cutoff: {cutoff}")

    # Save model
    model_path = os.path.join(args.model_dir, "catboost_model.joblib")
    joblib.dump(model, model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")

    # Save metadata
    metadata = {
        "model_type": "catboost",
        "scoring_method": "shap_values_pdo",
        "target_score": target_score,
        "target_odds": target_odds,
        "pts_double_odds": pts_double_odds,
        "factor": float(factor),
        "offset": float(offset),
        "cutoff": cutoff,
        "gini": float(gini),
        "feature_names": features,
        "categorical_features": cat_features,
        "numerical_features": num_features,
    }

    metadata_path = os.path.join(args.model_dir, "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to: {metadata_path}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # SageMaker environment variables
    parser.add_argument(
        "--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )
    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"),
    )

    # Hyperparameters
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--target-score", type=int, default=600)
    parser.add_argument("--target-odds", type=int, default=30)
    parser.add_argument("--pts-double-odds", type=int, default=20)

    args = parser.parse_args()

    train(args)

