"""
SageMaker training script for credit scorecard
This runs inside the SageMaker container
"""

import argparse
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd
from fastwoe import FastWoe
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(args):
    """Train credit scorecard model"""
    
    logger.info("=" * 80)
    logger.info("Credit Scorecard Training - SageMaker")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading data from {args.train}")
    train_files = [f for f in os.listdir(args.train) if f.endswith('.csv')]
    
    if not train_files:
        raise ValueError(f"No CSV files found in {args.train}")
    
    df = pd.read_csv(os.path.join(args.train, train_files[0]))
    logger.info(f"Data shape: {df.shape}")
    
    # Prepare features and target
    target_col = "Good_Bad"
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Reverse target: 0=good, 1=bad (for default prediction)
    y = 1 - y
    
    feature_names = X.columns.tolist()
    logger.info(f"Features: {feature_names}")
    logger.info(f"Target distribution: Default={y.sum()} ({y.mean():.1%})")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Create pipeline
    logger.info("\nTraining WOE + Logistic Regression pipeline...")
    
    logistic = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    
    pipeline = Pipeline([("woe", FastWoe()), ("logistic", logistic)])
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    gini = roc_auc_score(y_test, y_pred_proba) * 2 - 1
    
    logger.info(f"\nModel Performance:")
    logger.info(f"  Gini: {gini:.4f}")
    
    # Convert to scorecard
    logger.info("\nConverting to scorecard...")
    
    logistic = pipeline.named_steps["logistic"]
    intercept = logistic.intercept_[0]
    coefficients = logistic.coef_[0]
    
    # Scorecard parameters
    target_score = args.target_score
    target_odds = args.target_odds
    pts_double_odds = args.pts_double_odds
    
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)
    base_score = offset + factor * (-intercept)
    
    logger.info(f"  Base score: {base_score:.2f}")
    logger.info(f"  Factor: {factor:.4f}")
    
    # Calculate optimal cutoff
    X_train_woe = pipeline.named_steps["woe"].transform(X_train)
    log_odds = logistic.decision_function(X_train_woe)
    train_scores = offset - factor * log_odds
    
    # Find optimal cutoff
    cutoff = int(np.median(train_scores))
    logger.info(f"  Optimal cutoff: {cutoff}")
    
    # Save model
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    logger.info(f"\n✓ Model saved to: {model_path}")
    
    # Save metadata
    metadata = {
        "target_score": target_score,
        "target_odds": target_odds,
        "pts_double_odds": pts_double_odds,
        "factor": float(factor),
        "offset": float(offset),
        "base_score": float(base_score),
        "cutoff": cutoff,
        "gini": float(gini),
        "feature_names": feature_names,
    }
    
    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to: {metadata_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output"))
    
    # Hyperparameters
    parser.add_argument("--target-score", type=int, default=600)
    parser.add_argument("--target-odds", type=int, default=30)
    parser.add_argument("--pts-double-odds", type=int, default=20)
    
    args = parser.parse_args()
    
    train(args)






