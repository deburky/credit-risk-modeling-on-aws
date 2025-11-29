"""
Train Credit Scorecard using WOE + Logistic Regression
Converts to points-based scorecard with optimal cutoff
Uses real credit data from chapters/chapter_1/data/credit_example.csv
"""

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


def load_data(data_path="../data/credit_example.csv"):
    """Load credit data"""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def create_scorecard_pipeline():
    """Create WOE + Logistic Regression pipeline"""

    # Logistic Regression - no penalty for interpretable scorecard
    logistic = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # Handle imbalanced data
    )

    return Pipeline([("woe", FastWoe()), ("logistic", logistic)])


def convert_to_scorecard(
    pipeline,
    feature_names,
    target_score=600,
    target_odds=30,
    pts_double_odds=20,
):
    """
    Convert logistic regression to points-based scorecard

    Scorecard formula:
    Score = offset + factor * (β0 + β1*WOE1 + β2*WOE2 + ...)

    Where:
    - factor = pts_double_odds / ln(2)
    - offset = target_score - factor * ln(target_odds)

    IMPORTANT: We REVERSE the sign because:
    - Model predicts P(default=1) - higher probability = BAD
    - Scorecard should give higher scores to GOOD applicants
    - So we negate the logit to flip the direction
    """

    logger.info("\n" + "=" * 80)
    logger.info("Converting model to scorecard...")
    logger.info("=" * 80)

    # Get logistic regression coefficients
    logistic = pipeline.named_steps["logistic"]
    intercept = logistic.intercept_[0]
    coefficients = logistic.coef_[0]

    # Calculate scaling factors
    factor = pts_double_odds / np.log(2)
    offset = target_score - factor * np.log(target_odds)

    logger.info("\nScorecard parameters:")
    logger.info(f"  Target score: {target_score}")
    logger.info(f"  Target odds: {target_odds}")
    logger.info(f"  Points to double odds: {pts_double_odds}")
    logger.info(f"  Factor: {factor:.4f}")
    logger.info(f"  Offset: {offset:.4f}")

    # Create scorecard
    scorecard = pd.DataFrame(
        {
            "feature": ["intercept"] + list(feature_names),
            "coefficient": [intercept] + list(coefficients),
        }
    )

    # Calculate points (REVERSE sign for creditworthiness)
    scorecard["points_per_unit"] = factor * (-scorecard["coefficient"])

    # Base score (from intercept)
    base_score = offset + factor * (-intercept)

    logger.info(f"\nBase score (intercept): {base_score:.2f}")
    logger.info("\nFeature contributions (points per unit WOE):")
    logger.info("-" * 80)

    for _, row in scorecard[scorecard["feature"] != "intercept"].iterrows():
        logger.info(f"  {row['feature']:20s}: {row['points_per_unit']:+8.2f} points")

    return scorecard, base_score, factor, offset


def calculate_scores(X, pipeline, scorecard, base_score):
    """Calculate credit scores for all samples"""

    # Transform to WOE
    X_transformed = pipeline.named_steps["woe"].transform(X)

    # Get predictions (log-odds)
    logistic = pipeline.named_steps["logistic"]
    log_odds = logistic.decision_function(X_transformed)

    # Convert to scores (REVERSE sign)
    factor = scorecard.iloc[0]["points_per_unit"] / (-scorecard.iloc[0]["coefficient"])
    offset = base_score - factor * (-logistic.intercept_[0])

    return offset - factor * log_odds


def find_optimal_cutoff(scores, y_true):
    """
    Find optimal cutoff that maximizes separation between goods and bads

    Strategy: Find threshold where difference between %bads captured and %goods captured is maximum
    """

    dataframe_viz = pd.DataFrame({"is_bad": y_true, "score": scores})

    # Selecting the threshold
    thresholds = sorted(dataframe_viz["score"].unique(), reverse=True)

    thresholds_ar = {"threshold": [], "share_bads": [], "share_goods": [], "delta": []}

    total_defaults = dataframe_viz["is_bad"].sum()
    total_non_defaults = len(dataframe_viz) - total_defaults

    for thresh in thresholds:
        # Declined (score <= threshold)
        declined = dataframe_viz[dataframe_viz["score"] <= thresh]
        share_bads = declined["is_bad"].sum() / total_defaults
        share_goods = (1 - declined["is_bad"]).sum() / total_non_defaults

        thresholds_ar["threshold"].append(thresh)
        thresholds_ar["share_bads"].append(share_bads)
        thresholds_ar["share_goods"].append(share_goods)
        thresholds_ar["delta"].append(share_bads - share_goods)

    thresholds_ar_df = pd.DataFrame(thresholds_ar)

    # Find threshold with maximum delta (most bads captured, fewest goods captured)
    optimal_threshold = thresholds_ar_df.loc[
        thresholds_ar_df["delta"].idxmax(), "threshold"
    ]

    logger.info("\nOptimal threshold analysis:")
    logger.info(f"  Optimal cutoff: {optimal_threshold:.0f}")
    logger.info(f"  Max delta: {thresholds_ar_df['delta'].max():.2%}")

    return int(optimal_threshold), thresholds_ar_df


def main():
    logger.info("=" * 80)
    logger.info("Credit Scorecard Training - WOE + Logistic Regression")
    logger.info("=" * 80)

    # Load data
    df = load_data()

    # Prepare features and target
    # Target: Good_Bad (0=bad, 1=good) - we'll predict bad (default)
    target_col = "Good_Bad"

    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Reverse target: 0=good, 1=bad (for default prediction)
    y = 1 - y

    feature_names = X.columns.tolist()
    logger.info(f"\nFeatures ({len(feature_names)}): {feature_names}")
    logger.info(f"Target: {target_col} (reversed: 0=good/no default, 1=bad/default)")
    logger.info(
        f"Target distribution: Default={y.sum()} ({y.mean():.1%}), Good={len(y) - y.sum()} ({1 - y.mean():.1%})"
    )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

    # Train pipeline
    logger.info("\nTraining WOE + Logistic Regression pipeline...")
    pipeline = create_scorecard_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    gini = roc_auc_score(y_test, y_pred_proba) * 2 - 1
    logger.info("Model Performance:")
    logger.info(f"{'=' * 80}")
    logger.info(f"Gini score: {gini:.4f}")

    # Convert to scorecard
    scorecard, base_score, factor, offset = convert_to_scorecard(
        pipeline,
        feature_names,
        target_score=600,
        target_odds=30,
        pts_double_odds=20,
    )

    # Calculate scores
    logger.info("\nCalculating credit scores...")
    train_scores = calculate_scores(X_train, pipeline, scorecard, base_score)
    test_scores = calculate_scores(X_test, pipeline, scorecard, base_score)

    logger.info("Score Distribution:")
    logger.info(f"{'=' * 80}")
    logger.info(f"  Min:     {test_scores.min():.0f}")
    logger.info(f"  25%:     {np.percentile(test_scores, 25):.0f}")
    logger.info(f"  Median:  {np.median(test_scores):.0f}")
    logger.info(f"  75%:     {np.percentile(test_scores, 75):.0f}")
    logger.info(f"  Max:     {test_scores.max():.0f}")

    # Find optimal cutoff
    cutoff, threshold_df = find_optimal_cutoff(train_scores, y_train.values)

    # Save artifacts
    os.makedirs("model_output", exist_ok=True)

    logger.info(f"\n{'=' * 80}")
    logger.info("Saving artifacts...")
    logger.info(f"{'=' * 80}")

    # Save pipeline
    joblib.dump(pipeline, "model_output/scorecard_pipeline.joblib")
    logger.info("✓ Pipeline: model_output/scorecard_pipeline.joblib")

    # Save scorecard
    scorecard.to_csv("model_output/scorecard.csv", index=False)
    logger.info("✓ Scorecard: model_output/scorecard.csv")

    # Save threshold analysis
    threshold_df.to_csv("model_output/threshold_analysis.csv", index=False)
    logger.info("✓ Threshold analysis: model_output/threshold_analysis.csv")

    # Save metadata
    metadata = {
        "target_score": 600,
        "target_odds": 30,
        "pts_double_odds": 20,
        "factor": float(factor),
        "offset": float(offset),
        "base_score": float(base_score),
        "cutoff": cutoff,
        "gini": float(gini),
        "feature_names": feature_names,
    }

    with open("model_output/scorecard_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("✓ Metadata: model_output/scorecard_metadata.json")

    logger.info(f"\n{'=' * 80}")
    logger.info("✓ Scorecard training complete!")
    logger.info(f"{'=' * 80}\n")

    return pipeline, scorecard, metadata


if __name__ == "__main__":
    pipeline, scorecard, metadata = main()
