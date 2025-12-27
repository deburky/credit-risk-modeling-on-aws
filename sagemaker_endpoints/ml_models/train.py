#!/usr/bin/env python3
"""
Reusable CatBoost model training script.

Can be used:
1. Directly: python train.py --data-dir ./data --output-dir ./model_output
2. As SageMaker entry point: entry_point="train.py"
3. As a module: from train import train_model
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


def get_feature_definitions():
    """Get feature names and categorical features."""
    feature_names = [
        "Application_Score",
        "Bureau_Score",
        "Loan_Amount",
        "Time_with_Bank",
        "Time_in_Employment",
        "Loan_to_income",
        "Gross_Annual_Income",
    ]
    categorical_features = [
        "Loan_Payment_Frequency",
        "Residential_Status",
        "Cheque_Card_Flag",
        "Existing_Customer_Flag",
        "Home_Telephone_Number",
    ]
    return feature_names, categorical_features


def load_training_data(data_dir: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load training data from CSV file.

    Args:
        data_dir: Directory containing training data CSV

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    data_dir = Path(data_dir)
    csv_files = list[Path](data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    # Use first CSV file found
    data_file = csv_files[0]
    print(f"Loading training data from {data_file}")

    df = pd.read_csv(data_file)

    feature_names, categorical_features = get_feature_definitions()
    all_features = feature_names + categorical_features

    if missing := [f for f in all_features if f not in df.columns]:
        raise ValueError(f"Missing required features: {missing}")

    target_col = next(
        (col for col in ["target", "is_bad", "default", "y"] if col in df.columns),
        None,
    )
    if target_col is None:
        raise ValueError(
            "Could not find target column. Expected one of: target, is_bad, default, y"
        )

    X = df[all_features].copy()
    y = df[target_col].copy()

    # Type assertions for type checker
    assert isinstance(X, pd.DataFrame), "X must be a DataFrame"
    assert isinstance(y, pd.Series), "y must be a Series"

    print(f"Loaded {len(X)} samples with {len(all_features)} features")
    return X, y


def generate_dummy_data(
    n_samples: int = 100, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate dummy training data for testing.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    np.random.seed(seed)
    feature_names, categorical_features = get_feature_definitions()

    X = pd.DataFrame(
        {
            **{name: np.random.randn(n_samples) for name in feature_names},
            **{
                name: np.random.choice(["Y", "N", "M", "H"], n_samples)
                for name in categorical_features
            },
        }
    )
    # Generate binary classification targets (0/1) for CatBoostClassifier
    # Higher scores should correlate with lower default probability
    # Create a simple rule: if Application_Score + Bureau_Score > threshold, then low default risk
    score_sum = X["Application_Score"] + X["Bureau_Score"]
    threshold = score_sum.median()
    y = pd.Series((score_sum < threshold).astype(int))  # 1 = default/bad, 0 = good

    return X, y


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: str | Path,
    iterations: int = 100,
    depth: int = 6,
    learning_rate: float = 0.1,
    target_score: float = 600.0,
    target_odds: float = 30.0,
    pts_double_odds: float = 20.0,
    random_seed: int = 42,
    verbose: bool = True,
) -> tuple[CatBoostClassifier, dict]:
    """
    Train CatBoost model and save artifacts.

    Args:
        X: Training features DataFrame
        y: Training target Series
        output_dir: Directory to save model and metadata
        iterations: Number of boosting iterations
        depth: Tree depth
        learning_rate: Learning rate
        target_score: Target credit score
        target_odds: Target odds
        pts_double_odds: Points to double odds
        random_seed: Random seed
        verbose: Whether to print training progress

    Returns:
        Tuple of (trained_model, metadata_dict)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names, categorical_features = get_feature_definitions()
    all_features = feature_names + categorical_features

    # Get categorical feature indices
    cat_indices = [
        all_features.index(c) for c in categorical_features if c in all_features
    ]

    # Train model
    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        verbose=verbose,
        random_seed=random_seed,
    )

    if verbose:
        print(
            f"Training model with {iterations} iterations, depth={depth}, lr={learning_rate}"
        )

    model.fit(X, y, cat_features=cat_indices or None)

    # Save model
    model_path = output_dir / "catboost_model.joblib"
    joblib.dump(model, model_path)
    if verbose:
        print(f"Saved model to {model_path}")

    # Create metadata
    metadata = {
        "feature_names": all_features,
        "categorical_features": categorical_features,
        "factor": 20.0,
        "offset": 600.0,
        "training_params": {
            "iterations": iterations,
            "depth": depth,
            "learning_rate": learning_rate,
            "target_score": target_score,
            "target_odds": target_odds,
            "pts_double_odds": pts_double_odds,
        },
    }

    metadata_path = output_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"Saved metadata to {metadata_path}")

    return model, metadata


def main():
    """Main entry point for SageMaker training or direct execution."""
    parser = argparse.ArgumentParser(description="Train CatBoost credit scoring model")

    # SageMaker passes these arguments
    parser.add_argument("--data-dir", type=str, default="/opt/ml/input/data/train")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/model")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--target-score", type=float, default=600.0)
    parser.add_argument("--target-odds", type=float, default=30.0)
    parser.add_argument("--pts-double-odds", type=float, default=20.0)
    parser.add_argument(
        "--dummy-data",
        type=str,
        default="False",
        help="Generate dummy data for testing (True/False)",
    )
    parser.add_argument(
        "--dummy-samples", type=int, default=100, help="Number of dummy samples"
    )

    args = parser.parse_args()

    try:
        # Load or generate data
        # Check for dummy-data hyperparameter (SageMaker passes it as --dummy-data True)
        use_dummy_data = str(args.dummy_data).lower() in {"true", "1", "yes", "on"}
        if use_dummy_data:
            print("Generating dummy training data...")
            X, y = generate_dummy_data(n_samples=args.dummy_samples)
        else:
            try:
                X, y = load_training_data(args.data_dir)
            except FileNotFoundError:
                # If no data found and dummy-data not explicitly set, use dummy data as fallback
                print("No training data found, generating dummy data as fallback...")
                X, y = generate_dummy_data(n_samples=args.dummy_samples)

        # Train model
        model, metadata = train_model(
            X=X,
            y=y,
            output_dir=args.output_dir,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            target_score=args.target_score,
            target_odds=args.target_odds,
            pts_double_odds=args.pts_double_odds,
        )

        print("Training completed successfully!")
        return 0

    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
