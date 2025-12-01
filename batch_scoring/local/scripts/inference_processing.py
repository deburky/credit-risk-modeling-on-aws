"""
Processing script for SageMaker inference using CatBoost model
"""

import argparse
import json
import os
import sys

# Install dependencies if needed
try:
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import Pool
except ImportError:
    import subprocess

    print("Installing required packages...")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "numpy",
            "pandas",
            "joblib",
            "catboost",
        ]
    )
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import Pool


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/model")
    return parser.parse_args()


def load_model(model_dir):
    """Load CatBoost model and metadata"""
    import tarfile
    
    # Check if model is in a tar.gz file (from SageMaker training)
    tar_files = [f for f in os.listdir(model_dir) if f.endswith(".tar.gz")]
    if tar_files:
        # Extract the tar.gz file
        tar_path = os.path.join(model_dir, tar_files[0])
        print(f"Extracting model from {tar_path}...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        print(f"✓ Model extracted")
    
    model_path = os.path.join(model_dir, "catboost_model.joblib")
    metadata_path = os.path.join(model_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Model files not found in {model_dir}")

    model = joblib.load(model_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    return model, metadata


def calculate_score(model, metadata, customer_features):
    """Calculate score using CatBoost model with SHAP values"""
    feature_names = metadata["feature_names"]
    categorical_features = metadata.get("categorical_features", [])

    # Create DataFrame with correct feature order
    X_dict = {name: customer_features.get(name, 0) for name in feature_names}
    X = pd.DataFrame([X_dict])

    # Convert categorical features to string
    for cat_feat in categorical_features:
        if cat_feat in X.columns:
            X[cat_feat] = X[cat_feat].astype(str).fillna("NA")

    # Get categorical feature indices
    cat_feature_indices = [
        feature_names.index(f) for f in categorical_features if f in feature_names
    ]

    # Create CatBoost Pool
    pool = Pool(X, cat_features=cat_feature_indices or None)

    # Get SHAP values
    shap_values = model.get_feature_importance(type="ShapValues", data=pool)

    # SHAP values shape: (n_samples, n_features + 1)
    feature_shap = shap_values[:, :-1]  # Feature contributions
    base_shap = shap_values[:, -1]  # Base value

    # Sum SHAP values to get total log-odds
    log_odds = feature_shap.sum(axis=1)[0] + base_shap[0]

    # Convert to score using PDO formula
    factor = metadata["factor"]
    offset = metadata["offset"]
    score = offset + factor * (-log_odds)

    return float(score)


def generate_sample_features(customer, feature_names):
    """Generate sample features based on customer data"""
    current_score = customer["current_score"]
    current_limit = float(customer.get("current_limit", 5000))

    features = {}

    # Numerical features
    if "Application_Score" in feature_names:
        features["Application_Score"] = current_score
    if "Bureau_Score" in feature_names:
        features["Bureau_Score"] = current_score + np.random.randint(-20, 20)
    if "Loan_Amount" in feature_names:
        features["Loan_Amount"] = current_limit * 1.2
    if "Time_with_Bank" in feature_names:
        features["Time_with_Bank"] = 24 if current_score > 650 else 12
    if "Time_in_Employment" in feature_names:
        features["Time_in_Employment"] = 36 if current_score > 650 else 18
    if "Loan_to_income" in feature_names:
        features["Loan_to_income"] = 0.25 if current_score > 650 else 0.40
    if "Gross_Annual_Income" in feature_names:
        features["Gross_Annual_Income"] = 50000 if current_score > 650 else 35000

    # Categorical features
    if "Loan_Payment_Frequency" in feature_names:
        features["Loan_Payment_Frequency"] = "M"
    if "Residential_Status" in feature_names:
        features["Residential_Status"] = "H"
    if "Cheque_Card_Flag" in feature_names:
        features["Cheque_Card_Flag"] = "Y" if current_score > 650 else "N"
    if "Existing_Customer_Flag" in feature_names:
        features["Existing_Customer_Flag"] = "Y"
    if "Home_Telephone_Number" in feature_names:
        features["Home_Telephone_Number"] = "Y" if current_score > 650 else "N"

    return features


def main():
    """Main processing function"""
    args = parse_args()

    # Load model
    print(f"Loading model from {args.model_dir}...")
    model, metadata = load_model(args.model_dir)
    print("✓ Model loaded")

    # Load customers from input (handle S3 download structure)
    input_dir = "/opt/ml/processing/input/customers"

    # Find the JSON file (could be in subdirectories from S3)
    customers_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                customers_file = os.path.join(root, file)
                break
        if customers_file:
            break

    if not customers_file:
        # Try direct path
        customers_file = os.path.join(input_dir, "customers.json")

    print(f"Loading customers from {customers_file}...")
    with open(customers_file, "r") as f:
        customers = json.load(f)

    print(f"Found {len(customers)} customers to score")

    # Score customers
    feature_names = metadata.get("feature_names", [])
    scored_customers = []

    for customer in customers:
        # Generate features
        features = generate_sample_features(customer, feature_names)

        # Calculate score
        new_score = calculate_score(model, metadata, features)

        scored_customers.append(
            {
                "customer_id": customer["customer_id"],
                "current_score": customer["current_score"],
                "current_limit": customer["current_limit"],
                "new_score": new_score,
            }
        )

    print(f"✓ Scored {len(scored_customers)} customers")

    # Save to output
    output_path = "/opt/ml/processing/output/scored_customers.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(scored_customers, f, indent=2)

    print(f"✓ Scored customers saved to {output_path}")


if __name__ == "__main__":
    main()
