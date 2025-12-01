"""
Lambda function to call SageMaker endpoint for credit scoring inference
For local testing, uses the CatBoost model directly with SHAP values
"""

import json
import os

import boto3
import joblib
import numpy as np
import pandas as pd
from catboost import Pool

# For local testing, load model directly
# Try multiple possible paths
MODEL_DIRS = [
    os.environ.get("MODEL_DIR", ""),
    "/opt/ml/model",
    "../model_output",
    "model_output",
]
SAGEMAKER_ENDPOINT = os.environ.get("SAGEMAKER_ENDPOINT", "credit-scoring-endpoint")

# Try to load model (for local testing)
model = None
metadata = None
MODEL_LOADED = False

for model_dir in MODEL_DIRS:
    if not model_dir:
        continue
    try:
        model_path = os.path.join(model_dir, "catboost_model.joblib")
        metadata_path = os.path.join(model_dir, "model_metadata.json")
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            model = joblib.load(model_path)
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            MODEL_LOADED = True
            print(f"✓ Loaded CatBoost model from {model_dir}")
            print(
                f"  Using PDO method: factor={metadata.get('factor', 'N/A')}, offset={metadata.get('offset', 'N/A')}"
            )
            break
    except Exception as e:
        print(f"Error loading model from {model_dir}: {e}")
        continue

if not MODEL_LOADED:
    print("⚠ Model not found, will use SageMaker endpoint")
    sagemaker_runtime = boto3.client(
        "sagemaker-runtime", endpoint_url=os.environ.get("LOCALSTACK_ENDPOINT")
    )


def calculate_score_local(customer_features):
    """Calculate score using CatBoost model with SHAP values and PDO method"""
    if not MODEL_LOADED:
        raise ValueError("Model not loaded")

    # Create DataFrame with correct feature order
    feature_names = metadata["feature_names"]
    categorical_features = metadata.get("categorical_features", [])

    # Ensure features are in correct order and handle categoricals
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

    # Create CatBoost Pool (required for get_feature_importance)
    pool = Pool(X, cat_features=cat_feature_indices or None)

    # Get SHAP values (in log-odds space)
    shap_values = model.get_feature_importance(type="ShapValues", data=pool)

    # SHAP values shape: (n_samples, n_features + 1)
    # Last column is base value, first n_features are feature contributions
    feature_shap = shap_values[:, :-1]  # Feature contributions
    base_shap = shap_values[:, -1]  # Base value (expected log-odds)

    # Sum SHAP values to get total log-odds contribution
    log_odds = feature_shap.sum(axis=1)[0] + base_shap[0]

    # Convert to score using PDO formula (REVERSE sign: higher score = better credit)
    # Score = offset + factor * (-log_odds)
    factor = metadata["factor"]
    offset = metadata["offset"]
    score = offset + factor * (-log_odds)

    return float(score)


def invoke_sagemaker_endpoint(customer_features):
    """Invoke SageMaker endpoint (for cloud deployment)"""
    payload = json.dumps({"instances": [list(customer_features.values())]})

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=SAGEMAKER_ENDPOINT, ContentType="application/json", Body=payload
    )

    result = json.loads(response["Body"].read().decode())
    return result["predictions"][0]["score"]


def lambda_handler(event, context):
    """
    Score customers using SageMaker endpoint or local model

    Input: List of customers with features
    Output: List of customers with updated scores
    """
    try:
        customers = event.get("customers", [])

        # For demo, we'll use sample features based on current_score
        # In production, you'd fetch actual features from database
        scored_customers = []

        # Get feature names from metadata
        feature_names = metadata.get("feature_names", []) if MODEL_LOADED else []

        for customer in customers:
            customer_id = customer["customer_id"]
            current_score = customer["current_score"]

            # Generate sample features based on score (for demo)
            # In production, fetch actual features from database
            # Using new feature set from BankCaseStudyData.csv
            sample_features = {}

            # Numerical features - generate based on current_score
            if "Application_Score" in feature_names:
                sample_features["Application_Score"] = current_score
            if "Bureau_Score" in feature_names:
                sample_features["Bureau_Score"] = current_score + np.random.randint(
                    -20, 20
                )
            if "Loan_Amount" in feature_names:
                current_limit = float(customer.get("current_limit", 5000))
                sample_features["Loan_Amount"] = current_limit * 1.2
            if "Time_with_Bank" in feature_names:
                sample_features["Time_with_Bank"] = 24 if current_score > 650 else 12
            if "Time_in_Employment" in feature_names:
                sample_features["Time_in_Employment"] = (
                    36 if current_score > 650 else 18
                )
            if "Loan_to_income" in feature_names:
                sample_features["Loan_to_income"] = (
                    0.25 if current_score > 650 else 0.40
                )
            if "Gross_Annual_Income" in feature_names:
                sample_features["Gross_Annual_Income"] = (
                    50000 if current_score > 650 else 35000
                )

            # Categorical features - use reasonable defaults
            if "Loan_Payment_Frequency" in feature_names:
                sample_features["Loan_Payment_Frequency"] = "M"  # Monthly
            if "Residential_Status" in feature_names:
                sample_features["Residential_Status"] = "H"  # Homeowner
            if "Cheque_Card_Flag" in feature_names:
                sample_features["Cheque_Card_Flag"] = (
                    "Y" if current_score > 650 else "N"
                )
            if "Existing_Customer_Flag" in feature_names:
                sample_features["Existing_Customer_Flag"] = "Y"
            if "Home_Telephone_Number" in feature_names:
                sample_features["Home_Telephone_Number"] = (
                    "Y" if current_score > 650 else "N"
                )

            if MODEL_LOADED:
                # Use local CatBoost model with SHAP values
                new_score = calculate_score_local(sample_features)
            else:
                # Use SageMaker endpoint
                new_score = invoke_sagemaker_endpoint(sample_features)

            scored_customers.append(
                {
                    "customer_id": customer_id,
                    "current_score": current_score,
                    "current_limit": customer["current_limit"],
                    "new_score": new_score,
                }
            )

        print(f"Scored {len(scored_customers)} customers")

        return {"statusCode": 200, "scored_customers": scored_customers}

    except Exception as e:
        print(f"Error in SageMaker inference: {e}")
        return {"statusCode": 500, "error": str(e), "scored_customers": []}
