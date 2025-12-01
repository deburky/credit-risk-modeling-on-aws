"""
Lambda function for credit scoring
Loads trained scorecard model from S3 and scores applications
"""

import base64
import json
import os
import tempfile
import uuid
from datetime import datetime

import boto3
import joblib
import pandas as pd

# Initialize AWS clients
s3_client = boto3.client(
    "s3", endpoint_url=os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566")
)
dynamodb_client = boto3.client(
    "dynamodb",
    endpoint_url=os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566"),
)

# Environment variables
DYNAMODB_TABLE = os.environ.get("DYNAMODB_TABLE", "ApprovedApplications")
S3_BUCKET = os.environ.get("S3_BUCKET", "credit-scoring-models")
MODEL_S3_KEY = os.environ.get("MODEL_S3_KEY", "models/model.tar.gz")

# Global variables for model caching
_model = None
_metadata = None


def load_model_from_s3():  # sourcery skip: extract-method
    """Load model and metadata from S3 (with caching)"""
    global _model, _metadata

    if _model is not None and _metadata is not None:
        return _model, _metadata

    print(f"Loading model from s3://{S3_BUCKET}/{MODEL_S3_KEY}...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Download model.tar.gz from S3
        model_tar_path = os.path.join(tmpdir, "model.tar.gz")
        s3_client.download_file(S3_BUCKET, MODEL_S3_KEY, model_tar_path)

        # Extract tar.gz
        import tarfile

        with tarfile.open(model_tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)

        # Load model and metadata
        model_path = os.path.join(tmpdir, "model.joblib")
        metadata_path = os.path.join(tmpdir, "metadata.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found in archive: {model_path}")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found in archive: {metadata_path}"
            )

        _model = joblib.load(model_path)

        with open(metadata_path, "r") as f:
            _metadata = json.load(f)

        print(
            f"✓ Loaded model (Gini: {_metadata['gini']:.4f}, Cutoff: {_metadata['cutoff']})"
        )

        return _model, _metadata


def calculate_score(features_dict, pipeline, metadata):
    """
    Calculate credit score using the trained scorecard model

    Args:
        features_dict: Dictionary with feature names and values
        pipeline: Trained sklearn pipeline (WOE + Logistic Regression)
        metadata: Model metadata with scorecard parameters

    Returns:
        Credit score (300-850 scale)
    """
    # Create DataFrame from features
    feature_names = metadata["feature_names"]
    X = pd.DataFrame([{name: features_dict.get(name, 0) for name in feature_names}])

    # Transform to WOE
    X_woe = pipeline.named_steps["woe"].transform(X)

    # Get log-odds from logistic regression
    logistic = pipeline.named_steps["logistic"]
    log_odds = logistic.decision_function(X_woe)

    # Convert to credit scores using scorecard formula
    # Score = offset - factor * log_odds (reversed sign for creditworthiness)
    factor = metadata["factor"]
    base_score = metadata["base_score"]
    offset = base_score - factor * (-logistic.intercept_[0])

    score = offset - factor * log_odds[0]

    return float(score)


def score_application(payload):  # sourcery skip: extract-method
    """
    Score a loan application using the trained scorecard model

    Args:
        payload: CSV string with feature values in order:
                 Mortgage, Balance, Amount Past Due, Delinquency,
                 Inquiry, Open Trade, Utilization, Demographic

    Returns:
        dict with 'score' and 'decision'
    """
    try:
        # Parse CSV string
        values = [float(x.strip()) for x in payload.split(",")]

        # Load model (cached after first load)
        pipeline, metadata = load_model_from_s3()

        # Map values to feature names
        feature_names = metadata["feature_names"]
        if len(values) != len(feature_names):
            raise ValueError(
                f"Expected {len(feature_names)} features, got {len(values)}"
            )

        features_dict = dict(zip(feature_names, values))

        # Calculate score
        score = calculate_score(features_dict, pipeline, metadata)

        # Ensure score is in valid range
        score = max(300, min(850, score))

        # Decision based on cutoff
        cutoff = metadata["cutoff"]
        decision = "APPROVED" if score >= cutoff else "DECLINED"

        return {"score": score, "decision": decision, "cutoff": cutoff}

    except Exception as e:
        print(f"Error in credit scoring: {e}")
        import traceback

        traceback.print_exc()
        return {"score": 500, "decision": "DECLINED", "error": str(e)}


def store_approved_application(application_id, score, payload, decision):
    """Store approved application in DynamoDB"""
    try:
        return dynamodb_client.put_item(
            TableName=DYNAMODB_TABLE,
            Item={
                "application_id": {"S": application_id},
                "timestamp": {"S": datetime.now().isoformat()},
                "credit_score": {"N": str(int(score))},
                "decision": {"S": decision},
                "application_data": {"S": str(payload)},
            },
        )
    except Exception as e:
        print(f"Error storing in DynamoDB: {e}")
        return None


def lambda_handler(event, context):
    """Process loan applications from Kinesis stream"""
    declined_count = 0
    approved_count = 0
    records_processed = 0

    for record in event.get("Records", []):
        try:
            # Kinesis data is base64 encoded
            payload = base64.b64decode(record["kinesis"]["data"]).decode("utf-8")
            records_processed += 1

            # Generate application ID
            application_id = str(uuid.uuid4())

            # Score the application using trained model
            result = score_application(payload)
            score = result["score"]
            decision = result["decision"]

            # Store APPROVED applications in DynamoDB
            if decision == "APPROVED":
                store_approved_application(application_id, score, payload, decision)
                approved_count += 1
                print(
                    f"✅ APPROVED - Application {application_id[:8]} (score: {score:.0f})"
                )
            else:
                declined_count += 1
                print(
                    f"❌ DECLINED - Application {application_id[:8]} (score: {score:.0f})"
                )

        except Exception as e:
            print(f"Error processing record: {e}")
            import traceback

            traceback.print_exc()
            continue

    message = f"Processed {records_processed} applications: {approved_count} approved, {declined_count} declined"
    print(message)

    return {
        "statusCode": 200,
        "body": json.dumps(
            {
                "message": message,
                "records_processed": records_processed,
                "approved": approved_count,
                "declined": declined_count,
            }
        ),
    }
