"""
Training script for SageMaker Local Mode with MLflow tracking
This runs inside the SageMaker container during training
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# Configure logging BEFORE importing mlflow to suppress request header warnings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Filter to suppress MLflow request header warnings
class SuppressMLflowHeaderWarning(logging.Filter):
    def filter(self, record):
        return (
            "Encountered unexpected error during resolving request headers"
            not in record.getMessage()
        )


# Suppress MLflow request header registry warnings (harmless but noisy)
# Set level to ERROR for the specific logger and all parent loggers
header_filter = SuppressMLflowHeaderWarning()
for logger_name in [
    "mlflow.tracking.request_header.registry",
    "mlflow.tracking.request_header",
    "mlflow.tracking",
]:
    mlflow_logger = logging.getLogger(logger_name)
    mlflow_logger.setLevel(logging.ERROR)
    mlflow_logger.addFilter(header_filter)


def train():
    """Training function that runs in SageMaker container."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow-tracking-uri", type=str, required=True)
    parser.add_argument("--mlflow-experiment", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    args = parser.parse_args()

    # Configure MLflow S3 environment (for LocalStack S3)
    # These are set via SageMaker environment variables, ensure defaults if missing
    os.environ.setdefault("MLFLOW_S3_IGNORE_TLS", "true")

    # Configure boto3 for LocalStack S3 if endpoint is provided
    # MLflow uses boto3 internally, so we need to configure it properly
    s3_endpoint = os.environ.get("MLFLOW_S3_ENDPOINT_URL", "Not set")
    if s3_endpoint != "Not set" and s3_endpoint.startswith("http"):
        # Set boto3 S3 endpoint environment variable
        os.environ["AWS_ENDPOINT_URL_S3"] = s3_endpoint
        # Also set general endpoint URL for boto3
        os.environ["AWS_ENDPOINT_URL"] = s3_endpoint

    # Log environment configuration for debugging
    logger.info(f"MLflow S3 Endpoint: {s3_endpoint}")
    logger.info(
        f"MLflow S3 Ignore TLS: {os.environ.get('MLFLOW_S3_IGNORE_TLS', 'Not set')}"
    )
    logger.info(
        f"AWS Access Key ID: {'Set' if os.environ.get('AWS_ACCESS_KEY_ID') else 'Not set'}"
    )

    # Test LocalStack connectivity if endpoint is set
    if s3_endpoint != "Not set" and s3_endpoint.startswith("http"):
        try:
            import urllib.request

            # Convert region-specific endpoint to health check URL
            # Replace s3.{region}.host.docker.internal with host.docker.internal
            test_url = s3_endpoint
            if "s3." in test_url and "host.docker.internal" in test_url:
                # Extract port if present
                port = ":4566" if ":4566" in test_url else ""
                test_url = f"http://host.docker.internal{port}/_localstack/health"
            elif "s3." in test_url and "localhost" in test_url:
                port = ":4566" if ":4566" in test_url else ""
                test_url = f"http://localhost{port}/_localstack/health"
            else:
                test_url = s3_endpoint.rstrip("/") + "/_localstack/health"

            logger.info(f"Testing LocalStack connectivity: {test_url}")
            req = urllib.request.Request(test_url)
            req.add_header("User-Agent", "MLflow-Training")
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    logger.info("LocalStack is reachable")
                else:
                    logger.warning(f"LocalStack returned status {response.status}")
        except Exception as conn_error:
            logger.warning(
                f"Could not reach LocalStack at {s3_endpoint}: {conn_error}. "
                "Artifact uploads may fail or timeout."
            )

    # Configure MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    logger.info(f"MLflow Tracking URI: {args.mlflow_tracking_uri}")
    logger.info(f"MLflow Experiment: {args.mlflow_experiment}")
    logger.info(f"Training data: {args.train}")

    # Ensure no active run before starting (MLflow 3.0 may auto-start runs)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Load HELOC dataset
    data_file = Path(args.train) / "heloc_dataset_v1.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    df = pd.read_csv(data_file)
    logger.info(f"Loaded dataset: {df.shape}")

    # HELOC dataset: RiskPerformance is the label
    label = "RiskPerformance"
    df[label] = df[label].map({"Good": 0, "Bad": 1})
    features = [col for col in df.columns if col != label]

    # Split data
    X = df[features]
    y = df[label]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle missing values (-8 is missing value indicator in HELOC dataset)
    X_train = X_train.replace(-8, 0).fillna(0)
    X_test = X_test.replace(-8, 0).fillna(0)

    # Train with MLflow tracking
    with mlflow.start_run():
        mlflow.set_tag(
            "mlflow.note.content", "Basic RF model for HELOC risk prediction"
        )
        mlflow.set_tag("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("model_type", "RandomForest")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)

        # Evaluate
        test_pred = model.predict_proba(X_test)[:, 1]
        test_gini = roc_auc_score(y_test, test_pred) * 2 - 1
        test_log_loss = log_loss(y_test, test_pred)

        mlflow.log_metric("test_gini", test_gini)
        mlflow.log_metric("test_log_loss", test_log_loss)

        # Log model using mlflow.sklearn.log_model() (MLflow 3.0+ with logged-models API)
        # This uses the logged-models API which provides better model tracking
        try:
            logger.info("Logging model using mlflow.sklearn.log_model()...")
            mlflow.sklearn.log_model(model, "model")
            logger.info("✓ Model logged successfully to MLflow using logged-models API")
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            logger.error(f"Error logging model: {error_type}: {error_msg}")
            import traceback

            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            logger.warning(
                "⚠ Failed to log model to MLflow. "
                "Model metrics and parameters are still logged, and model is saved to SageMaker output directory."
            )
    logger.info(f"Test Gini: {test_gini:.4f}")
    logger.info("✓ Training complete!")

    # Save model for SageMaker in tar.gz format (SageMaker native format)
    import tarfile

    # Save model as joblib first
    model_joblib_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_joblib_path)
    logger.info(f"Model saved to: {model_joblib_path}")

    # Package as tar.gz for SageMaker deployment
    model_tar_path = os.path.join(args.model_dir, "model.tar.gz")
    with tarfile.open(model_tar_path, "w:gz") as tar:
        tar.add(model_joblib_path, arcname="model.joblib")
    logger.info(f"Model packaged as tar.gz for SageMaker: {model_tar_path}")


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
