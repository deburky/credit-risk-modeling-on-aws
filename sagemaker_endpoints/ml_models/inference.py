"""
SageMaker Model Serving Script

This script serves the ML model for SageMaker endpoint inference.
It loads the model and handles prediction requests.
"""

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import Pool
else:
    # Try to import dependencies
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from catboost import Pool

        print("Dependencies loaded successfully", file=sys.stderr)
    except ImportError as e:
        print(f"Warning: Failed to import dependencies: {e}", file=sys.stderr)
        print("Container will start but predictions will fail", file=sys.stderr)
        # Create dummy modules when import fails
        import types

        joblib = types.ModuleType("joblib")
        np = types.ModuleType("numpy")
        pd = types.ModuleType("pandas")
        Pool = types.ModuleType("Pool")

# Initialize globals
model = None
metadata = None

# Model directory (SageMaker convention)
MODEL_DIR = os.environ.get("MODEL_DIR", "/opt/ml/model")
print(f"MODEL_DIR: {MODEL_DIR}", file=sys.stderr)


def load_model():
    """Load model from S3 (LocalStack)."""
    if joblib is None:
        raise ImportError("joblib not available")

    model_dir = Path(MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "catboost_model.joblib"
    metadata_path = model_dir / "model_metadata.json"
    model_tar = model_dir / "model.tar.gz"

    # Always download from S3 (LocalStack)
    print("Downloading model from S3...", file=sys.stderr, flush=True)
    try:
        import boto3
        from botocore.config import Config

        # Get S3 configuration from environment
        # Use host.docker.internal to reach LocalStack from SageMaker local mode container
        s3_endpoint = os.environ.get(
            "AWS_ENDPOINT_URL_S3", "http://host.docker.internal:4566"
        )
        bucket_name = os.environ.get("S3_BUCKET", "credit-scoring-models")
        s3_key = os.environ.get("S3_MODEL_KEY", "models/model.tar.gz")

        # Configure boto3 for LocalStack
        s3_config = Config(
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            s3={
                "endpoint_url": s3_endpoint,
                "addressing_style": "path",
            },
        )

        s3_client = boto3.client(
            "s3",
            endpoint_url=s3_endpoint,
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", "test"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", "test"),
            config=s3_config,
        )

        # Download model.tar.gz from S3
        print(
            f"Downloading s3://{bucket_name}/{s3_key}...", file=sys.stderr, flush=True
        )
        s3_client.download_file(bucket_name, s3_key, str(model_tar))

        # Extract the downloaded tar.gz
        print(f"Extracting model from {model_tar}", file=sys.stderr, flush=True)
        import tarfile

        with tarfile.open(model_tar, "r:gz") as tar:
            tar.extractall(path=model_dir)

        print(
            "Model downloaded and extracted successfully", file=sys.stderr, flush=True
        )

    except Exception as e:
        print(f"Error downloading from S3: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        raise FileNotFoundError(f"S3 download failed: {e}") from e

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path} after S3 download")

    print(f"Loading model from {model_path}", file=sys.stderr, flush=True)
    model = joblib.load(model_path)

    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, metadata


def predict(model, metadata, features: dict) -> dict:
    """
    Make a prediction using the model.

    Args:
        model: Trained CatBoost model
        metadata: Model metadata with feature names and parameters
        features: Dictionary of customer features

    Returns:
        Dictionary with 'score' and 'proba' (probability of default)
    """
    if model is None or metadata is None:
        raise ValueError("Model or metadata not loaded")

    # Get feature names from metadata
    feature_names = metadata["feature_names"]
    categorical_features = metadata.get("categorical_features", [])

    # Ensure features are in correct order
    X_dict = {name: features.get(name, 0) for name in feature_names}
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

    # Get probability directly from model (probability of default/positive class)
    proba_array = model.predict_proba(
        pool
    )  # Returns array of shape (n_samples, n_classes)
    proba = proba_array[0, 1]

    # Get SHAP values (in log-odds space) for score calculation
    shap_values = model.get_feature_importance(type="ShapValues", data=pool)

    # SHAP values shape: (n_samples, n_features + 1)
    feature_shap = shap_values[:, :-1]  # Feature contributions (in log-odds)
    base_shap = shap_values[:, -1]  # Base value (expected log-odds)

    # Compute log-odds (matching batch_scoring pattern)
    # Extract scalar values, handling both array and scalar cases
    log_odds_sum = feature_shap.sum(axis=1)
    try:
        log_odds_sum = float(log_odds_sum[0])
    except (TypeError, IndexError):
        log_odds_sum = log_odds_sum

    try:
        log_odds_base = float(base_shap[0])
    except (TypeError, IndexError):
        log_odds_base = float(base_shap)

    log_odds = log_odds_sum + log_odds_base

    # Convert to score using PDO formula (REVERSE sign: higher score = better credit)
    factor = float(metadata["factor"])
    offset = float(metadata["offset"])
    score = offset + factor * (-log_odds)

    return {
        "proba": float(proba),
        "score": int(score),
    }


def handle_request(request_data: dict) -> dict:
    """
    Handle a prediction request.

    Expected format:
    {
        "instances": [
            {
                "Application_Score": 700,
                "Bureau_Score": 720,
                ...
            }
        ]
    }
    """
    try:
        if model is None or metadata is None:
            return {
                "error": "Model not loaded. Please ensure model files are available."
            }

        instances = request_data.get("instances", [])
        if not instances:
            return {"error": "No instances provided"}

        predictions = []
        for instance in instances:
            if isinstance(instance, list):
                # If features are provided as a list, convert to dict
                # This assumes the order matches feature_names
                if metadata is None:
                    return {"error": "Model metadata not available"}
                feature_names = metadata.get("feature_names", [])
                instance = dict(zip(feature_names, instance, strict=False))

            result = predict(model, metadata, instance)
            predictions.append(result)

        return {"predictions": predictions}

    except Exception as e:
        return {"error": str(e)}


# Load model at startup
def initialize_model():
    """Initialize the model, called at startup."""
    global model, metadata
    try:
        model, metadata = load_model()
        print(f"Model loaded successfully from {MODEL_DIR}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Warning: Error loading model: {e}", file=sys.stderr, flush=True)
        print(
            "Container will start but predictions will fail until model is available",
            file=sys.stderr,
            flush=True,
        )
        model = None
        metadata = None
        return False


# Simple HTTP server for local testing
if __name__ == "__main__":
    import signal

    def signal_handler(sig, frame):
        print("Received signal, shutting down gracefully", file=sys.stderr, flush=True)
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        from http.server import BaseHTTPRequestHandler, HTTPServer
    except ImportError as e:
        print(f"Fatal: Cannot import http.server: {e}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Initialize model before starting server
    try:
        model_loaded = initialize_model()
        if not model_loaded:
            print(
                "Starting server without model (health checks will work, predictions will fail)",
                file=sys.stderr,
                flush=True,
            )
    except Exception as e:
        print(
            f"Warning: Error during model initialization: {e}",
            file=sys.stderr,
            flush=True,
        )
        print(
            "Starting server anyway (health checks will work, predictions will fail)",
            file=sys.stderr,
            flush=True,
        )
        model_loaded = False

    class ModelHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            """Health check endpoint."""
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            status = {"status": "healthy", "model_loaded": model is not None}
            self.wfile.write(json.dumps(status).encode("utf-8"))

        def do_POST(self):
            """Handle prediction requests."""
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                if content_length == 0:
                    self.send_response(400)
                    self.send_header("Content-type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"error": "No content"}).encode("utf-8")
                    )
                    return

                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode("utf-8"))
                response = handle_request(request_data)

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except json.JSONDecodeError as e:
                self.send_response(400)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"error": f"Invalid JSON: {e}"}).encode("utf-8")
                )
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

        def log_message(self, format, *args):
            # Suppress default logging
            pass

    # Run server on port 8080 (SageMaker local mode convention)
    server = None
    try:
        server = HTTPServer(("0.0.0.0", 8080), ModelHandler)
        print(
            "Model server starting on http://0.0.0.0:8080", file=sys.stderr, flush=True
        )
        print("Server is ready to accept requests", file=sys.stderr, flush=True)
        # Keep server running - this blocks until interrupted
        server.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped by user", file=sys.stderr, flush=True)
        if server:
            server.shutdown()
        sys.exit(0)
    except SystemExit:
        # Re-raise SystemExit to allow clean shutdown
        raise
    except Exception as e:
        print(f"Fatal error in server: {e}", file=sys.stderr, flush=True)
        import traceback

        traceback.print_exc(file=sys.stderr)
        # Don't exit - keep trying to serve
        # SageMaker will handle container lifecycle
        if server:
            try:
                server.shutdown()
            except Exception:
                pass
        # Keep container alive even on error - SageMaker will handle lifecycle
        import time

        while True:
            time.sleep(60)  # Keep container alive
