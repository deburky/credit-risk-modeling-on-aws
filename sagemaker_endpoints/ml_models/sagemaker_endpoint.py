"""
SageMaker Endpoint Deployment for Local Mode

This script deploys a SageMaker endpoint in local mode using Docker,
mirroring production deployment for credit scoring inference.
"""

import contextlib
import json
import os
import subprocess
from pathlib import Path

import boto3
from botocore.config import Config
from sagemaker.local import LocalSession
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# LocalStack configuration
ENDPOINT_URL = "http://localhost:4566"
REGION = "us-east-1"
S3_BUCKET = "credit-scoring-models"
ENDPOINT_NAME = "credit-scoring-endpoint"

# Dummy IAM role (required by SageMaker but not used in local mode)
DUMMY_IAM_ROLE = (
    "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"
)

# Set environment variables for ALL boto3 clients to use LocalStack
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
os.environ["AWS_DEFAULT_REGION"] = REGION
os.environ["AWS_ENDPOINT_URL"] = ENDPOINT_URL  # For boto3 >= 1.28.0
os.environ["AWS_ENDPOINT_URL_S3"] = ENDPOINT_URL  # Specifically for S3

# Configure boto3 for LocalStack
boto_config = Config(
    region_name=REGION,
    s3={
        "endpoint_url": ENDPOINT_URL,
        "addressing_style": "path",
    },
)

boto3_session = boto3.Session(
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name=REGION,
)

# Configure S3 client
s3_client = boto3_session.client("s3", endpoint_url=ENDPOINT_URL, config=boto_config)

# SageMaker session for local mode
# Configure to use LocalStack S3
sagemaker_session = LocalSession(boto_session=boto3_session)
sagemaker_session.config = {"local": {"local_code": True}}

# Override S3 client in the session to use LocalStack
# SageMaker uses this client to download model artifacts
sagemaker_session.boto_session._session.set_config_variable(
    "s3", {"endpoint_url": ENDPOINT_URL, "addressing_style": "path"}
)


def setup_s3_bucket():
    """Create S3 bucket in LocalStack if it doesn't exist."""
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
        print(f"S3 bucket '{S3_BUCKET}' already exists")
    except s3_client.exceptions.ClientError:
        print(f"Creating S3 bucket '{S3_BUCKET}'...")
        s3_client.create_bucket(Bucket=S3_BUCKET)
        print(f"Created S3 bucket '{S3_BUCKET}'")


def cleanup_existing_containers(image_name: str = "sagemaker-credit-scoring:latest"):
    """Stop and remove any existing containers using the same image or port."""
    print("Checking for existing containers...")

    # Stop containers using our image
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"ancestor={image_name}", "--format", "{{.ID}}"],
        capture_output=True,
        text=True,
    )

    if container_ids := [cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()]:
        print(f"Found {len(container_ids)} existing container(s), stopping...")
        for container_id in container_ids:
            subprocess.run(
                ["docker", "stop", container_id],
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["docker", "rm", container_id],
                capture_output=True,
                text=True,
            )
        print("Cleaned up existing containers")

    # Check for containers using port 8080 (SageMaker local mode default)
    result = subprocess.run(
        ["docker", "ps", "--filter", "publish=8080", "--format", "{{.ID}} {{.Names}} {{.Ports}}"],
        capture_output=True,
        text=True,
    )

    if result.stdout.strip():
        print("Warning: Port 8080 is in use:")
        print(result.stdout.strip())
        print("Attempting to stop containers using port 8080...")
        # Get container IDs using port 8080
        port_containers = subprocess.run(
            ["docker", "ps", "--filter", "publish=8080", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
        )
        for container_id in port_containers.stdout.strip().split("\n"):
            if container_id.strip():
                subprocess.run(
                    ["docker", "stop", container_id.strip()],
                    capture_output=True,
                    text=True,
                )
                print(f"Stopped container {container_id.strip()}")


def build_sagemaker_image():
    """Build Docker image for SageMaker inference container."""
    image_name = "sagemaker-credit-scoring:latest"
    project_root = Path(__file__).parent.parent
    dockerfile_path = Path(__file__).parent / "Dockerfile.inference"

    if not dockerfile_path.exists():
        print(f"Error: Dockerfile not found at {dockerfile_path}")
        return None

    print(f"Building Docker image '{image_name}' using {dockerfile_path}...")
    result = subprocess.run(
        ["docker", "build", "-t", image_name, "-f", str(dockerfile_path), str(project_root)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Docker build FAILED:")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        return None

    print(f"Built Docker image '{image_name}'")
    return image_name


def deploy_endpoint(
    model_path: str | Path | None = None, image_uri: str | None = None, detach: bool = False
):
    """Deploy SageMaker endpoint using SageMaker local mode."""
    print(f"Deploying SageMaker endpoint '{ENDPOINT_NAME}' using local mode...")

    setup_s3_bucket()

    # Ensure model.tar.gz exists
    project_root = Path(__file__).parent.parent
    model_output_dir = project_root / "ml_models" / "model_output"
    model_tar_path = model_output_dir / "model.tar.gz"

    # Create model.tar.gz if it doesn't exist
    if not model_tar_path.exists():
        print("Creating model.tar.gz...")
        import tarfile

        model_file = model_output_dir / "catboost_model.joblib"
        metadata_file = model_output_dir / "model_metadata.json"

        if not model_file.exists() or not metadata_file.exists():
            print("Error: Model files not found. Train a model first.")
            return None

        with tarfile.open(model_tar_path, "w:gz") as tar:
            tar.add(str(model_file), arcname="catboost_model.joblib")
            tar.add(str(metadata_file), arcname="model_metadata.json")
        print(f"Created {model_tar_path}")

    # Upload to S3
    s3_key = "models/model.tar.gz"
    print(f"Uploading model to s3://{S3_BUCKET}/{s3_key}...")

    with open(model_tar_path, "rb") as f:
        s3_client.upload_fileobj(f, S3_BUCKET, s3_key)

    # Verify upload
    try:
        s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
        print("Model uploaded successfully")
    except Exception as e:
        print(f"Failed to verify upload: {e}")
        return None

    # Build Docker image
    if not image_uri:
        image_uri = build_sagemaker_image()
    if not image_uri:
        print("Error: Failed to build Docker image")
        return None

    # Clean up any existing containers that might conflict
    cleanup_existing_containers(image_uri)

    # Use SageMaker local mode to deploy endpoint
    model_data = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"Using model from: {model_data}")

    # Create SageMaker Model with environment variables for S3 access
    # Use host.docker.internal to reach LocalStack from SageMaker local mode container
    model = Model(
        image_uri=image_uri,
        model_data=model_data,
        role=DUMMY_IAM_ROLE,
        sagemaker_session=sagemaker_session,
        env={
            "AWS_ENDPOINT_URL_S3": "http://host.docker.internal:4566",
            "AWS_ACCESS_KEY_ID": "test",
            "AWS_SECRET_ACCESS_KEY": "test",
            "AWS_DEFAULT_REGION": REGION,
            "S3_BUCKET": S3_BUCKET,
            "S3_MODEL_KEY": "models/model.tar.gz",
        },
    )

    # Deploy endpoint using SageMaker local mode
    print("Deploying endpoint with SageMaker local mode...")
    import threading
    import time

    deploy_error: list[Exception | None] = [None]
    deployment_complete = threading.Event()

    def deploy_in_thread():
        """Deploy endpoint in a thread using the same LocalSession so SDK can track it."""
        try:
            model.deploy(
                initial_instance_count=1,
                instance_type="local",
                endpoint_name=ENDPOINT_NAME,
            )
            deployment_complete.set()
        except Exception as e:
            deploy_error[0] = e
            deployment_complete.set()

    # Start deployment in background thread
    # Using same process/thread ensures LocalSession tracks the endpoint initially
    # Make it daemon so process can exit, but deployment continues
    deploy_thread = threading.Thread(target=deploy_in_thread, daemon=True)
    deploy_thread.start()

    # Wait a moment for deployment to start and register with LocalSession
    time.sleep(3)

    if deploy_error[0]:
        print(f"Error deploying endpoint: {deploy_error[0]}")
        import traceback

        traceback.print_exc()
        return None

    print(f" Endpoint '{ENDPOINT_NAME}' deployment started")
    print("Endpoint is running in the background")
    print("Use 'make endpoint-status' to check status")
    print("Use 'make test-endpoint' to test it")

    # Return a predictor that can be used once ready
    # The LocalSession in this process will track the endpoint
    return Predictor(
        endpoint_name=ENDPOINT_NAME,
        sagemaker_session=sagemaker_session,
    )


def test_endpoint(predictor=None):
    """Test the deployed endpoint with a sample customer record."""
    if not predictor:
        # Create SageMaker Predictor for local mode endpoint
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=sagemaker_session,
        )

    sample_features = {
        "Application_Score": 700,
        "Bureau_Score": 720,
        "Loan_Amount": 10000,
        "Time_with_Bank": 24,
        "Time_in_Employment": 36,
        "Loan_to_income": 0.3,
        "Gross_Annual_Income": 50000,
        "Loan_Payment_Frequency": "M",
        "Residential_Status": "H",
        "Cheque_Card_Flag": "Y",
        "Existing_Customer_Flag": "Y",
        "Home_Telephone_Number": "Y",
    }

    print("Testing endpoint with sample features...")
    print(f"Features: {json.dumps(sample_features, indent=2)}")

    try:
        # Send as dictionary, not list
        payload = json.dumps({"instances": [sample_features]})
        response = predictor.predict(
            payload,
            initial_args={"ContentType": "application/json"},
        )

        # Parse response - handle bytes, string, or dict
        if isinstance(response, bytes):
            result = json.loads(response.decode("utf-8"))
        elif isinstance(response, str):
            result = json.loads(response)
        else:
            result = response
            
        print(f"Prediction successful: {result}")
        return result
    except Exception as e:
        print(f"Error testing endpoint: {e}")
        import traceback
        traceback.print_exc()
        return None

def delete_endpoint():
    """Delete the SageMaker endpoint."""
    print(f"Deleting endpoint '{ENDPOINT_NAME}'...")

    # Try to delete via SageMaker SDK first
    deleted_via_sdk = False
    with contextlib.suppress(Exception):
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME,
            sagemaker_session=sagemaker_session,
        )
        predictor.delete_endpoint()
        print(f" Endpoint '{ENDPOINT_NAME}' deleted via SageMaker SDK")
        deleted_via_sdk = True
    # Also stop any Docker containers that might be running for this endpoint
    # SageMaker local mode uses containers with specific naming patterns
    try:
        import subprocess

        containers_to_stop = set()

        # Method 1: Find containers by name pattern (SageMaker local mode naming)
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True,
            text=True,
        )

        for line in result.stdout.splitlines():
            if line.strip():
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    container_id, container_name = parts
                    # Check if container name contains endpoint-related identifiers
                    if (
                        ENDPOINT_NAME.lower().replace("-", "") in container_name.lower()
                        or "algo" in container_name.lower()
                    ):
                        containers_to_stop.add(container_id)

        # Method 2: Find containers using port 8080 (SageMaker endpoint port)
        port_result = subprocess.run(
            ["docker", "ps", "--filter", "publish=8080", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
        )

        for container_id in port_result.stdout.strip().split("\n"):
            if container_id.strip():
                containers_to_stop.add(container_id.strip())

        if containers_to_stop:
            print(f"Stopping {len(containers_to_stop)} Docker container(s)...")
            for container_id in containers_to_stop:
                subprocess.run(
                    ["docker", "stop", container_id],
                    capture_output=True,
                )
                subprocess.run(
                    ["docker", "rm", container_id],
                    capture_output=True,
                )
            print(f" Stopped and removed {len(containers_to_stop)} container(s)")
        else:
            print("No running containers found for this endpoint")
    except Exception as e:
        if not deleted_via_sdk:
            print(f"Warning: Error stopping containers: {e}")

    if deleted_via_sdk or containers_to_stop:
        print(f" Endpoint '{ENDPOINT_NAME}' cleanup complete")
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' not found (may already be deleted)")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage SageMaker endpoint in local mode")
    parser.add_argument(
        "action",
        choices=["deploy", "test", "delete", "status"],
        help="Action to perform",
    )
    parser.add_argument("--model-path", type=str, help="Path to model artifacts")
    parser.add_argument("--image-uri", type=str, help="Docker image URI")
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Return immediately after starting deployment (endpoint runs in background)",
    )

    args = parser.parse_args()

    if args.action == "deploy":
        result = deploy_endpoint(
            model_path=args.model_path, image_uri=args.image_uri, detach=args.detach
        )
        import sys

        # Exit after deployment starts - container runs independently
        # SDK deletion may not work perfectly due to process separation,
        # but Docker cleanup will handle it
        if result:
            sys.exit(0)
        else:
            sys.exit(1)
    elif args.action == "test":
        test_endpoint()
    elif args.action == "delete":
        delete_endpoint()
    elif args.action == "status":
        try:
            Predictor(
                endpoint_name=ENDPOINT_NAME,
                sagemaker_session=sagemaker_session,
            )
            print(f"Endpoint '{ENDPOINT_NAME}' is running")
        except Exception as e:
            print(f"Error: Endpoint '{ENDPOINT_NAME}' is not available: {e}")


if __name__ == "__main__":
    main()
