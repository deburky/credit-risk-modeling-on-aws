"""Setup S3 bucket in LocalStack for MLflow artifacts."""

import contextlib
import os
import sys
import time

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError

# Configuration
LOCALSTACK_ENDPOINT = os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "test")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "test")
MLFLOW_BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# LocalStack S3 requires this specific endpoint format
S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT", f"http://s3.{REGION}.localhost.localstack.cloud:4566"
)


def wait_for_localstack(max_retries: int = 30, delay: int = 2) -> None:
    """Wait for LocalStack to be ready."""
    print(f"Waiting for LocalStack to be ready at {LOCALSTACK_ENDPOINT}...")
    for i in range(max_retries):
        with contextlib.suppress(Exception):
            response = requests.get(
                f"{LOCALSTACK_ENDPOINT}/_localstack/health", timeout=2
            )
            if response.status_code == 200:
                print("LocalStack is ready")
                return
        if i < max_retries - 1:
            print(f"Waiting for LocalStack... ({i + 1}/{max_retries})")
            time.sleep(delay)
    raise RuntimeError(f"LocalStack not ready after {max_retries * delay} seconds")


def setup_s3_bucket() -> None:
    """Create S3 bucket in LocalStack if it doesn't exist."""
    print("Setting up LocalStack S3 for MLflow...")
    print(f"Endpoint: {LOCALSTACK_ENDPOINT}")
    print(f"Bucket: {MLFLOW_BUCKET}")
    print(f"Region: {REGION}")

    wait_for_localstack()

    # For bucket creation, use regular endpoint (region-specific format causes issues)
    # MLflow will use the region-specific endpoint format for operations
    boto_config = Config(
        region_name=REGION,
        s3={"endpoint_url": LOCALSTACK_ENDPOINT, "addressing_style": "path"},
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=LOCALSTACK_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=REGION,
        config=boto_config,
    )

    try:
        s3.head_bucket(Bucket=MLFLOW_BUCKET)
        print(f"Bucket '{MLFLOW_BUCKET}' already exists")
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code == "404":
            print(f"Creating bucket: {MLFLOW_BUCKET}")
            # When using region-specific endpoint format, always specify location constraint
            # For us-east-1, use empty string; for other regions, use region name
            if REGION == "us-east-1":
                s3.create_bucket(
                    Bucket=MLFLOW_BUCKET,
                    CreateBucketConfiguration={"LocationConstraint": ""},
                )
            else:
                s3.create_bucket(
                    Bucket=MLFLOW_BUCKET,
                    CreateBucketConfiguration={"LocationConstraint": REGION},
                )
            print(f"Created bucket: {MLFLOW_BUCKET}")
        else:
            print(f"Error checking bucket: {e}")
            sys.exit(1)

    print("LocalStack S3 setup complete!")
    print("To use with MLflow, set these environment variables:")
    print(f"export AWS_ACCESS_KEY_ID={AWS_ACCESS_KEY_ID}")
    print(f"export AWS_SECRET_ACCESS_KEY={AWS_SECRET_ACCESS_KEY}")
    print(f"export MLFLOW_S3_ENDPOINT_URL={S3_ENDPOINT}")
    print("export MLFLOW_S3_IGNORE_TLS=true")


if __name__ == "__main__":
    setup_s3_bucket()
