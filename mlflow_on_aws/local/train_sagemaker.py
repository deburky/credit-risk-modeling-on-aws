"""
Train model using SageMaker Local Mode with MLflow tracking
Uses SageMaker containerized training with MLflow integration
"""

import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from scripts.docker_utils import DockerManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
REGION = "us-east-1"
STACK_NAME = "mlflow-stack"
LOCALSTACK_ENDPOINT = "http://localhost:4566"
MLFLOW_TRACKING_URI = (
    "http://host.docker.internal:5001"  # Access host MLflow from container
)
# LocalStack S3 configuration for MLflow artifacts
# From SageMaker container, use host.docker.internal to reach host LocalStack
# Note: Region-specific format doesn't work from containers, use regular endpoint
LOCALSTACK_S3_ENDPOINT = (
    "http://host.docker.internal:4566"  # Access host LocalStack from container
)
AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
MLFLOW_BUCKET = "mlflow"


def get_stack_role_arn() -> str:
    """Get MLflow Tracking Server role ARN from CloudFormation stack

    Raises:
        RuntimeError: If stack is not deployed or role ARN cannot be found
    """
    # Check if LocalStack is running
    endpoint_url = None
    try:
        import urllib.request

        urllib.request.urlopen(f"{LOCALSTACK_ENDPOINT}/_localstack/health", timeout=2)
        endpoint_url = LOCALSTACK_ENDPOINT
        logger.info(f"LocalStack detected, using endpoint: {endpoint_url}")
    except Exception:
        # LocalStack not running, try real AWS
        logger.info("LocalStack not running, checking real AWS...")

    try:
        cfn = boto3.client(
            "cloudformation",
            endpoint_url=endpoint_url,
            region_name=REGION,
        )
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stack = response["Stacks"][0]
        stack_status = stack["StackStatus"]

        # Check if stack is in a valid state
        if "FAILED" in stack_status or "ROLLBACK" in stack_status:
            raise RuntimeError(
                f"Stack '{STACK_NAME}' is in failed state: {stack_status}. "
                "Please check stack events and fix the issue."
            )

        outputs = stack.get("Outputs", [])
        for output in outputs:
            if output["OutputKey"] == "MLflowTrackingServerRoleArn":
                role_arn = output["OutputValue"]
                logger.info(f"Found role ARN from stack '{STACK_NAME}': {role_arn}")
                return role_arn

        raise RuntimeError(
            f"Stack '{STACK_NAME}' exists but 'MLflowTrackingServerRoleArn' output not found. "
            "The stack may not have completed successfully."
        )
    except ClientError as e:
        if "does not exist" in str(e):
            raise RuntimeError(
                f"Stack '{STACK_NAME}' is not deployed. "
                f"Please deploy the stack first: make deploy-stack"
            ) from e
        else:
            raise RuntimeError(f"Could not query stack '{STACK_NAME}': {e}") from e
    except RuntimeError:
        # Re-raise RuntimeError as-is
        raise
    except Exception as e:
        raise RuntimeError(f"Error querying stack '{STACK_NAME}': {e}") from e


def build_docker_image() -> str:
    """Build Docker image for SageMaker training with MLflow.

    Returns:
        Image name with tag
    """
    dockerfile_path = Path(__file__).parent / "training" / "Dockerfile.train"

    if not dockerfile_path.exists():
        raise FileNotFoundError(
            f"Dockerfile not found: {dockerfile_path}. "
            "Please ensure Dockerfile.train exists in the training directory."
        )

    image_name = "mlflow-sagemaker-train:latest"
    build_context = Path(__file__).parent

    docker_mgr = DockerManager()
    return docker_mgr.build_image(
        image_name=image_name,
        dockerfile_path=dockerfile_path,
        build_context=build_context,
    )


def train_with_sagemaker():
    """Train using SageMaker Local Mode with MLflow"""
    logger.info("Training with SageMaker Local Mode + MLflow")

    # Build Docker image
    image_name = build_docker_image()

    # Setup SageMaker Local Session
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # Find data (data is at root level, go up from local/)
    data_file = Path(__file__).parent.parent / "data" / "heloc_dataset_v1.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset not found: {data_file}")

    train_data_local = f"file://{data_file.parent}"
    output_path_local = f"file://{Path(__file__).parent / 'model_output'}"

    # Get IAM role from stack (or fallback to dummy)
    try:
        iam_role = get_stack_role_arn()
    except RuntimeError as e:
        # If stack not deployed, use dummy role for LocalStack
        logger.warning(f"Could not get role from stack: {e}")
        logger.info("Using dummy role for LocalStack testing")
        iam_role = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

    # Create estimator with LocalStack S3 environment variables for MLflow
    training_dir = Path(__file__).parent / "training"
    estimator = Estimator(
        image_uri=image_name,
        role=iam_role,
        instance_type="local",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        output_path=output_path_local,
        source_dir=str(training_dir),
        hyperparameters={
            "mlflow-tracking-uri": MLFLOW_TRACKING_URI,
            "mlflow-experiment": "credit-scoring-sagemaker",
            "n-estimators": 100,
            "max-depth": 10,
        },
        entry_point="train.py",
        environment={
            # LocalStack S3 environment variables for MLflow artifact storage
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "MLFLOW_S3_ENDPOINT_URL": LOCALSTACK_S3_ENDPOINT,
            "MLFLOW_S3_IGNORE_TLS": "true",
        },
    )

    logger.info(f"Training data: {train_data_local}")
    logger.info(f"Output path: {output_path_local}")
    logger.info(f"MLflow URI: {MLFLOW_TRACKING_URI}")

    # Train
    estimator.fit({"train": train_data_local})

    logger.info("Training complete!")
    logger.info("MLflow UI: http://localhost:5001")


if __name__ == "__main__":
    try:
        train_with_sagemaker()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
