"""
Train CatBoost model using SageMaker Local Mode and save to S3.

This trains the model using SageMaker, which saves it to S3.
The model can then be deployed from S3.
"""

import logging
import subprocess
from pathlib import Path

import boto3
from botocore.config import Config
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LocalStack S3 configuration
ENDPOINT_URL = "http://localhost:4566"
S3_BUCKET = "credit-scoring-models"
REGION = "us-east-1"
DUMMY_IAM_ROLE = (
    "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"
)

# Configure boto3 to use LocalStack
boto_config = Config(
    region_name=REGION, s3={"endpoint_url": ENDPOINT_URL, "addressing_style": "path"}
)


def setup_s3_bucket():
    """Create S3 bucket in LocalStack if it doesn't exist"""
    s3 = boto3.client("s3", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config)

    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"S3 bucket '{S3_BUCKET}' already exists")
    except s3.exceptions.ClientError:
        logger.info(f"Creating S3 bucket '{S3_BUCKET}'...")
        s3.create_bucket(Bucket=S3_BUCKET)
        logger.info(f"Created S3 bucket '{S3_BUCKET}'")

    return s3


def build_docker_image(image_name="catboost-sagemaker:latest"):
    """Build custom Docker image for SageMaker training"""
    project_root = Path(__file__).parent.parent
    dockerfile_path = Path(__file__).parent / "Dockerfile.training"

    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    logger.info(f"Building Docker image: {image_name}")

    # Always rebuild to ensure latest changes are included
    logger.info(f"Rebuilding Docker image '{image_name}' to ensure latest changes...")

    # Build the image
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile_path),
        str(project_root),
    ]

    logger.info(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Docker build failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to build Docker image: {result.stderr}")

    logger.info(f"Successfully built Docker image: {image_name}")
    return image_name


def train_with_sagemaker():
    """Train CatBoost model using SageMaker Local Mode and save to S3"""
    logger.info("Training CatBoost Credit Scoring Model with SageMaker Local Mode")

    # Setup S3
    s3_client = setup_s3_bucket()

    # Use LocalSession for local training
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # Use local file paths for training
    project_root = Path(__file__).parent.parent
    base_dir = project_root / "ml_models"
    data_dir = project_root / "data"
    train_data_local = f"file://{data_dir}" if data_dir.exists() else None
    output_path_local = f"file://{base_dir / 'model_output'}"

    # Build custom Docker image if not already built
    image_name = build_docker_image("catboost-sagemaker:latest")

    # Determine if we need dummy data
    use_dummy_data = not (train_data_local and data_dir.exists() and any(data_dir.glob("*.csv")))

    # Set hyperparameters (SageMaker requires all values as strings)

    from sagemaker.workflow.parameters import PipelineVariable

    hyperparameters: dict[str, str | PipelineVariable] = {
        "iterations": "100",
        "learning-rate": "0.1",
        "depth": "6",
        "target-score": "600",
        "target-odds": "30",
        "pts-double-odds": "20",
    }

    if use_dummy_data:
        logger.info("No training data found, using dummy data")
        hyperparameters["dummy-data"] = "True"
        hyperparameters["dummy-samples"] = "100"

    # Create estimator with custom Docker image
    # Specify source_dir to mount ml_models directory to /opt/ml/code
    catboost_estimator = Estimator(
        image_uri=image_name,
        role=DUMMY_IAM_ROLE,
        instance_type="local",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        output_path=output_path_local,
        hyperparameters=hyperparameters,
        entry_point="train.py",  # SageMaker accepts .py files directly
        source_dir=str(base_dir),  # Mount ml_models to /opt/ml/code
    )

    logger.info("Starting SageMaker Local training...")
    logger.info(f"Training data: {train_data_local}")
    logger.info(f"Output path: {output_path_local}")

    # Train using local file paths or dummy data
    if use_dummy_data:
        catboost_estimator.fit()
    else:
        catboost_estimator.fit({"train": train_data_local})

    logger.info("Training complete!")
    logger.info(f"Model artifacts saved locally to: {output_path_local}")

    # Upload model to LocalStack S3
    logger.info("Uploading model to LocalStack S3...")
    upload_model_to_s3(s3_client)
    logger.info(f"Model uploaded to s3://{S3_BUCKET}/models/catboost_model.tar.gz")

    return catboost_estimator


def upload_model_to_s3(s3_client):
    """Upload model artifacts from local model_output to S3"""
    import tarfile
    from io import BytesIO

    project_root = Path(__file__).parent.parent
    model_dir = project_root / "ml_models" / "model_output"
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return

    # Check if we have the model files
    model_path = model_dir / "catboost_model.joblib"
    metadata_path = model_dir / "model_metadata.json"

    if not model_path.exists() or not metadata_path.exists():
        logger.warning("Model files not found locally. SageMaker may have saved directly to S3.")
        return

    # Create model.tar.gz
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add(str(model_path), arcname="catboost_model.joblib")
        tar.add(str(metadata_path), arcname="model_metadata.json")

    tar_buffer.seek(0)

    # Upload to S3
    s3_key = "models/catboost_model.tar.gz"
    logger.info(f"Uploading model to s3://{S3_BUCKET}/{s3_key}...")
    s3_client.upload_fileobj(tar_buffer, S3_BUCKET, s3_key)
    logger.info("Model uploaded to S3")


if __name__ == "__main__":
    train_with_sagemaker()
