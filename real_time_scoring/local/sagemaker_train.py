"""
Train credit scorecard using SageMaker Local Mode
Stores model in LocalStack S3
"""

import logging
import os
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
DUMMY_IAM_ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

# Configure boto3 to use LocalStack
boto_config = Config(
    region_name=REGION, s3={"endpoint_url": ENDPOINT_URL, "addressing_style": "path"}
)


def setup_s3_bucket():
    """Create S3 bucket in LocalStack if it doesn't exist"""
    s3 = boto3.client(
        "s3", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
    )

    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ S3 bucket '{S3_BUCKET}' already exists")
    except s3.exceptions.ClientError:
        logger.info(f"Creating S3 bucket '{S3_BUCKET}'...")
        s3.create_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ Created S3 bucket '{S3_BUCKET}'")

    return s3


def build_docker_image(image_name="credit-scoring-sagemaker:latest"):
    """Build custom Docker image for SageMaker training"""
    dockerfile_path = Path(__file__).parent / "Dockerfile"

    if not dockerfile_path.exists():
        raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

    logger.info(f"Building Docker image: {image_name}")
    logger.info(f"Dockerfile: {dockerfile_path}")

    # Check if image already exists
    result = subprocess.run(
        ["docker", "images", "-q", image_name], capture_output=True, text=True
    )

    if result.stdout.strip():
        logger.info(f"✓ Docker image '{image_name}' already exists, skipping build")
        return image_name

    # Build the image
    build_cmd = [
        "docker",
        "build",
        "-t",
        image_name,
        "-f",
        str(dockerfile_path),
        str(dockerfile_path.parent),
    ]

    logger.info(f"Running: {' '.join(build_cmd)}")
    result = subprocess.run(build_cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Docker build failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to build Docker image: {result.stderr}")

    logger.info(f"✓ Successfully built Docker image: {image_name}")
    return image_name


def upload_training_data(s3_client):
    """Upload training data to S3"""
    data_path = Path("../data/credit_example.csv")

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    s3_key = "training-data/credit_example.csv"
    logger.info(f"Uploading training data to s3://{S3_BUCKET}/{s3_key}...")

    s3_client.upload_file(str(data_path), S3_BUCKET, s3_key)
    logger.info("✓ Uploaded training data")

    return f"s3://{S3_BUCKET}/{s3_key}"


def train_with_sagemaker_local():
    """Train scorecard using SageMaker Local Mode and save to S3"""

    logger.info("Training Credit Scorecard with SageMaker Local Mode")

    # Setup S3
    s3_client = setup_s3_bucket()

    # Use LocalSession for local training
    # Note: SageMaker Local Mode works best with local file paths
    # We'll upload the model to S3 after training
    #
    # IMPORTANT: LocalSession() will use AWS credentials from ~/.aws/config to authenticate
    # with ECR to pull the Docker image. Pulling images is FREE - you only get charged for
    # compute/storage, not for pulling public images. Since instance_type="local", all training
    # runs locally on your machine, so there are NO compute charges.
    sagemaker_session = LocalSession()
    sagemaker_session.config = {"local": {"local_code": True}}

    # Use local file paths for training
    train_data_local = "file://../data"
    output_path_local = "file://./model_output"

    # Build custom Docker image if not already built
    image_name = build_docker_image("credit-scoring-sagemaker:latest")

    # Create estimator with custom Docker image
    sklearn_estimator = Estimator(
        image_uri=image_name,
        role=DUMMY_IAM_ROLE,
        instance_type="local",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        output_path=output_path_local,
        hyperparameters={
            "target-score": 600,
            "target-odds": 30,
            "pts-double-odds": 20,
        },
        entry_point="training/train.py",
    )

    logger.info("\nStarting SageMaker Local training...")
    logger.info(f"Training data: {train_data_local}")
    logger.info(f"Output path: {output_path_local}")

    # Train using local file paths
    sklearn_estimator.fit({"train": train_data_local})

    logger.info("\n" + "=" * 80)
    logger.info("✓ Training complete!")
    logger.info(f"Model artifacts saved locally to: {output_path_local}")
    logger.info("=" * 80)

    # Upload model to LocalStack S3
    logger.info("\nUploading model to LocalStack S3...")
    upload_model_to_s3(s3_client)
    logger.info(f"✓ Model uploaded to s3://{S3_BUCKET}/models/model.tar.gz")

    # Download model artifacts locally for reference
    logger.info("\nDownloading model artifacts locally for reference...")
    download_model_from_s3(s3_client)

    return sklearn_estimator


def upload_model_to_s3(s3_client):
    """Upload model artifacts from local model_output to S3"""
    import tarfile
    from io import BytesIO

    model_dir = "model_output"
    if not os.path.exists(model_dir):
        logger.warning(f"Model directory not found: {model_dir}")
        return

    # Check if we have the model files
    model_path = os.path.join(model_dir, "model.joblib")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        logger.warning(
            "Model files not found locally. SageMaker may have saved directly to S3."
        )
        return

    # Create model.tar.gz
    tar_buffer = BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        # Add model.joblib and metadata.json
        tar.add(model_path, arcname="model.joblib")
        tar.add(metadata_path, arcname="metadata.json")

    tar_buffer.seek(0)

    # Upload to S3
    s3_key = "models/model.tar.gz"
    logger.info(f"Uploading model to s3://{S3_BUCKET}/{s3_key}...")
    s3_client.upload_fileobj(tar_buffer, S3_BUCKET, s3_key)
    logger.info("✓ Model uploaded to S3")


def download_model_from_s3(s3_client):
    """Download model artifacts from S3 to local model_output directory"""
    import tarfile
    from io import BytesIO

    model_prefix = "models/"

    try:
        # List objects in the models prefix
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=model_prefix)

        if "Contents" not in response:
            logger.warning("No model artifacts found in S3")
            return

        # Find the model.tar.gz file
        model_tar_key = None
        for obj in response["Contents"]:
            if obj["Key"].endswith("model.tar.gz"):
                model_tar_key = obj["Key"]
                break

        if not model_tar_key:
            logger.warning("model.tar.gz not found in S3")
            return

        # Download model.tar.gz
        logger.info(f"Downloading {model_tar_key}...")
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=model_tar_key)
        tar_data = obj["Body"].read()

        # Extract to local model_output directory
        os.makedirs("model_output", exist_ok=True)
        with tarfile.open(fileobj=BytesIO(tar_data), mode="r:gz") as tar:
            tar.extractall("model_output")

        logger.info("✓ Model artifacts downloaded to model_output/")

    except Exception as e:
        logger.warning(f"Could not download model from S3: {e}")
        logger.info("Model is still available in S3 for Lambda to use")


if __name__ == "__main__":
    estimator = train_with_sagemaker_local()
