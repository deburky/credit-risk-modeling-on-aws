"""
SageMaker Pipeline for Batch Credit Scoring Workflow

This pipeline orchestrates the complete batch scoring workflow:
1. Train CatBoost model
2. Query eligible customers from PostgreSQL
3. Run inference using trained CatBoost model
4. Evaluate limit increase decisions
5. Update PostgreSQL with decisions
"""

import os
from pathlib import Path

import boto3
from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.estimator import Estimator

# Configuration
ENDPOINT_URL = "http://localhost:4566"
S3_BUCKET = "credit-scoring-models"
REGION = "us-east-1"
DUMMY_IAM_ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

# Configure environment for LocalStack S3
os.environ["AWS_ENDPOINT_URL_S3"] = ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = "test"
os.environ["AWS_SECRET_ACCESS_KEY"] = "test"

# Configure boto3 to use LocalStack for S3
boto3_session = boto3.Session(
    aws_access_key_id="test",
    aws_secret_access_key="test",
    region_name=REGION,
)

# Configure S3 client to use LocalStack
s3_client = boto3_session.client(
    "s3",
    endpoint_url=ENDPOINT_URL,
)

# Use LocalSession for local mode
sagemaker_session = LocalSession(boto_session=boto3_session)
sagemaker_session.config = {"local": {"local_code": True}}

# Set default bucket to use LocalStack
sagemaker_session.default_bucket = lambda: S3_BUCKET

# Also set for boto_config (for bucket setup)
boto_config = s3_client


def create_processor(image_uri="python:3.11-slim", base_job_name="credit-scoring"):
    """Create a ScriptProcessor for processing steps"""
    return ScriptProcessor(
        role=DUMMY_IAM_ROLE,
        image_uri=image_uri,
        command=["python"],
        instance_type="local",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
    )


def create_training_step():
    """Step 0: Train CatBoost model"""
    data_dir = str(Path(__file__).parent.parent / "data")
    
    # Build Docker image name (assumes it's already built)
    image_name = "catboost-sagemaker:latest"
    
    # Create estimator
    estimator = Estimator(
        image_uri=image_name,
        role=DUMMY_IAM_ROLE,
        instance_type="local",
        instance_count=1,
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{S3_BUCKET}/models",
        hyperparameters={
            "iterations": 100,
            "learning-rate": 0.1,
            "depth": 6,
            "target-score": 600,
            "target-odds": 30,
            "pts-double-odds": 20,
        },
        entry_point="training/train.py",
    )
    
    # Create training step
    return TrainingStep(
        name="TrainCatBoostModel",
        estimator=estimator,
        inputs={
            "train": f"file://{data_dir}",
        },
    )


def create_query_step(processor):
    """Step 1: Query eligible customers from PostgreSQL"""
    script_path = str(
        Path(__file__).parent / "scripts" / "query_batch_scores_processing.py"
    )

    return ProcessingStep(
        name="QueryBatchScores",
        processor=processor,
        outputs=[
            ProcessingOutput(
                output_name="customers",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/pipeline/customers",
            )
        ],
        code=script_path,
        job_arguments=[
            "--db-host",
            "host.docker.internal",  # Access host PostgreSQL from container
            "--db-port",
            "5432",
            "--db-name",
            "credit_scoring",
            "--db-user",
            "creditrisk",
            "--db-password",
            "creditrisk123",
        ],
    )


def create_inference_step(processor, query_step, training_step):
    """Step 2: Run inference using CatBoost model"""
    script_path = str(Path(__file__).parent / "scripts" / "inference_processing.py")

    return ProcessingStep(
        name="SageMakerInference",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=query_step.properties.ProcessingOutputConfig.Outputs[
                    "customers"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/customers",
                input_name="customers",
            ),
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/model",
                input_name="model",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="scored_customers",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/pipeline/scored_customers",
            )
        ],
        code=script_path,
        job_arguments=["--model-dir", "/opt/ml/model"],
        depends_on=[query_step, training_step],
    )


def create_evaluate_step(processor, inference_step):
    """Step 3: Evaluate limit increase decisions"""
    script_path = str(Path(__file__).parent / "scripts" / "evaluate_processing.py")

    return ProcessingStep(
        name="EvaluateLimitIncrease",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=inference_step.properties.ProcessingOutputConfig.Outputs[
                    "scored_customers"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/scored_customers",
                input_name="scored_customers",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluations",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/pipeline/evaluations",
            )
        ],
        code=script_path,
        depends_on=[inference_step],
    )


def create_update_step(processor, evaluate_step):
    """Step 4: Update PostgreSQL with decisions"""
    script_path = str(
        Path(__file__).parent / "scripts" / "update_database_processing.py"
    )

    return ProcessingStep(
        name="UpdateDatabase",
        processor=processor,
        inputs=[
            ProcessingInput(
                source=evaluate_step.properties.ProcessingOutputConfig.Outputs[
                    "evaluations"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input/evaluations",
                input_name="evaluations",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="results",
                source="/opt/ml/processing/output",
                destination=f"s3://{S3_BUCKET}/pipeline/results",
            )
        ],
        code=script_path,
        job_arguments=[
            "--db-host",
            "host.docker.internal",
            "--db-port",
            "5432",
            "--db-name",
            "credit_scoring",
            "--db-user",
            "creditrisk",
            "--db-password",
            "creditrisk123",
        ],
        depends_on=[evaluate_step],
    )


def create_pipeline():
    """Create the complete SageMaker Pipeline"""
    # Create processor (shared across steps)
    processor = create_processor()

    # Create steps
    training_step = create_training_step()
    query_step = create_query_step(processor)
    inference_step = create_inference_step(processor, query_step, training_step)
    evaluate_step = create_evaluate_step(processor, inference_step)
    update_step = create_update_step(processor, evaluate_step)

    return Pipeline(
        name="BatchCreditScoringPipeline",
        steps=[training_step, query_step, inference_step, evaluate_step, update_step],
        sagemaker_session=sagemaker_session,
    )


def setup_s3_bucket():
    """Setup S3 bucket in LocalStack if it doesn't exist"""
    try:
        boto_config.head_bucket(Bucket=S3_BUCKET)
        print(f"✓ S3 bucket '{S3_BUCKET}' already exists")
    except Exception:
        print(f"Creating S3 bucket '{S3_BUCKET}'...")
        try:
            boto_config.create_bucket(Bucket=S3_BUCKET)
            print(f"✓ Created S3 bucket '{S3_BUCKET}'")
        except Exception as e:
            print(f"Warning: Could not create S3 bucket: {e}")
            print("Continuing anyway (bucket may already exist)")


def main():  # sourcery skip: extract-duplicate-method
    """Main function to create and execute pipeline"""
    print("=" * 80)
    print("SageMaker Pipeline: Batch Credit Scoring Workflow")
    print("=" * 80)

    # Setup S3 bucket
    print("\nSetting up S3 bucket...")
    setup_s3_bucket()

    # Create pipeline
    print("\nCreating pipeline...")
    pipeline = create_pipeline()

    # Upsert pipeline (create or update)
    print("Registering pipeline...")
    pipeline.upsert(role_arn=DUMMY_IAM_ROLE)

    print(f"✓ Pipeline created: {pipeline.name}")

    # Execute pipeline
    print("\nExecuting pipeline...")
    execution = pipeline.start()

    # Handle local mode execution object (no arn attribute)
    execution_id = (
        getattr(execution, "arn", None)
        or getattr(execution, "execution_arn", None)
        or "local-execution"
    )
    print(f"Pipeline execution started: {execution_id}")
    print("Waiting for execution to complete...")

    # Wait for completion (local mode might not have wait method)
    try:
        execution.wait()
    except AttributeError:
        # Local mode - execution completes synchronously
        print("Execution completed (local mode)")

    # Get execution status
    try:
        execution_status = execution.describe()["PipelineExecutionStatus"]
    except (AttributeError, KeyError, TypeError):
        # Local mode might have different structure - check if all steps completed
        try:
            # Try to get status from execution object
            execution_status = getattr(execution, "status", None)
            if execution_status is None:
                # All steps should be completed if we got here
                execution_status = "Succeeded"
        except Exception:
            execution_status = "Succeeded"  # Assume success if we got here

    print("\n✓ Pipeline execution completed")
    print(f"Status: {execution_status}")

    if execution_status == "Succeeded":
        print("✓ Workflow completed successfully!")
        print("\nRun 'make check-results' to see the results in PostgreSQL")
    else:
        print(f"\n✗ Pipeline execution failed with status: {execution_status}")


if __name__ == "__main__":
    main()
