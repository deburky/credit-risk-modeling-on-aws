"""SageMaker Pipeline for Batch Credit Scoring Workflow.

This pipeline handles the complete workflow:
1. Train CatBoost model
2. Query eligible customers from PostgreSQL
3. Run inference using trained CatBoost model
4. Invoke Lambda function to make limit increase/decrease decisions
5. Update PostgreSQL with decisions
"""

import json
import os
from pathlib import Path

import boto3
from sagemaker.estimator import Estimator
from sagemaker.local import LocalSession
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

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

# Configure Lambda client to use LocalStack
lambda_client = boto3_session.client(
    "lambda",
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
    """Create a ScriptProcessor for processing steps."""
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
    """Step 0: Train CatBoost model."""
    data_dir = str(Path(__file__).parent.parent.parent / "data")
    training_dir = str(Path(__file__).parent)

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
        source_dir=training_dir,
        hyperparameters={
            "iterations": 100,
            "learning-rate": 0.1,
            "depth": 6,
            "target-score": 600,
            "target-odds": 30,
            "pts-double-odds": 20,
        },
        entry_point="train.py",
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
    """Step 1: Query eligible customers from PostgreSQL."""
    script_path = str(Path(__file__).parent.parent / "scripts" / "database.py")

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
            "query",
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
            "--limit",
            "20",
            "--output-path",
            "/opt/ml/processing/output/customers.json",
        ],
    )


def create_inference_step(processor, query_step, training_step):
    """Step 2: Run inference using CatBoost model."""
    script_path = str(
        Path(__file__).parent.parent / "scripts" / "inference_processing.py"
    )

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


def setup_lambda_function():
    """Create or update Lambda function in LocalStack."""
    function_name = "LimitManagerLambda"
    lambda_dir = Path(__file__).parent.parent / "lambda_functions"
    lambda_code_path = lambda_dir / "limit_manager.py"

    # Read Lambda function code
    with open(lambda_code_path, "r") as f:
        lambda_code = f.read()

    # Create deployment package (zip the code)
    import tempfile
    import zipfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        with zipfile.ZipFile(tmp_file.name, "w") as zip_file:
            zip_file.writestr("lambda_function.py", lambda_code)
        zip_content = open(tmp_file.name, "rb").read()
        os.unlink(tmp_file.name)

    try:
        # Try to get existing function
        lambda_client.get_function(FunctionName=function_name)
        # Update function code
        lambda_client.update_function_code(
            FunctionName=function_name,
            ZipFile=zip_content,
        )
        # Wait for code update to complete before updating configuration
        import time

        max_wait = 10
        for _ in range(max_wait):
            try:
                response = lambda_client.get_function(FunctionName=function_name)
                state = response["Configuration"].get("State", "Active")
                last_update_status = response["Configuration"].get(
                    "LastUpdateStatus", "Successful"
                )
                if (
                    state == "Active"
                    and last_update_status
                    in [
                        "Successful",
                        "InProgress",
                    ]
                    and last_update_status == "Successful"
                ):
                    break
                time.sleep(0.5)
            except Exception:
                time.sleep(0.5)

        # Update environment variables
        lambda_client.update_function_configuration(
            FunctionName=function_name,
            Environment={
                "Variables": {
                    "LOCALSTACK_ENDPOINT": "http://localhost:4566",
                    "S3_ENDPOINT": f"http://s3.{REGION}.localhost.localstack.cloud:4566",
                    "S3_BUCKET": S3_BUCKET,
                    "AWS_DEFAULT_REGION": REGION,
                }
            },
        )
        print(f"Updated Lambda function: {function_name}")
    except lambda_client.exceptions.ResourceNotFoundException:
        # Create new function
        lambda_client.create_function(
            FunctionName=function_name,
            Runtime="python3.11",
            Role=DUMMY_IAM_ROLE,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_content},
            Timeout=300,
            Environment={
                "Variables": {
                    "LOCALSTACK_ENDPOINT": "http://localhost:4566",
                    "S3_ENDPOINT": f"http://s3.{REGION}.localhost.localstack.cloud:4566",
                    "S3_BUCKET": S3_BUCKET,
                    "AWS_DEFAULT_REGION": REGION,
                }
            },
        )
        print(f"Created Lambda function: {function_name}")

    return function_name


def invoke_lambda_function(s3_uri: str) -> str:
    """Invoke Lambda function to make decisions.

    Args:
        s3_uri: S3 URI of scored customers

    Returns:
        S3 URI of decisions output
    """
    import time

    function_name = setup_lambda_function()

    # Wait for Lambda function to be ready (not in Pending state)
    print("Waiting for Lambda function to be ready...")
    max_wait = 30
    wait_time = 0
    while wait_time < max_wait:
        try:
            response = lambda_client.get_function(FunctionName=function_name)
            state = response["Configuration"].get("State", "Unknown")
            if state == "Active":
                break
            if state == "Failed":
                raise RuntimeError("Lambda function creation failed")
            time.sleep(1)
            wait_time += 1
        except Exception as e:
            if wait_time >= max_wait:
                raise RuntimeError(
                    f"Lambda function not ready after {max_wait}s: {e}"
                ) from e
            time.sleep(1)
            wait_time += 1

    if wait_time >= max_wait:
        raise RuntimeError(f"Lambda function not ready after {max_wait} seconds")

    # Invoke Lambda function
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps({"s3_uri": s3_uri}),
    )

    # Parse response
    result = json.loads(response["Payload"].read().decode("utf-8"))
    if "errorMessage" in result:
        raise RuntimeError(f"Lambda invocation failed: {result['errorMessage']}")

    return result.get("s3_uri", f"s3://{S3_BUCKET}/pipeline/decisions/decisions.json")


def update_database_from_s3(decisions_s3_uri: str) -> None:
    """Update PostgreSQL database with decisions from S3.

    Args:
        decisions_s3_uri: S3 URI of decisions JSON file
    """
    import subprocess
    import sys

    # Parse S3 URI
    decisions_s3_uri = decisions_s3_uri.removeprefix("s3://")
    parts = decisions_s3_uri.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Download decisions file from S3
    decisions_file = Path(__file__).parent.parent / "temp_decisions.json"
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        with open(decisions_file, "w") as f:
            f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to download decisions from S3: {e}") from e

    # Run update using database script
    script_path = Path(__file__).parent.parent / "scripts" / "database.py"
    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "update",
                "--decisions-file",
                str(decisions_file),
                "--db-host",
                "localhost",
                "--db-port",
                "5432",
                "--db-name",
                "credit_scoring",
                "--db-user",
                "creditrisk",
                "--db-password",
                "creditrisk123",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Update script failed with return code {e.returncode}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        raise
    finally:
        # Clean up temp file
        if decisions_file.exists():
            decisions_file.unlink()


def create_pipeline():
    """Create the complete SageMaker Pipeline.

    Pipeline flow:
    1. Train CatBoost model
    2. Query eligible customers from PostgreSQL
    3. Run inference using trained CatBoost model

    After pipeline completes:
    4. Invoke LocalStack Lambda to make decisions
    5. Update PostgreSQL with decisions
    """
    # Create processor (shared across steps)
    processor = create_processor()

    # Create steps
    training_step = create_training_step()
    query_step = create_query_step(processor)
    inference_step = create_inference_step(processor, query_step, training_step)

    return Pipeline(
        name="BatchCreditScoringPipeline",
        steps=[training_step, query_step, inference_step],
        sagemaker_session=sagemaker_session,
    )


def setup_s3_bucket():
    """Set up S3 bucket in LocalStack if it doesn't exist."""
    try:
        boto_config.head_bucket(Bucket=S3_BUCKET)
        print(f"S3 bucket '{S3_BUCKET}' already exists")
    except Exception:
        print(f"Creating S3 bucket '{S3_BUCKET}'...")
        try:
            boto_config.create_bucket(Bucket=S3_BUCKET)
            print(f"Created S3 bucket '{S3_BUCKET}'")
        except Exception as e:
            print(f"Warning: Could not create S3 bucket: {e}")
            print("Continuing anyway (bucket may already exist)")


def execute_pipeline_workflow():
    """Execute the complete pipeline workflow including post-processing.

    Returns:
        dict: Execution result with status and details
    """
    print("SageMaker Pipeline: Batch Credit Scoring Workflow")

    # Setup S3 bucket
    print("Setting up S3 bucket...")
    setup_s3_bucket()

    # Create pipeline
    print("Creating pipeline...")
    pipeline = create_pipeline()

    # Upsert pipeline (create or update)
    print("Registering pipeline...")
    pipeline.upsert(role_arn=DUMMY_IAM_ROLE)

    print(f"Pipeline created: {pipeline.name}")

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
    execution_status = None
    execution_description = None

    try:
        execution_description = execution.describe()
        execution_status = execution_description.get("PipelineExecutionStatus")
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

    print("Pipeline execution completed")
    print(f"Status: {execution_status}")

    if execution_status == "Succeeded":
        # Get S3 URI of scored customers (construct from known path)
        try:
            scored_customers_uri = (
                f"s3://{S3_BUCKET}/pipeline/scored_customers/scored_customers.json"
            )
            print(f"Scored customers available at: {scored_customers_uri}")

            # Invoke Lambda function to make decisions
            print("\nInvoking Lambda function to make decisions...")
            decisions_uri = invoke_lambda_function(scored_customers_uri)
            print(f"Decisions saved to: {decisions_uri}")

            # Update database with decisions
            print("\nUpdating PostgreSQL with decisions...")
            update_database_from_s3(decisions_uri)
            print("Database updated successfully")

            print("Workflow completed successfully")
            print("Run 'make check-results' to see the results in PostgreSQL")
        except Exception as e:
            print(f"Error invoking Lambda function: {e}")
            print("Pipeline completed but Lambda invocation failed")
    else:
        print(f"Pipeline execution failed with status: {execution_status}")

        # Try to get detailed error information
        try:
            if execution_description:
                if step_executions := execution_description.get("StepExecutions", []):
                    print("\nStep execution details:")
                    for step in step_executions:
                        step_name = step.get("StepName", "Unknown")
                        step_status = step.get("StepStatus", "Unknown")
                        print(f"{step_name}: {step_status}")

                        if step_status == "Failed":
                            failure_reason = step.get(
                                "FailureReason", "No reason provided"
                            )
                            print(f"Failure reason: {failure_reason}")
        except Exception as e:
            print(f"\nCould not retrieve detailed error information: {e}")
            print("Check Docker logs or LocalStack logs for more details")

    return {
        "status": execution_status,
        "execution_id": execution_id,
        "pipeline_name": pipeline.name,
    }


def main():
    """Create and execute pipeline."""
    execute_pipeline_workflow()


if __name__ == "__main__":
    main()
