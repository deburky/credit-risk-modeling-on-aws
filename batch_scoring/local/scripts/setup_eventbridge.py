"""Setup EventBridge rule to trigger SageMaker Pipeline via Lambda."""

import os
import time
import zipfile
from pathlib import Path

import boto3

# Configuration
ENDPOINT_URL = "http://localhost:4566"
REGION = "us-east-1"
DUMMY_IAM_ROLE = "arn:aws:iam::111111111111:role/service-role/AmazonSageMaker-ExecutionRole-20200101T000001"

# AWS clients
events_client = boto3.client("events", endpoint_url=ENDPOINT_URL, region_name=REGION)
lambda_client = boto3.client("lambda", endpoint_url=ENDPOINT_URL, region_name=REGION)

RULE_NAME = "batch-scoring-schedule"
LAMBDA_FUNCTION_NAME = "PipelineTriggerLambda"
SCHEDULE_EXPRESSION = "cron(0 2 * * ? *)"  # Daily at 2 AM UTC


def setup_lambda_function():
    """Create or update Lambda function to start pipeline."""
    base_dir = Path(__file__).parent.parent
    lambda_dir = base_dir / "lambda_functions"
    training_dir = base_dir / "training"
    lambda_code_path = lambda_dir / "pipeline_trigger.py"

    with open(lambda_code_path, "r") as f:
        lambda_code = f.read()

    # Create deployment package with training module
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        with zipfile.ZipFile(tmp_file.name, "w") as zip_file:
            # Add Lambda function
            zip_file.writestr("lambda_function.py", lambda_code)
            # Add training module files (including __init__.py)
            for py_file in training_dir.glob("*.py"):
                arcname = f"training/{py_file.name}"
                zip_file.write(py_file, arcname)
        zip_content = open(tmp_file.name, "rb").read()
        os.unlink(tmp_file.name)

    try:
        lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        lambda_client.update_function_code(
            FunctionName=LAMBDA_FUNCTION_NAME, ZipFile=zip_content
        )
        print(f"Updated Lambda function: {LAMBDA_FUNCTION_NAME}")
    except lambda_client.exceptions.ResourceNotFoundException:
        lambda_client.create_function(
            FunctionName=LAMBDA_FUNCTION_NAME,
            Runtime="python3.11",
            Role=DUMMY_IAM_ROLE,
            Handler="lambda_function.lambda_handler",
            Code={"ZipFile": zip_content},
            Timeout=900,
            Environment={
                "Variables": {
                    "LOCALSTACK_ENDPOINT": ENDPOINT_URL,
                    "S3_BUCKET": "credit-scoring-models",
                    "AWS_DEFAULT_REGION": REGION,
                }
            },
        )
        print(f"Created Lambda function: {LAMBDA_FUNCTION_NAME}")

    # Wait for Lambda to be active
    max_wait = 30
    for _ in range(max_wait):
        response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        if response["Configuration"]["State"] == "Active":
            break
        time.sleep(1)

    return LAMBDA_FUNCTION_NAME


def setup_eventbridge_rule():
    """Create EventBridge rule and add Lambda as target."""
    # Create or update rule
    try:
        events_client.put_rule(
            Name=RULE_NAME,
            ScheduleExpression=SCHEDULE_EXPRESSION,
            State="ENABLED",
            Description="Trigger batch scoring pipeline daily",
        )
        print(f"Created/updated EventBridge rule: {RULE_NAME}")
    except Exception as e:
        print(f"Error creating rule: {e}")
        raise

    # Get Lambda function ARN
    lambda_response = lambda_client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
    lambda_arn = lambda_response["Configuration"]["FunctionArn"]

    # Add Lambda as target
    try:
        events_client.put_targets(
            Rule=RULE_NAME,
            Targets=[
                {
                    "Id": "1",
                    "Arn": lambda_arn,
                }
            ],
        )
        print(f"Added Lambda {LAMBDA_FUNCTION_NAME} as target for rule {RULE_NAME}")
    except Exception as e:
        print(f"Error adding target: {e}")
        raise

    # Grant EventBridge permission to invoke Lambda
    try:
        lambda_client.add_permission(
            FunctionName=LAMBDA_FUNCTION_NAME,
            StatementId=f"eventbridge-{RULE_NAME}",
            Action="lambda:InvokeFunction",
            Principal="events.amazonaws.com",
            SourceArn=f"arn:aws:events:{REGION}:000000000000:rule/{RULE_NAME}",
        )
        print("Granted EventBridge permission to invoke Lambda")
    except lambda_client.exceptions.ResourceConflictException:
        print("Permission already exists")


def main():
    """Set up EventBridge and Lambda for pipeline triggering."""
    print("Setting up EventBridge trigger for SageMaker Pipeline")
    print(f"Schedule: {SCHEDULE_EXPRESSION}")

    # Setup Lambda function
    setup_lambda_function()

    # Setup EventBridge rule
    setup_eventbridge_rule()

    print("EventBridge setup complete")
    print(f"Rule '{RULE_NAME}' will trigger '{LAMBDA_FUNCTION_NAME}' on schedule")


if __name__ == "__main__":
    main()
