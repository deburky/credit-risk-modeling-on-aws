"""
Deploy CloudFormation stack to LocalStack for SageMaker Endpoint A/B Testing
"""

import argparse
import contextlib
import logging
import subprocess
import sys
from pathlib import Path

import boto3
from botocore.config import Config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# LocalStack configuration
ENDPOINT_URL = "http://localhost:4566"
REGION = "us-east-1"
STACK_NAME = "sagemaker-endpoint-stack"

# Configure boto3 for LocalStack
boto_config = Config(
    region_name=REGION,
    s3={"endpoint_url": ENDPOINT_URL, "addressing_style": "path"},
)

# AWS clients
cfn = boto3.client(
    "cloudformation", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)
sqs = boto3.client(
    "sqs", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)
dynamodb = boto3.client(
    "dynamodb", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)
lambda_client = boto3.client(
    "lambda", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)


def read_template():
    """Read CloudFormation template."""
    template_path = (
        Path(__file__).parent.parent / "cloudformation" / "infrastructure.yml"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    with open(template_path, "r") as f:
        return f.read()


def stack_exists():
    """Check if stack exists."""
    try:
        cfn.describe_stacks(StackName=STACK_NAME)
        return True
    except cfn.exceptions.ClientError as e:
        if "does not exist" in str(e):
            return False
        raise


def create_stack(template_body):
    """Create CloudFormation stack."""
    logger.info(f"Creating stack: {STACK_NAME}")
    try:
        cfn.create_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        logger.info("Stack creation initiated")
        return True
    except Exception as e:
        logger.error(f"Error creating stack: {e}")
        return False


def update_stack(template_body):
    """Update CloudFormation stack."""
    logger.info(f"Updating stack: {STACK_NAME}")
    try:
        cfn.update_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        logger.info("Stack update initiated")
        return True
    except cfn.exceptions.ClientError as e:
        if "No updates are to be performed" in str(e):
            logger.info("No updates needed")
            return False
        raise


def wait_for_stack(stack_operation):
    """Wait for stack operation to complete."""
    logger.info(f"Waiting for stack {stack_operation} to complete...")
    waiter = cfn.get_waiter(f"stack_{stack_operation}_complete")
    try:
        waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 2, "MaxAttempts": 60})
        logger.info(f"Stack {stack_operation} completed successfully")
        return True
    except Exception as e:
        logger.error(f"Stack {stack_operation} failed: {e}")
        # Show stack events for debugging
        with contextlib.suppress(Exception):
            events = cfn.describe_stack_events(StackName=STACK_NAME)["StackEvents"][:5]
            logger.error("Recent stack events:")
            for event in events:
                logger.error(
                    f"  {event['Timestamp']} {event['ResourceStatus']} {event['ResourceType']} - {event.get('ResourceStatusReason', '')}"
                )
        return False


def show_stack_outputs():
    """Show stack outputs."""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stack = response["Stacks"][0]
        if outputs := stack.get("Outputs", []):
            logger.info("\nStack Outputs:")
            for output in outputs:
                logger.info(f"  {output['OutputKey']}: {output['OutputValue']}")
    except Exception as e:
        logger.warning(f"Could not retrieve stack outputs: {e}")


def delete_stack():
    """Delete CloudFormation stack."""
    logger.info(f"Deleting stack: {STACK_NAME}")
    try:
        cfn.delete_stack(StackName=STACK_NAME)
        logger.info("Stack deletion initiated")
        logger.info("Waiting for stack deletion to complete...")
        waiter = cfn.get_waiter("stack_delete_complete")
        waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 2, "MaxAttempts": 60})
        logger.info("Stack deleted successfully")
        return True
    except Exception as e:
        logger.error(f"Error deleting stack: {e}")
        return False


def describe_stack():
    """Describe stack status."""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stack = response["Stacks"][0]
        logger.info(f"\nStack: {stack['StackName']}")
        logger.info(f"Status: {stack['StackStatus']}")
        logger.info(f"Created: {stack.get('CreationTime', 'N/A')}")

        if outputs := stack.get("Outputs", []):
            logger.info("\nOutputs:")
            for output in outputs:
                logger.info(f"  {output['OutputKey']}: {output['OutputValue']}")

        return True
    except cfn.exceptions.ClientError as e:
        if "does not exist" in str(e):
            logger.info(f"Stack '{STACK_NAME}' does not exist")
        else:
            logger.error(f"Error describing stack: {e}")
        return False


def package_lambda():
    """Package Lambda function."""
    logger.info("Packaging Lambda function...")
    package_script = (
        Path(__file__).parent.parent / "lambda_functions" / "package_lambda.py"
    )
    if not package_script.exists():
        logger.warning(f"Lambda packaging script not found: {package_script}")
        return None

    try:
        subprocess.run([sys.executable, str(package_script)], check=True)
        package_path = (
            Path(__file__).parent.parent / "lambda_packages" / "ab_processor.zip"
        )
        if package_path.exists():
            logger.info(f"✓ Lambda packaged: {package_path}")
            return package_path
        else:
            logger.error(f"Lambda package not found: {package_path}")
            return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error packaging Lambda: {e}")
        return None


def update_lambda_code(package_path):
    """Update Lambda function code."""
    if not package_path or not package_path.exists():
        logger.warning("Lambda package not available, skipping code update")
        return False

    logger.info("Updating Lambda function code...")
    try:
        with open(package_path, "rb") as f:
            lambda_client.update_function_code(
                FunctionName="ab-processor", ZipFile=f.read()
            )
        logger.info("✓ Lambda function code updated")
        return True
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.warning("Lambda function not found (may not be created yet)")
        return False
    except Exception as e:
        logger.error(f"Error updating Lambda code: {e}")
        return False


def deploy():
    """Deploy or update stack."""
    # Package Lambda first
    package_path = package_lambda()

    template_body = read_template()

    if stack_exists():
        logger.info("Stack exists, updating...")
        if update_stack(template_body):
            wait_for_stack("update")
    else:
        logger.info("Stack does not exist, creating...")
        if create_stack(template_body):
            wait_for_stack("create")

    show_stack_outputs()

    # Update Lambda code after stack is ready
    if package_path:
        logger.info("")
        update_lambda_code(package_path)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Manage CloudFormation stack")
    parser.add_argument(
        "action",
        choices=["deploy", "delete", "describe", "status"],
        help="Action to perform",
    )

    args = parser.parse_args()

    if args.action == "deploy":
        deploy()
    elif args.action == "delete":
        delete_stack()
    elif args.action in ["describe", "status"]:
        describe_stack()
    else:
        logger.error(f"Unknown action: {args.action}")
        sys.exit(1)


if __name__ == "__main__":
    main()
