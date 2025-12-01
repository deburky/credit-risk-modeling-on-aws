"""
Deploy CloudFormation stack to LocalStack
"""

import logging
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LocalStack endpoint
ENDPOINT_URL = "http://localhost:4566"
STACK_NAME = "limit-increase-stack"

# AWS clients
cfn = boto3.client("cloudformation", endpoint_url=ENDPOINT_URL, region_name="us-east-1")


def read_template():
    """Read CloudFormation template"""
    template_path = Path(__file__).parent / "cloudformation" / "infrastructure.yml"
    with open(template_path, "r") as f:
        return f.read()


def deploy_stack():
    """Deploy CloudFormation stack"""
    logger.info(f"Deploying stack: {STACK_NAME}")

    template_body = read_template()

    try:
        # Check if stack exists
        try:
            cfn.describe_stacks(StackName=STACK_NAME)
            logger.info("Stack exists, updating...")

            cfn.update_stack(
                StackName=STACK_NAME,
                TemplateBody=template_body,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            action = "UPDATE"
        except cfn.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise

            logger.info("Stack doesn't exist, creating...")

            cfn.create_stack(
                StackName=STACK_NAME,
                TemplateBody=template_body,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            action = "CREATE"
        # Wait for stack to complete
        logger.info(f"Waiting for stack {action} to complete...")
        waiter = cfn.get_waiter(f"stack_{action.lower()}_complete")

        try:
            waiter.wait(
                StackName=STACK_NAME, WaiterConfig={"Delay": 5, "MaxAttempts": 60}
            )
            logger.info(f"✓ Stack {action} completed successfully!")
        except Exception as e:
            logger.error(f"Stack {action} failed or timed out: {e}")
            # Show stack events for debugging
            show_stack_events()
            raise

        # Show stack outputs
        show_stack_outputs()

    except Exception as e:
        logger.error(f"Error deploying stack: {e}")
        raise


def show_stack_events():
    """Show recent stack events"""
    try:
        response = cfn.describe_stack_events(StackName=STACK_NAME)
        events = response.get("StackEvents", [])[:10]

        logger.info("\nRecent Stack Events:")
        logger.info("-" * 80)
        for event in events:
            timestamp = event.get("Timestamp", "")
            resource = event.get("LogicalResourceId", "")
            status = event.get("ResourceStatus", "")
            reason = event.get("ResourceStatusReason", "")
            logger.info(f"{timestamp} | {resource:30s} | {status}")
            if reason:
                logger.info(f"  Reason: {reason}")
        logger.info("-" * 80)
    except Exception as e:
        logger.warning(f"Could not fetch stack events: {e}")


def show_stack_outputs():
    """Show stack outputs"""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        if stacks := response.get("Stacks", []):
            if outputs := stacks[0].get("Outputs", []):
                logger.info("\nStack Outputs:")
                logger.info("=" * 80)
                for output in outputs:
                    key = output.get("OutputKey", "")
                    value = output.get("OutputValue", "")
                    description = output.get("Description", "")
                    logger.info(f"{key:30s} : {value}")
                    if description:
                        logger.info(f"{'':30s}   ({description})")
                logger.info("=" * 80)
    except Exception as e:
        logger.warning(f"Could not fetch stack outputs: {e}")


def delete_stack():
    """Delete CloudFormation stack"""
    logger.info(f"Deleting stack: {STACK_NAME}")

    try:
        cfn.delete_stack(StackName=STACK_NAME)
        logger.info("Waiting for stack deletion...")

        waiter = cfn.get_waiter("stack_delete_complete")
        waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 5, "MaxAttempts": 60})

        logger.info("✓ Stack deleted successfully!")
    except Exception as e:
        logger.error(f"Error deleting stack: {e}")
        raise


def get_stack_status():
    """Get current stack status"""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        if stacks := response.get("Stacks", []):
            status = stacks[0].get("StackStatus", "UNKNOWN")
            logger.info(f"Stack Status: {status}")
            return status
        else:
            logger.info("Stack does not exist")
            return None
    except cfn.exceptions.ClientError as e:
        if "does not exist" in str(e):
            logger.info("Stack does not exist")
            return None
        raise


if __name__ == "__main__":
    import sys

    logger.info("=" * 80)
    logger.info("CloudFormation Stack Deployment (LocalStack)")
    logger.info("=" * 80)

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "delete":
            delete_stack()
        elif command == "status":
            get_stack_status()
            show_stack_outputs()
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Usage: python deploy_stack.py [delete|status]")
    else:
        deploy_stack()

    logger.info("=" * 80)


