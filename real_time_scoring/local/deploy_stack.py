"""
Deploy CloudFormation stack to LocalStack
"""

import logging
import time
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LocalStack endpoint
ENDPOINT_URL = "http://localhost:4566"
STACK_NAME = "credit-scoring-stack"
S3_BUCKET = "credit-scoring-models"

# AWS clients
cfn = boto3.client("cloudformation", endpoint_url=ENDPOINT_URL, region_name="us-east-1")
s3 = boto3.client("s3", endpoint_url=ENDPOINT_URL, region_name="us-east-1")
lambda_client = boto3.client(
    "lambda", endpoint_url=ENDPOINT_URL, region_name="us-east-1"
)
kinesis_client = boto3.client(
    "kinesis", endpoint_url=ENDPOINT_URL, region_name="us-east-1"
)


def read_template():
    """Read CloudFormation template."""
    template_path = Path(__file__).parent / "cloudformation" / "infrastructure.yml"
    with open(template_path, "r") as f:
        return f.read()


def setup_s3_bucket():
    """Create S3 bucket if it doesn't exist."""
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ S3 bucket '{S3_BUCKET}' already exists")
    except s3.exceptions.ClientError:
        logger.info(f"Creating S3 bucket '{S3_BUCKET}'...")
        s3.create_bucket(Bucket=S3_BUCKET)
        logger.info(f"✓ Created S3 bucket '{S3_BUCKET}'")


def package_and_upload_lambda():
    """Package Lambda function and upload to S3."""
    from package_lambda import package_lambda

    logger.info("Packaging Lambda function...")
    zip_path = package_lambda()

    # Upload to S3
    s3_key = "lambda/credit-scoring-lambda.zip"
    logger.info(f"Uploading Lambda package to s3://{S3_BUCKET}/{s3_key}...")
    s3.upload_file(str(zip_path), S3_BUCKET, s3_key)
    logger.info("✓ Uploaded Lambda package")

    return s3_key


def force_cleanup_all_resources():
    """Aggressively clean up ALL resources before stack operations."""
    logger.info("Performing aggressive resource cleanup...")

    # 1. Delete ALL event source mappings (even ones not in our stack)
    cleanup_event_source_mappings(aggressive=True)

    # 2. Try to delete the Lambda function directly if it exists
    try:
        lambda_client.delete_function(FunctionName="CreditScoringLambda")
        logger.info("✓ Deleted Lambda function directly")
        time.sleep(2)  # Give LocalStack time to process
    except lambda_client.exceptions.ResourceNotFoundException:
        logger.debug("Lambda function doesn't exist")
    except Exception as e:
        logger.debug(f"Could not delete Lambda function: {e}")

    # 3. Try to delete the Kinesis stream directly if it exists
    try:
        kinesis_client.delete_stream(
            StreamName="LoanApplicationsStream", EnforceConsumerDeletion=True
        )
        logger.info("✓ Deleted Kinesis stream directly")
        time.sleep(2)  # Give LocalStack time to process
    except kinesis_client.exceptions.ResourceNotFoundException:
        logger.debug("Kinesis stream doesn't exist")
    except Exception as e:
        logger.debug(f"Could not delete Kinesis stream: {e}")

    logger.info("✓ Aggressive cleanup complete")


def cleanup_event_source_mappings(aggressive=False):
    """
    Clean up Kinesis Event Source Mappings.

    Args:
        aggressive: If True, delete ALL mappings. If False, only delete ones related to our stream.
    """
    try:
        # List ALL event source mappings
        response = lambda_client.list_event_source_mappings()
        all_mappings = response.get("EventSourceMappings", [])

        if not all_mappings:
            logger.info("No event source mappings found")
            return

        # Filter mappings to delete
        if aggressive:
            mappings_to_delete = all_mappings
            logger.info(
                f"Aggressive mode: Found {len(mappings_to_delete)} total mapping(s) to delete"
            )
        else:
            mappings_to_delete = [
                m
                for m in all_mappings
                if "LoanApplicationsStream" in m.get("EventSourceArn", "")
            ]
            logger.info(
                f"Found {len(mappings_to_delete)} mapping(s) related to LoanApplicationsStream"
            )

        if not mappings_to_delete:
            logger.info("No event source mappings to clean up")
            return

        # Delete each mapping
        for mapping in mappings_to_delete:
            uuid = mapping.get("UUID")
            state = mapping.get("State", "Unknown")
            arn = mapping.get("EventSourceArn", "Unknown")

            logger.info(f"Processing mapping {uuid} (State: {state}, ARN: {arn})")

            # Step 1: Disable if enabled
            if state in ["Enabled", "Enabling"]:
                try:
                    logger.info(f"  Disabling mapping {uuid}...")
                    lambda_client.update_event_source_mapping(UUID=uuid, Enabled=False)
                    _wait_for_mapping_state(
                        uuid, target_states=["Disabled", "Disabling"], timeout=15
                    )
                except Exception as e:
                    logger.debug(f"  Could not disable: {e}")

            # Step 2: Delete
            try:
                logger.info(f"Deleting mapping {uuid}...")
                lambda_client.delete_event_source_mapping(UUID=uuid)
                _wait_for_mapping_deletion(uuid, timeout=30)
                logger.info(f"✓ Deleted mapping {uuid}")
            except lambda_client.exceptions.ResourceNotFoundException:
                logger.info(f"✓ Mapping {uuid} already deleted")
            except Exception as e:
                logger.warning(f"  ✗ Could not delete mapping {uuid}: {e}")

        # Final verification
        time.sleep(2)
        response = lambda_client.list_event_source_mappings()
        remaining = response.get("EventSourceMappings", [])

        if not aggressive:
            remaining = [
                m
                for m in remaining
                if "LoanApplicationsStream" in m.get("EventSourceArn", "")
            ]
        remaining_count = len(remaining)
        if remaining_count > 0:
            logger.warning(f"⚠ {remaining_count} mapping(s) still remain after cleanup")
            for m in remaining:
                logger.warning(f"  - UUID: {m.get('UUID')}, State: {m.get('State')}")
        else:
            logger.info("✓ All event source mappings cleaned up successfully")

    except Exception as e:
        logger.warning(f"Error cleaning up event source mappings: {e}")


def _wait_for_mapping_state(uuid, target_states, timeout=30):
    """Wait for mapping to reach target state(s)."""
    for _ in range(timeout):
        try:
            mapping = lambda_client.get_event_source_mapping(UUID=uuid)
            current_state = mapping.get("State", "Unknown")
            if current_state in target_states:
                return True
            time.sleep(1)
        except lambda_client.exceptions.ResourceNotFoundException:
            return True  # Mapping was deleted
    return False


def _wait_for_mapping_deletion(uuid, timeout=30):
    """Wait for mapping to be fully deleted."""
    for _ in range(timeout):
        try:
            lambda_client.get_event_source_mapping(UUID=uuid)
            time.sleep(1)
        except lambda_client.exceptions.ResourceNotFoundException:
            return True
    return False


def deploy_stack():
    """Deploy CloudFormation stack."""
    logger.info(f"Deploying stack: {STACK_NAME}")

    # Setup S3 bucket
    setup_s3_bucket()

    # Package and upload Lambda
    lambda_s3_key = package_and_upload_lambda()

    template_body = read_template()

    try:
        # Check if stack exists
        stack_exists = False
        try:
            cfn.describe_stacks(StackName=STACK_NAME)
            stack_exists = True
        except cfn.exceptions.ClientError as e:
            if "does not exist" not in str(e):
                raise

        if stack_exists:
            logger.info("Stack exists, updating...")
            action = _update_stack(template_body)
        else:
            logger.info("Stack doesn't exist, creating...")
            action = _create_stack(template_body)

        # Wait for stack to complete
        if action != "NONE":
            _wait_for_stack_completion(action)

        # Show stack outputs
        show_stack_outputs()

        # Update Lambda function code
        logger.info("\nUpdating Lambda function code...")
        update_lambda_code(lambda_s3_key)

    except Exception as e:
        logger.error(f"Error deploying stack: {e}")
        raise


def _update_stack(template_body):
    """Update existing stack."""
    try:
        cfn.update_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        return "UPDATE"
    except cfn.exceptions.ClientError as e:
        error_msg = str(e)
        if "No updates are to be performed" in error_msg:
            logger.info("✓ Stack is already up to date")
            return "NONE"
        elif "ResourceConflictException" in error_msg:
            logger.error(
                "Resource conflict detected. Please run 'make restart-localstack' to start fresh."
            )
            raise
        else:
            raise


def _create_stack(template_body):
    """Create new stack."""
    try:
        cfn.create_stack(
            StackName=STACK_NAME,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        return "CREATE"
    except cfn.exceptions.ClientError as e:
        error_msg = str(e)
        if "ResourceConflictException" in error_msg:
            logger.error(
                "Resource conflict detected. Please run 'make restart-localstack' to start fresh."
            )
        raise


def _wait_for_stack_completion(action):
    """Wait for stack operation to complete."""
    logger.info(f"Waiting for stack {action} to complete...")
    waiter = cfn.get_waiter(f"stack_{action.lower()}_complete")

    try:
        waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        logger.info(f"✓ Stack {action} completed successfully!")
    except Exception as e:
        logger.error(f"Stack {action} failed or timed out: {e}")
        show_stack_events()
        show_stack_failure_details()
        raise


def update_lambda_code(s3_key):
    """Update Lambda function code from S3"""
    try:
        lambda_client.update_function_code(
            FunctionName="CreditScoringLambda", S3Bucket=S3_BUCKET, S3Key=s3_key
        )
        logger.info("✓ Lambda function code updated")
    except Exception as e:
        logger.warning(f"Could not update Lambda code: {e}")


def show_stack_events():
    """Show recent stack events"""
    try:
        response = cfn.describe_stack_events(StackName=STACK_NAME)
        events = response.get("StackEvents", [])

        logger.info("Recent Stack Events:")
        for event in events[:15]:
            timestamp = event.get("Timestamp", "")
            resource = event.get("LogicalResourceId", "")
            status = event.get("ResourceStatus", "")
            reason = event.get("ResourceStatusReason", "")
            logger.info(f"{timestamp} | {resource:35s} | {status}")
            if reason:
                logger.info(f"  Reason: {reason}")
    except Exception as e:
        logger.warning(f"Could not fetch stack events: {e}")


def show_stack_failure_details():
    """Show detailed failure information"""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stacks = response.get("Stacks", [])
        if not stacks:
            return

        stack = stacks[0]
        status = stack.get("StackStatus", "")
        status_reason = stack.get("StackStatusReason", "")

        logger.info("\n" + "=" * 80)
        logger.info("Stack Failure Details:")
        logger.info("=" * 80)
        logger.info(f"Stack Status: {status}")
        if status_reason:
            logger.info(f"Status Reason: {status_reason}")

        # Find failed resources
        response = cfn.describe_stack_events(StackName=STACK_NAME)
        events = response.get("StackEvents", [])
        if failed_resources := [
            e for e in events if e.get("ResourceStatus", "").endswith("_FAILED")
        ]:
            logger.info(f"\nFailed Resources ({len(failed_resources)}):")
            for event in failed_resources[:5]:
                resource = event.get("LogicalResourceId", "")
                status = event.get("ResourceStatus", "")
                reason = event.get("ResourceStatusReason", "")
                logger.info(f"  - {resource}: {status}")
                if reason:
                    logger.info(f"    Reason: {reason}")
    except Exception as e:
        logger.warning(f"Could not fetch stack failure details: {e}")


def show_stack_outputs():
    """Show stack outputs"""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stacks = response.get("Stacks", [])
        if not stacks:
            return

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
    except Exception as e:
        logger.warning(f"Could not fetch stack outputs: {e}")


def delete_stack():  # sourcery skip: extract-duplicate-method
    """Delete CloudFormation stack"""
    logger.info(f"Deleting stack: {STACK_NAME}")

    # STEP 1: Force cleanup of all resources FIRST
    logger.info("STEP 1: Pre-deletion cleanup")
    force_cleanup_all_resources()
    time.sleep(3)

    # STEP 2: Delete the stack
    logger.info("STEP 2: Deleting CloudFormation stack")

    try:
        cfn.delete_stack(StackName=STACK_NAME)
        logger.info("Waiting for stack deletion...")

        waiter = cfn.get_waiter("stack_delete_complete")
        waiter.wait(StackName=STACK_NAME, WaiterConfig={"Delay": 5, "MaxAttempts": 60})
        logger.info("✓ Stack deleted successfully!")

    except cfn.exceptions.ClientError as e:
        if "does not exist" in str(e):
            logger.info("Stack does not exist (already deleted)")
        else:
            logger.error(f"Error deleting stack: {e}")
            # Try cleanup anyway
            force_cleanup_all_resources()
            raise
    except Exception as e:
        # Check if stack failed to delete
        try:
            response = cfn.describe_stacks(StackName=STACK_NAME)
            if stacks := response.get("Stacks", []):
                status = stacks[0].get("StackStatus", "")
                if status == "DELETE_FAILED":
                    logger.warning(
                        "Stack deletion failed, retrying with aggressive cleanup..."
                    )
                    force_cleanup_all_resources()
                    time.sleep(5)

                    cfn.delete_stack(StackName=STACK_NAME)
                    waiter = cfn.get_waiter("stack_delete_complete")
                    waiter.wait(
                        StackName=STACK_NAME,
                        WaiterConfig={"Delay": 5, "MaxAttempts": 60},
                    )
                    logger.info("✓ Stack deleted successfully on retry!")
                    return
        except cfn.exceptions.ClientError as check_error:
            if "does not exist" in str(check_error):
                logger.info("✓ Stack deleted successfully")
                force_cleanup_all_resources()  # Final cleanup
                return

        logger.error(f"Error deleting stack: {e}")
        force_cleanup_all_resources()  # Try cleanup anyway
        raise

    # STEP 3: Final cleanup to ensure nothing is left
    logger.info("STEP 3: Post-deletion cleanup")
    force_cleanup_all_resources()

    logger.info("\n✓ Stack and all resources deleted successfully!")


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
        elif command == "cleanup":
            logger.info("Performing aggressive cleanup of all resources...")
            force_cleanup_all_resources()
            logger.info("✓ Cleanup complete")
        else:
            logger.error(f"Unknown command: {command}")
            logger.info("Usage: python deploy_stack.py [delete|status|cleanup]")
    else:
        deploy_stack()

    logger.info("=" * 80)
