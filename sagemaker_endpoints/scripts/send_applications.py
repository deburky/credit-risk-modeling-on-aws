"""
Generate and send test loan applications to SQS queue
"""

import json
import logging
import random
import sys
import time
import uuid

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
sqs = boto3.client(
    "sqs", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)
cfn = boto3.client(
    "cloudformation", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
)


def get_stack_outputs():
    """Get CloudFormation stack outputs."""
    try:
        response = cfn.describe_stacks(StackName=STACK_NAME)
        stack = response["Stacks"][0]
        return {
            out["OutputKey"]: out["OutputValue"] for out in stack.get("Outputs", [])
        }
    except Exception as e:
        logger.error(f"Error getting stack outputs: {e}")
        return {}


def get_queue_url():
    """Get SQS queue URL from stack outputs."""
    outputs = get_stack_outputs()
    queue_url = outputs.get("ApplicationQueueUrl")
    if not queue_url:
        # Fallback: try to get queue by name
        try:
            response = sqs.get_queue_url(QueueName="application-processing-queue")
            queue_url = response["QueueUrl"]
        except Exception as e:
            logger.error(f"Could not get queue URL: {e}")
            raise
    return queue_url


def generate_application():
    """Generate a random loan application."""
    # Generate realistic credit scoring features
    application_score = random.randint(500, 800)
    bureau_score = random.randint(450, 850)
    loan_amount = random.choice([5000, 10000, 15000, 20000, 25000])
    time_with_bank = random.randint(6, 120)  # months
    time_in_employment = random.randint(6, 240)  # months
    loan_to_income = round(random.uniform(0.1, 0.5), 2)
    gross_annual_income = random.randint(30000, 150000)

    return {
        "Application_Score": application_score,
        "Bureau_Score": bureau_score,
        "Loan_Amount": loan_amount,
        "Time_with_Bank": time_with_bank,
        "Time_in_Employment": time_in_employment,
        "Loan_to_income": loan_to_income,
        "Gross_Annual_Income": gross_annual_income,
        "Loan_Payment_Frequency": random.choice(
            ["M", "W", "B"]
        ),  # Monthly, Weekly, Bi-weekly
        "Residential_Status": random.choice(
            ["H", "R", "O"]
        ),  # Homeowner, Renter, Other
        "Cheque_Card_Flag": random.choice(["Y", "N"]),
        "Existing_Customer_Flag": random.choice(["Y", "N"]),
        "Home_Telephone_Number": random.choice(["Y", "N"]),
    }


def send_application(queue_url, application_data, application_id=None):
    """Send application to SQS queue."""
    if not application_id:
        application_id = f"app-{uuid.uuid4().hex[:8]}"

    message_body = {
        "application_id": application_id,
        "application_data": application_data,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
    }

    try:
        response = sqs.send_message(
            QueueUrl=queue_url,
            MessageBody=json.dumps(message_body),
            MessageAttributes={
                "ApplicationId": {
                    "StringValue": application_id,
                    "DataType": "String",
                },
            },
        )
        logger.info(
            f"✓ Sent application {application_id} (MessageId: {response['MessageId']})"
        )
        return application_id
    except Exception as e:
        logger.error(f"Error sending application {application_id}: {e}")
        raise


def send_batch_applications(num_applications=10, delay=0.5):
    """Send multiple applications to the queue."""
    queue_url = get_queue_url()
    logger.info(f"Sending {num_applications} applications to queue: {queue_url}")

    sent_count = 0
    failed_count = 0

    for i in range(num_applications):
        try:
            application_data = generate_application()
            send_application(queue_url, application_data)
            sent_count += 1

            if delay > 0 and i < num_applications - 1:
                time.sleep(delay)
        except Exception as e:
            logger.error(f"Failed to send application {i + 1}: {e}")
            failed_count += 1

    logger.info(f"Sent: {sent_count}")
    logger.info(f"Failed: {failed_count}")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description="Send test loan applications to SQS")
    parser.add_argument(
        "--num", type=int, default=10, help="Number of applications to send"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5, help="Delay between sends (seconds)"
    )

    args = parser.parse_args()

    try:
        send_batch_applications(num_applications=args.num, delay=args.delay)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
