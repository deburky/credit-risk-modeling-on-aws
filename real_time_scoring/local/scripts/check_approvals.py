"""
Check approved applications in DynamoDB
"""

import logging

import boto3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LocalStack endpoint
ENDPOINT_URL = "http://localhost:4566"

dynamodb = boto3.client("dynamodb", endpoint_url=ENDPOINT_URL, region_name="us-east-1")


def check_approvals():
    """Query DynamoDB for approved applications"""
    logger.info("Checking approved applications...")

    try:
        response = dynamodb.scan(TableName="ApprovedApplications")

        items = response.get("Items", [])

        if not items:
            logger.info("\n📊 No approved applications yet")
            logger.info("   Try sending applications: make send-applications")
            return

        logger.info(f"\n📊 Found {len(items)} approved application(s):\n")
        logger.info("=" * 80)

        for idx, item in enumerate(items, 1):
            app_id = item.get("application_id", {}).get("S", "N/A")
            timestamp = item.get("timestamp", {}).get("S", "N/A")
            score = item.get("credit_score", {}).get("N", None)
            decision = item.get("decision", {}).get("S", "N/A")
            app_data = item.get("application_data", {}).get("S", "N/A")

            logger.info(f"Application #{idx}")
            logger.info(f"  ID:                {app_id}")
            logger.info(f"  Timestamp:         {timestamp}")
            logger.info(f"  Decision:          ✅ {decision}")
            if score:
                logger.info(f"  Credit Score:      {float(score):.0f}")
            logger.info(f"  Application Data:  {app_data[:60]}...")
            logger.info("-" * 80)

        logger.info(f"\n✓ Total approved applications: {len(items)}")

    except Exception as e:
        logger.error(f"Error querying DynamoDB: {e}")


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Credit Scoring Results - Approved Applications")
    logger.info("=" * 80)

    check_approvals()
