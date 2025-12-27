"""
Check A/B test results from DynamoDB
Shows variant distribution and assignment verification
"""

import logging
from collections import Counter

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
dynamodb = boto3.client(
    "dynamodb", endpoint_url=ENDPOINT_URL, region_name=REGION, config=boto_config
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


def get_dynamodb_table_name():
    """Get DynamoDB table name from stack outputs."""
    outputs = get_stack_outputs()
    return outputs.get("ProcessingResultsTableName") or "ProcessingResults"


def scan_results():
    """Scan all results from DynamoDB."""
    table_name = get_dynamodb_table_name()
    logger.info(f"Scanning DynamoDB table: {table_name}")

    results = []
    last_evaluated_key = None

    while True:
        scan_kwargs = {"TableName": table_name}
        if last_evaluated_key:
            scan_kwargs["ExclusiveStartKey"] = last_evaluated_key

        response = dynamodb.scan(**scan_kwargs)
        items = response.get("Items", [])

        for item in items:
            # Convert DynamoDB format to Python dict
            result = {
                "application_id": item.get("application_id", {}).get("S", ""),
                "processed_at": item.get("processed_at", {}).get("S", ""),
                "variant": item.get("variant", {}).get("S", ""),
                "score": float(item.get("score", {}).get("N", "0")),
                "decision": item.get("decision", {}).get("S", ""),
                "threshold": float(item.get("threshold", {}).get("N", "0")),
                "reason": item.get("reason", {}).get("S", ""),
            }
            if "proba" in item:
                result["proba"] = float(item.get("proba", {}).get("N", "0"))
            results.append(result)

        last_evaluated_key = response.get("LastEvaluatedKey")
        if not last_evaluated_key:
            break

    return results


def analyze_results(results):
    """Analyze A/B test results."""
    if not results:
        logger.warning("No results found in DynamoDB")
        return

    logger.info(f"\n{'=' * 60}")
    logger.info("A/B Test Results Analysis")
    logger.info(f"{'=' * 60}")
    logger.info(f"Total applications processed: {len(results)}")

    # Variant distribution
    variant_counts = Counter(r["variant"] for r in results)
    logger.info("\nVariant Distribution:")
    for variant, count in sorted(variant_counts.items()):
        percentage = (count / len(results)) * 100
        logger.info(f"  Variant {variant}: {count} ({percentage:.1f}%)")

    # Decision breakdown by variant
    logger.info("\nDecision Breakdown by Variant:")
    for variant in sorted(variant_counts.keys()):
        variant_results = [r for r in results if r["variant"] == variant]
        decisions = Counter(r["decision"] for r in variant_results)
        logger.info(f"  Variant {variant}:")
        for decision, count in sorted(decisions.items()):
            percentage = (count / len(variant_results)) * 100
            logger.info(f"{decision}: {count} ({percentage:.1f}%)")

    # Threshold verification
    logger.info("\nThreshold Verification:")
    variant_a_results = [r for r in results if r["variant"] == "A"]
    variant_b_results = [r for r in results if r["variant"] == "B"]

    if variant_a_results:
        thresholds_a = {r["threshold"] for r in variant_a_results}
        logger.info(f"  Variant A thresholds used: {sorted(thresholds_a)}")
        expected_threshold_a = 600
        if thresholds_a == {expected_threshold_a}:
            logger.info(
                f"    ✓ Variant A using correct threshold ({expected_threshold_a})"
            )
        else:
            logger.warning(
                f"✗ Variant A threshold mismatch. Expected {expected_threshold_a}, got {thresholds_a}"
            )

    if variant_b_results:
        thresholds_b = {r["threshold"] for r in variant_b_results}
        logger.info(f"  Variant B thresholds used: {sorted(thresholds_b)}")
        expected_threshold_b = 550
        if thresholds_b == {expected_threshold_b}:
            logger.info(f"✓ Variant B using correct threshold ({expected_threshold_b})")
        else:
            logger.warning(
                f"✗ Variant B threshold mismatch. Expected {expected_threshold_b}, got {thresholds_b}"
            )

    # Score statistics by variant
    logger.info("\nScore Statistics by Variant:")
    for variant in sorted(variant_counts.keys()):
        if variant_results := [r for r in results if r["variant"] == variant]:
            logger.info(f"Variant {variant}:")
    # Show sample results
    logger.info("\nSample Results (first 5):")
    for i, result in enumerate(results[:5], 1):
        logger.info(
            f"{i}. {result['application_id']}: "
            f"Variant {result['variant']}, "
            f"Score {result['score']:.2f}, "
            f"Decision {result['decision']}, "
            f"Threshold {result['threshold']}"
        )


def main():
    """Main function."""
    try:
        results = scan_results()
        analyze_results(results)
    except Exception as e:
        logger.error(f"Error checking results: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
