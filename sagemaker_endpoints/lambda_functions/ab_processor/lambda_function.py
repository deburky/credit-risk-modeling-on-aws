"""
Lambda function for A/B Processing: Assigns A/B variant, calls SageMaker, applies logic, stores in DynamoDB

This Lambda function is triggered by SQS messages and:
1. Assigns A/B test variant based on application_id
2. Calls SageMaker endpoint for scoring
3. Applies A/B test logic (different thresholds per variant)
4. Stores results in DynamoDB
"""

import json
import logging
import os
import random
import time

import boto3
from botocore.config import Config
from sagemaker.local import LocalSession
from sagemaker.predictor import Predictor

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration from environment variables
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "credit-scoring-endpoint")
DYNAMODB_TABLE_NAME = os.environ.get("DYNAMODB_TABLE_NAME", "ProcessingResults")
ENDPOINT_URL = os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566")
REGION = os.environ.get("AWS_REGION", "us-east-1")

# Configure boto3 for LocalStack (if running locally)
boto_config = Config(
    region_name=REGION,
    s3={"endpoint_url": ENDPOINT_URL, "addressing_style": "path"}
    if ENDPOINT_URL
    else None,
)

# AWS clients
dynamodb = boto3.client(
    "dynamodb",
    endpoint_url=ENDPOINT_URL or None,
    region_name=REGION,
    config=boto_config,
)

# SageMaker session (for local mode, use LocalSession; for production, use default)
if ENDPOINT_URL:
    sagemaker_session = LocalSession()
else:
    import sagemaker

    sagemaker_session = sagemaker.Session()


def assign_ab_variant(application_id):
    """
    Assign A/B test variant based on application_id (consistent assignment).

    Uses hash for deterministic assignment - same application_id always gets same variant.
    """
    return "A" if hash(application_id) % 2 == 0 else "B"


def call_sagemaker_endpoint(application_data):
    """Call SageMaker endpoint with application features."""
    try:
        predictor = Predictor(
            endpoint_name=ENDPOINT_NAME, sagemaker_session=sagemaker_session
        )

        # Extract features in correct order (matching train.py)
        feature_values = [
            application_data.get("Application_Score", 0),
            application_data.get("Bureau_Score", 0),
            application_data.get("Loan_Amount", 0),
            application_data.get("Time_with_Bank", 0),
            application_data.get("Time_in_Employment", 0),
            application_data.get("Loan_to_income", 0),
            application_data.get("Gross_Annual_Income", 0),
            application_data.get("Loan_Payment_Frequency", ""),
            application_data.get("Residential_Status", ""),
            application_data.get("Cheque_Card_Flag", ""),
            application_data.get("Existing_Customer_Flag", ""),
            application_data.get("Home_Telephone_Number", ""),
        ]

        payload = json.dumps({"instances": [feature_values]})

        response = predictor.predict(
            payload,
            initial_args={"ContentType": "application/json"},
        )

        # Parse response - handle different response types
        if isinstance(response, bytes):
            response_str = response.decode("utf-8")
            response_data = json.loads(response_str)
        elif isinstance(response, str):
            response_data = json.loads(response)
        else:
            response_data = response

        # Check for error response
        if "error" in response_data:
            logger.error(f"SageMaker endpoint error: {response_data['error']}")
            return None

        # Extract prediction
        predictions = response_data.get("predictions", [])
        if not predictions:
            logger.error("No predictions in response")
            return None

        return predictions[0]
    except Exception as e:
        logger.error(f"Error calling SageMaker endpoint: {e}")
        raise


def apply_ab_test_logic(score, variant):
    """
    Apply A/B test logic: Variant A (control) vs Variant B (treatment).

    - Variant A: Standard threshold (600)
    - Variant B: Lower threshold (550) with 10% random approval for scores below threshold
    """
    if variant == "A":
        threshold = 600
        decision = "APPROVED" if score >= threshold else "DECLINED"
        reason = f"Score {score:.2f} {'>=' if score >= threshold else '<'} threshold {threshold}"
    else:  # Variant B
        threshold = 550
        if score >= threshold:
            decision = "APPROVED"
            reason = f"Score {score:.2f} >= threshold {threshold}"
        elif random.random() < 0.10:
            decision = "APPROVED"
            reason = f"Score {score:.2f} < threshold {threshold}, but randomly approved (10% chance)"
        else:
            decision = "DECLINED"
            reason = f"Score {score:.2f} < threshold {threshold}"

    return {
        "variant": variant,
        "decision": decision,
        "score": score,
        "threshold": threshold,
        "reason": reason,
    }


def store_result(application_id, application_data, ab_result, processed_at, proba=None):
    """Store processing result in DynamoDB."""
    item = {
        "application_id": {"S": application_id},
        "processed_at": {"S": processed_at},
        "score": {"N": str(ab_result["score"])},
        "variant": {"S": ab_result["variant"]},
        "decision": {"S": ab_result["decision"]},
        "threshold": {"N": str(ab_result["threshold"])},
        "reason": {"S": ab_result["reason"]},
        "application_data": {"S": json.dumps(application_data)},
    }

    if proba is not None:
        item["proba"] = {"N": str(proba)}

    try:
        dynamodb.put_item(TableName=DYNAMODB_TABLE_NAME, Item=item)
        logger.info(
            f"Stored result for {application_id}: {ab_result['decision']} (Variant {ab_result['variant']})"
        )
    except Exception as e:
        logger.error(f"Error storing result in DynamoDB: {e}")
        raise


def process_sqs_record(record):
    """Process a single SQS record."""
    try:
        # Parse SQS message
        body = json.loads(record["body"])
        application_id = body.get("application_id", f"app-{int(time.time() * 1000)}")
        application_data = body.get("application_data", body)

        logger.info(f"Processing application {application_id}...")

        # Assign A/B test variant (this is the key difference - assignment happens here)
        variant = assign_ab_variant(application_id)
        logger.info(f"Assigned Variant {variant} to application {application_id}")

        # Call SageMaker endpoint
        prediction = call_sagemaker_endpoint(application_data)
        if prediction is None or "score" not in prediction:
            raise ValueError("Failed to get score from SageMaker endpoint")

        score = prediction["score"]
        proba = prediction.get("proba")

        logger.info(
            f"Application {application_id} - Score: {score:.2f}"
            + (f", Proba: {proba:.4f}" if proba is not None else "")
        )

        # Apply A/B test logic using assigned variant
        ab_result = apply_ab_test_logic(score, variant)

        # Store result
        processed_at = time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime())
        store_result(
            application_id, application_data, ab_result, processed_at, proba=proba
        )

        return {"statusCode": 200, "application_id": application_id, "variant": variant}

    except Exception as e:
        logger.error(f"Error processing SQS record: {e}")
        raise


def lambda_handler(event, context):
    """
    Lambda handler for SQS-triggered A/B processing.

    Event structure (SQS event):
    {
        "Records": [
            {
                "body": "{\"application_id\": \"...\", \"application_data\": {...}}",
                ...
            }
        ]
    }
    """
    logger.info(f"Received {len(event.get('Records', []))} SQS record(s)")

    results = []
    for record in event.get("Records", []):
        try:
            result = process_sqs_record(record)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process record: {e}")
            # Re-raise to trigger SQS retry/DLQ
            raise

    logger.info(f"Successfully processed {len(results)} application(s)")
    return {"statusCode": 200, "processed": len(results)}
