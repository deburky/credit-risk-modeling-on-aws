"""
Lambda function to make limit increase decisions based on scored customers.

Note:
For using LocalStack S3 in this lambda function, we need to use this format:
http://s3.{REGION}.localhost.localstack.cloud:4566

This is because LocalStack S3 doesn't support the default endpoint format.
https://stackoverflow.com/questions/78018985/connection-refused-error-127-0-0-14566-for-localstack-s3
"""

import json
import os

import boto3

# Environment variables
LOCALSTACK_ENDPOINT = os.environ.get("LOCALSTACK_ENDPOINT", "http://localhost:4566")
S3_BUCKET = os.environ.get("S3_BUCKET", "credit-scoring-models")
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

# We need to use this format because LocalStack S3 doesn't support the default endpoint format
S3_ENDPOINT = os.environ.get(
    "S3_ENDPOINT", f"http://s3.{REGION}.localhost.localstack.cloud:4566"
)

# Initialize S3 client
s3_client = boto3.client("s3", endpoint_url=S3_ENDPOINT)


def lambda_handler(event, context):
    """Process scored customers and make limit increase decisions."""
    # Get S3 location from event (from previous pipeline step)
    s3_uri = event.get("s3_uri")
    if not s3_uri:
        raise ValueError("s3_uri not provided in event")

    # Parse S3 URI and read file directly
    if s3_uri.startswith("s3://"):
        s3_uri = s3_uri[5:]
    parts = s3_uri.split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""

    # Read scored customers file directly
    print(f"Reading scored customers from s3://{bucket}/{key}")
    print(f"Using S3 endpoint: {S3_ENDPOINT}")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response["Body"].read().decode("utf-8")
        scored_customers = json.loads(content)
        print(f"Loaded {len(scored_customers)} scored customers")
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"Error reading S3 object: {error_type}: {error_msg}")

        # If it's a connection error, provide more details
        if "Could not connect" in error_msg or "Connection" in error_type:
            print("S3 endpoint configuration:")
            print(f"Endpoint URL: {S3_ENDPOINT}")
            print(f"Bucket: {bucket}")
            print(f"Key: {key}")
            print(f"Region: {REGION}")
            raise RuntimeError(
                f"S3 connection failed. Endpoint: {S3_ENDPOINT}, Error: {error_msg}"
            ) from e

        if "NoSuchKey" not in error_type and "NoSuchKey" not in error_msg:
            raise

        print(f"File not found at {key}, trying to list and find JSON files...")
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket, Prefix=key or "pipeline/scored_customers/"
            )

            scored_customers = None
            for page in pages:
                if "Contents" not in page:
                    continue
                for obj in page["Contents"]:
                    if obj["Key"].endswith(".json"):
                        response = s3_client.get_object(Bucket=bucket, Key=obj["Key"])
                        content = response["Body"].read().decode("utf-8")
                        scored_customers = json.loads(content)
                        break
                if scored_customers:
                    break
            if not scored_customers:
                raise RuntimeError(
                    f"Could not find scored customers JSON file in s3://{bucket}/{key}"
                ) from e
        except Exception as list_error:
            raise RuntimeError(
                f"Could not read scored customers from S3. Original error: {error_msg}, List error: {list_error}"
            ) from list_error
    # Process each customer
    decisions = []
    for customer in scored_customers:
        customer_id = customer.get("customer_id")
        current_score = float(customer.get("current_score", 0))
        new_score = float(customer.get("new_score", current_score))
        current_limit = float(customer.get("current_limit", 0))

        # Business rules for limit increase/decrease
        decision = "KEEP"
        new_limit = current_limit
        reason = "No change"

        if new_score >= 700 and current_limit < 10000:
            # High score, increase limit
            increase_pct = min(50, (new_score - 600) / 2)
            new_limit = int(current_limit * (1 + increase_pct / 100))
            new_limit = min(new_limit, 10000)
            decision = "INCREASE"
            reason = f"High score ({new_score:.0f}), eligible for increase"
        elif new_score >= 600 and current_limit < 5000:
            # Medium score, moderate increase
            increase_pct = min(25, (new_score - 550) / 2)
            new_limit = int(current_limit * (1 + increase_pct / 100))
            new_limit = min(new_limit, 5000)
            decision = "INCREASE"
            reason = f"Good score ({new_score:.0f}), moderate increase"
        elif new_score < 500:
            # Low score, decrease limit
            decrease_pct = max(10, (500 - new_score) / 10)
            new_limit = int(current_limit * (1 - decrease_pct / 100))
            new_limit = max(new_limit, 1000)
            decision = "DECREASE"
            reason = f"Low score ({new_score:.0f}), reducing limit"

        decisions.append(
            {
                "customer_id": customer_id,
                "current_score": current_score,
                "new_score": new_score,
                "current_limit": current_limit,
                "new_limit": new_limit,
                "decision": decision,
                "reason": reason,
            }
        )

    # Save decisions to S3
    output_key = "pipeline/decisions/decisions.json"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=output_key,
        Body=json.dumps(decisions, indent=2),
        ContentType="application/json",
    )

    return {
        "statusCode": 200,
        "decisions_count": len(decisions),
        "s3_uri": f"s3://{S3_BUCKET}/{output_key}",
        "decisions": decisions[:10],  # Return first 10 for preview
    }
