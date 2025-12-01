"""
Processing script for SageMaker Pipeline to update PostgreSQL database

This script runs inside a SageMaker Processing job container
and updates the database with limit increase decisions.
"""

import argparse
import json
import os
import sys

# Install psycopg2 if not available (for container)
try:
    import psycopg2
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2

from datetime import datetime


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=str, default="5432")
    parser.add_argument("--db-name", type=str, default="credit_scoring")
    parser.add_argument("--db-user", type=str, default="creditrisk")
    parser.add_argument("--db-password", type=str, default="creditrisk123")

    return parser.parse_args()


def get_db_connection(args):
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )


def main():
    """Main processing function"""
    args = parse_args()

    # Load evaluations from input (handle S3 download structure)
    input_dir = "/opt/ml/processing/input/evaluations"

    # Find the JSON file (could be in subdirectories from S3)
    evaluations_file = None
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                evaluations_file = os.path.join(root, file)
                break
        if evaluations_file:
            break

    if not evaluations_file:
        # Try direct path
        evaluations_file = os.path.join(input_dir, "evaluations.json")

    print(f"Loading evaluations from {evaluations_file}...")
    with open(evaluations_file, "r") as f:
        evaluations = json.load(f)

    print(f"Found {len(evaluations)} evaluations to process")

    # Connect to database
    print(f"Connecting to PostgreSQL at {args.db_host}:{args.db_port}...")
    conn = get_db_connection(args)
    cursor = conn.cursor()

    # Update database
    updated_count = 0

    for eval_result in evaluations:
        update_query = """
            UPDATE batch_scores
            SET 
                limit_increase_decision = %s,
                new_limit = %s,
                decision_reason = %s,
                updated_at = %s
            WHERE customer_id = %s
        """

        cursor.execute(
            update_query,
            (
                eval_result["decision"],
                eval_result["new_limit"],
                eval_result["reason"],
                datetime.now(),
                eval_result["customer_id"],
            ),
        )
        updated_count += 1

    conn.commit()
    cursor.close()
    conn.close()

    print(f"✓ Updated {updated_count} customer records in database")

    # Save results to output (optional)
    output_path = "/opt/ml/processing/output/results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(
            {
                "updated_count": updated_count,
                "status": "success",
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    print(f"✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
