"""
Processing script to query PostgreSQL for eligible customers
"""

import argparse
import json
import os
import sys

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "psycopg2-binary"])
    import psycopg2
    from psycopg2.extras import RealDictCursor


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

    # Connect to database
    print(f"Connecting to PostgreSQL at {args.db_host}:{args.db_port}...")
    conn = get_db_connection(args)
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # Query eligible customers
    query = """
        SELECT 
            customer_id,
            current_score,
            current_limit,
            application_date
        FROM batch_scores
        WHERE current_score >= 600
          AND current_limit < 10000
          AND limit_increase_decision IS NULL
        ORDER BY current_score DESC
        LIMIT 20
    """

    cursor.execute(query)
    customers = cursor.fetchall()

    # Convert to list of dicts
    customer_list = [dict(row) for row in customers]

    cursor.close()
    conn.close()

    print(f"Found {len(customer_list)} eligible customers")

    # Save to output
    output_path = "/opt/ml/processing/output/customers.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(customer_list, f, indent=2, default=str)

    print(f"✓ Customers saved to {output_path}")


if __name__ == "__main__":
    main()
