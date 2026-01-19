"""Database operations for batch scoring.

Handles schema setup, data loading, updates, and queries.
"""

import json
import os
import sys
from datetime import datetime, timedelta

# Install psycopg2 if not available (for container)
try:
    import psycopg2
except ImportError:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "psycopg2-binary"]
    )
    import psycopg2


class Database:
    """Database operations for batch scoring."""

    def __init__(
        self,
        host=None,
        port=None,
        database=None,
        user=None,
        password=None,
    ):
        """Initialize database connection parameters."""
        self.host = host or os.environ.get("DB_HOST", "localhost")
        self.port = port or os.environ.get("DB_PORT", "5432")
        self.database = database or os.environ.get("DB_NAME", "credit_scoring")
        self.user = user or os.environ.get("DB_USER", "creditrisk")
        self.password = password or os.environ.get("DB_PASSWORD", "creditrisk123")

    def connect(self):
        """Create database connection."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )

    def setup_schema(self):
        """Create database schema."""
        conn = self.connect()
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS batch_scores CASCADE;")
        cursor.execute(
            """
            CREATE TABLE batch_scores (
                customer_id VARCHAR(50) PRIMARY KEY,
                current_score INTEGER NOT NULL,
                current_limit DECIMAL(10, 2) NOT NULL,
                application_date DATE NOT NULL,
                limit_increase_decision VARCHAR(20),
                new_limit DECIMAL(10, 2),
                decision_reason TEXT,
                updated_at TIMESTAMP
            );
        """
        )

        conn.commit()
        cursor.close()
        conn.close()
        print("Created batch_scores table")

    def load_sample_data(self, n_customers=50, seed=42):
        """Load sample batch scores."""
        import random

        conn = self.connect()
        cursor = conn.cursor()

        random.seed(seed)
        customers = []
        for i in range(n_customers):
            # Use random.gauss for normal distribution, random.uniform for uniform
            score = int(random.gauss(650, 50))
            score = max(550, min(750, score))
            limit = float(random.uniform(2000, 12000))
            days_ago = random.randint(0, 180)
            app_date = datetime.now() - timedelta(days=days_ago)
            customer_id = f"CUST_{i + 1:04d}"

            customers.append(
                {
                    "customer_id": customer_id,
                    "current_score": score,
                    "current_limit": round(limit, 2),
                    "application_date": app_date.date(),
                }
            )

        insert_query = """
            INSERT INTO batch_scores 
            (customer_id, current_score, current_limit, application_date)
            VALUES (%s, %s, %s, %s)
        """

        for customer in customers:
            cursor.execute(
                insert_query,
                (
                    customer["customer_id"],
                    customer["current_score"],
                    customer["current_limit"],
                    customer["application_date"],
                ),
            )

        conn.commit()
        cursor.close()

        # Print summary
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN current_score >= 600 THEN 1 END) as eligible,
                COUNT(CASE WHEN current_limit < 10000 THEN 1 END) as under_limit
            FROM batch_scores
        """
        )
        result = cursor.fetchone()
        cursor.close()
        conn.close()

        print(f"Loaded {len(customers)} sample customers")
        print(f"Total customers: {result[0]}")
        print(f"Score >= 600: {result[1]}")
        print(f"Limit < 10000: {result[2]}")

    def setup(self, n_customers=50):
        """Set up database schema and load sample data."""
        print("Setting up PostgreSQL database for batch scoring")
        self.setup_schema()
        self.load_sample_data(n_customers)
        print("Database setup complete")

    def update_decisions(self, decisions=None, decisions_file=None):
        """Update database with decisions from list or file."""
        if decisions_file:
            with open(decisions_file, "r") as f:
                decisions = json.load(f)
        elif decisions is None:
            raise ValueError("Either 'decisions' or 'decisions_file' must be provided")

        conn = self.connect()
        cursor = conn.cursor()

        updated_count = 0
        for decision_result in decisions:
            db_decision = decision_result["decision"]

            cursor.execute(
                """
                UPDATE batch_scores
                SET 
                    limit_increase_decision = %s,
                    new_limit = %s,
                    decision_reason = %s,
                    updated_at = %s
                WHERE customer_id = %s
            """,
                (
                    db_decision,
                    decision_result["new_limit"],
                    decision_result["reason"],
                    datetime.now(),
                    decision_result["customer_id"],
                ),
            )
            updated_count += 1

        conn.commit()
        cursor.close()
        conn.close()

        print(f"Updated {updated_count} customer records in database")
        return updated_count

    def check_results(self):
        """Query and display limit increase results."""
        conn = self.connect()
        cursor = conn.cursor()

        # Get summary - only count customers with decisions (eligible customers that were processed)
        cursor.execute(
            """
            SELECT 
                COUNT(CASE WHEN limit_increase_decision IS NOT NULL THEN 1 END) as total_processed,
                COUNT(CASE WHEN limit_increase_decision = 'INCREASE' THEN 1 END) as increase,
                COUNT(CASE WHEN limit_increase_decision = 'DECREASE' THEN 1 END) as decrease,
                COUNT(CASE WHEN limit_increase_decision = 'KEEP' THEN 1 END) as keep
            FROM batch_scores
            WHERE current_score >= 600
              AND current_limit < 10000
        """
        )
        summary = cursor.fetchone()

        print("Limit Increase Results Summary")
        print(f"Total eligible customers processed: {summary[0]}")
        print(f"Increase: {summary[1]}")
        print(f"Decrease: {summary[2]}")
        print(f"Keep: {summary[3]}")

        # Get increase customers
        cursor.execute(
            """
            SELECT 
                customer_id,
                current_score,
                current_limit,
                new_limit,
                decision_reason,
                updated_at
            FROM batch_scores
            WHERE limit_increase_decision = 'INCREASE'
            ORDER BY new_limit DESC
            LIMIT 10
        """
        )
        if increase := cursor.fetchall():
            print("Top Limit Increases:")
            for row in increase:
                (
                    customer_id,
                    current_score,
                    current_limit,
                    new_limit,
                    reason,
                    updated_at,
                ) = row
                increase = new_limit - current_limit
                increase_pct = (increase / current_limit) * 100
                print(
                    f"{customer_id}: ${current_limit:,.0f} -> ${new_limit:,.0f} (+{increase_pct:.0f}%)"
                )
                print(f"Score: {current_score}, Reason: {reason}")

        # Get declined customers
        cursor.execute(
            """
            SELECT 
                customer_id,
                current_score,
                current_limit,
                decision_reason
            FROM batch_scores
            WHERE limit_increase_decision = 'DECREASE'
            ORDER BY current_score DESC
            LIMIT 5
        """
        )
        if decrease := cursor.fetchall():
            print("Sample Limit Decreases:")
            for row in decrease:
                customer_id, current_score, current_limit, reason = row
                print(
                    f"{customer_id}: Score {current_score}, Limit ${current_limit:,.0f}"
                )
                print(f"Reason: {reason}")

        cursor.close()
        conn.close()

    def query_eligible_customers(self, limit=20, output_path=None):
        """Query eligible customers for batch scoring.

        Args:
            limit: Maximum number of customers to return
            output_path: Optional path to save JSON file (for ProcessingStep)

        Returns:
            List of customer dictionaries
        """
        conn = self.connect()
        cursor = conn.cursor()

        # Build query - if limit is None or 0, process all eligible customers
        if limit and limit > 0:
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
                LIMIT %s
            """
            cursor.execute(query, (limit,))
        else:
            # Process all eligible customers (no limit)
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
            """
            cursor.execute(query)
        customers = cursor.fetchall()

        # Convert to list of dicts
        customer_list = [
            {
                "customer_id": row[0],
                "current_score": row[1],
                "current_limit": float(row[2]),
                "application_date": str(row[3]),
            }
            for row in customers
        ]

        cursor.close()
        conn.close()

        print(f"Found {len(customer_list)} eligible customers")

        # Save to file if output_path provided (for ProcessingStep)
        if output_path:
            import json
            import os

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(customer_list, f, indent=2, default=str)
            print(f"Customers saved to {output_path}")

        return customer_list


def main():
    """CLI interface for database operations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Database operations for batch scoring"
    )
    parser.add_argument(
        "command",
        choices=["setup", "update", "check", "query"],
        help="Command to execute",
    )
    parser.add_argument(
        "--decisions-file", type=str, help="Path to decisions JSON file"
    )
    parser.add_argument("--db-host", type=str, default="localhost")
    parser.add_argument("--db-port", type=str, default="5432")
    parser.add_argument("--db-name", type=str, default="credit_scoring")
    parser.add_argument("--db-user", type=str, default="creditrisk")
    parser.add_argument("--db-password", type=str, default="creditrisk123")
    parser.add_argument(
        "--n-customers", type=int, default=50, help="Number of sample customers"
    )
    parser.add_argument("--limit", type=int, default=20, help="Limit for query command")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path for query command (for ProcessingStep)",
    )

    args = parser.parse_args()

    db = Database(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password,
    )

    try:
        if args.command == "setup":
            db.setup(n_customers=args.n_customers)
        elif args.command == "update":
            if not args.decisions_file:
                parser.error("--decisions-file is required for update command")
            db.update_decisions(decisions_file=args.decisions_file)
        elif args.command == "check":
            db.check_results()
        elif args.command == "query":
            output_path = args.output_path or "/opt/ml/processing/output/customers.json"
            db.query_eligible_customers(limit=args.limit, output_path=output_path)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
