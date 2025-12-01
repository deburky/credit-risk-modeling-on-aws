"""
Setup PostgreSQL database schema and load sample batch scores
"""
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import inference
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'chapter_1', 'local'))

# Database connection
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'credit_scoring')
DB_USER = os.environ.get('DB_USER', 'creditrisk')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'creditrisk123')

def create_schema(conn):
    """Create database schema"""
    cursor = conn.cursor()
    
    # Drop table if exists
    cursor.execute("DROP TABLE IF EXISTS batch_scores CASCADE;")
    
    # Create table
    cursor.execute("""
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
    """)
    
    conn.commit()
    cursor.close()
    print("✓ Created batch_scores table")

def load_sample_data(conn):
    """Load sample batch scores"""
    cursor = conn.cursor()
    
    # Generate sample customers
    np.random.seed(42)
    n_customers = 50
    
    customers = []
    for i in range(n_customers):
        # Generate scores between 550-750
        score = int(np.random.normal(650, 50))
        score = max(550, min(750, score))
        
        # Generate limits between 2000-12000
        limit = float(np.random.uniform(2000, 12000))
        
        # Application date (last 6 months)
        days_ago = np.random.randint(0, 180)
        app_date = datetime.now() - timedelta(days=days_ago)
        
        customer_id = f"CUST_{i+1:04d}"
        
        customers.append({
            'customer_id': customer_id,
            'current_score': score,
            'current_limit': round(limit, 2),
            'application_date': app_date.date()
        })
    
    # Insert customers
    insert_query = """
        INSERT INTO batch_scores 
        (customer_id, current_score, current_limit, application_date)
        VALUES (%s, %s, %s, %s)
    """
    
    for customer in customers:
        cursor.execute(
            insert_query,
            (
                customer['customer_id'],
                customer['current_score'],
                customer['current_limit'],
                customer['application_date']
            )
        )
    
    conn.commit()
    cursor.close()
    print(f"✓ Loaded {len(customers)} sample customers")
    
    # Print summary
    cursor = conn.cursor()
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN current_score >= 600 THEN 1 END) as eligible,
            COUNT(CASE WHEN current_limit < 10000 THEN 1 END) as under_limit
        FROM batch_scores
    """)
    result = cursor.fetchone()
    print(f"  Total customers: {result[0]}")
    print(f"  Score >= 600: {result[1]}")
    print(f"  Limit < 10000: {result[2]}")

def main():
    """Main setup function"""
    print("=" * 80)
    print("Setting up PostgreSQL database for batch scoring")
    print("=" * 80)
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        print(f"✓ Connected to PostgreSQL at {DB_HOST}:{DB_PORT}")
        
        # Create schema
        create_schema(conn)
        
        # Load sample data
        load_sample_data(conn)
        
        conn.close()
        
        print("\n" + "=" * 80)
        print("✓ Database setup complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()


