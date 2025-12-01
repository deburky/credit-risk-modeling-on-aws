"""
Lambda function to query PostgreSQL for batch of customers eligible for limit increase evaluation
"""
import json
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Database connection parameters
# When running in Docker, use service name 'postgres'
# When running locally, use 'localhost'
DB_HOST = os.environ.get('DB_HOST', 'postgres' if os.path.exists('/.dockerenv') else 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'credit_scoring')
DB_USER = os.environ.get('DB_USER', 'creditrisk')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'creditrisk123')

def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def lambda_handler(event, context):
    """
    Query PostgreSQL for customers eligible for limit increase evaluation
    
    Criteria:
    - current_score >= 600 (good credit)
    - current_limit < 10000
    - limit_increase_decision IS NULL (not yet evaluated)
    """
    try:
        conn = get_db_connection()
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
        
        return {
            'statusCode': 200,
            'customers': customer_list,
            'count': len(customer_list)
        }
        
    except Exception as e:
        print(f"Error querying database: {e}")
        return {
            'statusCode': 500,
            'error': str(e),
            'customers': [],
            'count': 0
        }

