"""
Lambda function to update PostgreSQL with limit increase decisions
"""
import json
import os
import psycopg2
from datetime import datetime

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
    Update PostgreSQL with limit increase decisions
    """
    try:
        evaluations = event.get('evaluations', [])
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
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
                    eval_result['decision'],
                    eval_result['new_limit'],
                    eval_result['reason'],
                    datetime.now(),
                    eval_result['customer_id']
                )
            )
            updated_count += 1
        
        conn.commit()
        cursor.close()
        conn.close()
        
        print(f"Updated {updated_count} customer records")
        
        return {
            'statusCode': 200,
            'updated_count': updated_count,
            'message': f'Successfully updated {updated_count} customers'
        }
        
    except Exception as e:
        print(f"Error updating database: {e}")
        return {
            'statusCode': 500,
            'error': str(e),
            'updated_count': 0
        }

