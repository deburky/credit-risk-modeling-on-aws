"""
Check limit increase results from PostgreSQL
"""
import psycopg2
import sys
import os

# Database connection
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'credit_scoring')
DB_USER = os.environ.get('DB_USER', 'creditrisk')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'creditrisk123')

def check_results():
    """Query and display limit increase results"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cursor = conn.cursor()
        
        # Get summary
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                COUNT(CASE WHEN limit_increase_decision = 'APPROVED' THEN 1 END) as approved,
                COUNT(CASE WHEN limit_increase_decision = 'DECLINED' THEN 1 END) as declined,
                COUNT(CASE WHEN limit_increase_decision IS NULL THEN 1 END) as pending
            FROM batch_scores
        """)
        summary = cursor.fetchone()
        
        print("=" * 80)
        print("Limit Increase Results Summary")
        print("=" * 80)
        print(f"Total customers: {summary[0]}")
        print(f"  ✅ Approved: {summary[1]}")
        print(f"  ❌ Declined: {summary[2]}")
        print(f"  ⏳ Pending: {summary[3]}")
        print()
        
        # Get approved customers
        cursor.execute("""
            SELECT 
                customer_id,
                current_score,
                current_limit,
                new_limit,
                decision_reason,
                updated_at
            FROM batch_scores
            WHERE limit_increase_decision = 'APPROVED'
            ORDER BY new_limit DESC
            LIMIT 10
        """)
        approved = cursor.fetchall()
        
        if approved:
            print("Top Approved Limit Increases:")
            print("-" * 80)
            for row in approved:
                customer_id, current_score, current_limit, new_limit, reason, updated_at = row
                increase = new_limit - current_limit
                increase_pct = (increase / current_limit) * 100
                print(f"  {customer_id}: ${current_limit:,.0f} → ${new_limit:,.0f} (+{increase_pct:.0f}%)")
                print(f"    Score: {current_score}, Reason: {reason}")
                print()
        
        # Get declined customers
        cursor.execute("""
            SELECT 
                customer_id,
                current_score,
                current_limit,
                decision_reason
            FROM batch_scores
            WHERE limit_increase_decision = 'DECLINED'
            ORDER BY current_score DESC
            LIMIT 5
        """)
        declined = cursor.fetchall()
        
        if declined:
            print("Sample Declined Customers:")
            print("-" * 80)
            for row in declined:
                customer_id, current_score, current_limit, reason = row
                print(f"  {customer_id}: Score {current_score}, Limit ${current_limit:,.0f}")
                print(f"    Reason: {reason}")
                print()
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Error checking results: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    check_results()


