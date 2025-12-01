"""
Run the limit increase workflow locally (simulates Step Functions)
This is easier for local testing than using Step Functions in LocalStack
"""
import sys
import os
import json

# Add lambda_functions to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lambda_functions'))

from query_batch_scores import lambda_handler as query_handler
from sagemaker_inference import lambda_handler as inference_handler
from evaluate_limit_increase import lambda_handler as evaluate_handler
from update_database import lambda_handler as update_handler

class MockContext:
    """Mock Lambda context"""
    def __init__(self):
        self.function_name = "test"
        self.memory_limit_in_mb = 256
        self.invoked_function_arn = "arn:aws:lambda:us-east-1:123456789012:function:test"
        self.aws_request_id = "test-request-id"

def run_workflow():
    """Run the complete workflow"""
    print("=" * 80)
    print("Running Limit Increase Workflow")
    print("=" * 80)
    
    context = MockContext()
    
    # Step 1: Query batch scores
    print("\n[Step 1] Querying batch scores from PostgreSQL...")
    query_result = query_handler({}, context)
    
    if query_result['statusCode'] != 200:
        print(f"✗ Error querying batch scores: {query_result.get('error')}")
        return
    
    customers = query_result.get('customers', [])
    print(f"✓ Found {len(customers)} eligible customers")
    
    if len(customers) == 0:
        print("  No eligible customers found. Exiting.")
        return
    
    # Step 2: SageMaker inference
    print("\n[Step 2] Running SageMaker inference...")
    inference_result = inference_handler({'customers': customers}, context)
    
    if inference_result['statusCode'] != 200:
        print(f"✗ Error in inference: {inference_result.get('error')}")
        return
    
    scored_customers = inference_result.get('scored_customers', [])
    print(f"✓ Scored {len(scored_customers)} customers")
    
    # Step 3: Evaluate limit increase
    print("\n[Step 3] Evaluating limit increase eligibility...")
    evaluate_result = evaluate_handler({'scored_customers': scored_customers}, context)
    
    if evaluate_result['statusCode'] != 200:
        print(f"✗ Error evaluating: {evaluate_result.get('error')}")
        return
    
    evaluations = evaluate_result.get('evaluations', [])
    approved_count = evaluate_result.get('approved_count', 0)
    declined_count = evaluate_result.get('declined_count', 0)
    
    print(f"✓ Evaluated {len(evaluations)} customers")
    print(f"  ✅ Approved: {approved_count}")
    print(f"  ❌ Declined: {declined_count}")
    
    # Show some examples
    print("\n  Sample evaluations:")
    for eval_result in evaluations[:5]:
        decision_emoji = "✅" if eval_result['decision'] == 'APPROVED' else "❌"
        print(f"    {decision_emoji} {eval_result['customer_id']}: "
              f"Score {eval_result['new_score']:.0f}, "
              f"Limit ${eval_result['current_limit']:,.0f} → ${eval_result['new_limit']:,.0f}")
    
    # Step 4: Update database
    print("\n[Step 4] Updating PostgreSQL...")
    update_result = update_handler({'evaluations': evaluations}, context)
    
    if update_result['statusCode'] != 200:
        print(f"✗ Error updating database: {update_result.get('error')}")
        return
    
    updated_count = update_result.get('updated_count', 0)
    print(f"✓ Updated {updated_count} customer records")
    
    print("\n" + "=" * 80)
    print("✓ Workflow completed successfully!")
    print("=" * 80)
    print(f"\n  Run 'make check-results' to see the results in PostgreSQL")

if __name__ == "__main__":
    run_workflow()


