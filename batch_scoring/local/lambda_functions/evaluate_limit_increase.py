"""
Lambda function to evaluate limit increase eligibility based on business rules
"""
import json

def lambda_handler(event, context):
    """
    Evaluate limit increase eligibility
    
    Business Rules:
    - New score >= 650: Eligible for increase
    - Current limit < 5000: Can increase by 50%
    - Current limit >= 5000: Can increase by 25%
    - Max new limit: 20000
    """
    try:
        scored_customers = event.get('scored_customers', [])
        
        evaluations = []
        
        for customer in scored_customers:
            customer_id = customer['customer_id']
            current_score = customer['current_score']
            new_score = customer['new_score']
            current_limit = customer['current_limit']
            
            # Business rules
            if new_score >= 650:
                # Eligible for increase
                if current_limit < 5000:
                    increase_percent = 0.50  # 50% increase
                else:
                    increase_percent = 0.25  # 25% increase
                
                new_limit = int(current_limit * (1 + increase_percent))
                new_limit = min(new_limit, 20000)  # Cap at 20000
                
                decision = 'APPROVED'
                reason = f"Score {new_score:.0f} >= 650, increase by {increase_percent*100:.0f}%"
            else:
                # Not eligible
                new_limit = current_limit
                decision = 'DECLINED'
                reason = f"Score {new_score:.0f} < 650, no increase"
            
            evaluations.append({
                'customer_id': customer_id,
                'current_score': current_score,
                'new_score': new_score,
                'current_limit': current_limit,
                'new_limit': new_limit,
                'decision': decision,
                'reason': reason
            })
        
        approved_count = sum(1 for e in evaluations if e['decision'] == 'APPROVED')
        declined_count = len(evaluations) - approved_count
        
        print(f"Evaluated {len(evaluations)} customers: {approved_count} approved, {declined_count} declined")
        
        return {
            'statusCode': 200,
            'evaluations': evaluations,
            'approved_count': approved_count,
            'declined_count': declined_count
        }
        
    except Exception as e:
        print(f"Error evaluating limit increases: {e}")
        return {
            'statusCode': 500,
            'error': str(e),
            'evaluations': []
        }


