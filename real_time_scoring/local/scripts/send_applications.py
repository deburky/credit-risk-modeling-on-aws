"""
Send test loan applications to Kinesis stream
Simulates credit applications with various risk profiles
"""

import boto3
import random
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LocalStack endpoint
ENDPOINT_URL = 'http://localhost:4566'
STREAM_NAME = 'LoanApplicationsStream'

kinesis = boto3.client('kinesis', endpoint_url=ENDPOINT_URL, region_name='us-east-1')


def generate_application(risk_profile='medium'):
    """
    Generate a synthetic loan application
    
    Features:
    - credit_score (300-850)
    - annual_income ($)
    - debt_to_income_ratio (0-1)
    - employment_length (years)
    - num_delinquencies
    - credit_utilization (0-1)
    - num_credit_lines
    - loan_amount ($)
    """
    
    if risk_profile == 'low':  # Good applicant - should be APPROVED
        credit_score = random.uniform(700, 850)
        income = random.uniform(60000, 150000)
        debt_to_income = random.uniform(0.1, 0.3)
        employment_length = random.uniform(3, 15)
        delinquencies = 0
        credit_utilization = random.uniform(0.1, 0.3)
        num_credit_lines = random.randint(3, 8)
        loan_amount = random.uniform(10000, 50000)
        
    elif risk_profile == 'high':  # Risky applicant - should be DECLINED
        credit_score = random.uniform(300, 600)
        income = random.uniform(20000, 40000)
        debt_to_income = random.uniform(0.5, 0.9)
        employment_length = random.uniform(0, 2)
        delinquencies = random.randint(1, 5)
        credit_utilization = random.uniform(0.7, 1.0)
        num_credit_lines = random.randint(1, 3)
        loan_amount = random.uniform(5000, 30000)
        
    else:  # Medium risk
        credit_score = random.uniform(600, 700)
        income = random.uniform(40000, 70000)
        debt_to_income = random.uniform(0.3, 0.5)
        employment_length = random.uniform(1, 5)
        delinquencies = random.choice([0, 0, 1])
        credit_utilization = random.uniform(0.3, 0.6)
        num_credit_lines = random.randint(2, 6)
        loan_amount = random.uniform(10000, 40000)
    
    # Format as CSV
    application = f"{credit_score:.2f},{income:.2f},{debt_to_income:.3f},{employment_length:.1f},{delinquencies},{credit_utilization:.3f},{num_credit_lines},{loan_amount:.2f}"
    
    return application, risk_profile


def send_applications(num_applications=30):
    """Send loan applications to Kinesis stream"""
    logger.info(f"Sending {num_applications} loan applications...")
    
    # Mix of risk profiles
    risk_distribution = ['low'] * 12 + ['medium'] * 10 + ['high'] * 8
    random.shuffle(risk_distribution)
    
    approved_sent = 0
    declined_sent = 0
    
    for i in range(num_applications):
        risk_profile = risk_distribution[i] if i < len(risk_distribution) else 'medium'
        application, profile = generate_application(risk_profile)
        
        # Estimate expected decision
        expected = "✅ APPROVED" if profile == 'low' else "❌ DECLINED"
        
        if profile == 'low':
            approved_sent += 1
        elif profile == 'high':
            declined_sent += 1
        
        try:
            kinesis.put_record(
                StreamName=STREAM_NAME,
                Data=application,
                PartitionKey=f"partition-{i % 3}"
            )
            
            logger.info(f"Application #{i+1:3d} sent - {profile.upper():6s} risk - {expected}")
            
            # Small delay to simulate real-time
            time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error sending application: {e}")
    
    logger.info(f"\n✓ Sent {num_applications} applications")
    logger.info(f"  Expected: ~{approved_sent} approved, ~{declined_sent} declined")


if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("Sending loan applications to Kinesis")
    logger.info("=" * 60)
    
    send_applications(num_applications=30)
    
    logger.info("=" * 60)
    logger.info("✓ Done! Check results with: make check-approvals")
    logger.info("=" * 60)

