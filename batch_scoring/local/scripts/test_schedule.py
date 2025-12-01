"""
Test EventBridge scheduled rule (simulate scheduled execution)
For LocalStack, we can manually trigger the rule to test it
"""
import boto3
import json
import sys
import os
import time

# LocalStack endpoint
ENDPOINT_URL = os.environ.get('LOCALSTACK_ENDPOINT', 'http://localhost:4566')
REGION = os.environ.get('AWS_REGION', 'us-east-1')

def test_scheduled_rule():
    """Test the EventBridge scheduled rule by manually triggering it"""
    try:
        # Create EventBridge client
        events_client = boto3.client(
            'events',
            endpoint_url=ENDPOINT_URL,
            region_name=REGION
        )
        
        # Create Step Functions client
        sfn_client = boto3.client(
            'stepfunctions',
            endpoint_url=ENDPOINT_URL,
            region_name=REGION
        )
        
        # Get the rule
        rule_name = 'LimitIncreaseDailySchedule'
        
        try:
            response = events_client.describe_rule(Name=rule_name)
            print(f"✓ Found EventBridge rule: {rule_name}")
            print(f"  Schedule: {response.get('ScheduleExpression', 'N/A')}")
            print(f"  State: {response.get('State', 'N/A')}")
        except events_client.exceptions.ResourceNotFoundException:
            print(f"✗ Rule '{rule_name}' not found")
            print("  Make sure you've deployed the CloudFormation stack")
            return
        
        # Get the state machine ARN from targets
        targets = events_client.list_targets_by_rule(Rule=rule_name)
        if not targets.get('Targets'):
            print("✗ No targets found for rule")
            return
        
        state_machine_arn = targets['Targets'][0]['Arn']
        print(f"\n✓ State Machine ARN: {state_machine_arn}")
        
        # Manually trigger the rule by starting Step Functions execution
        print("\n🚀 Manually triggering scheduled execution...")
        execution_name = f"scheduled-{int(time.time())}"
        input_data = {}  # Empty input
        
        response = sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data)
        )
        
        execution_arn = response['executionArn']
        print(f"✓ Started execution: {execution_name}")
        print(f"  Execution ARN: {execution_arn}")
        
        # Wait a bit and check status
        print("\n⏳ Waiting 5 seconds for execution to start...")
        time.sleep(5)
        
        exec_response = sfn_client.describe_execution(executionArn=execution_arn)
        status = exec_response.get('status', 'UNKNOWN')
        print(f"  Status: {status}")
        
        if status == 'RUNNING':
            print("\n✓ Execution is running!")
            print(f"  Monitor with: aws stepfunctions describe-execution --execution-arn {execution_arn} --endpoint-url {ENDPOINT_URL}")
        elif status == 'SUCCEEDED':
            print("\n✓ Execution completed successfully!")
        elif status == 'FAILED':
            print("\n✗ Execution failed")
            print(f"  Error: {exec_response.get('error', 'Unknown error')}")
        
        return execution_arn
        
    except Exception as e:
        print(f"✗ Error testing scheduled rule: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_scheduled_rule()


