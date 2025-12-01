"""
Trigger Step Functions workflow execution
"""
import boto3
import json
import sys
import os

# LocalStack endpoint
ENDPOINT_URL = os.environ.get('LOCALSTACK_ENDPOINT', 'http://localhost:4566')
REGION = os.environ.get('AWS_REGION', 'us-east-1')

def trigger_workflow():
    """Start Step Functions execution"""
    try:
        # Create Step Functions client
        sfn_client = boto3.client(
            'stepfunctions',
            endpoint_url=ENDPOINT_URL,
            region_name=REGION
        )
        
        # State machine name
        state_machine_name = 'LimitIncreaseWorkflow'
        
        # List state machines to get ARN
        response = sfn_client.list_state_machines()
        state_machine_arn = None
        
        for sm in response.get('stateMachines', []):
            if sm['name'] == state_machine_name:
                state_machine_arn = sm['stateMachineArn']
                break
        
        if not state_machine_arn:
            print(f"✗ State machine '{state_machine_name}' not found")
            print("  Make sure you've deployed the CloudFormation stack")
            return
        
        # Start execution
        execution_name = f"limit-increase-{int(__import__('time').time())}"
        input_data = {}  # Empty input, Lambda will query database
        
        response = sfn_client.start_execution(
            stateMachineArn=state_machine_arn,
            name=execution_name,
            input=json.dumps(input_data)
        )
        
        execution_arn = response['executionArn']
        print(f"✓ Started execution: {execution_name}")
        print(f"  Execution ARN: {execution_arn}")
        print(f"\n  Monitor execution:")
        print(f"  aws stepfunctions describe-execution --execution-arn {execution_arn} --endpoint-url {ENDPOINT_URL}")
        
        return execution_arn
        
    except Exception as e:
        print(f"✗ Error triggering workflow: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    trigger_workflow()


