"""
Test if EventBridge and Step Functions are available in LocalStack
"""
import boto3
import json
import sys

ENDPOINT_URL = "http://localhost:4566"
REGION = "us-east-1"

def test_eventbridge():
    """Test EventBridge service"""
    print("=" * 80)
    print("Testing EventBridge (CloudWatch Events)")
    print("=" * 80)
    
    try:
        events_client = boto3.client(
            'events',
            endpoint_url=ENDPOINT_URL,
            region_name=REGION
        )
        
        # Try to list rules
        print("\n1. Testing list_rules()...")
        response = events_client.list_rules()
        print(f"   ✓ EventBridge is available!")
        print(f"   Found {len(response.get('Rules', []))} existing rules")
        
        # Try to create a test rule
        print("\n2. Testing create_rule()...")
        try:
            test_rule = events_client.put_rule(
                Name='test-rule',
                ScheduleExpression='rate(1 hour)',
                State='DISABLED',
                Description='Test rule to verify EventBridge works'
            )
            print(f"   ✓ Successfully created test rule!")
            print(f"   Rule ARN: {test_rule.get('RuleArn', 'N/A')}")
            
            # Clean up - delete test rule
            print("\n3. Cleaning up test rule...")
            events_client.delete_rule(Name='test-rule')
            print("   ✓ Test rule deleted")
            
        except Exception as e:
            print(f"   ✗ Error creating rule: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ EventBridge is NOT available: {e}")
        return False

def test_stepfunctions():
    """Test Step Functions service"""
    print("\n" + "=" * 80)
    print("Testing Step Functions")
    print("=" * 80)
    
    try:
        sfn_client = boto3.client(
            'stepfunctions',
            endpoint_url=ENDPOINT_URL,
            region_name=REGION
        )
        
        # Try to list state machines
        print("\n1. Testing list_state_machines()...")
        response = sfn_client.list_state_machines()
        print(f"   ✓ Step Functions is available!")
        print(f"   Found {len(response.get('stateMachines', []))} existing state machines")
        
        # Try to create a simple test state machine
        print("\n2. Testing create_state_machine()...")
        try:
            # Simple pass state machine definition
            definition = {
                "Comment": "Test state machine",
                "StartAt": "HelloWorld",
                "States": {
                    "HelloWorld": {
                        "Type": "Pass",
                        "Result": "Hello World!",
                        "End": True
                    }
                }
            }
            
            test_sm = sfn_client.create_state_machine(
                name='test-state-machine',
                definition=json.dumps(definition),
                roleArn='arn:aws:iam::123456789012:role/test-role'  # Dummy role for testing
            )
            print(f"   ✓ Successfully created test state machine!")
            print(f"   State Machine ARN: {test_sm.get('stateMachineArn', 'N/A')}")
            
            # Try to start an execution
            print("\n3. Testing start_execution()...")
            execution = sfn_client.start_execution(
                stateMachineArn=test_sm['stateMachineArn'],
                name='test-execution'
            )
            print(f"   ✓ Successfully started execution!")
            print(f"   Execution ARN: {execution.get('executionArn', 'N/A')}")
            
            # Clean up - delete state machine
            print("\n4. Cleaning up test state machine...")
            sfn_client.delete_state_machine(
                stateMachineArn=test_sm['stateMachineArn']
            )
            print("   ✓ Test state machine deleted")
            
        except Exception as e:
            print(f"   ✗ Error creating state machine: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ Step Functions is NOT available: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_localstack_services():
    """Check what services LocalStack reports as available"""
    print("\n" + "=" * 80)
    print("Checking LocalStack Services")
    print("=" * 80)
    
    try:
        import requests
        response = requests.get(f"{ENDPOINT_URL}/_localstack/health")
        if response.status_code == 200:
            health = response.json()
            services = health.get('services', {})
            
            print("\nAvailable services:")
            for service, status in sorted(services.items()):
                status_icon = "✓" if status == "running" else "✗"
                print(f"  {status_icon} {service}: {status}")
            
            # Check specific services
            print("\n" + "-" * 80)
            print("Key Services for Chapter 2:")
            print(f"  {'✓' if services.get('events') == 'running' else '✗'} events (EventBridge): {services.get('events', 'not available')}")
            print(f"  {'✓' if services.get('stepfunctions') == 'running' else '✗'} stepfunctions: {services.get('stepfunctions', 'not available')}")
            print(f"  {'✓' if services.get('lambda') == 'running' else '✗'} lambda: {services.get('lambda', 'not available')}")
            print(f"  {'✓' if services.get('iam') == 'running' else '✗'} iam: {services.get('iam', 'not available')}")
            print(f"  {'✓' if services.get('cloudformation') == 'running' else '✗'} cloudformation: {services.get('cloudformation', 'not available')}")
            
        else:
            print(f"Could not check LocalStack health: {response.status_code}")
    except ImportError:
        print("requests library not available, skipping health check")
    except Exception as e:
        print(f"Error checking LocalStack health: {e}")

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LocalStack Service Availability Test")
    print("=" * 80)
    
    # Check LocalStack health first
    check_localstack_services()
    
    # Test EventBridge
    eventbridge_available = test_eventbridge()
    
    # Test Step Functions
    stepfunctions_available = test_stepfunctions()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"EventBridge:   {'✓ AVAILABLE' if eventbridge_available else '✗ NOT AVAILABLE'}")
    print(f"Step Functions: {'✓ AVAILABLE' if stepfunctions_available else '✗ NOT AVAILABLE'}")
    
    if eventbridge_available and stepfunctions_available:
        print("\n✓ Both services are available! You can use the full architecture.")
    elif stepfunctions_available:
        print("\n⚠ Step Functions is available, but EventBridge is not.")
        print("  You can still use Step Functions manually, but scheduled execution won't work.")
    else:
        print("\n✗ One or both services are not available.")
        print("  Consider using the Python workflow simulation (run_workflow.py) instead.")
    
    print("=" * 80)


