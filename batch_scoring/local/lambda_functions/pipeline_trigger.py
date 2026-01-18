"""Lambda function to start SageMaker Pipeline execution.

Triggered by EventBridge on schedule.
"""

# Import from training package (packaged with Lambda)
from training import execute_pipeline_workflow


def lambda_handler(event, context):
    """Start SageMaker Pipeline execution from EventBridge trigger."""
    print("Starting SageMaker Pipeline from EventBridge trigger")

    try:
        result = execute_pipeline_workflow()
        return {
            "statusCode": 200,
            "pipeline_name": result.get("pipeline_name"),
            "execution_id": result.get("execution_id"),
            "status": result.get("status"),
            "message": "Pipeline execution completed successfully",
        }
    except Exception as e:
        print(f"Error executing pipeline: {e}")
        import traceback

        traceback.print_exc()
        return {
            "statusCode": 500,
            "error": str(e),
            "message": "Pipeline execution failed",
        }
