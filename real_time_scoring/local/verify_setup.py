"""
Quick verification script to check if the end-to-end setup is ready
"""

import sys
from pathlib import Path


def check_localstack():
    """Check if LocalStack is running"""
    try:
        import requests

        response = requests.get("http://localhost:4566/_localstack/health", timeout=2)
        if response.status_code == 200:
            print("✓ LocalStack is running")
            return True
    except Exception as e:
        print(f"✗ LocalStack is not running: {e}")
        return False


def check_docker_image():
    """Check if Docker image exists"""
    import subprocess

    try:
        result = subprocess.run(
            ["docker", "images", "-q", "credit-scoring-sagemaker:latest"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.stdout.strip():
            print("✓ Docker image 'credit-scoring-sagemaker:latest' exists")
            return True
        else:
            print("✗ Docker image 'credit-scoring-sagemaker:latest' not found")
            print("  Run: make build-docker")
            return False
    except Exception as e:
        print(f"✗ Could not check Docker image: {e}")
        return False


def check_model_in_s3():
    """Check if model exists in S3"""
    try:
        import boto3

        s3 = boto3.client(
            "s3", endpoint_url="http://localhost:4566", region_name="us-east-1"
        )
        bucket = "credit-scoring-models"

        # Check if bucket exists
        try:
            s3.head_bucket(Bucket=bucket)
        except:
            print(f"✗ S3 bucket '{bucket}' does not exist")
            return False

        # Check for model files
        response = s3.list_objects_v2(Bucket=bucket, Prefix="models/")
        if "Contents" in response and len(response["Contents"]) > 0:
            print(f"✓ Found {len(response['Contents'])} model file(s) in S3")
            for obj in response["Contents"][:3]:
                print(f"  - {obj['Key']} ({obj['Size'] / 1024:.1f} KB)")
            return True
        else:
            print("✗ No model files found in S3")
            print("  Run: make train-sagemaker")
            return False
    except Exception as e:
        print(f"✗ Could not check S3: {e}")
        return False


def check_lambda_package():
    """Check if Lambda package exists"""
    package_path = Path(__file__).parent / "lambda_package.zip"
    if package_path.exists():
        size_mb = package_path.stat().st_size / 1024 / 1024
        print(f"✓ Lambda package exists ({size_mb:.2f} MB)")
        if size_mb > 150:
            print(f"  ⚠ Warning: Package is large ({size_mb:.2f} MB)")
            print("  Make sure LocalStack limits are increased in docker-compose.yml")
        return True
    else:
        print("✗ Lambda package not found")
        print("  Run: make package-lambda")
        return False


def check_stack():
    """Check if CloudFormation stack is deployed"""
    try:
        import boto3

        cfn = boto3.client(
            "cloudformation",
            endpoint_url="http://localhost:4566",
            region_name="us-east-1",
        )
        stack_name = "credit-scoring-stack"

        try:
            response = cfn.describe_stacks(StackName=stack_name)
            stack = response["Stacks"][0]
            status = stack["StackStatus"]
            print(f"✓ Stack '{stack_name}' exists (Status: {status})")
            return True
        except cfn.exceptions.ClientError as e:
            if "does not exist" in str(e):
                print(f"✗ Stack '{stack_name}' not deployed")
                print("  Run: make deploy-stack")
                return False
            else:
                raise
    except Exception as e:
        print(f"✗ Could not check stack: {e}")
        return False


def main():
    print("=" * 80)
    print("Real-Time Scoring Setup Verification")
    print("=" * 80)
    print()

    checks = [
        ("LocalStack", check_localstack),
        ("Docker Image", check_docker_image),
        ("Model in S3", check_model_in_s3),
        ("Lambda Package", check_lambda_package),
        ("CloudFormation Stack", check_stack),
    ]

    results = []
    for name, check_func in checks:
        print(f"\n[{name}]")
        try:
            results.append(check_func())
        except Exception as e:
            print(f"  Error: {e}")
            results.append(False)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    all_passed = all(results)
    if all_passed:
        print("✓ All checks passed! Ready to run end-to-end workflow.")
        print("\nNext steps:")
        print("  1. make send-applications  # Send test applications")
        print("  2. make check-approvals    # Check results")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print("\nTypical workflow:")
        print("  1. make start              # Start LocalStack")
        print("  2. make train-sagemaker    # Train model (saves to S3)")
        print("  3. make deploy-stack       # Deploy Lambda + infrastructure")
        print("  4. make send-applications  # Send test applications")
        print("  5. make check-approvals    # Check results")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
    sys.exit(main())
