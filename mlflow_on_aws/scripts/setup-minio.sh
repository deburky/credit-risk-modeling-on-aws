#!/bin/bash
# Setup MinIO bucket for MLflow artifacts
# Automatically creates bucket using boto3 (no console needed)

set -e

MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ROOT_USER="${MINIO_ROOT_USER:-minioadmin}"
MINIO_ROOT_PASSWORD="${MINIO_ROOT_PASSWORD:-minioadmin}"
MLFLOW_BUCKET="${MLFLOW_BUCKET:-mlflow}"

echo "🔧 Setting up MinIO for MLflow..."
echo "   Endpoint: $MINIO_ENDPOINT"
echo "   Bucket: $MLFLOW_BUCKET"

# Wait for MinIO to be ready
echo "⏳ Waiting for MinIO to be ready..."
until curl -f -s "$MINIO_ENDPOINT/minio/health/live" > /dev/null 2>&1; do
    echo "   Waiting for MinIO..."
    sleep 2
done
echo "✅ MinIO is ready"

# Create bucket using boto3 (MLflow dependency)
echo "📦 Creating bucket: $MLFLOW_BUCKET"
python3 << EOF
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

s3 = boto3.client(
    's3',
    endpoint_url='$MINIO_ENDPOINT',
    aws_access_key_id='$MINIO_ROOT_USER',
    aws_secret_access_key='$MINIO_ROOT_PASSWORD',
    config=Config(signature_version='s3v4')
)

try:
    s3.head_bucket(Bucket='$MLFLOW_BUCKET')
    print("✅ Bucket '$MLFLOW_BUCKET' already exists")
except ClientError as e:
    error_code = e.response['Error']['Code']
    if error_code == '404':
        s3.create_bucket(Bucket='$MLFLOW_BUCKET')
        print("✅ Created bucket: $MLFLOW_BUCKET")
    else:
        print(f"⚠️  Error: {e}")
        exit(1)
EOF

echo ""
echo "✅ MinIO setup complete!"
echo ""
echo "💡 To use with MLflow, set these environment variables:"
echo "   export AWS_ACCESS_KEY_ID=$MINIO_ROOT_USER"
echo "   export AWS_SECRET_ACCESS_KEY=$MINIO_ROOT_PASSWORD"
echo "   export MLFLOW_S3_ENDPOINT_URL=$MINIO_ENDPOINT"
echo "   export MLFLOW_S3_IGNORE_TLS=true"
