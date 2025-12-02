# MLflow on AWS

AWS SAM CloudFormation stack for deploying MLflow Tracking Server infrastructure for SageMaker Studio.

## What This Does

Deploys IAM roles and policies needed for MLflow Tracking Server in SageMaker Studio:
- **MLflowTrackingServerRole** - IAM role for the MLflow tracking server
- **SageMakerMLflowPolicy** - IAM policy attached to SageMaker execution role (optional)

## Architecture

**Backend Store:** PostgreSQL (stores experiments, runs, metrics, parameters)  
**Artifact Store:** S3/MinIO (stores model artifacts, files)

For local testing:
- **PostgreSQL** (port 5433) - MLflow backend database
- **MinIO** (ports 9000/9001) - S3-compatible artifact storage
- **MLflow Tracking Server** (port 5001) - MLflow UI

## Quick Start

### Local Testing

```bash
# Start services (PostgreSQL, MinIO, MLflow, LocalStack)
make start

# Deploy CloudFormation stack to LocalStack
make deploy-stack

# Train model with SageMaker Local Mode
make train
```

**Access:**
- MLflow UI: http://localhost:5001
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

### Real AWS Deployment

```bash
# Set in .env file
SAGEMAKER_EXECUTION_ROLE=YourSageMakerRole-Name
SAGEMAKER_DEFAULT_BUCKET=your-sagemaker-bucket-name
MLFLOW_ARTIFACTS_PREFIX=mlflow

# Deploy to AWS
AWS_DEPLOY=true make deploy-stack
```

## Commands

```bash
make start          # Start all services
make stop           # Stop all services
make train          # Train model with SageMaker Local Mode + MLflow
make deploy-stack   # Deploy stack (LocalStack if running)
make clean-all      # Clean all resources
```

## Configuration

For LocalStack testing, leave these empty in `.env`:
```bash
SAGEMAKER_EXECUTION_ROLE=
SAGEMAKER_DEFAULT_BUCKET=
MLFLOW_ARTIFACTS_PREFIX=mlflow
```

For real AWS, set your actual values.
