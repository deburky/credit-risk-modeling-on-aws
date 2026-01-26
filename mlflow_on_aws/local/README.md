# MLflow on AWS

AWS SAM CloudFormation stack for deploying MLflow Tracking Server infrastructure for SageMaker Studio.

## What This Does

Deploys IAM roles and policies needed for MLflow Tracking Server in SageMaker Studio:
- **MLflowTrackingServerRole** - IAM role for the MLflow tracking server
- **SageMakerMLflowPolicy** - IAM policy attached to SageMaker execution role (optional)

## Architecture

**Backend Store:** PostgreSQL (stores experiments, runs, metrics, parameters)  
**Artifact Store:** S3/LocalStack (stores model artifacts, files)  
**MLflow Version:** 3.0.0+ (uses logged-models API for better model tracking)

For local testing:
- **PostgreSQL** (port 5433) - MLflow backend database
- **LocalStack S3** (port 4566) - S3-compatible artifact storage
- **MLflow Tracking Server** (port 5001) - MLflow UI with logged-models API support

## Quick Start

### Local Testing

```bash
# Navigate to local directory
cd local

# Start services (PostgreSQL, LocalStack, MLflow)
make start

# Deploy CloudFormation stack to LocalStack (from root)
cd ..
make deploy-stack

# Train model with SageMaker Local Mode (from local/)
cd local
make train
```

**Access:**
- MLflow UI: http://localhost:5001
- LocalStack: http://localhost:4566

### Real AWS Deployment

```bash
# Set in .env file (at root)
SAGEMAKER_EXECUTION_ROLE=YourSageMakerRole-Name
SAGEMAKER_DEFAULT_BUCKET=your-sagemaker-bucket-name
MLFLOW_ARTIFACTS_PREFIX=mlflow

# Deploy to AWS (from root)
AWS_DEPLOY=true make deploy-stack
```

## Directory Structure

```
mlflow_on_aws/
├── local/                    # Local development
│   ├── docker-compose.yml   # Local services
│   ├── Dockerfile.mlflow    # MLflow server image
│   ├── Makefile             # Local commands
│   ├── scripts/             # Docker management scripts
│   ├── training/            # Training code
│   └── train_sagemaker.py   # SageMaker training script
├── template.yaml            # SAM template
├── samconfig.toml          # SAM configuration
├── iam-policies/           # IAM policy files
└── Makefile                # SAM deployment commands
```

## Features

- **MLflow 3.0+** with logged-models API support
- **Model logging** using `mlflow.sklearn.log_model()` (logged-models API)
- **SageMaker-ready** models packaged as `tar.gz` format
- **LocalStack S3** for local artifact storage
- **PostgreSQL** backend for metadata storage

## Commands

### Local Development (from `local/` directory)

```bash
cd local
make start          # Start all services (PostgreSQL, LocalStack, MLflow)
make stop           # Stop all services
make restart        # Restart all services
make status         # Check service status
make train          # Train model with SageMaker Local Mode + MLflow
make clean-mlflow   # Clean MLflow data (PostgreSQL, LocalStack S3)
make clean-all      # Clean all resources (services, data, images)
```

### SAM Deployment (from root directory)

```bash
make build          # Build SAM application
make validate       # Validate SAM template
make deploy-stack   # Deploy stack (LocalStack if running)
make delete-stack   # Delete stack
make describe-stack # Show stack status
```

## Model Logging

This setup uses **MLflow 3.0+** with the logged-models API:
- Models are logged using `mlflow.sklearn.log_model()` (logged-models API)
- Models are stored in S3 with MLflow 3.0 structure: `experiments/<exp_id>/models/<model_id>/artifacts/`
- Models are also packaged as `model.tar.gz` for SageMaker deployment
- All artifacts are accessible via MLflow UI and directly from S3

## Configuration

For LocalStack testing, the default dummy role is used. You can leave these empty in `.env` (at root):

```bash
SAGEMAKER_EXECUTION_ROLE=
SAGEMAKER_DEFAULT_BUCKET=
MLFLOW_ARTIFACTS_PREFIX=mlflow
```

Or explicitly set the dummy role name:

```bash
SAGEMAKER_EXECUTION_ROLE=AmazonSageMaker-ExecutionRole-20200101T000001
```

For real AWS, set your actual SageMaker execution role name in `.env`:

```bash
SAGEMAKER_EXECUTION_ROLE=YourActualSageMakerRoleName
SAGEMAKER_DEFAULT_BUCKET=your-sagemaker-bucket
MLFLOW_ARTIFACTS_PREFIX=mlflow
```

## Technical Details

- **MLflow Server:** v3.0.0 with boto3 for S3 artifact storage
- **Training Container:** MLflow 3.0+ with logged-models API support
- **Artifact Storage:** LocalStack S3 (local) or AWS S3 (production)
- **Model Format:** MLflow logged models + SageMaker tar.gz packages
