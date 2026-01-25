# MLflow on AWS

AWS SAM CloudFormation stack for deploying MLflow Tracking Server infrastructure for SageMaker Studio.

## What This Does

Deploys IAM roles and policies needed for MLflow Tracking Server in SageMaker Studio:
- **MLflowTrackingServerRole** - IAM role for the MLflow tracking server
- **SageMakerMLflowPolicy** - IAM policy attached to SageMaker execution role (optional)

## Architecture

**Backend Store:** PostgreSQL (stores experiments, runs, metrics, parameters)  
**Artifact Store:** S3/LocalStack (stores model artifacts, files)

For local testing:
- **PostgreSQL** (port 5433) - MLflow backend database
- **LocalStack S3** (port 4566) - S3-compatible artifact storage
- **MLflow Tracking Server** (port 5001) - MLflow UI

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

## Commands

### Local Development (from `local/` directory)

```bash
cd local
make start          # Start all services
make stop           # Stop all services
make status         # Check service status
make train          # Train model with SageMaker Local Mode + MLflow
make clean-mlflow   # Clean MLflow data
make clean-all      # Clean all resources
```

### SAM Deployment (from root directory)

```bash
make build          # Build SAM application
make validate       # Validate SAM template
make deploy-stack   # Deploy stack (LocalStack if running)
make delete-stack   # Delete stack
make describe-stack # Show stack status
```

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
