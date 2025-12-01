# Chapter 2: Batch Scoring & Limit Increase - Local Implementation

Run the batch scoring and limit increase workflow locally using **PostgreSQL** and **SageMaker Pipelines** for ML workflow orchestration.

## Architecture

```
SageMaker Pipeline (Orchestration)
         ↓
    ┌────┴────┐
    ↓         ↓
Query DB   PostgreSQL
    ↓
SageMaker Inference (Processing Step)
    ↓
Evaluate Decisions (Processing Step)
    ↓
Update Database (Processing Step)
    ↓
PostgreSQL (Updated Decisions)
```

**Note**: The workflow is now orchestrated using **SageMaker Pipelines** with Processing Steps, providing better ML workflow integration, versioning, and tracking.

## Quick Start

```bash
# 1. Start PostgreSQL and LocalStack
make start

# 2. Train CatBoost model (choose one):
make train-catboost      # Direct training (fast, saves locally)
# OR
make train-sagemaker     # SageMaker Local Mode (saves to S3 for Lambda)

# 3. Setup database and load sample batch scores
make setup-db

# 4. Run the workflow using SageMaker Pipeline
make run-workflow

# 6. Check results
make check-results

# 7. Stop services
make stop
```

## Testing

For comprehensive testing instructions, see [TESTING.md](TESTING.md).

**Quick Test:**
```bash
# Run all tests in sequence
make test-all
```

**Individual Tests:**
```bash
make test-services    # Test LocalStack services availability
make test-schedule    # Test EventBridge scheduled rule
make run-workflow     # Run complete workflow
make check-results    # Check results in database
```

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- `uv` package manager (or use `pip`)

## Installation

```bash
# Install dependencies
uv pip install -r requirements.txt

# Or with pip
pip install -r requirements.txt
```

## Workflow Orchestration

The workflow is orchestrated using **SageMaker Pipelines**, which provides:

- **ML-native integration**: Built for ML workflows
- **Versioning**: Pipeline executions are tracked and versioned
- **Reproducibility**: Each run is logged with inputs/outputs
- **Local mode**: Can run locally for development/testing
- **Monitoring**: Built-in monitoring and logging

### Pipeline Steps

1. **QueryBatchScores** (Processing Step): Queries PostgreSQL for eligible customers
2. **SageMakerInference** (Processing Step): Runs CatBoost model inference
3. **EvaluateLimitIncrease** (Processing Step): Applies business rules
4. **UpdateDatabase** (Processing Step): Updates PostgreSQL with decisions

### Running the Pipeline

```bash
# Run the complete pipeline
make run-workflow
```

The pipeline will:
- Execute all steps in sequence
- Handle data passing between steps via S3
- Track execution status and results
- Work in local mode for development

## Pipeline Steps

### 1. Query Batch Scores (Processing Step)
- Queries PostgreSQL for customers eligible for limit increase
- Criteria: `current_score >= 600` AND `current_limit < 10000` AND `limit_increase_decision IS NULL`
- Returns up to 20 eligible customers
- Outputs to S3 for next step

### 2. SageMaker Inference (Processing Step)
- Loads CatBoost model from local model directory
- Runs inference on eligible customers
- Uses SHAP values for feature importance and scoring
- Converts log-odds to credit scores using PDO method
- Outputs scored customers to S3

### 3. Evaluate Limit Increase (Processing Step)
- Applies business rules:
  - **New score >= 650**: Eligible for increase
  - **Current limit < $5,000**: Can increase by 50%
  - **Current limit >= $5,000**: Can increase by 25%
  - **Max new limit**: $20,000
- Outputs evaluations to S3

### 4. Update Database (Processing Step)
- Updates PostgreSQL with limit increase decisions
- Stores: `limit_increase_decision`, `new_limit`, `decision_reason`, `updated_at`
- Saves results summary to S3

## Database Schema

```sql
CREATE TABLE batch_scores (
    customer_id VARCHAR(50) PRIMARY KEY,
    current_score INTEGER NOT NULL,
    current_limit DECIMAL(10, 2) NOT NULL,
    application_date DATE NOT NULL,
    limit_increase_decision VARCHAR(20),
    new_limit DECIMAL(10, 2),
    decision_reason TEXT,
    updated_at TIMESTAMP
);
```

## Sample Data

The `setup-db` command loads 50 sample customers with:
- Scores between 550-750
- Limits between $2,000-$12,000
- Application dates from the last 6 months

## Training Options

### Option 1: Direct Training (Recommended for Development)
- **Command**: `make train-catboost`
- **Requirements**: Python only (no Docker)
- **Speed**: Fast (~30-60 seconds)
- **Use case**: Local development, quick iteration
- **Output**: Saves model to `model_output/` locally

### Option 2: SageMaker Local Mode (Recommended for Production-like Testing)
- **Command**: `make train-sagemaker`
- **Requirements**: Docker + LocalStack running
- **Speed**: Slower (builds Docker image first time, ~2-3 minutes)
- **Use case**: Training with SageMaker and saving model to S3 for Lambda to use
- **Note**: 
  - Uses custom Docker container with Python 3.11 and CatBoost
  - Container includes: pandas 2.x, numpy, scikit-learn, catboost, sagemaker-training
  - Runs training in SageMaker-compatible container locally
  - Saves model artifacts to LocalStack S3 (`s3://credit-scoring-models/models/catboost_model.tar.gz`)
  - Lambda functions can load model from S3 for inference
  - Docker image is built automatically on first run (or use `make build-docker`)

## Custom Docker Container

The SageMaker training uses a custom Docker container (`catboost-sagemaker:latest`) that includes:

- **Base**: Python 3.11 (from AWS Public ECR)
- **Package Manager**: `uv` for fast dependency installation
- **Dependencies**: 
  - pandas >= 2.0.0
  - numpy >= 1.24.0
  - scikit-learn >= 1.3.0
  - catboost >= 1.2.0
  - sagemaker-training < 5.0
- **Why Custom?**: Provides modern dependencies and CatBoost support while maintaining SageMaker compatibility.

The Docker image is automatically built on first run. To rebuild manually:

```bash
make build-docker
```

To remove the Docker image:

```bash
make clean-docker
```

## Files

### Model Training
- `train_catboost.py` - Direct CatBoost training (fast, local)
- `sagemaker_train.py` - SageMaker Local Mode training (saves to S3)
- `training/train.py` - Training script that runs inside SageMaker container
- `Dockerfile` - Custom SageMaker training container for CatBoost
- `model_output/` - Trained model artifacts

### Lambda Functions
- `lambda_functions/query_batch_scores.py` - Query PostgreSQL
- `lambda_functions/sagemaker_inference.py` - Call SageMaker endpoint
- `lambda_functions/evaluate_limit_increase.py` - Business logic
- `lambda_functions/update_database.py` - Update PostgreSQL

### Scripts
- `scripts/setup_database.py` - Create schema and load sample data
- `scripts/query_batch_scores_processing.py` - Query database (Processing step)
- `scripts/inference_processing.py` - Run inference (Processing step)
- `scripts/evaluate_processing.py` - Evaluate decisions (Processing step)
- `scripts/update_database_processing.py` - Update database (Processing step)
- `scripts/check_results.py` - Query and display results
- `sagemaker_pipeline.py` - Main pipeline definition and execution

### Infrastructure
- `cloudformation/infrastructure.yml` - CloudFormation template (Step Functions + Lambda + EventBridge)
- `docker-compose.yml` - PostgreSQL + LocalStack configuration
- `deploy_stack.py` - Deploy CloudFormation stack

### Scheduling
- **EventBridge Rule**: `LimitIncreaseDailySchedule` - Triggers Step Functions daily at 2 AM UTC
- **IAM Roles**: 
  - `EventBridgeStepFunctionsInvokeRole` - Allows EventBridge to invoke Step Functions
  - `LimitIncreaseStepFunctionsRole` - Allows Step Functions to invoke Lambda functions

## Example Workflow

```bash
# Setup
$ make start
✓ PostgreSQL running at localhost:5432
✓ LocalStack running at http://localhost:4566

$ make setup-db
✓ Created batch_scores table
✓ Loaded 50 sample customers
  Total customers: 50
  Score >= 600: 35
  Limit < 10000: 42

# Run workflow
$ make run-workflow
[Step 1] Querying batch scores from PostgreSQL...
✓ Found 20 eligible customers

[Step 2] Running SageMaker inference...
✓ Scored 20 customers

[Step 3] Evaluating limit increase eligibility...
✓ Evaluated 20 customers
  ✅ Approved: 12
  ❌ Declined: 8

[Step 4] Updating PostgreSQL...
✓ Updated 20 customer records

# Check results
$ make check-results
Total customers: 50
  ✅ Approved: 12
  ❌ Declined: 8
  ⏳ Pending: 30

Top Approved Limit Increases:
  CUST_0001: $8,500 → $10,625 (+25%)
  CUST_0002: $4,200 → $6,300 (+50%)
  ...
```

## SageMaker Pipeline Details

The pipeline is defined in `sagemaker_pipeline.py` and uses:

- **Processing Steps**: Each step runs in a containerized environment
- **Local Mode**: Works with LocalStack S3 for artifact storage
- **Step Dependencies**: Steps execute in sequence with data passing via S3
- **Error Handling**: Pipeline tracks execution status and failures

### Pipeline Execution

```bash
# Run the pipeline
make run-workflow

# The pipeline will:
# 1. Query eligible customers from PostgreSQL
# 2. Run inference using CatBoost model
# 3. Evaluate limit increase decisions
# 4. Update PostgreSQL with results
```

### Pipeline Outputs

Each step outputs data to S3 (LocalStack):
- `s3://credit-scoring-models/pipeline/customers/` - Eligible customers
- `s3://credit-scoring-models/pipeline/scored_customers/` - Scored customers
- `s3://credit-scoring-models/pipeline/evaluations/` - Evaluations
- `s3://credit-scoring-models/pipeline/results/` - Final results

### Scheduling (Optional)

For production, you can schedule the pipeline using:
- **EventBridge** to trigger pipeline executions
- **Step Functions** to orchestrate pipeline + other services
- **CloudWatch Events** for cron-based scheduling

## Key Differences from Chapter 1

| Aspect | Chapter 1 | Chapter 2 |
|--------|-----------|-----------|
| **Database** | DynamoDB (NoSQL) | PostgreSQL (SQL) |
| **Processing** | Real-time (Kinesis) | Batch (SageMaker Pipeline) |
| **Orchestration** | Event-driven | ML Pipeline orchestration |
| **Use Case** | New loan applications | Existing customer limit increases |
| **Decision** | Approve/Decline | Increase/No Change |
| **ML Integration** | Lambda functions | SageMaker Processing Steps |

## Troubleshooting

### PostgreSQL Connection Error
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker-compose logs postgres
```

### Lambda Function Errors
- Make sure model files from Chapter 1 are available
- Check that `MODEL_DIR` environment variable points to correct path
- For local testing, Lambda functions use the model directly (not SageMaker endpoint)

### Pipeline Execution Issues
- Make sure LocalStack S3 is running and accessible
- Check that PostgreSQL is accessible from Docker containers (use `host.docker.internal`)
- Verify model files exist in `model_output/` directory
- Check Docker is running (required for Processing steps)

## Cleanup

```bash
# Stop services and remove volumes
make clean
```

