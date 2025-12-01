# Chapter 2: Batch Scoring & Limit Increase - Local Implementation

Run the batch scoring and limit increase workflow locally using **PostgreSQL** and **SageMaker Pipelines** for ML workflow orchestration.

## Architecture

```
SageMaker Pipeline (Orchestration)
         ↓
    ┌────┴────┐
    ↓         ↓
Train Model  Query DB
    ↓         ↓
    └────┬────┘
         ↓
    PostgreSQL (Batch Scores)
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

# 2. Setup database and load sample batch scores
make setup-db

# 3. Run the complete workflow using SageMaker Pipeline
#    (This will train the model, query customers, run inference, evaluate, and update DB)
make run-workflow

# 4. Check results
make check-results

# 5. Stop services
make stop
```

**Note**: The pipeline includes model training as the first step. If you want to train separately:

```bash
# Optional: Train model separately (faster for iteration)
make train-catboost      # Direct training (fast, saves locally)
# OR
make train-sagemaker     # SageMaker Local Mode (saves to S3)
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
make test-services    # Test LocalStack services availability (optional)
make run-workflow     # Run complete SageMaker Pipeline workflow
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

- **ML-native integration**: Built specifically for ML workflows
- **Versioning**: Pipeline executions are tracked and versioned
- **Reproducibility**: Each run is logged with inputs/outputs
- **Local mode**: Can run locally for development/testing using LocalStack
- **Monitoring**: Built-in monitoring and logging
- **Processing Steps**: Each step runs in a containerized environment

### Pipeline Steps

The pipeline consists of 5 steps:

1. **TrainCatBoostModel** (Training Step): 
   - Trains CatBoost model using BankCaseStudyData.csv
   - Saves model artifacts to S3
   - Model is used by inference step

2. **QueryBatchScores** (Processing Step): 
   - Queries PostgreSQL for eligible customers
   - Outputs customer data to S3 for next step

3. **SageMakerInference** (Processing Step): 
   - Loads CatBoost model from training step output
   - Runs inference on eligible customers
   - Uses SHAP values for scoring
   - Outputs scored customers to S3

4. **EvaluateLimitIncrease** (Processing Step): 
   - Applies business rules for limit increases
   - Evaluates each customer's eligibility
   - Outputs evaluations to S3

5. **UpdateDatabase** (Processing Step): 
   - Updates PostgreSQL with limit increase decisions
   - Saves results summary to S3

**Note**: If you've already trained a model, you can skip the training step by modifying the pipeline or using a pre-trained model.

### Running the Pipeline

```bash
# Run the complete pipeline
make run-workflow
```

The pipeline will:
- Execute all steps in sequence with proper dependencies
- Handle data passing between steps via S3 (LocalStack)
- Track execution status and results
- Work in local mode for development (no AWS account needed)
- Use Docker containers for each processing step

## Pipeline Step Details

### Step 1: Train CatBoost Model
**Type**: Training Step

- Trains CatBoost model using `BankCaseStudyData.csv` from `../data/`
- Uses custom Docker container: `catboost-sagemaker:latest`
- Saves model artifacts to S3: `s3://credit-scoring-models/models/`
- Model includes metadata (feature names, categorical features, scoring parameters)

### Step 2: Query Batch Scores
**Script**: `scripts/query_batch_scores_processing.py`

- Queries PostgreSQL for customers eligible for limit increase
- **Criteria**: 
  - `current_score >= 600` 
  - `current_limit < 10000` 
  - `limit_increase_decision IS NULL`
- Returns up to 20 eligible customers per run
- Outputs customer data to S3: `s3://credit-scoring-models/pipeline/customers/`

### Step 3: SageMaker Inference
**Script**: `scripts/inference_processing.py`

- Loads CatBoost model from training step output (S3) or local `model_output/` directory
- Runs inference on eligible customers using the trained model
- Uses SHAP values for feature importance and scoring
- Converts log-odds to credit scores using PDO (Points to Double Odds) method
- **Features**: Uses the new dataset features from `BankCaseStudyData.csv`:
  - Numerical: Application_Score, Bureau_Score, Loan_Amount, Time_with_Bank, Time_in_Employment, Loan_to_income, Gross_Annual_Income
  - Categorical: Loan_Payment_Frequency, Residential_Status, Cheque_Card_Flag, Existing_Customer_Flag, Home_Telephone_Number
- Outputs scored customers to S3: `s3://credit-scoring-models/pipeline/scored_customers/`

### Step 4: Evaluate Limit Increase
**Script**: `scripts/evaluate_processing.py`

- Applies business rules for limit increases:
  - **New score >= 650**: Eligible for increase
  - **Current limit < $5,000**: Can increase by 50%
  - **Current limit >= $5,000**: Can increase by 25%
  - **Max new limit**: $20,000
- Generates decision reasons for each customer
- Outputs evaluations to S3: `s3://credit-scoring-models/pipeline/evaluations/`

### Step 5: Update Database
**Script**: `scripts/update_database_processing.py`

- Updates PostgreSQL with limit increase decisions
- Stores: `limit_increase_decision`, `new_limit`, `decision_reason`, `updated_at`
- Saves results summary to S3: `s3://credit-scoring-models/pipeline/results/`

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
- **Use case**: Training with SageMaker and saving model to S3 for pipeline use
- **Note**: 
  - Uses custom Docker container with Python 3.11 and CatBoost
  - Container includes: pandas 2.x, numpy, scikit-learn, catboost, sagemaker-training
  - Runs training in SageMaker-compatible container locally
  - Saves model artifacts to LocalStack S3 (`s3://credit-scoring-models/models/catboost_model.tar.gz`)
  - Pipeline processing steps can load model from S3 for inference
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

### Pipeline & Processing Scripts
- `sagemaker_pipeline.py` - Main SageMaker Pipeline definition and execution
- `scripts/query_batch_scores_processing.py` - Query database (Processing step)
- `scripts/inference_processing.py` - Run inference (Processing step)
- `scripts/evaluate_processing.py` - Evaluate decisions (Processing step)
- `scripts/update_database_processing.py` - Update database (Processing step)
- `scripts/check_results.py` - Query and display results
- `scripts/setup_database.py` - Create schema and load sample data

### Legacy Lambda Functions (for reference)
- `lambda_functions/query_batch_scores.py` - Query PostgreSQL (legacy)
- `lambda_functions/sagemaker_inference.py` - Inference (legacy)
- `lambda_functions/evaluate_limit_increase.py` - Business logic (legacy)
- `lambda_functions/update_database.py` - Update PostgreSQL (legacy)

**Note**: The current implementation uses SageMaker Processing Steps instead of Lambda functions for better ML workflow integration.

### Infrastructure
- `docker-compose.yml` - PostgreSQL + LocalStack configuration
- `cloudformation/infrastructure.yml` - CloudFormation template (optional, for Step Functions/EventBridge if needed)
- `deploy_stack.py` - Deploy CloudFormation stack (optional)

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

# Run workflow (SageMaker Pipeline)
$ make run-workflow
Starting SageMaker Pipeline execution...
Pipeline: BatchCreditScoringPipeline
Step 1/5: TrainCatBoostModel - Training model...
Step 2/5: QueryBatchScores - Querying eligible customers...
Step 3/5: SageMakerInference - Running CatBoost inference...
Step 4/5: EvaluateLimitIncrease - Evaluating decisions...
Step 5/5: UpdateDatabase - Updating PostgreSQL...
✓ Pipeline execution completed successfully

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

## SageMaker Pipeline Technical Details

The pipeline is defined in `sagemaker_pipeline.py` and uses:

- **Processing Steps**: Each step runs in a Docker containerized environment
- **Local Mode**: Works with LocalStack S3 for artifact storage (no AWS account needed)
- **Step Dependencies**: Steps execute in sequence with automatic data passing via S3
- **Error Handling**: Pipeline tracks execution status and failures
- **Container Images**: Uses `python:3.11-slim` for processing steps, `catboost-sagemaker:latest` for training

### Pipeline Execution Flow

```bash
# Run the complete pipeline
make run-workflow

# Execution flow:
# 1. Pipeline starts → Creates/updates pipeline definition
# 2. Step 1: TrainCatBoostModel → Trains model, saves to S3
# 3. Step 2: QueryBatchScores → Queries PostgreSQL, outputs to S3
# 4. Step 3: SageMakerInference → Loads model, scores customers, outputs to S3
# 5. Step 4: EvaluateLimitIncrease → Applies rules, outputs to S3
# 6. Step 5: UpdateDatabase → Updates PostgreSQL, saves summary to S3
# 7. Pipeline completes → All artifacts stored in S3
```

### Pipeline Outputs (S3 Locations)

Each step outputs data to LocalStack S3:
- `s3://credit-scoring-models/pipeline/customers/` - Eligible customers (JSON)
- `s3://credit-scoring-models/pipeline/scored_customers/` - Scored customers (JSON)
- `s3://credit-scoring-models/pipeline/evaluations/` - Evaluations (JSON)
- `s3://credit-scoring-models/pipeline/results/` - Final results summary (JSON)
- `s3://credit-scoring-models/models/` - Trained model artifacts

### Data Flow Between Steps

```
[Training Step] → S3 (Model)
     ↓
PostgreSQL → [Query Step] → S3 (Customers) → [Inference Step] → S3 (Scored) → 
[Evaluate Step] → S3 (Evaluations) → [Update Step] → PostgreSQL
```

Each step reads from the previous step's S3 output and writes its own output for the next step. The training step runs in parallel with the query step, and the inference step depends on both completing.

### Scheduling (Optional - Production)

For production deployment, you can schedule the pipeline using:
- **EventBridge Rules**: Trigger pipeline executions on a schedule
- **Step Functions**: Orchestrate pipeline + other AWS services
- **CloudWatch Events**: Cron-based scheduling
- **SageMaker Pipelines API**: Programmatic execution via API/CLI

## Key Differences from Chapter 1

| Aspect | Chapter 1 | Chapter 2 |
|--------|-----------|-----------|
| **Database** | DynamoDB (NoSQL) | PostgreSQL (SQL) |
| **Processing** | Real-time (Kinesis streams) | Batch (SageMaker Pipeline) |
| **Orchestration** | Event-driven (Kinesis → Lambda) | ML Pipeline orchestration (SageMaker Pipelines) |
| **Use Case** | New loan applications | Existing customer limit increases |
| **Decision** | Approve/Decline | Increase/No Change |
| **Model** | WOE + Logistic Regression | CatBoost with SHAP values |
| **ML Integration** | Lambda functions | SageMaker Processing Steps |
| **Workflow** | Event-driven streaming | Scheduled batch processing |
| **Data** | Real-time loan applications | Batch customer scores |

## Troubleshooting

### PostgreSQL Connection Error
```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker-compose logs postgres
```

### Model Not Found
- Make sure model is trained: `make train-catboost` or `make train-sagemaker`
- Check that model files exist in `model_output/` directory
- Verify model files are accessible from Docker containers

### Pipeline Execution Issues
- **LocalStack S3**: Make sure LocalStack is running (`make start`) and S3 is accessible
- **PostgreSQL Access**: Processing steps need to access PostgreSQL. Use `host.docker.internal` as hostname from Docker containers
- **Model Files**: Verify model files exist in `model_output/` directory or S3
- **Docker**: Check Docker is running (required for Processing steps)
- **Docker Image**: Ensure `catboost-sagemaker:latest` image is built (`make build-docker`)
- **Data Path**: Verify data file exists at `../data/BankCaseStudyData.csv`

## Cleanup

```bash
# Stop services and remove volumes
make clean
```

