# Chapter 1: Credit Scoring - Local Implementation

Run the credit scoring architecture locally using LocalStack with **CloudFormation** and a trained **WOE + Logistic Regression** scorecard.

## Architecture

```
Loan Application → Kinesis Stream → Lambda (Scorecard) → DynamoDB (Approved Applications)
```

## Quick Start

```bash
# 1. Start LocalStack (required for S3)
make start

# 2. Train the scorecard model with SageMaker (saves to S3)
make train-sagemaker          # SageMaker Local Mode - saves model to S3
# OR for quick testing without SageMaker:
make train-scorecard          # Direct training (saves locally only)

# 3. Test inference locally
make test-inference

# 4. Deploy CloudFormation stack (includes Lambda with model loading from S3)
make deploy-stack

# 5. Send test loan applications
make send-applications

# 6. Check approved applications
make check-approvals

# 7. Stop LocalStack
make stop
```

## Training Options

### Option 1: Direct Training (Recommended)
- **Command**: `make train-scorecard`
- **Requirements**: Python only (no Docker)
- **Speed**: Fast (~5 seconds)
- **Use case**: Local development, quick iteration

### Option 2: SageMaker Local Mode (Recommended for Lambda)
- **Command**: `make train-sagemaker`
- **Requirements**: Docker + LocalStack running
- **Speed**: Slower (builds Docker image first time, ~2-3 minutes)
- **Use case**: Training with SageMaker and saving model to S3 for Lambda to use
- **Note**: 
  - Uses custom Docker container with Python 3.11 and modern dependencies
  - Container includes: pandas 2.x, scikit-learn 1.3+, fastwoe, sagemaker-training
  - Runs training in SageMaker-compatible container locally
  - Saves model artifacts to LocalStack S3 (`s3://credit-scoring-models/models/`)
  - Lambda function loads model from S3 for inference
  - Docker image is built automatically on first run (or use `make build-docker`)

## Model Details

### Scorecard Training

The credit scorecard uses:
- **Algorithm**: Weight of Evidence (WOE) + Logistic Regression
- **Library**: `fastwoe` for WOE transformation
- **Data**: 20,000 credit applications from `../data/credit_example.csv`
- **Target**: Binary (0=good, 1=default)
- **Performance**: Gini ~0.63

### Features

1. **Mortgage** - Mortgage amount
2. **Balance** - Current balance
3. **Amount Past Due** - Past due amount
4. **Delinquency** - Number of delinquencies
5. **Inquiry** - Credit inquiries
6. **Open Trade** - Open trades
7. **Utilization** - Credit utilization ratio
8. **Demographic** - Demographic indicator

### Scorecard Formula

```
Score = offset + factor × (-logit)

Where:
- factor = 20 / ln(2) = 28.85
- offset = 600 - factor × ln(30) = 501.86
- logit = β₀ + β₁×WOE₁ + β₂×WOE₂ + ...
```

**Note**: We reverse the sign because the model predicts default probability, but the scorecard should give higher scores to creditworthy applicants.

### Decision Rule

- **Score ≥ 502**: ✅ **APPROVED**
- **Score < 502**: ❌ **DECLINED**

The cutoff of 502 is optimally determined to maximize the separation between good and bad applicants.

## Infrastructure (via CloudFormation)

- **S3 Bucket**: `credit-scoring-models` (stores trained model artifacts)
- **Kinesis Stream**: `LoanApplicationsStream` (1 shard, 24h retention)
- **Lambda Function**: `CreditScoringLambda` (loads model from S3, applies scorecard)
- **DynamoDB Table**: `ApprovedApplications` (stores approved loans)
- **IAM Role**: `CreditScoringLambdaRole` (S3, DynamoDB, Kinesis permissions)
- **Event Source Mapping**: Kinesis → Lambda (batch size: 10)

### Lambda Function Details

The Lambda function:
1. **Loads model from S3**: Downloads `model.tar.gz` from `s3://credit-scoring-models/models/`
2. **Caches model**: Model is cached in memory after first load
3. **Scores applications**: Uses trained WOE + Logistic Regression scorecard
4. **Makes decisions**: Approves if score >= cutoff (typically 502)
5. **Stores results**: Saves approved applications to DynamoDB

## Files

### Directory Structure
```
local/
├── src/                          # Source code
│   └── scorecard/                # Scorecard inference code
│       ├── inference_scorecard.py  # Scorecard inference class
│       └── lambda_function.py      # Lambda handler
├── scripts/                      # Utility and deployment scripts
│   ├── local_train.py            # Train WOE + LR scorecard (direct)
│   ├── sagemaker_train.py        # SageMaker Local Mode training
│   ├── deploy_stack.py           # Deploy/delete/status commands
│   ├── package_lambda.py         # Package Lambda function
│   ├── send_applications.py      # Generate and send test applications
│   ├── check_approvals.py        # Query approved applications
│   ├── plot_score_distribution.py # Generate score distribution plot
│   └── verify_setup.py           # Verify end-to-end setup
├── config/                       # Configuration files
│   ├── localstack-config.toml    # LocalStack configuration
│   ├── requirements.txt          # Python dependencies
│   └── requirements_lambda.txt   # Lambda dependencies
├── cloudformation/               # Infrastructure as code
│   └── infrastructure.yml        # CloudFormation template
├── training/                     # SageMaker training code
│   ├── train.py                  # Training entry point
│   └── requirements.txt          # Training dependencies
├── model_output/                 # Trained model artifacts (generated)
│   ├── scorecard_pipeline.joblib # Full pipeline (WOE + LR)
│   ├── scorecard.csv             # Scorecard with points per WOE unit
│   ├── scorecard_metadata.json   # Metadata (cutoff, gini, etc.)
│   └── threshold_analysis.csv    # Threshold optimization results
├── docker-compose.yml            # LocalStack configuration
├── Dockerfile                    # Custom SageMaker training container
├── Makefile                      # Build automation
└── README.md                     # This file
```

### Model Training & Inference
- `scripts/local_train.py` - Train WOE + LR scorecard, find optimal cutoff
- `src/scorecard/inference_scorecard.py` - Load model and score new applications
- `model_output/` - Trained model artifacts (generated after training)

### Infrastructure
- `cloudformation/infrastructure.yml` - CloudFormation template
- `scripts/deploy_stack.py` - Deploy/delete/status commands
- `docker-compose.yml` - LocalStack configuration
- `Dockerfile` - Custom SageMaker training container (Python 3.11 + dependencies)
- `scripts/sagemaker_train.py` - SageMaker Local Mode training script

### Testing
- `scripts/send_applications.py` - Generate and send test applications
- `scripts/check_approvals.py` - Query approved applications from DynamoDB

## Application Data Format

Applications are sent as CSV strings with 8 features:

```
Mortgage,Balance,Amount Past Due,Delinquency,Inquiry,Open Trade,Utilization,Demographic
300000,500,0,0,0,0,0.2,1
```

## Example Workflow

```bash
# Train model
$ make train-scorecard
INFO: Gini score: 0.6295
INFO: Optimal cutoff: 502

# Test inference
$ make test-inference
✅ Good applicant - Score: 562 - APPROVED
❌ Risky applicant - Score: 427 - DECLINED

# Deploy infrastructure
$ make start
$ make deploy-stack

# Send applications
$ make send-applications
INFO: Sent 30 applications
INFO: Expected: ~12 approved, ~8 declined

# Check results
$ make check-approvals
📊 Found 12 approved application(s)
```

## Custom Docker Container

The SageMaker training uses a custom Docker container (`credit-scoring-sagemaker:latest`) that includes:

- **Base**: Python 3.11 (from AWS Public ECR)
- **Package Manager**: `uv` for fast dependency installation
- **Dependencies**: 
  - pandas >= 2.0.0
  - scikit-learn >= 1.3.0
  - fastwoe >= 0.1.5rc1
  - sagemaker-training < 5.0
- **Why Custom?**: The official SageMaker scikit-learn container (1.2-1) uses older package versions that conflict with `fastwoe`'s requirements. The custom container provides modern dependencies while maintaining SageMaker compatibility.

The Docker image is automatically built on first run. To rebuild manually:

```bash
make build-docker
```

To remove the Docker image:

```bash
make clean-docker
```

## Cleanup

```bash
# Stop LocalStack and clean up
make stop
make clean

# Optionally remove Docker image
make clean-docker
```

## Configuration

### LocalStack Lambda Size Limits

The Lambda package is large (~100MB zipped, ~250-300MB unzipped) due to ML dependencies. To allow this in LocalStack, we've configured:

1. **Environment variables** in `docker-compose.yml`:
   - `LAMBDA_LIMITS_CODE_SIZE_ZIPPED=157286400` (150MB)
   - `LAMBDA_LIMITS_MAX_FUNCTION_PAYLOAD_SIZE_BYTES=524288000` (500MB)

2. **Config file** `config/localstack-config.toml` (mounted to `/etc/localstack/localstack-config.toml`)

Both are needed for LocalStack to respect the increased limits.

## Notes

- **Lambda Package Size**: The Lambda function includes pandas, numpy, scikit-learn, and fastwoe, which creates a **very large package**:
  - **Compressed (zip)**: ~100 MB
  - **Uncompressed**: ~250-300 MB (exceeds LocalStack's 250MB limit!)
  
  **Why so large?**
  - scikit-learn: ~120 MB (compiled C/C++ code)
  - pandas: ~65 MB (C extensions, compiled code)
  - numpy: ~41 MB (BLAS/LAPACK linear algebra libraries)
  - numba + llvmlite: ~70-90 MB (LLVM JIT compiler for fastwoe)
  - Plus dependencies: ~10 MB
  
  This is a **well-known challenge** with ML packages in Lambda. For production, consider:
  - Using AWS Lambda Layers for large dependencies (separate from function code)
  - Using Lambda Container Images (up to 10GB limit)
  - Using lighter alternatives (e.g., onnxruntime for inference)
- The Lambda function loads the trained model from S3 and applies the scorecard
- The cutoff (502) is based on optimal separation of goods/bads on training data
- Approval rate: ~55%, Bad rate in approved: ~24%

## Troubleshooting

### Lambda Package Too Large

If you encounter "Unzipped size must be smaller than 262144000 bytes" error:

1. **For Local Testing**: Use `make train-scorecard` and test inference locally with `make test-inference`
2. **For Production**: 
   - Use AWS Lambda Layers to separate dependencies
   - Use Lambda Container Images instead of zip packages
   - Optimize the package by removing unnecessary files
