# Chapter 1: Credit Scoring - Local Implementation

Run the credit scoring architecture locally using LocalStack with **CloudFormation** and a trained **WOE + Logistic Regression** scorecard.

## Architecture

```
Loan Application → Kinesis Stream → Lambda (Scorecard) → DynamoDB (Approved Applications)
```

## Quick Start

```bash
# 1. Train the scorecard model
make train-scorecard          # Direct training (recommended)
# OR
make train-sagemaker          # SageMaker Local Mode (requires Docker)

# 2. Test inference
make test-inference

# 3. Start LocalStack
make start

# 4. Deploy CloudFormation stack
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

### Option 2: SageMaker Local Mode
- **Command**: `make train-sagemaker`
- **Requirements**: Docker + AWS credentials
- **Speed**: Slower (pulls Docker image first time)
- **Use case**: Testing SageMaker workflows before deploying to AWS
- **Note**: Runs training in SageMaker sklearn container locally

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

- **Kinesis Stream**: `LoanApplicationsStream` (1 shard, 24h retention)
- **Lambda Function**: `CreditScoringLambda` (applies scorecard cutoff=502)
- **DynamoDB Table**: `ApprovedApplications` (stores approved loans)
- **IAM Role**: `CreditScoringLambdaRole` (permissions)
- **Event Source Mapping**: Kinesis → Lambda (batch size: 10)

## Files

### Model Training & Inference
- `train_scorecard.py` - Train WOE + LR scorecard, find optimal cutoff
- `inference_scorecard.py` - Load model and score new applications
- `model_output/` - Trained model artifacts
  - `scorecard_pipeline.joblib` - Full pipeline (WOE + LR)
  - `scorecard.csv` - Scorecard with points per WOE unit
  - `scorecard_metadata.json` - Metadata (cutoff, gini, etc.)
  - `threshold_analysis.csv` - Threshold optimization results

### Infrastructure
- `cloudformation/infrastructure.yml` - CloudFormation template
- `deploy_stack.py` - Deploy/delete/status commands
- `docker-compose.yml` - LocalStack configuration

### Testing
- `send_applications.py` - Generate and send test applications
- `check_approvals.py` - Query approved applications from DynamoDB

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

## Cleanup

```bash
# Stop LocalStack and clean up
make stop
make clean
```

## Notes

- The Lambda function uses a simplified rule-based scoring for demo purposes
- For production, you would deploy the actual trained model to Lambda
- The cutoff (502) is based on optimal separation of goods/bads on training data
- Approval rate: ~55%, Bad rate in approved: ~24%
