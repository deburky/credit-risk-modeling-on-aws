# Testing Guide for Batch Scoring

This guide explains how to test the batch scoring and limit increase workflow.

## Quick Test

Run all tests in sequence:

```bash
make test-all
```

This will:
1. Test if LocalStack services are available
2. Run the complete workflow
3. Check and display results

## Step-by-Step Testing

### 1. Start Services

```bash
make start
```

This starts:
- PostgreSQL on `localhost:5432`
- LocalStack on `http://localhost:4566`

### 2. Setup Database

```bash
make setup-db
```

This creates the `batch_scores` table and loads 50 sample customers.

### 3. Train Model (Optional)

If you want to train a fresh model:

```bash
# Fast training (local)
make train-catboost

# OR SageMaker training (saves to S3)
make train-sagemaker
```

### 4. Test Services

Check if LocalStack services are available:

```bash
make test-services
```

This verifies:
- EventBridge (CloudWatch Events) is available
- Step Functions is available
- Shows health status of all LocalStack services

**Expected Output:**
```
✓ EventBridge is available!
✓ Step Functions is available!
```

### 5. Test Individual Components

#### Test Database Connection

```bash
# Check if database is accessible
psql -h localhost -U creditrisk -d credit_scoring -c "SELECT COUNT(*) FROM batch_scores;"
```

#### Test Lambda Functions Individually

You can test each Lambda function directly:

```bash
# Test query function
uv run python -c "
import sys
sys.path.insert(0, 'lambda_functions')
from query_batch_scores import lambda_handler
result = lambda_handler({}, None)
print(result)
"

# Test inference function (requires model)
uv run python -c "
import sys, json
sys.path.insert(0, 'lambda_functions')
from sagemaker_inference import lambda_handler
event = {'customers': [{'customer_id': 'CUST_0001', 'current_score': 650, 'current_limit': 5000}]}
result = lambda_handler(event, None)
print(json.dumps(result, indent=2))
"
```

### 6. Run Complete Workflow

#### Option A: Python Simulation (Recommended for Local Testing)

```bash
make run-workflow
```

This runs the complete workflow using Python (simulates Step Functions):
- Queries eligible customers from PostgreSQL
- Runs SageMaker inference
- Evaluates limit increases
- Updates database

**Expected Output:**
```
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
```

#### Option B: Step Functions (If Available in LocalStack)

```bash
# Deploy stack first
make deploy-stack

# Test scheduled rule
make test-schedule
```

**Note:** Step Functions support in LocalStack Community Edition may be limited. Use Option A for reliable local testing.

### 7. Check Results

```bash
make check-results
```

This displays:
- Summary of approved/declined/pending decisions
- Top approved limit increases
- Sample declined customers

**Expected Output:**
```
Total customers: 50
  ✅ Approved: 12
  ❌ Declined: 8
  ⏳ Pending: 30

Top Approved Limit Increases:
  CUST_0001: $8,500 → $10,625 (+25%)
  CUST_0002: $4,200 → $6,300 (+50%)
  ...
```

## Testing Scenarios

### Scenario 1: Fresh Setup Test

Test everything from scratch:

```bash
# Clean start
make clean
make start
make setup-db
make train-catboost
make test-all
```

### Scenario 2: Re-run Workflow

Test running the workflow multiple times:

```bash
# First run
make run-workflow

# Check results
make check-results

# Re-run (should process remaining eligible customers)
make run-workflow

# Check updated results
make check-results
```

### Scenario 3: Test with Different Data

Modify sample data in `scripts/setup_database.py`:
- Change number of customers
- Adjust score ranges
- Modify limit ranges

Then re-run:

```bash
make setup-db
make run-workflow
make check-results
```

## Verification Checklist

After running tests, verify:

- [ ] PostgreSQL is running and accessible
- [ ] LocalStack services are available
- [ ] Database has sample data (50 customers)
- [ ] Model is trained (if using SageMaker)
- [ ] Workflow completes without errors
- [ ] Results show approved/declined decisions
- [ ] Database is updated with decisions

## Troubleshooting Tests

### PostgreSQL Connection Error

```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# Check logs
docker-compose logs postgres

# Test connection
psql -h localhost -U creditrisk -d credit_scoring
```

### LocalStack Services Not Available

```bash
# Check LocalStack health
curl http://localhost:4566/_localstack/health

# Check logs
docker-compose logs localstack

# Restart LocalStack
make stop
make start
```

### Model Not Found

```bash
# Train model first
make train-catboost

# OR if using SageMaker
make train-sagemaker
```

### No Eligible Customers

Check database:

```bash
psql -h localhost -U creditrisk -d credit_scoring -c "
SELECT 
    COUNT(*) as total,
    COUNT(CASE WHEN current_score >= 600 THEN 1 END) as score_ok,
    COUNT(CASE WHEN current_limit < 10000 THEN 1 END) as limit_ok,
    COUNT(CASE WHEN limit_increase_decision IS NULL THEN 1 END) as pending
FROM batch_scores;
"
```

If all customers have been processed, reset:

```bash
# Reset decisions
psql -h localhost -U creditrisk -d credit_scoring -c "
UPDATE batch_scores 
SET limit_increase_decision = NULL, 
    new_limit = NULL, 
    decision_reason = NULL, 
    updated_at = NULL;
"
```

## Integration Tests

### Test End-to-End Flow

```bash
# Complete workflow test
make start
make setup-db
make train-catboost
make deploy-stack
make run-workflow
make check-results
```

### Test Scheduled Execution

```bash
# Deploy stack
make deploy-stack

# Test schedule (manually trigger)
make test-schedule

# Check if execution started
aws stepfunctions list-executions \
  --state-machine-arn <arn> \
  --endpoint-url http://localhost:4566
```

## Performance Testing

### Test with Larger Dataset

Modify `scripts/setup_database.py` to generate more customers:

```python
n_customers = 500  # Increase from 50
```

Then test:

```bash
make setup-db
time make run-workflow
```

### Test Batch Size

Modify query limit in `lambda_functions/query_batch_scores.py`:

```python
LIMIT 50  # Increase from 20
```

## Continuous Testing

For development, you can set up a test loop:

```bash
# Watch mode (requires entr or similar)
while true; do
  make run-workflow
  make check-results
  sleep 5
done
```

## Next Steps

After testing:
- Review results in database
- Check logs for any errors
- Verify business logic (approval criteria)
- Test edge cases (boundary conditions)
- Test error handling
