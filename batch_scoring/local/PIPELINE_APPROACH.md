# Using SageMaker Pipelines for Database Updates

This document explains how to use **SageMaker Pipelines** to orchestrate the database update step in the batch scoring workflow.

## Overview

SageMaker Pipelines can orchestrate the database update step using a **Processing Step**, which runs custom Python scripts in a containerized environment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              SageMaker Pipeline                              │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │   Inference  │───▶│  Evaluation  │───▶│   Update DB  │ │
│  │    Step      │    │    Step      │    │  (Processing)│ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │   Database      │
                    └─────────────────┘
```

## Benefits

1. **Native ML Integration**: Part of the SageMaker ecosystem
2. **Versioning**: Pipeline executions are tracked and versioned
3. **Reproducibility**: Each run is logged with inputs/outputs
4. **Local Mode**: Can run locally for development/testing
5. **Monitoring**: Built-in monitoring and logging

## Implementation

### Step 1: Create Processing Script

The processing script (`scripts/update_database_processing.py`) runs inside a SageMaker Processing container and:
- Loads evaluation results from S3
- Connects to PostgreSQL
- Updates customer records
- Saves results back to S3

### Step 2: Create Pipeline

```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import ScriptProcessor

# Create processor
processor = ScriptProcessor(
    role=role,
    image_uri="python:3.11-slim",
    command=["python"],
    instance_type="local",  # For local mode
    sagemaker_session=local_session,
)

# Create processing step
update_step = ProcessingStep(
    name="UpdateDatabase",
    processor=processor,
    inputs=[ProcessingInput(
        source="s3://bucket/evaluations.json",
        destination="/opt/ml/processing/input",
    )],
    code="scripts/update_database_processing.py",
    job_arguments=[
        "--db-host", "host.docker.internal",
        "--db-port", "5432",
        # ... other DB params
    ],
)

# Create pipeline
pipeline = Pipeline(
    name="CreditLimitIncreasePipeline",
    steps=[update_step],
)
```

### Step 3: Execute Pipeline

```python
# Create/update pipeline
pipeline.upsert(role_arn=role)

# Execute
execution = pipeline.start()
execution.wait()
```

## Local Mode Setup

For local development:

```python
from sagemaker.local import LocalSession

sagemaker_session = LocalSession()
sagemaker_session.config = {"local": {"local_code": True}}
```

## Hybrid Approach

You can combine SageMaker Pipelines with Step Functions:

```
Step Functions (Orchestration)
    │
    ├─▶ Query Database (Lambda)
    │
    ├─▶ SageMaker Inference (Lambda)
    │
    └─▶ SageMaker Pipeline
            │
            ├─▶ Evaluate Decisions (Processing)
            │
            └─▶ Update Database (Processing)
```

## Comparison: Processing Step vs Lambda

| Aspect | SageMaker Processing Step | Lambda Function |
|--------|---------------------------|-----------------|
| **Execution** | Container-based | Serverless |
| **Duration** | Up to 1 hour | 15 minutes max |
| **Resources** | Configurable CPU/memory | Fixed memory |
| **Cost** | Pay per second | Pay per request |
| **ML Integration** | Native | Manual |
| **Local Testing** | Yes (local mode) | Yes (LocalStack) |
| **Database Access** | Via network | Via network |

## When to Use Processing Step

✅ **Use Processing Step when:**
- You need more than 15 minutes execution time
- You need custom CPU/memory resources
- You want native ML pipeline integration
- You're already using SageMaker Pipelines

✅ **Use Lambda when:**
- Execution is quick (< 15 minutes)
- You want serverless, event-driven architecture
- You need tight integration with other AWS services
- You prefer simpler deployment

## Example: Full Pipeline

```python
# Complete pipeline with all steps
pipeline = Pipeline(
    name="CreditLimitIncreasePipeline",
    steps=[
        query_step,      # Query database (Processing)
        inference_step,  # SageMaker inference (Transform)
        evaluate_step,   # Evaluate decisions (Processing)
        update_step,     # Update database (Processing)
    ],
)
```

## Running the Example

```bash
# Install dependencies
pip install sagemaker psycopg2-binary

# Run pipeline
python sagemaker_pipeline_example.py
```

## Notes

1. **Database Access**: Processing containers need network access to PostgreSQL
   - Use `host.docker.internal` for local PostgreSQL
   - Use VPC configuration for production

2. **Dependencies**: Install `psycopg2-binary` in the container
   - Can be done in the processing script
   - Or use a custom Docker image

3. **Local Mode**: Works with LocalStack S3 for artifact storage

4. **Error Handling**: Pipeline steps can have retry logic and error handling

## Next Steps

1. Create custom Docker image with PostgreSQL client
2. Add more pipeline steps (inference, evaluation)
3. Integrate with Step Functions for hybrid orchestration
4. Add pipeline versioning and experiment tracking
