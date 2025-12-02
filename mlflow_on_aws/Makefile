# ============================================================================
# MLflow on AWS - SAM Deployment
# ============================================================================

.PHONY: help build validate deploy-stack delete-stack describe-stack start stop status clean clean-mlflow clean-all train

# Configuration
REGION = eu-west-1
LOCALSTACK_ENDPOINT = http://localhost:4566
USE_LOCALSTACK ?= $(shell docker ps --filter "publish=4566" --format "{{.Names}}" | grep -q localstack && echo "true" || echo "false")
AWS_DEPLOY ?= false
AWS_ENDPOINT_URL = $(if $(filter true,$(USE_LOCALSTACK)),$(LOCALSTACK_ENDPOINT),$(if $(filter true,$(AWS_DEPLOY)),,))

# Load .env file if it exists
ifneq (,$(wildcard .env))
    include .env
    export
endif

# Stack configuration
CF_STACK_NAME = mlflow-stack
CF_STACK_CODE = staging
MLFLOW_ARTIFACTS_PREFIX ?= $(shell grep MLFLOW_ARTIFACTS_PREFIX .env 2>/dev/null | cut -d '=' -f2 || echo "mlflow")
SAGEMAKER_EXECUTION_ROLE ?= $(shell grep SAGEMAKER_EXECUTION_ROLE .env 2>/dev/null | cut -d '=' -f2 || echo "")
SAGEMAKER_DEFAULT_BUCKET ?= $(shell grep SAGEMAKER_DEFAULT_BUCKET .env 2>/dev/null | cut -d '=' -f2 || echo "")

help:
	@echo "MLflow on AWS - SAM Deployment"
	@echo "=============================="
	@echo ""
	@echo "Services:"
	@echo "  make start              - Start all services (LocalStack, PostgreSQL, MinIO, MLflow)"
	@echo "  make stop               - Stop all services"
	@echo "  make status             - Check service status"
	@echo ""
	@echo "Training:"
	@echo "  make train              - Train with SageMaker Local Mode + MLflow"
	@echo ""
	@echo "SAM Deployment:"
	@echo "  make build              - Build SAM application"
	@echo "  make validate           - Validate SAM template"
	@echo "  make deploy-stack       - Deploy stack (LocalStack if running)"
	@echo "  make delete-stack       - Delete stack"
	@echo "  make describe-stack     - Show stack status"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean              - Clean build artifacts"
	@echo "  make clean-mlflow       - Clean MLflow data (PostgreSQL, MinIO)"
	@echo "  make clean-all          - Clean everything (services, data, stack, images)"
	@echo ""
	@echo "  ⚠️  To deploy to REAL AWS: AWS_DEPLOY=true make deploy-stack"
	@echo ""

# Start all services
start:
	@echo "🚀 Starting services..."
	@docker-compose up -d
	@echo "⏳ Waiting for services to be ready..."
	@sleep 10
	@echo "📦 Setting up MinIO bucket..."
	@MINIO_ENDPOINT=http://localhost:9000 \
	 MLFLOW_BUCKET=$(MLFLOW_ARTIFACTS_PREFIX) \
	 bash scripts/setup-minio.sh
	@echo ""
	@echo "✅ Services ready:"
	@echo "   - MLflow UI: http://localhost:5001"
	@echo "   - PostgreSQL: localhost:5433"
	@echo "   - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
	@echo "   - LocalStack: http://localhost:4566"

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	@docker-compose down
	@echo "✅ Services stopped"

# Check service status
status:
	@docker-compose ps

# Build SAM application
build:
	@echo "📦 Building SAM application..."
	@sam build
	@echo "✅ SAM build complete"

# Validate SAM template
validate: build
	@echo "🔍 Validating SAM template..."
	@sam validate
	@echo "✅ Template validated"

# Deploy stack
deploy-stack: build
	@if [ "$(USE_LOCALSTACK)" = "true" ]; then \
		echo "📍 Deploying to LocalStack at $(LOCALSTACK_ENDPOINT)"; \
	elif [ "$(AWS_DEPLOY)" != "true" ]; then \
		echo "⚠️  WARNING: LocalStack not running and AWS_DEPLOY not set!"; \
		echo "   This will deploy to REAL AWS and may incur charges."; \
		echo "   To deploy to AWS, set: AWS_DEPLOY=true make deploy-stack"; \
		echo "   To use LocalStack, run: make start"; \
		exit 1; \
	else \
		echo "📍 Deploying to AWS (Region: $(REGION))"; \
		echo "⚠️  WARNING: Deploying to real AWS account!"; \
	fi
	@if [ -n "$(AWS_ENDPOINT_URL)" ]; then \
		aws cloudformation deploy \
			--endpoint-url "$(AWS_ENDPOINT_URL)" \
			--stack-name "$(CF_STACK_NAME)" \
			--template-file .aws-sam/build/template.yaml \
			--capabilities CAPABILITY_NAMED_IAM \
			--parameter-overrides StackCode="$(CF_STACK_CODE)" SageMakerExecutionRole="$(SAGEMAKER_EXECUTION_ROLE)" SageMakerBucket="$(SAGEMAKER_DEFAULT_BUCKET)" MLflowPrefix="$(MLFLOW_ARTIFACTS_PREFIX)" \
			--region us-east-1; \
	else \
		sam deploy \
			--stack-name "$(CF_STACK_NAME)" \
			--parameter-overrides StackCode="$(CF_STACK_CODE)" SageMakerExecutionRole="$(SAGEMAKER_EXECUTION_ROLE)" SageMakerBucket="$(SAGEMAKER_DEFAULT_BUCKET)" MLflowPrefix="$(MLFLOW_ARTIFACTS_PREFIX)" \
			--capabilities CAPABILITY_NAMED_IAM \
			--region $(REGION) \
			--no-confirm-changeset \
			--no-fail-on-empty-changeset; \
	fi
	@echo "✅ Deployment complete!"
	@$(MAKE) describe-stack

# Describe stack
describe-stack:
	@USE_LOCALSTACK_CHECK=$$(docker ps --filter "publish=4566" --format "{{.Names}}" | grep -q localstack && echo "true" || echo "false"); \
	if [ "$$USE_LOCALSTACK_CHECK" = "true" ]; then \
		ENDPOINT_ARG="--endpoint-url $(LOCALSTACK_ENDPOINT)"; \
		REGION_ARG=us-east-1; \
	else \
		ENDPOINT_ARG=""; \
		REGION_ARG=$(REGION); \
	fi; \
	aws cloudformation describe-stacks \
		$$ENDPOINT_ARG \
		--stack-name "$(CF_STACK_NAME)" \
		--region $$REGION_ARG \
		--query 'Stacks[0].{Status:StackStatus,Outputs:Outputs}' \
		--output table 2>/dev/null || echo "Stack not found"

# Delete stack
delete-stack:
	@if [ "$(USE_LOCALSTACK)" != "true" ] && [ "$(AWS_DEPLOY)" != "true" ]; then \
		echo "⚠️  WARNING: LocalStack not running and AWS_DEPLOY not set!"; \
		echo "   This will delete from REAL AWS."; \
		echo "   To delete from AWS, set: AWS_DEPLOY=true make delete-stack"; \
		echo "   To use LocalStack, run: make start"; \
		exit 1; \
	fi
	@read -p "Delete stack $(CF_STACK_NAME)? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		if [ -n "$(AWS_ENDPOINT_URL)" ]; then \
			aws cloudformation delete-stack \
				--endpoint-url "$(AWS_ENDPOINT_URL)" \
				--stack-name "$(CF_STACK_NAME)" \
				--region us-east-1; \
			REGION_ARG=us-east-1; \
		else \
			aws cloudformation delete-stack \
				--stack-name "$(CF_STACK_NAME)" \
				--region $(REGION); \
			REGION_ARG=$(REGION); \
		fi; \
		echo "✅ Delete initiated"; \
		aws cloudformation wait stack-delete-complete \
			$$ENDPOINT_ARG \
			--stack-name "$(CF_STACK_NAME)" \
			--region $$REGION_ARG 2>/dev/null || echo "Stack deleted"; \
	fi

# Train model with SageMaker Local Mode + MLflow
train:
	@echo "🧪 Training with SageMaker Local Mode + MLflow..."
	@uv run python train_sagemaker.py

# Clean MLflow data (PostgreSQL and MinIO)
clean-mlflow:
	@echo "🧹 Cleaning MLflow data..."
	@docker-compose down -v 2>/dev/null || true
	@rm -rf postgres_data/ minio_data/
	@echo "✅ MLflow data cleaned"

# Clean
clean:
	@echo "🧹 Cleaning..."
	@rm -rf .aws-sam/ iam-policies/*-resolved.json model_output/ Dockerfile.train
	@echo "✅ Cleaned"

# Clean everything (services, data, stack, images)
clean-all: stop
	@echo "🧹 Cleaning all resources..."
	@echo "Deleting CloudFormation stack..."
	@aws cloudformation delete-stack --stack-name "$(CF_STACK_NAME)" --endpoint-url "$(LOCALSTACK_ENDPOINT)" --region us-east-1 2>/dev/null || echo "Stack already deleted or LocalStack not running"
	@$(MAKE) clean-mlflow
	@$(MAKE) clean
	@echo "Removing Docker images..."
	@docker rmi mlflow-sagemaker-train:latest 2>/dev/null || true
	@docker rmi mlflow-on-aws:latest 2>/dev/null || true
	@rm -rf localstack_data/ 2>/dev/null || true
	@echo "✅ All resources cleaned"
