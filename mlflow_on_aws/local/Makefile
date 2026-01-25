# ============================================================================
# MLflow on AWS - Local Development
# ============================================================================

.PHONY: help start stop restart status train clean clean-mlflow clean-all build validate deploy-stack delete-stack describe-stack

# Configuration
MLFLOW_ARTIFACTS_PREFIX ?= mlflow
REGION = us-east-1
LOCALSTACK_ENDPOINT = http://localhost:4566
USE_LOCALSTACK ?= $(shell uv run python scripts/docker_services.py check-localstack 2>/dev/null || echo "false")
AWS_DEPLOY ?= false
AWS_ENDPOINT_URL = $(if $(filter true,$(USE_LOCALSTACK)),$(LOCALSTACK_ENDPOINT),$(if $(filter true,$(AWS_DEPLOY)),,))

# Root directory (parent of local/)
ROOT_DIR = $(shell cd .. && pwd)

# Load .env file from root if it exists
ifneq (,$(wildcard $(ROOT_DIR)/.env))
    include $(ROOT_DIR)/.env
    export
endif

# Stack configuration
CF_STACK_NAME = mlflow-stack
CF_STACK_CODE = staging
MLFLOW_ARTIFACTS_PREFIX ?= $(shell grep MLFLOW_ARTIFACTS_PREFIX $(ROOT_DIR)/.env 2>/dev/null | cut -d '=' -f2 || echo "mlflow")
SAGEMAKER_EXECUTION_ROLE ?= $(shell grep SAGEMAKER_EXECUTION_ROLE $(ROOT_DIR)/.env 2>/dev/null | cut -d '=' -f2 || echo "AmazonSageMaker-ExecutionRole-20200101T000001")
SAGEMAKER_DEFAULT_BUCKET ?= $(shell grep SAGEMAKER_DEFAULT_BUCKET $(ROOT_DIR)/.env 2>/dev/null | cut -d '=' -f2 || echo "")

help: ## Show available targets
	@echo "MLflow on AWS - Local Development"
	@echo "=================================="
	@echo ""
	@echo "Services:"
	@echo "  make start              - Start all services (LocalStack, PostgreSQL, MLflow)"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services (stop then start)"
	@echo "  make status             - Check service status"
	@echo ""
	@echo "Training:"
	@echo "  make train              - Train with SageMaker Local Mode + MLflow"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-mlflow       - Clean MLflow data (PostgreSQL, LocalStack S3)"
	@echo "  make clean-all          - Clean everything (services, data, images)"
	@echo ""
	@echo "SAM Deployment:"
	@echo "  make build              - Build SAM application"
	@echo "  make validate           - Validate SAM template"
	@echo "  make deploy-stack       - Deploy stack (LocalStack if running)"
	@echo "  make delete-stack       - Delete stack"
	@echo "  make describe-stack     - Show stack status"
	@echo ""
	@echo "  To deploy to REAL AWS: AWS_DEPLOY=true make deploy-stack"
	@echo ""

start: ## Start all services
	@echo "Starting services..."
	@uv run python scripts/docker_services.py start
	@echo "Waiting for services to be ready..."
	@sleep 10
	@echo "Setting up LocalStack S3 bucket..."
	@BUCKET_NAME=$${MLFLOW_ARTIFACTS_PREFIX:-mlflow}; \
	 LOCALSTACK_ENDPOINT=http://localhost:4566 \
	 MLFLOW_BUCKET=$$BUCKET_NAME \
	 uv run python scripts/setup_s3.py
	@echo "Services ready:"
	@echo "- MLflow UI: http://localhost:5001"
	@echo "- PostgreSQL: localhost:5433"
	@echo "- LocalStack: http://localhost:4566"

stop: ## Stop all services
	@echo "Stopping services..."
	@uv run python scripts/docker_services.py stop
	@echo "Services stopped"

restart: ## Restart all services
	@echo "Restarting services..."
	@$(MAKE) stop
	@echo "Waiting for containers to fully stop..."
	@sleep 2
	@$(MAKE) start

status: ## Check service status
	@uv run python scripts/docker_services.py status

train: ## Train with SageMaker Local Mode + MLflow
	@echo "Training with SageMaker Local Mode + MLflow..."
	@uv run python train_sagemaker.py

clean-mlflow: ## Clean MLflow data
	@echo "Cleaning MLflow data..."
	@uv run python scripts/docker_services.py stop-volumes 2>/dev/null || true
	@rm -rf postgres_data/
	@echo "MLflow data cleaned"

clean: ## Clean build artifacts
	@echo "Cleaning..."
	@rm -rf model_output/ training/Dockerfile.train
	@echo "Cleaned"

clean-all: stop ## Clean everything
	@echo "Cleaning all resources..."
	@$(MAKE) clean-mlflow
	@$(MAKE) clean
	@echo "Removing Docker images..."
	@uv run python scripts/remove_images.py 2>/dev/null || true
	@rm -rf localstack_data/ 2>/dev/null || true
	@echo "All resources cleaned"

build: ## Build SAM application
	@echo "Building SAM application..."
	@sam build --template template.yaml
	@echo "SAM build complete"

validate: build ## Validate SAM template
	@echo "Validating SAM template..."
	@sam validate --template template.yaml
	@echo "Template validated"

deploy-stack: build ## Deploy stack
	@if [ "$(USE_LOCALSTACK)" = "true" ]; then \
		echo "Deploying to LocalStack at $(LOCALSTACK_ENDPOINT)"; \
	elif [ "$(AWS_DEPLOY)" != "true" ]; then \
		echo "WARNING: LocalStack not running and AWS_DEPLOY not set!"; \
		echo "   This will deploy to REAL AWS and may incur charges."; \
		echo "   To deploy to AWS, set: AWS_DEPLOY=true make deploy-stack"; \
		echo "   To use LocalStack, run: make start"; \
		exit 1; \
	else \
		echo "Deploying to AWS (Region: $(REGION))"; \
		echo "WARNING: Deploying to real AWS account!"; \
	fi
	@if [ -n "$(AWS_ENDPOINT_URL)" ]; then \
		aws cloudformation deploy \
		--endpoint-url "$(AWS_ENDPOINT_URL)" \
		--stack-name "$(CF_STACK_NAME)" \
		--template-file .aws-sam/build/template.yaml \
		--capabilities CAPABILITY_NAMED_IAM \
		--parameter-overrides StackCode="$(CF_STACK_CODE)" SageMakerExecutionRole="$(SAGEMAKER_EXECUTION_ROLE)" SageMakerBucket="$(SAGEMAKER_DEFAULT_BUCKET)" MLflowPrefix="$(MLFLOW_ARTIFACTS_PREFIX)" \
		--region $(REGION); \
	else \
		sam deploy \
		--stack-name "$(CF_STACK_NAME)" \
		--parameter-overrides StackCode="$(CF_STACK_CODE)" SageMakerExecutionRole="$(SAGEMAKER_EXECUTION_ROLE)" SageMakerBucket="$(SAGEMAKER_DEFAULT_BUCKET)" MLflowPrefix="$(MLFLOW_ARTIFACTS_PREFIX)" \
		--capabilities CAPABILITY_NAMED_IAM \
		--region $(REGION) \
		--no-confirm-changeset \
		--no-fail-on-empty-changeset; \
	fi
	@echo "Deployment complete!"
	@$(MAKE) describe-stack

describe-stack: ## Show stack status
	@USE_LOCALSTACK_CHECK=$$(uv run python scripts/docker_services.py check-localstack 2>/dev/null || echo "false"); \
	if [ "$$USE_LOCALSTACK_CHECK" = "true" ]; then \
		ENDPOINT_ARG="--endpoint-url $(LOCALSTACK_ENDPOINT)"; \
		REGION_ARG=$(REGION); \
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

delete-stack: ## Delete stack
	@if [ "$(USE_LOCALSTACK)" != "true" ] && [ "$(AWS_DEPLOY)" != "true" ]; then \
		echo "WARNING: LocalStack not running and AWS_DEPLOY not set!"; \
		echo "This will delete from REAL AWS."; \
		echo "To delete from AWS, set: AWS_DEPLOY=true make delete-stack"; \
		echo "To use LocalStack, run: make start"; \
		exit 1; \
	fi
	@read -p "Delete stack $(CF_STACK_NAME)? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		if [ -n "$(AWS_ENDPOINT_URL)" ]; then \
			aws cloudformation delete-stack \
			--endpoint-url "$(AWS_ENDPOINT_URL)" \
			--stack-name "$(CF_STACK_NAME)" \
			--region $(REGION); \
			REGION_ARG=$(REGION); \
		else \
			aws cloudformation delete-stack \
			--stack-name "$(CF_STACK_NAME)" \
			--region $(REGION); \
			REGION_ARG=$(REGION); \
		fi; \
		echo "Delete initiated"; \
		aws cloudformation wait stack-delete-complete \
			$$ENDPOINT_ARG \
			--stack-name "$(CF_STACK_NAME)" \
			--region $$REGION_ARG 2>/dev/null || echo "Stack deleted"; \
	fi
