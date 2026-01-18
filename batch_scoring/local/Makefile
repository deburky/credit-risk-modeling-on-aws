.PHONY: help start stop setup-db run-workflow check-results setup-eventbridge train-catboost train-sagemaker test-lambda test-db-update clean

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Batch Scoring & Loan Limit Management"
	@echo "=================================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

start: ## Start PostgreSQL and LocalStack
	@if docker ps --filter "publish=4566" --format "{{.Names}}" | grep -q localstack; then \
		docker-compose up -d postgres; \
	else \
		docker-compose up -d; \
	fi
	@sleep 5
	@echo "Services started:"
	@docker-compose ps --format "table {{.Service}}\t{{.Status}}\t{{.Ports}}"
	@if docker ps --filter "publish=4566" --format "{{.Names}}" | grep -q localstack && ! docker-compose ps --services | grep -q localstack; then \
		LS_NAME=$$(docker ps --filter "publish=4566" --format "{{.Names}}" | head -1); \
		LS_STATUS=$$(docker ps --filter "publish=4566" --format "{{.Status}}" | head -1); \
		echo "localstack (external)  $$LS_STATUS  0.0.0.0:4566->4566/tcp"; \
	fi
	@echo ""
	@echo "✓ PostgreSQL running at localhost:5432"
	@echo "✓ LocalStack running at http://localhost:4566"

stop: ## Stop PostgreSQL and LocalStack
	docker-compose down

setup-db: ## Setup PostgreSQL database and load sample data
	uv run python scripts/database.py setup

train-catboost: ## Train CatBoost model with SHAP values (direct)
	uv run python training/train_catboost.py

build-docker: ## Build custom SageMaker training Docker image for CatBoost
	@if docker images -q catboost-sagemaker:latest | grep -q .; then \
		echo "✓ Docker image 'catboost-sagemaker:latest' already exists, skipping build"; \
	else \
		echo "Building custom SageMaker training container for CatBoost..."; \
		docker build -t catboost-sagemaker:latest -f training/Dockerfile .; \
		echo "✓ Docker image built: catboost-sagemaker:latest"; \
	fi

train-sagemaker: build-docker ## Train CatBoost with SageMaker Local Mode using custom container (saves to S3)
	uv run python training/sagemaker_train.py

run-workflow: start build-docker ## Run the complete workflow (start services, setup, build, EventBridge, pipeline, results)
	@echo "=================================================================================="
	@echo "Running Complete Workflow"
	@echo "=================================================================================="
	@echo ""
	@echo "Step 1/5: Setting up database..."
	@$(MAKE) setup-db
	@echo ""
	@echo "Step 2/5: Setting up EventBridge..."
	@$(MAKE) setup-eventbridge || (echo "✗ EventBridge setup failed"; exit 1)
	@echo ""
	@echo "Step 3/5: Running pipeline..."
	@uv run python training/sagemaker_pipeline.py || (echo "✗ Pipeline failed - skipping results check"; exit 1)
	@echo ""
	@echo "Step 4/5: Checking results..."
	@$(MAKE) check-results
	@echo ""
	@echo "=================================================================================="
	@echo "✓ Workflow complete!"
	@echo "=================================================================================="

check-results: ## Check limit increase results from PostgreSQL
	uv run python scripts/database.py check

setup-eventbridge: ## Setup EventBridge rule to trigger pipeline via Lambda
	uv run python scripts/setup_eventbridge.py

test-lambda: ## Test Lambda function independently
	uv run python scripts/test_lambda.py

test-db-update: ## Test database update independently
	@echo "Creating test decisions..."
	@echo '[{"customer_id": "CUST_0001", "current_score": 650, "new_score": 720, "current_limit": 5000.0, "new_limit": 7500, "decision": "INCREASE", "reason": "High score"}, {"customer_id": "CUST_0002", "current_score": 580, "new_score": 450, "current_limit": 3000.0, "new_limit": 2700, "decision": "DECREASE", "reason": "Low score"}]' > test_decisions.json
	@uv run python scripts/database.py update --decisions-file test_decisions.json
	@rm -f test_decisions.json
	@echo "Test complete - check results with 'make check-results'"


clean: ## Clean up everything
	docker-compose down -v
	rm -rf localstack_data/
