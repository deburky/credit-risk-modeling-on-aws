.PHONY: help start stop setup-db run-workflow check-results test-schedule test-services test-all clean

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Chapter 2: Batch Scoring & Limit Increase (Local)"
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
	uv run python scripts/setup_database.py

train-catboost: ## Train CatBoost model with SHAP values (direct)
	uv run python train_catboost.py

build-docker: ## Build custom SageMaker training Docker image for CatBoost
	@if docker images -q catboost-sagemaker:latest | grep -q .; then \
		echo "✓ Docker image 'catboost-sagemaker:latest' already exists, skipping build"; \
	else \
		echo "Building custom SageMaker training container for CatBoost..."; \
		docker build -t catboost-sagemaker:latest -f Dockerfile .; \
		echo "✓ Docker image built: catboost-sagemaker:latest"; \
	fi

train-sagemaker: build-docker ## Train CatBoost with SageMaker Local Mode using custom container (saves to S3)
	uv run python sagemaker_train.py

clean-docker: ## Remove custom Docker image
	docker rmi catboost-sagemaker:latest 2>/dev/null || echo "Image not found or already removed"

run-workflow: ## Run the limit increase workflow (SageMaker Pipeline)
	uv run python sagemaker_pipeline.py

check-results: ## Check limit increase results from PostgreSQL
	uv run python scripts/check_results.py

deploy-stack: ## Deploy CloudFormation stack (Step Functions + EventBridge)
	uv run python deploy_stack.py

delete-stack: ## Delete CloudFormation stack
	uv run python deploy_stack.py delete

stack-status: ## Show stack status
	uv run python deploy_stack.py status

test-services: ## Test if LocalStack services (EventBridge, Step Functions) are available
	uv run python scripts/test_services.py

test-schedule: ## Test EventBridge scheduled rule (manually trigger)
	uv run python scripts/test_schedule.py

test-all: ## Run all tests (services, workflow, results)
	@echo "=================================================================================="
	@echo "Running All Tests"
	@echo "=================================================================================="
	@echo ""
	@echo "Step 1/3: Testing LocalStack services..."
	@$(MAKE) test-services
	@echo ""
	@echo "Step 2/3: Running workflow..."
	@$(MAKE) run-workflow
	@echo ""
	@echo "Step 3/3: Checking results..."
	@$(MAKE) check-results
	@echo ""
	@echo "=================================================================================="
	@echo "✓ All tests complete!"
	@echo "=================================================================================="

diagram: ## Generate architecture diagram
	cd .. && uv run python draw_diagram.py

pdf: ## Convert technical document to letter-size PDF
	cd .. && \
	gs -sPAPERSIZE=letter \
		-dFIXEDMEDIA \
		-dPDFFitPage \
		-dCompatibilityLevel=1.4 \
		-sDEVICE=pdfwrite \
		-o batch_scoring_credit_limit_management.pdf \
		technical_doc.pdf && \
	echo "✓ Created batch_scoring_credit_limit_management.pdf"

clean: ## Clean up everything
	docker-compose down -v
	rm -rf localstack_data/

