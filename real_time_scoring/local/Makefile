.PHONY: help start stop restart-localstack deploy-stack delete-stack stack-status send-applications check-approvals clean run-workflow

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Chapter 1: Credit Scoring (Local)"
	@echo "=================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

start: ## Start LocalStack
	docker-compose up -d
	@sleep 5
	@echo "✓ LocalStack running at http://localhost:4566"

stop: ## Stop LocalStack
	docker-compose down

restart-localstack: ## Stop LocalStack, remove all data, and restart fresh
	@echo "Stopping LocalStack..."
	@docker-compose down -v
	@echo "Removing LocalStack data..."
	@rm -rf localstack_data/ 2>/dev/null || true
	@echo "Starting fresh LocalStack..."
	@docker-compose up -d
	@sleep 8
	@echo "✓ LocalStack restarted fresh at http://localhost:4566"

deploy-stack: ## Deploy CloudFormation stack
	uv run python scripts/deploy_stack.py
	@echo "✓ Stack deployed"

delete-stack: ## Delete CloudFormation stack
	uv run python scripts/deploy_stack.py delete
	@echo "✓ Stack deletion complete"

stack-status: ## Show stack status
	uv run python scripts/deploy_stack.py status

stack-events: ## Show recent stack events (for debugging)
	@echo "Fetching stack events..."
	@uv run python -c "import sys; sys.path.insert(0, 'scripts'); from deploy_stack import show_stack_events; show_stack_events()"

train-scorecard: ## Train credit scorecard model (direct)
	uv run python scripts/local_train.py

build-docker: ## Build custom SageMaker training Docker image
	@if docker images -q credit-scoring-sagemaker:latest | grep -q .; then \
		echo "✓ Docker image 'credit-scoring-sagemaker:latest' already exists, skipping build"; \
	else \
		echo "Building custom SageMaker training container..."; \
		docker build -t credit-scoring-sagemaker:latest -f Dockerfile .; \
		echo "✓ Docker image built: credit-scoring-sagemaker:latest"; \
	fi

train-sagemaker: build-docker ## Train with SageMaker Local Mode using custom container (saves to S3)
	uv run python scripts/sagemaker_train.py

package-lambda: ## Package Lambda function with dependencies
	uv run python scripts/package_lambda.py

test-inference: ## Test scorecard inference
	uv run python -m src.scorecard.inference_scorecard

send-applications: ## Send test loan applications
	uv run python scripts/send_applications.py

check-approvals: ## Check approved applications
	uv run python scripts/check_approvals.py

verify-setup: ## Verify end-to-end setup is ready
	uv run python scripts/verify_setup.py

run-workflow: ## Run complete end-to-end workflow with SageMaker (start → train → deploy → send → check)
	@echo "=================================================================================="
	@echo "Running Complete End-to-End Workflow"
	@echo "=================================================================================="
	@echo ""
	@echo "Step 1/5: Starting LocalStack..."
	@$(MAKE) start
	@echo ""
	@echo "Step 2/5: Training model with SageMaker (saves to S3)..."
	@$(MAKE) train-sagemaker
	@echo ""
	@echo "Step 3/5: Deploying CloudFormation stack..."
	@$(MAKE) deploy-stack
	@echo ""
	@echo "Step 4/5: Sending test applications..."
	@$(MAKE) send-applications
	@echo ""
	@echo "Step 5/5: Checking approved applications..."
	@$(MAKE) check-approvals
	@echo ""
	@echo "=================================================================================="
	@echo "✓ Workflow complete!"
	@echo "=================================================================================="

clean: ## Clean up everything
	docker-compose down -v
	rm -rf localstack_data/ model_output/

clean-docker: ## Remove custom Docker image
	docker rmi credit-scoring-sagemaker:latest 2>/dev/null || echo "Image not found or already removed"
