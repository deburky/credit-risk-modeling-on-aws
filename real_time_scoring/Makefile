.PHONY: help start stop deploy-stack delete-stack stack-status send-applications check-approvals clean

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

deploy-stack: ## Deploy CloudFormation stack
	uv run python deploy_stack.py
	@echo "✓ Stack deployed"

delete-stack: ## Delete CloudFormation stack
	uv run python deploy_stack.py delete

stack-status: ## Show stack status
	uv run python deploy_stack.py status

train-scorecard: ## Train credit scorecard model (direct)
	uv run python train_scorecard.py

train-sagemaker: ## Train with SageMaker Local Mode (Docker)
	uv run python sagemaker_train.py

test-inference: ## Test scorecard inference
	uv run python inference_scorecard.py

send-applications: ## Send test loan applications
	uv run python send_applications.py

check-approvals: ## Check approved applications
	uv run python check_approvals.py

clean: ## Clean up everything
	docker-compose down -v
	rm -rf localstack_data/ model_output/
