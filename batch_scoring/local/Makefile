.PHONY: help start stop setup-db check-results setup-eventbridge train-catboost train-sagemaker run-workflow clean

.DEFAULT_GOAL := help

help: ## Show this help
	@echo "Batch Scoring & Loan Limit Management"
	@echo "=================================================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

start: ## Start PostgreSQL and LocalStack
	uv run python scripts/docker_services.py start

stop: ## Stop PostgreSQL and LocalStack
	uv run python scripts/docker_services.py stop

setup-db: ## Setup PostgreSQL database and load sample data
	uv run python scripts/database.py setup

check-results: ## Check limit increase results from PostgreSQL
	uv run python scripts/database.py check

setup-eventbridge: ## Setup EventBridge rule to trigger pipeline via Lambda
	uv run python scripts/setup_eventbridge.py

train-catboost: ## Train CatBoost model with SHAP values (direct)
	uv run python training/train_catboost.py

train-sagemaker: ## Train CatBoost with SageMaker Local Mode using custom container (saves to S3)
	uv run python training/sagemaker_train.py

run-workflow: ## Run the complete workflow (start services, setup, build, EventBridge, pipeline, results)
	uv run python scripts/workflow.py

clean: ## Clean up everything
	uv run python scripts/docker_services.py stop
	rm -rf localstack_data/
