# ==============================================================================
# Makefile for LocalStack A/B Testing with SageMaker
# Author: Denis Burakov (@deburky)
# ==============================================================================

.PHONY: help install start stop restart health check-appconfig ensure-running setup clean logs \
	deploy-endpoint test-endpoint endpoint-status delete-endpoint \
	generate-customers score-customers update-appconfig-rules test-traffic-split test-rule-traffic-split test-ab-dynamic-config test-ab-endpoint test-ab \
	deploy-stack describe-stack delete-stack \
	deploy-all test-all lint type-check format install-hooks test test-fast test-all-versions test-version test-ci \
	version status

# Silent mode for certain targets
.SILENT: help health check-appconfig

# Variables (lowercase for local variables)
uv := uv
python := $(uv) run python
docker_compose := docker-compose

# Script paths (DRY principle)
setup_script := ab_tests/setup.py
update_rules_script := ab_tests/update_appconfig_rules.py
generate_customers_script := ab_tests/generate_customer_stream.py
score_customers_script := ab_tests/score_with_appconfig.py
test_traffic_split_script := ab_tests/test_traffic_split.py
test_ab_endpoint_script := ab_tests/test_ab_endpoint.py
test_rule_traffic_split_script := tests/appconfig/test_rule_based_traffic_split.py
test_ab_dynamic_config_script := tests/appconfig/test_ab_testing_dynamic_config.py
sagemaker_train_script := ml_models/sagemaker_train.py
sagemaker_endpoint_script := ml_models/sagemaker_endpoint.py
deploy_stack_script := scripts/deploy_stack.py
send_applications_script := scripts/send_applications.py
check_ab_results_script := scripts/check_ab_results.py

# ============================================================================
# Infrastructure & Setup
# ============================================================================

help:
	@echo "LocalStack A/B Testing with SageMaker - Makefile"
	@echo ""
	@echo "Infrastructure:"
	@echo "make install       - Install Python dependencies"
	@echo "make start         - Start LocalStack"
	@echo "make stop          - Stop LocalStack"
	@echo "make restart       - Restart LocalStack"
	@echo "make health        - Check LocalStack health"
	@echo "make setup         - Set up AppConfig resources"
	@echo "make clean         - Stop LocalStack and clean up data"
	@echo "make logs          - Show LocalStack logs"
	@echo ""
	@echo "SageMaker (ML Models):"
	@echo "make deploy-endpoint - Train model (if needed) and deploy SageMaker endpoint"
	@echo "make test-endpoint   - Test the deployed SageMaker endpoint"
	@echo "make endpoint-status - Check if the SageMaker endpoint is running"
	@echo "make delete-endpoint - Stop and delete the SageMaker endpoint"
	@echo ""
	@echo "A/B Testing:"
	@echo "make test-ab                - Full A/B test (auto-setup + endpoint test)"
	@echo "make test-ab-endpoint        - End-to-end A/B test with endpoint"
	@echo "make test-traffic-split     - Test traffic splitting logic (no dependencies)"
	@echo "make test-rule-traffic-split - Test rule-based traffic splitting"
	@echo "make test-ab-dynamic-config - Test dynamic AppConfig changes"
	@echo "make generate-customers     - Generate customer records for testing"
	@echo "make score-customers        - Score customers using SageMaker + AppConfig"
	@echo "make update-appconfig-rules - Update AppConfig with rules-based configuration"
	@echo ""
	@echo "Deployment:"
	@echo "make deploy-stack    - Deploy CloudFormation stack (SQS + DynamoDB)"
	@echo "make describe-stack - Describe CloudFormation stack"
	@echo "make delete-stack   - Delete CloudFormation stack"
	@echo ""
	@echo "SQS Processing & A/B Testing:"
	@echo "make test-workflow    - Complete workflow: send apps + Lambda processes (recommended)"
	@echo "make send-applications - Send test applications to SQS queue"
	@echo "make check-ab-results  - Check A/B variant assignment and results"
	@echo ""
	@echo "End-to-End:"
	@echo "make deploy-all     - Deploy everything: endpoint + stack + AppConfig setup"
	@echo "make test-all       - Run all tests"
	@echo ""
	@echo "Development:"
	@echo "make lint           - Run ruff linting checks"
	@echo "make type-check     - Run mypy type checking"
	@echo "make format         - Format code with ruff"
	@echo "make install-hooks  - Install pre-commit hooks"
	@echo "make test-ci        - Test CI/CD workflow locally with act"
	@echo ""
	@echo "Utilities:"
	@echo "make version        - Show version information for tools"
	@echo "make status         - Show status dashboard (LocalStack, endpoint, stack)"

install:
	@echo "Installing Python dependencies with uv..."
	$(uv) sync
	@echo "Dependencies installed"

start:
	@echo "Starting LocalStack..."
	$(docker_compose) up -d
	@echo "Waiting for LocalStack to be ready..."
	@for i in $$(seq 1 30); do \
		if curl -s http://localhost:4566/_localstack/health > /dev/null 2>&1; then \
			echo "LocalStack is ready"; \
			break; \
		fi; \
		echo "Waiting... ($$i/30)"; \
		sleep 2; \
	done
	@$(MAKE) health

stop:
	@echo "Stopping LocalStack..."
	$(docker_compose) down
	@echo "LocalStack stopped"

restart: stop start

health:
	@echo "Checking LocalStack health..."
	@if ! curl -sf http://localhost:4566/_localstack/health > /dev/null 2>&1; then \
		echo "LocalStack is not running. Start it with: make start"; \
		exit 1; \
	fi
	@response=$$(curl -sf http://localhost:4566/_localstack/health 2>&1); \
	if [ -z "$$response" ]; then \
		echo "LocalStack health check returned empty response"; \
		exit 1; \
	fi; \
	echo "$$response" | python3 -m json.tool > /dev/null 2>&1 || { \
		echo "LocalStack health check failed - response: $$response"; \
		exit 1; \
	}
	@echo "LocalStack is healthy"

check-appconfig:
	@echo "Checking AppConfig availability..."
	@curl -s http://localhost:4566/_localstack/health | python3 -c "import sys, json; d=json.load(sys.stdin); e=d.get('edition', 'unknown'); print('LocalStack edition:', e); [print('Community edition does not support AppConfig'), print('Use localstack/localstack-pro:latest with Base/Ultimate plan'), print('Get a trial: https://app.localstack.cloud'), sys.exit(1)] if e == 'community' else None"

ensure-running: health

setup: ensure-running
	@echo "Setting up AppConfig resources..."
	@$(python) $(setup_script) || { echo "Setup failed"; exit 1; }
	@echo "AppConfig setup complete"

clean:
	@echo "Cleaning up..."
	$(docker_compose) down -v
	@echo "Cleanup complete"

logs:
	@echo "Showing LocalStack logs..."
	$(docker_compose) logs -f

version:
	@echo "=== Version Information ==="
	@echo "Python: $$($(python) --version 2>&1 || echo 'Not available')"
	@echo "UV: $$($(uv) --version 2>&1 || echo 'Not available')"
	@echo "Docker Compose: $$($(docker_compose) version --short 2>&1 || echo 'Not available')"

status: ensure-running
	@echo "=== LocalStack Status ==="
	@$(MAKE) health
	@echo ""
	@echo "=== SageMaker Endpoint ==="
	@$(MAKE) endpoint-status || echo "No endpoint deployed"
	@echo ""
	@echo "=== CloudFormation Stack ==="
	@$(MAKE) describe-stack || echo "No stack deployed"

# ============================================================================
# SageMaker (ML Models)
# ============================================================================

deploy-endpoint: ensure-running
	@echo "Deploying SageMaker endpoint..."
	@echo "Step 1: Checking if model exists in S3..."
	@if ! $(python) -c "import boto3; from botocore.config import Config; s3=boto3.client('s3', endpoint_url='http://localhost:4566', config=Config(region_name='us-east-1', s3={'endpoint_url': 'http://localhost:4566', 'addressing_style': 'path'})); s3.head_bucket(Bucket='credit-scoring-models'); objs=s3.list_objects_v2(Bucket='credit-scoring-models', Prefix='models/'); exit(0 if objs.get('Contents') else 1)" 2>/dev/null; then \
		echo "Model not found in S3, training with SageMaker..."; \
		$(python) $(sagemaker_train_script); \
	fi
	@echo "Step 2: Deploying endpoint from S3..."
	@if $(python) $(sagemaker_endpoint_script) deploy; then \
		echo "Endpoint deployed and ready"; \
	else \
		echo "Failed to deploy endpoint"; \
		exit 1; \
	fi

test-endpoint: ensure-running
	@echo "Testing SageMaker endpoint..."
	@$(python) $(sagemaker_endpoint_script) test

endpoint-status: ensure-running
	@echo "Checking endpoint status..."
	@$(python) $(sagemaker_endpoint_script) status

delete-endpoint: ensure-running
	@echo "Stopping SageMaker endpoint..."
	@$(python) $(sagemaker_endpoint_script) delete
	@echo "Endpoint stopped"

# ============================================================================
# A/B Testing
# ============================================================================

generate-customers: ensure-running
	@echo "Generating customer records..."
	$(python) $(generate_customers_script) --num-records 20 --output /tmp/customers.json
	@echo "Customer records generated: /tmp/customers.json"

score-customers: ensure-running
	@echo "Scoring customers with SageMaker + AppConfig..."
	@if [ ! -f /tmp/customers.json ]; then \
		echo "Customer records not found, generating..."; \
		$(MAKE) generate-customers; \
	fi
	$(python) $(score_customers_script) --input /tmp/customers.json --output /tmp/scored_customers.json
	@echo "Customers scored: /tmp/scored_customers.json"

update-appconfig-rules: ensure-running
	@echo "Updating AppConfig with rules-based configuration..."
	@$(python) $(update_rules_script) || { echo "Failed to update AppConfig rules"; exit 1; }
	@echo "AppConfig rules updated"

test-traffic-split:
	@echo "Testing traffic splitting logic..."
	$(python) $(test_traffic_split_script)
	@echo "Traffic split test complete"

test-rule-traffic-split: ensure-running
	@echo "Testing rule-based traffic splitting..."
	@echo "Updating AppConfig with rules-based configuration..."
	@$(python) $(update_rules_script) || echo "AppConfig rules update skipped (may already exist)"
	@echo "Running traffic split test..."
	@$(python) $(test_rule_traffic_split_script)
	@echo "Rule-based traffic split test complete"

test-ab-dynamic-config: ensure-running setup
	@echo "Testing dynamic AppConfig changes..."
	$(python) $(test_ab_dynamic_config_script)
	@echo "Dynamic config test complete"

test-ab-endpoint: ensure-running
	@echo "Running end-to-end A/B test with endpoint..."
	@echo "Prerequisites: endpoint deployed (make deploy-endpoint) and AppConfig set up (make setup && make update-appconfig-rules)"
	$(python) $(test_ab_endpoint_script)
	@echo "End-to-end A/B test complete"

test-ab: ensure-running
	@echo "Running full A/B test workflow..."
	@echo "Step 1: Setting up AppConfig (if needed)..."
	@$(python) $(setup_script) || echo "AppConfig setup skipped (may already exist)"
	@echo "Step 2: Updating AppConfig rules..."
	@$(python) $(update_rules_script) || echo "AppConfig rules update skipped"
	@echo "Step 3: Checking/Deploying endpoint..."
	@if ! curl -s http://localhost:8080/ping > /dev/null 2>&1; then \
		echo "Endpoint not running, deploying..."; \
		$(MAKE) deploy-endpoint; \
	else \
		echo "Endpoint is already running"; \
	fi
	@echo "Step 4: Running end-to-end A/B test..."
	@$(python) $(test_ab_endpoint_script)
	@echo ""
	@echo "Full A/B test workflow complete!"
	@echo "Results saved to: /tmp/ab_test_results.json"

# ============================================================================
# Deployment (CloudFormation, SQS, DynamoDB, Lambda)
# ============================================================================

deploy-stack: ensure-running
	@echo "Deploying CloudFormation stack (SQS + DynamoDB + Lambda)..."
	@echo "Note: Lambda function will be packaged automatically"
	$(python) $(deploy_stack_script) deploy
	@echo "Stack deployment complete (Lambda code included)"

describe-stack: ensure-running
	@echo "Describing CloudFormation stack..."
	$(python) $(deploy_stack_script) describe

delete-stack: ensure-running
	@echo "Deleting CloudFormation stack..."
	$(python) $(deploy_stack_script) delete
	@echo "Stack deletion complete"

# ============================================================================
# SQS Processing & A/B Testing
# ============================================================================

send-applications: ensure-running
	@echo "Sending test applications to SQS..."
	$(python) $(send_applications_script) --num 10
	@echo "Applications sent"

test-workflow: ensure-running deploy-stack
	@echo "Running complete workflow test..."
	@echo "1. Sending applications to SQS..."
	@$(python) $(send_applications_script) --num 10
	@echo ""
	@echo "2. Waiting for Lambda to process messages..."
	@sleep 5
	@echo ""
	@echo "3. Showing A/B test results (variant assignments)..."
	@$(python) $(check_ab_results_script)

check-ab-results: ensure-running
	@echo "Checking A/B test results from DynamoDB..."
	@$(python) $(check_ab_results_script)

# ============================================================================
# End-to-End Commands
# ============================================================================

deploy-all: ensure-running deploy-endpoint
	@echo ""
	@echo "Deployment complete!"
	@echo "SageMaker endpoint: deployed"
	@echo ""
	@echo "Next steps:"
	@echo "make test-endpoint        - Test the SageMaker endpoint"
	@echo "make endpoint-status      - Check endpoint status"

test-all: ensure-running
	@echo "Running all tests..."
	@echo "Testing traffic splitting logic..."
	@$(MAKE) test-traffic-split
	@echo ""
	@echo "Testing rule-based traffic splitting..."
	@$(MAKE) test-rule-traffic-split
	@echo ""
	@echo "Testing dynamic AppConfig changes..."
	@$(MAKE) test-ab-dynamic-config
	@echo ""
	@echo "All tests completed"

# ============================================================================
# Development Tools
# ============================================================================

lint:
	@echo "Running ruff linting checks..."
	$(uv) run ruff check .
	@echo "Linting checks complete"

type-check:
	@echo "Running mypy type checking..."
	$(uv) run mypy ab_tests/ ml_models/ deployment/ tests/ || true
	@echo "Type checking complete"

format:
	@echo "Formatting code with ruff..."
	$(uv) run ruff format .
	@echo "Code formatted"

install-hooks:
	@echo "Installing pre-commit hooks with prek..."
	@if ! $(uv) run prek --version > /dev/null 2>&1; then \
		echo "Error: prek is not installed. Install it with: make install"; \
		exit 1; \
	fi
	$(uv) run prek install
	@echo "Pre-commit hooks installed successfully with prek"
	@echo "Hooks will run automatically on git commit"
	@echo "To run manually: $(uv) run prek run --all-files"

test:
	@echo "Running pytest with coverage..."
	$(uv) run pytest --cov=ab_tests --cov=ml_models --cov=deployment --cov-report=term --cov-report=html
	@echo "Tests complete. Coverage report: htmlcov/index.html"

test-fast:
	@echo "Running pytest without coverage..."
	$(uv) run pytest -v
	@echo "Tests complete"

test-all-versions:
	@echo "Running tests on all Python versions (3.10, 3.11, 3.12, 3.13)..."
	@echo "Note: uvx will automatically install Python versions if needed"
	@uvx --python 3.10 tox -e py310 || echo "Python 3.10 tests failed or skipped"
	@uvx --python 3.11 tox -e py311 || echo "Python 3.11 tests failed or skipped"
	@uvx --python 3.12 tox -e py312 || echo "Python 3.12 tests failed or skipped"
	@uvx --python 3.13 tox -e py313 || echo "Python 3.13 tests failed or skipped"
	@echo "All version tests complete"

test-version:
	@echo "Usage: make test-version PYTHON=3.12"
	@echo "Available: 3.10, 3.11, 3.12, 3.13"
	@if [ -z "$(PYTHON)" ]; then \
		echo "Error: PYTHON not specified. Example: make test-version PYTHON=3.12"; \
		exit 1; \
	fi
	@uvx --python $(PYTHON) tox -e py$(PYTHON | tr -d '.')
	@echo "Test complete for Python $(PYTHON)"

test-ci:
	@echo "⚠️  WARNING: act can consume significant memory and disk space!"
	@echo "⚠️  This will run CI jobs SEQUENTIALLY (one at a time) to reduce resource usage."
	@echo "⚠️  For production CI testing, use GitHub Actions or a dedicated CI environment."
	@echo ""
	@echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
	@sleep 3
	@if ! command -v act > /dev/null 2>&1; then \
		echo "Error: act is not installed. Install it with:"; \
		echo "  brew install act  # macOS"; \
		echo "  curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash  # Linux"; \
		exit 1; \
	fi
	@echo "Running CI jobs sequentially with resource limits..."
	@echo "Note: Tests requiring LocalStack will be skipped in CI containers"
	act --container-architecture linux/amd64 --rm --container-options "--memory=4g --memory-swap=4g" --env LOCALSTACK_ENDPOINT=http://host.docker.internal:4566
