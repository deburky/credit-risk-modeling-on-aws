"""Complete workflow orchestrator for batch scoring pipeline."""

import contextlib
import importlib.util
from pathlib import Path


def _import_module(module_path: Path, module_name: str):
    """Import module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_services(base_dir: Path) -> None:
    """Start PostgreSQL and LocalStack services."""
    print("Step 1/5: Starting services...")

    # Import docker_utils
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"

    # Check if external LocalStack is running
    localstack_running = False
    with contextlib.suppress(Exception):
        containers = docker_mgr.client.containers.list(
            filters={"publish": "4566"}, all=True
        )
        if containers:
            localstack_running = True
            print("External LocalStack detected, starting only PostgreSQL")
    # Start services
    if localstack_running:
        containers = docker_mgr.start_containers(compose_file, services=["postgres"])
    else:
        containers = docker_mgr.start_containers(compose_file)

    # Display status
    print("Services started:")
    for service, info in containers.items():
        status = info.get("status", "unknown")
        ports = info.get("ports", [])
        port_str = (
            ", ".join(
                [
                    f"{p.get('PublishedPort', '')}:{p.get('TargetPort', '')}"
                    for p in ports
                ]
            )
            if ports
            else "N/A"
        )
        print(f"{service}: {status} ({port_str})")

    # Show external LocalStack if present
    if localstack_running and "localstack" not in containers:
        with contextlib.suppress(Exception):
            if ext_containers := docker_mgr.client.containers.list(
                filters={"publish": "4566"}, all=True
            ):
                print(f"localstack (external): {ext_containers[0].status}")
    print("PostgreSQL running at localhost:5432")
    print("LocalStack running at http://localhost:4566")


def build_docker_image(base_dir: Path) -> None:
    """Build custom SageMaker training Docker image."""
    print("Step 2/5: Building Docker image...")

    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    dockerfile_path = base_dir / "training" / "Dockerfile"
    build_context = base_dir

    try:
        image_name = docker_mgr.build_image(
            image_name="catboost-sagemaker:latest",
            dockerfile_path=dockerfile_path,
            build_context=build_context,
        )
        print(f"Docker image ready: {image_name}")
    except Exception as e:
        print(f"Docker build failed: {e}")
        raise


def setup_database(base_dir: Path) -> None:
    """Set up PostgreSQL database and load sample data."""
    print("Step 3/5: Setting up database...")

    database_path = base_dir / "scripts" / "database.py"
    database = _import_module(database_path, "database")
    db = database.Database()

    try:
        db.setup(n_customers=50)
        print("Database setup complete")
    except Exception as e:
        print(f"Database setup failed: {e}")
        raise


def setup_eventbridge(base_dir: Path) -> None:
    """Set up EventBridge rule to trigger pipeline via Lambda."""
    print("Step 4/5: Setting up EventBridge...")
    try:
        setup_script = base_dir / "scripts" / "setup_eventbridge.py"
        setup_module = _import_module(setup_script, "setup_eventbridge")
        setup_module.main()
        print("EventBridge setup complete")
    except Exception as e:
        print(f"EventBridge setup failed: {e}")
        raise


def run_pipeline(base_dir: Path) -> None:
    """Run SageMaker pipeline."""
    print("Step 5/5: Running pipeline...")
    try:
        pipeline_script = base_dir / "training" / "sagemaker_pipeline.py"
        pipeline_module = _import_module(pipeline_script, "sagemaker_pipeline")

        # Execute the pipeline
        result = pipeline_module.execute_pipeline_workflow()
        print(f"Pipeline execution started: {result.get('execution_id', 'N/A')}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        raise


def check_results(base_dir: Path) -> None:
    """Check limit increase results from PostgreSQL."""
    print("Checking results...")

    database_path = base_dir / "scripts" / "database.py"
    database = _import_module(database_path, "database")
    db = database.Database()

    try:
        db.check_results()
        print("Results check complete")
    except Exception as e:
        print(f"Results check failed: {e}")
        raise


def run_workflow() -> None:
    """Run the complete workflow."""
    import sys

    base_dir = Path(__file__).parent.parent

    print("Running Complete Workflow")

    try:
        start_services(base_dir)
        build_docker_image(base_dir)
        setup_database(base_dir)
        setup_eventbridge(base_dir)
        run_pipeline(base_dir)
        check_results(base_dir)

        print("Workflow complete")
    except Exception as e:
        print(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run_workflow()
