"""Manage Docker services (LocalStack) using docker-py."""

import importlib.util
import sys
from pathlib import Path


def _import_module(module_path: Path, module_name: str):
    """Import module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_services():
    """Start LocalStack services."""
    base_dir = Path(__file__).parent.parent
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    containers = docker_mgr.start_containers(compose_file)

    print("LocalStack running at http://localhost:4566")
    for service, info in containers.items():
        status = info.get("status", "unknown")
        print(f"{service}: {status}")


def stop_services():
    """Stop LocalStack services."""
    base_dir = Path(__file__).parent.parent
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    container_name = "real-time-scoring-localstack"

    # Check if our LocalStack container is running
    if docker_mgr.container_running(container_name):
        docker_mgr.stop_containers(compose_file)
        print("LocalStack stopped (port 4566 freed)")
        return

    # Check for other LocalStack instances on port 4566
    print("No real-time-scoring LocalStack container found.")
    print("Checking for other LocalStack instances on port 4566...")

    try:
        containers = docker_mgr.client.containers.list(
            filters={"publish": "4566"}, all=True
        )
        if containers:
            other_ls = containers[0]
            print(f"Found: {other_ls.name} (from another module)")
            response = input("Stop it to free port 4566? [y/N] ").strip().lower()
            if response == "y":
                other_ls.stop()
                print(f"Stopped {other_ls.name} (port 4566 freed)")
            else:
                print(f"Keeping {other_ls.name} running")
        else:
            print("No LocalStack instances found on port 4566")
    except Exception as e:
        print(f"Could not check for other containers: {e}")


def restart_services():
    """Restart LocalStack services with fresh data."""
    base_dir = Path(__file__).parent.parent
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"

    print("Stopping LocalStack...")
    docker_mgr.stop_containers(compose_file, remove_volumes=True)

    print("Removing LocalStack data...")
    localstack_data = base_dir / "localstack_data"
    if localstack_data.exists():
        import shutil

        shutil.rmtree(localstack_data, ignore_errors=True)

    print("Starting fresh LocalStack...")
    import time

    containers = docker_mgr.start_containers(compose_file)
    time.sleep(3)  # Additional wait for fresh start

    print("LocalStack restarted fresh at http://localhost:4566")
    for service, info in containers.items():
        status = info.get("status", "unknown")
        print(f"{service}: {status}")


def main():
    """CLI interface."""
    if len(sys.argv) < 2:
        print("Usage: python docker_services.py {start|stop|restart}")
        sys.exit(1)

    command = sys.argv[1]

    try:
        if command == "start":
            start_services()
        elif command == "stop":
            stop_services()
        elif command == "restart":
            restart_services()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python docker_services.py {start|stop|restart}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
