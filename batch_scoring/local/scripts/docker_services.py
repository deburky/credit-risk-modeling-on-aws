"""Manage Docker services using docker-py."""

import importlib.util
from pathlib import Path


def _import_module(module_path: Path, module_name: str):
    """Import module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def start_services():
    """Start PostgreSQL and LocalStack services."""
    base_dir = Path(__file__).parent.parent
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"

    # Check if external LocalStack is running
    import contextlib

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


def stop_services():
    """Stop PostgreSQL and LocalStack services."""
    base_dir = Path(__file__).parent.parent
    docker_utils_path = base_dir / "scripts" / "docker_utils.py"
    docker_utils = _import_module(docker_utils_path, "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    docker_mgr.stop_containers(compose_file)
    print("Services stopped")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docker_services.py {start|stop}")
        sys.exit(1)

    command = sys.argv[1]
    try:
        if command == "start":
            start_services()
        elif command == "stop":
            stop_services()
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
