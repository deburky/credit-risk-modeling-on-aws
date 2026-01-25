"""Manage Docker services using docker-py."""

import importlib.util
import sys
from pathlib import Path


def _import_module(module_path: Path, module_name: str):
    """Import module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Import docker_utils from same directory
_scripts_dir = Path(__file__).parent
docker_utils = _import_module(_scripts_dir / "docker_utils.py", "docker_utils")
DockerManager = docker_utils.DockerManager


def start_services() -> None:
    """Start all services (LocalStack, PostgreSQL, MLflow)."""
    base_dir = Path(__file__).parent.parent
    compose_file = base_dir / "docker-compose.yml"

    if not compose_file.exists():
        print(f"Error: docker-compose.yml not found at {compose_file}")
        sys.exit(1)

    try:
        docker_mgr = DockerManager()
    except Exception as e:
        print(f"Error connecting to Docker: {e}")
        print("Make sure Docker is running")
        sys.exit(1)

    print(f"Starting services from {compose_file}...")
    print("Note: This may take a while if building MLflow image for the first time")
    try:
        containers = docker_mgr.start_containers(compose_file)
    except Exception as e:
        print(f"Failed to start containers: {e}")
        print("\nTroubleshooting:")
        print("1. Check if Docker is running: docker ps")
        print("2. Check if docker-compose is installed: docker-compose --version")
        print("3. Try running manually: docker-compose up -d")
        raise

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


def stop_services() -> None:
    """Stop all services."""
    base_dir = Path(__file__).parent.parent
    docker_mgr = DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    docker_mgr.stop_containers(compose_file)
    print("Services stopped")


def stop_services_with_volumes() -> None:
    """Stop all services and remove volumes."""
    base_dir = Path(__file__).parent.parent
    docker_mgr = DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    docker_mgr.stop_containers(compose_file, remove_volumes=True)
    print("Services stopped and volumes removed")


def get_status() -> None:
    """Get status of all services."""
    base_dir = Path(__file__).parent.parent
    docker_mgr = DockerManager()

    compose_file = base_dir / "docker-compose.yml"
    containers = docker_mgr.get_container_status(compose_file)

    if not containers:
        print("No containers found")
        return

    for service, info in containers.items():
        status = info.get("status", "unknown")
        name = info.get("name", "unknown")
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
        print(f"{service} ({name}): {status} - {port_str}")


def is_localstack_running() -> bool:
    """Check if LocalStack is running."""
    docker_mgr = DockerManager()
    return docker_mgr.is_localstack_running()


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print(
            "Usage: python docker_services.py {start|stop|stop-volumes|status|check-localstack}"
        )
        sys.exit(1)

    command = sys.argv[1]
    try:
        if command == "start":
            start_services()
        elif command == "stop":
            stop_services()
        elif command == "stop-volumes":
            stop_services_with_volumes()
        elif command == "status":
            get_status()
        elif command == "check-localstack":
            if is_localstack_running():
                print("true")
                sys.exit(0)
            else:
                print("false")
                sys.exit(1)
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
