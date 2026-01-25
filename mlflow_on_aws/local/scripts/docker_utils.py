"""Docker utilities for managing containers and images."""

import contextlib
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import docker
from docker.errors import APIError, ImageNotFound


class DockerManager:
    """Manage Docker containers and images programmatically."""

    def __init__(self) -> None:
        """Initialize Docker client."""
        try:
            self.client = docker.from_env()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Docker daemon: {e}") from e

    def image_exists(self, image_name: str) -> bool:
        """Check if Docker image exists locally."""
        try:
            self.client.images.get(image_name)
            return True
        except ImageNotFound:
            return False

    def build_image(
        self,
        image_name: str,
        dockerfile_path: Path,
        build_context: Path,
        tag: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Build Docker image from Dockerfile.

        Args:
            image_name: Name for the image (e.g., "mlflow-sagemaker-train:latest")
            dockerfile_path: Path to Dockerfile
            build_context: Build context directory
            tag: Optional tag (defaults to image_name)
            force: Force rebuild even if image exists

        Returns:
            Image name with tag

        Raises:
            FileNotFoundError: If Dockerfile doesn't exist
            RuntimeError: If build fails
        """
        if not dockerfile_path.exists():
            raise FileNotFoundError(f"Dockerfile not found: {dockerfile_path}")

        if not force and self.image_exists(image_name):
            print(f"Docker image '{image_name}' already exists, skipping build")
            return image_name

        print(f"Building Docker image: {image_name}")
        print(f"Dockerfile: {dockerfile_path}")
        print(f"Build context: {build_context}")

        tag = tag or image_name
        try:
            image, build_logs = self.client.images.build(
                path=str(build_context),
                dockerfile=str(dockerfile_path.relative_to(build_context)),
                tag=tag,
                rm=True,
            )

            for log in build_logs:
                if "stream" in log:
                    print(log["stream"].strip())

            print(f"Successfully built Docker image: {tag}")
            return tag

        except APIError as e:
            raise RuntimeError(f"Failed to build Docker image: {e}") from e

    def remove_image(self, image_name: str) -> None:
        """Remove Docker image.

        Args:
            image_name: Name of image to remove
        """
        try:
            self.client.images.remove(image_name, force=True)
            print(f"Removed Docker image: {image_name}")
        except ImageNotFound:
            print(f"Docker image '{image_name}' not found, skipping removal")
        except APIError as e:
            print(f"Warning: Could not remove image '{image_name}': {e}")

    def container_exists(self, container_name: str) -> bool:
        """Check if container exists (running or stopped)."""
        try:
            self.client.containers.get(container_name)
            return True
        except docker.errors.NotFound:
            return False

    def container_running(self, container_name: str) -> bool:
        """Check if container is currently running."""
        try:
            container = self.client.containers.get(container_name)
            return container.status == "running"
        except docker.errors.NotFound:
            return False

    def is_localstack_running(self) -> bool:
        """Check if LocalStack container is running on port 4566."""
        try:
            containers = self.client.containers.list(
                filters={"publish": "4566"}, all=True
            )
            return len(containers) > 0
        except Exception:
            return False

    def start_containers(
        self,
        compose_file: Path,
        services: Optional[list[str]] = None,
        detach: bool = True,
    ) -> dict[str, dict]:
        """Start containers using docker-compose.

        Args:
            compose_file: Path to docker-compose.yml
            services: Optional list of service names to start
            detach: Run in detached mode

        Returns:
            Dictionary mapping service names to container info
        """
        if not compose_file.exists():
            raise FileNotFoundError(f"docker-compose.yml not found: {compose_file}")

        # Check if docker-compose is available
        compose_check = subprocess.run(
            ["docker-compose", "--version"], capture_output=True, text=True
        )
        if compose_check.returncode != 0:
            raise RuntimeError(
                "docker-compose not found. Please install docker-compose."
            ) from None

        compose_dir = compose_file.parent

        # Stop and remove any existing containers first to avoid conflicts
        # This handles the case where containers from a previous run still exist
        stop_cmd = [
            "docker-compose",
            "-f",
            str(compose_file),
            "down",
            "--remove-orphans",
        ]
        subprocess.run(stop_cmd, cwd=compose_dir, capture_output=True, text=True)
        # Ignore errors from stop - containers may not exist

        # Also force-remove containers by name using Docker API as fallback
        # This handles containers that docker-compose doesn't know about
        container_names = [
            "mlflow-on-aws-localstack",
            "mlflow-on-aws-postgres",
            "mlflow-on-aws-server",
        ]
        for container_name in container_names:
            try:
                container = self.client.containers.get(container_name)
                print(f"Force removing existing container: {container_name}")
                container.remove(force=True)
            except docker.errors.NotFound:
                # Container doesn't exist, which is fine
                pass
            except Exception as e:
                # Log but don't fail - we'll try to start anyway
                print(f"Warning: Could not remove container {container_name}: {e}")

        # Give Docker a moment to clean up
        time.sleep(1)

        cmd = [
            "docker-compose",
            "-f",
            str(compose_file),
            "up",
            "-d",
            "--force-recreate",
            "--remove-orphans",
        ]

        if services:
            cmd.extend(services)

        print(f"Running: {' '.join(cmd)}")
        print("This may take a while if images need to be pulled or built...")
        result = subprocess.run(cmd, cwd=compose_dir, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            if result.stdout:
                print(f"Output: {result.stdout}")
            raise RuntimeError(f"Failed to start containers: {result.stderr}") from None

        if result.stdout:
            print(result.stdout)
        else:
            print("Containers started successfully")

        # Wait for services to be ready
        time.sleep(5)

        # Get container status
        status_cmd = [
            "docker-compose",
            "-f",
            str(compose_file),
            "ps",
            "--format",
            "json",
        ]
        status_result = subprocess.run(
            status_cmd, cwd=compose_dir, capture_output=True, text=True
        )

        containers = {}
        if status_result.returncode == 0:
            for line in status_result.stdout.strip().split("\n"):
                if line:
                    info = json.loads(line)
                    containers[info.get("Service", "")] = {
                        "name": info.get("Name", ""),
                        "status": info.get("State", ""),
                        "ports": info.get("Publishers", []),
                    }

        return containers

    def stop_containers(self, compose_file: Path, remove_volumes: bool = False) -> None:
        """Stop containers using docker-compose.

        Args:
            compose_file: Path to docker-compose.yml
            remove_volumes: Also remove volumes
        """
        if not compose_file.exists():
            raise FileNotFoundError(f"docker-compose.yml not found: {compose_file}")

        compose_dir = compose_file.parent
        cmd = ["docker-compose", "-f", str(compose_file), "down"]

        if remove_volumes:
            cmd.append("-v")

        result = subprocess.run(cmd, cwd=compose_dir, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Failed to stop containers: {result.stderr}") from None

    def get_container_status(self, compose_file: Path) -> dict[str, dict]:
        """Get status of containers from docker-compose.

        Args:
            compose_file: Path to docker-compose.yml

        Returns:
            Dictionary mapping service names to container info
        """
        if not compose_file.exists():
            raise FileNotFoundError(f"docker-compose.yml not found: {compose_file}")

        compose_dir = compose_file.parent
        status_cmd = [
            "docker-compose",
            "-f",
            str(compose_file),
            "ps",
            "--format",
            "json",
        ]
        status_result = subprocess.run(
            status_cmd, cwd=compose_dir, capture_output=True, text=True
        )

        containers = {}
        if status_result.returncode == 0:
            for line in status_result.stdout.strip().split("\n"):
                if line:
                    info = json.loads(line)
                    containers[info.get("Service", "")] = {
                        "name": info.get("Name", ""),
                        "status": info.get("State", ""),
                        "ports": info.get("Publishers", []),
                    }

        return containers

    def wait_for_container_health(
        self, container_name: str, timeout: int = 60, interval: int = 2
    ) -> bool:
        """Wait for container to become healthy.

        Args:
            container_name: Name of container
            timeout: Maximum time to wait in seconds
            interval: Check interval in seconds

        Returns:
            True if healthy, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            with contextlib.suppress(docker.errors.NotFound):
                container = self.client.containers.get(container_name)
                health = (
                    container.attrs.get("State", {}).get("Health", {}).get("Status")
                )
                if health == "healthy":
                    return True
            time.sleep(interval)
        return False
