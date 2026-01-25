"""Remove Docker images."""

from pathlib import Path

import importlib.util


def _import_module(module_path: Path, module_name: str):
    """Import module from file path."""
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Remove Docker images."""
    _scripts_dir = Path(__file__).parent
    docker_utils = _import_module(_scripts_dir / "docker_utils.py", "docker_utils")
    docker_mgr = docker_utils.DockerManager()

    images = ["mlflow-sagemaker-train:latest", "mlflow-on-aws:latest"]
    for image in images:
        docker_mgr.remove_image(image)


if __name__ == "__main__":
    main()
