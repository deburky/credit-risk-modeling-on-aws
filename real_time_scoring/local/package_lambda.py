"""
Package Lambda function with dependencies for deployment
Optimized for size using uv and aggressive cleanup
"""

import contextlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


def package_lambda():  # sourcery skip: extract-method
    """Package Lambda function with dependencies"""

    print("Packaging Lambda Function (Optimized for Size)")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Copy Lambda function
        lambda_file = Path(__file__).parent / "lambda_function.py"
        shutil.copy(lambda_file, tmpdir_path / "index.py")
        print("✓ Copied Lambda function")

        # Install dependencies to temp directory
        # IMPORTANT: Use Python 3.11 to match Lambda runtime (python3.11)
        print("\nInstalling dependencies with uv (Python 3.11, optimized for size)...")

        # Try to use python3.11 explicitly, fallback to python3 if not available
        python_cmd = shutil.which("python3.11") or shutil.which("python3")
        if python_cmd and "3.11" not in python_cmd:
            # Verify Python version
            version_check = subprocess.run(
                [python_cmd, "--version"],
                capture_output=True,
                text=True,
            )
            if "3.11" not in version_check.stdout:
                print(f"⚠ Warning: Using {python_cmd} (not Python 3.11)")
                print(
                    "Lambda runtime is python3.11 - there may be compatibility issues"
                )

        if uv_cmd := shutil.which("uv"):
            print("  Using uv for optimized package installation...")
            subprocess.run(
                [
                    uv_cmd or "uv",
                    "pip",
                    "install",
                    "--python",
                    python_cmd or "python3",
                    "-r",
                    str(Path(__file__).parent / "requirements_lambda.txt"),
                    "--target",
                    str(tmpdir_path),
                    "--quiet",
                    "--no-cache",
                ],
                check=True,
            )
        else:
            print("Using pip (uv not found)...")
            subprocess.run(
                [
                    python_cmd or "python3",
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(Path(__file__).parent / "requirements_lambda.txt"),
                    "-t",
                    str(tmpdir_path),
                    "--quiet",
                    "--no-cache-dir",
                ],
                check=True,
            )

        print("  Cleaning up unnecessary files to reduce package size...")

        # Aggressively remove unnecessary files and directories
        # Patterns for directories to remove
        dirs_to_remove = {
            "tests",
            "test",
            "testing",
            "tests.py",
            "docs",
            "doc",
            "documentation",
            "examples",
            "example",
            "demos",
            "demo",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".cache",
        }

        # File extensions to remove
        file_extensions_to_remove = {
            ".pyc",
            ".pyo",
            ".pyd",  # Compiled Python
            ".pyx",
            ".pxd",
            ".pxi",  # Cython source
            ".c",
            ".cpp",
            ".h",
            ".hpp",
            ".hxx",  # C/C++ source (keep .so)
            ".cmake",
            ".cmake.in",
            ".ipynb",  # Jupyter notebooks
            ".md",
            ".txt",
            ".rst",
            ".html",
            ".css",  # Documentation
            ".pyi",  # Type stubs
        }

        # Remove directories
        for root, dirs, files in os.walk(tmpdir_path):
            dirs[:] = [d for d in dirs if d not in dirs_to_remove]

            # Remove files
            for file in files:
                file_path = Path(root) / file
                should_remove = False

                # Check file extension
                if any(file.endswith(ext) for ext in file_extensions_to_remove):
                    should_remove = True
                # Check for documentation files
                elif file.lower() in [
                    "readme",
                    "license",
                    "changelog",
                    "authors",
                    "contributors",
                ]:
                    should_remove = True
                # Check for CMake files
                elif file.lower() in ["cmakelists.txt", "setup.py"]:
                    should_remove = True

                if should_remove:
                    with contextlib.suppress(Exception):
                        file_path.unlink()
        # Clean up .dist-info directories (remove large files but keep essential metadata)
        for dist_info in tmpdir_path.rglob("*.dist-info"):
            if dist_info.is_dir():
                # Remove documentation files from dist-info
                for item in dist_info.iterdir():
                    if item.is_file():
                        if item.suffix in [".txt", ".md", ".rst"]:
                            item.unlink(missing_ok=True)
                        # Keep METADATA, RECORD, WHEEL, but remove others
                        elif item.name not in [
                            "METADATA",
                            "RECORD",
                            "WHEEL",
                            "INSTALLER",
                        ]:
                            if (
                                item.suffix in [".json"]
                                and item.name != "direct_url.json"
                            ):
                                item.unlink(missing_ok=True)

        # Calculate size before zipping
        total_size = sum(
            f.stat().st_size for f in tmpdir_path.rglob("*") if f.is_file()
        )
        print(f"  Package size before compression: {total_size / 1024 / 1024:.2f} MB")

        if total_size > 500 * 1024 * 1024:  # 500MB
            print(
                f"Warning: Package size ({total_size / 1024 / 1024:.2f} MB) exceeds 500MB limit"
            )
            print("Consider using Lambda Layers or Container Images for production")

        print("✓ Installed and cleaned dependencies")

        # Create zip file
        zip_path = Path(__file__).parent / "lambda_package.zip"
        if zip_path.exists():
            zip_path.unlink()

        print(f"\nCreating zip package: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(tmpdir_path):
                # Skip __pycache__ directories
                dirs[:] = [d for d in dirs if d != "__pycache__"]
                for file in files:
                    if file.endswith(".pyc"):
                        continue
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(tmpdir_path)
                    zipf.write(file_path, arcname)

        zip_size = zip_path.stat().st_size
        print(f"✓ Created {zip_path} ({zip_size / 1024 / 1024:.2f} MB compressed)")

        # Estimate unzipped size (rough approximation)
        # Decompress to check actual size
        with tempfile.TemporaryDirectory() as check_dir:
            with zipfile.ZipFile(zip_path, "r") as zipf:
                zipf.extractall(check_dir)
            unzipped_size = sum(
                f.stat().st_size for f in Path(check_dir).rglob("*") if f.is_file()
            )
            print(f"  Unzipped size: {unzipped_size / 1024 / 1024:.2f} MB")

            if unzipped_size > 500 * 1024 * 1024:
                print(
                    f"ERROR: Unzipped size ({unzipped_size / 1024 / 1024:.2f} MB) exceeds 500MB limit!"
                )
                print("The package will fail to deploy to Lambda.")
                print("Solutions:")
                print("1. Use Lambda Layers for large dependencies")
                print("2. Use Lambda Container Images (10GB limit)")
                print("3. Remove more dependencies or use lighter alternatives")

        print("=" * 80)

        return zip_path


if __name__ == "__main__":
    package_lambda()
