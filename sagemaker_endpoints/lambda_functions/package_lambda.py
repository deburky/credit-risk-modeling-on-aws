"""
Package Lambda function for deployment.

Creates a zip file with the Lambda function code and dependencies.
"""

import shutil
import subprocess
import sys
from pathlib import Path

LAMBDA_DIR = Path(__file__).parent / "ab_processor"
OUTPUT_DIR = Path(__file__).parent.parent / "lambda_packages"
PACKAGE_NAME = "ab_processor.zip"


def package_lambda():
    """Package the Lambda function."""
    print(f"Packaging Lambda function from {LAMBDA_DIR}...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    package_path = OUTPUT_DIR / PACKAGE_NAME

    # Remove existing package
    if package_path.exists():
        print(f"Removing existing package: {package_path}")
        package_path.unlink()

    # Create temporary directory for packaging
    temp_dir = OUTPUT_DIR / "temp_lambda"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        # Copy Lambda function code
        lambda_function = LAMBDA_DIR / "lambda_function.py"
        if not lambda_function.exists():
            raise FileNotFoundError(f"Lambda function not found: {lambda_function}")

        shutil.copy2(lambda_function, temp_dir / "lambda_function.py")
        print("✓ Copied lambda_function.py")

        # Install dependencies
        requirements = LAMBDA_DIR / "requirements.txt"
        if requirements.exists():
            print(f"Installing dependencies from {requirements}...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(requirements),
                    "-t",
                    str(temp_dir),
                    "--quiet",
                ],
                check=True,
            )
            print("✓ Dependencies installed")
        else:
            print("⚠ No requirements.txt found, skipping dependencies")

        # Create zip file
        print(f"Creating zip package: {package_path}")
        shutil.make_archive(str(package_path.with_suffix("")), "zip", temp_dir)
        print(f"✓ Package created: {package_path}")
        print(f"  Size: {package_path.stat().st_size / 1024:.1f} KB")

    finally:
        # Cleanup temp directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("✓ Cleaned up temporary files")


if __name__ == "__main__":
    try:
        package_lambda()
        print("\n✓ Lambda packaging complete!")
    except Exception as e:
        print(f"\n✗ Error packaging Lambda: {e}")
        sys.exit(1)
