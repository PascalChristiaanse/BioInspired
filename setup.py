"""
Setup script for the BioInspired project.
This script helps you get started with the database infrastructure.
Requires the tudat-space conda environment to be installed.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"[RUNNING] {description}...")
    try:
        result = subprocess.run(
            command, shell=True, check=True, capture_output=True, text=True
        )
        print(f"[OK] {description} completed successfully!")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error during {description.lower()}: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False


def check_docker():
    """Check if Docker is available."""
    return run_command("docker --version", "Checking Docker")


def check_docker_compose():
    """Check if Docker Compose is available."""
    return run_command("docker-compose --version", "Checking Docker Compose")


def check_conda_environment():
    """Check if the tudat-space conda environment exists."""
    print("[INFO] Checking for tudat-space conda environment...")
    try:
        result = subprocess.run(
            "conda env list", shell=True, check=True, capture_output=True, text=True
        )
        if "tudat-space" in result.stdout:
            print("[OK] tudat-space conda environment found!")
            return True
        else:
            print("[ERROR] tudat-space conda environment not found!")
            print("Please create the tudat-space conda environment first.")
            print("You can install it following the TudatPy installation guide:")
            print("https://docs.tudat.space/en/stable/_src_getting_started/_src_installation.html")
            return False
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error checking conda environments: {e}")
        print("Please make sure conda is installed and available in your PATH.")
        return False


def get_conda_python():
    """Get the command to run Python in the tudat-space conda environment."""
    return "conda run -n tudat-space python"


def get_conda_pip():
    """Get the command to run pip in the tudat-space conda environment."""
    return "conda run -n tudat-space pip"


def setup_python_environment():
    """Install Python dependencies in tudat-space conda environment."""
    pip_command = get_conda_pip()
    return run_command(
        f'{pip_command} install -r requirements.txt',
        "Installing Python dependencies in tudat-space environment",
    )


def start_database():
    """Start the PostgreSQL database using Docker Compose."""
    return run_command("docker-compose up -d", "Starting PostgreSQL database")


def initialize_database():
    """Initialize the database tables."""
    python_command = get_conda_python()
    return run_command(f'{python_command} init_db.py', "Initializing database tables")


def main():
    """Main setup process."""
    print("BioInspired Project Setup")
    print("=" * 40)

    # Check prerequisites
    if not check_docker():
        print("[ERROR] Docker is required but not found. Please install Docker first.")
        return False

    if not check_docker_compose():
        print(
            "[ERROR] Docker Compose is required but not found. Please install Docker Compose first."
        )
        return False
    # Setup steps
    steps = [
        (check_conda_environment, "Checking tudat-space conda environment"),
        (setup_python_environment, "Setting up Python environment"),
        (start_database, "Starting PostgreSQL database"),
        (initialize_database, "Initializing database"),
    ]

    for step_func, step_name in steps:
        print(f"\n[INFO] Step: {step_name}")
        if not step_func():
            print(f"[ERROR] Setup failed at step: {step_name}")
            return False
    print("\n[OK] Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate the tudat-space conda environment:")
    print("   conda activate tudat-space")
    print("2. Check that the database is running: docker-compose ps")
    print("3. Start developing your evolutionary algorithms!")
    print("4. Use the DatabaseManager class to store and retrieve your data")
    print("\nTo run scripts in the future, use:")
    python_command = get_conda_python()
    print(f'   {python_command} your_script.py')
    print("   or activate the environment first:")
    print("   conda activate tudat-space")
    print("   python your_script.py")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
