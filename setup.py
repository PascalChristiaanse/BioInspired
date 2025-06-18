"""
Setup script for the BioInspired project.
This script helps you get started with the database infrastructure.
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


def create_virtual_environment():
    """Create a virtual environment."""
    venv_path = os.path.join(os.getcwd(), "venv")
    if os.path.exists(venv_path):
        print("[OK] Virtual environment already exists, skipping creation")
        return True
    return run_command("python -m venv venv", "Creating virtual environment")


def get_venv_python():
    """Get the path to the Python executable in the virtual environment."""
    if os.name == "nt":  # Windows
        return os.path.join("venv", "Scripts", "python.exe")
    else:  # Unix/Linux/Mac
        return os.path.join("venv", "bin", "python")


def get_venv_pip():
    """Get the path to the pip executable in the virtual environment."""
    if os.name == "nt":  # Windows
        return os.path.join("venv", "Scripts", "pip.exe")
    else:  # Unix/Linux/Mac
        return os.path.join("venv", "bin", "pip")


def setup_python_environment():
    """Install Python dependencies in virtual environment."""
    pip_path = get_venv_pip()
    return run_command(
        f'"{pip_path}" install -r requirements.txt',
        "Installing Python dependencies in venv",
    )


def start_database():
    """Start the PostgreSQL database using Docker Compose."""
    return run_command("docker-compose up -d", "Starting PostgreSQL database")


def initialize_database():
    """Initialize the database tables."""
    python_path = get_venv_python()
    return run_command(f'"{python_path}" init_db.py', "Initializing database tables")


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
        (create_virtual_environment, "Creating virtual environment"),
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
    print("1. Activate the virtual environment:")
    if os.name == "nt":  # Windows
        print("   .\\venv\\Scripts\\Activate.ps1")
    else:  # Unix/Linux/Mac
        print("   source venv/bin/activate")
    print("2. Check that the database is running: docker-compose ps")
    print("3. Start developing your evolutionary algorithms!")
    print("4. Use the DatabaseManager class to store and retrieve your data")
    print("\nTo run scripts in the future, use:")
    python_path = get_venv_python()
    print(f'   "{python_path}" your_script.py')

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
