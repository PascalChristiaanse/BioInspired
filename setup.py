def is_admin():
    """Check if the script is running with admin rights (Windows only)."""
    try:
        import ctypes

        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False


def relaunch_as_admin():
    """Relaunch the script with admin rights (Windows only)."""
    import sys
    import os
    import ctypes

    params = " ".join([f'"{arg}"' for arg in sys.argv])
    executable = sys.executable
    script = os.path.abspath(sys.argv[0])
    # Use ShellExecuteW to relaunch as admin
    ret = ctypes.windll.shell32.ShellExecuteW(
        None, "runas", executable, f'"{script}"', None, 1
    )
    if int(ret) <= 32:
        print(
            "[ERROR] Failed to elevate permissions. Please run this script as administrator."
        )
        sys.exit(1)
    sys.exit(0)


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
            print(
                "https://docs.tudat.space/en/stable/_src_getting_started/_src_installation.html"
            )
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
    """Install Python dependencies in tudat-space conda environment and Blender's Python."""
    
    # First, configure conda channels and install conda packages
    print("[INFO] Configuring conda channels...")
    if not run_command(
        "conda config --add channels conda-forge",
        "Adding conda-forge channel"
    ):
        return False
    
    if not run_command(
        "conda config --add channels pytorch",
        "Adding pytorch channel"
    ):
        return False
    
    # Set flexible channel priority to avoid conflicts
    if not run_command(
        "conda config --set channel_priority flexible",
        "Setting channel priority to flexible"
    ):
        return False
    
    print("[INFO] Installing conda packages in tudat-space environment...")
    # Install PyTorch from pytorch channel
    if not run_command(
        "conda install -n tudat-space -c pytorch pytorch -y",
        "Installing PyTorch in tudat-space environment"
    ):
        return False
    
    # Install pygmo from conda-forge
    if not run_command(
        "conda install -n tudat-space -c conda-forge pygmo -y",
        "Installing pygmo in tudat-space environment"
    ):
        return False
    
    # Then install pip requirements
    pip_command = get_conda_pip()
    ok = run_command(
        f"{pip_command} install -r requirements.txt",
        "Installing Python dependencies in tudat-space environment",
    )

    print("[INFO] Checking for Blender in PATH...")
    blender_path = None
    # Try 'where blender' in CMD
    try:
        result = subprocess.run(
            ["cmd", "/c", "where blender"], check=True, capture_output=True, text=True
        )
        blender_path = result.stdout.strip().splitlines()[0]
        print(f"[OK] Blender found at: {blender_path}")
    except subprocess.CalledProcessError:
        # Try 'Get-Command blender' in PowerShell
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    "Get-Command blender | Select-Object -ExpandProperty Source",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            blender_path = result.stdout.strip().splitlines()[0]
            if blender_path:
                print(f"[OK] Blender found at: {blender_path}")
            else:
                raise subprocess.CalledProcessError(1, "Get-Command")
        except subprocess.CalledProcessError:
            print("[ERROR] Blender executable not found in PATH!")
            print(
                "Please add the folder containing 'blender.exe' to your PATH environment variable."
            )
            print(
                "For example, if Blender is installed in C:\\Program Files\\Blender Foundation\\Blender, add that folder to PATH."
            )
            print(
                "After updating PATH, restart your terminal or computer and re-run this setup script."
            )
            return False
    except FileNotFoundError:
        # CMD not found, assume platform is linux
        try:
            result = subprocess.run(
                ["which", "blender"], check=True, capture_output=True, text=True
            )
            blender_path = result.stdout.strip().splitlines()[0]
            if blender_path:
                print(f"[OK] Blender found at: {blender_path}")
            else:
                raise subprocess.CalledProcessError(1, "Get-Command")
        
        except Exception as e:
            print("[ERROR] Blender executable not found!")
            print(
                "Please install Blender and ensure it can be found from your terminal."
            )
            print("Exception was: ", e)
            return False

    # Get Blender's Python executable
    print("[INFO] Locating Blender's bundled Python...")
    try:
        # Run Blender in background to get its Python executable path
        get_py_cmd = f'"{blender_path}" --background --python-expr "import sys; print(sys.executable)"'
        result = subprocess.run(
            get_py_cmd, shell=True, check=True, capture_output=True, text=True
        )
        # Blender prints a lot, so look for the last line containing 'python'
        python_lines = [
            line for line in result.stdout.splitlines() if "python" in line.lower()
        ]
        if not python_lines:
            print("[ERROR] Could not determine Blender's Python executable.")
            return False
        blender_python = python_lines[-1].strip()
        print(f"[OK] Blender's Python found at: {blender_python}")
    except Exception as e:
        print(f"[ERROR] Failed to get Blender's Python executable: {e}")
        return False

    # Install requirements into Blender's Python
    print("[INFO] Installing requirements into Blender's Python environment...")
    # Ensure pip is available in Blender's Python
    ensure_pip_cmd = f'"{blender_python}" -m ensurepip'
    run_command(ensure_pip_cmd, "Ensuring pip is available in Blender's Python")
    # Upgrade pip
    upgrade_pip_cmd = f'"{blender_python}" -m pip install --upgrade pip'
    run_command(upgrade_pip_cmd, "Upgrading pip in Blender's Python")
    # Install requirements
    install_cmd = f'"{blender_python}" -m pip install -r requirements.txt'
    ok_blender = run_command(
        install_cmd, "Installing Python dependencies in Blender's Python"
    )

    return ok and ok_blender


def start_database():
    """Start the PostgreSQL database using Docker Compose."""
    return run_command("docker-compose up -d", "Starting PostgreSQL database")


def initialize_database():
    """Initialize the database tables."""
    python_command = get_conda_python()
    return run_command(
        f"{python_command} -m src.bioinspired.data.init_db",
        "Initializing database tables",
    )


def main():
    """Main setup process."""

    # Check for admin rights and relaunch if not admin
    if os.name == "nt" and not is_admin():
        print("[INFO] Relaunching setup with administrator privileges...")
        relaunch_as_admin()

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
            if os.name == "nt":
                input("Press Enter to exit...")
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
    print(f"   {python_command} your_script.py")
    print("   or activate the environment first:")
    print("   conda activate tudat-space")
    print("   python your_script.py")
    if os.name == "nt":
        input("Press Enter to exit...")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
