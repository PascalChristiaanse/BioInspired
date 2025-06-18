#!/bin/bash
# Activation script for the BioInspired project using tudat-space conda environment

echo "🧬 Activating BioInspired with tudat-space conda environment"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda is not installed or not in PATH."
    echo "Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if tudat-space environment exists
if conda env list | grep -q "tudat-space"; then
    echo "[OK] tudat-space conda environment found"
    
    # Activate the environment
    eval "$(conda shell.bash hook)"
    conda activate tudat-space
    
    if [ "$CONDA_DEFAULT_ENV" = "tudat-space" ]; then
        echo "[OK] tudat-space environment activated"
    else
        echo "[ERROR] Failed to activate tudat-space environment"
        exit 1
    fi
else
    echo "[ERROR] tudat-space conda environment not found."
    echo "Please install TudatPy following the guide:"
    echo "https://docs.tudat.space/en/stable/_src_getting_started/_src_installation.html"
    exit 1
fi

echo "📊 Python version: $(python --version)"
echo "📦 Key installed packages:"
conda list | grep -E "(sqlalchemy|psycopg2|numpy|matplotlib|tudat)" || echo "   (checking packages...)"

echo ""
echo "🚀 Ready to work on BioInspired!"
echo "   Environment: $CONDA_DEFAULT_ENV"
echo "   To deactivate: conda deactivate"
echo "   To run examples: python example_usage.py"
