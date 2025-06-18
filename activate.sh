#!/bin/bash
# Activation script for the BioInspired virtual environment

echo "🧬 Activating BioInspired Virtual Environment"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "[OK] Virtual environment activated (Unix/Linux/Mac)"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo "[OK] Virtual environment activated (Windows Git Bash)"
else
    echo "[ERROR] Virtual environment not found. Run 'python setup.py' first."
    exit 1
fi

echo "📊 Python version: $(python --version)"
echo "📦 Installed packages:"
pip list | grep -E "(sqlalchemy|psycopg2|numpy|matplotlib)"

echo ""
echo "🚀 Ready to work on BioInspired!"
echo "   To deactivate: deactivate"
echo "   To run examples: python example_usage.py"
