# PowerShell activation script for the BioInspired virtual environment

Write-Host "[INFO] Activating BioInspired Virtual Environment" -ForegroundColor Cyan

$venvPath = "venv\Scripts\Activate.ps1"

if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
    
    Write-Host "[INFO] Python version: " -NoNewline -ForegroundColor Yellow
    python --version
    
    Write-Host "[RUNNING] Installed packages:" -ForegroundColor Yellow
    pip list | Select-String "sqlalchemy|psycopg2|numpy|matplotlib"
    
    Write-Host ""
    Write-Host "[OK] Ready to work on BioInspired!" -ForegroundColor Green
    Write-Host "   To deactivate: deactivate" -ForegroundColor White
    Write-Host "   To run examples: python example_usage.py" -ForegroundColor White
} else {
    Write-Host "[ERROR] Virtual environment not found. Run 'python setup.py' first." -ForegroundColor Red
    exit 1
}
