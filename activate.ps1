# PowerShell activation script for the BioInspired project using tudat-space conda environment

Write-Host "🧬 Activating BioInspired with tudat-space conda environment" -ForegroundColor Cyan

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Conda is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Anaconda or Miniconda first." -ForegroundColor Red
    exit 1
}

# Check if tudat-space environment exists
$envList = conda env list
if ($envList -match "tudat-space") {
    Write-Host "[OK] tudat-space conda environment found" -ForegroundColor Green
    
    # Activate the environment
    conda activate tudat-space
    
    if ($env:CONDA_DEFAULT_ENV -eq "tudat-space") {
        Write-Host "[OK] tudat-space environment activated" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to activate tudat-space environment" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "[ERROR] tudat-space conda environment not found." -ForegroundColor Red
    Write-Host "Please install TudatPy following the guide:" -ForegroundColor Red
    Write-Host "https://docs.tudat.space/en/stable/_src_getting_started/_src_installation.html" -ForegroundColor Red
    exit 1
}

Write-Host "[INFO] Python version: " -NoNewline -ForegroundColor Yellow
python --version

Write-Host "[INFO] Key installed packages:" -ForegroundColor Yellow
$packages = conda list | Select-String "sqlalchemy|psycopg2|numpy|matplotlib|tudat"
if ($packages) {
    $packages
} else {
    Write-Host "   (checking packages...)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "🚀 Ready to work on BioInspired!" -ForegroundColor Green
Write-Host "   Environment: $env:CONDA_DEFAULT_ENV" -ForegroundColor White
Write-Host "   To deactivate: conda deactivate" -ForegroundColor White
Write-Host "   To run examples: python example_usage.py" -ForegroundColor White
