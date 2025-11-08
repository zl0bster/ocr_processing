param(
    [string]$PythonExe = "python",
    [string]$EnvDir = "venv"
)

Write-Host "Creating virtual environment '$EnvDir' using '$PythonExe'..."
if (-not (Test-Path $EnvDir)) {
    & $PythonExe -m venv $EnvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create virtual environment."
        exit 1
    }
    Write-Host "Virtual environment created."
}
else {
    Write-Host "Virtual environment already exists. Skipping creation."
}

$activateScript = Join-Path $EnvDir "Scripts\\Activate.ps1"

Write-Host ""
Write-Host "To activate the virtual environment, run:"
Write-Host "    . $activateScript"

$pythonInEnv = Join-Path $EnvDir "Scripts\\python.exe"
if (-not (Test-Path $pythonInEnv)) {
    Write-Error "Python executable not found in the virtual environment."
    exit 1
}

Write-Host ""
Write-Host "Upgrading pip and installing dependencies from requirements.txt..."
& $pythonInEnv -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to upgrade pip."
    exit 1
}

& $pythonInEnv -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install dependencies."
    exit 1
}

Write-Host ""
Write-Host "Environment setup complete."

