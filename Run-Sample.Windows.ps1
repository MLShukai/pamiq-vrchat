# PAMIQ VRChat Sample Runner for Windows
# This script automatically sets up dependencies and runs the PAMIQ VRChat sample

# Set error action preference
$ErrorActionPreference = "Stop"

# Helper functions for colored output
function Write-Error-Custom {
    param([string]$Message)
    Write-Host "❌ ERROR: $Message" -ForegroundColor Red
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning-Custom {
    param([string]$Message)
    Write-Host "⚠️  WARNING: $Message" -ForegroundColor Yellow
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor Blue
}

# Check if command exists
function Test-Command {
    param([string]$CommandName)
    try {
        Get-Command $CommandName -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

# Check if process is running
function Test-Process {
    param([string]$ProcessName)
    try {
        $processes = Get-Process -Name $ProcessName -ErrorAction Stop
        return $processes.Count -gt 0
    }
    catch {
        return $false
    }
}

# =============================================================================
#                              DEPENDENCY RESOLUTION
# =============================================================================

Write-Info "Checking dependencies..."

# 1. Check and install uv
if (-not (Test-Command "uv")) {
    Write-Warning-Custom "uv is not installed."
    $response = Read-Host "Do you want to install uv? (y/N)"
    if ($response -match "^[Yy]$") {
        Write-Info "Installing uv..."
        try {
            # Download and install uv using the official installer
            Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
            if (-not $?) {
                throw "Installation failed"
            }

            # Refresh PATH for current session
            $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "User") + ";" + [System.Environment]::GetEnvironmentVariable("PATH", "Machine")

            Write-Success "uv installed successfully"
        }
        catch {
            Write-Error-Custom "Failed to install uv: $($_.Exception.Message)"
            exit 1
        }
    }
    else {
        Write-Error-Custom "uv installation was declined. Cannot proceed."
        exit 1
    }
}

# Verify uv is accessible
if (-not (Test-Command "uv")) {
    Write-Error-Custom "uv is still not accessible after installation. Please restart PowerShell."
    exit 1
}

Write-Success "uv is available"

# 2. Install dependencies
Write-Info "Installing Python dependencies..."
try {
    & uv sync --all-extras
    & uv pip install torch -U --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) {
        throw "uv sync failed with exit code $LASTEXITCODE"
    }
}
catch {
    Write-Error-Custom "Failed to install dependencies with uv sync --all-extras"
    exit 1
}
Write-Success "Dependencies installed"

# 3. Check CUDA availability using nvidia-smi
Write-Info "Checking CUDA availability..."
try {
    # Check PyTorch CUDA availability
    $cudaCheck = & uv run python -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaCheck -ne "True") {
        throw "PyTorch CUDA is not available. torch.cuda.is_available() returned: $cudaCheck"
    }

    Write-Success "CUDA is available"
}
catch {
    Write-Error-Custom "CUDA is not available: $($_.Exception.Message)"
    Write-Error-Custom "Please ensure you have:"
    Write-Error-Custom "1. A CUDA-compatible NVIDIA GPU"
    Write-Error-Custom "2. NVIDIA drivers installed"
    Write-Error-Custom "3. PyTorch with CUDA support"
    exit 1
}

# =============================================================================
#                          SOFTWARE STARTUP VERIFICATION
# =============================================================================

Write-Info "Checking required software..."

# 1. Check VRChat process
if (-not (Test-Process "VRChat")) {
    Write-Warning-Custom "VRChat.exe process not found!"
    Write-Warning-Custom "Please start VRChat from Steam before running this script."
    Write-Warning-Custom "Launch Steam → Library → VRChat → Play"
}
else {
    Write-Success "VRChat is running"
}

# 2. Check OBS
$obsProcesses = @("obs64", "obs32", "obs")
$obsRunning = $false
foreach ($obsProcess in $obsProcesses) {
    if (Test-Process $obsProcess) {
        $obsRunning = $true
        break
    }
}

if (-not $obsRunning) {
    Write-Warning-Custom "OBS is not running!"
    Write-Warning-Custom "Please:"
    Write-Warning-Custom "1. Install OBS Studio if not installed"
    Write-Warning-Custom "2. Start OBS"
    Write-Warning-Custom "3. Set up VRChat window capture"
    Write-Warning-Custom "4. Enable Virtual Camera (Start Virtual Camera button)"
}
else {
    Write-Success "OBS is running"
}

# =============================================================================
#                                    LAUNCH
# =============================================================================

# 1. Start pamiq-kbctl in a new window
Write-Info "Starting pamiq-kbctl in background..."

if (Test-Process "pamiq-kbctl") {
    Write-Warning-Custom "pamiq-kbctl is already running"
}
else {
    try {
        # Start pamiq-kbctl in a new PowerShell window
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "& uv run pamiq-kbctl"
        Write-Success "pamiq-kbctl started in new window"
    }
    catch {
        Write-Warning-Custom "Failed to start pamiq-kbctl: $($_.Exception.Message)"
    }
}

# 2. Model size selection
Write-Host ""
Write-Info "Model size selection:"
Write-Host "Available sizes:"
Write-Host "  1) tiny    - ~2.5GiB VRAM (recommended for most systems)"
Write-Host "  2) small   - ~4GiB VRAM"
Write-Host "  3) medium  - ~6.5GiB VRAM"
Write-Host "  4) large   - ~12GiB VRAM"
Write-Host "  5) huge    - ~23GiB VRAM"
Write-Host "  6) [Enter] - Use default (tiny)"
Write-Host ""

$selection = Read-Host "Select model size (1-6 or Enter for default)"

$modelSize = ""
switch ($selection) {
    "1" { $modelSize = "tiny" }
    "2" { $modelSize = "small" }
    "3" { $modelSize = "medium" }
    "4" { $modelSize = "large" }
    "5" { $modelSize = "huge" }
    { $_ -in @("6", "") } { $modelSize = "" }
    default {
        Write-Warning-Custom "Invalid selection, using default (tiny)"
        $modelSize = ""
    }
}

# 3. Run the sample
Write-Info "Starting PAMIQ VRChat sample..."

try {
    if ($modelSize) {
        Write-Info "Running with model size: $modelSize"
        & uv run python src/run_sample.py --model-size $modelSize
    }
    else {
        Write-Info "Running with default model size"
        & uv run python src/run_sample.py
    }

    if ($LASTEXITCODE -eq 0) {
        Write-Success "PAMIQ VRChat sample completed"
    }
    else {
        Write-Error-Custom "PAMIQ VRChat sample failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}
catch {
    Write-Error-Custom "Failed to run PAMIQ VRChat sample: $($_.Exception.Message)"
    exit 1
}

# Keep window open if running interactively
if ($Host.Name -eq "ConsoleHost") {
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
