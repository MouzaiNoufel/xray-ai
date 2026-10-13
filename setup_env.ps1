# One-time setup on Windows. From the project folder in a VS Code terminal (PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
# Options:
#   -Cuda cu126   force one CUDA build (default tries cu130, then cu128, then cu126)
#                 GTX 10xx / older GPUs: use -Cuda cu126
#   -Cpu          no NVIDIA GPU (testing only; training will be very slow)
#   -Python 3.11  pick another Python version installed with the py launcher
param(
    [string]$Cuda = "auto",
    [switch]$Cpu,
    [string]$Python = "3.12"
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
function Check($what) { if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $what" -ForegroundColor Red; exit 1 } }

Write-Host "`n[1/6] Checking Python $Python" -ForegroundColor Cyan
$py = "py"
$found = $true
try { & $py -$Python --version } catch { $found = $false }
if (-not $found -or $LASTEXITCODE -ne 0) {
    Write-Host "Python $Python not found via the 'py' launcher. Install it from python.org (tick 'Add to PATH')." -ForegroundColor Red
    exit 1
}

Write-Host "`n[2/6] Creating .venv" -ForegroundColor Cyan
if (-not (Test-Path ".venv")) { & $py -$Python -m venv .venv; Check "create venv" }
$vpy = ".\.venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip wheel; Check "pip upgrade"

Write-Host "`n[3/6] Installing PyTorch" -ForegroundColor Cyan
if ($Cpu) {
    & $vpy -m pip install torch torchvision; Check "torch install"
} else {
    try { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv } catch {
        Write-Host "nvidia-smi not found: install/update the NVIDIA driver first (nvidia.com/drivers)." -ForegroundColor Yellow
    }
    $builds = if ($Cuda -eq "auto") { @("cu130", "cu128", "cu126") } else { @($Cuda) }
    $ok = $false
    foreach ($b in $builds) {
        Write-Host "Trying PyTorch build $b ..."
        & $vpy -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$b"
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
    if (-not $ok) { Write-Host "FAILED: no PyTorch CUDA build installed. Check https://pytorch.org/get-started/locally/" -ForegroundColor Red; exit 1 }
}

Write-Host "`n[4/6] Installing project + dev requirements" -ForegroundColor Cyan
& $vpy -m pip install -r requirements-dev.txt; Check "requirements"

Write-Host "`n[5/6] Local config and git hooks" -ForegroundColor Cyan
if (-not (Test-Path "configs\local.yaml")) {
    Copy-Item "configs\local.example.yaml" "configs\local.yaml"
    Write-Host "Created configs\local.yaml -> edit it with your dataset paths."
}
if (Test-Path ".git") { & ".\.venv\Scripts\pre-commit.exe" install }

Write-Host "`n[6/6] GPU check" -ForegroundColor Cyan
& $vpy scripts\check_gpu.py
Write-Host "`nDone. In VS Code: Ctrl+Shift+P > 'Python: Select Interpreter' > .venv" -ForegroundColor Green
