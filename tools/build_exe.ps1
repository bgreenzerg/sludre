param(
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$root = Split-Path -Parent $root
Set-Location $root

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv first: https://docs.astral.sh/uv/"
}

function Invoke-Uv {
    param([string[]]$Args)
    $uvArgs = @()
    if ($env:VIRTUAL_ENV) {
        $uvArgs += "--active"
    }
    $uvArgs += $Args
    & uv @uvArgs
    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed: uv $($uvArgs -join ' ')"
    }
}

Write-Host "Syncing dependencies (build group)..."
Invoke-Uv @("sync", "--group", "build")

if (-not $SkipTests) {
    Write-Host "Running tests..."
    Invoke-Uv @("run", "python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
}

Write-Host "Building Sludre.exe with PyInstaller..."
Invoke-Uv @("run", "--group", "build", "pyinstaller", "--noconfirm", "--clean", "packaging/pyinstaller.spec")

$distDir = Join-Path $root "dist"
$bundleDir = Join-Path $distDir "Sludre"
$zipPath = Join-Path $distDir "Sludre-win64.zip"
if (-not (Test-Path $bundleDir)) {
    throw "Build output not found: $bundleDir"
}
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

Write-Host "Creating zip artifact: $zipPath"
Compress-Archive -Path (Join-Path $bundleDir "*") -DestinationPath $zipPath -Force
Write-Host "Done."
