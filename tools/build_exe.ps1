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
    $uvArgs = @($Args)
    $usedActive = $false
    if ($env:VIRTUAL_ENV -and $Args.Count -ge 1) {
        if ($Args.Count -eq 1) {
            $uvArgs = @($Args[0], "--active")
        } else {
            $uvArgs = @($Args[0], "--active") + $Args[1..($Args.Count - 1)]
        }
        $usedActive = $true
    }

    & uv @uvArgs
    if ($LASTEXITCODE -eq 0) {
        return
    }

    if ($usedActive) {
        Write-Warning "uv command with --active failed; retrying without --active."
        & uv @Args
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed: uv $($Args -join ' ')"
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
