param(
    [switch]$SkipTests,
    [switch]$IncludeBundledModel,
    [string]$BundledModelSource = "models\syvai--hviske-v2"
)

$ErrorActionPreference = "Stop"
$root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $root

if (-not (Test-Path (Join-Path $root "pyproject.toml"))) {
    throw "pyproject.toml not found at expected project root: $root"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required. Install uv first: https://docs.astral.sh/uv/"
}

function Invoke-Uv {
    param([string[]]$CommandArgs)
    if (-not $CommandArgs -or $CommandArgs.Count -eq 0) {
        throw "Invoke-Uv requires at least one uv command argument."
    }
    $uvArgs = @($CommandArgs)
    $usedActive = $false
    if ($env:VIRTUAL_ENV -and $CommandArgs.Count -ge 1) {
        if ($CommandArgs.Count -eq 1) {
            $uvArgs = @($CommandArgs[0], "--active")
        } else {
            $uvArgs = @($CommandArgs[0], "--active") + $CommandArgs[1..($CommandArgs.Count - 1)]
        }
        $usedActive = $true
    }

    & uv @uvArgs
    if ($LASTEXITCODE -eq 0) {
        return
    }

    if ($usedActive) {
        Write-Warning "uv command with --active failed; retrying without --active."
        & uv @CommandArgs
        if ($LASTEXITCODE -eq 0) {
            return
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw "uv command failed: uv $($CommandArgs -join ' ')"
    }
}

function New-ZipWithPython {
    param(
        [Parameter(Mandatory = $true)][string]$SourceDir,
        [Parameter(Mandatory = $true)][string]$DestinationZip
    )

    $zipScript = @'
import sys
import zipfile
from pathlib import Path

src = Path(sys.argv[1]).resolve()
dst = Path(sys.argv[2]).resolve()

if not src.exists():
    raise FileNotFoundError(f"Source directory not found: {src}")
if dst.exists():
    dst.unlink()

with zipfile.ZipFile(dst, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
    for path in src.rglob("*"):
        if path.is_file():
            zf.write(path, path.relative_to(src).as_posix())

print(f"Zip created: {dst}")
'@
    $zipScript | uv run python - $SourceDir $DestinationZip
    if ($LASTEXITCODE -ne 0) {
        throw "Zip creation failed: $DestinationZip"
    }
}

function Clear-BundleRuntimeState {
    param(
        [Parameter(Mandatory = $true)][string]$BundleDir,
        [switch]$RemoveModels
    )

    $items = @(".env", "config.json", "wordlist.json", "logs")
    if ($RemoveModels) {
        $items += "models"
    }
    foreach ($name in $items) {
        $path = Join-Path $BundleDir $name
        if (Test-Path $path) {
            Remove-Item $path -Recurse -Force
        }
    }
}

Write-Host "Syncing dependencies (build group)..."
Invoke-Uv -CommandArgs @("sync", "--group", "build")

if (-not $SkipTests) {
    Write-Host "Running tests..."
    Invoke-Uv -CommandArgs @("run", "python", "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py")
}

Write-Host "Building Sludre.exe with PyInstaller..."
if (Test-Path (Join-Path $root "dist\Sludre")) {
    Remove-Item (Join-Path $root "dist\Sludre") -Recurse -Force
}
Invoke-Uv -CommandArgs @("run", "--group", "build", "pyinstaller", "--noconfirm", "--clean", "packaging/pyinstaller.spec")

$distDir = Join-Path $root "dist"
$bundleDir = Join-Path $distDir "Sludre"
$liteZipPath = Join-Path $distDir "Sludre-win64-lite.zip"
$withModelZipPath = Join-Path $distDir "Sludre-win64-with-model.zip"
if (-not (Test-Path $bundleDir)) {
    throw "Build output not found: $bundleDir"
}
if (Test-Path $liteZipPath) {
    Remove-Item $liteZipPath -Force
}

Clear-BundleRuntimeState -BundleDir $bundleDir -RemoveModels

Write-Host "Creating lite zip artifact: $liteZipPath"
New-ZipWithPython -SourceDir $bundleDir -DestinationZip $liteZipPath

if ($IncludeBundledModel) {
    $modelSourcePath = $BundledModelSource
    if (-not [System.IO.Path]::IsPathRooted($modelSourcePath)) {
        $modelSourcePath = Join-Path $root $modelSourcePath
    }
    if (-not (Test-Path $modelSourcePath)) {
        throw "Bundled model source not found: $modelSourcePath"
    }
    $modelSourcePath = (Resolve-Path $modelSourcePath).Path
    $modelDestRoot = Join-Path $bundleDir "models"
    New-Item -ItemType Directory -Path $modelDestRoot -Force | Out-Null
    $modelDestPath = Join-Path $modelDestRoot (Split-Path $modelSourcePath -Leaf)
    if (Test-Path $modelDestPath) {
        Remove-Item $modelDestPath -Recurse -Force
    }
    Write-Host "Bundling local model from: $modelSourcePath"
    Copy-Item -Path $modelSourcePath -Destination $modelDestRoot -Recurse -Force
    Clear-BundleRuntimeState -BundleDir $bundleDir

    if (Test-Path $withModelZipPath) {
        Remove-Item $withModelZipPath -Force
    }
    Write-Host "Creating with-model zip artifact: $withModelZipPath"
    New-ZipWithPython -SourceDir $bundleDir -DestinationZip $withModelZipPath
}

Write-Host "Done."
