<p align="center">
  <img src="assets/sludre_logo.png" alt="Sludre logo" width="180" />
</p>

<h1 align="center">Sludre</h1>
<p align="center">Windows desktop app for fast Danish speech-to-text on local CUDA, with optional LLM cleanup and direct cursor insertion.</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-0A2540">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-1F6FEB">
  <img alt="GPU" src="https://img.shields.io/badge/inference-CUDA-3FB950">
  <img alt="UI" src="https://img.shields.io/badge/UI-PySide6-0E7490">
</p>

## What Sludre Does
Sludre listens while you hold `Ctrl + Space`, transcribes locally with a Whisper-family model, optionally cleans the text with an LLM, and pastes the final result where your cursor is active.

## Core Features
- Global hold-to-talk hotkey: `Ctrl + Space`.
- Local STT with `syvai/hviske-v2` on CUDA via `faster-whisper`.
- Explicit model download button (no auto-download from Hugging Face).
- Manual model path support.
- One-time conversion from Transformers Whisper format to CTranslate2 when needed.
- Optional LLM cleanup pipeline before insert.
- LLM providers: OpenAI-compatible endpoint and Mistral API.
- Named system prompt presets (create/update/delete).
- Custom wordlist with deterministic replacements and preferred term injection.
- Output history in UI with per-entry `Kopier` button.

## Requirements
- Windows 10 or 11
- NVIDIA GPU with working CUDA drivers
- Python 3.11+
- `uv` installed (`https://docs.astral.sh/uv/`)

## Quick Start (Source with uv)
```powershell
uv sync
uv run -m src.app
```

## Project Runtime Paths
Sludre resolves paths from the active runtime root:
- Source run: repository root
- Frozen `.exe`: folder containing `Sludre.exe`

This means the following files are local and visible next to your source or exe bundle:
- secrets: `.\.env`
- config: `.\config.json`
- models: `.\models\`
- logs: `.\logs\sludre.log`
- wordlist: `.\wordlist.json`

## Model Setup
Default model target:

`.\models\syvai--hviske-v2`

Download strategy after clicking `Download model` in settings:
1. Hugging Face CLI (`hf` / `huggingface-cli`)
2. Python SDK fallback (`huggingface_hub.snapshot_download`)

If the model is Transformers format (`*.safetensors`), Sludre auto-converts once to:

`.\models\syvai--hviske-v2\ctranslate2`

Manual model folder can always be configured in `Indstillinger`.

## Secret Storage
API keys are stored in project-local `.env`:
- `HF_TOKEN`
- `LLM_API_KEY`

Plaintext secrets are not persisted in `config.json`.

## Build Windows `.exe` (PyInstaller)
### Local build
```powershell
.\tools\build_exe.ps1
```

This produces:
- `dist\Sludre\Sludre.exe`
- `dist\Sludre-win64.zip`

### CI build
GitHub Actions workflow is included at:

`.github/workflows/windows-build.yml`

It runs on `v*` tags and manual dispatch.

## Development
Run tests:
```powershell
uv run python -m unittest discover -s tests -p "test_*.py"
```

Compile check:
```powershell
uv run python -m compileall src
```

## Project Structure
```text
.github/workflows/windows-build.yml
packaging/pyinstaller.spec
tools/build_exe.ps1
src/
  app.py
  core/
  ui/
assets/
tests/
pyproject.toml
```
