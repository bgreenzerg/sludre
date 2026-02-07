# Sludre (Windows, Python, CUDA)

Windows app with GUI, global `Ctrl+Space` hold-to-talk, local STT model on CUDA, optional LLM cleanup, and direct cursor insertion.

## Features
- Global hotkey: hold `Ctrl+Space` to record.
- Local transcription with `syvai/hviske-v2`.
- Auto model download to `.\models` or optional manual model folder.
- Clean UI split into tabs:
  - `Sludre` tab for recording and output history.
  - `Indstillinger` tab for all configuration.
- Optional LLM cleanup step before insertion:
  - OpenAI-compatible provider.
  - Mistral API provider.
- Multiple named system prompt presets (save/select/delete by name).
- Custom wordlist:
  - Replacement rules.
  - Preferred terms added to LLM prompt.
- Output history table containing transcription + final output, with `Kopier` button per entry.
- Styled UI theme with cinematic background image (`assets/background.jpg`).
- Detailed runtime logs in `.\logs\sludre.log`.

## Requirements
- Windows 10/11
- NVIDIA GPU with working CUDA drivers
- Python 3.11+

## Installation
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run
```powershell
python -m src.app
```

## Model download notes
- First startup can take a long time (several GB).
- Download target: `.\models\syvai--hviske-v2`.
- If the model is private, set HF token in app settings.
- Downloader uses Hugging Face CLI first (`hf` / `huggingface-cli`) and falls back to Python SDK.
- If downloaded model is Hugging Face Transformers format (`*.safetensors`), app auto-converts it once to CTranslate2 in `.\models\syvai--hviske-v2\ctranslate2`.
- Auto-conversion requires `transformers` and `torch` installed. If missing, app logs a clear install hint.

## LLM cleanup notes
- LLM cleanup is off by default.
- Configure provider, API key, model, timeout, and system prompt in the app.
- On LLM failure, app shows a popup:
  - Insert raw transcription, or
  - Cancel insertion.

## Wordlist notes
- Default wordlist file: `.\wordlist.json`.
- Edit it from the app ("Edit wordlist").
- You can enable/disable:
  - rule-based replacements
  - preferred terms in LLM prompt

## Logging
- File logs: `.\logs\sludre.log`.
- UI log panel mirrors important runtime events.
- API keys are never written to logs.

## Visual Customization
- Default background image: `assets/background.jpg`.
- Replace that file with your own image (same filename) to customize the app look.
