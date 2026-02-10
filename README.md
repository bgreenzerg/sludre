<p align="center">
  <img src="assets/sludre_logo.png" alt="Sludre logo" width="180" />
</p>

<h1 align="center">Sludre</h1>
<p align="center">Windows-app til hurtig dansk tale-til-tekst lokalt på CUDA, med valgfri LLM-oprydning og direkte indsættelse ved cursor.</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Windows-0A2540">
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-1F6FEB">
  <img alt="GPU" src="https://img.shields.io/badge/inference-CUDA-3FB950">
  <img alt="UI" src="https://img.shields.io/badge/UI-PySide6-0E7490">
</p>

## Hvad Er Sludre?
Sludre lytter, mens du holder `Ctrl + Space`, transskriberer lokalt med en Whisper-model, rydder teksten op med LLM hvis du ønsker det, og indsætter resultatet dér hvor markøren står.

## Funktioner
- Global hold-to-talk hotkey: `Ctrl + Space`
- Lokal STT med `syvai/hviske-v2` via `faster-whisper`
- Eksplicit `Download model`-knap (ingen auto-download fra Hugging Face)
- Mulighed for manuel modelsti
- Engangskonvertering fra Transformers-format til CTranslate2 ved behov
  - Konverteringsafhængigheder (`transformers`, `torch`) er inkluderet i projekt/build-afhængigheder
- Valgfri LLM-oprydning inden indsættelse
- LLM-providere: OpenAI-kompatibel endpoint og Mistral API
- Prompt-presets (opret/opdatér/slet)
- Ordliste med deterministic replacements og foretrukne termer
- Output-historik i UI med `Kopier`-knap pr. række

## Systemkrav (Brugere)
- Windows 10 eller 11
- NVIDIA GPU med fungerende CUDA-drivere

## Kom Godt I Gang (Anbefalet)
### 1) Vælg installer-type
Nemmeste start (ingen HF-opsætning):
- `Sludre-win64-with-model.zip`
  - Direkte download: [SourceForge (with-model)](https://sourceforge.net/projects/sludre/files/Sludre-win64-with-model.zip/download)
  - Større download
  - Model er allerede med i pakken

Mindre download (model hentes i appen):
- `Sludre-win64-lite.zip`
  - Hent fra: [GitHub Releases](https://github.com/bgreenzerg/sludre/releases)
  - Kræver `HF key` + klik på `Download model`

### 2) Installer
1. Udpak zip-filen til en mappe, fx `C:\Sludre`.
2. Kør `Sludre.exe`.

### 3) Standard placering af modeller (installeret version)
Når du kører den installerede `Sludre.exe`, ligger standardstier relativt til samme mappe som exe'en:
- `models`-mappe: `.\models\`
- Eksempel hvis appen ligger i `C:\Sludre`:
  - Modelmappe: `C:\Sludre\models\`
  - Standard modelsti: `C:\Sludre\models\syvai--hviske-v2`

### 4) Første opsætning i appen
1. Åbn `Indstillinger`.
2. Hvis du bruger `with-model`:
   - Appen er normalt klar med det samme.
3. Hvis du bruger `lite`:
   - Indsæt Hugging Face key i `HF key`.
   - Klik `Download model`.
   - Vent til status er `Ready`.

Du kan følge download i `System log`:
- download-plan (`files` + total size)
- fremdrift (`X%` + bytes)
- detaljerede fejl (CLI/SDK fallback)

### 5) Brug appen
- Hold `Ctrl + Space` for at optage
- Slip for at transskribere og indsætte tekst

## Runtime-filer og stier
Sludre bruger runtime-roden:
- Source-kørsel: repository-roden
- Exe-kørsel: mappen hvor `Sludre.exe` ligger

Følgende filer ligger i runtime-roden:
- `.\.env`
- `.\config.json`
- `.\models\`
- `.\logs\sludre.log`
- `.\wordlist.json`

## Modelopsætning
Standard modelmål:

`.\models\syvai--hviske-v2`

Downloadstrategi når du klikker `Download model`:
1. Hugging Face CLI (`hf` / `huggingface-cli`)
2. Python SDK fallback (`huggingface_hub.snapshot_download`)

Hvis modellen er i Transformers-format (`*.safetensors`), laver Sludre automatisk engangskonvertering til:

`.\models\syvai--hviske-v2\ctranslate2`

Manuel modelsti:
- Kan sættes i `Indstillinger`
- Hvis mappen ikke findes, oprettes den automatisk
- Hvis mappen findes men ikke har gyldig model endnu, bruges den som download-target

## Nøgler Og Hemmeligheder
API-nøgler gemmes i lokal `.env`:
- `HF_TOKEN`
- `LLM_API_KEY`

Plaintext nøgler gemmes ikke i `config.json`.

## Fejlsøgning
### `Download model` gør ingenting
- Tjek at `HF key` er udfyldt
- Tjek `System log` for model/download-linjer

### Fejl med manglende `model.bin`
- Den valgte modelmappe indeholder ikke en komplet model
- Klik `Download model` og lad den køre færdig
- Ryd evt. ufuldstændig modelmappe og prøv igen

### Konverteringsfejl med manglende `transformers` / `torch`
Kør:
```powershell
uv sync
```

### Hugging Face auth/netværksfejl
- Verificér token og repo-adgang på Hugging Face
- Se `System log` for CLI stderr/fallback-detaljer

## Til Udviklere: Kør Fra Source (uv)
Krav:
- Python 3.11+
- `uv` installeret (`https://docs.astral.sh/uv/`)

Installér dependencies:
```powershell
uv sync
```

Start appen:
```powershell
uv run -m src.app
```

## Byg Windows `.exe` (PyInstaller)
### Lokal build
```powershell
.\tools\build_exe.ps1
```

Dette producerer:
- `dist\Sludre\Sludre.exe`
- `dist\Sludre-win64-lite.zip`

Valgfri build med indbygget model:
```powershell
.\tools\build_exe.ps1 -IncludeBundledModel
```

Valgfri build med custom modelkilde:
```powershell
.\tools\build_exe.ps1 -IncludeBundledModel -BundledModelSource "C:\path\to\model-folder"
```

Dette producerer desuden:
- `dist\Sludre-win64-with-model.zip`

Build-note:
- Runtime-state (`config.json`, `.env`, `wordlist.json`, `logs/`) fjernes fra bundle før zip
- Det sikrer en ren pakke uden lokale stier eller nøgler

### CI-build
Workflow:

`.github/workflows/windows-build.yml`

Kører på `v*` tags og manuel dispatch.
Publicerer både `lite` og `with-model`.
`with-model` forventer model i `models/syvai--hviske-v2` i workspace.

## Udvikling
Kør tests:
```powershell
uv run python -m unittest discover -s tests -p "test_*.py"
```

Compile-check:
```powershell
uv run python -m compileall src
```

## Projektstruktur
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
