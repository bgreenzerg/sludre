# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules


project_root = Path(SPECPATH).parent
assets_root = project_root / "assets"

datas = []
for name in ("sludre_logo.png", "sludre_background_stick.png", "background.jpg"):
    path = assets_root / name
    if path.exists():
        datas.append((str(path), "assets"))
# Include faster-whisper runtime assets (Silero VAD ONNX model, etc.).
datas += collect_data_files("faster_whisper", includes=["assets/*"])

hiddenimports = []
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += ["transformers", "torch"]


a = Analysis(
    [str(project_root / "src" / "app.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Sludre",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Sludre",
)
