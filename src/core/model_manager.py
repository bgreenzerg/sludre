from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path


INFERENCE_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "normalizer.json",
]

CONVERTER_COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "normalizer.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def _snapshot_download(*args, **kwargs) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(*args, **kwargs)


def _subprocess_run(*args, **kwargs):
    import subprocess

    return subprocess.run(*args, **kwargs)


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _create_transformers_converter(*args, **kwargs):
    from ctranslate2.converters import TransformersConverter

    return TransformersConverter(*args, **kwargs)


class ModelManager:
    def __init__(
        self,
        repo_id: str,
        cache_dir: Path,
        manual_model_path: str | None = None,
        hf_token: str | None = None,
        log_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.repo_id = repo_id
        self.cache_dir = cache_dir
        self.manual_model_path = manual_model_path or ""
        self.hf_token = hf_token or None
        self.log_callback = log_callback
        self.logger = logging.getLogger("sludre.model_manager")

    def _log_info(self, message: str, notify_ui: bool = False) -> None:
        self.logger.info(message)
        if notify_ui and self.log_callback:
            self.log_callback(message)

    def _log_warning(self, message: str, notify_ui: bool = False) -> None:
        self.logger.warning(message)
        if notify_ui and self.log_callback:
            self.log_callback(message)

    def _download_dir(self) -> Path:
        safe_name = self.repo_id.replace("/", "--")
        return self.cache_dir / safe_name

    def download_dir(self) -> Path:
        return self._download_dir()

    def _download_snapshot(self, target_dir: Path) -> Path:
        self._log_info(
            f"Starting Python SDK fallback download to: {target_dir}",
            notify_ui=True,
        )
        base_kwargs = {
            "repo_id": self.repo_id,
            "local_dir": str(target_dir),
            "local_files_only": False,
            "token": self.hf_token,
            "resume_download": True,
            "etag_timeout": 30,
            "max_workers": 4,
            "local_dir_use_symlinks": False,
            "allow_patterns": INFERENCE_ALLOW_PATTERNS,
        }
        try:
            model_path = _snapshot_download(**base_kwargs)
            self._log_info("Python SDK fallback download completed.", notify_ui=True)
            return Path(model_path)
        except TypeError:
            # Compatibility for huggingface_hub versions that removed/renamed args.
            fallback_kwargs = dict(base_kwargs)
            fallback_kwargs.pop("local_dir_use_symlinks", None)
            fallback_kwargs.pop("resume_download", None)
            model_path = _snapshot_download(**fallback_kwargs)
            self._log_info(
                "Python SDK fallback download completed with compatibility args.",
                notify_ui=True,
            )
            return Path(model_path)

    def _hf_cli_command(self, target_dir: Path) -> list[str]:
        command = self._candidate_cli_bases()[0] + [
            "download",
            self.repo_id,
            "--local-dir",
            str(target_dir),
            "--max-workers",
            "4",
            "--include",
            *INFERENCE_ALLOW_PATTERNS,
        ]
        if self.hf_token:
            command.extend(["--token", self.hf_token])
        return command

    @staticmethod
    def _candidate_cli_bases() -> list[list[str]]:
        exe_dir = Path(sys.executable).resolve().parent
        candidates = [
            [str(exe_dir / "hf.exe")],
            [str(exe_dir / "hf")],
            [str(exe_dir / "huggingface-cli.exe")],
            [str(exe_dir / "huggingface-cli")],
            ["hf"],
            ["huggingface-cli"],
            [sys.executable, "-m", "huggingface_hub.cli.hf"],
            [sys.executable, "-m", "huggingface_hub.commands.huggingface_cli"],
        ]
        unique: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        for cmd in candidates:
            key = tuple(cmd)
            if key in seen:
                continue
            unique.append(cmd)
            seen.add(key)
        return unique

    def _build_cli_command(self, cli_base: list[str], target_dir: Path) -> list[str]:
        command = cli_base + [
            "download",
            self.repo_id,
            "--local-dir",
            str(target_dir),
            "--max-workers",
            "4",
            "--include",
            *INFERENCE_ALLOW_PATTERNS,
        ]
        if self.hf_token:
            command.extend(["--token", self.hf_token])
        return command

    @staticmethod
    def _looks_like_missing_entrypoint(message: str) -> bool:
        lowered = message.lower()
        patterns = [
            "no module named",
            "error while finding module specification",
            "not recognized as an internal or external command",
            "is not recognized as an internal or external command",
            "no such file or directory",
            "cannot find the file specified",
        ]
        return any(pattern in lowered for pattern in patterns)

    @staticmethod
    def _format_command_for_log(command: list[str], token: str | None) -> str:
        if not token:
            return " ".join(command)
        return " ".join(part if part != token else "***REDACTED***" for part in command)

    @staticmethod
    def _has_required_model_files(target_dir: Path) -> bool:
        if not (target_dir / "config.json").exists():
            return False
        if (target_dir / "model.safetensors.index.json").exists():
            return True
        return any(target_dir.glob("model-*.safetensors"))

    @staticmethod
    def _is_ctranslate2_model(path: Path) -> bool:
        return (path / "model.bin").exists()

    @staticmethod
    def _looks_like_transformers_whisper_model(path: Path) -> bool:
        if not (path / "config.json").exists():
            return False
        if (path / "model.safetensors.index.json").exists():
            return True
        return (path / "model.safetensors").exists() or any(
            path.glob("model-*.safetensors")
        )

    def _ensure_runtime_model_format(self, model_dir: Path) -> Path:
        if self._is_ctranslate2_model(model_dir):
            self._log_info(f"CTranslate2 model detected: {model_dir}")
            return model_dir

        if not self._looks_like_transformers_whisper_model(model_dir):
            self._log_warning(
                "Model folder does not look like either CTranslate2 or "
                f"Transformers Whisper: {model_dir}",
                notify_ui=True,
            )
            return model_dir

        converted_dir = model_dir / "ctranslate2"
        if self._is_ctranslate2_model(converted_dir):
            self._log_info(
                f"Using cached converted CTranslate2 model: {converted_dir}",
                notify_ui=True,
            )
            return converted_dir

        missing = [
            name for name in ("transformers", "torch") if not _module_available(name)
        ]
        if missing:
            missing_joined = ", ".join(missing)
            raise RuntimeError(
                "Downloaded model is in Transformers format and must be converted "
                "to CTranslate2 before faster-whisper can load it.\n"
                f"Missing dependencies for conversion: {missing_joined}\n"
                "Install and retry: pip install transformers torch"
            )

        self._log_info(
            "Transformers model detected. Starting one-time conversion to "
            f"CTranslate2 in: {converted_dir}",
            notify_ui=True,
        )
        if converted_dir.exists() and not self._is_ctranslate2_model(converted_dir):
            self._log_warning(
                "Found existing incomplete conversion directory. "
                f"Resetting: {converted_dir}",
                notify_ui=True,
            )
            shutil.rmtree(converted_dir, ignore_errors=True)
        converted_dir.mkdir(parents=True, exist_ok=True)
        try:
            converter = _create_transformers_converter(
                str(model_dir),
                copy_files=CONVERTER_COPY_FILES,
                load_as_float16=True,
            )
            converter.convert(
                str(converted_dir),
                quantization="float16",
                force=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to convert model to CTranslate2 format. "
                f"Details: {exc}"
            ) from exc

        if not self._is_ctranslate2_model(converted_dir):
            raise RuntimeError(
                "Conversion finished but model.bin was not created in "
                f"{converted_dir}"
            )
        self._log_info(
            f"CTranslate2 conversion completed successfully: {converted_dir}",
            notify_ui=True,
        )
        return converted_dir

    def _download_with_cli(self, target_dir: Path) -> Path:
        env = os.environ.copy()
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        self._log_info("Starting Hugging Face CLI download...", notify_ui=True)
        entrypoint_errors: list[str] = []
        for cli_base in self._candidate_cli_bases():
            command = self._build_cli_command(cli_base, target_dir)
            cli_name = " ".join(cli_base)
            self._log_info(
                f"Trying CLI entrypoint: {cli_name}",
                notify_ui=True,
            )
            self._log_info(
                f"CLI command: {self._format_command_for_log(command, self.hf_token)}"
            )
            try:
                result = self._run_with_heartbeat(command, env)
            except FileNotFoundError as exc:
                msg = f"{cli_name}: not found ({exc})"
                self._log_warning(msg)
                entrypoint_errors.append(msg)
                continue

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            if stdout:
                self._log_info(f"CLI stdout:\n{stdout}")
            if stderr:
                self._log_warning(f"CLI stderr:\n{stderr}")
            self._log_info(
                f"{cli_name} finished with return code {result.returncode}",
                notify_ui=True,
            )

            if result.returncode == 0:
                if not self._has_required_model_files(target_dir):
                    raise RuntimeError(
                        "Hugging Face CLI finished but required model files "
                        "are missing."
                    )
                self._log_info(
                    f"Required model files detected in: {target_dir}",
                    notify_ui=True,
                )
                return target_dir

            details = stderr or stdout or "No output."
            if self._looks_like_missing_entrypoint(details):
                msg = f"{cli_name}: unavailable ({details})"
                entrypoint_errors.append(msg)
                self._log_warning(msg)
                continue
            raise RuntimeError(f"{cli_name} failed: {details}")

        joined = "\n".join(entrypoint_errors) if entrypoint_errors else "No output."
        raise RuntimeError(
            "No compatible Hugging Face CLI entrypoint worked.\n"
            f"Details:\n{joined}"
        )

    def _run_with_heartbeat(self, command: list[str], env: dict[str, str]):
        state: dict[str, object] = {}
        error: dict[str, Exception] = {}
        started = time.monotonic()

        def worker() -> None:
            try:
                state["result"] = _subprocess_run(
                    command,
                    check=False,
                    text=True,
                    capture_output=True,
                    env=env,
                )
            except Exception as exc:
                error["exception"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while thread.is_alive():
            thread.join(timeout=10.0)
            if thread.is_alive():
                elapsed = int(time.monotonic() - started)
                self._log_info(
                    f"huggingface-cli still running... elapsed {elapsed}s",
                    notify_ui=True,
                )
        if "exception" in error:
            raise error["exception"]
        result = state.get("result")
        if result is None:
            raise RuntimeError("No subprocess result returned.")
        return result

    def ensure_model_available(self) -> Path:
        if self.manual_model_path:
            model_path = Path(self.manual_model_path).expanduser()
            if model_path.exists():
                self._log_info(
                    f"Using manual model path: {model_path}",
                    notify_ui=True,
                )
                return self._ensure_runtime_model_format(model_path)
            raise FileNotFoundError(
                f"Manual model path does not exist: {model_path}"
            )

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        target_dir = self._download_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        self._log_info(f"Model target directory: {target_dir}", notify_ui=True)
        try:
            downloaded = self._download_with_cli(target_dir)
        except Exception as cli_exc:
            self._log_warning(
                f"huggingface-cli download failed. Falling back to Python SDK. "
                f"Error: {cli_exc}",
                notify_ui=True,
            )
            try:
                downloaded = self._download_snapshot(target_dir)
            except Exception as sdk_exc:
                raise RuntimeError(
                    "Model download failed with both huggingface-cli and "
                    f"Python SDK fallback.\nCLI error: {cli_exc}\nSDK error: {sdk_exc}"
                ) from sdk_exc
        return self._ensure_runtime_model_format(downloaded)
