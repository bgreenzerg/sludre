from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QLinearGradient, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from src.core.audio_capture import AudioCapture
from src.core.app_logging import default_log_file
from src.core.config import AppConfig, ConfigStore, DEFAULT_LLM_SYSTEM_PROMPT, MISTRAL_MODEL_PRESETS
from src.core.env_secrets import EnvSecretsStore
from src.core.hotkey_controller import HotkeyController
from src.core.llm_refiner import LlmRefiner
from src.core.model_manager import ModelManager
from src.core.pipeline import PostProcessResult, TranscriptionPostProcessor
from src.core.text_inserter import TextInserter
from src.core.transcriber import Transcriber
from src.core.wordlist_store import ReplacementRule, WordlistData, WordlistStore

THEME = {
    "bg_fallback": "#071521",
    "card": "rgba(8, 16, 28, 195)",
    "card_soft": "rgba(8, 16, 28, 150)",
    "text": "#ECF2FF",
    "muted": "#A9BBD6",
    "accent": "#32D27C",
    "accent_hover": "#22B56A",
    "border": "rgba(165, 190, 220, 90)",
    "input_bg": "rgba(4, 12, 22, 180)",
}


class StyledBackgroundWidget(QWidget):
    def __init__(self, image_path: Path | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self._pixmap = QPixmap(str(image_path)) if image_path and image_path.exists() else QPixmap()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        rect = self.rect()
        painter.fillRect(rect, QColor(THEME["bg_fallback"]))

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                rect.size(),
                Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (rect.width() - scaled.width()) // 2
            y = (rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        overlay = QLinearGradient(0, 0, 0, rect.height())
        overlay.setColorAt(0.0, QColor(4, 10, 20, 128))
        overlay.setColorAt(0.6, QColor(4, 10, 20, 192))
        overlay.setColorAt(1.0, QColor(4, 10, 20, 234))
        painter.fillRect(rect, overlay)

        # Decorative ambient circles for depth.
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(49, 126, 255, 40))
        painter.drawEllipse(int(rect.width() * 0.66), -130, 420, 420)
        painter.setBrush(QColor(50, 210, 124, 32))
        painter.drawEllipse(-160, int(rect.height() * 0.54), 450, 450)


class UiBridge(QObject):
    status_changed = Signal(str)
    log_message = Signal(str)
    ready_changed = Signal(bool)
    postprocess_ready = Signal(object)
    postprocess_failed = Signal(str)


class WordlistEditorDialog(QDialog):
    def __init__(self, initial: WordlistData, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Wordlist")
        self.resize(720, 460)
        self._data = initial
        layout = QVBoxLayout(self)
        self.rules = QTableWidget(0, 4, self)
        self.rules.setHorizontalHeaderLabels(["From", "To", "Match Case", "Whole Word"])
        self.rules.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.rules)
        action = QHBoxLayout()
        add = QPushButton("Add Rule")
        rem = QPushButton("Remove Selected")
        add.clicked.connect(self._add)
        rem.clicked.connect(self._remove)
        action.addWidget(add)
        action.addWidget(rem)
        layout.addLayout(action)
        self.terms = QPlainTextEdit(self)
        self.terms.setPlaceholderText("Preferred terms, one per line")
        layout.addWidget(self.terms)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        for rule in self._data.replacements:
            self._add(rule)
        self.terms.setPlainText("\n".join(self._data.preferred_terms))

    def _add(self, rule: ReplacementRule | None = None) -> None:
        r = rule if isinstance(rule, ReplacementRule) else ReplacementRule("", "")
        row = self.rules.rowCount()
        self.rules.insertRow(row)
        self.rules.setItem(row, 0, QTableWidgetItem(r.source))
        self.rules.setItem(row, 1, QTableWidgetItem(r.target))
        for col, value in [(2, r.match_case), (3, r.whole_word)]:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if value else Qt.CheckState.Unchecked)
            self.rules.setItem(row, col, item)

    def _remove(self) -> None:
        for idx in reversed(self.rules.selectionModel().selectedRows()):
            self.rules.removeRow(idx.row())

    def to_wordlist_data(self) -> WordlistData:
        reps: list[ReplacementRule] = []
        for row in range(self.rules.rowCount()):
            src = (self.rules.item(row, 0).text() if self.rules.item(row, 0) else "").strip()
            dst = (self.rules.item(row, 1).text() if self.rules.item(row, 1) else "").strip()
            if not src:
                continue
            reps.append(
                ReplacementRule(
                    source=src,
                    target=dst,
                    match_case=self.rules.item(row, 2).checkState() == Qt.CheckState.Checked,
                    whole_word=self.rules.item(row, 3).checkState() == Qt.CheckState.Checked,
                )
            )
        terms = [t.strip() for t in self.terms.toPlainText().splitlines() if t.strip()]
        return WordlistData(replacements=reps, preferred_terms=terms)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.logger = logging.getLogger("sludre.ui")
        self.setWindowTitle("Sludre")
        self.resize(1020, 820)
        self._state_lock = threading.Lock()
        self._ready = False
        self._listening = False
        self._transcribing = False
        self._model_thread: threading.Thread | None = None

        self.bridge = UiBridge()
        self.bridge.status_changed.connect(self._set_status_label)
        self.bridge.log_message.connect(self._append_log)
        self.bridge.ready_changed.connect(self._set_ready_state)
        self.bridge.postprocess_ready.connect(self._on_postprocess_ready)
        self.bridge.postprocess_failed.connect(self._on_postprocess_failed)

        self.config_store = ConfigStore.default()
        self.config = self.config_store.load()
        self.secrets_store = EnvSecretsStore.default()
        self._secrets_migrated = False
        self._sync_runtime_secrets()
        self._prompt_presets = self._normalize_prompts(self.config.llm_prompt_presets)

        self.audio_capture = AudioCapture(sample_rate=self.config.sample_rate, channels=self.config.channels, max_record_seconds=self.config.max_record_seconds, silence_trim=self.config.silence_trim)
        self.text_inserter = TextInserter(restore_clipboard=self.config.restore_clipboard)
        self.wordlist_store = WordlistStore(Path(self.config.wordlist_path))
        self.llm_refiner = LlmRefiner()
        self.post_processor = TranscriptionPostProcessor(self.wordlist_store, self.llm_refiner)
        self.model_manager = self._make_model_manager(self.config)
        self.transcriber = Transcriber(model_path=self._initial_model_path(self.config), device="cuda")
        self.hotkey = HotkeyController(self._on_listen_start, self._on_listen_stop)
        self._logo_path = self._find_asset(["sludre_logo.jpg", "sludre_logo.png"])
        self._background_path = self._find_asset(["sludre_background_stick.jpg", "sludre_background_stick.png", "background.jpg", "background.png"])
        if self._logo_path:
            self.setWindowIcon(QIcon(str(self._logo_path)))
        self._build_ui()
        self._apply_theme()
        self._ui_log(f"Secrets file: {self.secrets_store.path}")
        if self._secrets_migrated:
            self._ui_log("Migrated legacy keys from config.json to project .env")
        self._ui_log(
            f"UI assets: logo={self._logo_path if self._logo_path else 'not found'}, "
            f"background={self._background_path if self._background_path else 'not found'}"
        )
        self._ui_log(f"Detailed log file: {default_log_file()}")
        self._register_hotkey()
        self._start_model_init()

    @staticmethod
    def _normalize_prompts(raw: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned = [{"name": str(x.get("name", "")).strip(), "prompt": str(x.get("prompt", "")).strip()} for x in raw if isinstance(x, dict)]
        cleaned = [x for x in cleaned if x["name"] and x["prompt"]]
        return cleaned if cleaned else [{"name": "Standard", "prompt": DEFAULT_LLM_SYSTEM_PROMPT}]

    @staticmethod
    def _find_asset(candidates: list[str]) -> Path | None:
        roots = [Path("assets"), Path(".")]
        for root in roots:
            for name in candidates:
                candidate = root / name
                if candidate.exists() and candidate.is_file():
                    return candidate
        return None

    def _build_ui(self) -> None:
        root = StyledBackgroundWidget(self._background_path, self)
        root.setObjectName("rootBackground")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)
        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("mainTabs")
        root_layout.addWidget(self.tabs)
        main = QWidget(self.tabs); settings = QWidget(self.tabs); self.tabs.addTab(main, "Sludre"); self.tabs.addTab(settings, "Indstillinger")

        main_layout = QVBoxLayout(main)
        hero = QWidget(main)
        hero.setObjectName("heroCard")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(18, 14, 18, 14)
        hero_layout.setSpacing(14)
        self.logo_label = QLabel(hero)
        self.logo_label.setObjectName("heroLogo")
        self.logo_label.setMinimumSize(78, 78)
        self.logo_label.setMaximumSize(78, 78)
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if self._logo_path:
            pix = QPixmap(str(self._logo_path))
            if not pix.isNull():
                self.logo_label.setPixmap(
                    pix.scaled(
                        74,
                        74,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        else:
            self.logo_label.setText("S")
        hero_layout.addWidget(self.logo_label, 0, Qt.AlignmentFlag.AlignTop)
        hero_text = QVBoxLayout()
        title = QLabel("Sludre")
        title.setObjectName("heroTitle")
        subtitle = QLabel("Lokal transkribering med valgfri LLM-oprydning")
        subtitle.setObjectName("heroSubtitle")
        hero_text.addWidget(title)
        hero_text.addWidget(subtitle)
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setObjectName("statusLabel")
        hero_text.addWidget(self.status_label)
        hero_layout.addLayout(hero_text, 1)
        main_layout.addWidget(hero)
        form = QFormLayout(); form.addRow("Global hotkey:", QLabel(self.config.hotkey.upper())); form.addRow("Language:", QLabel(self.config.language)); main_layout.addLayout(form)
        actions = QHBoxLayout(); self.hold_btn = QPushButton("Hold to Talk"); self.retry_btn = QPushButton("Retry Model Init"); self.hold_btn.pressed.connect(self._on_listen_start); self.hold_btn.released.connect(self._on_listen_stop); self.retry_btn.clicked.connect(self._start_model_init); actions.addWidget(self.hold_btn); actions.addWidget(self.retry_btn); actions.addStretch(1); main_layout.addLayout(actions)
        main_layout.addWidget(QLabel("Output historik:"))
        self.output_table = QTableWidget(0, 4, self); self.output_table.setHorizontalHeaderLabels(["Tid", "Transkribering", "Output", "Kopi"]); self.output_table.horizontalHeader().setStretchLastSection(False); self.output_table.setColumnWidth(0, 90); self.output_table.setColumnWidth(3, 80); main_layout.addWidget(self.output_table, 2)
        self.output_table.verticalHeader().setVisible(False)
        self.output_table.setAlternatingRowColors(True)
        self.output_table.setWordWrap(True)
        self.output_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.output_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        main_layout.addWidget(QLabel("System log:")); self.log_output = QTextEdit(); self.log_output.setReadOnly(True); main_layout.addWidget(self.log_output, 1)

        s = QVBoxLayout(settings)
        model_row = QHBoxLayout(); self.manual_model_input = QLineEdit(self.config.manual_model_path); browse = QPushButton("Browse..."); browse.clicked.connect(self._browse_model_path); model_row.addWidget(self.manual_model_input); model_row.addWidget(browse); s.addLayout(model_row)
        self.hf_token_input = QLineEdit(self.config.hf_token); self.hf_token_input.setEchoMode(QLineEdit.EchoMode.Password); self.hf_token_input.setPlaceholderText("Optional Hugging Face token"); s.addWidget(self.hf_token_input)
        self.llm_enabled = QCheckBox("Enable LLM cleanup"); self.llm_enabled.setChecked(self.config.llm_enabled); s.addWidget(self.llm_enabled)
        llm = QFormLayout(); self.provider = QComboBox(); self.provider.addItem("OpenAI-compatible", "openai_compatible"); self.provider.addItem("Mistral API", "mistral_api"); self.provider.currentIndexChanged.connect(self._on_provider_changed); self._set_combo(self.provider, self.config.llm_provider)
        self.base_url = QLineEdit(self.config.llm_base_url); self.mistral_base = QLineEdit(self.config.mistral_base_url); self.api_key = QLineEdit(self.config.llm_api_key); self.api_key.setEchoMode(QLineEdit.EchoMode.Password); self.model_name = QLineEdit(self.config.llm_model)
        self.mistral_preset = QComboBox(); [self.mistral_preset.addItem(m, m) for m in MISTRAL_MODEL_PRESETS]; self._set_combo(self.mistral_preset, self.config.mistral_model_preset)
        self.timeout = QSpinBox(); self.timeout.setRange(1, 60); self.timeout.setValue(self.config.llm_timeout_seconds); self.temperature = QDoubleSpinBox(); self.temperature.setRange(0.0, 2.0); self.temperature.setDecimals(2); self.temperature.setValue(self.config.llm_temperature)
        llm.addRow("Provider:", self.provider); llm.addRow("OpenAI base URL:", self.base_url); llm.addRow("Mistral base URL:", self.mistral_base); llm.addRow("API key:", self.api_key); llm.addRow("Mistral preset:", self.mistral_preset); llm.addRow("Custom model:", self.model_name); llm.addRow("Timeout (s):", self.timeout); llm.addRow("Temperature:", self.temperature); s.addLayout(llm)
        s.addWidget(QLabel("System prompt presets:"))
        pr = QHBoxLayout(); self.prompt_select = QComboBox(); self.prompt_select.currentIndexChanged.connect(self._load_prompt); self.prompt_name = QLineEdit(); self.prompt_name.setPlaceholderText("Prompt navn"); pr.addWidget(self.prompt_select); pr.addWidget(self.prompt_name); s.addLayout(pr)
        self.prompt_text = QPlainTextEdit(); self.prompt_text.setMaximumHeight(160); s.addWidget(self.prompt_text)
        pbtn = QHBoxLayout(); savep = QPushButton("Gem/Opdater prompt"); delp = QPushButton("Slet valgt prompt"); savep.clicked.connect(self._save_prompt); delp.clicked.connect(self._delete_prompt); pbtn.addWidget(savep); pbtn.addWidget(delp); pbtn.addStretch(1); s.addLayout(pbtn)
        self.wordlist_enabled = QCheckBox("Enable wordlist"); self.wordlist_enabled.setChecked(self.config.wordlist_enabled); self.wordlist_replace = QCheckBox("Apply replacements"); self.wordlist_replace.setChecked(self.config.wordlist_apply_replacements); self.wordlist_prompt = QCheckBox("Include preferred terms in prompt"); self.wordlist_prompt.setChecked(self.config.wordlist_include_in_prompt)
        wr = QHBoxLayout(); edit_wordlist = QPushButton("Edit wordlist"); edit_wordlist.clicked.connect(self._open_wordlist_editor); wr.addWidget(self.wordlist_enabled); wr.addWidget(self.wordlist_replace); wr.addWidget(self.wordlist_prompt); wr.addWidget(edit_wordlist); s.addLayout(wr)
        s.addWidget(QLabel(self.config.wordlist_path)); save = QPushButton("Gem indstillinger"); save.clicked.connect(self._save_settings); s.addWidget(save); s.addStretch(1)
        self._refresh_prompt_combo(self.config.llm_selected_prompt_name); self._load_prompt(); self._on_provider_changed()

    def _sync_runtime_secrets(self) -> None:
        self.secrets_store.ensure_exists()
        legacy_hf = self.config.hf_token.strip()
        legacy_llm = self.config.llm_api_key.strip()
        current_hf = self.secrets_store.get_secret("HF_TOKEN").strip()
        current_llm = self.secrets_store.get_secret("LLM_API_KEY").strip()
        migrated = False
        if legacy_hf and not current_hf:
            self.secrets_store.set_secret("HF_TOKEN", legacy_hf)
            current_hf = legacy_hf
            migrated = True
        if legacy_llm and not current_llm:
            self.secrets_store.set_secret("LLM_API_KEY", legacy_llm)
            current_llm = legacy_llm
            migrated = True
        self.config.hf_token = current_hf or legacy_hf
        self.config.llm_api_key = current_llm or legacy_llm
        if legacy_hf or legacy_llm:
            # Remove plaintext secrets from config.json after migration/runtime load.
            self.config_store.save(self.config)
        self._secrets_migrated = migrated

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"""
            QWidget {{
                color: {THEME["text"]};
                font-family: "Segoe UI Variable";
                font-size: 12px;
            }}
            #heroCard {{
                border: 1px solid {THEME["border"]};
                border-radius: 16px;
                background: rgba(7, 20, 37, 176);
            }}
            #heroLogo {{
                border-radius: 14px;
                background: rgba(6, 16, 30, 210);
                border: 1px solid rgba(173, 201, 233, 105);
                font-size: 38px;
                font-weight: 700;
            }}
            #heroTitle {{
                font-size: 28px;
                font-weight: 700;
                letter-spacing: 0.8px;
            }}
            #heroSubtitle {{
                color: {THEME["muted"]};
                font-size: 13px;
            }}
            #statusLabel {{
                font-size: 16px;
                font-weight: 650;
                color: #E5F8FF;
                margin-top: 2px;
            }}
            #mainTabs::pane {{
                border: 1px solid {THEME["border"]};
                border-radius: 14px;
                background: {THEME["card"]};
                margin-top: 8px;
            }}
            #mainTabs QTabBar::tab {{
                background: rgba(6, 14, 26, 145);
                border: 1px solid {THEME["border"]};
                border-bottom: none;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                padding: 8px 14px;
                margin-right: 6px;
                color: {THEME["muted"]};
            }}
            #mainTabs QTabBar::tab:selected {{
                background: {THEME["card"]};
                color: {THEME["text"]};
            }}
            QPushButton {{
                background: {THEME["accent"]};
                color: #0B2318;
                border: none;
                border-radius: 10px;
                padding: 7px 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {THEME["accent_hover"]};
            }}
            QPushButton:disabled {{
                background: rgba(140, 165, 190, 110);
                color: rgba(11, 35, 24, 140);
            }}
            QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableWidget {{
                background: {THEME["input_bg"]};
                border: 1px solid {THEME["border"]};
                border-radius: 10px;
                selection-background-color: rgba(50, 210, 124, 80);
                alternate-background-color: rgba(12, 21, 34, 170);
            }}
            QComboBox {{
                padding-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background: rgba(7, 17, 30, 245);
                color: {THEME["text"]};
                border: 1px solid {THEME["border"]};
                border-radius: 10px;
                selection-background-color: rgba(50, 210, 124, 135);
                selection-color: #061B12;
                outline: 0;
            }}
            QComboBox QAbstractItemView::item {{
                background: transparent;
                color: {THEME["text"]};
                min-height: 24px;
                padding: 4px 8px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: rgba(50, 210, 124, 60);
                color: {THEME["text"]};
            }}
            QComboBox QAbstractItemView::item:selected {{
                background: rgba(50, 210, 124, 135);
                color: #061B12;
            }}
            QHeaderView::section {{
                background: rgba(5, 12, 24, 190);
                color: {THEME["muted"]};
                border: none;
                padding: 6px;
            }}
            QCheckBox {{
                spacing: 6px;
            }}
            """
        )

    @staticmethod
    def _set_combo(combo: QComboBox, value: str) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _refresh_prompt_combo(self, selected: str | None = None) -> None:
        current = selected or str(self.prompt_select.currentData() or "")
        self.prompt_select.blockSignals(True); self.prompt_select.clear()
        for p in self._prompt_presets: self.prompt_select.addItem(p["name"], p["name"])
        if current: self._set_combo(self.prompt_select, current)
        if self.prompt_select.currentIndex() < 0 and self.prompt_select.count() > 0: self.prompt_select.setCurrentIndex(0)
        self.prompt_select.blockSignals(False)

    def _load_prompt(self) -> None:
        name = str(self.prompt_select.currentData() or "")
        p = next((x for x in self._prompt_presets if x["name"] == name), self._prompt_presets[0])
        self.prompt_name.setText(p["name"]); self.prompt_text.setPlainText(p["prompt"])

    def _save_prompt(self) -> bool:
        name = self.prompt_name.text().strip(); prompt = self.prompt_text.toPlainText().strip()
        if not name or not prompt: QMessageBox.warning(self, "Prompt mangler", "Navn og prompt skal udfyldes."); return False
        existing = next((x for x in self._prompt_presets if x["name"] == name), None)
        if existing: existing["prompt"] = prompt
        else: self._prompt_presets.append({"name": name, "prompt": prompt})
        self._refresh_prompt_combo(name); self._ui_log(f"Prompt preset saved: {name}"); return True

    def _delete_prompt(self) -> None:
        selected = str(self.prompt_select.currentData() or "")
        if len(self._prompt_presets) <= 1: QMessageBox.information(self, "Kan ikke slette", "Mindst ét preset skal eksistere."); return
        self._prompt_presets = [p for p in self._prompt_presets if p["name"] != selected]
        self._refresh_prompt_combo(); self._load_prompt(); self._ui_log(f"Prompt preset deleted: {selected}")

    def _save_settings(self) -> None:
        if not self._save_prompt(): return
        selected = str(self.prompt_select.currentData() or self._prompt_presets[0]["name"])
        selected_text = next((p["prompt"] for p in self._prompt_presets if p["name"] == selected), self.prompt_text.toPlainText().strip())
        self.config.manual_model_path = self.manual_model_input.text().strip()
        hf_token = self.hf_token_input.text().strip()
        llm_api_key = self.api_key.text().strip()
        try:
            self.secrets_store.set_secret("HF_TOKEN", hf_token)
            self.secrets_store.set_secret("LLM_API_KEY", llm_api_key)
        except Exception as exc:
            QMessageBox.critical(self, "Kunne ikke gemme nøgler", f"Kunne ikke skrive .env filen.\n\n{exc}")
            self._ui_log(f"Failed to write secrets file: {exc}", level=logging.ERROR)
            return
        self.config.hf_token = hf_token
        self.config.llm_enabled = self.llm_enabled.isChecked()
        self.config.llm_provider = str(self.provider.currentData())
        self.config.llm_base_url = self.base_url.text().strip()
        self.config.mistral_base_url = self.mistral_base.text().strip()
        self.config.llm_api_key = llm_api_key
        self.config.llm_model = self.model_name.text().strip()
        self.config.mistral_model_preset = str(self.mistral_preset.currentData())
        self.config.llm_timeout_seconds = int(self.timeout.value())
        self.config.llm_temperature = float(self.temperature.value())
        self.config.llm_prompt_presets = list(self._prompt_presets)
        self.config.llm_selected_prompt_name = selected
        self.config.llm_system_prompt = selected_text
        self.config.wordlist_enabled = self.wordlist_enabled.isChecked()
        self.config.wordlist_apply_replacements = self.wordlist_replace.isChecked()
        self.config.wordlist_include_in_prompt = self.wordlist_prompt.isChecked()
        self.config_store.save(self.config)
        self._ui_log("Indstillinger gemt (.env opdateret).")
        self._start_model_init()

    def _on_provider_changed(self) -> None:
        is_mistral = str(self.provider.currentData()) == "mistral_api"
        self.base_url.setEnabled(not is_mistral); self.mistral_base.setEnabled(is_mistral); self.mistral_preset.setEnabled(is_mistral)

    def _register_hotkey(self) -> None:
        try: self.hotkey.register(); self._ui_log("Global hotkey active: hold Ctrl+Space to record.")
        except Exception as exc: self._ui_log(f"Global hotkey registration failed: {exc}", level=logging.ERROR)

    def _make_model_manager(self, config: AppConfig) -> ModelManager:
        return ModelManager(repo_id=config.model_repo_id, cache_dir=Path(config.model_cache_dir), manual_model_path=config.manual_model_path.strip() or None, hf_token=self._resolve_hf_token(config), log_callback=self._ui_log)

    def _resolve_hf_token(self, config: AppConfig) -> str | None:
        token = (
            config.hf_token.strip()
            or self.secrets_store.get_secret("HF_TOKEN").strip()
            or os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
        )
        return token or None

    @staticmethod
    def _initial_model_path(config: AppConfig) -> Path:
        return Path(config.manual_model_path.strip() or config.model_cache_dir)

    def _start_model_init(self) -> None:
        if self._model_thread and self._model_thread.is_alive(): return
        self.bridge.ready_changed.emit(False); self.bridge.status_changed.emit("Status: Preparing model..."); self.model_manager = self._make_model_manager(self.config)
        def worker() -> None:
            try:
                model_path = self.model_manager.ensure_model_available()
                t = Transcriber(model_path=model_path, device="cuda"); t.load()
                with self._state_lock: self.transcriber = t
                self.bridge.status_changed.emit("Status: Ready"); self.bridge.ready_changed.emit(True); self._ui_log(f"Model loaded: {model_path}")
            except Exception as exc:
                self.bridge.status_changed.emit("Status: Error loading model"); self.bridge.ready_changed.emit(False); self._ui_log(f"Model initialization failed.\nDetails: {exc}", level=logging.ERROR)
        self._model_thread = threading.Thread(target=worker, daemon=True); self._model_thread.start()

    def _open_wordlist_editor(self) -> None:
        dialog = WordlistEditorDialog(self.wordlist_store.load(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        data = dialog.to_wordlist_data(); self.wordlist_store.save(data); self._ui_log(f"Wordlist saved. rules={len(data.replacements)} terms={len(data.preferred_terms)}")

    def _browse_model_path(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder", str(Path.home()))
        if folder: self.manual_model_input.setText(folder)

    def _on_listen_start(self) -> None:
        with self._state_lock:
            if not self._ready or self._transcribing or self._listening: return
            self._listening = True
        try: self.audio_capture.start(); self.bridge.status_changed.emit("Status: Listening...")
        except Exception as exc:
            with self._state_lock: self._listening = False
            self.bridge.status_changed.emit("Status: Audio error"); self._ui_log(f"Audio start failed: {exc}", level=logging.ERROR)

    def _on_listen_stop(self) -> None:
        with self._state_lock:
            if not self._listening: return
            self._listening = False
        audio = self.audio_capture.stop()
        if audio.size == 0: self.bridge.status_changed.emit("Status: Ready"); return
        self._start_transcription(audio)

    def _start_transcription(self, audio) -> None:
        with self._state_lock:
            if self._transcribing: return
            self._transcribing = True
        self.bridge.status_changed.emit("Status: Transcribing...")
        def worker() -> None:
            try:
                with self._state_lock: t = self.transcriber
                stt = t.transcribe(audio=audio, sample_rate=self.config.sample_rate, language=self.config.language)
                post = self.post_processor.process(raw_text=stt.text, config=self.config)
                self.bridge.postprocess_ready.emit({"post_result": post})
            except Exception as exc:
                self.bridge.postprocess_failed.emit(str(exc))
        self._processing_thread = threading.Thread(target=worker, daemon=True); self._processing_thread.start()

    def _on_postprocess_ready(self, payload: object) -> None:
        post = payload.get("post_result") if isinstance(payload, dict) else None
        if not isinstance(post, PostProcessResult): self._on_postprocess_failed("Invalid payload"); return
        out = post.text.strip()
        if post.llm_error and not self._ask_raw(post.llm_error): self._append_output(post.raw_text, post.text, post.llm_used, False); self._finish_cycle(); return
        if post.llm_error: out = post.raw_text.strip()
        if not out: self._append_output(post.raw_text, out, post.llm_used, False); self._finish_cycle(); return
        self.text_inserter.insert_text_at_cursor(out); self.bridge.status_changed.emit("Status: Inserted"); self._append_output(post.raw_text, out, post.llm_used, True); self._finish_cycle()

    def _append_output(self, raw_text: str, final_text: str, llm_used: bool, inserted: bool) -> None:
        row = self.output_table.rowCount(); self.output_table.insertRow(row)
        self.output_table.setItem(row, 0, QTableWidgetItem(time.strftime("%H:%M:%S")))
        self.output_table.setItem(row, 1, QTableWidgetItem(raw_text))
        self.output_table.setItem(row, 2, QTableWidgetItem(final_text))
        btn = QPushButton("Kopier"); btn.clicked.connect(lambda _, txt=final_text: QApplication.clipboard().setText(txt))
        self.output_table.setCellWidget(row, 3, btn)
        self.output_table.scrollToBottom()

    def _ask_raw(self, err: str) -> bool:
        dlg = QMessageBox(self); dlg.setIcon(QMessageBox.Icon.Warning); dlg.setWindowTitle("LLM Cleanup Failed"); dlg.setText("LLM cleanup failed. Insert raw transcription?"); dlg.setInformativeText(err[:300] + ("..." if len(err) > 300 else ""))
        yes = dlg.addButton("Insert raw transcription", QMessageBox.ButtonRole.AcceptRole); dlg.addButton("Cancel insertion", QMessageBox.ButtonRole.RejectRole); dlg.exec()
        return dlg.clickedButton() is yes

    def _on_postprocess_failed(self, error_message: str) -> None:
        self.bridge.status_changed.emit("Status: Transcription error"); self._ui_log(f"Transcription failed: {error_message}", level=logging.ERROR); self._finish_cycle()

    def _finish_cycle(self) -> None:
        with self._state_lock: self._transcribing = False
        if self._ready: self.bridge.status_changed.emit("Status: Ready")

    def _set_status_label(self, text: str) -> None:
        self.status_label.setText(text)

    def _append_log(self, text: str) -> None:
        self.log_output.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def _ui_log(self, text: str, level: int = logging.INFO) -> None:
        self.logger.log(level, text); self.bridge.log_message.emit(text)

    def _set_ready_state(self, ready: bool) -> None:
        with self._state_lock: self._ready = ready
        self.hold_btn.setEnabled(ready); self.retry_btn.setEnabled(not ready)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.hotkey.unregister()
        if self.audio_capture.is_recording(): self.audio_capture.stop()
        super().closeEvent(event)
