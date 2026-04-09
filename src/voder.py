import sys
import os

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

MODELS_DIR = os.path.join(_src_dir, "models")

MODELS_TMP_DIR = os.path.join(MODELS_DIR, "tmp")
MODELS_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

QWEN_TTS_VOICEDESIGN_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "qwen_tts_voicedesign")
QWEN_TTS_BASE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "qwen_tts_base")
ACESTEP_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "acestep")
SEED_VC_V1_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v1")
SEED_VC_V2_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v2")
WHISPER_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "whisper")
UNISE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "unise")
TANGOFLUX_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "tangoflux")

os.environ["HF_HOME"] = MODELS_DIR
os.environ["HF_HUB_CACHE"] = MODELS_TMP_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_TMP_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_TMP_DIR

os.environ["XDG_CACHE_HOME"] = MODELS_TMP_DIR

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_TMP_DIR, exist_ok=True)
os.makedirs(MODELS_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(QWEN_TTS_VOICEDESIGN_DIR, exist_ok=True)
os.makedirs(QWEN_TTS_BASE_DIR, exist_ok=True)
os.makedirs(ACESTEP_DIR, exist_ok=True)
os.makedirs(SEED_VC_V1_DIR, exist_ok=True)
os.makedirs(SEED_VC_V2_DIR, exist_ok=True)
os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(UNISE_DIR, exist_ok=True)
os.makedirs(TANGOFLUX_DIR, exist_ok=True)

import time
import math
import tempfile
import shutil
import gc
import traceback
import numpy as np
import torch
import torchaudio
import yaml
import soundfile as sf
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QMessageBox, QProgressBar, QFrame, QSizePolicy,
                             QDesktopWidget, QComboBox, QMenu, QAction, QSlider,
                             QGridLayout, QInputDialog, QTextEdit, QSplitter,
                             QListWidget, QListWidgetItem, QLineEdit, QSpinBox,
                             QScrollArea, QDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon, QFont, QColor, QPalette, QPainter, QPen, QBrush
from omegaconf import DictConfig
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
import subprocess
import json
import re

HF_TOKEN_FILE = "HF_TOKEN.txt"

def setup_hf_token():
    if not os.path.exists(HF_TOKEN_FILE):
        with open(HF_TOKEN_FILE, 'w') as f:
            f.write("# Paste your HuggingFace token here\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            f.write("# Some models may require a token for gated repositories\n")
        return None
    with open(HF_TOKEN_FILE, 'r') as f:
        content = f.read().strip()
        lines = [line for line in content.split('\n') if line and not line.startswith('#')]
        if lines:
            return lines[0]
    return None

hf_token = setup_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    possible_paths = [
        "HF_TOKEN.txt",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "HF_TOKEN.txt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "1", "HF_TOKEN.txt"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read().strip()
                lines = [line for line in content.split('\n') if line and not line.startswith('#')]
                if lines:
                    hf_token = lines[0]
                    os.environ["HF_TOKEN"] = hf_token
                    break
    if not hf_token:
        print("\n" + "="*60)
        print("WARNING: HuggingFace token not found!")
        print("="*60)
        print("To use pyannote speaker diarization, you need to:")
        print("1. Get a token from: https://huggingface.co/settings/tokens")
        print("2. Create a file called 'HF_TOKEN.txt' with your token")
        print("3. Make sure the token has access to pyannote models")
        print("   (Accept conditions at: https://huggingface.co/pyannote/speaker-diarization-community-1)")
        print("="*60 + "\n")

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None, target_dir=None):
    if target_dir is None:
        target_dir = SEED_VC_V2_DIR
    os.makedirs(target_dir, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=target_dir)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=target_dir)
    return model_path, config_path

THEME = {
    'background': '#0A0A0A',
    'surface': '#1a1a1a',
    'surface_hover': '#2a2a2a',
    'surface_active': '#3a3a3a',
    'text': '#E5E5E5',
    'text_secondary': '#A0A0A0',
    'accent': '#4CAF50',
    'accent_hover': '#45a049',
    'accent_pressed': '#3d8b40',
    'accent_disabled': '#2d5a30',
    'border': '#404040',
    'border_light': '#E5E5E5',
    'border_disabled': '#555555',
    'error': '#f44336',
    'warning': '#ff9800',
    'success': '#4CAF50',
    'panel_background': '#121212',
    'panel_border': '#E5E5E5',
}

def get_main_button_style():
    return """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #121212, stop:0.3 #121212, stop:0.7 #1a1a1a, stop:1 #121212);
            border: 2px solid #E5E5E5;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #121212, stop:0.3 #161616, stop:0.7 #1e1e1e, stop:1 #121212);
            border: 2px solid #E5E5E5;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0e0e0e, stop:0.3 #121212, stop:0.7 #161616, stop:1 #0e0e0e);
            border: 2px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }
    """

def get_secondary_button_style():
    return """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #121212, stop:0.3 #121212, stop:0.7 #1a1a1a, stop:1 #121212);
            border: 2px solid #E5E5E5;
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            color: white;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #121212, stop:0.3 #161616, stop:0.7 #1e1e1e, stop:1 #121212);
            border: 2px solid #E5E5E5;
        }
        QPushButton:pressed {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0e0e0e, stop:0.3 #121212, stop:0.7 #161616, stop:1 #0e0e0e);
            border: 2px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }
    """

def get_surface_button_style():
    return """
        QPushButton {
            background-color: #2a2a2a;
            color: white;
            border: 1px solid #3a3a3a;
            border-radius: 5px;
            font-size: 12px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #3a3a3a;
            border: 1px solid #E5E5E5;
        }
        QPushButton:pressed {
            background-color: #4a4a4a;
            border: 1px solid #E5E5E5;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            border: 1px solid #404040;
            color: #666666;
        }
    """

def get_panel_style():
    return f"""
        QFrame {{
            background-color: {THEME['panel_background']};
            border: 2px solid {THEME['panel_border']};
            border-radius: 8px;
        }}
    """

def get_title_label_style():
    return f"""
        color: {THEME['text']};
        font-weight: bold;
        font-size: 16px;
    """

def get_subtitle_label_style():
    return f"""
        color: {THEME['text_secondary']};
        font-size: 12px;
    """

def get_status_bar_style():
    return f"""
        color: {THEME['text_secondary']};
        padding: 6px 12px;
        font-size: 12px;
    """

def get_progress_bar_style():
    return f"""
        QProgressBar {{
            border: 1px solid {THEME['border']};
            background-color: {THEME['surface']};
            height: 8px;
            border-radius: 4px;
            text-align: center;
            color: {THEME['text_secondary']};
        }}
        QProgressBar::chunk {{
            background-color: {THEME['text_secondary']};
            border-radius: 3px;
        }}
    """

def get_window_style():
    return f"""
        background-color: {THEME['background']};
        color: {THEME['text']};
    """

def get_text_edit_style():
    return f"""
        QTextEdit {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 2px solid {THEME['border']};
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            padding: 8px;
        }}
        QTextEdit:focus {{
            border: 2px solid {THEME['accent']};
        }}
    """

def get_combo_box_style():
    return f"""
        QComboBox {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 2px solid {THEME['border_light']};
            border-radius: 6px;
            padding: 6px 12px;
            min-width: 80px;
            font-size: 13px;
            selection-background-color: {THEME['surface_hover']};
            selection-color: {THEME['text']};
        }}
        QComboBox::drop-down {{
            border: none;
            subcontrol-origin: padding;
            subcontrol-position: right center;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none();
            width: 0px;
            height: 0px;
        }}
        QComboBox:hover {{
            border: 2px solid #E5E5E5;
        }}
        QComboBox:disabled {{
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }}
        QComboBox QAbstractItemView {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border_light']};
            border-radius: 4px;
            selection-background-color: {THEME['surface_hover']};
            selection-color: {THEME['text']};
        }}
    """

def get_line_edit_style():
    return f"""
        QLineEdit {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 2px solid {THEME['border']};
            border-radius: 6px;
            padding: 6px 8px;
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border: 2px solid {THEME['accent']};
        }}
        QLineEdit:disabled {{
            background-color: #2a2a2a;
            border: 2px solid #555555;
            color: #666666;
        }}
    """

def get_list_widget_style():
    return f"""
        QListWidget {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border']};
            border-radius: 4px;
            font-size: 12px;
        }}
        QListWidget::item {{
            padding: 4px;
            border-bottom: 1px solid {THEME['border']};
        }}
        QListWidget::item:selected {{
            background-color: {THEME['accent']};
            color: white;
        }}
        QListWidget::item:hover {{
            background-color: {THEME['surface_hover']};
        }}
    """

class DialogueScriptWidget(QWidget):
    characters_changed = pyqtSignal(set)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        self.setup_ui()
        self.add_row()
        self.update_characters()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(scroll_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def add_row(self, character="", text=""):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        char_edit = QLineEdit()
        char_edit.setPlaceholderText("Character")
        char_edit.setStyleSheet(get_line_edit_style())
        char_edit.setMinimumWidth(100)
        char_edit.setText(character)
        char_edit.textChanged.connect(self.on_text_changed)
        row_layout.addWidget(char_edit)

        text_edit = QLineEdit()
        text_edit.setPlaceholderText("Dialogue text")
        text_edit.setStyleSheet(get_line_edit_style())
        text_edit.setText(text)
        text_edit.textChanged.connect(self.on_text_changed)
        row_layout.addWidget(text_edit, stretch=1)

        delete_btn = QPushButton("×")
        delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: bold;
                padding: 4px 8px;
                min-width: 24px;
                max-width: 24px;
            }
            QPushButton:hover {
                background-color: #f44336;
            }
        """)
        delete_btn.setCursor(Qt.PointingHandCursor)
        delete_btn.clicked.connect(lambda: self.delete_row(row_widget))
        row_layout.addWidget(delete_btn)

        self.rows_layout.addWidget(row_widget)
        self.rows.append((char_edit, text_edit, delete_btn, row_widget))

        if len(self.rows) == 1:
            delete_btn.setEnabled(False)
            delete_btn.setVisible(False)

    def delete_row(self, row_widget):
        for i, (_, _, _, w) in enumerate(self.rows):
            if w == row_widget:
                if len(self.rows) == 1:
                    return
                self.rows_layout.removeWidget(w)
                w.deleteLater()
                del self.rows[i]
                break
        for idx, (_, _, btn, _) in enumerate(self.rows):
            if idx == 0:
                btn.setEnabled(False)
                btn.setVisible(False)
            else:
                btn.setEnabled(True)
                btn.setVisible(True)
        self.on_text_changed()

    def on_text_changed(self):
        if self.rows:
            last_char, last_text, _, _ = self.rows[-1]
            if last_char.text().strip() and last_text.text().strip():
                if not (len(self.rows) > 1 and not (self.rows[-2][0].text().strip() and self.rows[-2][1].text().strip())):
                    self.add_row()
        self.update_characters()

    def update_characters(self):
        chars = set()
        for char_edit, _, _, _ in self.rows:
            text = char_edit.text().strip()
            if text:
                chars.add(text.lower())
        self.characters_changed.emit(chars)

    def get_dialogue_items(self):
        items = []
        for idx, (char_edit, text_edit, _, _) in enumerate(self.rows):
            char = char_edit.text().strip()
            text = text_edit.text().strip()
            if char and text:
                items.append((idx + 1, char, text))
        return items

    def validate(self):
        active_rows = 0
        for char_edit, text_edit, _, _ in self.rows:
            char = char_edit.text().strip()
            text = text_edit.text().strip()
            if char or text:
                if not char or not text:
                    return False, "Each active line must have both Character and Text."
                active_rows += 1
        if active_rows == 0:
            return False, "No dialogue entered."
        return True, ""

    def clear(self):
        while len(self.rows) > 1:
            _, _, _, w = self.rows.pop()
            self.rows_layout.removeWidget(w)
            w.deleteLater()
        char_edit, text_edit, delete_btn, _ = self.rows[0]
        char_edit.clear()
        text_edit.clear()
        delete_btn.setEnabled(False)
        delete_btn.setVisible(False)
        self.update_characters()

class VoicePromptWidget(QWidget):
    prompts_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = 'text'
        self.characters = set()
        self.character_rows = {}
        self.audio_numbers = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(scroll_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(6)
        self.scroll.setWidget(scroll_widget)
        layout.addWidget(self.scroll)

    def set_mode(self, mode):
        if mode not in ('text', 'combo'):
            raise ValueError("Mode must be 'text' or 'combo'")
        self.mode = mode
        self.rebuild()

    def set_characters(self, chars_set):
        if chars_set == self.characters:
            return
        old_prompts = self.get_all_prompts()
        self.characters = chars_set
        self.rebuild()
        for char_lower, prompt in old_prompts.items():
            if char_lower in self.character_rows and prompt is not None:
                _, inp, _ = self.character_rows[char_lower]
                if self.mode == 'text':
                    inp.setText(prompt)
                else:
                    index = inp.findData(prompt)
                    if index >= 0:
                        inp.setCurrentIndex(index)

    def set_audio_numbers(self, numbers):
        self.audio_numbers = numbers
        if self.mode == 'combo':
            old_prompts = self.get_all_prompts()
            self.rebuild()
            for char_lower, num_str in old_prompts.items():
                if char_lower in self.character_rows and num_str is not None:
                    _, inp, _ = self.character_rows[char_lower]
                    index = inp.findData(num_str)
                    if index >= 0:
                        inp.setCurrentIndex(index)

    def rebuild(self):
        for widget in self.character_rows.values():
            label, inp, row_widget = widget
            self.rows_layout.removeWidget(row_widget)
            label.deleteLater()
            inp.deleteLater()
            row_widget.deleteLater()
        self.character_rows.clear()

        sorted_chars = sorted(self.characters)
        for char_lower in sorted_chars:
            label = QLabel(char_lower)
            label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px; min-width: 80px;")
            if self.mode == 'text':
                inp = QLineEdit()
                inp.setStyleSheet(get_line_edit_style())
                inp.setPlaceholderText("Describe the voice...")
                inp.textChanged.connect(lambda: self.prompts_changed.emit())
            else:
                inp = QComboBox()
                inp.setStyleSheet(get_combo_box_style())
                inp.setEditable(False)
                inp.addItem("", None)
                for num in self.audio_numbers:
                    inp.addItem(num, num)
                inp.setCurrentIndex(0)
                inp.setFocusPolicy(Qt.StrongFocus)
                inp.wheelEvent = lambda event: event.ignore()
                inp.currentIndexChanged.connect(lambda: self.prompts_changed.emit())
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(label)
            row_layout.addWidget(inp, stretch=1)
            self.rows_layout.addWidget(row_widget)
            self.character_rows[char_lower] = (label, inp, row_widget)

    def get_prompt(self, character):
        char_lower = character.lower()
        if char_lower not in self.character_rows:
            return None
        _, inp, _ = self.character_rows[char_lower]
        if self.mode == 'text':
            text = inp.text().strip()
            return text if text else None
        else:
            return inp.currentData()

    def get_all_prompts(self):
        result = {}
        for char_lower in self.character_rows:
            result[char_lower] = self.get_prompt(char_lower)
        return result

    def has_all_prompts(self):
        for char_lower in self.characters:
            if char_lower not in self.character_rows:
                return False
            prompt = self.get_prompt(char_lower)
            if prompt is None or prompt == "":
                return False
        return True

    def clear(self):
        self.characters.clear()
        self.rebuild()

class AudioWaveformWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setStyleSheet(f"background-color: {THEME['surface']}; border: 1px solid {THEME['border']};")
        self.audio_data = None
        self.sample_rate = 44100

    def set_audio(self, audio_path):
        if audio_path and os.path.exists(audio_path):
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                self.audio_data = waveform[0].numpy()
                self.sample_rate = sample_rate
                self.update()
            except:
                self.audio_data = None
                self.update()
        else:
            self.audio_data = None
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        painter.fillRect(self.rect(), QColor(THEME['surface']))
        if self.audio_data is None:
            painter.setPen(QColor(THEME['text_secondary']))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Audio")
            return
        painter.setPen(QColor(THEME['accent']))
        samples = len(self.audio_data)
        if samples == 0:
            return
        step = max(1, samples // width)
        for x in range(width):
            start_idx = x * step
            end_idx = min(start_idx + step, samples)
            if start_idx < samples:
                chunk = self.audio_data[start_idx:end_idx]
                max_val = np.max(np.abs(chunk))
                y_center = height // 2
                y_offset = int(max_val * (height // 2) * 0.9)
                painter.drawLine(x, y_center - y_offset, x, y_center + y_offset)

class WhisperSTT:
    def __init__(self, model_dir=None):
        self.model_dir = WHISPER_DIR if model_dir is None else model_dir
        self.model = None
        self.checkpoint_path = os.path.join(self.model_dir, "whisper-turbo.pt")
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        if self.model is None:
            try:
                import whisper
                import torch
                if os.path.exists(self.checkpoint_path):
                    self.model = whisper.load_model(self.checkpoint_path)
                else:
                    self.model = whisper.load_model("large-v3-turbo")
                    self._save_checkpoint()
            except Exception as e:
                print(f"Error loading Whisper: {e}")

    def _save_checkpoint(self):
        import torch
        checkpoint = {
            "dims": {
                "n_mels": self.model.dims.n_mels,
                "n_audio_ctx": self.model.dims.n_audio_ctx,
                "n_audio_state": self.model.dims.n_audio_state,
                "n_audio_head": self.model.dims.n_audio_head,
                "n_audio_layer": self.model.dims.n_audio_layer,
                "n_vocab": self.model.dims.n_vocab,
                "n_text_ctx": self.model.dims.n_text_ctx,
                "n_text_state": self.model.dims.n_text_state,
                "n_text_head": self.model.dims.n_text_head,
                "n_text_layer": self.model.dims.n_text_layer,
            },
            "model_state_dict": self.model.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)

    def transcribe(self, audio_path):
        if self.model is None:
            return None
        try:
            result = self.model.transcribe(audio_path, word_timestamps=True)
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

class EasyOCRReader:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.easyocr_dir = os.path.join(self.model_dir, "easyocr")
        self.model = None
        self.reader = None
        os.makedirs(self.easyocr_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.easyocr_dir, exist_ok=True)
        if self.reader is None:
            try:
                import easyocr
                print("Loading EasyOCR model...")
                self.reader = easyocr.Reader(
                    ['en'],
                    model_storage_directory=self.easyocr_dir,
                    download_enabled=True,
                    gpu=False
                )
                print("EasyOCR model loaded successfully")
            except Exception as e:
                print(f"Error loading EasyOCR: {e}")
                print("Note: EasyOCR will use CPU for text recognition.")

    def read_text(self, image_path):
        if self.reader is None:
            return None
        try:
            result = self.reader.readtext(image_path)
            return result
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None

    def extract_text_from_image(self, image_path):
        if self.reader is None:
            return False, None, "EasyOCR model not loaded"

        try:
            result = self.read_text(image_path)
            if not result:
                return False, None, "No text found in image"

            texts = []
            for detection in result:
                text = detection[1].strip()
                if text:
                    texts.append(text)

            if not texts:
                return False, None, "No text found in image"

            full_text = ' '.join(texts)
            return True, full_text, None

        except Exception as e:
            return False, None, f"Error extracting text: {str(e)}"

    def cleanup(self):
        self.reader = None
        gc.collect()

class SpeakerDiarization:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.diarization_dir = os.path.join(self.model_dir, "pyannote")
        self.model = None
        self.pipeline = None
        os.makedirs(self.diarization_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.diarization_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
                if libs_path not in sys.path:
                    sys.path.insert(0, libs_path)

                os.environ["PYANNOTE_SKIP_DEPENDENCY_CHECK"] = "1"

                from pyannote.audio import Pipeline
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Loading pyannote speaker diarization model...")

                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("Error: HuggingFace token is required for pyannote")
                    return

                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    cache_dir=self.diarization_dir,
                    token=token
                )
                self.pipeline = self.pipeline.to(device)
                print("Speaker diarization model loaded successfully")
            except ImportError as e:
                print(f"Error: pyannote.audio not available in local libs")
                print(f"Import error: {e}")
            except Exception as e:
                error_str = str(e).lower()
                if 'audio_metadata' in error_str or 'torchaudio' in error_str:
                    print(f"Error loading speaker diarization model: torchaudio compatibility issue")
                    print("Note: Your torchaudio version may be incompatible with pyannote.audio")
                    print("Try upgrading: pip install --upgrade torchaudio")
                elif 'token' in error_str or 'auth' in error_str:
                    print(f"Error loading speaker diarization model: Authentication failed")
                    print("Make sure your HF_TOKEN is valid and has accepted the model conditions")
                    print("Visit: https://huggingface.co/pyannote/speaker-diarization-community-1")
                else:
                    print(f"Error loading speaker diarization model: {e}")
                print("Note: pyannote requires authentication token.")
                print("Set HF_TOKEN in HF_TOKEN.txt file with your HuggingFace token.")

    def diarize(self, audio_path):
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline(audio_path, min_speakers=1)
            if hasattr(result, 'speaker_diarization'):
                return result.speaker_diarization
            return result
        except Exception as e:
            print(f"Diarization error: {e}")
            return None

    def format_diarization(self, diarization, transcription_result):
        if diarization is None or transcription_result is None:
            return []

        try:
            segments = transcription_result.get("segments", [])
            if not segments:
                return []

            transcription_segments = []
            for seg in segments:
                words = seg.get("words", [])
                if words:
                    for word in words:
                        transcription_segments.append({
                            "start": word.get("start", seg.get("start", 0)),
                            "end": word.get("end", seg.get("end", 0)),
                            "text": word.get("word", "").strip()
                        })
                else:
                    transcription_segments.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip()
                    })

            if not transcription_segments:
                return []

            diarization_turns = []
            for turn in diarization.itertracks(yield_label=True):
                segment, track, speaker = turn

                start_time = float(segment.start)
                end_time = float(segment.end)

                diarization_turns.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker
                })

            diarization_turns.sort(key=lambda x: x["start"])

            result = []
            for t_seg in transcription_segments:
                best_speaker = None
                best_overlap = 0

                for turn in diarization_turns:
                    overlap_start = max(t_seg["start"], turn["start"])
                    overlap_end = min(t_seg["end"], turn["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)

                    if overlap_duration > 0:
                        if t_seg["start"] >= turn["start"] and t_seg["end"] <= turn["end"]:
                            if best_overlap < 2:
                                best_speaker = turn["speaker"]
                                best_overlap = 2
                        elif overlap_duration > best_overlap:
                            best_speaker = turn["speaker"]
                            best_overlap = overlap_duration

                if best_speaker is not None:
                    result.append({
                        "speaker": best_speaker,
                        "start": t_seg["start"],
                        "end": t_seg["end"],
                        "text": t_seg["text"]
                    })

            if result:
                for i, seg in enumerate(result):
                    if seg["speaker"] is None:
                        prev_speaker = None
                        next_speaker = None

                        for j in range(i - 1, -1, -1):
                            if result[j]["speaker"] is not None:
                                prev_speaker = result[j]["speaker"]
                                break

                        for j in range(i + 1, len(result)):
                            if result[j]["speaker"] is not None:
                                next_speaker = result[j]["speaker"]
                                break

                        if prev_speaker and next_speaker:
                            prev_dist = i - next((idx for idx in range(i - 1, -1, -1) if result[idx]["speaker"] is not None), i)
                            next_dist = next((idx for idx in range(i + 1, len(result)) if result[idx]["speaker"] is not None), i) - i
                            seg["speaker"] = prev_speaker if prev_dist <= next_dist else next_speaker
                        elif prev_speaker:
                            seg["speaker"] = prev_speaker
                        elif next_speaker:
                            seg["speaker"] = next_speaker

                min_utterance_duration = 0.5
                i = 0
                while i < len(result):
                    duration = result[i]["end"] - result[i]["start"]
                    if duration < min_utterance_duration:
                        prev_idx = i - 1
                        while prev_idx >= 0 and result[prev_idx]["speaker"] == result[i]["speaker"]:
                            prev_idx -= 1

                        next_idx = i + 1
                        while next_idx < len(result) and result[next_idx]["speaker"] == result[i]["speaker"]:
                            next_idx += 1

                        should_merge_with_prev = False
                        should_merge_with_next = False

                        if prev_idx >= 0:
                            prev_duration = result[prev_idx]["end"] - result[prev_idx]["start"]
                            should_merge_with_prev = (prev_idx >= 0 and
                                (next_idx >= len(result) or
                                 result[prev_idx]["speaker"] != result[i]["speaker"] and
                                 prev_duration > duration))

                        if next_idx < len(result):
                            next_duration = result[next_idx]["end"] - result[next_idx]["start"]
                            should_merge_with_next = (next_idx < len(result) and
                                result[next_idx]["speaker"] != result[i]["speaker"] and
                                (next_idx >= len(result) or
                                 not should_merge_with_prev or
                                 next_duration > prev_duration))

                        if should_merge_with_prev:
                            result[prev_idx]["text"] += " " + result[i]["text"]
                            result[prev_idx]["end"] = result[i]["end"]
                            result.pop(i)
                        elif should_merge_with_next:
                            result[next_idx]["text"] = result[i]["text"] + " " + result[next_idx]["text"]
                            result[next_idx]["start"] = result[i]["start"]
                            result.pop(i)
                        else:
                            i += 1
                    else:
                        i += 1

            return result
        except Exception as e:
            print(f"Error formatting diarization: {e}")
            return []

import re

def _get_audio_duration(path):
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 30

def _parse_script_directives(text):
    tokens = text.split()
    directives = {}
    content_end = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        if token.startswith('/time:'):
            directives['time_raw'] = token[6:]
            content_end = i
        elif token.startswith('/level:'):
            directives['level_raw'] = token[7:]
            content_end = i
        elif token.startswith('/duration:'):
            directives['duration_raw'] = token[10:]
            content_end = i
        else:
            break
    clean_text = ' '.join(tokens[:content_end])
    return clean_text.strip(), directives

def _validate_time_directive(time_str):
    time_str = time_str.strip()
    if not time_str:
        return 0, 0, 0, None
    if not re.match(r'^[+-]?\d+([+-]\d+)*$', time_str):
        return 0, 0, 0, "Invalid time format"
    tokens = re.findall(r'[+-]?\d+', time_str)
    start_pad = 0
    cut_start = 0
    cut_end = 0
    position_set = False
    for token in tokens:
        if token.startswith('+'):
            cut_start += int(token[1:])
        elif token.startswith('-'):
            cut_end += int(token[1:])
        else:
            if not position_set:
                start_pad = int(token)
                position_set = True
            else:
                cut_end += int(token)
    return start_pad, cut_start, cut_end, None

def _validate_level_directive(level_str):
    level_str = level_str.strip()
    if not re.match(r'^\d+$', level_str):
        return None, "Invalid level: must be a number"
    val = int(level_str)
    if val < 0 or val > 100:
        return None, "Invalid level: must be 0-100"
    return val, None

def _validate_duration_directive(dur_str):
    dur_str = dur_str.strip()
    if not re.match(r'^\d+$', dur_str):
        return None, "Invalid duration: must be a number"
    val = int(dur_str)
    if val < 1 or val > 30:
        return None, "Invalid duration: must be 1-30"
    return val, None

def _parse_directives_for_line(directives):
    result = {'time_end': 0, 'time_start': 0, 'time_pad': 0, 'level': 100, 'duration': None, 'has_time': False}
    errors = []
    if 'time_raw' in directives:
        start_pad, cut_start, cut_end, err = _validate_time_directive(directives['time_raw'])
        if err:
            errors.append(f"/time: {err}")
        else:
            result['time_pad'] = start_pad
            result['time_end'] = cut_end
            result['time_start'] = cut_start
            result['has_time'] = True
    if 'level_raw' in directives:
        val, err = _validate_level_directive(directives['level_raw'])
        if err:
            errors.append(f"/level: {err}")
        else:
            result['level'] = val
    if 'duration_raw' in directives:
        val, err = _validate_duration_directive(directives['duration_raw'])
        if err:
            errors.append(f"/duration: {err}")
        else:
            result['duration'] = val
    return result, errors

def _apply_clip_effects(input_path, output_path, cut_start=0, cut_end=0, level=100):
    clip_duration = _get_audio_duration(input_path)
    effective_duration = clip_duration - cut_start - cut_end
    if effective_duration <= 0.01:
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono',
            '-t', '0.01', '-y', output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return
    filters = []
    if level != 100:
        filters.append(f"volume={level / 100.0}")
    filter_str = ",".join(filters)
    cmd = ['ffmpeg', '-ss', str(cut_start), '-i', input_path, '-t', str(effective_duration)]
    if filter_str:
        cmd.extend(['-af', filter_str])
    cmd.extend(['-y', output_path])
    subprocess.run(cmd, capture_output=True, text=True)

def _parse_music_level_spec(spec_str):
    if not spec_str or not spec_str.strip():
        return []
    spec_str = spec_str.strip()
    segments = []
    parts = spec_str.split()
    for part in parts:
        m = re.match(r'^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)-(\d+)$', part)
        if m:
            from_sec = float(m.group(1))
            to_sec = float(m.group(2))
            level_pct = int(m.group(3))
        else:
            m = re.match(r'^(\d+(?:\.\d+)?)-(\d+)$', part)
            if m:
                from_sec = float(m.group(1))
                to_sec = None
                level_pct = int(m.group(2))
            else:
                m = re.match(r'^(\d+)$', part)
                if m:
                    from_sec = 0.0
                    to_sec = None
                    level_pct = int(m.group(1))
                else:
                    return None
        if to_sec is not None and from_sec >= to_sec:
            return None
        if level_pct < 0:
            level_pct = 0
        if level_pct > 100:
            level_pct = 100
        segments.append((from_sec, to_sec, level_pct))
    return segments

def _build_music_volume_expression(segments, total_duration, default_vol=0.35, fade_dur=1.0):
    if not segments:
        return f"volume={default_vol}"
    default_v = f"{default_vol:.6f}"
    expr = default_v
    for from_sec, to_sec, level_pct in reversed(segments):
        if to_sec is None:
            to_sec = total_duration
        if to_sec > total_duration:
            to_sec = total_duration
        if from_sec >= to_sec:
            continue
        vol = level_pct / 100.0
        v = f"{vol:.6f}"
        seg_dur = to_sec - from_sec
        actual_fade = min(fade_dur, seg_dur / 2.0) if seg_dur > 0.01 else 0.01
        af = f"{actual_fade:.2f}"
        fi = max(0, from_sec - actual_fade)
        fo = max(fi, to_sec - actual_fade)
        expr = (
            f"if(between(t,{fi:.3f},{from_sec:.3f}),"
            f"{default_v}+({v}-{default_v})*(t-{fi:.3f})/{af},"
            f"if(between(t,{from_sec:.3f},{fo:.3f}),"
            f"{v},"
            f"if(between(t,{fo:.3f},{to_sec:.3f}),"
            f"{v}+({default_v}-{v})*(t-{fo:.3f})/{af},"
            f"{expr}"
            f")))"
        )
    return f"volume='{expr}':eval=frame"

def _mix_dialogue_with_music(dialogue_path, music_path, output_path, music_level_spec=None):
    duration = _get_audio_duration(dialogue_path)
    segments = _parse_music_level_spec(music_level_spec)
    if segments is None:
        print("Warning: Invalid music level spec, using default 35%")
        segments = []
    vol_filter = _build_music_volume_expression(segments, duration)
    mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    mixed_temp.close()
    cmd = [
        'ffmpeg', '-i', dialogue_path, '-i', music_path,
        '-filter_complex', f'[1:a]{vol_filter}[music];[0:a][music]amix=inputs=2:duration=longest',
        '-y', mixed_temp.name
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg mixing failed: {result.stderr}")
        return False
    shutil.move(mixed_temp.name, output_path)
    return True

def _generate_music_and_mix(ace, music_description, dialogue_path, output_path, music_level_spec=None):
    duration = _get_audio_duration(dialogue_path)
    print(f"Dialogue duration: {duration:.2f}s")
    print("Generating background music...")
    music_result = generate_background_music(ace, music_description, duration)
    if music_result is None:
        print("Error: Background music generation failed")
        return False
    music_temp_path, music_temp_dir = music_result
    print("Mixing dialogue with music...")
    success = _mix_dialogue_with_music(dialogue_path, music_temp_path, output_path, music_level_spec)
    if music_temp_dir is not None:
        shutil.rmtree(music_temp_dir, ignore_errors=True)
    return success

def _assemble_enhanced_dialogue(dialogue_items, voice_data, tts_design_obj=None, tts_vc_obj=None, vc_voice_data=None, output_path=None, mode='tts'):
    temp_dir = tempfile.mkdtemp()
    try:
        clips = []
        sfx_generator = None
        for i, item in enumerate(dialogue_items):
            num = item[0]
            char = item[1]
            text = item[2]
            directives = item[3] if len(item) > 3 else {}
            cut_start = directives.get('time_start', 0)
            cut_end = directives.get('time_end', 0)
            start_pad = directives.get('time_pad', 0)
            level = directives.get('level', 100)
            raw_file = os.path.join(temp_dir, f"raw_{i:03d}.wav")
            processed_file = os.path.join(temp_dir, f"processed_{i:03d}.wav")
            if char.lower() == 'sfx':
                duration = directives.get('duration')
                if duration is None:
                    print(f"Error: SFX line {num} requires /duration:nn (1-30)")
                    return False, "Missing duration for SFX line"
                if sfx_generator is None:
                    from tangoflux import TangoFluxGenerator
                    sfx_generator = TangoFluxGenerator(TANGOFLUX_DIR)
                    sfx_generator.ensure_model()
                    if sfx_generator.model is None:
                        return False, "Failed to load TangoFlux model"
                print(f"  Generating SFX line {num}: \"{text[:50]}\" ({duration}s)")
                audio = sfx_generator.generate(text, duration)
                if audio is None:
                    return False, f"SFX generation failed for line {num}"
                sfx_generator.save(audio, raw_file)
            else:
                char_lower = char.lower()
                is_vc = vc_voice_data is not None and char_lower in vc_voice_data
                is_tts = char_lower in voice_data
                if not is_vc and not is_tts:
                    print(f"Error: No voice data for '{char}'")
                    return False, f"Missing voice data for '{char}'"
                if is_vc:
                    if tts_vc_obj is None:
                        return False, "TTS+VC object not provided for cloned voice character"
                    tts_vc_obj.voice_prompt = vc_voice_data[char_lower]
                    success = tts_vc_obj.synthesize(text, raw_file)
                    if not success:
                        return False, f"Failed to synthesize line {num}"
                else:
                    if tts_design_obj is None:
                        return False, "TTS design object not provided"
                    voice_instruct = voice_data[char_lower]
                    success = tts_design_obj.synthesize(text, voice_instruct, raw_file)
                    if not success:
                        return False, f"Failed to synthesize line {num}"
            if not os.path.exists(raw_file):
                return False, f"Audio file not generated for line {num}"
            if cut_start > 0 or cut_end > 0 or level != 100:
                _apply_clip_effects(raw_file, processed_file, cut_start, cut_end, level)
                try:
                    os.unlink(raw_file)
                except:
                    pass
            else:
                shutil.move(raw_file, processed_file)
            has_time = directives.get('has_time', False)
            clips.append((has_time, start_pad, processed_file))
        if len(clips) < 1:
            return False, "No audio segments generated"
        tracks = []
        cursor = 0
        for has_time, orig_pad, fpath in clips:
            if has_time:
                pos = orig_pad
            else:
                pos = cursor
            tracks.append((pos, fpath))
            dur = _get_audio_duration(fpath)
            end = pos + dur
            if end > cursor:
                cursor = end
        if len(tracks) == 1:
            pad_ms, fpath = tracks[0]
            if pad_ms > 0:
                cmd = [
                    'ffmpeg', '-i', fpath,
                    '-af', f'adelay={int(pad_ms * 1000)}|{int(pad_ms * 1000)}',
                    '-y', output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return False, f"FFmpeg delay failed: {result.stderr}"
            else:
                shutil.copy(fpath, output_path)
            return True, "Dialogue assembled successfully"
        total_duration = 0
        for pad_sec, fpath in tracks:
            d = _get_audio_duration(fpath)
            end = pad_sec + d
            if end > total_duration:
                total_duration = end
        if total_duration <= 0:
            return False, "Total duration is zero"
        cmd = ['ffmpeg']
        for _, fpath in tracks:
            cmd.extend(['-i', fpath])
        filter_parts = []
        for idx, (pad_sec, _) in enumerate(tracks):
            if pad_sec > 0:
                delay_ms = int(pad_sec * 1000)
                filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[d{idx}]")
            else:
                filter_parts.append(f"[{idx}:a]acopy[d{idx}]")
        input_labels = "".join(f"[d{i}]" for i in range(len(tracks)))
        filter_parts.append(f"{input_labels}amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[out]")
        filter_str = ";".join(filter_parts)
        cmd.extend(['-filter_complex', filter_str, '-map', '[out]', '-y', output_path])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg mix failed: {result.stderr}"
        return True, "Dialogue assembled successfully"
    finally:
        if sfx_generator:
            sfx_generator.cleanup()
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

class QwenTTSVoiceDesign:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.model_dir_full = QWEN_TTS_VOICEDESIGN_DIR if model_dir is None else os.path.join(model_dir, "qwen_tts_voice_design")
        self.model = None
        os.makedirs(self.model_dir_full, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.model_dir_full, exist_ok=True)
        if self.model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                model_path = os.path.join(self.model_dir_full, "model")
                if os.path.exists(model_path):
                    self.model = Qwen3TTSModel.from_pretrained(model_path, device_map=device, dtype=dtype)
                else:
                    self.model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        device_map=device,
                        dtype=dtype
                    )
            except Exception as e:
                print(f"Error loading Qwen-TTS VoiceDesign: {e}")

    def synthesize(self, text, voice_instruct, output_path, language="English"):
        if self.model is None:
            return False
        try:
            import soundfile as sf
            import torch
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_instruct
            )
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"VoiceDesign synthesis error: {e}")
            return False

    def synthesize_dialogue(self, dialogue_items, voice_prompts, output_path, language="English"):
        if self.model is None:
            return False, "Model not loaded"
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        try:
            for i, (num, char, script_text) in enumerate(dialogue_items):
                char_lower = char.lower()
                voice_instruct = voice_prompts.get(char_lower, voice_prompts.get(char, ""))
                if not voice_instruct:
                    return False, f"Missing voice prompt for character '{char}'"
                temp_file = os.path.join(temp_dir, f"segment_{i+1:03d}.wav")
                temp_files.append(temp_file)
                success = self.synthesize(script_text, voice_instruct, temp_file, language)
                if not success:
                    return False, f"Failed to synthesize segment {i+1}"
            if len(temp_files) < 2:
                if temp_files:
                    shutil.copy(temp_files[0], output_path)
                return len(temp_files) > 0, "Single segment processed" if temp_files else "No segments generated"
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"FFmpeg concatenation failed: {result.stderr}"
            return True, "Dialogue compiled successfully"
        except Exception as e:
            return False, f"Dialogue processing error: {str(e)}"
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

class QwenTTS:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.model_dir_base = QWEN_TTS_BASE_DIR if model_dir is None else os.path.join(model_dir, "qwen_tts_base")
        self.model = None
        self.voice_prompt = None
        os.makedirs(self.model_dir_base, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        if self.model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                if os.path.exists(os.path.join(self.model_dir_base, "config.json")):
                    print("Loading Qwen-TTS from local cache...")
                    self.model = Qwen3TTSModel.from_pretrained(
                        self.model_dir_base,
                        device_map=device,
                        dtype=dtype
                    )
                else:
                    print("Downloading Qwen-TTS from HuggingFace...")
                    self.model = Qwen3TTSModel.from_pretrained(
                        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        device_map=device,
                        dtype=dtype
                    )
            except Exception as e:
                print(f"Error loading Qwen-TTS: {e}")

    def extract_voice(self, audio_path):
        if self.model is None:
            return None
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform_np = waveform.cpu().numpy().flatten()
            self.voice_prompt = self.model.create_voice_clone_prompt(
                ref_audio=(waveform_np, sample_rate),
                x_vector_only_mode=True
            )
            return True
        except Exception as e:
            print(f"Voice extraction error: {e}")
            return None

    def synthesize(self, text, output_path):
        if self.model is None or self.voice_prompt is None:
            return False
        try:
            import soundfile as sf
            import torch
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language="English",
                voice_clone_prompt=self.voice_prompt
            )
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"Synthesis error: {e}")
            return False

class SeedVCV2:
    def __init__(self):
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.checkpoints_dir = SEED_VC_V2_DIR
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from hf_utils import load_custom_model_from_hf
                from modules.v2.vc_wrapper import (
                    DEFAULT_CE_REPO_ID, DEFAULT_CE_NARROW_CHECKPOINT,
                    DEFAULT_CE_WIDE_CHECKPOINT, DEFAULT_SE_REPO_ID, DEFAULT_SE_CHECKPOINT
                )
                cfm_path = self.download_checkpoint(
                    repo_id="Plachta/Seed-VC",
                    filename="v2/cfm_small.pth",
                    local_name="cfm_small.pth"
                )
                ar_path = self.download_checkpoint(
                    repo_id="Plachta/Seed-VC",
                    filename="v2/ar_base.pth",
                    local_name="ar_base.pth"
                )
                if not all([cfm_path, ar_path]):
                    return
                config_path = os.path.join(os.path.dirname(__file__), "configs", "v2", "vc_wrapper.yaml")
                cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
                self.model = instantiate(cfg)
                try:
                    from modules.bigvgan import bigvgan
                    self.model.vocoder = bigvgan.BigVGAN.from_pretrained(
                        "nvidia/bigvgan_v2_22khz_80band_256x",
                        use_cuda_kernel=False
                    )
                    print("Vocoder loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load vocoder: {e}")
                self.model.load_checkpoints(
                    cfm_checkpoint_path=cfm_path,
                    ar_checkpoint_path=ar_path
                )
                ce_narrow_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_NARROW_CHECKPOINT,
                    local_name="bsq32_light.pth"
                )
                if ce_narrow_path:
                    ce_narrow_checkpoint = torch.load(ce_narrow_path, map_location="cpu")
                    self.model.content_extractor_narrow.load_state_dict(ce_narrow_checkpoint, strict=False)
                ce_wide_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_WIDE_CHECKPOINT,
                    local_name="bsq2048_light.pth"
                )
                if ce_wide_path:
                    ce_wide_checkpoint = torch.load(ce_wide_path, map_location="cpu")
                    self.model.content_extractor_wide.load_state_dict(ce_wide_checkpoint, strict=False)
                se_path = self.download_checkpoint(
                    repo_id=DEFAULT_SE_REPO_ID,
                    filename=DEFAULT_SE_CHECKPOINT,
                    local_name="campplus_cn_common.bin"
                )
                if se_path:
                    se_checkpoint = torch.load(se_path, map_location="cpu")
                    self.model.style_encoder.load_state_dict(se_checkpoint, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model.setup_ar_caches(
                    max_batch_size=1,
                    max_seq_len=8192,
                    dtype=self.dtype,
                    device=self.device
                )
            except ImportError as e:
                print(f"Missing dependency for Seed-VC: {e}")
            except Exception as e:
                print(f"Error loading Seed-VC v2: {e}")

    def download_checkpoint(self, repo_id, filename, local_name):
        local_path = os.path.join(self.checkpoints_dir, local_name)
        if os.path.exists(local_path):
            return local_path
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.checkpoints_dir,
                force_filename=local_name
            )
            return downloaded_path if os.path.exists(downloaded_path) else local_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def convert(self, source_path, reference_path, output_path):
        if self.model is None:
            return False
        try:
            result = self.model.convert_voice(
                source_audio_path=source_path,
                target_audio_path=reference_path,
                device=torch.device(self.device),
                dtype=self.dtype
            )
            if result is not None:
                sf.write(output_path, result, 22050)
                return True
            return False
        except Exception as e:
            print(f"Seed-VC conversion error: {e}")
            return False

class SeedVCV1:
    def __init__(self):
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.checkpoints_dir = SEED_VC_V1_DIR
        self.whisper_model = None
        self.whisper_feature_extractor = None
        self.campplus_model = None
        self.bigvgan_model = None
        self.rmvpe = None
        self.to_mel = None
        self.sr = 44100
        self.hop_length = 512
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from hf_utils import load_custom_model_from_hf
                from modules.commons import build_model, load_checkpoint, recursive_munch
                from modules.campplus.DTDNN import CAMPPlus
                from modules.bigvgan import bigvgan
                from modules.audio import mel_spectrogram
                from modules.rmvpe import RMVPE
                from transformers import WhisperModel, AutoFeatureExtractor

                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
                    target_dir=SEED_VC_V1_DIR
                )
                config = yaml.safe_load(open(dit_config_path, 'r'))
                model_params = recursive_munch(config['model_params'])
                self.model = build_model(model_params, stage='DiT')
                self.hop_length = config['preprocess_params']['spect_params']['hop_length']
                self.sr = config['preprocess_params']['sr']

                self.model, _, _, _ = load_checkpoint(
                    self.model, None, dit_checkpoint_path,
                    load_only_params=True, ignore_modules=[], is_distributed=False
                )
                for key in self.model:
                    self.model[key].eval()
                    self.model[key].to(self.device)
                self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

                mel_fn_args = {
                    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
                    "win_size": config['preprocess_params']['spect_params']['win_length'],
                    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
                    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
                    "sampling_rate": self.sr,
                    "fmin": 0,
                    "fmax": None,
                    "center": False
                }
                self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

                whisper_name = "openai/whisper-small"
                self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
                del self.whisper_model.decoder
                self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

                campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", target_dir=SEED_VC_V1_DIR)
                self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
                self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
                self.campplus_model.eval()
                self.campplus_model.to(self.device)

                self.bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
                self.bigvgan_model.remove_weight_norm()
                self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

                rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", target_dir=SEED_VC_V1_DIR)
                self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)

                print("Seed-VC v1 (seed-uvit-whisper-base-f0-44k) loaded successfully")
            except ImportError as e:
                print(f"Missing dependency for Seed-VC v1: {e}")
            except Exception as e:
                print(f"Error loading Seed-VC v1: {e}")

    def download_checkpoint(self, repo_id, filename, local_name):
        local_path = os.path.join(self.checkpoints_dir, local_name)
        if os.path.exists(local_path):
            return local_path
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.checkpoints_dir,
                force_filename=local_name
            )
            return downloaded_path if os.path.exists(downloaded_path) else local_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def _process_whisper_features(self, audio_16k):
        if audio_16k.size(-1) <= 16000 * 30:
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, :audio_16k.size(-1) // 320 + 1]
        else:
            overlapping_time = 5
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:
                    chunk = audio_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer,
                        audio_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features, attention_mask=inputs.attention_mask
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, :chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            features = torch.cat(features_list, dim=1)
        return features

    def convert(self, source_path, reference_path, output_path):
        if self.model is None:
            return False
        try:
            import librosa
            source_audio = librosa.load(source_path, sr=self.sr)[0]
            ref_audio = librosa.load(reference_path, sr=self.sr)[0]

            source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
            ref_audio = torch.tensor(ref_audio[:self.sr * 25]).unsqueeze(0).float().to(self.device)

            ref_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

            S_alt = self._process_whisper_features(converted_waves_16k)
            S_ori = self._process_whisper_features(ref_waves_16k)

            mel = self.to_mel(source_audio.to(self.device).float())
            mel2 = self.to_mel(ref_audio.to(self.device).float())

            target_lengths = torch.LongTensor([int(mel.size(2))]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(
                ref_waves_16k,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))

            F0_ori = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
            F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

            if self.device.type == "mps":
                F0_ori = torch.from_numpy(F0_ori).float().to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).float().to(self.device)[None]
            else:
                F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)

            shifted_log_f0_alt = log_f0_alt.clone()
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)

            cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

            max_context_window = self.sr // self.hop_length * 30
            max_source_window = max_context_window - mel2.size(2)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                vc_target = self.model.cfm.inference(
                    torch.cat([prompt_condition, cond], dim=1),
                    torch.LongTensor([torch.cat([prompt_condition, cond], dim=1).size(1)]).to(mel2.device),
                    mel2, style2, None, 10,
                    inference_cfg_rate=0.7
                )
            vc_target = vc_target[:, :, mel2.size(-1):]

            del self.whisper_model, self.whisper_feature_extractor
            del self.campplus_model, self.rmvpe, self.model
            self.whisper_model = None
            self.whisper_feature_extractor = None
            self.campplus_model = None
            self.rmvpe = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vc_wave = self.bigvgan_model(vc_target.clone().float())[0]

            output_audio = vc_wave[0].cpu().numpy()
            sf.write(output_path, output_audio, self.sr)
            return True
        except Exception as e:
            print(f"Seed-VC v1 conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

class AceStepWrapper:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.checkpoints_dir = os.path.join(script_dir, "checkpoints")
        self.handler = None
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        if self.handler is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from acestep.handler import AceStepHandler
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                self.handler = AceStepHandler()
                status, success = self.handler.initialize_service(
                    project_root="",
                    config_path="acestep-v15-turbo",
                    device=device
                )
                if not success:
                    print(f"Error initializing ACE-Step: {status}")
                    self.handler = None
            except Exception as e:
                print(f"Error loading ACE-Step model: {e}")
                self.handler = None

    def generate(self, lyrics, style_prompt, output_path, duration=10):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            result = self.handler.generate_music(
                captions=style_prompt,
                lyrics=lyrics,
                vocal_language="unknown",
                inference_steps=8,
                guidance_scale=7.0,
                use_random_seed=True,
                seed=-1,
                audio_duration=duration,
                batch_size=1,
                task_type="text2music",
                shift=1.0,
            )
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step generation error: {e}")
            return False

def generate_background_music(ace_wrapper, music_description, total_duration, progress_callback=None):
    min_duration = 10
    
    if total_duration < min_duration:
        total_duration = min_duration
    
    if total_duration <= 250:
        music_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        music_temp.close()
        success = ace_wrapper.generate(
            lyrics="...",
            style_prompt=music_description,
            output_path=music_temp.name,
            duration=int(total_duration)
        )
        if not success:
            os.unlink(music_temp.name)
            return None
        return music_temp.name, None
    
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    chunk_size = 250
    
    num_chunks = math.ceil(total_duration / chunk_size)
    
    for i in range(num_chunks):
        if progress_callback:
            progress_callback(i, num_chunks)
        
        chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        chunk_files.append(chunk_file)
        
        if i == num_chunks - 1:
            current_duration = total_duration - (i * chunk_size)
            if current_duration < min_duration:
                current_duration = min_duration
        else:
            current_duration = chunk_size
        
        success = ace_wrapper.generate(
            lyrics="...",
            style_prompt=music_description,
            output_path=chunk_file,
            duration=int(current_duration)
        )
        
        if not success:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
    
    if progress_callback:
        progress_callback(num_chunks, num_chunks)
    
    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, 'w') as f:
        for chunk in chunk_files:
            f.write(f"file '{chunk}'\n")
    
    output_file = os.path.join(temp_dir, "music.wav")
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, '-y', output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.unlink(chunk_file)
    
    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    return output_file, temp_dir

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, mode, base_path=None, target_path=None, text=None, output_path=None,
                 voice_instruct=None, dialogue_data=None, voice_prompts=None, duration=None,
                 music_description=None, assignments=None, is_music=False):
        super().__init__()
        self.mode = mode
        self.base_path = base_path
        self.target_path = target_path
        self.text = text
        self.output_path = output_path
        self.voice_instruct = voice_instruct
        self.dialogue_data = dialogue_data
        self.voice_prompts = voice_prompts
        self.duration = duration
        self.music_description = music_description
        self.assignments = assignments
        self.is_music = is_music
        self.stt = None
        self.tts = None
        self.tts_voice_design = None
        self.seed_vc = None
        self.ace_tt = None

    def cleanup(self):
        if self.stt is not None:
            del self.stt
            self.stt = None
        if self.tts is not None:
            del self.tts
            self.tts = None
        if self.tts_voice_design is not None:
            del self.tts_voice_design
            self.tts_voice_design = None
        if self.seed_vc is not None:
            del self.seed_vc
            self.seed_vc = None
        if self.ace_tt is not None:
            del self.ace_tt
            self.ace_tt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(self):
        try:
            if self.mode == "analyze_base":
                self.status_signal.emit("Loading Whisper model...")
                self.stt = WhisperSTT()
                self.progress_signal.emit(20)
                self.status_signal.emit("Transcribing base audio...")
                result = self.stt.transcribe(self.base_path)
                self.progress_signal.emit(50)
                if result:
                    segments = []
                    for segment in result.get("segments", []):
                        segments.append({
                            "start": segment["start"],
                            "end": segment["end"],
                            "text": segment["text"].strip()
                        })
                    text = result.get("text", "").strip()
                    self.finished_signal.emit(json.dumps({"text": text, "segments": segments}))
                    self.cleanup()
                else:
                    self.error_signal.emit("Transcription failed")

            elif self.mode == "analyze_target":
                self.status_signal.emit("Loading Qwen-TTS model...")
                self.tts = QwenTTS()
                self.progress_signal.emit(50)
                self.status_signal.emit("Extracting voice characteristics...")
                success = self.tts.extract_voice(self.target_path)
                self.progress_signal.emit(70)
                if success:
                    self.finished_signal.emit("Voice extracted successfully")
                    self.cleanup()
                else:
                    self.error_signal.emit("Voice extraction failed")

            elif self.mode == "synthesize":
                self.status_signal.emit("Generating speech...")
                if self.tts is None:
                    self.tts = QwenTTS()
                    self.tts.extract_voice(self.target_path)
                self.progress_signal.emit(70)
                success = self.tts.synthesize(self.text, self.output_path)
                self.progress_signal.emit(100)
                if success and os.path.exists(self.output_path):
                    self.finished_signal.emit(self.output_path)
                    self.cleanup()
                else:
                    self.error_signal.emit("Synthesis failed")

            elif self.mode == "tts_voice_design":
                self.status_signal.emit("Loading Qwen-TTS VoiceDesign model...")
                self.tts_voice_design = QwenTTSVoiceDesign()
                self.progress_signal.emit(20)
                if self.tts_voice_design.model is None:
                    self.error_signal.emit("Failed to load VoiceDesign model")
                    return
                self.status_signal.emit("Generating speech with voice design...")
                success = self.tts_voice_design.synthesize(self.text, self.voice_instruct, self.output_path)
                self.progress_signal.emit(80)
                if success and os.path.exists(self.output_path):
                    self.finished_signal.emit(self.output_path)
                    self.cleanup()
                else:
                    self.error_signal.emit("VoiceDesign synthesis failed")

            elif self.mode == "tts_voice_design_dialogue":
                self.status_signal.emit("Loading Qwen-TTS VoiceDesign model...")
                self.tts_voice_design = QwenTTSVoiceDesign()
                self.progress_signal.emit(10)
                if self.tts_voice_design.model is None:
                    self.error_signal.emit("Failed to load VoiceDesign model")
                    return
                self.status_signal.emit("Generating dialogue...")
                dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                dialogue_temp.close()
                success, message = self.tts_voice_design.synthesize_dialogue(
                    self.dialogue_data,
                    self.voice_prompts,
                    dialogue_temp.name
                )
                if not success:
                    self.error_signal.emit(message)
                    return
                self.progress_signal.emit(50)
                if self.music_description:
                    self.status_signal.emit("Generating background music...")
                    try:
                        info = sf.info(dialogue_temp.name)
                        duration = info.duration
                        print(f"Dialogue duration: {duration:.2f}s")
                    except Exception as e:
                        print(f"Could not get audio duration with soundfile: {e}")
                        try:
                            info = torchaudio.info(dialogue_temp.name)
                            duration = info.num_frames / info.sample_rate
                        except Exception as e2:
                            print(f"Torchaudio also failed: {e2}")
                            duration = 30
                    del self.tts_voice_design
                    self.tts_voice_design = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.ace_tt = AceStepWrapper()
                    if self.ace_tt.handler is None:
                        self.error_signal.emit("Failed to load ACE-Step model")
                        return
                    self.progress_signal.emit(60)
                    def progress_callback(current, total):
                        progress = 60 + int((current / total) * 20)
                        self.progress_signal.emit(progress)
                    music_result = generate_background_music(self.ace_tt, self.music_description, duration, progress_callback)
                    del self.ace_tt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if music_result is None:
                        self.error_signal.emit("Background music generation failed")
                        return
                    music_temp_path, music_temp_dir = music_result
                    self.progress_signal.emit(85)
                    self.status_signal.emit("Mixing dialogue with music...")
                    mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    mixed_temp.close()
                    cmd = [
                        'ffmpeg', '-i', dialogue_temp.name, '-i', music_temp_path,
                        '-filter_complex', '[1:a]volume=0.35[music];[0:a][music]amix=inputs=2:duration=longest',
                        '-y', mixed_temp.name
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    shutil.rmtree(music_temp_dir, ignore_errors=True)
                    if result.returncode != 0:
                        self.error_signal.emit(f"FFmpeg mixing failed: {result.stderr}")
                        return
                    shutil.move(mixed_temp.name, self.output_path)
                    os.unlink(dialogue_temp.name)
                else:
                    shutil.move(dialogue_temp.name, self.output_path)
                self.progress_signal.emit(100)
                self.finished_signal.emit(self.output_path)
                self.cleanup()

            elif self.mode == "tts_vc_dialogue":
                self.status_signal.emit("Loading Qwen-TTS model...")
                self.tts = QwenTTS()
                self.progress_signal.emit(10)
                if self.tts.model is None:
                    self.error_signal.emit("Failed to load Qwen-TTS model")
                    return
                unique_chars = set()
                for _, char, _ in self.dialogue_data:
                    unique_chars.add(char.lower())
                voice_prompts = {}
                for char_lower in unique_chars:
                    audio_path = self.assignments[char_lower]
                    self.status_signal.emit(f"Extracting voice for '{char_lower}'...")
                    success = self.tts.extract_voice(audio_path)
                    if not success:
                        self.error_signal.emit(f"Voice extraction failed for {char_lower}")
                        return
                    voice_prompts[char_lower] = self.tts.voice_prompt
                temp_dir = tempfile.mkdtemp()
                temp_files = []
                try:
                    total = len(self.dialogue_data)
                    for i, (num, char, script_text) in enumerate(self.dialogue_data):
                        char_lower = char.lower()
                        self.status_signal.emit(f"Generating line {num}/{total} for '{char}'...")
                        progress = int((i / total) * 40)
                        self.progress_signal.emit(progress + 10)
                        self.tts.voice_prompt = voice_prompts[char_lower]
                        temp_file = os.path.join(temp_dir, f"line_{num}.wav")
                        temp_files.append((num, temp_file))
                        success = self.tts.synthesize(script_text, temp_file)
                        if not success:
                            self.error_signal.emit(f"Synthesis failed for line {num}")
                            return
                    self.progress_signal.emit(50)
                    temp_files.sort(key=lambda x: x[0])
                    dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    dialogue_temp.close()
                    concat_list = os.path.join(temp_dir, "concat_list.txt")
                    with open(concat_list, 'w') as f:
                        for _, tf in temp_files:
                            f.write(f"file '{tf}'\n")
                    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', dialogue_temp.name]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        self.error_signal.emit(f"FFmpeg concatenation failed: {result.stderr}")
                        return
                    self.progress_signal.emit(70)
                    if self.music_description:
                        self.status_signal.emit("Generating background music...")
                        try:
                            info = sf.info(dialogue_temp.name)
                            duration = info.duration
                            print(f"Dialogue duration: {duration:.2f}s")
                        except Exception as e:
                            print(f"Could not get audio duration with soundfile: {e}")
                            try:
                                info = torchaudio.info(dialogue_temp.name)
                                duration = info.num_frames / info.sample_rate
                            except Exception as e2:
                                print(f"Torchaudio also failed: {e2}")
                                duration = 30
                        music_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        del self.tts
                        self.tts = None
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        self.ace_tt = AceStepWrapper()
                        if self.ace_tt.handler is None:
                            self.error_signal.emit("Failed to load ACE-Step model")
                            return
                        self.progress_signal.emit(80)
                        def progress_callback(current, total):
                            progress = 80 + int((current / total) * 5)
                            self.progress_signal.emit(progress)
                        music_result = generate_background_music(self.ace_tt, self.music_description, duration, progress_callback)
                        del self.ace_tt
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if music_result is None:
                            self.error_signal.emit("Background music generation failed")
                            return
                        music_temp_path, music_temp_dir = music_result
                        self.progress_signal.emit(87)
                        self.status_signal.emit("Mixing dialogue with music...")
                        mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                        mixed_temp.close()
                        cmd = [
                            'ffmpeg', '-i', dialogue_temp.name, '-i', music_temp_path,
                            '-filter_complex', '[1:a]volume=0.35[music];[0:a][music]amix=inputs=2:duration=longest',
                            '-y', mixed_temp.name
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        shutil.rmtree(music_temp_dir, ignore_errors=True)
                        if result.returncode != 0:
                            self.error_signal.emit(f"FFmpeg mixing failed: {result.stderr}")
                            return
                        shutil.move(mixed_temp.name, self.output_path)
                        os.unlink(dialogue_temp.name)
                    else:
                        shutil.move(dialogue_temp.name, self.output_path)
                finally:
                    try:
                        shutil.rmtree(temp_dir)
                    except:
                        pass
                self.progress_signal.emit(100)
                self.finished_signal.emit(self.output_path)
                self.cleanup()

            elif self.mode == "seed_vc_convert":
                if self.is_music:
                    self.status_signal.emit("Loading Seed-VC v1 model...")
                    self.seed_vc = SeedVCV1()
                    self.progress_signal.emit(20)
                    if self.seed_vc.model is None:
                        self.error_signal.emit("Failed to load Seed-VC v1 model")
                        return
                    temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    try:
                        self.status_signal.emit("Resampling inputs to 44100Hz...")
                        waveform_base, sr_base = torchaudio.load(self.base_path)
                        if sr_base != 44100:
                            resampler_base = torchaudio.transforms.Resample(sr_base, 44100)
                            waveform_base = resampler_base(waveform_base)
                        torchaudio.save(temp_base.name, waveform_base, 44100)
                        waveform_target, sr_target = torchaudio.load(self.target_path)
                        if sr_target != 44100:
                            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
                            waveform_target = resampler_target(waveform_target)
                        torchaudio.save(temp_target.name, waveform_target, 44100)
                        self.progress_signal.emit(40)
                        self.status_signal.emit("Converting voice...")
                        success = self.seed_vc.convert(
                            source_path=temp_base.name,
                            reference_path=temp_target.name,
                            output_path=temp_output_44k.name
                        )
                        self.progress_signal.emit(70)
                        if success:
                            shutil.copy(temp_output_44k.name, self.output_path)
                            self.progress_signal.emit(90)
                            self.finished_signal.emit(self.output_path)
                            self.cleanup()
                        else:
                            self.error_signal.emit("Voice conversion failed")
                    finally:
                        for temp_file in [temp_base.name, temp_target.name, temp_output_44k.name]:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                else:
                    self.status_signal.emit("Loading Seed-VC v2 model...")
                    self.seed_vc = SeedVCV2()
                    self.progress_signal.emit(20)
                    if self.seed_vc.model is None:
                        self.error_signal.emit("Failed to load Seed-VC model")
                        return
                    temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    try:
                        self.status_signal.emit("Resampling inputs to 22050Hz...")
                        waveform_base, sr_base = torchaudio.load(self.base_path)
                        if sr_base != 22050:
                            resampler_base = torchaudio.transforms.Resample(sr_base, 22050)
                            waveform_base = resampler_base(waveform_base)
                        torchaudio.save(temp_base.name, waveform_base, 22050)
                        waveform_target, sr_target = torchaudio.load(self.target_path)
                        if sr_target != 22050:
                            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
                            waveform_target = resampler_target(waveform_target)
                        torchaudio.save(temp_target.name, waveform_target, 22050)
                        self.progress_signal.emit(40)
                        self.status_signal.emit("Converting voice...")
                        success = self.seed_vc.convert(
                            source_path=temp_base.name,
                            reference_path=temp_target.name,
                            output_path=temp_output_22k.name
                        )
                        self.progress_signal.emit(70)
                        if success:
                            self.status_signal.emit("Upsampling output to 44100Hz...")
                            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
                            if sr_out != 44100:
                                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                                waveform_out = resampler_out(waveform_out)
                            torchaudio.save(self.output_path, waveform_out, 44100)
                            self.progress_signal.emit(90)
                            self.finished_signal.emit(self.output_path)
                            self.cleanup()
                        else:
                            self.error_signal.emit("Voice conversion failed")
                    finally:
                        for temp_file in [temp_base.name, temp_target.name, temp_output_22k.name]:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)

            elif self.mode == "ttm_generate":
                self.status_signal.emit("Loading ACE-Step model...")
                self.ace_tt = AceStepWrapper()
                self.progress_signal.emit(20)
                if self.ace_tt.handler is None:
                    self.error_signal.emit("Failed to load ACE-Step model")
                    return
                duration = self.duration if self.duration else 30
                self.status_signal.emit(f"Generating music ({duration}s duration)...")
                self.progress_signal.emit(40)
                success = self.ace_tt.generate(
                    lyrics=self.text,
                    style_prompt=self.voice_instruct,
                    output_path=self.output_path,
                    duration=duration
                )
                self.progress_signal.emit(90)
                if success and os.path.exists(self.output_path):
                    self.finished_signal.emit(self.output_path)
                    self.cleanup()
                else:
                    self.error_signal.emit("Music generation failed")

            elif self.mode == "ttm_vc_generate":
                temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_ttm_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_target_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                try:
                    self.status_signal.emit("Loading ACE-Step model...")
                    self.ace_tt = AceStepWrapper()
                    self.progress_signal.emit(10)
                    if self.ace_tt.handler is None:
                        self.error_signal.emit("Failed to load ACE-Step model")
                        return
                    duration = self.duration if self.duration else 30
                    self.status_signal.emit(f"Generating music ({duration}s duration)...")
                    self.progress_signal.emit(30)
                    success = self.ace_tt.generate(
                        lyrics=self.text,
                        style_prompt=self.voice_instruct,
                        output_path=temp_ttm_output.name,
                        duration=duration
                    )
                    if not success or not os.path.exists(temp_ttm_output.name):
                        self.error_signal.emit("Music generation failed")
                        return
                    self.status_signal.emit("Resampling TTM output to 44100Hz...")
                    self.progress_signal.emit(50)
                    waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
                    if sr_ttm != 44100:
                        resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 44100)
                        waveform_ttm = resampler_ttm(waveform_ttm)
                    torchaudio.save(temp_ttm_22k.name, waveform_ttm, 44100)
                    self.status_signal.emit("Clearing ACE-Step from memory...")
                    del self.ace_tt
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.status_signal.emit("Resampling target voice to 44100Hz...")
                    self.progress_signal.emit(60)
                    waveform_target, sr_target = torchaudio.load(self.target_path)
                    if sr_target != 44100:
                        resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
                        waveform_target = resampler_target(waveform_target)
                    torchaudio.save(temp_target_22k.name, waveform_target, 44100)
                    self.status_signal.emit("Loading Seed-VC v1 model...")
                    self.seed_vc = SeedVCV1()
                    self.progress_signal.emit(70)
                    if self.seed_vc.model is None:
                        self.error_signal.emit("Failed to load Seed-VC v1 model")
                        return
                    self.status_signal.emit("Converting voice...")
                    self.progress_signal.emit(80)
                    vc_success = self.seed_vc.convert(
                        source_path=temp_ttm_22k.name,
                        reference_path=temp_target_22k.name,
                        output_path=temp_vc_output.name
                    )
                    if not vc_success:
                        self.error_signal.emit("Voice conversion failed")
                        return
                    self.status_signal.emit("Saving output...")
                    self.progress_signal.emit(95)
                    shutil.copy(temp_vc_output.name, self.output_path)
                    self.progress_signal.emit(100)
                    self.finished_signal.emit(self.output_path)
                    self.cleanup()
                finally:
                    for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output.name]:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
        except Exception as e:
            self.error_signal.emit(str(e))

def parse_dialogue_script(script_text):
    pattern = r'^(\d+)\s*:\s*([A-Za-z0-9_]+)\s*:\s*(.+)$'
    lines = script_text.strip().split('\n')
    items = []
    numbers_found = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if match:
            num = int(match.group(1))
            char = match.group(2)
            text = match.group(3).strip()
            items.append((num, char, text))
            numbers_found.add(num)
    if not items:
        return None, "No valid dialogue format found"
    expected_range = set(range(1, len(items) + 1))
    if numbers_found != expected_range:
        missing = expected_range - numbers_found
        if missing:
            return None, f"Missing dialogue numbers: {sorted(missing)}"
        else:
            return None, f"Unexpected dialogue numbers found"
    items.sort(key=lambda x: x[0])
    return items, None

def parse_voice_prompts(prompt_text):
    prompts = {}
    lines = prompt_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        if line.startswith('---'):
            continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            char = parts[0].strip()
            instruct = parts[1].strip()
            prompts[char.lower()] = instruct
            prompts[char] = instruct
    return prompts

def is_dialogue_mode(script_text):
    pattern = r'^(\d+)\s*:\s*[A-Za-z0-9_]+\s*:.+$'
    lines = [l.strip() for l in script_text.strip().split('\n') if l.strip()]
    if not lines:
        return False
    return all(re.match(pattern, line) for line in lines)

TTS_HELPER = """# Single Mode Example:
Character: Your dialogue text here.

# Dialogue Mode Example:
James: Welcome to our podcast! Today we'll discuss AI.
Sarah: Thanks James! I'm excited to share my research.
James: Let's start with the basics. What is AI?

# Voice prompts will appear below for each character automatically."""

TTM_HELPER = """# Example Song Structure:

Verse 1:
Walking down the empty street
Feeling the rhythm in my feet
The city lights are shining bright
Guiding me through the night

Chorus:
This is our moment, this is our time
Everything's gonna be just fine
Dancing under the moonlight
Everything feels so right

Verse 2:
The music plays, I start to move
Grooving to the funky groove
Don't care what tomorrow brings
Tonight my heart just sings"""

class BackgroundMusicDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Background Music")
        self.setModal(True)
        layout = QVBoxLayout()
        label = QLabel("Enter music description (or press Skip):")
        layout.addWidget(label)
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("e.g., soft piano, cinematic strings, ambient")
        layout.addWidget(self.text_edit)
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.setStyleSheet(get_main_button_style())
        self.ok_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn = QPushButton("Skip")
        self.skip_btn.setStyleSheet(get_secondary_button_style())
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.skip_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.ok_btn.clicked.connect(self.on_ok)
        self.skip_btn.clicked.connect(self.on_skip)
        self.result = None

    def on_ok(self):
        desc = self.text_edit.text().strip()
        if not desc:
            QMessageBox.warning(self, "Warning", "Description cannot be empty. Press Skip to skip music.")
            return
        self.result = desc
        self.accept()

    def on_skip(self):
        self.result = None
        self.reject()

class MusicInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Musical Inputs?")
        self.setModal(True)
        self.setMinimumWidth(350)
        self.result_value = False
        layout = QVBoxLayout(self)
        label = QLabel("Are the inputs musical?")
        label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 20px;")
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        yes_btn = QPushButton("Yes")
        yes_btn.setStyleSheet(get_secondary_button_style())
        yes_btn.setMinimumWidth(100)
        yes_btn.clicked.connect(self.on_yes)
        no_btn = QPushButton("No")
        no_btn.setStyleSheet(get_secondary_button_style())
        no_btn.setMinimumWidth(100)
        no_btn.clicked.connect(self.on_no)
        button_layout.addWidget(yes_btn)
        button_layout.addWidget(no_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)
        layout.setContentsMargins(20, 20, 20, 20)

    def on_yes(self):
        self.result_value = True
        self.accept()

    def on_no(self):
        self.result_value = False
        self.reject()

    def get_result(self):
        return self.result_value


class VODERGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VODER - Voice Blender")
        self.resize(1400, 900)
        self.setStyleSheet(get_window_style())
        self.setWindowIcon(self.load_icon())
        self.base_audio_path = None
        self.target_audio_path = None
        self.output_audio_path = None
        self.transcription_data = None
        self.voice_embedded = False
        self.original_cwd = os.getcwd()
        self.results_dir = os.path.join(self.original_cwd, "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.setup_ui()

    def load_icon(self):
        icon_path = self.get_resource_path("voder.png")
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return QIcon()

    def get_resource_path(self, relative_path):
        if hasattr(sys, "_MEIPASS"):
            return os.path.join(sys._MEIPASS, relative_path)
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        header_layout = QHBoxLayout()
        title = QLabel("VODER: Voice Blender")
        title.setStyleSheet(get_title_label_style())
        header_layout.addWidget(title, alignment=Qt.AlignCenter)
        header_layout.addStretch(1)
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet(get_subtitle_label_style())
        header_layout.addWidget(mode_label)
        self.mode_combo = QComboBox()
        self.mode_combo.setStyleSheet(get_combo_box_style())
        self.mode_combo.addItem("STT+TTS")
        self.mode_combo.addItem("TTS")
        self.mode_combo.addItem("TTS+VC")
        self.mode_combo.addItem("STS")
        self.mode_combo.addItem("TTM")
        self.mode_combo.addItem("TTM+VC")
        self.mode_combo.currentIndexChanged.connect(self.on_mode_changed)
        header_layout.addWidget(self.mode_combo)
        main_layout.addLayout(header_layout)

        subtitle = QLabel("They say what you want them to say.")
        subtitle.setStyleSheet(get_subtitle_label_style())
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        self.content_splitter = QSplitter(Qt.Horizontal)
        self.base_panel = self.create_audio_panel("Base Audio (Content)", True)
        self.content_splitter.addWidget(self.base_panel)
        self.work_panel = self.create_work_panel()
        self.content_splitter.addWidget(self.work_panel)
        self.target_panel = self.create_audio_panel("Target Audio (Voice)", False)
        self.content_splitter.addWidget(self.target_panel)
        self.tts_panel = self.create_tts_panel()
        self.content_splitter.addWidget(self.tts_panel)
        self.ttm_panel = self.create_ttm_panel()
        self.content_splitter.addWidget(self.ttm_panel)
        self.tts_vc_target_panel = self.create_tts_vc_target_panel()
        self.content_splitter.addWidget(self.tts_vc_target_panel)
        self.tts_panel.hide()
        self.tts_vc_target_panel.hide()
        self.content_splitter.setSizes([400, 600, 400, 0, 0, 0])
        main_layout.addWidget(self.content_splitter, stretch=1)

        self.output_panel = self.create_output_panel()
        main_layout.addWidget(self.output_panel)

        self.status_bar = QLabel("Ready")
        self.status_bar.setStyleSheet(get_status_bar_style())
        main_layout.addWidget(self.status_bar)

        self.progress = QProgressBar()
        self.progress.setStyleSheet(get_progress_bar_style())
        main_layout.addWidget(self.progress)

        self.worker = None
        self.check_ready()

    def on_mode_changed(self, index):
        mode = self.mode_combo.currentText()
        if mode == "STS":
            self.work_panel.hide()
            self.tts_panel.hide()
            self.ttm_panel.hide()
            self.tts_vc_target_panel.hide()
            self.base_panel.show()
            self.target_panel.show()
            self.base_analyze_btn.hide()
            self.target_analyze_btn.hide()
            self.sts_patch_btn.show()
            self.patch_btn.setText("Patch")
            try:
                self.patch_btn.clicked.disconnect()
            except:
                pass
            self.patch_btn.clicked.connect(self.patch_audio_sts)
            self.clear_btn.hide()
            self.text_edit.setEnabled(False)
            self.segments_list.setEnabled(False)
            self.ttm_patch_btn.hide()
            self.ttm_vc_patch_btn.hide()
            self.content_splitter.setSizes([500, 0, 500, 0, 0, 0])
        elif mode == "TTS":
            self.work_panel.hide()
            self.tts_panel.show()
            self.ttm_panel.hide()
            self.tts_vc_target_panel.hide()
            self.base_panel.hide()
            self.target_panel.hide()
            self.patch_btn.setText("Generate")
            try:
                self.patch_btn.clicked.disconnect()
            except:
                pass
            self.patch_btn.clicked.connect(self.patch_audio_tts)
            self.clear_btn.show()
            self.clear_btn.clicked.connect(self.clear_tts_inputs)
            self.ttm_patch_btn.hide()
            self.ttm_vc_patch_btn.hide()
            self.content_splitter.setSizes([0, 0, 0, 1400, 0, 0])
            self.tts_voice_prompt_widget.set_mode('text')
            self.tts_script_widget.clear()
            self.tts_voice_prompt_widget.clear()
            self.update_tts_prompts_from_script()
            self.tts_script_widget.characters_changed.connect(self.update_tts_prompts_from_script)
            self.tts_voice_prompt_widget.prompts_changed.connect(self.check_ready)
        elif mode == "TTS+VC":
            self.work_panel.hide()
            self.tts_panel.show()
            self.ttm_panel.hide()
            self.base_panel.hide()
            self.target_panel.hide()
            self.target_analyze_btn.hide()
            self.tts_vc_target_panel.show()
            self.patch_btn.setText("Generate")
            try:
                self.patch_btn.clicked.disconnect()
            except:
                pass
            self.patch_btn.clicked.connect(self.patch_audio_tts_vc)
            self.clear_btn.show()
            self.clear_btn.clicked.connect(self.clear_tts_inputs)
            self.ttm_patch_btn.hide()
            self.ttm_vc_patch_btn.hide()
            self.content_splitter.setSizes([0, 0, 0, 700, 0, 700])
            self.tts_voice_prompt_widget.set_mode('combo')
            self.tts_script_widget.clear()
            self.tts_voice_prompt_widget.clear()
            self.update_tts_prompts_from_script()
            self.tts_script_widget.characters_changed.connect(self.update_tts_prompts_from_script)
            self.tts_voice_prompt_widget.prompts_changed.connect(self.check_ready)
            self.update_audio_numbers_in_prompts()
        elif mode == "TTM":
            self.work_panel.hide()
            self.tts_panel.hide()
            self.ttm_panel.show()
            self.tts_vc_target_panel.hide()
            self.base_panel.hide()
            self.target_panel.hide()
            self.patch_btn.hide()
            self.clear_btn.hide()
            self.ttm_patch_btn.show()
            self.ttm_vc_patch_btn.hide()
            self.ttm_clear_btn.show()
            try:
                self.ttm_patch_btn.clicked.disconnect()
            except:
                pass
            self.ttm_patch_btn.clicked.connect(self.patch_audio_ttm)
            self.content_splitter.setSizes([0, 0, 0, 0, 1400, 0])
        elif mode == "TTM+VC":
            self.work_panel.hide()
            self.tts_panel.hide()
            self.tts_vc_target_panel.hide()
            self.base_panel.hide()
            self.target_panel.show()
            self.target_analyze_btn.hide()
            self.ttm_panel.show()
            self.patch_btn.hide()
            self.clear_btn.hide()
            self.ttm_patch_btn.hide()
            self.ttm_vc_patch_btn.show()
            self.ttm_clear_btn.show()
            try:
                self.ttm_vc_patch_btn.clicked.disconnect()
            except:
                pass
            self.ttm_vc_patch_btn.clicked.connect(self.patch_audio_ttm_vc)
            self.content_splitter.setSizes([0, 0, 700, 0, 700, 0])
        else:
            self.work_panel.show()
            self.tts_panel.hide()
            self.ttm_panel.hide()
            self.tts_vc_target_panel.hide()
            self.base_panel.show()
            self.target_panel.show()
            self.base_analyze_btn.show()
            self.target_analyze_btn.show()
            self.sts_patch_btn.hide()
            self.patch_btn.setText("Patch")
            self.patch_btn.show()
            try:
                self.patch_btn.clicked.disconnect()
            except:
                pass
            self.patch_btn.clicked.connect(self.patch_audio)
            self.clear_btn.show()
            self.clear_btn.clicked.connect(self.clear_text)
            self.ttm_patch_btn.hide()
            self.ttm_vc_patch_btn.hide()
            self.content_splitter.setSizes([400, 600, 400, 0, 0, 0])
        self.check_ready()

    def create_audio_panel(self, title, is_base):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        waveform = AudioWaveformWidget()
        waveform.setMinimumHeight(120)
        if is_base:
            self.base_waveform = waveform
        else:
            self.target_waveform = waveform
        layout.addWidget(waveform)

        info_lbl = QLabel("No audio loaded")
        info_lbl.setStyleSheet(get_subtitle_label_style())
        info_lbl.setAlignment(Qt.AlignCenter)
        if is_base:
            self.base_info = info_lbl
        else:
            self.target_info = info_lbl
        layout.addWidget(info_lbl)

        btn_layout = QHBoxLayout()
        load_btn = QPushButton("Load Audio/Video")
        load_btn.setStyleSheet(get_main_button_style())
        load_btn.setCursor(Qt.PointingHandCursor)
        if is_base:
            load_btn.clicked.connect(self.load_base)
        else:
            load_btn.clicked.connect(self.load_target)
        btn_layout.addWidget(load_btn)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.setStyleSheet(get_secondary_button_style())
        analyze_btn.setCursor(Qt.PointingHandCursor)
        analyze_btn.setEnabled(False)
        if is_base:
            self.base_analyze_btn = analyze_btn
            analyze_btn.clicked.connect(self.analyze_base)
        else:
            self.target_analyze_btn = analyze_btn
            analyze_btn.clicked.connect(self.analyze_target)
        btn_layout.addWidget(analyze_btn)
        layout.addLayout(btn_layout)

        play_patch_layout = QHBoxLayout()
        play_btn = QPushButton("Play")
        play_btn.setStyleSheet(get_surface_button_style())
        play_btn.setCursor(Qt.PointingHandCursor)
        play_btn.setEnabled(False)
        if is_base:
            self.base_play_btn = play_btn
            play_btn.clicked.connect(lambda: self.play_audio(self.base_audio_path))
        else:
            self.target_play_btn = play_btn
            play_btn.clicked.connect(lambda: self.play_audio(self.target_audio_path))
        play_patch_layout.addWidget(play_btn)

        if is_base:
            self.sts_patch_btn = QPushButton("Patch")
            self.sts_patch_btn.setStyleSheet(get_main_button_style())
            self.sts_patch_btn.setCursor(Qt.PointingHandCursor)
            self.sts_patch_btn.setEnabled(False)
            self.sts_patch_btn.setVisible(False)
            self.sts_patch_btn.clicked.connect(self.patch_audio_sts)
            play_patch_layout.addWidget(self.sts_patch_btn)
        layout.addLayout(play_patch_layout)

        return panel

    def create_work_panel(self):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel("Transcription & Editing")
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setStyleSheet(get_text_edit_style())
        self.text_edit.setPlaceholderText("Transcribed text will appear here...\nYou can edit this text and click 'Patch' to synthesize with target voice.")
        self.text_edit.setEnabled(False)
        layout.addWidget(self.text_edit, stretch=1)

        self.segments_list = QListWidget()
        self.segments_list.setStyleSheet(get_list_widget_style())
        self.segments_list.setMaximumHeight(150)
        self.segments_list.setEnabled(False)
        layout.addWidget(self.segments_list)

        controls_layout = QHBoxLayout()
        self.patch_btn = QPushButton("Patch")
        self.patch_btn.setStyleSheet(get_main_button_style())
        self.patch_btn.setCursor(Qt.PointingHandCursor)
        self.patch_btn.setEnabled(False)
        self.patch_btn.clicked.connect(self.patch_audio)
        controls_layout.addWidget(self.patch_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(get_surface_button_style())
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_text)
        controls_layout.addWidget(self.clear_btn)
        layout.addLayout(controls_layout)

        return panel

    def create_tts_panel(self):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel("Text-to-Speech")
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        script_label = QLabel("Script")
        script_label.setStyleSheet(get_subtitle_label_style())
        layout.addWidget(script_label)

        self.tts_script_widget = DialogueScriptWidget()
        layout.addWidget(self.tts_script_widget, stretch=3)

        prompt_label = QLabel("Voice Prompt")
        prompt_label.setStyleSheet(get_subtitle_label_style())
        layout.addWidget(prompt_label)

        self.tts_voice_prompt_widget = VoicePromptWidget()
        self.tts_voice_prompt_widget.set_mode('text')
        layout.addWidget(self.tts_voice_prompt_widget, stretch=2)

        controls_layout = QHBoxLayout()
        self.patch_btn = QPushButton("Generate")
        self.patch_btn.setStyleSheet(get_main_button_style())
        self.patch_btn.setCursor(Qt.PointingHandCursor)
        self.patch_btn.clicked.connect(self.patch_audio_tts)
        controls_layout.addWidget(self.patch_btn)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(get_surface_button_style())
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_tts_inputs)
        controls_layout.addWidget(self.clear_btn)
        layout.addLayout(controls_layout)

        return panel

    def create_ttm_panel(self):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel("Text-to-Music")
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        lyrics_label = QLabel("Song Lyrics")
        lyrics_label.setStyleSheet(get_subtitle_label_style())
        layout.addWidget(lyrics_label)

        self.ttm_lyrics_edit = QTextEdit()
        self.ttm_lyrics_edit.setStyleSheet(get_text_edit_style())
        self.ttm_lyrics_edit.setMinimumHeight(150)
        self.ttm_lyrics_edit.setPlaceholderText(TTM_HELPER)
        self.ttm_lyrics_edit.textChanged.connect(self.check_ready)
        layout.addWidget(self.ttm_lyrics_edit, stretch=1)

        prompt_label = QLabel("Style Prompt")
        prompt_label.setStyleSheet(get_subtitle_label_style())
        layout.addWidget(prompt_label)

        self.ttm_prompt_edit = QTextEdit()
        self.ttm_prompt_edit.setStyleSheet(get_text_edit_style())
        self.ttm_prompt_edit.setMinimumHeight(80)
        self.ttm_prompt_edit.setPlaceholderText("# Describe the music style:\nupbeat pop with male vocals, energetic drums, synth bass, cheerful melody\n\n# OR detailed:\ngenre: electronic pop, vocals: female soft dreamy, instruments: piano strings, mood: romantic relaxing")
        self.ttm_prompt_edit.textChanged.connect(self.check_ready)
        layout.addWidget(self.ttm_prompt_edit)

        duration_layout = QHBoxLayout()
        duration_label = QLabel("Duration:")
        duration_label.setStyleSheet(get_subtitle_label_style())
        duration_layout.addWidget(duration_label)
        self.ttm_minutes_spin = QSpinBox()
        self.ttm_minutes_spin.setStyleSheet(get_text_edit_style())
        self.ttm_minutes_spin.setRange(0, 5)
        self.ttm_minutes_spin.setValue(0)
        self.ttm_minutes_spin.setSuffix(" m")
        duration_layout.addWidget(self.ttm_minutes_spin)
        self.ttm_seconds_spin = QSpinBox()
        self.ttm_seconds_spin.setStyleSheet(get_text_edit_style())
        self.ttm_seconds_spin.setRange(10, 59)
        self.ttm_seconds_spin.setValue(0)
        self.ttm_seconds_spin.setSuffix(" s")
        duration_layout.addWidget(self.ttm_seconds_spin)
        self.ttm_minutes_spin.valueChanged.connect(self.on_ttm_minutes_changed)
        duration_layout.addStretch(1)
        layout.addLayout(duration_layout)

        controls_layout = QHBoxLayout()
        self.ttm_patch_btn = QPushButton("Generate")
        self.ttm_patch_btn.setStyleSheet(get_main_button_style())
        self.ttm_patch_btn.setCursor(Qt.PointingHandCursor)
        self.ttm_patch_btn.hide()
        controls_layout.addWidget(self.ttm_patch_btn)
        self.ttm_vc_patch_btn = QPushButton("Generate")
        self.ttm_vc_patch_btn.setStyleSheet(get_main_button_style())
        self.ttm_vc_patch_btn.setCursor(Qt.PointingHandCursor)
        self.ttm_vc_patch_btn.hide()
        controls_layout.addWidget(self.ttm_vc_patch_btn)
        self.ttm_clear_btn = QPushButton("Clear")
        self.ttm_clear_btn.setStyleSheet(get_surface_button_style())
        self.ttm_clear_btn.setCursor(Qt.PointingHandCursor)
        self.ttm_clear_btn.clicked.connect(self.clear_ttm_inputs)
        self.ttm_clear_btn.hide()
        controls_layout.addWidget(self.ttm_clear_btn)
        layout.addLayout(controls_layout)

        return panel

    def on_ttm_minutes_changed(self, minutes):
        if minutes == 5:
            self.ttm_seconds_spin.setValue(0)
            self.ttm_seconds_spin.setEnabled(False)
            self.ttm_seconds_spin.lineEdit().setReadOnly(True)
        else:
            self.ttm_seconds_spin.setEnabled(True)
            self.ttm_seconds_spin.lineEdit().setReadOnly(False)

    def create_tts_vc_target_panel(self):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel("Voice Reference Files")
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        subtitle_lbl = QLabel("Add audio files for voice cloning")
        subtitle_lbl.setStyleSheet(get_subtitle_label_style())
        subtitle_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle_lbl)

        self.tts_vc_audio_list = QListWidget()
        self.tts_vc_audio_list.setStyleSheet(get_list_widget_style())
        self.tts_vc_audio_list.setMinimumHeight(200)
        layout.addWidget(self.tts_vc_audio_list)

        btn_layout = QHBoxLayout()
        self.tts_vc_add_btn = QPushButton("Add Audio")
        self.tts_vc_add_btn.setStyleSheet(get_main_button_style())
        self.tts_vc_add_btn.setCursor(Qt.PointingHandCursor)
        self.tts_vc_add_btn.clicked.connect(self.tts_vc_add_audio)
        btn_layout.addWidget(self.tts_vc_add_btn)
        layout.addLayout(btn_layout)

        self.tts_vc_audio_files = {}
        self.tts_vc_next_number = 1
        return panel

    def tts_vc_add_audio(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Add Voice Reference Audio", "",
                                               "Audio Files (*.wav *.mp3 *.flac *.m4a)")
        if fname:
            audio_number = self.tts_vc_next_number
            self.tts_vc_next_number += 1
            self.tts_vc_audio_files[audio_number] = fname
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 14, 5, 14)
            item_layout.setSpacing(10)
            name_lbl = QLabel(f"{audio_number}")
            name_lbl.setStyleSheet(f"color: {THEME['text']}; font-weight: bold; min-width: 30px;")
            item_layout.addWidget(name_lbl)
            play_btn = QPushButton("Play")
            play_btn.setStyleSheet(get_surface_button_style())
            play_btn.setCursor(Qt.PointingHandCursor)
            play_btn.setFixedWidth(60)
            play_btn.setMinimumHeight(35)
            play_btn.clicked.connect(lambda: self.tts_vc_play_audio(audio_number))
            item_layout.addWidget(play_btn)
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet(get_surface_button_style())
            delete_btn.setCursor(Qt.PointingHandCursor)
            delete_btn.setFixedWidth(60)
            delete_btn.setMinimumHeight(35)
            delete_btn.clicked.connect(lambda: self.tts_vc_delete_audio(audio_number, item_widget))
            item_layout.addWidget(delete_btn)
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            self.tts_vc_audio_list.addItem(item)
            self.tts_vc_audio_list.setItemWidget(item, item_widget)
            self.update_audio_numbers_in_prompts()

    def tts_vc_play_audio(self, audio_number):
        if audio_number in self.tts_vc_audio_files:
            audio_path = self.tts_vc_audio_files[audio_number]
            if os.path.exists(audio_path):
                self.play_audio(audio_path)

    def tts_vc_delete_audio(self, audio_number, item_widget):
        if audio_number in self.tts_vc_audio_files:
            del self.tts_vc_audio_files[audio_number]
            for i in range(self.tts_vc_audio_list.count()):
                item = self.tts_vc_audio_list.item(i)
                if self.tts_vc_audio_list.itemWidget(item) == item_widget:
                    self.tts_vc_audio_list.takeItem(i)
                    break
            self.update_audio_numbers_in_prompts()

    def tts_vc_get_audio_count(self):
        return len(self.tts_vc_audio_files)

    def tts_vc_get_audio_path(self, audio_number):
        return self.tts_vc_audio_files.get(audio_number, None)

    def tts_vc_get_all_audio_files(self):
        return self.tts_vc_audio_files.copy()

    def update_audio_numbers_in_prompts(self):
        if hasattr(self, 'tts_voice_prompt_widget'):
            numbers = [str(num) for num in sorted(self.tts_vc_audio_files.keys())]
            self.tts_voice_prompt_widget.set_audio_numbers(numbers)

    def update_tts_prompts_from_script(self):
        if hasattr(self, 'tts_script_widget') and hasattr(self, 'tts_voice_prompt_widget'):
            items = self.tts_script_widget.get_dialogue_items()
            chars = set()
            for _, char, _ in items:
                chars.add(char.lower())
            self.tts_voice_prompt_widget.set_characters(chars)

    def create_output_panel(self):
        panel = QFrame()
        panel.setStyleSheet(get_panel_style())
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title_lbl = QLabel("Output Preview")
        title_lbl.setStyleSheet(get_title_label_style())
        title_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_lbl)

        self.output_waveform = AudioWaveformWidget()
        self.output_waveform.setMinimumHeight(80)
        layout.addWidget(self.output_waveform)

        btn_layout = QHBoxLayout()
        self.output_play_btn = QPushButton("Play")
        self.output_play_btn.setStyleSheet(get_secondary_button_style())
        self.output_play_btn.setCursor(Qt.PointingHandCursor)
        self.output_play_btn.setEnabled(False)
        self.output_play_btn.clicked.connect(lambda: self.play_audio(self.output_audio_path))
        btn_layout.addWidget(self.output_play_btn)
        layout.addLayout(btn_layout)

        return panel

    def extract_audio_from_video(self, video_path):
        try:
            temp_dir = tempfile.gettempdir()
            audio_path = os.path.join(temp_dir, f"voder_{int(time.time())}.wav")
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', audio_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if os.path.exists(audio_path):
                return audio_path
            return None
        except Exception as e:
            print(f"FFmpeg error: {e}")
            return None

    def load_base(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Base Audio/Video", "",
                                               "Audio/Video Files (*.wav *.mp3 *.flac *.m4a *.mp4 *.avi *.mov *.mkv)")
        if fname:
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.status_bar.setText("Extracting audio from video...")
                audio_path = self.extract_audio_from_video(fname)
                if audio_path:
                    self.base_audio_path = audio_path
                else:
                    QMessageBox.warning(self, "Error", "Could not extract audio from video")
                    return
            else:
                self.base_audio_path = fname
            self.base_waveform.set_audio(self.base_audio_path)
            try:
                info = torchaudio.info(self.base_audio_path)
                duration = info.num_frames / info.sample_rate
                self.base_info.setText(f"{os.path.basename(fname)}\n{duration:.1f}s | {info.sample_rate}Hz")
            except:
                self.base_info.setText(os.path.basename(fname))
            self.base_analyze_btn.setEnabled(True)
            self.base_play_btn.setEnabled(True)
            self.check_ready()

    def load_target(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Target Voice Audio/Video", "",
                                               "Audio/Video Files (*.wav *.mp3 *.flac *.m4a *.mp4 *.avi *.mov *.mkv)")
        if fname:
            if fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                self.status_bar.setText("Extracting audio from video...")
                audio_path = self.extract_audio_from_video(fname)
                if audio_path:
                    self.target_audio_path = audio_path
                else:
                    QMessageBox.warning(self, "Error", "Could not extract audio from video")
                    return
            else:
                self.target_audio_path = fname
            self.target_waveform.set_audio(self.target_audio_path)
            try:
                info = torchaudio.info(self.target_audio_path)
                duration = info.num_frames / info.sample_rate
                self.target_info.setText(f"{os.path.basename(fname)}\n{duration:.1f}s | {info.sample_rate}Hz")
            except:
                self.target_info.setText(os.path.basename(fname))
            self.target_analyze_btn.setEnabled(True)
            self.target_play_btn.setEnabled(True)
            self.check_ready()

    def check_ready(self):
        mode = self.mode_combo.currentText()
        if mode == "STS":
            if self.base_audio_path and self.target_audio_path:
                self.patch_btn.setEnabled(True)
                self.sts_patch_btn.setEnabled(True)
            else:
                self.patch_btn.setEnabled(False)
                self.sts_patch_btn.setEnabled(False)
        elif mode == "TTS+VC":
            script_valid, _ = self.tts_script_widget.validate()
            if not script_valid:
                self.patch_btn.setEnabled(False)
            else:
                if self.tts_vc_get_audio_count() == 0:
                    self.patch_btn.setEnabled(False)
                else:
                    if self.tts_voice_prompt_widget.has_all_prompts():
                        self.patch_btn.setEnabled(True)
                    else:
                        self.patch_btn.setEnabled(False)
        elif mode == "TTS":
            script_valid, _ = self.tts_script_widget.validate()
            if not script_valid:
                self.patch_btn.setEnabled(False)
            else:
                if self.tts_voice_prompt_widget.has_all_prompts():
                    self.patch_btn.setEnabled(True)
                else:
                    self.patch_btn.setEnabled(False)
        elif mode == "TTM":
            lyrics = self.ttm_lyrics_edit.toPlainText().strip()
            style_prompt = self.ttm_prompt_edit.toPlainText().strip()
            if lyrics and style_prompt:
                self.ttm_patch_btn.setEnabled(True)
            else:
                self.ttm_patch_btn.setEnabled(False)
        elif mode == "TTM+VC":
            lyrics = self.ttm_lyrics_edit.toPlainText().strip()
            style_prompt = self.ttm_prompt_edit.toPlainText().strip()
            has_target = self.target_audio_path is not None
            if lyrics and style_prompt and has_target:
                self.ttm_vc_patch_btn.setEnabled(True)
            else:
                self.ttm_vc_patch_btn.setEnabled(False)
        else:
            if self.transcription_data and self.voice_embedded:
                self.patch_btn.setEnabled(True)
            else:
                self.patch_btn.setEnabled(False)
        if mode in ("TTM", "TTM+VC"):
            self.ttm_clear_btn.setEnabled(True)
        elif mode in ("TTS", "TTS+VC"):
            self.clear_btn.setEnabled(True)

    def analyze_base(self):
        if not self.base_audio_path:
            return
        self.set_processing_state(True)
        self.status_bar.setText("Analyzing base audio with Whisper...")
        self.progress.setValue(0)
        self.worker = ProcessingThread("analyze_base", base_path=self.base_audio_path)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_base_analyzed)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_base_analyzed(self, result_json):
        try:
            data = json.loads(result_json)
            self.transcription_data = data
            self.text_edit.setText(data["text"])
            self.text_edit.setEnabled(True)
            self.segments_list.clear()
            for seg in data.get("segments", []):
                item_text = f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, seg)
                self.segments_list.addItem(item)
            self.segments_list.setEnabled(True)
            self.status_bar.setText("Base audio transcribed successfully")
            self.check_ready()
        except Exception as e:
            self.on_error(f"Failed to parse transcription: {e}")
        finally:
            self.set_processing_state(False)

    def analyze_target(self):
        if not self.target_audio_path:
            return
        self.set_processing_state(True)
        self.status_bar.setText("Analyzing target voice...")
        self.progress.setValue(0)
        self.worker = ProcessingThread("analyze_target", target_path=self.target_audio_path)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_target_analyzed)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_target_analyzed(self, message):
        self.voice_embedded = True
        self.status_bar.setText(f"Target voice: {message}")
        self.check_ready()
        self.set_processing_state(False)

    def patch_audio(self):
        if not self.transcription_data or not self.voice_embedded:
            return
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "Error", "No text to synthesize")
            return
        self.set_processing_state(True)
        self.status_bar.setText("Synthesizing with target voice...")
        self.progress.setValue(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"voder_output_{timestamp}.wav")
        self.worker = ProcessingThread("synthesize", target_path=self.target_audio_path,
                                       text=text, output_path=output_path)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def patch_audio_tts(self):
        script_valid, script_msg = self.tts_script_widget.validate()
        if not script_valid:
            QMessageBox.warning(self, "Script Error", script_msg)
            return
        dialogue_items = self.tts_script_widget.get_dialogue_items()
        if not dialogue_items:
            QMessageBox.warning(self, "Error", "No dialogue entered.")
            return
        prompts = self.tts_voice_prompt_widget.get_all_prompts()
        valid_prompts = {char: prompt for char, prompt in prompts.items() if prompt is not None}
        missing = []
        for _, char, _ in dialogue_items:
            char_lower = char.lower()
            if char_lower not in valid_prompts or not valid_prompts[char_lower]:
                missing.append(char)
        if missing:
            QMessageBox.warning(self, "Missing Voice Prompts",
                                f"The following characters have no voice prompt:\n{', '.join(set(missing))}")
            return
        music_description = None
        if len(dialogue_items) > 1:
            dlg = BackgroundMusicDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                music_description = dlg.result
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"voder_tts_dialogue_{timestamp}"
        if music_description:
            base_name += "_m"
        output_path = os.path.join(self.results_dir, f"{base_name}.wav")
        self.set_processing_state(True)
        self.status_bar.setText("Processing dialogue..." + (" with music" if music_description else ""))
        self.progress.setValue(0)
        self.worker = ProcessingThread("tts_voice_design_dialogue",
                                       dialogue_data=dialogue_items,
                                       voice_prompts=valid_prompts,
                                       output_path=output_path,
                                       music_description=music_description)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def patch_audio_tts_vc(self):
        script_valid, script_msg = self.tts_script_widget.validate()
        if not script_valid:
            QMessageBox.warning(self, "Script Error", script_msg)
            return
        dialogue_items = self.tts_script_widget.get_dialogue_items()
        if not dialogue_items:
            QMessageBox.warning(self, "Error", "No dialogue entered.")
            return
        if self.tts_vc_get_audio_count() == 0:
            QMessageBox.warning(self, "Error", "No voice reference audio files loaded.")
            return
        assignments = self.tts_voice_prompt_widget.get_all_prompts()
        valid_assignments = {char: num for char, num in assignments.items() if num is not None}
        missing = []
        for _, char, _ in dialogue_items:
            char_lower = char.lower()
            if char_lower not in valid_assignments:
                missing.append(char)
        if missing:
            QMessageBox.warning(self, "Missing Audio Assignments",
                                f"The following characters have no audio file assigned:\n{', '.join(set(missing))}")
            return
        audio_files = self.tts_vc_get_all_audio_files()
        assignments_paths = {}
        for char, num_str in valid_assignments.items():
            try:
                num = int(num_str)
            except:
                QMessageBox.warning(self, "Invalid Audio Number",
                                    f"Invalid audio number for character '{char}': {num_str}")
                return
            if num not in audio_files:
                QMessageBox.warning(self, "Audio File Missing",
                                    f"Audio file number {num} not found. It may have been deleted.")
                return
            assignments_paths[char] = audio_files[num]
        if len(dialogue_items) == 1:
            _, char, text = dialogue_items[0]
            audio_path = assignments_paths[char.lower()]
            self.generate_tts_vc_single(text, audio_path)
        else:
            music_description = None
            dlg = BackgroundMusicDialog(self)
            if dlg.exec_() == QDialog.Accepted:
                music_description = dlg.result
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"voder_tts_vc_dialogue_{timestamp}"
            if music_description:
                base_name += "_m"
            output_path = os.path.join(self.results_dir, f"{base_name}.wav")
            self.set_processing_state(True)
            self.status_bar.setText("Processing dialogue with voice clone..." + (" with music" if music_description else ""))
            self.progress.setValue(0)
            self.worker = ProcessingThread("tts_vc_dialogue",
                                           dialogue_data=dialogue_items,
                                           assignments=assignments_paths,
                                           output_path=output_path,
                                           music_description=music_description)
            self.worker.progress_signal.connect(self.progress.setValue)
            self.worker.status_signal.connect(self.status_bar.setText)
            self.worker.finished_signal.connect(self.on_synthesis_finished)
            self.worker.error_signal.connect(self.on_error)
            self.worker.start()

    def generate_tts_vc_single(self, script_text, audio_path):
        self.set_processing_state(True)
        self.status_bar.setText("Extracting voice from reference...")
        self.progress.setValue(0)
        tts = QwenTTS()
        success = tts.extract_voice(audio_path)
        if not success:
            QMessageBox.warning(self, "Error", "Failed to extract voice from reference audio")
            self.set_processing_state(False)
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"voder_tts_vc_single_{timestamp}.wav")
        self.status_bar.setText("Generating speech with cloned voice...")
        self.progress.setValue(50)
        success = tts.synthesize(script_text, output_path)
        if success and os.path.exists(output_path):
            self.on_synthesis_finished(output_path)
        else:
            QMessageBox.warning(self, "Error", "Speech generation failed")
            self.set_processing_state(False)

    def patch_audio_sts(self):
        if not self.base_audio_path or not self.target_audio_path:
            return
        dialog = MusicInputDialog(self)
        dialog.exec_()
        is_music = dialog.get_result()
        mode_str = "M-STS" if is_music else "STS"
        self.set_processing_state(True)
        self.status_bar.setText(f"Converting voice with Seed-VC ({mode_str})...")
        self.progress.setValue(0)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"voder_m_sts_{timestamp}.wav" if is_music else f"voder_sts_output_{timestamp}.wav"
        output_path = os.path.join(self.results_dir, output_filename)
        self.worker = ProcessingThread("seed_vc_convert", base_path=self.base_audio_path,
                                       target_path=self.target_audio_path, output_path=output_path,
                                       is_music=is_music)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def patch_audio_ttm(self):
        lyrics_text = self.ttm_lyrics_edit.toPlainText().strip()
        style_prompt = self.ttm_prompt_edit.toPlainText().strip()
        minutes = self.ttm_minutes_spin.value()
        seconds = self.ttm_seconds_spin.value()
        duration = minutes * 60 + seconds
        duration = max(10, min(300, duration))
        if not lyrics_text:
            QMessageBox.warning(self, "Error", "Please enter song lyrics")
            return
        if not style_prompt:
            QMessageBox.warning(self, "Error", "Please enter style prompt")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"voder_ttm_output_{timestamp}.wav")
        self.set_processing_state(True)
        self.status_bar.setText("Generating music with ACE-Step...")
        self.progress.setValue(0)
        self.worker = ProcessingThread("ttm_generate",
                                       text=lyrics_text,
                                       voice_instruct=style_prompt,
                                       output_path=output_path,
                                       duration=duration)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def patch_audio_ttm_vc(self):
        lyrics_text = self.ttm_lyrics_edit.toPlainText().strip()
        style_prompt = self.ttm_prompt_edit.toPlainText().strip()
        minutes = self.ttm_minutes_spin.value()
        seconds = self.ttm_seconds_spin.value()
        duration = minutes * 60 + seconds
        duration = max(10, min(300, duration))
        if not lyrics_text:
            QMessageBox.warning(self, "Error", "Please enter song lyrics")
            return
        if not style_prompt:
            QMessageBox.warning(self, "Error", "Please enter style prompt")
            return
        if not self.target_audio_path:
            QMessageBox.warning(self, "Error", "Please load target voice audio")
            return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.results_dir, f"voder_ttm_vc_output_{timestamp}.wav")
        self.set_processing_state(True)
        self.status_bar.setText("Generating music with TTM+VC...")
        self.progress.setValue(0)
        self.worker = ProcessingThread("ttm_vc_generate",
                                       text=lyrics_text,
                                       voice_instruct=style_prompt,
                                       target_path=self.target_audio_path,
                                       output_path=output_path,
                                       duration=duration)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def on_synthesis_finished(self, output_path):
        self.output_audio_path = output_path
        self.output_waveform.set_audio(output_path)
        self.output_play_btn.setEnabled(True)
        self.status_bar.setText(f"Conversion complete: {os.path.basename(output_path)}")
        self.set_processing_state(False)

    def play_audio(self, audio_path):
        if not audio_path or not os.path.exists(audio_path):
            return
        try:
            if sys.platform == "darwin":
                subprocess.run(["afplay", audio_path])
            elif sys.platform == "win32":
                os.startfile(audio_path)
            else:
                subprocess.run(["aplay", audio_path], stderr=subprocess.DEVNULL)
        except:
            pass

    def clear_text(self):
        self.text_edit.clear()
        if self.transcription_data:
            self.text_edit.setText(self.transcription_data.get("text", ""))

    def clear_tts_inputs(self):
        self.tts_script_widget.clear()
        self.tts_voice_prompt_widget.clear()

    def clear_ttm_inputs(self):
        self.ttm_lyrics_edit.clear()
        self.ttm_prompt_edit.clear()
        self.ttm_minutes_spin.setValue(0)
        self.ttm_seconds_spin.setValue(30)

    def set_processing_state(self, processing):
        mode = self.mode_combo.currentText()
        if mode == "STS":
            self.base_analyze_btn.setEnabled(False)
            self.target_analyze_btn.setEnabled(False)
            self.patch_btn.setEnabled(False)
            self.sts_patch_btn.setEnabled(False)
        elif mode == "TTS":
            if processing:
                self.patch_btn.setEnabled(False)
            else:
                self.check_ready()
        elif mode == "TTS+VC":
            if processing:
                self.patch_btn.setEnabled(False)
            else:
                self.check_ready()
        elif mode == "TTM":
            if processing:
                self.ttm_patch_btn.setEnabled(False)
            else:
                self.check_ready()
        elif mode == "TTM+VC":
            if processing:
                self.ttm_vc_patch_btn.setEnabled(False)
            else:
                self.check_ready()
        else:
            self.base_analyze_btn.setEnabled(not processing and self.base_audio_path is not None)
            self.target_analyze_btn.setEnabled(not processing and self.target_audio_path is not None)
            self.patch_btn.setEnabled(not processing and self.transcription_data is not None and self.voice_embedded)

    def on_error(self, error_msg):
        self.status_bar.setText(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.set_processing_state(False)
        self.progress.setValue(0)

def print_banner():
    print("""
██    ██  ██████  ██████  ███████ ██████
██    ██ ██    ██ ██   ██ ██      ██   ██
██    ██ ██    ██ ██   ██ █████   ██████
 ██  ██  ██    ██ ██   ██ ██      ██   ██
  ████    ██████  ██████  ███████ ██   ██
""")
    print("=" * 60)
    print("Interactive CLI Mode - Voice Blender Tool")
    print("=" * 60)

def validate_file_exists(path):
    if os.path.exists(path):
        return True
    print(f"Error: File not found: {path}")
    return False

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}

def validate_audio_file(path):
    if not os.path.exists(path):
        return False, "File does not exist."
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return True, "video"
    try:
        torchaudio.load(path)
        return True, "audio"
    except Exception as e:
        return False, f"Unsupported or corrupt audio/video format: {str(e)}"

def is_youtube_url(url):
    youtube_patterns = [
        'youtube.com',
        'youtu.be',
        'youtube.com/watch',
        'youtube.com/shorts',
        'bilibili.com',
        'tiktok.com',
    ]
    url_lower = url.lower()
    return any(pattern in url_lower for pattern in youtube_patterns)

def download_youtube_audio(url, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    try:
        import yt_dlp

        output_path = os.path.join(temp_dir, f"voder_yt_{int(time.time())}")

        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
        }

        print(f"Downloading audio from: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                if info is None:
                    return False, "Failed to fetch video information", None

                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                print(f"Video: {title} ({duration}s)")

                if not info:
                    return False, "Network error: Could not access video", None

            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                if 'is not a valid URL' in error_msg:
                    return False, "Invalid YouTube URL", None
                elif 'Video unavailable' in error_msg:
                    return False, "Video is unavailable", None
                elif 'HTTP Error' in error_msg:
                    return False, f"Network error: {error_msg}", None
                elif 'Connection' in error_msg:
                    return False, "Connection error: Check your internet connection", None
                else:
                    return False, f"Download error: {error_msg}", None
            except Exception as e:
                return False, f"Error checking video: {str(e)}", None

            try:
                print("Extracting audio...")
                ydl.download([url])
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                if 'HTTP Error' in error_msg:
                    return False, f"Network error during download: {error_msg}", None
                elif 'Connection' in error_msg:
                    return False, "Connection lost during download", None
                else:
                    return False, f"Download failed: {error_msg}", None
            except Exception as e:
                return False, f"Download error: {str(e)}", None

        mp3_path = output_path + '.mp3'
        if os.path.exists(mp3_path):
            print(f"Audio downloaded successfully: {mp3_path}")
            return True, None, mp3_path

        for ext in ['.m4a', '.wav', '.webm']:
            alt_path = output_path + ext
            if os.path.exists(alt_path):
                if ext != '.mp3':
                    try:
                        import torchaudio
                        waveform, sr = torchaudio.load(alt_path)
                        torchaudio.save(mp3_path, waveform, sr)
                        os.unlink(alt_path)
                        print(f"Audio downloaded and converted: {mp3_path}")
                        return True, None, mp3_path
                    except:
                        return True, None, alt_path
                return True, None, alt_path

        return False, "Downloaded file not found", None

    except ImportError:
        return False, "yt-dlp not installed. Run: pip install yt-dlp", None
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", None

def extract_audio_from_video_cli(video_path):
    try:
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"voder_cli_{int(time.time())}.wav")
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(audio_path):
            return audio_path
        return None
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return None

def validate_dialogue_source_file(file_path):
    if is_youtube_url(file_path):
        return True, "youtube", None

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}", None

    ext = file_path.lower()
    if ext.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.flac', '.m4a')):
        return True, "audio", None
    elif ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        return True, "image", None
    elif ext.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return False, "Empty text file", None

            lines = content.split('\n')
            dialogue_items = []
            mode_detected = None
            auto_formatted = False

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                has_colon = ':' in line
                if mode_detected is None:
                    mode_detected = 'dialogue' if has_colon else 'single'

                if mode_detected == 'single':
                    if len(lines) == 1:
                        dialogue_items.append((1, 'text', line))
                    else:
                        dialogue_items.append((len(dialogue_items) + 1, 'text', line))
                        auto_formatted = True
                else:
                    if ':' not in line:
                        dialogue_items.append((len(dialogue_items) + 1, 'text', line))
                        auto_formatted = True
                    else:
                        parts = line.split(':', 1)
                        speaker = parts[0].strip()
                        text = parts[1].strip()
                        if not speaker or not text:
                            dialogue_items.append((len(dialogue_items) + 1, 'text', line.split(':', 1)[1].strip() if ':' in line else line))
                            auto_formatted = True
                        else:
                            dialogue_items.append((len(dialogue_items) + 1, speaker, text))

            if not dialogue_items:
                return False, "No valid dialogue found in file", None

            if auto_formatted:
                print(f"\n[Auto-format] TXT file has been reformatted for compatibility:")
                print("  - Lines without speaker name are prefixed with 'text:'")
                print("  - Empty lines have been removed")
                print(f"  - Total lines after formatting: {len(dialogue_items)}")

            return True, "txt", dialogue_items

        except Exception as e:
            return False, f"Error reading file: {str(e)}", None
    else:
        return False, f"Unsupported file format: {file_path}", None

def analyze_dialogue_source(file_path, source_type="audio"):
    if source_type == "txt":
        return True, None, None

    if source_type == "image":
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            return False, "Failed to load EasyOCR model", None

        print(f"Extracting text from image: {os.path.basename(file_path)}")
        success, text, error_msg = ocr.extract_text_from_image(file_path)

        ocr.cleanup()
        del ocr
        gc.collect()

        if not success:
            return False, error_msg or "Failed to extract text from image", None

        if not text:
            return False, "No text found in image", None

        dialogue_items = [(1, 'text', text)]
        return True, None, dialogue_items

    if source_type == "youtube":
        print(f"Downloading audio from YouTube...")
        success, error_msg, audio_path = download_youtube_audio(file_path)
        if not success:
            return False, error_msg, None

        file_path = audio_path

    audio_path = file_path
    needs_cleanup = False
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(file_path)
        if not audio_path:
            return False, "Failed to extract audio from video", None
        needs_cleanup = True
    elif not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        return False, f"Unsupported audio format: {file_path}", None

    try:
        print("Loading Whisper model...")
        stt = WhisperSTT()
        if stt.model is None:
            return False, "Failed to load Whisper model", None

        print("Transcribing audio...")
        result = stt.transcribe(audio_path)

        del stt
        stt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not result:
            return False, "Transcription failed", None

        print("Performing speaker diarization...")
        diarization = SpeakerDiarization()

        if diarization.pipeline is None:
            text = result.get("text", "").strip()
            if not text:
                return False, "No text transcribed", None
            dialogue_items = [(1, 'text', text)]
            return True, None, dialogue_items

        diar_result = diarization.diarize(audio_path)

        formatted_segments = diarization.format_diarization(diar_result, result)

        del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not formatted_segments:
            text = result.get("text", "").strip()
            dialogue_items = [(1, 'text', text)]
            return True, None, dialogue_items

        original_speakers = []
        for seg in formatted_segments:
            speaker = seg["speaker"]
            if speaker not in original_speakers:
                original_speakers.append(speaker)

        speaker_mapping = {spk: idx for idx, spk in enumerate(sorted(original_speakers), 1)}

        if len(original_speakers) == 1:
            content = " ".join(seg["text"] for seg in formatted_segments)
            dialogue_items = [(1, 'text', content)]
        else:
            dialogue_items = []
            current_speaker_num = None
            current_text_parts = []
            current_start_time = None
            last_end_time = None
            min_words_before_switch = 3
            min_gap_before_switch = 0.3

            for seg in formatted_segments:
                speaker_num = speaker_mapping[seg["speaker"]]
                text = seg["text"]

                if speaker_num == current_speaker_num:
                    current_text_parts.append(text)
                    last_end_time = seg["end"]
                else:
                    time_gap = seg["start"] - last_end_time if last_end_time else 0
                    words_accumulated = len(current_text_parts)

                    should_switch = (words_accumulated >= min_words_before_switch) or (time_gap >= min_gap_before_switch)

                    if should_switch and current_text_parts:
                        content = " ".join(current_text_parts)
                        dialogue_items.append((current_speaker_num, str(current_speaker_num), content))
                        current_speaker_num = speaker_num
                        current_text_parts = [text]
                        current_start_time = seg["start"]
                        last_end_time = seg["end"]
                    else:
                        current_text_parts.append(text)
                        last_end_time = seg["end"]

            if current_text_parts:
                content = " ".join(current_text_parts)
                dialogue_items.append((current_speaker_num, str(current_speaker_num), content))

        return True, None, dialogue_items

    except Exception as e:
        return False, f"Error analyzing audio: {str(e)}", None
    finally:
        if needs_cleanup and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        if source_type == "youtube" and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

def extract_voice_clips_from_multispeaker(file_path, num_speakers, source_type="audio"):
    audio_path = file_path
    needs_cleanup = False
    youtube_temp = None

    if source_type == "youtube":
        print(f"Downloading audio from YouTube...")
        success, error_msg, downloaded_path = download_youtube_audio(file_path)
        if not success:
            return False, error_msg, None
        audio_path = downloaded_path
        youtube_temp = downloaded_path
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(file_path)
        if not audio_path:
            return False, "Failed to extract audio from video", None
        needs_cleanup = True

    try:
        print("Loading Whisper model...")
        stt = WhisperSTT()
        if stt.model is None:
            return False, "Failed to load Whisper model", None

        print("Transcribing audio with word timestamps...")
        result = stt.transcribe(audio_path)

        del stt
        stt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not result:
            return False, "Transcription failed", None

        print("Performing speaker diarization...")
        diarization = SpeakerDiarization()

        if diarization.pipeline is None:
            return False, "Speaker diarization model not available", None

        diar_result = diarization.diarize(audio_path)

        del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if diar_result is None:
            return False, "Speaker diarization failed", None

        speaker_segments = {}

        segments = result.get("segments", [])
        if not segments:
            return False, "No transcription segments found", None

        transcription_segments = []
        for seg in segments:
            words = seg.get("words", [])
            if words:
                for word in words:
                    transcription_segments.append({
                        "start": word.get("start", seg.get("start", 0)),
                        "end": word.get("end", seg.get("end", 0)),
                        "text": word.get("word", "").strip()
                    })
            else:
                transcription_segments.append({
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip()
                })

        for t_seg in transcription_segments:
            assigned_speaker = None
            for turn in diar_result.itertracks(yield_label=True):
                start, end, speaker = turn
                if t_seg["start"] >= start and t_seg["end"] <= end:
                    assigned_speaker = speaker
                    break
                elif t_seg["start"] >= start and t_seg["start"] < end:
                    assigned_speaker = speaker
                    break

            if assigned_speaker is not None:
                if assigned_speaker not in speaker_segments:
                    speaker_segments[assigned_speaker] = []
                speaker_segments[assigned_speaker].append({
                    "start": t_seg["start"],
                    "end": t_seg["end"],
                    "text": t_seg["text"]
                })

        if not speaker_segments:
            return False, "No speaker segments found", None

        sorted_speakers = sorted(speaker_segments.keys())

        speaker_to_num = {}
        for idx, speaker in enumerate(sorted_speakers, 1):
            speaker_to_num[speaker] = str(idx)

        clips_dict = {}
        temp_dir = tempfile.mkdtemp()

        for speaker in sorted_speakers:
            if len(clips_dict) >= num_speakers:
                break

            segments_list = speaker_segments[speaker]
            if not segments_list:
                continue

            longest_seg = max(segments_list, key=lambda x: x["end"] - x["start"])

            speaker_num = speaker_to_num[speaker]
            clip_path = os.path.join(temp_dir, f"{speaker_num}.wav")

            cmd = [
                'ffmpeg', '-i', audio_path,
                '-ss', str(longest_seg["start"]),
                '-t', str(longest_seg["end"] - longest_seg["start"]),
                '-y', clip_path
            ]

            result_ffmpeg = subprocess.run(cmd, capture_output=True, text=True)
            if result_ffmpeg.returncode != 0:
                print(f"Warning: Failed to extract clip for speaker {speaker_num}: {result_ffmpeg.stderr}")
                continue

            clips_dict[speaker_num] = clip_path
            print(f"Extracted voice clip for speaker {speaker_num} ({longest_seg['end'] - longest_seg['start']:.2f}s)")

        if not clips_dict:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            return False, "Failed to extract any voice clips", None

        return True, None, clips_dict

    except Exception as e:
        return False, f"Error extracting voice clips: {str(e)}", None
    finally:
        if needs_cleanup and os.path.exists(audio_path) and audio_path != file_path:
            try:
                os.unlink(audio_path)
            except:
                pass
        if youtube_temp and os.path.exists(youtube_temp):
            try:
                os.unlink(youtube_temp)
            except:
                pass

def cli_tts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTS Mode ---")

    print("\nDo you have a dialogue source file? (audio/video/txt/image)")
    print("Press Y to provide a file, or N to enter manually")
    has_source = input("> ").strip().lower()

    dialogue_items = None
    mode_detected = None

    if has_source in ['y', 'yes']:
        while True:
            print("\nEnter the path to your dialogue source (file path or YouTube URL):")
            file_path = input("> ").strip()
            if not file_path:
                print("Error: No file path provided")
                continue

            success, msg, items = validate_dialogue_source_file(file_path)
            if not success:
                print(f"Error: {msg}")
                retry = input("Try another source? (Y/N): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return False
                continue

            if msg == "txt":
                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 or (len(items) == 1 and items[0][1] != 'text') else 'single'
                break
            elif msg == "image":
                print(f"\nAnalyzing image: {os.path.basename(file_path)}...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="image")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            elif msg == "youtube":
                print(f"\nProcessing YouTube video...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="youtube")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            else:
                print(f"\nAnalyzing {os.path.basename(file_path)}...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="audio")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
    else:
        print("\nEnter script lines. Use format 'Character: text' for dialogue, or plain text for single speech.")
        print("Empty line finishes script entry.")
        lines = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            has_colon = ':' in line
            if mode_detected is None:
                mode_detected = 'dialogue' if has_colon else 'single'
            else:
                if (mode_detected == 'dialogue' and not has_colon) or (mode_detected == 'single' and has_colon):
                    print("Error: Inconsistent format. All lines must be either plain text (single mode) or contain 'Character: text' (dialogue mode).")
                    return False
            lines.append(line)

        if not lines:
            print("Error: No script provided")
            return False

        if mode_detected == 'single':
            script = "\n".join(lines)
            print("Enter voice prompt:")
            voice_prompt = input("> ").strip()
            if not voice_prompt:
                print("Error: No voice prompt provided")
                return False
            print("\nLoading Qwen-TTS VoiceDesign model...")
            tts_design = QwenTTSVoiceDesign()
            if tts_design.model is None:
                print("Error: Failed to load VoiceDesign model")
                return False
            print("Generating speech...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
            success = tts_design.synthesize(script, voice_prompt, output_path)
            if not success:
                print("Error: VoiceDesign synthesis failed")
                return False
            print(f"\n✓ Success! Output saved to: {output_path}")
            del tts_design
            tts_design = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        else:
            dialogue_items = []
            for i, line in enumerate(lines, start=1):
                if ':' not in line:
                    print(f"Error: Invalid dialogue line (missing ':'): {line}")
                    return False
                char, text = line.split(':', 1)
                char = char.strip()
                text = text.strip()
                if not char:
                    print(f"Error: Empty character in line: {line}")
                    return False
                if char.lower() == 'sfx' and not text:
                    print(f"Error: Empty SFX prompt in line: {line}")
                    return False
                clean_text, directives_raw = _parse_script_directives(text)
                parsed_directives, errors = _parse_directives_for_line(directives_raw)
                if errors:
                    print(f"Error in line {i}: {'; '.join(errors)}")
                    print("  Please re-enter this line.")
                    while True:
                        retry_line = input("> ").strip()
                        if not retry_line:
                            print("Error: Line cannot be empty. Please try again.")
                            continue
                        if ':' not in retry_line:
                            print("Error: Line must contain ':'. Please try again.")
                            continue
                        rchar, rtext = retry_line.split(':', 1)
                        rchar = rchar.strip()
                        rtext = rtext.strip()
                        if rchar.lower() != char.lower():
                            print(f"Error: Character must be '{char}'. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and not rtext:
                            print("Error: SFX prompt cannot be empty. Please try again.")
                            continue
                        rclean_text, rdirectives_raw = _parse_script_directives(rtext)
                        rparsed_directives, rerrors = _parse_directives_for_line(rdirectives_raw)
                        if rerrors:
                            print(f"Error: {'; '.join(rerrors)}. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and rparsed_directives.get('duration') is None:
                            print("Error: SFX line requires /duration:nn (1-30). Please try again.")
                            continue
                        if not rclean_text and rchar.lower() != 'sfx':
                            print("Error: Empty text. Please try again.")
                            continue
                        clean_text = rclean_text
                        parsed_directives = rparsed_directives
                        break
                if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                    while True:
                        dur_input = input(f"  SFX duration for line {i} (1-30): ").strip()
                        if not dur_input:
                            print("Error: Duration is required for SFX lines.")
                            continue
                        val, err = _validate_duration_directive(dur_input)
                        if err:
                            print(f"Error: {err}. Please enter a number between 1 and 30.")
                            continue
                        parsed_directives['duration'] = val
                        break
                if not clean_text and char.lower() != 'sfx':
                    print(f"Error: Empty text in line: {line}")
                    return False
                dialogue_items.append((i, char, clean_text, parsed_directives))

        chars = set()
        for _, char, _, _ in dialogue_items:
            if char.lower() != 'sfx':
                chars.add(char.lower())

        _is_all_sfx_interactive = len(chars) == 0

        if not _is_all_sfx_interactive:
            print(f"\nVoice prompts for {len(chars)} character(s):")
        voice_prompts = {}
        sorted_chars = sorted(chars)
        for i, char_lower in enumerate(sorted_chars):
            orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
            prompt = input(f"{orig_char}: ").strip()
            if not prompt:
                print(f"Error: No voice prompt for {orig_char}")
                return False
            voice_prompts[char_lower] = prompt
            print(f"Progress: {i+1}/{len(chars)} completed")

        music_description = None
        music_level_spec = None
        add_music = input("\nAdd background music? (y/N): ").strip().lower()
        if add_music in ('y', 'yes'):
            music_desc = input("Music description: ").strip()
            if music_desc:
                music_description = music_desc
        if music_description:
            level_input = input("Sound level (optional, press Enter for default 35%): ").strip()
            if level_input:
                parsed_level = _parse_music_level_spec(level_input)
                if parsed_level is None:
                    print("Warning: Invalid level format, using default 35%")
                else:
                    music_level_spec = level_input

        tts_design = None
        if not _is_all_sfx_interactive:
            print("\nLoading Qwen-TTS VoiceDesign model...")
            tts_design = QwenTTSVoiceDesign()
            if tts_design.model is None:
                print("Error: Failed to load VoiceDesign model")
                return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"voder_tts_dialogue_{timestamp}"
        if music_description:
            base_name += "_m"
        output_path = os.path.join(results_dir, f"{base_name}.wav")

        has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
        has_effects = any(
            item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
            for item in dialogue_items
        ) if len(dialogue_items) > 0 else False

        dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dialogue_temp.close()

        if has_sfx or has_effects:
            success, msg = _assemble_enhanced_dialogue(
                dialogue_items, voice_prompts, tts_design_obj=tts_design,
                output_path=dialogue_temp.name, mode='tts'
            )
            if not success:
                print(f"Error: {msg}")
                return False
        elif len(dialogue_items) == 1:
            _, char, text = dialogue_items[0]
            voice_instruct = voice_prompts[char.lower()]
            success = tts_design.synthesize(text, voice_instruct, dialogue_temp.name)
            if not success:
                print("Error: VoiceDesign synthesis failed")
                return False
        else:
            simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
            success, msg = tts_design.synthesize_dialogue(simple_items, voice_prompts, dialogue_temp.name)
            if not success:
                print(f"Error: {msg}")
                return False

        if music_description:
            if tts_design is not None:
                del tts_design
                tts_design = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ace = AceStepWrapper()
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False
            success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec)
            del ace
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not success:
                return False
            os.unlink(dialogue_temp.name)
        else:
            shutil.move(dialogue_temp.name, output_path)
        print(f"\n✓ Success! Output saved to: {output_path}")
        if tts_design is not None:
            del tts_design
            tts_design = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return True

def cli_tts_vc_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTS+VC Mode ---")

    print("\nDo you have a dialogue source file? (audio/video/txt/image)")
    print("Press Y to provide a file, or N to enter manually")
    has_source = input("> ").strip().lower()

    dialogue_items = None
    mode_detected = None

    if has_source in ['y', 'yes']:
        while True:
            print("\nEnter the path to your dialogue source (file path or YouTube URL):")
            file_path = input("> ").strip()
            if not file_path:
                print("Error: No file path provided")
                continue

            success, msg, items = validate_dialogue_source_file(file_path)
            if not success:
                print(f"Error: {msg}")
                retry = input("Try another source? (Y/N): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return False
                continue

            if msg == "txt":
                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 or (len(items) == 1 and items[0][1] != 'text') else 'single'
                break
            elif msg == "image":
                print(f"\nAnalyzing image: {os.path.basename(file_path)}...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="image")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            elif msg == "youtube":
                print(f"\nProcessing YouTube video...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="youtube")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            else:
                print(f"\nAnalyzing {os.path.basename(file_path)}...")
                success, error_msg, items = analyze_dialogue_source(file_path, source_type="audio")
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
    else:
        print("\nEnter script lines. Use format 'Character: text' for dialogue, or plain text for single speech.")
        print("Empty line finishes script entry.")
        lines = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            has_colon = ':' in line
            if mode_detected is None:
                mode_detected = 'dialogue' if has_colon else 'single'
            else:
                if (mode_detected == 'dialogue' and not has_colon) or (mode_detected == 'single' and has_colon):
                    print("Error: Inconsistent format. All lines must be either plain text (single mode) or contain 'Character: text' (dialogue mode).")
                    return False
            lines.append(line)

        if not lines:
            print("Error: No script provided")
            return False

        if mode_detected == 'single':
            script = "\n".join(lines)
            print("Enter target voice audio/video path:")
            target_path = input("> ").strip()
            valid, msg = validate_audio_file(target_path)
            if not valid:
                print(f"Error: {msg}")
                return False
            if msg == "video":
                print("Extracting audio from video...")
                extracted = extract_audio_from_video_cli(target_path)
                if not extracted:
                    print("Error: Could not extract audio from video")
                    return False
                target_path = extracted
            print("\nLoading Qwen-TTS model...")
            tts = QwenTTS()
            print("Extracting voice characteristics...")
            success = tts.extract_voice(target_path)
            if not success:
                print("Error: Voice extraction failed")
                return False
            print("Generating speech with cloned voice...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_tts_vc_{timestamp}.wav")
            success = tts.synthesize(script, output_path)
            if not success:
                print("Error: Synthesis failed")
                return False
            print(f"\n✓ Success! Output saved to: {output_path}")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        else:
            dialogue_items = []
            for i, line in enumerate(lines, start=1):
                if ':' not in line:
                    print(f"Error: Invalid dialogue line (missing ':'): {line}")
                    return False
                char, text = line.split(':', 1)
                char = char.strip()
                text = text.strip()
                if not char:
                    print(f"Error: Empty character in line: {line}")
                    return False
                if char.lower() == 'sfx' and not text:
                    print(f"Error: Empty SFX prompt in line: {line}")
                    return False
                clean_text, directives_raw = _parse_script_directives(text)
                parsed_directives, errors = _parse_directives_for_line(directives_raw)
                if errors:
                    print(f"Error in line {i}: {'; '.join(errors)}")
                    print("  Please re-enter this line.")
                    while True:
                        retry_line = input("> ").strip()
                        if not retry_line:
                            print("Error: Line cannot be empty. Please try again.")
                            continue
                        if ':' not in retry_line:
                            print("Error: Line must contain ':'. Please try again.")
                            continue
                        rchar, rtext = retry_line.split(':', 1)
                        rchar = rchar.strip()
                        rtext = rtext.strip()
                        if rchar.lower() != char.lower():
                            print(f"Error: Character must be '{char}'. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and not rtext:
                            print("Error: SFX prompt cannot be empty. Please try again.")
                            continue
                        rclean_text, rdirectives_raw = _parse_script_directives(rtext)
                        rparsed_directives, rerrors = _parse_directives_for_line(rdirectives_raw)
                        if rerrors:
                            print(f"Error: {'; '.join(rerrors)}. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and rparsed_directives.get('duration') is None:
                            print("Error: SFX line requires /duration:nn (1-30). Please try again.")
                            continue
                        if not rclean_text and rchar.lower() != 'sfx':
                            print("Error: Empty text. Please try again.")
                            continue
                        clean_text = rclean_text
                        parsed_directives = rparsed_directives
                        break
                if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                    while True:
                        dur_input = input(f"  SFX duration for line {i} (1-30): ").strip()
                        if not dur_input:
                            print("Error: Duration is required for SFX lines.")
                            continue
                        val, err = _validate_duration_directive(dur_input)
                        if err:
                            print(f"Error: {err}. Please enter a number between 1 and 30.")
                            continue
                        parsed_directives['duration'] = val
                        break
                if not clean_text and char.lower() != 'sfx':
                    print(f"Error: Empty text in line: {line}")
                    return False
                dialogue_items.append((i, char, clean_text, parsed_directives))

        chars = set()
        for _, char, _, _ in dialogue_items:
            if char.lower() != 'sfx':
                chars.add(char.lower())

        _is_all_sfx_vc_interactive = len(chars) == 0

        assignments = {}
        temp_clip_dir = None

        if not _is_all_sfx_vc_interactive:
            sorted_chars = sorted(chars)
            print(f"\nDo you have a multi-speaker audio source? (for auto voice cloning)")
            print("Press Y to provide a file, or N to enter manually for each character")
            has_multispeaker = input("> ").strip().lower()

            if has_multispeaker in ['y', 'yes']:
                while True:
                    print("\nEnter the path to your multi-speaker audio source (file path or YouTube URL):")
                    file_path = input("> ").strip()
                    if not file_path:
                        print("Error: No file path provided")
                        continue

                    source_type = "audio"
                    if "youtube.com" in file_path.lower() or "youtu.be" in file_path.lower():
                        source_type = "youtube"
                        success, msg, _ = validate_dialogue_source_file(file_path)
                        if not success:
                            print(f"Error: {msg}")
                            retry = input("Try another source? (Y/N): ").strip().lower()
                            if retry not in ['y', 'yes']:
                                return False
                            continue
                    elif not os.path.exists(file_path):
                        print(f"Error: File not found: {file_path}")
                        retry = input("Try another source? (Y/N): ").strip().lower()
                        if retry not in ['y', 'yes']:
                            return False
                        continue

                    print(f"\nExtracting voice clips from multi-speaker source...")
                    success, error_msg, clips_dict = extract_voice_clips_from_multispeaker(
                        file_path, len(sorted_chars), source_type=source_type
                    )

                    if not success:
                        print(f"Error: {error_msg}")
                        retry = input("Try another source? (Y/N): ").strip().lower()
                        if retry not in ['y', 'yes']:
                            return False
                        continue

                    print(f"\nExtracted {len(clips_dict)} voice clip(s). Assigning to characters...")

                    clip_keys = sorted(clips_dict.keys(), key=lambda x: int(x))

                    for i, char_lower in enumerate(sorted_chars):
                        orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                        if i < len(clip_keys):
                            clip_path = clips_dict[clip_keys[i]]
                            assignments[char_lower] = clip_path
                            print(f"  {orig_char} -> speaker {clip_keys[i]} (auto)")
                        else:
                            path = input(f"{orig_char} (need more): ").strip()
                            if not path:
                                print(f"Error: No audio path provided for {orig_char}")
                                return False
                            valid, msg = validate_audio_file(path)
                            if not valid:
                                print(f"Error: {msg}")
                                return False
                            if msg == "video":
                                print(f"Extracting audio from video for {orig_char}...")
                                extracted = extract_audio_from_video_cli(path)
                                if not extracted:
                                    print(f"Error: Could not extract audio from video for {orig_char}")
                                    return False
                                path = extracted
                            assignments[char_lower] = path
                            print(f"  {orig_char} -> manual")

                    temp_clip_dir = os.path.dirname(list(clips_dict.values())[0]) if clips_dict else None
                    break
            else:
                print(f"\nAudio file paths for {len(chars)} character(s):")
                for i, char_lower in enumerate(sorted_chars):
                    orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                    path = input(f"{orig_char}: ").strip()
                    if not path:
                        print(f"Error: No audio path provided for {orig_char}")
                        return False
                    valid, msg = validate_audio_file(path)
                    if not valid:
                        print(f"Error: {msg}")
                        return False
                    if msg == "video":
                        print(f"Extracting audio from video for {orig_char}...")
                        extracted = extract_audio_from_video_cli(path)
                        if not extracted:
                            print(f"Error: Could not extract audio from video for {orig_char}")
                            return False
                        path = extracted
                    assignments[char_lower] = path
                    print(f"Progress: {i+1}/{len(chars)} completed")

        music_description = None
        music_level_spec = None
        add_music = input("\nAdd background music? (y/N): ").strip().lower()
        if add_music in ('y', 'yes'):
            music_desc = input("Music description: ").strip()
            if music_desc:
                music_description = music_desc
        if music_description:
            level_input = input("Sound level (optional, press Enter for default 35%): ").strip()
            if level_input:
                parsed_level = _parse_music_level_spec(level_input)
                if parsed_level is None:
                    print("Warning: Invalid level format, using default 35%")
                else:
                    music_level_spec = level_input

        tts = None
        if not _is_all_sfx_vc_interactive:
            print("\nLoading Qwen-TTS model...")
            tts = QwenTTS()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"voder_tts_vc_dialogue_{timestamp}"
        if music_description:
            base_name += "_m"
        output_path = os.path.join(results_dir, f"{base_name}.wav")

        dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dialogue_temp.close()

        has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
        has_effects = any(
            item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
            for item in dialogue_items
        ) if len(dialogue_items) > 0 else False

        if has_sfx or has_effects:
            unique_chars = set()
            for _, char, _, _ in dialogue_items:
                if char.lower() != 'sfx':
                    unique_chars.add(char.lower())
            voice_prompts = {}
            for char_lower in unique_chars:
                audio_path = assignments[char_lower]
                print(f"Extracting voice for '{char_lower}'...")
                success = tts.extract_voice(audio_path)
                if not success:
                    print(f"Error: Failed to extract voice from {audio_path}")
                    return False
                voice_prompts[char_lower] = tts.voice_prompt
            success, msg = _assemble_enhanced_dialogue(
                dialogue_items, voice_prompts, tts_vc_obj=tts,
                output_path=dialogue_temp.name, mode='tts_vc'
            )
            if not success:
                print(f"Error: {msg}")
                return False
        elif len(dialogue_items) == 1:
            _, char, text = dialogue_items[0]
            audio_path = assignments[char.lower()]
            success = tts.extract_voice(audio_path)
            if not success:
                print("Error: Voice extraction failed")
                return False
            success = tts.synthesize(text, dialogue_temp.name)
            if not success:
                print("Error: Synthesis failed")
                return False
        else:
            temp_dir = tempfile.mkdtemp()
            try:
                temp_files = []
                simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
                unique_chars = set()
                for _, char, _ in simple_items:
                    unique_chars.add(char.lower())
                voice_prompts = {}
                for char_lower in unique_chars:
                    audio_path = assignments[char_lower]
                    print(f"Extracting voice for '{char_lower}'...")
                    success = tts.extract_voice(audio_path)
                    if not success:
                        print(f"Error: Failed to extract voice from {audio_path}")
                        return False
                    voice_prompts[char_lower] = tts.voice_prompt
                for i, (num, char, script_text) in enumerate(simple_items):
                    char_lower = char.lower()
                    print(f"Processing line {num} for '{char}'...")
                    tts.voice_prompt = voice_prompts[char_lower]
                    temp_file = os.path.join(temp_dir, f"line_{num}.wav")
                    temp_files.append((num, temp_file))
                    success = tts.synthesize(script_text, temp_file)
                    if not success:
                        print(f"Error: Failed to generate speech for line {num}")
                        return False
                temp_files.sort(key=lambda x: x[0])
                concat_list = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list, 'w') as f:
                    for _, tf in temp_files:
                        f.write(f"file '{tf}'\n")
                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', dialogue_temp.name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error: FFmpeg concatenation failed: {result.stderr}")
                    return False
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
                if temp_clip_dir and os.path.exists(temp_clip_dir):
                    try:
                        shutil.rmtree(temp_clip_dir)
                    except:
                        pass

        if music_description:
            if tts is not None:
                del tts
                tts = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ace = AceStepWrapper()
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False
            success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec)
            del ace
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not success:
                return False
            os.unlink(dialogue_temp.name)
        else:
            shutil.move(dialogue_temp.name, output_path)
        print(f"\n✓ Success! Output saved to: {output_path}")
        if tts is not None:
            del tts
            tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True

def cli_stt_tts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- STT+TTS Mode ---")
    print("Convert speech from base audio to target voice")
    print()
    base_path = input("Enter base audio/video path: ").strip()
    if not validate_file_exists(base_path):
        return False
    if base_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(base_path)
        if not audio_path:
            print("Error: Could not extract audio from video")
            return False
        base_path = audio_path
    print("\nLoading Whisper model...")
    stt = WhisperSTT()
    print("Transcribing base audio...")
    result = stt.transcribe(base_path)
    if not result:
        print("Error: Transcription failed")
        return False
    text = result.get("text", "").strip()
    print(f"\nExtracted text ({len(text)} chars):")
    display_text = text.replace('\n', '\\n').replace('\r', '\\r')
    print(display_text)
    print()

    print("Offloading Whisper model...")
    del stt
    stt = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    edited_text = input("Edit text (or press Enter to keep as is): ").strip()
    if edited_text:
        text = edited_text.replace('\\n', '\n')
    if not text:
        print("Error: No text to synthesize")
        return False
    print()
    target_path = input("Enter target voice audio path: ").strip()
    if not validate_file_exists(target_path):
        return False
    print("\nLoading Qwen-TTS model...")
    tts = QwenTTS()
    print("Extracting voice characteristics...")
    success = tts.extract_voice(target_path)
    if not success:
        print("Error: Voice extraction failed")
        return False
    print("\nSynthesizing speech with target voice...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"voder_stt_tts_{timestamp}.wav")
    success = tts.synthesize(text, output_path)
    if not success:
        print("Error: Synthesis failed")
        return False
    print(f"\n✓ Success! Output saved to: {output_path}")
    del tts
    tts = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def cli_sts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- STS Mode ---")
    print("Convert voice from base audio to target voice")
    print()
    base_path = input("Enter base audio/video path: ").strip()
    if not validate_file_exists(base_path):
        return False
    if base_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(base_path)
        if not audio_path:
            print("Error: Could not extract audio from video")
            return False
        base_path = audio_path
    print()
    target_path = input("Enter target voice audio path: ").strip()
    if not validate_file_exists(target_path):
        return False
    print()
    while True:
        music_input = input("Are the inputs musical? (Y/N): ").strip().lower()
        if music_input in ['y', 'yes']:
            is_music = True
            break
        elif music_input in ['n', 'no']:
            is_music = False
            break
        else:
            print("Please enter Y or N")
    if is_music:
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            return False
        print("Resampling inputs to 44100Hz...")
        import torchaudio
        waveform_base, sr_base = torchaudio.load(base_path)
        if sr_base != 44100:
            resampler_base = torchaudio.transforms.Resample(sr_base, 44100)
            waveform_base = resampler_base(waveform_base)
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_base.name, waveform_base, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_base.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
            shutil.copy(temp_output_44k.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for temp_file in [temp_base.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    else:
        print("\nLoading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            return False
        print("Resampling inputs to 22050Hz...")
        import torchaudio
        waveform_base, sr_base = torchaudio.load(base_path)
        if sr_base != 22050:
            resampler_base = torchaudio.transforms.Resample(sr_base, 22050)
            waveform_base = resampler_base(waveform_base)
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_base.name, waveform_base, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_base.name,
                reference_path=temp_target.name,
                output_path=temp_output_22k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            print("Upsampling output to 44100Hz...")
            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
            if sr_out != 44100:
                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                waveform_out = resampler_out(waveform_out)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
            torchaudio.save(output_path, waveform_out, 44100)
            print(f"\n✓ Success! Output saved to: {output_path}")
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for temp_file in [temp_base.name, temp_target.name, temp_output_22k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

def cli_ttm_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTM Mode ---")
    print("Generate music from lyrics and style")
    print()
    print("Enter song lyrics (use \\n for new lines):")
    lyrics = input("> ").strip()
    if not lyrics:
        print("Error: No lyrics provided")
        return False
    lyrics = lyrics.replace('\\n', '\n')
    print()
    print("Enter style prompt (use \\n for new lines, e.g., 'upbeat pop, female vocals'):")
    style = input("> ").strip()
    if not style:
        print("Error: No style prompt provided")
        return False
    style = style.replace('\\n', '\n')
    print()
    print("Enter duration in seconds (10-300, where 300 = 5 minutes max):")
    while True:
        try:
            duration = int(input("> ").strip())
            if 10 <= duration <= 300:
                break
            else:
                print("Error: Duration must be between 10 and 300 seconds")
        except ValueError:
            print("Error: Please enter a valid number")
    print("\nLoading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False
    print(f"Generating music ({duration}s duration)...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
    success = ace_step.generate(
        lyrics=lyrics,
        style_prompt=style,
        output_path=output_path,
        duration=duration
    )
    if not success:
        print("Error: Music generation failed")
        return False
    print(f"\n✓ Success! Output saved to: {output_path}")
    del ace_step
    ace_step = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True

def cli_ttm_vc_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTM+VC Mode ---")
    print("Generate music then convert to target voice")
    print()
    print("Enter song lyrics (use \\n for new lines):")
    lyrics = input("> ").strip()
    if not lyrics:
        print("Error: No lyrics provided")
        return False
    lyrics = lyrics.replace('\\n', '\n')
    print()
    print("Enter style prompt (use \\n for new lines, e.g., 'upbeat pop, female vocals'):")
    style = input("> ").strip()
    if not style:
        print("Error: No style prompt provided")
        return False
    style = style.replace('\\n', '\n')
    print()
    print("Enter duration in seconds (10-300, where 300 = 5 minutes max):")
    while True:
        try:
            duration = int(input("> ").strip())
            if 10 <= duration <= 300:
                break
            else:
                print("Error: Duration must be between 10 and 300 seconds")
        except ValueError:
            print("Error: Please enter a valid number")
    print()
    target_path = input("Enter target voice audio path: ").strip()
    if not validate_file_exists(target_path):
        return False
    print("\nLoading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False
    temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_ttm_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_target_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        print(f"Generating music ({duration}s duration)...")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=temp_ttm_output.name,
            duration=duration
        )
        if not success:
            print("Error: Music generation failed")
            return False
        print("Resampling TTM output to 44100Hz...")

        import torchaudio
        waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
        if sr_ttm != 44100:
            resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 44100)
            waveform_ttm = resampler_ttm(waveform_ttm)
        torchaudio.save(temp_ttm_22k.name, waveform_ttm, 44100)
        print("Resampling target voice to 44100Hz...")
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        torchaudio.save(temp_target_22k.name, waveform_target, 44100)
        print("Clearing ACE-Step from memory...")
        del ace_step
        ace_step = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Loading Seed-VC v1 model...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            return False
        print("Converting voice...")
        vc_success = seed_vc.convert(
            source_path=temp_ttm_22k.name,
            reference_path=temp_target_22k.name,
            output_path=temp_vc_output.name
        )
        if not vc_success:
            print("Error: Voice conversion failed")
            return False
        print("Saving output...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
        shutil.copy(temp_vc_output.name, output_path)
        print(f"\n✓ Success! Output saved to: {output_path}")
        del seed_vc
        seed_vc = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    finally:
        for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def parse_oneline_args(args):
    if not args:
        return {'error': 'No arguments provided'}
    mode = args[0].lower()
    result = {'mode': mode, 'params': {}, 'error': None, 'is_music': False}
    valid_keywords = ['script', 'voice', 'lyrics', 'styling', 'base', 'target', 'duration', 'timestamp', 'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr']
    i = 1
    current_keyword = None
    result_path = None

    if mode == 'stt':
        file_paths = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower in ['timestamp', 'dialogue']:
                result['params'][arg_lower] = True
                i += 1
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif os.path.exists(arg) or is_youtube_url(arg):
                file_paths.append(arg)
                i += 1
            else:
                result['error'] = f'File not found: {arg}'
                return result

        if not file_paths:
            result['error'] = 'STT mode requires at least one audio/video file path'
            return result

        result['params']['files'] = file_paths
        result['params']['result_path'] = result_path
        return result

    if mode == 'se':
        file_paths = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif os.path.exists(arg):
                file_paths.append(arg)
                i += 1
            else:
                result['error'] = f'File not found: {arg}'
                return result

        if not file_paths:
            result['error'] = 'SE mode requires at least one audio/video file path'
            return result

        result['params']['files'] = file_paths
        result['params']['result_path'] = result_path
        return result

    if mode == 'sfx':
        prompt = None
        duration = None
        steps = None
        guide = None
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'sound':
                if i + 1 < len(args):
                    prompt = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'sound keyword requires a prompt argument'
                    return result
            elif arg_lower == 'duration':
                if i + 1 < len(args):
                    try:
                        duration = int(args[i + 1])
                        i += 2
                    except ValueError:
                        result['error'] = 'duration must be a number between 1 and 30'
                        return result
                else:
                    result['error'] = 'duration keyword requires a number argument'
                    return result
            elif arg_lower == 'steps':
                if i + 1 < len(args):
                    try:
                        steps = int(args[i + 1])
                        i += 2
                    except ValueError:
                        print("Warning: Invalid steps value, using default (30).")
                        steps = None
                        i += 2
                else:
                    print("Warning: steps keyword requires a number, using default (30).")
                    i += 1
            elif arg_lower == 'guide':
                if i + 1 < len(args):
                    try:
                        guide = float(args[i + 1])
                        i += 2
                    except ValueError:
                        print("Warning: Invalid guide value, using default (4.5).")
                        guide = None
                        i += 2
                else:
                    print("Warning: guide keyword requires a number, using default (4.5).")
                    i += 1
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                result['error'] = f'Unknown parameter: {arg}'
                return result

        if not prompt:
            result['error'] = 'SFX mode requires a sound prompt (sound "your prompt")'
            return result
        if duration is None:
            result['error'] = 'SFX mode requires a duration (duration <1-30>)'
            return result
        if duration < 1:
            result['error'] = 'Duration must be at least 1 second'
            return result

        use_steps = 30
        use_guide = 4.5
        if steps is not None:
            if 1 <= steps <= 100:
                use_steps = steps
            else:
                print("Warning: steps must be between 1-100, using default (30).")
        if guide is not None:
            guide = round(guide * 2) / 2
            if 1.0 <= guide <= 10.0:
                use_guide = guide
            else:
                print("Warning: guide must be between 1.0-10.0, using default (4.5).")

        result['params']['prompt'] = prompt
        result['params']['duration'] = duration
        result['params']['steps'] = use_steps
        result['params']['guide'] = use_guide
        result['params']['result_path'] = result_path
        return result

    while i < len(args):
        arg = args[i]
        arg_lower = arg.lower()
        if arg_lower == 'result':
            if i + 1 < len(args):
                result_path = args[i + 1]
                i += 2
            else:
                result['error'] = 'result keyword requires a path argument'
                return result
        elif arg_lower in valid_keywords:
            current_keyword = arg_lower
            result['params'].setdefault(current_keyword, [])
            i += 1
        elif mode == 'sts' and arg_lower == 'music':
            result['is_music'] = True
            current_keyword = None
            i += 1
        elif current_keyword is not None:
            try:
                duration_val = int(arg)
                remaining = args[i+1:]
                is_duration = all(is_num(x) for x in remaining)
                if is_duration:
                    result['params']['duration'] = duration_val
                elif current_keyword == 'duration':
                    result['params']['duration'] = duration_val
                else:
                    result['params'][current_keyword].append(arg)
                i += 1
            except ValueError:
                result['params'][current_keyword].append(arg)
                i += 1
        else:
                try:
                    duration = int(arg)
                    result['params']['duration'] = duration
                    i += 1
                except ValueError:
                    if mode == 'sts':
                        result['error'] = f'invalid parameter "{arg}" only next parameter should be music or empty'
                    else:
                        result['error'] = f'Unknown parameter: {arg}'
                    return result

    result['params']['result_path'] = result_path
    return result

def is_num(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

def validate_oneline_mode(mode_name):
    valid_modes = ['tts', 'tts+vc', 'sts', 'ttm', 'ttm+vc', 'stt', 'se', 'sfx']
    if mode_name.lower() in ['stt+tts', 'stt_tts', 'stttts']:
        return 'stt+tts_rejected'
    if mode_name.lower() in valid_modes:
        return mode_name.lower()
    return None

def show_oneline_usage():
    print("VODER One-Line Command Usage:")
    print("=" * 60)
    print()
    print("Available modes:")
    print("  tts      - Text-to-Speech")
    print("  tts+vc   - Text-to-Speech + Voice Clone")
    print("  sts      - Speech-to-Speech (Voice Conversion)")
    print("  ttm      - Text-to-Music")
    print("  ttm+vc   - Text-to-Music + Voice Conversion")
    print("  stt      - Speech-to-Text (Transcription with optional diarization)")
    print("  se       - Speech Enhancement (denoise, dereverb, restore)")
    print("  sfx      - Sound Effects (text prompt + duration → audio)")
    print()
    print("Note: STT+TTS mode is not available in one-line mode.")
    print("      Use 'tts' mode with your text, or use interactive CLI.")
    print()
    print("Single mode examples:")
    print('  python voder.py tts script "hello world" voice "male voice"')
    print('  python voder.py tts+vc script "hello" target "voice.wav"')
    print('  python voder.py tts ocr "path/to/image.png" voice "text: female voice"')
    print('  python voder.py tts+vc ocr "path/to/image.png" target "text: voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav" music')
    print('  python voder.py ttm lyrics "song" styling "pop" 30')
    print('  python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
    print()
    print("STT examples (Speech-to-Text transcription):")
    print('  python voder.py stt "path/to/audio.wav"')
    print('  python voder.py stt "audio1.wav" "audio2.wav"')
    print('  python voder.py stt "audio.wav" timestamp')
    print('  python voder.py stt "audio.wav" dialogue')
    print('  python voder.py stt "audio.wav" timestamp dialogue')
    print('  python voder.py stt "https://youtube.com/watch?v=..."')
    print()
    print("SE examples (Speech Enhancement):")
    print('  python voder.py se "path/to/audio.wav"')
    print('  python voder.py se "audio1.wav" "audio2.wav"')
    print('  python voder.py se "path/to/video.mp4"')
    print()
    print("SFX examples (Sound Effects Generation):")
    print('  python voder.py sfx sound "thunder cracking" duration 5')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 result "output.wav"')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 steps 50')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 steps 50 guide 3.5 result "output.wav"')
    print()
    print("Dialogue mode examples:")
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"')
    print('  python voder.py tts+vc script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano"')
    print('  python voder.py tts script "James: Hello" script "sfx: thunder /duration:3" voice "James: deep male" music "soft piano" level "10:20-50"')
    print()
    print("Parameters (can appear multiple times):")
    print("  script   - Dialogue line in 'Character: text' format, or plain text for single mode")
    print("  voice    - Voice prompt in 'Character: description' format (TTS)")
    print("  target   - Audio file path in 'Character: path' format (TTS+VC) or single path (STS/TTM+VC)")
    print("  lyrics   - Song lyrics for TTM (single)")
    print("  styling  - Style prompt for TTM (single)")
    print("  base     - Base audio/video path")
    print("  music    - Music flag for STS mode (uses 44.1kHz v1 model)")
    print("  timestamp - Keep Whisper timestamps in output (STT mode)")
    print("  dialogue - Enable speaker diarization (STT mode)")
    print("  sound    - Sound effect prompt (SFX mode)")
    print("  duration - Duration in seconds (10-300 for TTM, 1-30 for SFX)")
    print("  steps    - Inference steps (1-100, SFX mode, default: 30)")
    print("  guide    - Guidance scale (1.0-10.0, SFX mode, default: 4.5)")
    print("  music    - Background music description (dialogue modes)")
    print("  level    - Music volume levels e.g. \"10:20-50 30:60-80\" (dialogue modes, default: 35%)")
    print("  ocr      - Image file path for OCR text extraction (TTS/TTS+VC modes)")
    print("  <number> - Duration in seconds (10-300, for TTM modes)")
    print()
    print("Script directives (per line, at end of text):")
    print("  /time:nn-nn+nn  - Cut nn seconds from end (-nn) and/or start (+nn)")
    print("  /level:0-100     - Volume level for that line (default: 100)")
    print("  /duration:1-30    - SFX duration (required for sfx: lines)")
    print("  sfx: prompt      - Special character: generates SFX via TangoFlux")

def execute_oneline_command(parsed):
    mode = parsed['mode']
    params = parsed['params']
    if 'is_music' in parsed:
        params['is_music'] = parsed['is_music']

    success = False
    if mode == 'tts':
        success = oneline_tts(params)
    elif mode == 'tts+vc':
        success = oneline_tts_vc(params)
    elif mode == 'sts':
        success = oneline_sts(params)
    elif mode == 'ttm':
        success = oneline_ttm(params)
    elif mode == 'ttm+vc':
        success = oneline_ttm_vc(params)
    elif mode == 'stt':
        success = oneline_stt(params)
    elif mode == 'se':
        success = oneline_se(params)
    elif mode == 'sfx':
        success = oneline_sfx(params)
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_oneline_usage()
        return False

    if success and params.get('result_path'):
        copy_result_to_path(params['result_path'])

    return success

def copy_result_to_path(result_path):
    if result_path is None:
        return
    try:
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            return
        files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
        if not files:
            return
        latest_file = max(files, key=os.path.getmtime)

        result_dir = os.path.dirname(result_path)
        result_filename = os.path.basename(result_path)

        if not result_dir:
            destination = os.path.join(".", result_filename)
            os.makedirs(".", exist_ok=True)
        else:
            os.makedirs(result_dir, exist_ok=True)
            destination = result_path

        shutil.copy2(latest_file, destination)
        print(f"Result copied to: {destination}")
    except Exception as e:
        print(f"Note: Could not copy to result path: {e}")

def oneline_tts(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    scripts = params.get('script', [])
    voices = params.get('voice', [])
    targets = params.get('target', [])
    music_params = params.get('music', [])
    music_description = music_params[0] if music_params else None
    level_params = params.get('level', [])
    music_level_spec = level_params[0] if level_params else None
    ocr_param = params.get('ocr', [])

    if ocr_param:
        ocr_path = ocr_param[0]
        if not os.path.exists(ocr_path):
            print(f"Error: Image file not found: {ocr_path}")
            return False
        ext = os.path.splitext(ocr_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']:
            print(f"Error: Input must be an image file. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP")
            return False
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            print("Error: Failed to load EasyOCR model")
            return False
        print(f"Extracting text from image...")
        success, extracted_text, error_msg = ocr.extract_text_from_image(ocr_path)
        ocr.cleanup()
        del ocr
        gc.collect()
        if not success:
            print(f"Error: {error_msg or 'Failed to extract text from image'}")
            return False
        if not extracted_text:
            print("Error: No text found in image")
            return False
        scripts = [f"text: {extracted_text}"]
        print(f"Extracted text: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")

    if not scripts:
        print("Error: TTS mode requires at least one 'script' parameter")
        return False

    _is_all_sfx = all(s.strip().lower().startswith('sfx:') for s in scripts)
    if not voices and not targets and not _is_all_sfx:
        print("Error: TTS mode requires at least one 'voice' or 'target' parameter")
        return False

    has_colon_script = any(':' in s for s in scripts)
    has_colon_voice = any(':' in v for v in voices) if voices else False
    has_colon_target = any(':' in t for t in targets) if targets else False

    if not has_colon_script and not has_colon_voice and not has_colon_target:
        if len(scripts) != 1:
            print("Error: Single mode expects exactly one script argument")
            return False
        if len(voices) != 1:
            print("Error: Single mode expects exactly one voice argument")
            return False
        if music_description:
            print("Warning: Background music is only supported for dialogue mode. Ignoring music parameter.")
        script = scripts[0].replace('\\n', '\n')
        voice_prompt = voices[0]
        print("Loading Qwen-TTS VoiceDesign model...")
        tts_design = QwenTTSVoiceDesign()
        if tts_design.model is None:
            print("Error: Failed to load VoiceDesign model")
            return False
        print("Generating speech...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
        success = tts_design.synthesize(script, voice_prompt, output_path)
        if not success:
            print("Error: VoiceDesign synthesis failed")
            return False
        print(f"✓ Success! Output saved to: {output_path}")
        return True
    else:
        if not has_colon_script:
            print("Error: Dialogue script must be in format 'Character: text', got: {s}")
            return False
        if not _is_all_sfx and not has_colon_voice and not has_colon_target:
            print("Error: Dialogue mode requires voice or target parameters in 'Character: value' format.")
            return False
        dialogue_items = []
        for idx, s in enumerate(scripts, start=1):
            if ':' not in s:
                print(f"Error: Dialogue script must be in format 'Character: text', got: {s}")
                return False
            char, text = s.split(':', 1)
            char = char.strip()
            text = text.strip()
            if not char:
                print(f"Error: Empty character in script: {s}")
                return False
            if char.lower() == 'sfx' and not text:
                print(f"Error: Empty SFX prompt in script: {s}")
                return False
            clean_text, directives_raw = _parse_script_directives(text)
            parsed_directives, errors = _parse_directives_for_line(directives_raw)
            if errors:
                print(f"Error in script line {idx}: {'; '.join(errors)}")
                return False
            if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                print(f"Error: SFX line {idx} requires /duration:nn (1-30)")
                return False
            if not clean_text and char.lower() != 'sfx':
                print(f"Error: Empty text in script: {s}")
                return False
            dialogue_items.append((idx, char, clean_text, parsed_directives))

        voice_prompts = {}
        for v in voices:
            if ':' not in v:
                print(f"Error: Voice prompt must be in format 'Character: prompt', got: {v}")
                return False
            char, prompt = v.split(':', 1)
            char = char.strip()
            prompt = prompt.strip()
            if not char or not prompt:
                print(f"Error: Empty character or prompt in voice: {v}")
                return False
            voice_prompts[char.lower()] = prompt

        target_assignments = {}
        for t in targets:
            if ':' not in t:
                print(f"Error: Target assignment must be in format 'Character: path', got: {t}")
                return False
            char, path = t.split(':', 1)
            char = char.strip()
            path = path.strip()
            if not char or not path:
                print(f"Error: Empty character or path in target: {t}")
                return False
            valid, msg = validate_audio_file(path)
            if not valid:
                print(f"Error: {msg}")
                return False
            if msg == "video":
                print(f"Extracting audio from video for character '{char}'...")
                extracted = extract_audio_from_video_cli(path)
                if not extracted:
                    print(f"Error: Could not extract audio from video for character '{char}'")
                    return False
                path = extracted
            target_assignments[char.lower()] = path

        overlap = set(voice_prompts.keys()) & set(target_assignments.keys())
        if overlap:
            print(f"Error: Character(s) specified in both voice and target: {', '.join(overlap)}")
            return False

        script_chars = set()
        for _, char, _, _ in dialogue_items:
            if char.lower() != 'sfx':
                script_chars.add(char.lower())
        all_assigned = set(voice_prompts.keys()) | set(target_assignments.keys())
        missing = script_chars - all_assigned
        if missing:
            print(f"Error: Missing voice/target for characters: {', '.join(missing)}")
            return False

        has_tts_chars = len(voice_prompts) > 0
        has_vc_chars = len(target_assignments) > 0

        if music_description and music_description.strip() == "":
            music_description = None

        if music_level_spec and not music_description:
            print("Warning: Level spec ignored (no music description provided)")

        tts_design = None
        if has_tts_chars:
            print("Loading Qwen-TTS VoiceDesign model...")
            tts_design = QwenTTSVoiceDesign()
            if tts_design.model is None:
                print("Error: Failed to load VoiceDesign model")
                return False

        vc_voice_prompts = None
        tts_obj = None
        if has_vc_chars:
            print("Loading Qwen-TTS model...")
            tts_obj = QwenTTS()
            vc_voice_prompts = {}
            for char_lower, audio_path in target_assignments.items():
                print(f"Extracting voice for '{char_lower}'...")
                success = tts_obj.extract_voice(audio_path)
                if not success:
                    print(f"Error: Failed to extract voice from {audio_path}")
                    return False
                vc_voice_prompts[char_lower] = tts_obj.voice_prompt

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"voder_tts_dialogue_{timestamp}"
        if music_description:
            base_name += "_m"
        output_path = os.path.join(results_dir, f"{base_name}.wav")

        dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dialogue_temp.close()

        has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
        has_effects = any(
            item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
            for item in dialogue_items
        ) if len(dialogue_items) > 0 else False

        if has_sfx or has_effects or has_vc_chars:
            success, msg = _assemble_enhanced_dialogue(
                dialogue_items, voice_prompts, tts_design_obj=tts_design,
                tts_vc_obj=tts_obj, vc_voice_data=vc_voice_prompts,
                output_path=dialogue_temp.name, mode='tts'
            )
            if not success:
                print(f"Error: {msg}")
                return False
        else:
            simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
            success, msg = tts_design.synthesize_dialogue(simple_items, voice_prompts, dialogue_temp.name)
            if not success:
                print(f"Error: {msg}")
                return False

        if music_description:
            if tts_design is not None:
                del tts_design
                tts_design = None
            if tts_obj is not None:
                del tts_obj
                tts_obj = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ace = AceStepWrapper()
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False
            success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec)
            del ace
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not success:
                return False
            os.unlink(dialogue_temp.name)
        else:
            shutil.move(dialogue_temp.name, output_path)
        print(f"✓ Success! Output saved to: {output_path}")
        if tts_design is not None:
            del tts_design
        if tts_obj is not None:
            del tts_obj
        return True

def oneline_tts_vc(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    scripts = params.get('script', [])
    voices = params.get('voice', [])
    targets = params.get('target', [])
    music_params = params.get('music', [])
    music_description = music_params[0] if music_params else None
    level_params = params.get('level', [])
    music_level_spec = level_params[0] if level_params else None
    ocr_param = params.get('ocr', [])

    if ocr_param:
        ocr_path = ocr_param[0]
        if not os.path.exists(ocr_path):
            print(f"Error: Image file not found: {ocr_path}")
            return False
        ext = os.path.splitext(ocr_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']:
            print(f"Error: Input must be an image file. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP")
            return False
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            print("Error: Failed to load EasyOCR model")
            return False
        print(f"Extracting text from image...")
        success, extracted_text, error_msg = ocr.extract_text_from_image(ocr_path)
        ocr.cleanup()
        del ocr
        gc.collect()
        if not success:
            print(f"Error: {error_msg or 'Failed to extract text from image'}")
            return False
        if not extracted_text:
            print("Error: No text found in image")
            return False
        scripts = [f"text: {extracted_text}"]
        print(f"Extracted text: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")

    if not scripts:
        print("Error: TTS+VC mode requires at least one 'script' parameter")
        return False

    _is_all_sfx_vc = all(s.strip().lower().startswith('sfx:') for s in scripts)
    if not voices and not targets and not _is_all_sfx_vc:
        print("Error: TTS+VC mode requires at least one 'voice' or 'target' parameter")
        return False

    has_colon_script = any(':' in s for s in scripts)
    has_colon_voice = any(':' in v for v in voices) if voices else False
    has_colon_target = any(':' in t for t in targets) if targets else False

    if not has_colon_script and not has_colon_target:
        if len(scripts) != 1:
            print("Error: Single mode expects exactly one script argument")
            return False
        if len(targets) != 1:
            print("Error: Single mode expects exactly one target argument")
            return False
        if music_description:
            print("Warning: Background music is only supported for dialogue mode. Ignoring music parameter.")
        script = scripts[0].replace('\\n', '\n')
        target_path = targets[0]
        valid, msg = validate_audio_file(target_path)
        if not valid:
            print(f"Error: {msg}")
            return False
        if msg == "video":
            print("Extracting audio from video...")
            extracted = extract_audio_from_video_cli(target_path)
            if not extracted:
                print("Error: Could not extract audio from video")
                return False
            target_path = extracted
        print("Loading Qwen-TTS model...")
        tts = QwenTTS()
        print("Extracting voice characteristics...")
        success = tts.extract_voice(target_path)
        if not success:
            print("Error: Voice extraction failed")
            return False
        print("Generating speech with cloned voice...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_vc_{timestamp}.wav")
        success = tts.synthesize(script, output_path)
        if not success:
            print("Error: Synthesis failed")
            return False
        print(f"✓ Success! Output saved to: {output_path}")
        return True
    else:
        if not has_colon_script:
            print("Error: Dialogue script must be in format 'Character: text'")
            return False
        if not _is_all_sfx_vc and not has_colon_voice and not has_colon_target:
            print("Error: Dialogue mode requires voice or target parameters in 'Character: value' format.")
            return False
        dialogue_items = []
        for idx, s in enumerate(scripts, start=1):
            if ':' not in s:
                print(f"Error: Dialogue script must be in format 'Character: text', got: {s}")
                return False
            char, text = s.split(':', 1)
            char = char.strip()
            text = text.strip()
            if not char:
                print(f"Error: Empty character in script: {s}")
                return False
            if char.lower() == 'sfx' and not text:
                print(f"Error: Empty SFX prompt in script: {s}")
                return False
            clean_text, directives_raw = _parse_script_directives(text)
            parsed_directives, errors = _parse_directives_for_line(directives_raw)
            if errors:
                print(f"Error in script line {idx}: {'; '.join(errors)}")
                return False
            if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                print(f"Error: SFX line {idx} requires /duration:nn (1-30)")
                return False
            if not clean_text and char.lower() != 'sfx':
                print(f"Error: Empty text in script: {s}")
                return False
            dialogue_items.append((idx, char, clean_text, parsed_directives))

        voice_prompts = {}
        for v in voices:
            if ':' not in v:
                print(f"Error: Voice prompt must be in format 'Character: prompt', got: {v}")
                return False
            char, prompt = v.split(':', 1)
            char = char.strip()
            prompt = prompt.strip()
            if not char or not prompt:
                print(f"Error: Empty character or prompt in voice: {v}")
                return False
            voice_prompts[char.lower()] = prompt

        target_assignments = {}
        for t in targets:
            if ':' not in t:
                print(f"Error: Target assignment must be in format 'Character: path', got: {t}")
                return False
            char, path = t.split(':', 1)
            char = char.strip()
            path = path.strip()
            if not char or not path:
                print(f"Error: Empty character or path in target: {t}")
                return False
            valid, msg = validate_audio_file(path)
            if not valid:
                print(f"Error: {msg}")
                return False
            if msg == "video":
                print(f"Extracting audio from video for character '{char}'...")
                extracted = extract_audio_from_video_cli(path)
                if not extracted:
                    print(f"Error: Could not extract audio from video for character '{char}'")
                    return False
                path = extracted
            target_assignments[char.lower()] = path

        overlap = set(voice_prompts.keys()) & set(target_assignments.keys())
        if overlap:
            print(f"Error: Character(s) specified in both voice and target: {', '.join(overlap)}")
            return False

        script_chars = set()
        for _, char, _, _ in dialogue_items:
            if char.lower() != 'sfx':
                script_chars.add(char.lower())
        all_assigned = set(voice_prompts.keys()) | set(target_assignments.keys())
        missing = script_chars - all_assigned
        if missing:
            print(f"Error: Missing voice/target for characters: {', '.join(missing)}")
            return False

        has_tts_chars = len(voice_prompts) > 0
        has_vc_chars = len(target_assignments) > 0

        if music_description and music_description.strip() == "":
            music_description = None

        if music_level_spec and not music_description:
            print("Warning: Level spec ignored (no music description provided)")

        tts_design = None
        if has_tts_chars:
            print("Loading Qwen-TTS VoiceDesign model...")
            tts_design = QwenTTSVoiceDesign()
            if tts_design.model is None:
                print("Error: Failed to load VoiceDesign model")
                return False

        tts = None
        vc_voice_prompts = None
        if has_vc_chars:
            print("Loading Qwen-TTS model...")
            tts = QwenTTS()
            vc_voice_prompts = {}
            for char_lower, audio_path in target_assignments.items():
                print(f"Extracting voice for '{char_lower}'...")
                success = tts.extract_voice(audio_path)
                if not success:
                    print(f"Error: Failed to extract voice from {audio_path}")
                    return False
                vc_voice_prompts[char_lower] = tts.voice_prompt
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"voder_tts_vc_dialogue_{timestamp}"
        if music_description:
            base_name += "_m"
        output_path = os.path.join(results_dir, f"{base_name}.wav")

        dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        dialogue_temp.close()

        has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
        has_effects = any(
            item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
            for item in dialogue_items
        ) if len(dialogue_items) > 0 else False

        if has_sfx or has_effects or has_tts_chars or has_vc_chars:
            success, msg = _assemble_enhanced_dialogue(
                dialogue_items, voice_prompts, tts_design_obj=tts_design,
                tts_vc_obj=tts, vc_voice_data=vc_voice_prompts,
                output_path=dialogue_temp.name, mode='tts_vc'
            )
            if not success:
                print(f"Error: {msg}")
                return False
        else:
            temp_dir = tempfile.mkdtemp()
            try:
                temp_files = []
                simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
                for i, (num, char, script_text) in enumerate(simple_items):
                    char_lower = char.lower()
                    print(f"Processing line {num} for '{char}'...")
                    tts.voice_prompt = vc_voice_prompts[char_lower]
                    temp_file = os.path.join(temp_dir, f"line_{num}.wav")
                    temp_files.append((num, temp_file))
                    success = tts.synthesize(script_text, temp_file)
                    if not success:
                        print(f"Error: Failed to generate speech for line {num}")
                        return False
                temp_files.sort(key=lambda x: x[0])
                concat_list = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list, 'w') as f:
                    for _, tf in temp_files:
                        f.write(f"file '{tf}'\n")
                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', dialogue_temp.name]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"Error: FFmpeg concatenation failed: {result.stderr}")
                    return False
            finally:
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

        if music_description:
            if tts_design is not None:
                del tts_design
                tts_design = None
            if tts is not None:
                del tts
                tts = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ace = AceStepWrapper()
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False
            success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec)
            del ace
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not success:
                return False
            os.unlink(dialogue_temp.name)
        else:
            shutil.move(dialogue_temp.name, output_path)
        print(f"✓ Success! Output saved to: {output_path}")
        if tts_design is not None:
            del tts_design
        if tts is not None:
            del tts
        return True

def oneline_sts(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    is_music = params.get('is_music', False)

    if 'base' not in params or len(params['base']) != 1:
        print("Error: STS mode requires exactly one 'base' parameter")
        return False
    if 'target' not in params or len(params['target']) != 1:
        print("Error: STS mode requires exactly one 'target' parameter")
        return False
    base_path = params['base'][0]
    target_path = params['target'][0]
    if not os.path.exists(base_path):
        print(f"Error: Base file not found: {base_path}")
        return False
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return False
    if base_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"voder_cli_{int(time.time())}.wav")
        cmd = ['ffmpeg', '-i', base_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(audio_path):
            print("Error: Could not extract audio from video")
            return False
        base_path = audio_path
    if is_music:
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            return False
        print("Resampling inputs to 44100Hz...")
        import torchaudio
        waveform_base, sr_base = torchaudio.load(base_path)
        if sr_base != 44100:
            resampler_base = torchaudio.transforms.Resample(sr_base, 44100)
            waveform_base = resampler_base(waveform_base)
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_base.name, waveform_base, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_base.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
            shutil.copy(temp_output_44k.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_base.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    else:
        print("Loading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            return False
        print("Resampling inputs to 22050Hz...")
        import torchaudio
        waveform_base, sr_base = torchaudio.load(base_path)
        if sr_base != 22050:
            resampler_base = torchaudio.transforms.Resample(sr_base, 22050)
            waveform_base = resampler_base(waveform_base)
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_base = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_base.name, waveform_base, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_base.name,
                reference_path=temp_target.name,
                output_path=temp_output_22k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            print("Upsampling output to 44100Hz...")
            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
            if sr_out != 44100:
                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                waveform_out = resampler_out(waveform_out)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
            torchaudio.save(output_path, waveform_out, 44100)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_base.name, temp_target.name, temp_output_22k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

def oneline_ttm(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    if 'lyrics' not in params or len(params['lyrics']) != 1:
        print("Error: TTM mode requires exactly one 'lyrics' parameter")
        return False
    if 'styling' not in params or len(params['styling']) != 1:
        print("Error: TTM mode requires exactly one 'styling' parameter")
        return False
    if 'duration' not in params:
        print("Error: TTM mode requires duration (10-300 seconds)")
        return False
    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False
    lyrics = params['lyrics'][0].replace('\\n', '\n')
    style = params['styling'][0].replace('\\n', '\n')
    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False
    print(f"Generating music ({duration}s duration)...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
    success = ace_step.generate(
        lyrics=lyrics,
        style_prompt=style,
        output_path=output_path,
        duration=duration
    )
    if not success:
        print("Error: Music generation failed")
        return False
    print(f"✓ Success! Output saved to: {output_path}")
    return True

def oneline_ttm_vc(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    if 'lyrics' not in params or len(params['lyrics']) != 1:
        print("Error: TTM+VC mode requires exactly one 'lyrics' parameter")
        return False
    if 'styling' not in params or len(params['styling']) != 1:
        print("Error: TTM+VC mode requires exactly one 'styling' parameter")
        return False
    if 'target' not in params or len(params['target']) != 1:
        print("Error: TTM+VC mode requires exactly one 'target' parameter")
        return False
    if 'duration' not in params:
        print("Error: TTM+VC mode requires duration (10-300 seconds)")
        return False
    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False
    target_path = params['target'][0]
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return False
    lyrics = params['lyrics'][0].replace('\\n', '\n')
    style = params['styling'][0].replace('\\n', '\n')
    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False
    temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_ttm_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_target_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        print(f"Generating music ({duration}s duration)...")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=temp_ttm_output.name,
            duration=duration
        )
        if not success:
            print("Error: Music generation failed")
            return False
        print("Resampling TTM output to 44100Hz...")
        import torchaudio
        waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
        if sr_ttm != 44100:
            resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 44100)
            waveform_ttm = resampler_ttm(waveform_ttm)
        torchaudio.save(temp_ttm_22k.name, waveform_ttm, 44100)
        print("Resampling target voice to 44100Hz...")
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        torchaudio.save(temp_target_22k.name, waveform_target, 44100)
        print("Clearing ACE-Step from memory...")
        del ace_step
        ace_step = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Loading Seed-VC v1 model...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            return False
        print("Converting voice...")
        vc_success = seed_vc.convert(
            source_path=temp_ttm_22k.name,
            reference_path=temp_target_22k.name,
            output_path=temp_vc_output.name
        )
        if not vc_success:
            print("Error: Voice conversion failed")
            return False
        print("Saving output...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
        shutil.copy(temp_vc_output.name, output_path)
        print(f"✓ Success! Output saved to: {output_path}")
        print("Clearing Seed-VC from memory...")
        del seed_vc
        seed_vc = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    finally:
        for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def oneline_stt(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])
    keep_timestamp = params.get('timestamp', False)
    enable_dialogue = params.get('dialogue', False)

    if not files:
        print("Error: STT mode requires at least one audio/video/image file path or YouTube URL")
        return False

    for file_path in files:
        if not os.path.exists(file_path) and not is_youtube_url(file_path):
            print(f"Error: File not found or invalid YouTube URL: {file_path}")
            return False

    success_count = 0
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        print("=" * 60)

        is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))

        if is_image:
            try:
                print("Loading EasyOCR model...")
                ocr = EasyOCRReader()
                if ocr.reader is None:
                    print("Error: Failed to load EasyOCR model")
                    continue

                print(f"Extracting text from image...")
                success, text, error_msg = ocr.extract_text_from_image(file_path)

                ocr.cleanup()
                del ocr
                gc.collect()

                if not success:
                    print(f"Error: {error_msg or 'Failed to extract text from image'}")
                    continue

                if not text:
                    print(f"Error: No text found in image")
                    continue

                formatted_text = f"image: {text}"

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                suffix = "stt_ocr"

                output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
                output_path = os.path.join(results_dir, output_filename)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)

                print(f"✓ Success! Output saved to: {output_path}")
                success_count += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            continue

        audio_path = file_path
        needs_youtube_download = is_youtube_url(file_path)
        needs_extraction = False

        if needs_youtube_download:
            print("Downloading audio from YouTube...")
            success, error_msg, audio_path = download_youtube_audio(file_path)
            if not success:
                print(f"Error: {error_msg}")
                continue
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            extracted = extract_audio_from_video_cli(file_path)
            if not extracted:
                print(f"Error: Could not extract audio from {file_path}")
                continue
            audio_path = extracted
            needs_extraction = True

        try:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                continue

            print("Transcribing audio...")
            result = stt.transcribe(audio_path)
            if not result:
                print(f"Error: Transcription failed for {file_path}")
                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            def format_time_range(start, end):
                def format_single(seconds):
                    if seconds is None:
                        seconds = 0
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 100)
                    return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                return f"[{format_single(start)}-{format_single(end)}]"

            def format_time(seconds):
                if seconds is None:
                    seconds = 0
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 100)
                return f"[{minutes:02d}:{secs:02d}:{millis:02d}]"

            if enable_dialogue:
                print("Performing speaker diarization...")
                diarization = SpeakerDiarization()
                if diarization.pipeline is None:
                    print("Warning: Speaker diarization model not available, proceeding without it")
                    if keep_timestamp and result.get("segments"):
                        lines = []
                        for seg in result.get("segments", []):
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            text = seg.get("text", "").strip()
                            if text:
                                lines.append(f"{format_time_range(start, end)} text: {text}")
                        if lines:
                            formatted_text = "\n".join(lines)
                        else:
                            formatted_text = result.get("text", "").strip()
                    else:
                        formatted_text = result.get("text", "").strip()
                else:
                    diar_result = diarization.diarize(audio_path)
                    formatted_segments = diarization.format_diarization(diar_result, result)

                    if formatted_segments:
                        original_speakers = []
                        for seg in formatted_segments:
                            speaker = seg["speaker"]
                            if speaker not in original_speakers:
                                original_speakers.append(speaker)

                        speaker_mapping = {spk: idx for idx, spk in enumerate(sorted(original_speakers), 1)}

                        if len(original_speakers) == 1:
                            content = " ".join(seg["text"] for seg in formatted_segments)
                            if keep_timestamp:
                                first_time = formatted_segments[0]["start"]
                                last_time = formatted_segments[-1]["end"]
                                formatted_text = f"{format_time_range(first_time, last_time)} text: {content}"
                            else:
                                formatted_text = f"text: {content}"
                        else:
                            lines = []
                            current_speaker_num = None
                            current_text_parts = []
                            current_first_time = None
                            current_last_time = None
                            min_words_before_switch = 3
                            min_gap_before_switch = 0.3

                            for seg in formatted_segments:
                                speaker_num = speaker_mapping[seg["speaker"]]
                                text = seg["text"]
                                seg_start = seg.get("start", 0) or 0
                                seg_end = seg.get("end", 0) or 0

                                if current_speaker_num is None:
                                    current_speaker_num = speaker_num
                                    current_text_parts = [text]
                                    current_first_time = seg_start
                                    current_last_time = seg_end
                                elif speaker_num == current_speaker_num:
                                    current_text_parts.append(text)
                                    current_last_time = seg_end
                                else:
                                    time_gap = seg_start - current_last_time if current_last_time else 0
                                    words_accumulated = len(current_text_parts)

                                    should_switch = (words_accumulated >= min_words_before_switch) or (time_gap >= min_gap_before_switch)

                                    if should_switch and current_text_parts:
                                        content = " ".join(current_text_parts)
                                        if keep_timestamp:
                                            lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                        else:
                                            lines.append(f"{current_speaker_num}: {content}")
                                        current_speaker_num = speaker_num
                                        current_text_parts = [text]
                                        current_first_time = seg_start
                                        current_last_time = seg_end
                                    else:
                                        current_text_parts.append(text)
                                        current_last_time = seg_end

                            if current_text_parts:
                                content = " ".join(current_text_parts)
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                else:
                                    lines.append(f"{current_speaker_num}: {content}")

                            formatted_text = "\n".join(lines)

                        del diarization
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        if keep_timestamp and result.get("segments"):
                            lines = []
                            for seg in result.get("segments", []):
                                start = seg.get("start", 0)
                                end = seg.get("end", 0)
                                text = seg.get("text", "").strip()
                                if text:
                                    lines.append(f"{format_time_range(start, end)} text: {text}")
                            if lines:
                                formatted_text = "\n".join(lines)
                            else:
                                formatted_text = result.get("text", "").strip()
                        else:
                            formatted_text = result.get("text", "").strip()
            else:
                formatted_text = result.get("text", "").strip()

                if keep_timestamp and result.get("segments"):
                    lines = []
                    for seg in result.get("segments", []):
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "").strip()
                        if text:
                            lines.append(f"{format_time_range(start, end)} text: {text}")
                    if lines:
                        formatted_text = "\n".join(lines)
                    else:
                        formatted_text = result.get("text", "").strip()

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if is_youtube_url(file_path):
                base_name = f"youtube_{len(files)}_{success_count + 1}"
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]

            suffix_parts = ["stt"]
            if keep_timestamp:
                suffix_parts.append("timestamp")
            if enable_dialogue:
                suffix_parts.append("dialogue")
            suffix = "_".join(suffix_parts)

            output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
            output_path = os.path.join(results_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)

            print(f"✓ Success! Output saved to: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            if file_path != audio_path and os.path.exists(audio_path):
                try:
                    os.unlink(audio_path)
                except:
                    pass

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {success_count}/{len(files)} files successful")
    return success_count > 0

def oneline_se(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])

    if not files:
        print("Error: SE mode requires at least one audio/video file path")
        return False

    for file_path in files:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return False

    print("Loading UniSE Speech Enhancement model...")
    from unise import UniSEEnhancer
    enhancer = UniSEEnhancer(UNISE_DIR)
    enhancer.ensure_model()
    if enhancer.model is None:
        print("Error: Failed to load UniSE model")
        return False

    success_count = 0
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        print("=" * 60)

        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if is_video:
                output_filename = f"voder_se_{timestamp}.mp4"
                output_path = os.path.join(results_dir, output_filename)
                print("Enhancing speech in video...")
                success = enhancer.enhance_video(file_path, output_path)
            else:
                output_filename = f"voder_se_{timestamp}.wav"
                output_path = os.path.join(results_dir, output_filename)
                print("Enhancing speech in audio...")
                success = enhancer.enhance(file_path, output_path)

            if success:
                print(f"\n✓ Success! Output saved to: {output_path}")
                success_count += 1
            else:
                print(f"Error: Enhancement failed for {file_path}")

        except Exception as e:
            traceback.print_exc()
            print(f"Error processing {file_path}: {e}")

    enhancer.cleanup()
    del enhancer
    enhancer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {success_count}/{len(files)} files successful")
    return success_count > 0

def cli_se_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SE Mode ---")
    print("Speech Enhancement - denoise, dereverb, restore audio")
    print("Note: Outputs 16kHz audio. Not for musical enhancement.")
    print()

    while True:
        file_path = input("Enter audio/video file path: ").strip()
        if os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            is_valid_audio = False
            is_video = ext in VIDEO_EXTENSIONS
            if is_video:
                break
            try:
                torchaudio.load(file_path)
                is_valid_audio = True
                break
            except Exception:
                pass
            if not is_valid_audio:
                print("Error: Unsupported or corrupt file format.")
        else:
            print("Error: File not found. Please try again.")

    print("Loading UniSE Speech Enhancement model...")
    from unise import UniSEEnhancer
    enhancer = UniSEEnhancer(UNISE_DIR)
    enhancer.ensure_model()
    if enhancer.model is None:
        print("Error: Failed to load UniSE model")
        return False

    try:
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if is_video:
            output_filename = f"voder_se_{timestamp}.mp4"
            output_path = os.path.join(results_dir, output_filename)
            print("Enhancing speech in video...")
            success = enhancer.enhance_video(file_path, output_path)
        else:
            output_filename = f"voder_se_{timestamp}.wav"
            output_path = os.path.join(results_dir, output_filename)
            print("Enhancing speech in audio...")
            success = enhancer.enhance(file_path, output_path)

        if success:
            print(f"\n✓ Success! Output saved to: {output_path}")
        else:
            print("Error: Enhancement failed")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        enhancer.cleanup()
        del enhancer
        enhancer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return True

def oneline_sfx(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    prompt = params.get('prompt', '')
    duration = params.get('duration', 10)
    steps = params.get('steps', 30)
    guide = params.get('guide', 4.5)

    if duration > 30:
        print("Warning: Duration >30s clamped to 30s (model maximum).")
        duration = 30

    print(f"SFX Generation")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: {duration}s")
    print(f"  Steps: {steps}")
    print(f"  Guidance: {guide}")

    print("Loading TangoFlux SFX model...")
    from tangoflux import TangoFluxGenerator
    generator = TangoFluxGenerator(TANGOFLUX_DIR)
    generator.ensure_model()
    if generator.model is None:
        print("Error: Failed to load TangoFlux model")
        return False

    try:
        print(f"\nGenerating sound effect...")
        audio = generator.generate(prompt, duration, steps=steps, guidance_scale=guide)
        if audio is None:
            print("Error: Generation failed")
            return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"voder_sfx_{timestamp}.wav"
        output_path = os.path.join(results_dir, output_filename)

        if generator.save(audio, output_path):
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        else:
            print("Error: Failed to save output")
            return False

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return False
    finally:
        generator.cleanup()
        del generator
        generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def cli_sfx_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SFX Mode ---")
    print("Sound Effects Generation - text prompt + duration → audio")
    print("Steps: 30, Guidance: 4.5 (hardcoded for best results)")
    print()

    while True:
        prompt = input("Enter sound prompt: ").strip()
        if prompt:
            break
        print("Error: Prompt cannot be empty. Please try again.")

    while True:
        duration_input = input("Enter duration in seconds (1-30): ").strip()
        if not duration_input:
            print("Error: Duration cannot be empty. Please try again.")
            continue
        try:
            duration = int(duration_input)
            if duration < 1:
                print("Error: Duration must be at least 1 second. Please try again.")
                continue
            if duration > 30:
                print("Warning: Duration >30s clamped to 30s (model maximum).")
                duration = 30
            break
        except ValueError:
            print("Error: Invalid number. Please enter a number between 1 and 30.")

    print("\nLoading TangoFlux SFX model...")
    from tangoflux import TangoFluxGenerator
    generator = TangoFluxGenerator(TANGOFLUX_DIR)
    generator.ensure_model()
    if generator.model is None:
        print("Error: Failed to load TangoFlux model")
        return False

    try:
        print(f"\nGenerating sound effect...")
        audio = generator.generate(prompt, duration, steps=30, guidance_scale=4.5)
        if audio is None:
            print("Error: Generation failed")
            return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"voder_sfx_{timestamp}.wav"
        output_path = os.path.join(results_dir, output_filename)

        if generator.save(audio, output_path):
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        else:
            print("Error: Failed to save output")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        generator.cleanup()
        del generator
        generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def interactive_cli_mode():
    while True:
        print_banner()
        print("\nSelect Mode:")
        print("1. STT+TTS (Speech-to-Text + Text-to-Speech)")
        print("2. TTS (Text-to-Speech)")
        print("3. TTS+VC (Text-to-Speech + Voice Clone)")
        print("4. STS (Speech-to-Speech / Voice Conversion)")
        print("5. TTM (Text-to-Music)")
        print("6. TTM+VC (Text-to-Music + Voice Conversion)")
        print("7. SE (Speech Enhancement)")
        print("8. SFX (Sound Effects Generation)")
        choice = input("\nEnter your choice (1-8): ").strip()
        success = False
        if choice == '1':
            success = cli_stt_tts_mode()
        elif choice == '2':
            success = cli_tts_mode()
        elif choice == '3':
            success = cli_tts_vc_mode()
        elif choice == '4':
            success = cli_sts_mode()
        elif choice == '5':
            success = cli_ttm_mode()
        elif choice == '6':
            success = cli_ttm_vc_mode()
        elif choice == '7':
            success = cli_se_mode()
        elif choice == '8':
            success = cli_sfx_mode()
        else:
            print("Invalid choice. Please enter 1-8.")
            continue
        print("\n--- What's Next? ---")
        print("1. Blend Again")
        print("2. Exit")
        while True:
            next_choice = input("\nEnter your choice (1-2): ").strip()
            if next_choice == '1':
                print("\n" + "=" * 60 + "\n")
                break
            elif next_choice == '2':
                print("\nThank you for using VODER! Goodbye!")
                print("Results saved to: results/")
                return
            else:
                print("Invalid choice. Please enter 1 or 2.")

def parse_and_execute_oneline(args):
    parsed = parse_oneline_args(args)
    if parsed.get('error'):
        print(f"Error: {parsed['error']}")
        show_oneline_usage()
        return False
    mode = validate_oneline_mode(parsed['mode'])
    if mode == 'stt+tts_rejected':
        print("Error: STT+TTS mode is not available in one-line mode.")
        print("Reason: This mode requires interactive text editing.")
        print("Solutions:")
        print("  - Use 'tts' mode with your text directly")
        print("  - Use 'sts' mode to convert speech to target voice")
        print("  - Use interactive CLI: python voder.py cli")
        return False
    if mode is None:
        print(f"Error: Invalid mode '{parsed['mode']}'")
        show_oneline_usage()
        return False
    parsed['mode'] = mode
    return execute_oneline_command(parsed)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cli' and len(sys.argv) == 2:
            interactive_cli_mode()
            sys.exit(0)
        arg_offset = 1
        if sys.argv[1] == 'cli':
            arg_offset = 2
        if len(sys.argv) > arg_offset:
            result = parse_and_execute_oneline(sys.argv[arg_offset:])
            sys.exit(0 if result else 1)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = VODERGUI()
    window.show()
    sys.exit(app.exec_())
