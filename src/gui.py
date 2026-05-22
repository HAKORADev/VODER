import sys
import os
import subprocess
import platform
import re
import time

def _q(v):
    if not v:
        return '""'
    if re.match(r'^[a-zA-Z0-9_\.\-/]+$', v):
        return v
    return '"' + v.replace('"', '\\"') + '"'

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QFileDialog,
                             QMessageBox, QProgressBar, QFrame, QComboBox,
                             QTextEdit, QListWidget, QListWidgetItem, QLineEdit,
                             QSpinBox, QScrollArea, QCheckBox, QRadioButton,
                             QTabWidget, QDoubleSpinBox, QShortcut)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QColor, QPainter, QPalette, QKeySequence

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

try:
    import torchaudio
    HAS_TORCHAUDIO = True
except Exception:
    HAS_TORCHAUDIO = False

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

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
    return get_main_button_style()

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

def get_checkbox_style():
    return f"""
        QCheckBox {{
            color: {THEME['text']};
            font-size: 13px;
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {THEME['border_light']};
            border-radius: 4px;
            background-color: {THEME['surface']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {THEME['accent']};
            border: 2px solid {THEME['accent']};
        }}
        QCheckBox::indicator:hover {{
            border: 2px solid #E5E5E5;
        }}
    """

def get_group_box_style():
    return f"""
        QGroupBox {{
            color: {THEME['text']};
            font-weight: bold;
            font-size: 13px;
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 16px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
        }}
    """

def get_radio_button_style():
    return f"""
        QRadioButton {{
            color: {THEME['text']};
            font-size: 13px;
            spacing: 8px;
        }}
        QRadioButton::indicator {{
            width: 18px;
            height: 18px;
            border: 2px solid {THEME['border_light']};
            border-radius: 9px;
            background-color: {THEME['surface']};
        }}
        QRadioButton::indicator:checked {{
            background-color: {THEME['accent']};
            border: 2px solid {THEME['accent']};
        }}
    """

def get_spin_box_style():
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 2px solid {THEME['border']};
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 13px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {THEME['accent']};
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: {THEME['surface_hover']};
            border: 1px solid {THEME['border']};
            width: 16px;
        }}
    """


class AudioWaveformWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        self.setStyleSheet(f"background-color: {THEME['surface']}; border: 1px solid {THEME['border']};")
        self.audio_data = None
        self.sample_rate = 44100

    def set_audio(self, audio_path):
        if audio_path and os.path.exists(audio_path) and HAS_TORCHAUDIO and HAS_NUMPY:
            try:
                waveform, sample_rate = torchaudio.load(audio_path)
                self.audio_data = waveform[0].numpy()
                self.sample_rate = sample_rate
                self.update()
            except Exception:
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


class SubprocessThread(QThread):
    output_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(int, str)

    def __init__(self, command_args):
        super().__init__()
        self.command_args = command_args
        self.process = None

    def run(self):
        python = sys.executable
        voder_path = os.path.join(_src_dir, 'voder.py')
        cmd = [python, voder_path] + self.command_args
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=_src_dir,
                bufsize=1
            )
            output_lines = []
            for line in self.process.stdout:
                line = line.rstrip('\n')
                output_lines.append(line)
                self.output_signal.emit(line)
            self.process.wait()
            full_output = '\n'.join(output_lines)
            self.finished_signal.emit(self.process.returncode, full_output)
        except Exception as e:
            self.output_signal.emit(f"Error launching process: {e}")
            self.finished_signal.emit(1, str(e))

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class AudioPlayerThread(QThread):
    finished_signal = pyqtSignal()

    def __init__(self, path):
        super().__init__()
        self.path = path
        self.process = None

    def run(self):
        if not self.path or not os.path.exists(self.path):
            self.finished_signal.emit()
            return
        try:
            system = platform.system()
            if system == "Darwin":
                self.process = subprocess.Popen(["afplay", self.path])
            elif system == "Windows":
                os.startfile(self.path)
                self.process = None
            else:
                self.process = subprocess.Popen(
                    ["aplay", self.path],
                    stderr=subprocess.DEVNULL
                )
            if self.process:
                self.process.wait()
        except Exception:
            pass
        self.finished_signal.emit()

    def stop(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


class HelperWidgets:

    @staticmethod
    def make_file_picker(parent_layout, label_text, placeholder="", file_filter="Audio Files (*.wav *.mp3 *.flac *.ogg);;Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(get_line_edit_style())
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(get_surface_button_style())
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setFixedWidth(80)

        def pick_file():
            path, _ = QFileDialog.getOpenFileName(None, "Select File", "", file_filter)
            if path:
                line_edit.setText(path)

        browse_btn.clicked.connect(pick_file)
        row.addWidget(label)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(browse_btn)
        parent_layout.addLayout(row)
        return line_edit

    @staticmethod
    def make_save_picker(parent_layout, label_text, placeholder=""):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(get_line_edit_style())
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(get_surface_button_style())
        browse_btn.setCursor(Qt.PointingHandCursor)
        browse_btn.setFixedWidth(80)

        def pick_file():
            path, _ = QFileDialog.getSaveFileName(None, "Save To", "", "WAV Files (*.wav);;All Files (*)")
            if path:
                line_edit.setText(path)

        browse_btn.clicked.connect(pick_file)
        row.addWidget(label)
        row.addWidget(line_edit, stretch=1)
        row.addWidget(browse_btn)
        parent_layout.addLayout(row)
        return line_edit

    @staticmethod
    def make_label(parent_layout, text):
        label = QLabel(text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px;")
        parent_layout.addWidget(label)
        return label

    @staticmethod
    def make_subtitle(parent_layout, text):
        label = QLabel(text)
        label.setStyleSheet(get_subtitle_label_style())
        parent_layout.addWidget(label)
        return label

    @staticmethod
    def make_line_edit(parent_layout, label_text, placeholder=""):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setStyleSheet(get_line_edit_style())
        row.addWidget(label)
        row.addWidget(line_edit, stretch=1)
        parent_layout.addLayout(row)
        return line_edit

    @staticmethod
    def make_text_edit(parent_layout, label_text, placeholder="", min_height=80):
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px;")
        parent_layout.addWidget(label)
        text_edit = QTextEdit()
        text_edit.setPlaceholderText(placeholder)
        text_edit.setStyleSheet(get_text_edit_style())
        text_edit.setMinimumHeight(min_height)
        text_edit.setMaximumHeight(min_height * 2)
        parent_layout.addWidget(text_edit)
        return text_edit

    @staticmethod
    def make_checkbox(parent_layout, text):
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet(get_checkbox_style())
        parent_layout.addWidget(checkbox)
        return checkbox

    @staticmethod
    def make_spinbox(parent_layout, label_text, min_val=0, max_val=300, default=30):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setStyleSheet(get_spin_box_style())
        row.addWidget(label)
        row.addWidget(spinbox)
        row.addStretch()
        parent_layout.addLayout(row)
        return spinbox

    @staticmethod
    def make_double_spinbox(parent_layout, label_text, min_val=0.0, max_val=100.0, default=4.5, step=0.5, decimals=1):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setStyleSheet(get_spin_box_style())
        row.addWidget(label)
        row.addWidget(spinbox)
        row.addStretch()
        parent_layout.addLayout(row)
        return spinbox

    @staticmethod
    def make_separator(parent_layout):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color: {THEME['border']};")
        parent_layout.addWidget(line)

    @staticmethod
    def make_button(parent_layout, text, style=None):
        btn = QPushButton(text)
        btn.setStyleSheet(style or get_surface_button_style())
        btn.setCursor(Qt.PointingHandCursor)
        parent_layout.addWidget(btn)
        return btn

    @staticmethod
    def make_button_row(parent_layout, buttons_data):
        row = QHBoxLayout()
        btns = []
        for text, style in buttons_data:
            btn = QPushButton(text)
            btn.setStyleSheet(style or get_surface_button_style())
            btn.setCursor(Qt.PointingHandCursor)
            row.addWidget(btn)
            btns.append(btn)
        row.addStretch()
        parent_layout.addLayout(row)
        return btns


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

        delete_btn = QPushButton("\u00d7")
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
        seen = set()
        for char_edit, _, _, _ in self.rows:
            text = char_edit.text().strip()
            if text and text.lower() not in seen:
                chars.add(text.lower())
                seen.add(text.lower())
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

    def set_text(self, text):
        self.clear()
        for line in text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                char, dialogue = line.split(':', 1)
                self.rows[-1][0].setText(char.strip())
                self.rows[-1][1].setText(dialogue.strip())
                self.add_row()
            else:
                self.rows[-1][1].setText(line)


class KeyValueRow(QWidget):
    remove_signal = pyqtSignal(object)

    def __init__(self, key_label="Key", value_label="Value", key_placeholder="", value_placeholder="", parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText(key_placeholder or key_label)
        self.key_edit.setStyleSheet(get_line_edit_style())
        self.key_edit.setMinimumWidth(100)
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText(value_placeholder or value_label)
        self.val_edit.setStyleSheet(get_line_edit_style())
        self.browse_btn = None
        self.del_btn = QPushButton("x")
        self.del_btn.setFixedSize(24, 24)
        self.del_btn.setStyleSheet("""
            QPushButton { background-color: #3a3a3a; color: white; border: none; border-radius: 12px; font-size: 14px; font-weight: bold; min-width: 24px; max-width: 24px; }
            QPushButton:hover { background-color: #f44336; }
        """)
        self.del_btn.setCursor(Qt.PointingHandCursor)
        self.del_btn.clicked.connect(lambda: self.remove_signal.emit(self))
        layout.addWidget(self.key_edit)
        layout.addWidget(self.val_edit, stretch=1)
        layout.addWidget(self.del_btn)

    def set_browse(self, file_filter="Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"):
        if self.browse_btn is None:
            self.browse_btn = QPushButton("Browse")
            self.browse_btn.setStyleSheet(get_surface_button_style())
            self.browse_btn.setCursor(Qt.PointingHandCursor)
            self.browse_btn.setFixedWidth(80)
            self.layout().insertWidget(2, self.browse_btn)

        def pick():
            path, _ = QFileDialog.getOpenFileName(None, "Select", "", file_filter)
            if path:
                self.val_edit.setText(path)

        self.browse_btn.clicked.connect(pick)

    def get_key(self):
        return self.key_edit.text().strip()

    def get_value(self):
        return self.val_edit.text().strip()


class KeyValueList(QWidget):
    def __init__(self, key_label="Key", value_label="Value", key_placeholder="", value_placeholder="", with_browse=False, file_filter="", parent=None):
        super().__init__(parent)
        self.key_label = key_label
        self.value_label = value_label
        self.key_placeholder = key_placeholder
        self.value_placeholder = value_placeholder
        self.with_browse = with_browse
        self.file_filter = file_filter
        self.rows = []
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(4)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        self.rows_layout = QVBoxLayout(scroll_widget)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        self.scroll_area.setWidget(scroll_widget)
        self.layout.addWidget(self.scroll_area)
        spacer = QWidget()
        spacer.setFixedHeight(8)
        self.layout.addWidget(spacer)
        self.add_btn = QPushButton(f"+ Add {key_label}")
        self.add_btn.setStyleSheet(get_surface_button_style())
        self.add_btn.setCursor(Qt.PointingHandCursor)
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)
        self.add_row()

    def setMinimumHeight(self, h):
        super().setMinimumHeight(h)
        self.scroll_area.setMinimumHeight(h)

    def set_scroll_height(self, h):
        self.scroll_area.setMinimumHeight(h)

    def add_row(self, key="", value=""):
        row = KeyValueRow(self.key_label, self.value_label, self.key_placeholder, self.value_placeholder)
        row.remove_signal.connect(self.remove_row)
        if self.with_browse:
            row.set_browse(self.file_filter)
        row.key_edit.setText(key)
        row.val_edit.setText(value)
        if self.rows:
            prev = self.rows[-1]
            prev.key_edit.textChanged.disconnect()
            prev.val_edit.textChanged.disconnect()
            prev.key_edit.textChanged.connect(prev._kv_emit)
            prev.val_edit.textChanged.connect(prev._kv_emit)
        row._kv_emit = lambda: self._on_last_row_changed()
        row.key_edit.textChanged.connect(row._kv_emit)
        row.val_edit.textChanged.connect(row._kv_emit)
        self.rows_layout.addWidget(row)
        self.rows.append(row)
        self.update_delete_buttons()

    def _on_last_row_changed(self):
        if not self.rows:
            return
        last = self.rows[-1]
        if last.key_edit.text().strip() and last.val_edit.text().strip():
            if len(self.rows) > 1:
                prev = self.rows[-2]
                if not prev.key_edit.text().strip() and not prev.val_edit.text().strip():
                    return
            self.add_row()

    def remove_row(self, row_widget):
        if len(self.rows) <= 1:
            return
        self.rows_layout.removeWidget(row_widget)
        row_widget.deleteLater()
        if row_widget in self.rows:
            self.rows.remove(row_widget)
        self.update_delete_buttons()
        if self.rows:
            last = self.rows[-1]
            try:
                last.key_edit.textChanged.disconnect()
            except Exception:
                pass
            try:
                last.val_edit.textChanged.disconnect()
            except Exception:
                pass
            last._kv_emit = lambda: self._on_last_row_changed()
            last.key_edit.textChanged.connect(last._kv_emit)
            last.val_edit.textChanged.connect(last._kv_emit)

    def update_delete_buttons(self):
        for i, row in enumerate(self.rows):
            row.del_btn.setEnabled(len(self.rows) > 1)
            row.del_btn.setVisible(len(self.rows) > 1)

    def get_items(self):
        items = []
        for row in self.rows:
            k = row.get_key()
            v = row.get_value()
            if k and v:
                items.append((k, v))
        return items

    def ensure_keys(self, keys):
        key_list = sorted(keys, key=str.lower)
        existing_filled = []
        for row in self.rows:
            k = row.get_key()
            if k:
                existing_filled.append(row)
        filled_keys = [r.get_key().lower() for r in existing_filled]
        empty_rows = [r for r in self.rows if not r.get_key()]
        ei = 0
        for key in key_list:
            if key.lower() not in filled_keys:
                if ei < len(empty_rows):
                    empty_rows[ei].key_edit.setText(key)
                    filled_keys.append(key.lower())
                    ei += 1
                else:
                    self.add_row(key=key)
                    filled_keys.append(key.lower())

    def clear(self):
        for row in list(self.rows):
            self.rows_layout.removeWidget(row)
            row.deleteLater()
        self.rows.clear()
        self.add_row()


class FileListWidget(QWidget):
    def __init__(self, placeholder="Click + to add files", file_filter="Audio Files (*.wav *.mp3 *.flac *.ogg);;Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self.paths = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.list_widget = QListWidget()
        self.list_widget.setStyleSheet(get_list_widget_style())
        self.list_widget.setMinimumHeight(60)
        self.list_widget.setMaximumHeight(120)
        layout.addWidget(self.list_widget)
        btn_row = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.setStyleSheet(get_surface_button_style())
        add_btn.setCursor(Qt.PointingHandCursor)
        add_btn.setFixedWidth(80)
        add_btn.clicked.connect(self.add_file)
        del_btn = QPushButton("- Remove")
        del_btn.setStyleSheet(get_surface_button_style())
        del_btn.setCursor(Qt.PointingHandCursor)
        del_btn.setFixedWidth(80)
        del_btn.clicked.connect(self.remove_selected)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def add_file(self):
        paths, _ = QFileDialog.getOpenFileNames(None, "Select Files", "", self.file_filter)
        for p in paths:
            if p not in self.paths:
                self.paths.append(p)
                self.list_widget.addItem(os.path.basename(p))
        self.list_widget.addItems = None

    def remove_selected(self):
        for item in self.list_widget.selectedItems():
            idx = self.list_widget.row(item)
            self.list_widget.takeItem(idx)
            if idx < len(self.paths):
                self.paths.pop(idx)

    def get_paths(self):
        return list(self.paths)

    def clear(self):
        self.paths.clear()
        self.list_widget.clear()


class TTSTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        mode_row = QHBoxLayout()
        self.mode_single = QRadioButton("Single")
        self.mode_single.setStyleSheet(get_radio_button_style())
        self.mode_single.setChecked(True)
        self.mode_dialogue = QRadioButton("Dialogue")
        self.mode_dialogue.setStyleSheet(get_radio_button_style())
        mode_row.addWidget(self.mode_single)
        mode_row.addWidget(self.mode_dialogue)
        mode_row.addStretch()
        inner.addLayout(mode_row)

        self.script_label = QLabel("Script")
        self.script_label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px;")
        inner.addWidget(self.script_label)

        self.script_text = QTextEdit()
        self.script_text.setPlaceholderText("Enter text...")
        self.script_text.setStyleSheet(get_text_edit_style())
        self.script_text.setMinimumHeight(100)
        self.script_text.setMaximumHeight(200)
        inner.addWidget(self.script_text)

        self.dialogue_widget = DialogueScriptWidget()
        self.dialogue_widget.setMinimumHeight(120)
        self.dialogue_widget.setMaximumHeight(240)
        inner.addWidget(self.dialogue_widget)
        self.dialogue_widget.hide()
        self.script_label.hide()

        self.mode_single.toggled.connect(self.on_mode_changed)

        HelperWidgets.make_label(inner, "Voice Descriptions (one per line: Character: description)")

        self.voice_list = KeyValueList("Character", "Voice Description", "Character name", "e.g. deep male voice")
        self.voice_list.setMinimumHeight(80)
        self.voice_list.add_btn.hide()
        inner.addWidget(self.voice_list)

        HelperWidgets.make_separator(inner)
        HelperWidgets.make_label(inner, "Voice Clone Targets (one per line: Character: audio path)")

        self.clone_list = KeyValueList("Character", "Audio Path", "Character name", "path/to/voice.wav", with_browse=True)
        self.clone_list.setMinimumHeight(80)
        self.clone_list.add_btn.hide()
        inner.addWidget(self.clone_list)

        HelperWidgets.make_separator(inner)
        self.music_edit = HelperWidgets.make_line_edit(inner, "Music Desc", "Background music description (dialogue mode)")
        self.level_edit = HelperWidgets.make_line_edit(inner, "Music Level", 'e.g. "10:20-50 30:60-80"')
        self.reference_edit = HelperWidgets.make_line_edit(inner, "Music Ref", "Path or URL to reference audio for music style")
        self.ocr_edit = HelperWidgets.make_file_picker(inner, "OCR Image", "Path to image for text extraction", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp)")
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run TTS")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def on_mode_changed(self, checked):
        if checked:
            self.script_text.show()
            self.script_label.show()
            self.dialogue_widget.hide()
        else:
            self.script_text.hide()
            self.script_label.hide()
            self.dialogue_widget.show()

    def build_args(self):
        args = ["tts"]
        is_dialogue = self.mode_dialogue.isChecked()

        if self.ocr_edit.text().strip():
            args.extend(["ocr", self.ocr_edit.text().strip()])

        if is_dialogue:
            items = self.dialogue_widget.get_dialogue_items()
            if not items and not self.ocr_edit.text().strip():
                return None
            for idx, char, text in items:
                args.extend(["script", f"{char}: {text}"])
        else:
            script_text = self.script_text.toPlainText().strip()
            if not script_text and not self.ocr_edit.text().strip():
                return None
            if script_text:
                args.extend(["script", script_text])

        for char, desc in self.voice_list.get_items():
            args.extend(["voice", f"{char}: {desc}"])

        for char, path in self.clone_list.get_items():
            args.extend(["target", f"{char}: {path}"])

        if self.music_edit.text().strip():
            args.extend(["music", self.music_edit.text().strip()])
        if self.level_edit.text().strip():
            args.extend(["level", self.level_edit.text().strip()])
        if self.reference_edit.text().strip():
            args.extend(["reference", self.reference_edit.text().strip()])

        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])

        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class STTTTSTab(QWidget):
    transcribe_signal = pyqtSignal(list)
    synthesize_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        HelperWidgets.make_label(inner, "Step 1: Transcribe Audio")
        self.base_edit = HelperWidgets.make_file_picker(inner, "Base Audio", "Source audio or video file to transcribe")

        self.transcribe_btn = QPushButton("Transcribe")
        self.transcribe_btn.setStyleSheet(get_main_button_style())
        self.transcribe_btn.setCursor(Qt.PointingHandCursor)
        self.transcribe_btn.clicked.connect(self.on_transcribe)
        inner.addWidget(self.transcribe_btn)

        HelperWidgets.make_separator(inner)
        HelperWidgets.make_label(inner, "Step 2: Review & Edit Transcription")

        self.script_text = QTextEdit()
        self.script_text.setPlaceholderText("Transcribed text will appear here. Edit as needed before synthesizing...")
        self.script_text.setStyleSheet(get_text_edit_style())
        self.script_text.setMinimumHeight(120)
        self.script_text.setMaximumHeight(240)
        inner.addWidget(self.script_text)

        HelperWidgets.make_separator(inner)
        HelperWidgets.make_label(inner, "Step 3: Synthesize with Target Voice")

        self.target_edit = HelperWidgets.make_file_picker(inner, "Target Voice", "Reference voice audio for synthesis")
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.synthesize_btn = QPushButton("Synthesize")
        self.synthesize_btn.setStyleSheet(get_main_button_style())
        self.synthesize_btn.setCursor(Qt.PointingHandCursor)
        self.synthesize_btn.clicked.connect(self.on_synthesize)
        inner.addWidget(self.synthesize_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def on_transcribe(self):
        base_path = self.base_edit.text().strip()
        if not base_path:
            return
        args = ["stt", base_path]
        self.transcribe_signal.emit(args)

    def on_synthesize(self):
        script = self.script_text.toPlainText().strip()
        target = self.target_edit.text().strip()
        if not script or not target:
            return
        args = ["tts", "script", script, "target", target]
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        self.synthesize_signal.emit(args)

    def set_transcription(self, text):
        self.script_text.setText(text)

    def build_transcribe_args(self):
        return self.on_transcribe()

    def build_synthesize_args(self):
        return self.on_synthesize()


class STSTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        self.base_edit = HelperWidgets.make_file_picker(inner, "Base Audio", "Source audio or video (or URL)")
        self.target_edit = HelperWidgets.make_file_picker(inner, "Target Voice", "Reference voice audio")
        self.music_cb = HelperWidgets.make_checkbox(inner, "Music mode (Seed-VC v1, 44.1kHz)")
        self.mimic_cb = HelperWidgets.make_checkbox(inner, "Mimic mode (style + voice)")
        self.music_cb.toggled.connect(lambda checked: self.mimic_cb.setChecked(False) if checked else None)
        self.mimic_cb.toggled.connect(lambda checked: self.music_cb.setChecked(False) if checked else None)
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run STS")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        if not self.base_edit.text().strip() or not self.target_edit.text().strip():
            return None
        args = ["sts", "base", self.base_edit.text().strip(), "target", self.target_edit.text().strip()]
        if self.music_cb.isChecked():
            args.append("music")
        if self.mimic_cb.isChecked():
            args.append("mimic")
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class StemSelectorWidget(QWidget):
    stems_changed = pyqtSignal()

    AVAILABLE_STEMS = [
        ("vocals", "Vocals"),
        ("backing_vocals", "Backing Vocals"),
        ("drums", "Drums"),
        ("bass", "Bass"),
        ("guitar", "Guitar"),
        ("keyboard", "Keyboard"),
        ("strings", "Strings"),
        ("percussion", "Percussion"),
        ("brass", "Brass"),
        ("woodwinds", "Woodwinds"),
        ("synth", "Synth"),
        ("fx", "FX / Sound Effects"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.checkboxes = {}
        self.custom_edit = None
        self._expanded = False
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.toggle_btn = QPushButton("Select Stems")
        self.toggle_btn.setStyleSheet(get_surface_button_style())
        self.toggle_btn.setCursor(Qt.PointingHandCursor)
        self.toggle_btn.clicked.connect(self.toggle_panel)
        layout.addWidget(self.toggle_btn)

        self.panel = QFrame()
        self.panel.setStyleSheet(f"background-color: {THEME['surface']}; border: 1px solid {THEME['border']}; border-radius: 6px; padding: 8px;")
        self.panel_layout = QVBoxLayout(self.panel)
        self.panel_layout.setContentsMargins(8, 8, 8, 8)
        self.panel_layout.setSpacing(4)

        for stem_key, stem_label in self.AVAILABLE_STEMS:
            cb = QCheckBox(stem_label)
            cb.setStyleSheet(get_checkbox_style())
            cb.stateChanged.connect(self.on_changed)
            self.checkboxes[stem_key] = cb
            self.panel_layout.addWidget(cb)

        quick_row = QHBoxLayout()
        quick_row.setSpacing(4)
        for label_text, stems_list in [
            ("All", [s[0] for s in self.AVAILABLE_STEMS]),
            ("Inst", ["drums", "bass", "guitar", "keyboard", "strings", "percussion", "brass", "woodwinds", "synth", "fx"]),
            ("Voice", ["vocals", "backing_vocals"]),
            ("None", []),
        ]:
            btn = QPushButton(label_text)
            btn.setFixedHeight(28)
            btn.setStyleSheet(get_surface_button_style())
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, sl=stems_list: self.set_stems(sl))
            quick_row.addWidget(btn)
        quick_row.addStretch()
        self.panel_layout.addLayout(quick_row)

        custom_row = QHBoxLayout()
        custom_row.setSpacing(4)
        custom_label = QLabel("Custom:")
        custom_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px;")
        self.custom_edit = QLineEdit()
        self.custom_edit.setPlaceholderText("e.g. everything, instruments, voices")
        self.custom_edit.setStyleSheet(get_line_edit_style())
        self.custom_edit.textChanged.connect(self.on_changed)
        custom_row.addWidget(custom_label)
        custom_row.addWidget(self.custom_edit, stretch=1)
        self.panel_layout.addLayout(custom_row)

        self.panel.hide()
        layout.addWidget(self.panel)

    def toggle_panel(self):
        self._expanded = not self._expanded
        self.panel.setVisible(self._expanded)
        self.update_label()

    def set_stems(self, stems_list):
        for key, cb in self.checkboxes.items():
            cb.setChecked(key in stems_list)
        self.on_changed()

    def on_changed(self):
        self.update_label()
        self.stems_changed.emit()

    def update_label(self):
        selected = self.get_stems_list()
        custom = self.custom_edit.text().strip() if self.custom_edit else ""
        if custom:
            if selected:
                self.toggle_btn.setText(f"Select Stems ({len(selected)} + custom)")
            else:
                self.toggle_btn.setText(f"Select Stems (custom)")
        elif selected:
            display = ", ".join(selected[:3])
            if len(selected) > 3:
                display += f" +{len(selected)-3}"
            self.toggle_btn.setText(f"Select Stems: {display}")
        else:
            self.toggle_btn.setText("Select Stems")

    def get_stems_list(self):
        selected = [key for key, cb in self.checkboxes.items() if cb.isChecked()]
        return selected

    def get_stems_string(self):
        selected = self.get_stems_list()
        custom = self.custom_edit.text().strip() if self.custom_edit else ""
        if custom and selected:
            return " ".join(selected) + " " + custom
        elif custom:
            return custom
        elif selected:
            return " ".join(selected)
        return ""


class TTMTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        self.inner = QVBoxLayout(scroll_widget)
        self.inner.setSpacing(8)

        self.sub_mode = QComboBox()
        self.sub_mode.setStyleSheet(get_combo_box_style())
        self.sub_mode.addItems(["Generate", "Voice Clone (VC)", "Remix", "Repaint", "Complete", "Lego", "Extract", "BGM"])
        self.sub_mode.currentTextChanged.connect(self.on_submode_changed)
        mode_row = QHBoxLayout()
        mode_label = QLabel("Sub-Mode")
        mode_label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.sub_mode, stretch=1)
        mode_row.addStretch()
        self.inner.addLayout(mode_row)

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(8)
        self.inner.addWidget(self.container)

        self.overdose_cb = HelperWidgets.make_checkbox(self.inner, "Overdose tier (ACE-Step XL-Turbo)")
        self.result_edit = HelperWidgets.make_save_picker(self.inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run TTM")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        self.inner.addWidget(self.run_btn)
        self.inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.on_submode_changed("Generate")

    def clear_container(self):
        while self.container_layout.count():
            item = self.container_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    sw = sub.widget()
                    if sw:
                        sw.deleteLater()

    def on_submode_changed(self, mode):
        self.clear_container()
        getattr(self, f'build_{mode.lower().replace(" ", "_").replace("(vc)", "vc")}_ui', self.build_generate_ui)()

    def build_generate_ui(self):
        self.gen_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics", "Song lyrics (use \\n for line breaks)", 100)
        self.gen_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "Style/mood prompt", 80)
        self.gen_duration = HelperWidgets.make_spinbox(self.container_layout, "Duration (s)", 10, 300, 30)
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.gen_target_type = QComboBox()
        self.gen_target_type.setStyleSheet(get_combo_box_style())
        self.gen_target_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Target Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.gen_target_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.gen_target_path = HelperWidgets.make_file_picker(self.container_layout, "Target Path", "Reference audio path")

    def build_voice_clone_vc_ui(self):
        self.vc_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics", "Song lyrics (use \\n for line breaks)", 100)
        self.vc_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "Style/mood prompt", 80)
        self.vc_duration = HelperWidgets.make_spinbox(self.container_layout, "Duration (s)", 10, 300, 30)
        self.vc_clone = HelperWidgets.make_file_picker(self.container_layout, "Clone Voice", "Source voice audio for cloning")
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.vc_target_type = QComboBox()
        self.vc_target_type.setStyleSheet(get_combo_box_style())
        self.vc_target_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Target Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.vc_target_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.vc_target_path = HelperWidgets.make_file_picker(self.container_layout, "Target Path", "Reference audio path")

    def build_remix_ui(self):
        self.remix_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL to remix")
        self.remix_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "New style for the remix", 80)
        self.remix_bias = HelperWidgets.make_spinbox(self.container_layout, "Bias (0-100)", 0, 100, 40)
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.remix_ref_type = QComboBox()
        self.remix_ref_type.setStyleSheet(get_combo_box_style())
        self.remix_ref_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Ref Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.remix_ref_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.remix_ref_path = HelperWidgets.make_file_picker(self.container_layout, "Reference Path", "Reference audio or URL")

    def build_repaint_ui(self):
        self.repaint_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL to repaint")
        self.repaint_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "New style for the repainted section", 80)
        self.repaint_time = HelperWidgets.make_line_edit(self.container_layout, "Time Range", "e.g. 20-80 or 20.5-80.5")
        self.repaint_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics (opt)", "Optional lyrics for repainted section", 80)
        self.repaint_bias = HelperWidgets.make_spinbox(self.container_layout, "Bias (0-100)", 0, 100, 40)
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.repaint_ref_type = QComboBox()
        self.repaint_ref_type.setStyleSheet(get_combo_box_style())
        self.repaint_ref_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Ref Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.repaint_ref_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.repaint_ref_path = HelperWidgets.make_file_picker(self.container_layout, "Reference Path", "Reference audio or URL")

    def build_complete_ui(self):
        self.complete_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL")
        HelperWidgets.make_label(self.container_layout, "Instruments to Add")
        self.complete_stems_selector = StemSelectorWidget()
        self.container_layout.addWidget(self.complete_stems_selector)
        self.complete_add = HelperWidgets.make_line_edit(self.container_layout, "Custom Stems", "e.g. everything, or additional stems not listed above")
        self.complete_voice_cb = HelperWidgets.make_checkbox(self.container_layout, "Pre-extract vocals (voice)")
        self.complete_music_cb = HelperWidgets.make_checkbox(self.container_layout, "Pre-extract instruments (music)")
        self.complete_voice_cb.toggled.connect(lambda checked: self.complete_music_cb.setChecked(False) if checked else None)
        self.complete_music_cb.toggled.connect(lambda checked: self.complete_voice_cb.setChecked(False) if checked else None)
        self.complete_video_cb = HelperWidgets.make_checkbox(self.container_layout, "Preserve video")
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.complete_ref_type = QComboBox()
        self.complete_ref_type.setStyleSheet(get_combo_box_style())
        self.complete_ref_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Ref Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.complete_ref_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.complete_ref_path = HelperWidgets.make_file_picker(self.container_layout, "Reference Path", "Reference audio or URL")

    def build_lego_ui(self):
        self.lego_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL")
        HelperWidgets.make_label(self.container_layout, "Instruments to Make")
        self.lego_stems_selector = StemSelectorWidget()
        self.container_layout.addWidget(self.lego_stems_selector)
        self.lego_make = HelperWidgets.make_line_edit(self.container_layout, "Custom Stems", "e.g. everything, or additional stems not listed above")
        self.lego_voice_cb = HelperWidgets.make_checkbox(self.container_layout, "Pre-extract vocals (voice)")
        self.lego_music_cb = HelperWidgets.make_checkbox(self.container_layout, "Pre-extract instruments (music)")
        self.lego_voice_cb.toggled.connect(lambda checked: self.lego_music_cb.setChecked(False) if checked else None)
        self.lego_music_cb.toggled.connect(lambda checked: self.lego_voice_cb.setChecked(False) if checked else None)
        self.lego_mix_cb = HelperWidgets.make_checkbox(self.container_layout, "Mix all tracks into one file")
        self.lego_blend_cb = HelperWidgets.make_checkbox(self.container_layout, "Mix and blend with original")
        self.lego_mix_cb.toggled.connect(lambda checked: self.lego_blend_cb.setChecked(False) if checked else None)
        self.lego_blend_cb.toggled.connect(lambda checked: self.lego_mix_cb.setChecked(False) if checked else None)
        ref_row = QHBoxLayout()
        ref_label = QLabel("References (optional, multiple)")
        ref_label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px;")
        ref_row.addWidget(ref_label)
        ref_row.addStretch()
        lego_add_btn = QPushButton("+Add")
        lego_add_btn.setStyleSheet(get_surface_button_style())
        lego_add_btn.setCursor(Qt.PointingHandCursor)
        lego_add_btn.clicked.connect(lambda: self.lego_ref_list.add_row())
        ref_row.addWidget(lego_add_btn)
        self.container_layout.addLayout(ref_row)
        self.lego_ref_list = KeyValueList("Stem or fallback", "Audio Path", "e.g. drums or leave empty", "path/to/ref.wav", with_browse=True)
        self.lego_ref_list.setMinimumHeight(120)
        self.lego_ref_list.add_btn.hide()
        self.container_layout.addWidget(self.lego_ref_list)

    def build_extract_ui(self):
        self.extract_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL")
        HelperWidgets.make_label(self.container_layout, "Stems")
        self.extract_stems_selector = StemSelectorWidget()
        self.container_layout.addWidget(self.extract_stems_selector)
        self.extract_only_cb = HelperWidgets.make_checkbox(self.container_layout, "Only (extract everything EXCEPT specified)")
        self.extract_mix_cb = HelperWidgets.make_checkbox(self.container_layout, "Mix all extracted stems into one file")
        self.extract_only_cb.toggled.connect(lambda checked: self.extract_mix_cb.setChecked(False) if checked else None)
        self.extract_mix_cb.toggled.connect(lambda checked: self.extract_only_cb.setChecked(False) if checked else None)

    def build_bgm_ui(self):
        self.bgm_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Audio/video file or URL to add background music on")
        self.bgm_music = HelperWidgets.make_text_edit(self.container_layout, "Music Desc", "Background music style/description", 80)
        self.bgm_level = HelperWidgets.make_spinbox(self.container_layout, "Music Level", 0, 100, 35)
        self.bgm_reference = HelperWidgets.make_file_picker(self.container_layout, "Reference (opt)", "Reference audio/video or URL for music style")

    def _add_target_ref(self, args, target_type, target_path):
        if target_type == "As-is (full audio)" and target_path.strip():
            args.extend(["target", target_path.strip()])
        elif target_type == "Extract vocals" and target_path.strip():
            args.extend(["target voice", target_path.strip()])
        elif target_type == "Extract instruments" and target_path.strip():
            args.extend(["target music", target_path.strip()])

    def _add_ref(self, args, ref_type, ref_path):
        if ref_type == "As-is (full audio)" and ref_path.strip():
            args.extend(["reference", ref_path.strip()])
        elif ref_type == "Extract vocals" and ref_path.strip():
            args.extend(["reference voice", ref_path.strip()])
        elif ref_type == "Extract instruments" and ref_path.strip():
            args.extend(["reference music", ref_path.strip()])

    def build_args(self):
        mode = self.sub_mode.currentText().lower()
        args = ["ttm"]
        if self.overdose_cb.isChecked():
            args.append("overdose")

        if mode == "generate":
            lyrics = self.gen_lyrics.toPlainText().strip()
            styling = self.gen_styling.toPlainText().strip()
            if not lyrics or not styling:
                return None
            args.extend(["lyrics", lyrics, "styling", styling])
            args.append(str(self.gen_duration.value()))
            self._add_target_ref(args, self.gen_target_type.currentText(), self.gen_target_path.text())

        elif mode == "voice clone (vc)":
            lyrics = self.vc_lyrics.toPlainText().strip()
            styling = self.vc_styling.toPlainText().strip()
            clone = self.vc_clone.text().strip()
            if not lyrics or not styling or not clone:
                return None
            args.append("vc")
            args.extend(["lyrics", lyrics, "styling", styling])
            args.append(str(self.vc_duration.value()))
            args.extend(["clone", clone])
            self._add_target_ref(args, self.vc_target_type.currentText(), self.vc_target_path.text())

        elif mode == "remix":
            source = self.remix_source.text().strip()
            styling = self.remix_styling.toPlainText().strip()
            if not source or not styling:
                return None
            args.extend(["remix", source, "styling", styling, "bias", str(self.remix_bias.value())])
            self._add_ref(args, self.remix_ref_type.currentText(), self.remix_ref_path.text())

        elif mode == "repaint":
            source = self.repaint_source.text().strip()
            styling = self.repaint_styling.toPlainText().strip()
            time_range = self.repaint_time.text().strip()
            if not source or not styling or not time_range:
                return None
            args.extend(["repaint", source, "styling", styling, f"time:{time_range}", "bias", str(self.repaint_bias.value())])
            lyrics = self.repaint_lyrics.toPlainText().strip()
            if lyrics:
                args.extend(["lyrics", lyrics])
            self._add_ref(args, self.repaint_ref_type.currentText(), self.repaint_ref_path.text())

        elif mode == "complete":
            source = self.complete_source.text().strip()
            selector_stems = self.complete_stems_selector.get_stems_string()
            custom_stems = self.complete_add.text().strip()
            combined = " ".join(filter(None, [selector_stems, custom_stems])).strip()
            if not source or not combined:
                return None
            args.append("complete")
            if self.complete_voice_cb.isChecked():
                args.append("voice")
            if self.complete_music_cb.isChecked():
                args.append("music")
            if self.complete_video_cb.isChecked():
                args.append("video")
            args.append(source)
            args.extend(["add", combined])
            self._add_ref(args, self.complete_ref_type.currentText(), self.complete_ref_path.text())

        elif mode == "lego":
            source = self.lego_source.text().strip()
            selector_stems = self.lego_stems_selector.get_stems_string()
            custom_stems = self.lego_make.text().strip()
            combined = " ".join(filter(None, [selector_stems, custom_stems])).strip()
            if not source or not combined:
                return None
            args.append("lego")
            if self.lego_mix_cb.isChecked():
                args.append("mix")
            if self.lego_blend_cb.isChecked():
                args.append("blend")
            if self.lego_voice_cb.isChecked():
                args.append("voice")
            if self.lego_music_cb.isChecked():
                args.append("music")
            args.append(source)
            args.extend(["make", combined])
            for stem, path in self.lego_ref_list.get_items():
                if stem.strip():
                    args.extend(["reference", f"{stem.strip()}:{path.strip()}"])
                else:
                    args.extend(["reference", path.strip()])

        elif mode == "extract":
            source = self.extract_source.text().strip()
            stems = self.extract_stems_selector.get_stems_string()
            if not source or not stems:
                return None
            args.append("extract")
            if self.extract_only_cb.isChecked():
                args.append("only")
            if self.extract_mix_cb.isChecked():
                args.append("mix")
            args.append(source)
            args.extend(["stems", stems])

        elif mode == "bgm":
            source = self.bgm_source.text().strip()
            music_desc = self.bgm_music.toPlainText().strip()
            if not source or not music_desc:
                return None
            args.extend(["bgm", source, "music", music_desc])
            args.extend(["level", str(self.bgm_level.value())])
            ref = self.bgm_reference.text().strip()
            if ref:
                args.extend(["reference", ref])

        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class STTTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        HelperWidgets.make_label(inner, "Input Files (audio, video, image, or URLs)")
        self.file_list = FileListWidget(inner, file_filter="Audio (*.wav *.mp3 *.flac *.ogg);;Video (*.mp4 *.avi *.mov *.mkv);;Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)")
        inner.addWidget(self.file_list)

        self.timestamp_cb = HelperWidgets.make_checkbox(inner, "Timestamp (keep word-level timestamps)")
        self.dialogue_cb = HelperWidgets.make_checkbox(inner, "Dialogue (speaker diarization, requires HF_TOKEN)")
        self.translate_cb = HelperWidgets.make_checkbox(inner, "Translate to English")
        self.se_cb = HelperWidgets.make_checkbox(inner, "Speech Enhancement (denoise/dereverb before transcription)")
        self.overdose_cb = HelperWidgets.make_checkbox(inner, "Overdose (VibeVoice ASR, requires 24GB+ VRAM or 48GB+ RAM)")
        self.translate_cb.toggled.connect(lambda checked: self.overdose_cb.setChecked(False) if checked else None)
        self.overdose_cb.toggled.connect(lambda checked: self.translate_cb.setChecked(False) if checked else None)
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run STT")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        paths = self.file_list.get_paths()
        if not paths:
            return None
        args = ["stt"]
        args.extend(paths)
        if self.timestamp_cb.isChecked():
            args.append("timestamp")
        if self.dialogue_cb.isChecked():
            args.append("dialogue")
        if self.translate_cb.isChecked():
            args.append("translate")
        if self.se_cb.isChecked():
            args.append("se")
        if self.overdose_cb.isChecked():
            args.append("overdose")
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class SETab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        HelperWidgets.make_label(inner, "Input Files (audio, video, or URLs)")
        self.file_list = FileListWidget(inner, file_filter="Audio (*.wav *.mp3 *.flac *.ogg);;Video (*.mp4 *.avi *.mov *.mkv);;All Files (*)")
        inner.addWidget(self.file_list)
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run SE")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        paths = self.file_list.get_paths()
        if not paths:
            return None
        args = ["se"]
        args.extend(paths)
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class SFXTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        self.sound_edit = HelperWidgets.make_line_edit(inner, "Sound Prompt", "e.g. thunder cracking, rain on a tin roof")
        self.duration_spin = HelperWidgets.make_spinbox(inner, "Duration (s)", 1, 30, 5)
        self.steps_spin = HelperWidgets.make_spinbox(inner, "Steps", 1, 100, 30)
        self.guide_spin = HelperWidgets.make_double_spinbox(inner, "Guidance", 1.0, 10.0, 4.5, 0.5)
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run SFX")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        sound = self.sound_edit.text().strip()
        if not sound:
            return None
        args = ["sfx", "sound", sound, "duration", str(self.duration_spin.value())]
        if self.steps_spin.value() != 30:
            args.extend(["steps", str(self.steps_spin.value())])
        if self.guide_spin.value() != 4.5:
            args.extend(["guide", str(self.guide_spin.value())])
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class SVSTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        self.file_edit = HelperWidgets.make_file_picker(inner, "Input File", "Audio or video file (or URL)")
        self.voice_cb = HelperWidgets.make_checkbox(inner, "Extract vocals (remove instruments)")
        self.music_cb = HelperWidgets.make_checkbox(inner, "Extract instruments (remove vocals)")
        self.both_cb = HelperWidgets.make_checkbox(inner, "Extract both vocals and instruments")
        self.voice_cb.toggled.connect(lambda checked: (self.music_cb.setChecked(False), self.both_cb.setChecked(False)) if checked else None)
        self.music_cb.toggled.connect(lambda checked: (self.voice_cb.setChecked(False), self.both_cb.setChecked(False)) if checked else None)
        self.both_cb.toggled.connect(lambda checked: (self.voice_cb.setChecked(False), self.music_cb.setChecked(False)) if checked else None)
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run SVS")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        if not self.file_edit.text().strip():
            return None
        if not self.voice_cb.isChecked() and not self.music_cb.isChecked() and not self.both_cb.isChecked():
            return None
        args = ["svs"]
        if self.voice_cb.isChecked():
            args.append("voice")
        if self.music_cb.isChecked():
            args.append("music")
        if self.both_cb.isChecked():
            args.append("both")
        args.append(self.file_edit.text().strip())
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class SLCTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        self.input_edit = HelperWidgets.make_file_picker(inner, "Input Audio", "Audio file or URL (audio only, no video)")
        self.translate_cb = HelperWidgets.make_checkbox(inner, "Translate to English")
        self.target_edit = HelperWidgets.make_file_picker(inner, "Target Voice", "Optional: target voice reference audio")
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run SLC")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        if not self.input_edit.text().strip():
            return None
        args = ["slc"]
        if self.translate_cb.isChecked():
            args.append("translate")
        if self.target_edit.text().strip():
            args.extend(["target", self.target_edit.text().strip()])
        args.append(self.input_edit.text().strip())
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class SSTab(QWidget):
    run_signal = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet("background: transparent;")
        inner = QVBoxLayout(scroll_widget)
        inner.setSpacing(8)

        self.source_edit = HelperWidgets.make_file_picker(inner, "Source", "Audio, video, or URL")
        self.target_edit = HelperWidgets.make_file_picker(inner, "Target Voice", "Optional: extract specific speaker matching this voice")
        self.se_cb = HelperWidgets.make_checkbox(inner, "Speech Enhancement (denoise/dereverb before separation)")
        self.overdose_cb = HelperWidgets.make_checkbox(inner, "Overdose (VibeVoice ASR for better accuracy)")
        self.result_edit = HelperWidgets.make_save_picker(inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run SS")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        if not self.source_edit.text().strip():
            return None
        args = ["ss"]
        if self.target_edit.text().strip():
            args.extend(["target", self.target_edit.text().strip()])
        if self.se_cb.isChecked():
            args.append("se")
        if self.overdose_cb.isChecked():
            args.append("overdose")
        args.append(self.source_edit.text().strip())
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class VoderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VODER - Voice Design & Audio Processing")
        self.setMinimumSize(900, 700)
        self.resize(1100, 800)
        self.setStyleSheet(get_window_style())
        self.running = False
        self.thread = None
        self.audio_thread = None
        self.start_time = None
        self.stt_tts_tab = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        central.setStyleSheet("background: transparent;")
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        title = QLabel("VODER")
        title.setStyleSheet("color: #E5E5E5; font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        self.tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 2px solid {THEME['border']};
                border-radius: 8px;
                background-color: {THEME['background']};
                padding: 8px;
            }}
            QTabBar::tab {{
                background-color: {THEME['surface']};
                color: {THEME['text']};
                border: 2px solid {THEME['border']};
                border-bottom: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                padding: 8px 16px;
                margin-right: 4px;
                font-size: 13px;
                font-weight: bold;
                min-width: 60px;
            }}
            QTabBar::tab:selected {{
                background-color: {THEME['panel_background']};
                border: 2px solid {THEME['panel_border']};
                border-bottom: none;
            }}
            QTabBar::tab:hover {{
                background-color: {THEME['surface_hover']};
            }}
        """)

        self.tts_tab = TTSTab()
        self.stt_tts_tab = STTTTSTab()
        self.sts_tab = STSTab()
        self.ttm_tab = TTMTab()
        self.stt_tab = STTTab()
        self.se_tab = SETab()
        self.sfx_tab = SFXTab()
        self.svs_tab = SVSTab()
        self.slc_tab = SLCTab()
        self.ss_tab = SSTab()

        for tab, name in [
            (self.tts_tab, "TTS"), (self.stt_tts_tab, "STT\u2192TTS"),
            (self.sts_tab, "STS"), (self.ttm_tab, "TTM"),
            (self.stt_tab, "STT"), (self.se_tab, "SE"), (self.sfx_tab, "SFX"),
            (self.svs_tab, "SVS"), (self.slc_tab, "SLC"), (self.ss_tab, "SS")
        ]:
            self.tabs.addTab(tab, name)
            if hasattr(tab, 'run_signal'):
                tab.run_signal.connect(self.run_command)

        self.stt_tts_tab.transcribe_signal.connect(self.run_stt_tts_transcribe)
        self.stt_tts_tab.synthesize_signal.connect(self.run_command)

        main_layout.addWidget(self.tabs, stretch=1)

        console_frame = QFrame()
        console_frame.setStyleSheet(get_panel_style())
        console_layout = QVBoxLayout(console_frame)
        console_layout.setContentsMargins(8, 8, 8, 8)
        console_layout.setSpacing(4)
        console_label = QLabel("Console Output")
        console_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px; font-weight: bold;")
        console_layout.addWidget(console_label)
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(get_text_edit_style())
        self.console.setMaximumHeight(180)
        self.console.setMinimumHeight(80)
        console_layout.addWidget(self.console)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        self.progress_bar.setMaximumHeight(8)
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()
        console_layout.addWidget(self.progress_bar)

        audio_row = QHBoxLayout()
        audio_label = QLabel("Output Audio")
        audio_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 12px; min-width: 80px;")
        audio_row.addWidget(audio_label)
        self.output_audio_edit = QLineEdit()
        self.output_audio_edit.setPlaceholderText("Audio path auto-detected from output")
        self.output_audio_edit.setStyleSheet(get_line_edit_style())
        self.output_audio_edit.setReadOnly(True)
        audio_row.addWidget(self.output_audio_edit, stretch=1)
        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet(get_surface_button_style())
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setFixedWidth(60)
        self.play_btn.clicked.connect(self.play_audio)
        self.play_btn.setEnabled(False)
        audio_row.addWidget(self.play_btn)

        self.waveform_widget = AudioWaveformWidget()
        audio_row.addWidget(self.waveform_widget, stretch=1)
        console_layout.addLayout(audio_row)

        console_btn_row = QHBoxLayout()
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(get_surface_button_style())
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.setFixedWidth(80)
        self.clear_btn.clicked.connect(self.console.clear)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(get_surface_button_style())
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setFixedWidth(80)
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)
        self.copy_cmd_btn = QPushButton("Copy Command")
        self.copy_cmd_btn.setStyleSheet(get_surface_button_style())
        self.copy_cmd_btn.setCursor(Qt.PointingHandCursor)
        self.copy_cmd_btn.setMinimumWidth(120)
        self.copy_cmd_btn.clicked.connect(self.copy_command)
        console_btn_row.addWidget(self.clear_btn)
        console_btn_row.addWidget(self.stop_btn)
        console_btn_row.addWidget(self.copy_cmd_btn)
        console_btn_row.addStretch()
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(get_status_bar_style())
        console_btn_row.addWidget(self.status_label)
        console_layout.addLayout(console_btn_row)
        main_layout.addWidget(console_frame)

        self.statusBar().showMessage("Ready")

        run_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        run_shortcut.activated.connect(self.run_current_tab)

        self.ready_indicator = QLabel("")
        self.ready_indicator.setAlignment(Qt.AlignCenter)
        self.ready_indicator.setStyleSheet("color: #4CAF50; font-size: 13px; font-weight: bold;")
        console_btn_row.addWidget(self.ready_indicator)

        self._ready_timer = QTimer()
        self._ready_timer.timeout.connect(self.check_readiness)
        self._ready_timer.start(500)

    def check_readiness(self):
        if self.running:
            self.ready_indicator.setText("")
            return
        current = self.tabs.currentWidget()
        if current and hasattr(current, 'build_args'):
            try:
                args = current.build_args()
                if args is not None:
                    self.ready_indicator.setText("[ READY ]")
                    self.ready_indicator.setStyleSheet("color: #4CAF50; font-size: 12px; font-weight: bold;")
                else:
                    self.ready_indicator.setText("")
            except Exception:
                self.ready_indicator.setText("")
        else:
            self.ready_indicator.setText("")

    def run_current_tab(self):
        current = self.tabs.currentWidget()
        if current and hasattr(current, 'run_signal'):
            if isinstance(current, STTTTSTab):
                return
            btn = getattr(current, 'run_btn', None)
            if btn and btn.isEnabled():
                btn.click()

    def run_stt_tts_transcribe(self, args):
        if self.running:
            self.console.append("[!] A process is already running. Stop it first.")
            return
        self.running = True
        self.stop_btn.setEnabled(True)
        self.last_args = args
        cmd_display = "python voder.py " + " ".join(_q(a) for a in args)
        self.console.append(f"$ {cmd_display}")
        self.console.append("")
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet(f"color: {THEME['warning']}; padding: 6px 12px; font-size: 12px;")
        self.start_time = time.time()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.show()
        self.thread = SubprocessThread(args)
        self.thread.output_signal.connect(self.on_stt_tts_transcribe_output)
        self.thread.finished_signal.connect(self.on_stt_tts_transcribe_finished)
        self.thread.start()

    def on_stt_tts_transcribe_output(self, line):
        self.console.append(line)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_stt_tts_transcribe_finished(self, returncode, output):
        self.running = False
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.start_time = None
        if returncode == 0:
            self.status_label.setText(f"Done ({elapsed:.1f}s)")
            self.status_label.setStyleSheet(f"color: {THEME['success']}; padding: 6px 12px; font-size: 12px;")
            self.console.append("")
            self.console.append("[OK] Process completed successfully.")
            self.stt_tts_tab.set_transcription(output.strip())
            detected_path = self._detect_audio_path(output)
            if detected_path:
                self.output_audio_edit.setText(detected_path)
                self.play_btn.setEnabled(True)
                self.waveform_widget.set_audio(detected_path)
        else:
            self.status_label.setText(f"Failed (code {returncode}) ({elapsed:.1f}s)")
            self.status_label.setStyleSheet(f"color: {THEME['error']}; padding: 6px 12px; font-size: 12px;")
            self.console.append("")
            self.console.append(f"[ERR] Process exited with code {returncode}.")
        self.statusBar().showMessage(self.status_label.text())

    def run_command(self, args):
        if self.running:
            self.console.append("[!] A process is already running. Stop it first.")
            return
        self.running = True
        self.stop_btn.setEnabled(True)
        self.last_args = args
        cmd_display = "python voder.py " + " ".join(_q(a) for a in args)
        self.console.append(f"$ {cmd_display}")
        self.console.append("")
        self.status_label.setText("Running...")
        self.status_label.setStyleSheet(f"color: {THEME['warning']}; padding: 6px 12px; font-size: 12px;")
        self.start_time = time.time()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.progress_bar.show()
        self.thread = SubprocessThread(args)
        self.thread.output_signal.connect(self.on_output)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def on_output(self, line):
        self.console.append(line)
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def on_finished(self, returncode, output):
        self.running = False
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        elapsed = time.time() - self.start_time if self.start_time else 0
        self.start_time = None
        if returncode == 0:
            self.status_label.setText(f"Done ({elapsed:.1f}s)")
            self.status_label.setStyleSheet(f"color: {THEME['success']}; padding: 6px 12px; font-size: 12px;")
            self.console.append("")
            self.console.append("[OK] Process completed successfully.")
        else:
            self.status_label.setText(f"Failed (code {returncode}) ({elapsed:.1f}s)")
            self.status_label.setStyleSheet(f"color: {THEME['error']}; padding: 6px 12px; font-size: 12px;")
            self.console.append("")
            self.console.append(f"[ERR] Process exited with code {returncode}.")
        detected_path = self._detect_audio_path(output)
        if detected_path:
            self.output_audio_edit.setText(detected_path)
            self.play_btn.setEnabled(True)
            self.waveform_widget.set_audio(detected_path)
        self.statusBar().showMessage(self.status_label.text())

    def _detect_audio_path(self, output):
        if not output:
            return None
        extensions = r'\.(wav|mp3|flac|ogg|m4a|aac|wma)(?:"|\s|$|>)'
        pattern = r'(/[^\s<>"\'`]+?' + extensions + r')'
        matches = re.findall(pattern, output)
        if matches:
            last = matches[-1]
            if isinstance(last, tuple):
                return last[0]
            return last
        pattern2 = r'([A-Za-z]:\\[^\s<>"\'`]+?' + extensions + r')'
        matches2 = re.findall(pattern2, output)
        if matches2:
            last = matches2[-1]
            if isinstance(last, tuple):
                return last[0]
            return last
        return None

    def play_audio(self):
        path = self.output_audio_edit.text().strip()
        if not path:
            return
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()
            self.audio_thread.wait(1000)
        self.audio_thread = AudioPlayerThread(path)
        self.audio_thread.finished_signal.connect(self.on_playback_finished)
        self.audio_thread.start()
        self.play_btn.setText("Stop")
        self.play_btn.clicked.disconnect(self.play_audio)
        self.play_btn.clicked.connect(self.stop_audio)

    def stop_audio(self):
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.stop()

    def on_playback_finished(self):
        self.play_btn.setText("Play")
        self.play_btn.clicked.disconnect(self.stop_audio)
        self.play_btn.clicked.connect(self.play_audio)

    def stop_process(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.console.append("[!] Stopping process...")
            self.status_label.setText("Stopped")
            self.status_label.setStyleSheet(f"color: {THEME['warning']}; padding: 6px 12px; font-size: 12px;")
            self.progress_bar.hide()

    def copy_command(self):
        current = self.tabs.currentWidget()
        if current and hasattr(current, 'build_args'):
            try:
                args = current.build_args()
                if args:
                    cmd = "python voder.py " + " ".join(_q(a) for a in args)
                    QApplication.clipboard().setText(cmd)
                    self.console.append(f"[i] Copied: {cmd}")
                    return
            except Exception:
                pass
        if hasattr(self, 'last_args') and self.last_args:
            cmd = "python voder.py " + " ".join(_q(a) for a in self.last_args)
            QApplication.clipboard().setText(cmd)
            self.console.append(f"[i] Copied last: {cmd}")


def launch():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(THEME['background']))
    palette.setColor(QPalette.WindowText, QColor(THEME['text']))
    palette.setColor(QPalette.Base, QColor(THEME['surface']))
    palette.setColor(QPalette.AlternateBase, QColor(THEME['surface_hover']))
    palette.setColor(QPalette.ToolTipBase, QColor(THEME['surface']))
    palette.setColor(QPalette.ToolTipText, QColor(THEME['text']))
    palette.setColor(QPalette.Text, QColor(THEME['text']))
    palette.setColor(QPalette.Button, QColor(THEME['surface']))
    palette.setColor(QPalette.ButtonText, QColor(THEME['text']))
    palette.setColor(QPalette.BrightText, QColor(THEME['error']))
    palette.setColor(QPalette.Link, QColor(THEME['accent']))
    palette.setColor(QPalette.Highlight, QColor(THEME['accent']))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    window = VoderGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
