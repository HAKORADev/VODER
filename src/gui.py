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
                             QStackedWidget, QDoubleSpinBox, QShortcut)
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
    'background': '#0D0D0D',
    'surface': '#151515',
    'surface_hover': '#1E1E1E',
    'surface_active': '#282828',
    'text': '#F0F0F0',
    'text_secondary': '#808080',
    'accent': '#FFFFFF',
    'accent_hover': '#E8E8E8',
    'accent_pressed': '#D0D0D0',
    'accent_disabled': '#555555',
    'border': '#2A2A2A',
    'border_light': '#3A3A3A',
    'border_disabled': '#222222',
    'error': '#FF4444',
    'warning': '#FF9800',
    'success': '#FFFFFF',
    'panel_background': '#111111',
    'panel_border': '#2A2A2A',
    'sidebar_background': '#0A0A0A',
    'sidebar_item_hover': '#1A1A1A',
    'sidebar_item_active': '#FFFFFF',
}

def contrast_color(bg_hex):
    hex_clean = bg_hex.lstrip('#')
    if len(hex_clean) != 6:
        return '#000000'
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return '#000000' if luminance > 0.5 else '#FFFFFF'

def get_main_button_style():
    return f"""
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {THEME['surface']}, stop:0.3 {THEME['surface']}, stop:0.7 {THEME['surface_hover']}, stop:1 {THEME['surface']});
            border: 2px solid {THEME['accent']};
            border-radius: 8px;
            font-size: 14px;
            font-weight: bold;
            color: {THEME['accent']};
            padding: 8px 16px;
        }}
        QPushButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {THEME['surface']}, stop:0.3 {THEME['surface_hover']}, stop:0.7 {THEME['surface_active']}, stop:1 {THEME['surface']});
            border: 2px solid {THEME['accent']};
        }}
        QPushButton:pressed {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {THEME['surface_hover']}, stop:0.3 {THEME['surface_active']}, stop:0.7 {THEME['surface_hover']}, stop:1 {THEME['surface_hover']});
            border: 2px solid {THEME['accent_hover']};
        }}
        QPushButton:disabled {{
            background-color: {THEME['surface']};
            border: 2px solid {THEME['border_disabled']};
            color: {THEME['accent_disabled']};
        }}
    """

def get_secondary_button_style():
    return get_main_button_style()

def get_surface_button_style():
    return f"""
        QPushButton {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border_light']};
            border-radius: 6px;
            font-size: 12px;
            padding: 6px 12px;
        }}
        QPushButton:hover {{
            background-color: {THEME['surface_hover']};
            border: 1px solid {THEME['accent']};
        }}
        QPushButton:pressed {{
            background-color: {THEME['surface_active']};
            border: 1px solid {THEME['accent']};
        }}
        QPushButton:disabled {{
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border_disabled']};
            color: {THEME['accent_disabled']};
        }}
    """

def get_panel_style():
    return f"""
        QFrame {{
            background-color: {THEME['panel_background']};
            border: 1px solid {THEME['panel_border']};
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
            background-color: {THEME['accent']};
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
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            padding: 8px;
        }}
        QTextEdit:focus {{
            border: 1px solid {THEME['accent']};
        }}
    """

def get_combo_box_style():
    return f"""
        QComboBox {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border_light']};
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
            border: 1px solid {THEME['accent']};
        }}
        QComboBox:disabled {{
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border_disabled']};
            color: {THEME['accent_disabled']};
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
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 6px 8px;
            font-size: 13px;
        }}
        QLineEdit:focus {{
            border: 1px solid {THEME['accent']};
        }}
        QLineEdit:disabled {{
            background-color: {THEME['surface']};
            border: 1px solid {THEME['border_disabled']};
            color: {THEME['accent_disabled']};
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
            background-color: {THEME['sidebar_item_active']};
            color: {contrast_color(THEME['sidebar_item_active'])};
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
            color: {contrast_color(THEME['accent'])};
        }}
        QCheckBox::indicator:hover {{
            border: 2px solid {THEME['accent']};
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
            color: {contrast_color(THEME['accent'])};
        }}
    """

def get_spin_box_style():
    return f"""
        QSpinBox, QDoubleSpinBox {{
            background-color: {THEME['surface']};
            color: {THEME['text']};
            border: 1px solid {THEME['border']};
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 13px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {THEME['accent']};
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
    def make_combo_row(parent_layout, label_text, items, default_index=0):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        combo = QComboBox()
        combo.setStyleSheet(get_combo_box_style())
        combo.addItems(items)
        if default_index < len(items):
            combo.setCurrentIndex(default_index)
        row.addWidget(label)
        row.addWidget(combo, stretch=1)
        row.addStretch()
        parent_layout.addLayout(row)
        return combo

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


class TTSTab(QWidget):
    run_signal = pyqtSignal(list)
    transcribe_signal = pyqtSignal(list)

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
        self.sub_mode.addItems(["TTS", "Modify Speech", "SLC", "SVC", "Dub"])
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

        self.overdose_cb = HelperWidgets.make_checkbox(self.inner, "Overdose (VibeVoice ASR for dialogue source)")
        self.extreme_cb = HelperWidgets.make_checkbox(self.inner, "Extreme (Fish Speech S2Pro, .ttse voice files)")
        self.result_edit = HelperWidgets.make_save_picker(self.inner, "Result Path", "Optional: copy output to this path")

        self.run_btn = QPushButton("Run TTS")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        self.inner.addWidget(self.run_btn)
        self.inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        self.on_submode_changed("TTS")

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
        if mode == "TTS":
            self.build_tts_ui()
        elif mode == "Modify Speech":
            self.build_modify_speech_ui()
        elif mode == "SLC":
            self.build_slc_ui()
        elif mode == "SVC":
            self.build_svc_ui()
        elif mode == "Dub":
            self.build_dub_ui()

    def build_tts_ui(self):
        mode_row = QHBoxLayout()
        self.tts_mode_single = QRadioButton("Single")
        self.tts_mode_single.setStyleSheet(get_radio_button_style())
        self.tts_mode_single.setChecked(True)
        self.tts_mode_dialogue = QRadioButton("Dialogue")
        self.tts_mode_dialogue.setStyleSheet(get_radio_button_style())
        mode_row.addWidget(self.tts_mode_single)
        mode_row.addWidget(self.tts_mode_dialogue)
        mode_row.addStretch()
        self.container_layout.addLayout(mode_row)

        self.tts_script_label = QLabel("Script")
        self.tts_script_label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px;")
        self.container_layout.addWidget(self.tts_script_label)

        self.tts_script_text = QTextEdit()
        self.tts_script_text.setPlaceholderText("Enter text...")
        self.tts_script_text.setStyleSheet(get_text_edit_style())
        self.tts_script_text.setMinimumHeight(100)
        self.tts_script_text.setMaximumHeight(200)
        self.container_layout.addWidget(self.tts_script_text)

        self.tts_dialogue_widget = DialogueScriptWidget()
        self.tts_dialogue_widget.setMinimumHeight(120)
        self.tts_dialogue_widget.setMaximumHeight(240)
        self.container_layout.addWidget(self.tts_dialogue_widget)
        self.tts_dialogue_widget.hide()
        self.tts_script_label.hide()

        self.tts_mode_single.toggled.connect(self._on_tts_mode_changed)
        self.tts_dialogue_widget.characters_changed.connect(self._on_characters_changed)

        HelperWidgets.make_label(self.container_layout, "Voice Descriptions (Character: description)")
        self.tts_voice_list = KeyValueList("Character", "Voice Description", "Character name", "e.g. deep male voice")
        self.tts_voice_list.setMinimumHeight(80)
        self.tts_voice_list.add_btn.hide()
        self.container_layout.addWidget(self.tts_voice_list)

        HelperWidgets.make_separator(self.container_layout)
        HelperWidgets.make_label(self.container_layout, "Voice Clone Targets (Character: audio path)")
        self.tts_clone_list = KeyValueList("Character", "Audio Path", "Character name", "path/to/voice.wav", with_browse=True)
        self.tts_clone_list.setMinimumHeight(80)
        self.tts_clone_list.add_btn.hide()
        self.container_layout.addWidget(self.tts_clone_list)

        self.tts_first_cb = HelperWidgets.make_checkbox(self.container_layout, "First (use first reference as speaker template)")

        HelperWidgets.make_label(self.container_layout, "Additional References (for multi-reference)")
        self.tts_extra_refs = FileListWidget(self.container_layout, file_filter="Audio (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        self.container_layout.addWidget(self.tts_extra_refs)

        self.tts_sts_prefix_cb = HelperWidgets.make_checkbox(self.container_layout, "STS mode (route target through voice conversion)")

        HelperWidgets.make_separator(self.container_layout)
        self.tts_music_edit = HelperWidgets.make_line_edit(self.container_layout, "Music Desc", "Background music description (dialogue mode)")
        self.tts_level_edit = HelperWidgets.make_line_edit(self.container_layout, "Music Level", 'e.g. "10:20-50 30:60-80"')
        self.tts_reference_edit = HelperWidgets.make_line_edit(self.container_layout, "Music Ref", "Path or URL to reference audio for music style")
        self.tts_ocr_edit = HelperWidgets.make_file_picker(self.container_layout, "OCR Image", "Path to image for text extraction", "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tiff *.webp)")

    def build_modify_speech_ui(self):
        self.ms_source_edit = HelperWidgets.make_file_picker(self.container_layout, "Source Audio", "Audio/video file or URL to transcribe and modify")
        self.ms_overdose_cb = HelperWidgets.make_checkbox(self.container_layout, "Overdose (VibeVoice ASR)")
        HelperWidgets.make_label(self.container_layout, "Transcribed Text (editable):")
        self.ms_script_text = QTextEdit()
        self.ms_script_text.setPlaceholderText("Click Transcribe first, then edit the text here...")
        self.ms_script_text.setStyleSheet(get_text_edit_style())
        self.ms_script_text.setMinimumHeight(120)
        self.ms_script_text.setMaximumHeight(240)
        self.container_layout.addWidget(self.ms_script_text)
        self.ms_target_edit = HelperWidgets.make_file_picker(self.container_layout, "Target Voice", "Reference voice audio (leave empty to use source voice)")
        self.ms_extreme_cb = HelperWidgets.make_checkbox(self.container_layout, "Extreme (Fish Speech S2Pro)")
        self.ms_preserve_cb = HelperWidgets.make_checkbox(self.container_layout, "Preserve non-vocals (keep music/instruments)")
        ms_btn_row = QHBoxLayout()
        self.ms_transcribe_btn = QPushButton("Transcribe")
        self.ms_transcribe_btn.setStyleSheet(get_surface_button_style())
        self.ms_transcribe_btn.setCursor(Qt.PointingHandCursor)
        self.ms_transcribe_btn.clicked.connect(self.on_ms_transcribe)
        ms_btn_row.addWidget(self.ms_transcribe_btn)
        self.ms_synthesize_btn = QPushButton("Synthesize")
        self.ms_synthesize_btn.setStyleSheet(get_main_button_style())
        self.ms_synthesize_btn.setCursor(Qt.PointingHandCursor)
        self.ms_synthesize_btn.clicked.connect(self.on_ms_synthesize)
        ms_btn_row.addWidget(self.ms_synthesize_btn)
        ms_btn_row.addStretch()
        self.container_layout.addLayout(ms_btn_row)

    def on_ms_transcribe(self):
        source = self.ms_source_edit.text().strip() if hasattr(self, 'ms_source_edit') else ""
        if not source:
            return
        args = ["stt", source]
        self.transcribe_signal.emit(args)

    def on_ms_synthesize(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)

    def set_ms_transcription(self, text):
        if hasattr(self, 'ms_script_text'):
            self.ms_script_text.setText(text)

    def build_slc_ui(self):
        self.slc_input_edit = HelperWidgets.make_file_picker(self.container_layout, "Input Audio", "Audio file or URL to convert")
        self.slc_target_edit = HelperWidgets.make_file_picker(self.container_layout, "Target Voice", "Reference voice audio (optional)")
        self.slc_music_cb = HelperWidgets.make_checkbox(self.container_layout, "Music mode (SLC with music)")
        self.slc_translate_cb = HelperWidgets.make_checkbox(self.container_layout, "Translate")
        self.slc_lang_source = HelperWidgets.make_line_edit(self.container_layout, "Source Lang", "Source language code (e.g. en, auto)")
        self.slc_lang_target = HelperWidgets.make_line_edit(self.container_layout, "Target Lang", "Target language code (e.g. ja, ar)")
        self.slc_extreme_cb = HelperWidgets.make_checkbox(self.container_layout, "Extreme (Fish Speech S2Pro)")

    def build_svc_ui(self):
        self.svc_input_edit = HelperWidgets.make_file_picker(self.container_layout, "Input Audio", "Audio file or URL to change voice of")
        HelperWidgets.make_label(self.container_layout, "Voice Target (use one):")
        self.svc_target_edit = HelperWidgets.make_file_picker(self.container_layout, "Target Voice", "Reference voice audio file")
        self.svc_voice_edit = HelperWidgets.make_line_edit(self.container_layout, "Voice Desc", "e.g. deep male voice (alternative to target file)")
        self.svc_extreme_cb = HelperWidgets.make_checkbox(self.container_layout, "Extreme (Fish Speech S2Pro)")

    def build_dub_ui(self):
        self.dub_source_edit = HelperWidgets.make_file_picker(self.container_layout, "Source Video/Audio", "Video or audio file or URL to dub")
        self.dub_subtitle_cb = HelperWidgets.make_checkbox(self.container_layout, "Generate subtitles")
        self.dub_subtitle_original_cb = HelperWidgets.make_checkbox(self.container_layout, "Keep original language subtitles")
        self.dub_subtitle_lang_source = HelperWidgets.make_line_edit(self.container_layout, "Subtitle Lang", "Source-Target e.g. auto-ar (optional)")
        self.dub_translate_lang_source = HelperWidgets.make_line_edit(self.container_layout, "Translate Lang", "Source-Target e.g. auto-ar")
        self.dub_translate_cb = HelperWidgets.make_checkbox(self.container_layout, "Translate (enable with lang spec above)")
        self.dub_se_cb = HelperWidgets.make_checkbox(self.container_layout, "Apply sound enhancement during dub")
        self.dub_video_cb = HelperWidgets.make_checkbox(self.container_layout, "Output video file")
        self.dub_overdose_cb = HelperWidgets.make_checkbox(self.container_layout, "Overdose (VibeVoice ASR)")

    def _on_tts_mode_changed(self, checked):
        if checked:
            self.tts_script_text.show()
            self.tts_script_label.show()
            self.tts_dialogue_widget.hide()
        else:
            self.tts_script_text.hide()
            self.tts_script_label.hide()
            self.tts_dialogue_widget.show()

    def _on_characters_changed(self, chars):
        self.tts_voice_list.ensure_keys(chars)
        self.tts_clone_list.ensure_keys(chars)

    def build_args(self):
        mode = self.sub_mode.currentText()
        args = ["tts"]

        if mode == "TTS":
            is_dialogue = self.tts_mode_dialogue.isChecked()

            ocr = self.tts_ocr_edit.text().strip() if hasattr(self, 'tts_ocr_edit') else ""
            if ocr:
                args.extend(["ocr", ocr])

            if is_dialogue:
                items = self.tts_dialogue_widget.get_dialogue_items()
                if not items and not ocr:
                    return None
                for idx, char, text in items:
                    args.extend(["script", f"{char}: {text}"])
            else:
                script = self.tts_script_text.toPlainText().strip() if hasattr(self, 'tts_script_text') else ""
                if not script and not ocr:
                    return None
                if script:
                    args.extend(["script", script])

            if hasattr(self, 'tts_voice_list'):
                for char, desc in self.tts_voice_list.get_items():
                    args.extend(["voice", f"{char}: {desc}"])

            if hasattr(self, 'tts_clone_list'):
                for char, path in self.tts_clone_list.get_items():
                    target_val = path
                    if hasattr(self, 'tts_extra_refs') and self.tts_extra_refs.get_paths():
                        extra_paths = self.tts_extra_refs.get_paths()
                        target_val = "(" + path + ")" + "".join(f"({ep})" for ep in extra_paths)
                    if hasattr(self, 'tts_sts_prefix_cb') and self.tts_sts_prefix_cb.isChecked():
                        target_val = "sts:" + target_val
                    args.extend(["target", f"{char}: {target_val}"])

            if hasattr(self, 'tts_first_cb') and self.tts_first_cb.isChecked():
                args.append("first")

            music = self.tts_music_edit.text().strip() if hasattr(self, 'tts_music_edit') else ""
            if music:
                args.extend(["music", music])

            level = self.tts_level_edit.text().strip() if hasattr(self, 'tts_level_edit') else ""
            if level:
                args.extend(["level", level])

            reference = self.tts_reference_edit.text().strip() if hasattr(self, 'tts_reference_edit') else ""
            if reference:
                args.extend(["reference", reference])

        elif mode == "Modify Speech":
            script = self.ms_script_text.toPlainText().strip() if hasattr(self, 'ms_script_text') else ""
            if not script:
                return None
            args.extend(["script", script])
            target = self.ms_target_edit.text().strip() if hasattr(self, 'ms_target_edit') else ""
            source = self.ms_source_edit.text().strip() if hasattr(self, 'ms_source_edit') else ""
            if target:
                args.extend(["target", target])
            elif source:
                args.extend(["target", source])

        elif mode == "SLC":
            input_path = self.slc_input_edit.text().strip() if hasattr(self, 'slc_input_edit') else ""
            if not input_path:
                return None
            args.append("slc")
            if hasattr(self, 'slc_translate_cb') and self.slc_translate_cb.isChecked():
                args.append("translate")
                lang_src = self.slc_lang_source.text().strip() if hasattr(self, 'slc_lang_source') else ""
                lang_tgt = self.slc_lang_target.text().strip() if hasattr(self, 'slc_lang_target') else ""
                if lang_src or lang_tgt:
                    src = lang_src or "auto"
                    tgt = lang_tgt or "en"
                    args.append(f"({src}-{tgt})")
            if hasattr(self, 'slc_music_cb') and self.slc_music_cb.isChecked():
                args.append("music")
            args.append(input_path)
            if hasattr(self, 'slc_target_edit') and self.slc_target_edit.text().strip():
                args.extend(["target", self.slc_target_edit.text().strip()])

        elif mode == "SVC":
            input_path = self.svc_input_edit.text().strip() if hasattr(self, 'svc_input_edit') else ""
            if not input_path:
                return None
            args.append("svc")
            args.append(input_path)
            target = self.svc_target_edit.text().strip() if hasattr(self, 'svc_target_edit') else ""
            voice = self.svc_voice_edit.text().strip() if hasattr(self, 'svc_voice_edit') else ""
            if target:
                args.extend(["target", target])
            elif voice:
                args.extend(["voice", voice])

        elif mode == "Dub":
            source = self.dub_source_edit.text().strip() if hasattr(self, 'dub_source_edit') else ""
            if not source:
                return None
            args.append("dub")
            args.append(source)
            if hasattr(self, 'dub_subtitle_cb') and self.dub_subtitle_cb.isChecked():
                args.append("subtitle")
                if hasattr(self, 'dub_subtitle_original_cb') and self.dub_subtitle_original_cb.isChecked():
                    args.append("original")
                sub_lang = self.dub_subtitle_lang_source.text().strip() if hasattr(self, 'dub_subtitle_lang_source') else ""
                if sub_lang:
                    args.append(f"({sub_lang})")
            if hasattr(self, 'dub_translate_cb') and self.dub_translate_cb.isChecked():
                args.append("translate")
                trans_lang = self.dub_translate_lang_source.text().strip() if hasattr(self, 'dub_translate_lang_source') else ""
                if trans_lang:
                    args.append(f"({trans_lang})")
                else:
                    args.append("(auto-en)")
            if hasattr(self, 'dub_se_cb') and self.dub_se_cb.isChecked():
                args.append("se")
            if hasattr(self, 'dub_video_cb') and self.dub_video_cb.isChecked():
                args.append("video")
            if hasattr(self, 'dub_overdose_cb') and self.dub_overdose_cb.isChecked():
                args.append("overdose")

        if self.overdose_cb.isChecked() and mode not in ("Dub", "Modify Speech"):
            args.append("overdose")
        if mode == "Modify Speech" and hasattr(self, 'ms_overdose_cb') and self.ms_overdose_cb.isChecked():
            args.append("overdose")
        if mode == "SLC" and hasattr(self, 'slc_extreme_cb') and self.slc_extreme_cb.isChecked():
            args.append("extreme")
        elif mode == "SVC" and hasattr(self, 'svc_extreme_cb') and self.svc_extreme_cb.isChecked():
            args.append("extreme")
        elif mode == "Modify Speech" and hasattr(self, 'ms_extreme_cb') and self.ms_extreme_cb.isChecked():
            args.append("extreme")
        elif self.extreme_cb.isChecked() and mode not in ("Dub", "SLC", "SVC", "Modify Speech"):
            args.append("extreme")
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


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
        self.nomusic_cb = HelperWidgets.make_checkbox(inner, "No music (disable music handling)")
        self.original_cb = HelperWidgets.make_checkbox(inner, "Original (use original audio)")
        self.overdose_cb = HelperWidgets.make_checkbox(inner, "Overdose (VibeVoice ASR)")
        self.extreme_cb = HelperWidgets.make_checkbox(inner, "Extreme (Fish Speech S2Pro)")

        self.music_cb.toggled.connect(lambda checked: self.mimic_cb.setChecked(False) if checked else None)
        self.mimic_cb.toggled.connect(lambda checked: self.music_cb.setChecked(False) if checked else None)
        self.music_cb.toggled.connect(lambda checked: self.nomusic_cb.setChecked(False) if checked else None)
        self.nomusic_cb.toggled.connect(lambda checked: self.music_cb.setChecked(False) if checked else None)
        self.original_cb.toggled.connect(lambda checked: self.music_cb.setChecked(False) if checked else None)
        self.music_cb.toggled.connect(lambda checked: self.original_cb.setChecked(False) if checked else None)

        self.sts_first_cb = HelperWidgets.make_checkbox(inner, "First (multi-ref speaker extraction)")

        HelperWidgets.make_label(inner, "Additional Target References")
        self.sts_extra_refs = FileListWidget(inner, file_filter="Audio (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        inner.addWidget(self.sts_extra_refs)

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
        target_val = self.target_edit.text().strip()
        if hasattr(self, 'sts_extra_refs') and self.sts_extra_refs.get_paths():
            extra_paths = self.sts_extra_refs.get_paths()
            target_val = "(" + target_val + ")" + "".join(f"({ep})" for ep in extra_paths)
        args = ["sts", "base", self.base_edit.text().strip(), "target", target_val]
        if self.music_cb.isChecked():
            args.append("music")
        if self.mimic_cb.isChecked():
            args.append("mimic")
        if self.nomusic_cb.isChecked():
            args.append("nomusic")
        if self.original_cb.isChecked():
            args.append("original")
        if hasattr(self, 'sts_first_cb') and self.sts_first_cb.isChecked():
            args.append("first")
        if self.overdose_cb.isChecked():
            args.append("overdose")
        if self.extreme_cb.isChecked():
            args.append("extreme")
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


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
        self.sub_mode.addItems(["Generate", "Voice", "Voice Clone (VC)", "Remix", "Repaint", "Complete", "Lego", "Extract", "BGM"])
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
        method_name = f'build_{mode.lower().replace(" ", "_").replace("(vc)", "vc")}_ui'
        getattr(self, method_name, self.build_generate_ui)()

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

    def build_voice_ui(self):
        self.voice_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics", "Song lyrics (use \\n for line breaks)", 100)
        self.voice_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "Style/mood prompt", 80)
        self.voice_duration = HelperWidgets.make_spinbox(self.container_layout, "Duration (s)", 10, 300, 30)
        HelperWidgets.make_label(self.container_layout, "Reference Audio (optional)")
        self.voice_target_type = QComboBox()
        self.voice_target_type.setStyleSheet(get_combo_box_style())
        self.voice_target_type.addItems(["None", "As-is (full audio)", "Extract vocals", "Extract instruments"])
        r = QHBoxLayout()
        rl = QLabel("Target Type")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.voice_target_type, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.voice_target_path = HelperWidgets.make_file_picker(self.container_layout, "Target Path", "Reference audio path")

    def build_voice_clone_vc_ui(self):
        self.vc_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics", "Song lyrics (use \\n for line breaks)", 100)
        self.vc_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "Style/mood prompt", 80)
        self.vc_duration = HelperWidgets.make_spinbox(self.container_layout, "Duration (s)", 10, 300, 30)
        self.vc_clone = HelperWidgets.make_file_picker(self.container_layout, "Clone Voice", "Source voice audio for cloning")
        self.vc_clone_first_cb = HelperWidgets.make_checkbox(self.container_layout, "Clone first (use first segment only)")
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
        self.remix_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics (opt)", "Optional new lyrics", 80)
        self.remix_bias = HelperWidgets.make_spinbox(self.container_layout, "Bias (0-100)", 0, 100, 40)
        HelperWidgets.make_label(self.container_layout, "Reference Audio (up to 3, optional)")
        self.remix_ref_list = KeyValueList("Ref Type", "Audio Path", "voice, music, or empty", "path/to/ref.wav", with_browse=True)
        self.remix_ref_list.setMinimumHeight(80)
        self.container_layout.addWidget(self.remix_ref_list)

    def build_repaint_ui(self):
        self.repaint_source_prefix = QComboBox()
        self.repaint_source_prefix.setStyleSheet(get_combo_box_style())
        self.repaint_source_prefix.addItems(["None", "voice", "music"])
        r = QHBoxLayout()
        rl = QLabel("Source Prefix")
        rl.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        r.addWidget(rl)
        r.addWidget(self.repaint_source_prefix, stretch=1)
        r.addStretch()
        self.container_layout.addLayout(r)
        self.repaint_source = HelperWidgets.make_file_picker(self.container_layout, "Source", "Source audio/video or URL to repaint")
        self.repaint_styling = HelperWidgets.make_text_edit(self.container_layout, "Styling", "New style for the repainted section", 80)
        self.repaint_time = HelperWidgets.make_line_edit(self.container_layout, "Time Range", "e.g. 20-80 or 20.5-80.5")
        self.repaint_lyrics = HelperWidgets.make_text_edit(self.container_layout, "Lyrics (opt)", "Optional lyrics for repainted section", 80)
        self.repaint_bias = HelperWidgets.make_spinbox(self.container_layout, "Bias (0-100)", 0, 100, 40)
        self.repaint_multipass = HelperWidgets.make_line_edit(self.container_layout, "Multipass", 'e.g. "20-80/orchestral" "80-100/cinematic" (optional)')
        HelperWidgets.make_label(self.container_layout, "Reference Audio (up to 3, optional)")
        self.repaint_ref_list = KeyValueList("Ref Type", "Audio Path", "voice, music, or empty", "path/to/ref.wav", with_browse=True)
        self.repaint_ref_list.setMinimumHeight(80)
        self.container_layout.addWidget(self.repaint_ref_list)

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
        self.complete_noblend_cb = HelperWidgets.make_checkbox(self.container_layout, "No blending")
        self.complete_usrc_cb = HelperWidgets.make_checkbox(self.container_layout, "Blend with original source (usrc)")
        self.complete_video_cb = HelperWidgets.make_checkbox(self.container_layout, "Preserve video")
        self.complete_sfx_edit = HelperWidgets.make_line_edit(self.container_layout, "SFX Specs", 'e.g. "sfx:thunder/5-30/50" (optional, multiple separated by space)')
        HelperWidgets.make_label(self.container_layout, "Reference Audio (up to 3, optional)")
        self.complete_ref_list = KeyValueList("Ref Type", "Audio Path", "voice, music, or empty", "path/to/ref.wav", with_browse=True)
        self.complete_ref_list.setMinimumHeight(80)
        self.container_layout.addWidget(self.complete_ref_list)

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
        HelperWidgets.make_label(self.container_layout, "Reference Audio (up to 3, optional)")
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
        self.bgm_video_cb = HelperWidgets.make_checkbox(self.container_layout, "Output video file")
        self.bgm_reference = HelperWidgets.make_file_picker(self.container_layout, "Reference (opt)", "Reference audio/video or URL for music style")
        self.bgm_sfx_edit = HelperWidgets.make_line_edit(self.container_layout, "SFX Specs", 'e.g. "sfx:rain/10-0/30" (optional, multiple by space)')

    def _add_target_ref(self, args, target_type, target_path):
        if target_type == "As-is (full audio)" and target_path.strip():
            args.extend(["target", target_path.strip()])
        elif target_type == "Extract vocals" and target_path.strip():
            args.extend(["target voice", target_path.strip()])
        elif target_type == "Extract instruments" and target_path.strip():
            args.extend(["target music", target_path.strip()])

    def _add_ref_list(self, args, ref_list):
        for stem, path in ref_list.get_items():
            stem_clean = stem.strip().lower()
            path_clean = path.strip()
            if stem_clean in ("voice", "music"):
                args.extend(["reference", stem_clean, path_clean])
            else:
                args.extend(["reference", path_clean])

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
            args.extend(["duration", str(self.gen_duration.value())])
            self._add_target_ref(args, self.gen_target_type.currentText(), self.gen_target_path.text())

        elif mode == "voice":
            lyrics = self.voice_lyrics.toPlainText().strip()
            styling = self.voice_styling.toPlainText().strip()
            if not lyrics or not styling:
                return None
            args.extend(["lyrics", lyrics, "styling", styling])
            args.extend(["duration", str(self.voice_duration.value())])
            args.append("voice")
            self._add_target_ref(args, self.voice_target_type.currentText(), self.voice_target_path.text())

        elif mode == "voice clone (vc)":
            lyrics = self.vc_lyrics.toPlainText().strip()
            styling = self.vc_styling.toPlainText().strip()
            clone = self.vc_clone.text().strip()
            if not lyrics or not styling or not clone:
                return None
            args.append("vc")
            args.extend(["lyrics", lyrics, "styling", styling])
            args.extend(["duration", str(self.vc_duration.value())])
            if hasattr(self, 'vc_clone_first_cb') and self.vc_clone_first_cb.isChecked():
                args.extend(["clone", "first", clone])
            else:
                args.extend(["clone", clone])
            self._add_target_ref(args, self.vc_target_type.currentText(), self.vc_target_path.text())

        elif mode == "remix":
            source = self.remix_source.text().strip()
            styling = self.remix_styling.toPlainText().strip()
            if not source or not styling:
                return None
            args.extend(["remix", source, "styling", styling, "bias", str(self.remix_bias.value())])
            lyrics = self.remix_lyrics.toPlainText().strip() if hasattr(self, 'remix_lyrics') else ""
            if lyrics:
                args.extend(["lyrics", lyrics])
            if hasattr(self, 'remix_ref_list'):
                self._add_ref_list(args, self.remix_ref_list)

        elif mode == "repaint":
            source = self.repaint_source.text().strip()
            styling = self.repaint_styling.toPlainText().strip()
            time_range = self.repaint_time.text().strip()
            if not source or not styling or not time_range:
                return None
            prefix = self.repaint_source_prefix.currentText() if hasattr(self, 'repaint_source_prefix') else "None"
            if prefix != "None":
                args.extend(["repaint", prefix, source])
            else:
                args.extend(["repaint", source])
            args.extend(["styling", styling, f"time:{time_range}", "bias", str(self.repaint_bias.value())])
            lyrics = self.repaint_lyrics.toPlainText().strip() if hasattr(self, 'repaint_lyrics') else ""
            if lyrics:
                args.extend(["lyrics", lyrics])
            multipass = self.repaint_multipass.text().strip() if hasattr(self, 'repaint_multipass') else ""
            if multipass:
                for spec in multipass.split():
                    args.append(spec)
            if hasattr(self, 'repaint_ref_list'):
                self._add_ref_list(args, self.repaint_ref_list)

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
            if hasattr(self, 'complete_noblend_cb') and self.complete_noblend_cb.isChecked():
                args.append("noblend")
            if hasattr(self, 'complete_usrc_cb') and self.complete_usrc_cb.isChecked():
                args.append("usrc")
            if self.complete_video_cb.isChecked():
                args.append("video")
            args.append(source)
            args.extend(["add", combined])
            sfx = self.complete_sfx_edit.text().strip() if hasattr(self, 'complete_sfx_edit') else ""
            if sfx:
                for spec in sfx.split():
                    if spec.startswith("sfx:"):
                        args.append(spec)
            if hasattr(self, 'complete_ref_list'):
                self._add_ref_list(args, self.complete_ref_list)

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
            if hasattr(self, 'bgm_video_cb') and self.bgm_video_cb.isChecked():
                args.append("video")
            ref = self.bgm_reference.text().strip() if hasattr(self, 'bgm_reference') else ""
            if ref:
                args.extend(["reference", ref])
            sfx = self.bgm_sfx_edit.text().strip() if hasattr(self, 'bgm_sfx_edit') else ""
            if sfx:
                for spec in sfx.split():
                    if spec.startswith("sfx:"):
                        args.append(spec)

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
        self.se_cb = HelperWidgets.make_checkbox(inner, "Speech Enhancement (denoise/dereverb before transcription)")
        self.overdose_cb = HelperWidgets.make_checkbox(inner, "Overdose (VibeVoice ASR, requires 24GB+ VRAM or 48GB+ RAM)")
        self.subtitle_cb = HelperWidgets.make_checkbox(inner, "Subtitle mode (implies overdose, structured output)")

        HelperWidgets.make_separator(inner)
        HelperWidgets.make_label(inner, "Translation (optional)")
        self.translate_cb = HelperWidgets.make_checkbox(inner, "Translate")
        self.translate_source_edit = HelperWidgets.make_line_edit(inner, "Source Lang", "Source language code or auto (e.g. ja, auto)")
        self.translate_target_edit = HelperWidgets.make_line_edit(inner, "Target Lang", "Target language code (e.g. en, ar)")

        self.subtitle_cb.toggled.connect(lambda checked: self.overdose_cb.setChecked(True) if checked else None)

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
        if self.se_cb.isChecked():
            args.append("se")
        if self.overdose_cb.isChecked():
            args.append("overdose")
        if self.subtitle_cb.isChecked():
            args.append("subtitle")
        if self.translate_cb.isChecked():
            args.append("translate")
            source = self.translate_source_edit.text().strip()
            target = self.translate_target_edit.text().strip()
            if source or target:
                src = source or "auto"
                tgt = target or "en"
                args.append(f"({src}-{tgt})")
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

        HelperWidgets.make_separator(inner)
        HelperWidgets.make_label(inner, "Enhancement Mode")

        self.se_mode = QComboBox()
        self.se_mode.setStyleSheet(get_combo_box_style())
        self.se_mode.addItems([
            "Basic Enhancement",
            "Voice (extract/enhance voice only)",
            "SR (super-resolution)",
            "SR + Voice",
            "SR + Music",
            "SR + Voice + Music",
        ])
        mode_row = QHBoxLayout()
        mode_label = QLabel("Mode")
        mode_label.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; min-width: 90px;")
        mode_row.addWidget(mode_label)
        mode_row.addWidget(self.se_mode, stretch=1)
        mode_row.addStretch()
        inner.addLayout(mode_row)

        self.blend_cb = HelperWidgets.make_checkbox(inner, "Blend (mix enhanced with original)")

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
        mode = self.se_mode.currentText()
        if mode.startswith("Voice"):
            args.append("voice")
        elif mode.startswith("SR + Voice + Music"):
            args.append("sr")
            args.append("voice")
            args.append("music")
        elif mode.startswith("SR + Voice"):
            args.append("sr")
            args.append("voice")
        elif mode.startswith("SR + Music"):
            args.append("sr")
            args.append("music")
        elif mode.startswith("SR"):
            args.append("sr")
        if self.blend_cb.isChecked():
            args.append("blend")
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
        args.extend(["steps", str(self.steps_spin.value())])
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

        self.source_edit = HelperWidgets.make_file_picker(inner, "Source", "Audio or video file or URL")
        self.target_edit = HelperWidgets.make_file_picker(inner, "Target Voice", "Reference voice audio (optional)")
        self.se_cb = HelperWidgets.make_checkbox(inner, "Speech Enhancement (before separation)")
        self.overdose_cb = HelperWidgets.make_checkbox(inner, "Overdose (VibeVoice ASR)")
        self.blend_cb = HelperWidgets.make_checkbox(inner, "Blend (mix each speaker with non-vocals)")
        self.video_cb = HelperWidgets.make_checkbox(inner, "Video (mux audio with original video)")
        self.extreme_cb = HelperWidgets.make_checkbox(inner, "Extreme (Fish Speech S2Pro)")
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
        source = self.source_edit.text().strip()
        if not source:
            return None
        args = ["ss"]
        if self.target_edit.text().strip():
            args.extend(["target", self.target_edit.text().strip()])
        if self.se_cb.isChecked():
            args.append("se")
        if self.overdose_cb.isChecked():
            args.append("overdose")
        if self.blend_cb.isChecked():
            args.append("blend")
        if self.video_cb.isChecked():
            args.append("video")
        if self.extreme_cb.isChecked():
            args.append("extreme")
        args.append(source)
        if self.result_edit.text().strip():
            args.extend(["result", self.result_edit.text().strip()])
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class TrainTab(QWidget):
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

        self.voice_name_edit = HelperWidgets.make_line_edit(inner, "Voice Name", "e.g. james, sarah (used for voice: lookup)")
        HelperWidgets.make_label(inner, "Reference Audio Files (1 or more)")
        self.ref_list = FileListWidget(inner, file_filter="Audio (*.wav *.mp3 *.flac *.ogg);;All Files (*)")
        inner.addWidget(self.ref_list)
        self.test_cb = HelperWidgets.make_checkbox(inner, "Test after training")
        self.test_script_edit = HelperWidgets.make_line_edit(inner, "Test Script", "Custom test script text (optional, uses default if empty)")
        self.first_cb = HelperWidgets.make_checkbox(inner, "First (use first segment only)")
        self.extreme_cb = HelperWidgets.make_checkbox(inner, "Extreme (train for .ttse file, Fish Speech S2Pro)")

        self.run_btn = QPushButton("Run Train")
        self.run_btn.setStyleSheet(get_main_button_style())
        self.run_btn.setCursor(Qt.PointingHandCursor)
        self.run_btn.clicked.connect(self.on_run)
        inner.addWidget(self.run_btn)
        inner.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

    def build_args(self):
        voice_name = self.voice_name_edit.text().strip()
        paths = self.ref_list.get_paths()
        if not voice_name or not paths:
            return None
        args = ["train", f"voice:{voice_name}"]
        args.extend(paths)
        if self.test_cb.isChecked():
            test_script = self.test_script_edit.text().strip()
            args.append("test")
            if test_script:
                args.append(test_script)
        if self.first_cb.isChecked():
            args.append("first")
        if self.extreme_cb.isChecked():
            args.append("extreme")
        return args

    def on_run(self):
        args = self.build_args()
        if args:
            self.run_signal.emit(args)


class VoderGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VODER")
        self.setMinimumSize(960, 700)
        self.setStyleSheet(get_window_style())
        self.current_thread = None
        self.audio_player = None
        self.last_output_path = None
        self.terminal_visible = False
        self.running = False
        self.player_thread = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setFixedHeight(48)
        top_bar.setStyleSheet(f"background-color: {THEME['background']}; border-bottom: 1px solid {THEME['border']};")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("VODER")
        title.setStyleSheet(f"color: {THEME['text']}; font-size: 18px; font-weight: bold; letter-spacing: 2px;")
        top_layout.addWidget(title)
        top_layout.addStretch()

        self.terminal_btn = QPushButton(">_<")
        self.terminal_btn.setFixedSize(36, 36)
        self.terminal_btn.setCursor(Qt.PointingHandCursor)
        self.terminal_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a1a;
                color: #FFFFFF;
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a2a2a;
                border: 1px solid #FFFFFF;
            }
            QPushButton:pressed {
                background-color: #FFFFFF;
                color: #000000;
            }
        """)
        self.terminal_btn.clicked.connect(self.toggle_terminal)
        top_layout.addWidget(self.terminal_btn)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px; margin-left: 12px;")
        top_layout.addWidget(self.status_label)

        main_layout.addWidget(top_bar)

        body = QHBoxLayout()
        body.setSpacing(0)
        body.setContentsMargins(0, 0, 0, 0)

        sidebar = QFrame()
        sidebar.setFixedWidth(160)
        sidebar.setStyleSheet(f"background-color: {THEME['sidebar_background']}; border-right: 1px solid {THEME['border']};")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        self.mode_list = QListWidget()
        self.mode_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {THEME['sidebar_background']};
                border: none;
                outline: none;
                padding: 8px 0;
            }}
            QListWidget::item {{
                color: {THEME['text']};
                padding: 12px 16px;
                font-size: 13px;
                border: none;
                border-radius: 0;
            }}
            QListWidget::item:hover {{
                background-color: {THEME['sidebar_item_hover']};
            }}
            QListWidget::item:selected {{
                background-color: {THEME['sidebar_item_active']};
                color: {contrast_color(THEME['sidebar_item_active'])};
            }}
        """)
        modes = ["TTS", "STS", "TTM", "STT", "SE", "SFX", "SVS", "SS", "Train"]
        self.mode_list.addItems(modes)
        self.mode_list.setCurrentRow(0)
        self.mode_list.currentRowChanged.connect(self.on_mode_changed)
        sidebar_layout.addWidget(self.mode_list)

        sidebar_layout.addStretch()

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"color: {THEME['border']};")
        sidebar_layout.addWidget(sep)

        self.ready_label = QLabel("Ready")
        self.ready_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px; padding: 8px 16px;")
        sidebar_layout.addWidget(self.ready_label)

        body.addWidget(sidebar)

        self.content_stack = QStackedWidget()

        self.tts_tab = TTSTab()
        self.sts_tab = STSTab()
        self.ttm_tab = TTMTab()
        self.stt_tab = STTTab()
        self.se_tab = SETab()
        self.sfx_tab = SFXTab()
        self.svs_tab = SVSTab()
        self.ss_tab = SSTab()
        self.train_tab = TrainTab()

        self.content_stack.addWidget(self.tts_tab)
        self.content_stack.addWidget(self.sts_tab)
        self.content_stack.addWidget(self.ttm_tab)
        self.content_stack.addWidget(self.stt_tab)
        self.content_stack.addWidget(self.se_tab)
        self.content_stack.addWidget(self.sfx_tab)
        self.content_stack.addWidget(self.svs_tab)
        self.content_stack.addWidget(self.ss_tab)
        self.content_stack.addWidget(self.train_tab)

        content_wrapper = QWidget()
        content_layout = QVBoxLayout(content_wrapper)
        content_layout.setContentsMargins(8, 8, 8, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.content_stack, stretch=1)

        audio_frame = QFrame()
        audio_frame.setStyleSheet(f"background-color: {THEME['panel_background']}; border-top: 1px solid {THEME['border']};")
        audio_layout = QVBoxLayout(audio_frame)
        audio_layout.setContentsMargins(12, 8, 12, 8)
        audio_layout.setSpacing(4)

        self.waveform = AudioWaveformWidget()
        self.waveform.setFixedHeight(48)
        audio_layout.addWidget(self.waveform)

        audio_row = QHBoxLayout()
        audio_row.setSpacing(8)
        self.audio_path_edit = QLineEdit()
        self.audio_path_edit.setReadOnly(True)
        self.audio_path_edit.setPlaceholderText("Output audio")
        self.audio_path_edit.setStyleSheet(get_line_edit_style())
        audio_row.addWidget(self.audio_path_edit, stretch=1)

        self.play_btn = QPushButton("Play")
        self.play_btn.setStyleSheet(get_surface_button_style())
        self.play_btn.setCursor(Qt.PointingHandCursor)
        self.play_btn.setFixedWidth(60)
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_audio)
        audio_row.addWidget(self.play_btn)

        self.audio_stop_btn = QPushButton("Stop")
        self.audio_stop_btn.setStyleSheet(get_surface_button_style())
        self.audio_stop_btn.setCursor(Qt.PointingHandCursor)
        self.audio_stop_btn.setFixedWidth(60)
        self.audio_stop_btn.setEnabled(False)
        self.audio_stop_btn.clicked.connect(self.stop_audio)
        audio_row.addWidget(self.audio_stop_btn)

        audio_layout.addLayout(audio_row)
        content_layout.addWidget(audio_frame)

        body.addWidget(content_wrapper, stretch=1)
        main_layout.addLayout(body)

        self.terminal_panel = QFrame(self.content_stack)
        self.terminal_panel.setStyleSheet(f"""
            QFrame {{
                background-color: {THEME['panel_background']};
                border: 1px solid {THEME['border_light']};
                border-radius: 8px;
            }}
        """)
        terminal_layout = QVBoxLayout(self.terminal_panel)
        terminal_layout.setContentsMargins(12, 8, 12, 8)
        terminal_layout.setSpacing(6)

        term_header = QHBoxLayout()
        term_title = QLabel("Terminal")
        term_title.setStyleSheet(f"color: {THEME['text']}; font-size: 13px; font-weight: bold;")
        term_header.addWidget(term_title)
        term_header.addStretch()

        self.term_close_btn = QPushButton("\u00d7")
        self.term_close_btn.setFixedSize(24, 24)
        self.term_close_btn.setCursor(Qt.PointingHandCursor)
        self.term_close_btn.setStyleSheet("""
            QPushButton { background-color: #3a3a3a; color: white; border: none; border-radius: 12px; font-size: 14px; font-weight: bold; }
            QPushButton:hover { background-color: #f44336; }
        """)
        self.term_close_btn.clicked.connect(self.toggle_terminal)
        term_header.addWidget(self.term_close_btn)
        terminal_layout.addLayout(term_header)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet(get_text_edit_style())
        self.console.setMinimumHeight(150)
        self.console.setMaximumHeight(300)
        terminal_layout.addWidget(self.console)

        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(get_progress_bar_style())
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(4)
        terminal_layout.addWidget(self.progress_bar)

        term_btn_row = QHBoxLayout()
        term_btn_row.setSpacing(6)

        self.copy_cmd_btn = QPushButton("Copy Command")
        self.copy_cmd_btn.setStyleSheet(get_surface_button_style())
        self.copy_cmd_btn.setCursor(Qt.PointingHandCursor)
        self.copy_cmd_btn.clicked.connect(self.copy_command)
        term_btn_row.addWidget(self.copy_cmd_btn)

        self.copy_output_btn = QPushButton("Copy Output")
        self.copy_output_btn.setStyleSheet(get_surface_button_style())
        self.copy_output_btn.setCursor(Qt.PointingHandCursor)
        self.copy_output_btn.clicked.connect(self.copy_output)
        term_btn_row.addWidget(self.copy_output_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setStyleSheet(get_surface_button_style())
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.clear_console)
        term_btn_row.addWidget(self.clear_btn)

        self.stop_btn = QPushButton("Stop Process")
        self.stop_btn.setStyleSheet(get_surface_button_style())
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_process)
        term_btn_row.addWidget(self.stop_btn)

        term_btn_row.addStretch()
        terminal_layout.addLayout(term_btn_row)

        self.terminal_panel.hide()

        self.tts_tab.run_signal.connect(self.run_command)
        self.tts_tab.transcribe_signal.connect(self.run_modify_speech_transcribe)
        self.sts_tab.run_signal.connect(self.run_command)
        self.ttm_tab.run_signal.connect(self.run_command)
        self.stt_tab.run_signal.connect(self.run_command)
        self.se_tab.run_signal.connect(self.run_command)
        self.sfx_tab.run_signal.connect(self.run_command)
        self.svs_tab.run_signal.connect(self.run_command)
        self.ss_tab.run_signal.connect(self.run_command)
        self.train_tab.run_signal.connect(self.run_command)

        self.run_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.run_shortcut.activated.connect(self.run_current_tab)

        self.ready_timer = QTimer()
        self.ready_timer.timeout.connect(self.update_readiness)
        self.ready_timer.start(500)

    def toggle_terminal(self):
        self.terminal_visible = not self.terminal_visible
        if self.terminal_visible:
            self.terminal_panel.setGeometry(self.content_stack.rect())
            self.terminal_panel.raise_()
            self.terminal_panel.show()
        else:
            self.terminal_panel.hide()

    def on_mode_changed(self, row):
        self.content_stack.setCurrentIndex(row)

    def run_command(self, args):
        if self.running:
            return
        self.running = True
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.console.append(f"\n> python voder.py {' '.join(_q(a) for a in args)}\n")
        self.current_thread = SubprocessThread(args)
        self.current_thread.output_signal.connect(self.on_output)
        self.current_thread.finished_signal.connect(self.on_process_finished)
        self.current_thread.start()

    def run_modify_speech_transcribe(self, args):
        if self.running:
            return
        self.running = True
        self.stop_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.console.append(f"\n> python voder.py {' '.join(_q(a) for a in args)}\n")
        self.current_thread = SubprocessThread(args)
        self.current_thread.output_signal.connect(self.on_output)
        self.current_thread.finished_signal.connect(self.on_modify_speech_transcribe_finished)
        self.current_thread.start()

    def on_output(self, line):
        self.console.append(line)
        detected = self.detect_output_path(line)
        if detected:
            self.last_output_path = detected
            self.audio_path_edit.setText(detected)
            self.play_btn.setEnabled(True)
            self.waveform.set_audio(detected)

    def on_process_finished(self, returncode, output):
        self.running = False
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        self.current_thread = None
        audio_path = self.detect_output_path(output)
        if audio_path:
            self.last_output_path = audio_path
            self.audio_path_edit.setText(audio_path)
            self.play_btn.setEnabled(True)
            self.waveform.set_audio(audio_path)

    def on_modify_speech_transcribe_finished(self, returncode, output):
        self.on_process_finished(returncode, output)
        if returncode == 0:
            text = self._extract_transcription(output)
            if text:
                self.tts_tab.set_ms_transcription(text)

    def _extract_transcription(self, output):
        lines = output.strip().split('\n')
        text_lines = []
        capture = False
        for line in lines:
            if 'Transcription' in line or 'transcription' in line or 'Result' in line:
                capture = True
                continue
            if capture:
                stripped = line.strip()
                if stripped and not stripped.startswith('[') and not stripped.startswith('===') and not stripped.startswith('---'):
                    text_lines.append(stripped)
        return '\n'.join(text_lines) if text_lines else output.strip()

    def detect_output_path(self, output):
        patterns = [
            r'(?:Saved|saved|Output|output|Result|result|Created|created|Written|written)\s*(?:to|at|:)?\s*["\']?([\w/\-\\\.]+\.(?:wav|mp3|flac|ogg))["\']?',
            r'["\']?([\w/\-\\\.]+\.(?:wav|mp3|flac|ogg))["\']?',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, output)
            for match in reversed(matches):
                if os.path.exists(match):
                    return match
        for line in reversed(output.split('\n')):
            line = line.strip()
            for ext in ['.wav', '.mp3', '.flac', '.ogg']:
                if ext in line.lower():
                    for word in line.split():
                        word = word.strip('"\'.,;:')
                        if word.lower().endswith(ext) and os.path.exists(word):
                            return word
        return ""

    def play_audio(self):
        if self.last_output_path and os.path.exists(self.last_output_path):
            self.stop_audio()
            self.player_thread = AudioPlayerThread(self.last_output_path)
            self.player_thread.finished_signal.connect(self.on_play_finished)
            self.player_thread.start()
            self.play_btn.setEnabled(False)
            self.audio_stop_btn.setEnabled(True)

    def stop_audio(self):
        if self.player_thread:
            self.player_thread.stop()
            self.player_thread = None
        self.play_btn.setEnabled(bool(self.last_output_path and os.path.exists(self.last_output_path) if self.last_output_path else False))
        self.audio_stop_btn.setEnabled(False)

    def on_play_finished(self):
        self.player_thread = None
        self.play_btn.setEnabled(bool(self.last_output_path and os.path.exists(self.last_output_path) if self.last_output_path else False))
        self.audio_stop_btn.setEnabled(False)

    def stop_process(self):
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread = None
        self.running = False
        self.stop_btn.setEnabled(False)
        self.progress_bar.setVisible(False)

    def clear_console(self):
        self.console.clear()

    def copy_command(self):
        tab = self.content_stack.currentWidget()
        args = None
        if hasattr(tab, 'build_args'):
            args = tab.build_args()
        if args:
            cmd = ' '.join(_q(a) for a in args)
            clipboard = QApplication.clipboard()
            clipboard.setText(f"python voder.py {cmd}")

    def copy_output(self):
        text = self.console.toPlainText()
        if text:
            clipboard = QApplication.clipboard()
            clipboard.setText(text)

    def run_current_tab(self):
        tab = self.content_stack.currentWidget()
        if hasattr(tab, 'on_run'):
            tab.on_run()

    def update_readiness(self):
        tab = self.content_stack.currentWidget()
        if hasattr(tab, 'build_args'):
            args = tab.build_args()
            if args is not None:
                self.ready_label.setText("Ready")
                self.ready_label.setStyleSheet(f"color: {THEME['accent']}; font-size: 11px; padding: 8px 16px; font-weight: bold;")
            else:
                self.ready_label.setText("Ready")
                self.ready_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px; padding: 8px 16px;")
        else:
            self.ready_label.setText("Ready")
            self.ready_label.setStyleSheet(f"color: {THEME['text_secondary']}; font-size: 11px; padding: 8px 16px;")


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
    palette.setColor(QPalette.Highlight, QColor(THEME['accent']))
    palette.setColor(QPalette.HighlightedText, QColor(contrast_color(THEME['accent'])))
    app.setPalette(palette)

    window = VoderGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
