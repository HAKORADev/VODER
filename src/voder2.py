import sys
import os
import time
import tempfile
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
                             QListWidget, QListWidgetItem, QLineEdit, QSpinBox)
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

SEEDVC_CHECKPOINTS_DIR = "./models/seed_vc_v2/checkpoints"
QWEN_TTS_MODEL_DIR = "./models/qwen_tts_voice_design"
ACE_STEP_MODEL_DIR = "./models/ace_step_1_5"

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    os.makedirs(SEEDVC_CHECKPOINTS_DIR, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=SEEDVC_CHECKPOINTS_DIR)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=SEEDVC_CHECKPOINTS_DIR)
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
            min-width: 120px;
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
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
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

class QwenTTSVoiceDesign:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.model_dir_full = os.path.join(model_dir, "qwen_tts_voice_design")
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
                    #self.model.save_pretrained(model_path)
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
                    import shutil
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
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

class QwenTTS:
    def __init__(self, model_dir="./models"):
        self.model_dir = model_dir
        self.model_dir_base = os.path.join(model_dir, "qwen_tts_base")
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

                    print("Saving Qwen-TTS locally...")
                    #self.model.save_pretrained(self.model_dir_base)
                    
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
        # Use same checkpoints dir as hf_utils.load_custom_model_from_hf expects
        self.checkpoints_dir = "checkpoints"
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

                # Load vocoder separately (Hydra's instantiate doesn't work well with HuggingFace's HubMixin)
                try:
                    from modules.bigvgan import bigvgan
                    self.model.vocoder = bigvgan.BigVGAN.from_pretrained(
                        "nvidia/bigvgan_v2_22khz_80band_256x",
                        use_cuda_kernel=False
                    )
                    print("Vocoder loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load vocoder: {e}")

                # Load main checkpoints from our downloads
                self.model.load_checkpoints(
                    cfm_checkpoint_path=cfm_path,
                    ar_checkpoint_path=ar_path
                )

                # Load content extractor narrow
                ce_narrow_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_NARROW_CHECKPOINT,
                    local_name="bsq32_light.pth"
                )
                if ce_narrow_path:
                    ce_narrow_checkpoint = torch.load(ce_narrow_path, map_location="cpu")
                    self.model.content_extractor_narrow.load_state_dict(ce_narrow_checkpoint, strict=False)

                # Load content extractor wide
                ce_wide_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_WIDE_CHECKPOINT,
                    local_name="bsq2048_light.pth"
                )
                if ce_wide_path:
                    ce_wide_checkpoint = torch.load(ce_wide_path, map_location="cpu")
                    self.model.content_extractor_wide.load_state_dict(ce_wide_checkpoint, strict=False)

                # Load style encoder
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
            except ImportError as e:
                print(f"Missing dependency for Seed-VC: {e}")
            except Exception as e:
                print(f"Error loading Seed-VC v2: {e}")

    def download_checkpoint(self, repo_id, filename, local_name):
        local_path = os.path.join(self.checkpoints_dir, local_name)
        if os.path.exists(local_path):
            return local_path
        try:
            # hf_hub_download returns the actual path to the downloaded file
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

class AceStepWrapper:
    def __init__(self):
        # Determine the correct checkpoints path that AceStepHandler expects
        # AceStepHandler._get_project_root() returns the parent of acestep/ directory
        # which is where voder2.py is located (voder/)
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
                # project_root parameter is ignored by initialize_service
                # It always uses _get_project_root() which returns the parent of acestep/
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

class ProcessingThread(QThread):
    progress_signal = pyqtSignal(int)
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, mode, base_path=None, target_path=None, text=None, output_path=None, 
                 voice_instruct=None, dialogue_data=None, voice_prompts=None, duration=None):
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
        self.stt = None
        self.tts = None
        self.tts_voice_design = None
        self.seed_vc = None
        self.ace_tt = None

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
                else:
                    self.error_signal.emit("VoiceDesign synthesis failed")

            elif self.mode == "tts_voice_design_dialogue":
                self.status_signal.emit("Loading Qwen-TTS VoiceDesign model...")
                self.tts_voice_design = QwenTTSVoiceDesign()
                self.progress_signal.emit(10)

                if self.tts_voice_design.model is None:
                    self.error_signal.emit("Failed to load VoiceDesign model")
                    return

                total_steps = len(self.dialogue_data)
                success, message = self.tts_voice_design.synthesize_dialogue(
                    self.dialogue_data, 
                    self.voice_prompts, 
                    self.output_path
                )

                if success:
                    self.progress_signal.emit(100)
                    self.finished_signal.emit(self.output_path)
                else:
                    self.error_signal.emit(message)

            elif self.mode == "seed_vc_convert":
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
                else:
                    self.error_signal.emit("Music generation failed")

            elif self.mode == "ttm_vc_generate":
                temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_ttm_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_target_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_vc_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                
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

                    self.status_signal.emit("Resampling TTM output to 22050Hz...")
                    self.progress_signal.emit(50)
                    
                    waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
                    if sr_ttm != 22050:
                        resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 22050)
                        waveform_ttm = resampler_ttm(waveform_ttm)
                    torchaudio.save(temp_ttm_22k.name, waveform_ttm, 22050)

                    self.status_signal.emit("Resampling target voice to 22050Hz...")
                    self.progress_signal.emit(60)
                    
                    waveform_target, sr_target = torchaudio.load(self.target_path)
                    if sr_target != 22050:
                        resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
                        waveform_target = resampler_target(waveform_target)
                    torchaudio.save(temp_target_22k.name, waveform_target, 22050)

                    self.status_signal.emit("Loading Seed-VC model...")
                    self.seed_vc = SeedVCV2()
                    self.progress_signal.emit(70)

                    if self.seed_vc.model is None:
                        self.error_signal.emit("Failed to load Seed-VC model")
                        return

                    self.status_signal.emit("Converting voice...")
                    self.progress_signal.emit(80)
                    
                    vc_success = self.seed_vc.convert(
                        source_path=temp_ttm_22k.name,
                        reference_path=temp_target_22k.name,
                        output_path=temp_vc_output_22k.name
                    )
                    
                    if not vc_success:
                        self.error_signal.emit("Voice conversion failed")
                        return

                    self.status_signal.emit("Upsampling output to 44100Hz...")
                    self.progress_signal.emit(95)
                    
                    waveform_out, sr_out = torchaudio.load(temp_vc_output_22k.name)
                    if sr_out != 44100:
                        resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                        waveform_out = resampler_out(waveform_out)
                    torchaudio.save(self.output_path, waveform_out, 44100)
                    
                    self.progress_signal.emit(100)
                    self.finished_signal.emit(self.output_path)
                    
                finally:
                    for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output_22k.name]:
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
Hello, this is a friendly voice speaking clearly and naturally.

---

# Dialogue Mode Example:
1:James: "Welcome to our podcast! Today we'll discuss AI."
2:Sarah: "Thanks James! I'm excited to share my research."
3:James: "Let's start with the basics. What is AI?"

# For dialogue, add voice prompts in the Voice Prompt area below"""

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

        os.makedirs("results", exist_ok=True)

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
            
            self.tts_prompt_edit.setPlaceholderText("# Single Mode:\na clean adult male with exciting tone and clear pronunciation\n\n# OR Dialogue Mode (Character:Voice):\nJames: professional male with deep authoritative voice\nSarah: young female with energetic and friendly tone")
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

            
            self.tts_prompt_edit.setPlaceholderText("# Single Mode: No prompt needed\n\n# Dialogue Mode - Character:AudioNumber\n# The number refers to the audio file number in the list\nJames: 1\nSarah: 2\nAlex: 3")
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

        self.tts_script_edit = QTextEdit()
        self.tts_script_edit.setStyleSheet(get_text_edit_style())
        self.tts_script_edit.setMinimumHeight(200)
        self.tts_script_edit.setPlaceholderText(TTS_HELPER)
        self.tts_script_edit.textChanged.connect(self.check_ready)
        layout.addWidget(self.tts_script_edit, stretch=1)

        prompt_label = QLabel("Voice Prompt")
        prompt_label.setStyleSheet(get_subtitle_label_style())
        layout.addWidget(prompt_label)

        self.tts_prompt_edit = QTextEdit()
        self.tts_prompt_edit.setStyleSheet(get_text_edit_style())
        self.tts_prompt_edit.setMinimumHeight(120)
        self.tts_prompt_edit.setPlaceholderText("# Single Mode:\na clean adult male with exciting tone and clear pronunciation\n\n# OR Dialogue Mode (Character:Voice):\nJames: professional male with deep authoritative voice\nSarah: young female with energetic and friendly tone")
        self.tts_prompt_edit.textChanged.connect(self.check_ready)
        layout.addWidget(self.tts_prompt_edit)

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
        self.ttm_seconds_spin.setRange(0, 59)
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

    def tts_vc_get_audio_count(self):
        return len(self.tts_vc_audio_files)

    def tts_vc_get_audio_path(self, audio_number):
        return self.tts_vc_audio_files.get(audio_number, None)

    def tts_vc_get_all_audio_files(self):
        return self.tts_vc_audio_files.copy()

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
            script = self.tts_script_edit.toPlainText().strip()
            audio_count = self.tts_vc_get_audio_count()
            if script and audio_count > 0:
                self.patch_btn.setEnabled(True)
            else:
                self.patch_btn.setEnabled(False)
        elif mode == "TTS":  
            script = self.tts_script_edit.toPlainText().strip()
            voice_prompt = self.tts_prompt_edit.toPlainText().strip()
            if script and voice_prompt:
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
    
        if mode == "TTM" or mode == "TTM+VC":
            self.ttm_clear_btn.setEnabled(True)
        elif mode == "TTS" or mode == "TTS+VC":
            self.clear_btn.setEnabled(True)
        else:
            pass

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
        output_path = os.path.join("results", f"voder_output_{timestamp}.wav")

        self.worker = ProcessingThread("synthesize", target_path=self.target_audio_path, 
                                       text=text, output_path=output_path)
        self.worker.progress_signal.connect(self.progress.setValue)
        self.worker.status_signal.connect(self.status_bar.setText)
        self.worker.finished_signal.connect(self.on_synthesis_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

    def patch_audio_tts(self):
        script_text = self.tts_script_edit.toPlainText().strip()
        voice_prompt = self.tts_prompt_edit.toPlainText().strip()

        if not script_text:
            QMessageBox.warning(self, "Error", "Please enter script text")
            return

        if not voice_prompt:
            QMessageBox.warning(self, "Error", "Please enter voice prompt")
            return

        if is_dialogue_mode(script_text):
            dialogue_data, error = parse_dialogue_script(script_text)
            if error:
                QMessageBox.warning(self, "Dialogue Script Error", error)
                return

            voice_prompts = parse_voice_prompts(voice_prompt)
            if not voice_prompts:
                QMessageBox.warning(self, "Error", "No valid voice prompts found")
                return

            script_chars = set(char.lower() for _, char, _ in dialogue_data)
            prompt_chars = set(voice_prompts.keys())

            missing_chars = script_chars - prompt_chars
            if missing_chars:
                QMessageBox.warning(self, "Character Mismatch", 
                    f"Characters in script but missing voice prompts:\n{', '.join(sorted(missing_chars))}")
                return

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("results", f"voder_tts_dialogue_{timestamp}.wav")

            self.set_processing_state(True)
            self.status_bar.setText("Processing dialogue...")
            self.progress.setValue(0)

            self.worker = ProcessingThread("tts_voice_design_dialogue",
                                           dialogue_data=dialogue_data,
                                           voice_prompts=voice_prompts,
                                           output_path=output_path)
            self.worker.progress_signal.connect(self.progress.setValue)
            self.worker.status_signal.connect(self.status_bar.setText)
            self.worker.finished_signal.connect(self.on_synthesis_finished)
            self.worker.error_signal.connect(self.on_error)
            self.worker.start()
        else:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("results", f"voder_tts_single_{timestamp}.wav")

            self.set_processing_state(True)
            self.status_bar.setText("Generating speech with VoiceDesign...")
            self.progress.setValue(0)

            self.worker = ProcessingThread("tts_voice_design",
                                           text=script_text,
                                           voice_instruct=voice_prompt,
                                           output_path=output_path)
            self.worker.progress_signal.connect(self.progress.setValue)
            self.worker.status_signal.connect(self.status_bar.setText)
            self.worker.finished_signal.connect(self.on_synthesis_finished)
            self.worker.error_signal.connect(self.on_error)
            self.worker.start()

    def patch_audio_tts_vc(self):
        script_text = self.tts_script_edit.toPlainText().strip()
        audio_count = self.tts_vc_get_audio_count()

        if not script_text:
            QMessageBox.warning(self, "Error", "Please enter script text")
            return

        if audio_count == 0:
            QMessageBox.warning(self, "Error", "Please add at least one voice reference audio file")
            return

        audio_files = self.tts_vc_get_all_audio_files()

        if audio_count == 1:
            audio_path = list(audio_files.values())[0]
            self.generate_tts_vc_single(script_text, audio_path)
        else:
            dialogue_data, error = parse_dialogue_script(script_text)
            if error:
                QMessageBox.warning(self, "Dialogue Script Error", error)
                return

            if len(audio_files) < 2:
                QMessageBox.warning(self, "Error", 
                    "Dialogue mode requires 2+ audio files. Please add more voice references or use single mode.")
                return

            voice_mapping = {}
            voice_prompt_text = self.tts_prompt_edit.toPlainText().strip()
            if voice_prompt_text:
                voice_mapping = parse_voice_prompts(voice_prompt_text)

            script_chars = set(char.lower() for _, char, _ in dialogue_data)
            prompt_chars = set(voice_mapping.keys())

            missing_chars = script_chars - prompt_chars
            if missing_chars:
                QMessageBox.warning(self, "Character Mismatch", 
                    f"Characters in script but missing voice references:\n{', '.join(sorted(missing_chars))}\n\n"
                    f"Format: CHARACTER_NAME:AUDIO_NUMBER (e.g., James:1, Sara:2)")
                return

            self.generate_tts_vc_dialogue(dialogue_data, voice_mapping, audio_files)

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
        output_path = os.path.join("results", f"voder_tts_vc_single_{timestamp}.wav")

        self.status_bar.setText("Generating speech with cloned voice...")
        self.progress.setValue(50)

        success = tts.synthesize(script_text, output_path)

        if success and os.path.exists(output_path):
            self.on_synthesis_finished(output_path)
        else:
            QMessageBox.warning(self, "Error", "Speech generation failed")
            self.set_processing_state(False)

    def generate_tts_vc_dialogue(self, dialogue_data, voice_mapping, audio_files):
        self.set_processing_state(True)
        self.status_bar.setText("Processing dialogue with voice cloning...")
        self.progress.setValue(0)

        temp_dir = tempfile.mkdtemp()
        temp_files = []
        tts = QwenTTS()

        try:
            total_steps = len(dialogue_data)
            for i, (num, char, script_text) in enumerate(dialogue_data):
                char_lower = char.lower()

                if char_lower not in voice_mapping:
                    QMessageBox.warning(self, "Error", f"Character '{char}' not found in voice prompts")
                    self.set_processing_state(False)
                    return

                audio_num = voice_mapping[char_lower]
                if audio_num not in audio_files:
                    QMessageBox.warning(self, "Error", f"Audio file {audio_num} not found")
                    self.set_processing_state(False)
                    return

                audio_path = audio_files[audio_num]

                self.status_bar.setText(f"Generating line {num}/{total_steps} for '{char}'...")
                progress = int((i / total_steps) * 80)
                self.progress.setValue(progress)

                success = tts.extract_voice(audio_path)
                if not success:
                    QMessageBox.warning(self, "Error", f"Failed to extract voice from audio {audio_num}")
                    self.set_processing_state(False)
                    return

                temp_file = os.path.join(temp_dir, f"line_{num}.wav")
                temp_files.append((num, temp_file))

                success = tts.synthesize(script_text, temp_file)
                if not success:
                    QMessageBox.warning(self, "Error", f"Failed to generate speech for line {num}")
                    self.set_processing_state(False)
                    return

            self.status_bar.setText("Compiling dialogue...")
            self.progress.setValue(90)

            if len(temp_files) < 1:
                QMessageBox.warning(self, "Error", "No audio files generated")
                self.set_processing_state(False)
                return

            temp_files.sort(key=lambda x: x[0])

            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for _, tf in temp_files:
                    f.write(f"file '{tf}'\n")

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join("results", f"voder_tts_vc_dialogue_{timestamp}.wav")

            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                QMessageBox.warning(self, "Error", f"FFmpeg concatenation failed: {result.stderr}")
                self.set_processing_state(False)
                return

            self.on_synthesis_finished(output_path)

        finally:
            import shutil
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

    def patch_audio_sts(self):
        if not self.base_audio_path or not self.target_audio_path:
            return

        self.set_processing_state(True)
        self.status_bar.setText("Converting voice with Seed-VC...")
        self.progress.setValue(0)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"voder_sts_output_{timestamp}.wav")

        self.worker = ProcessingThread("seed_vc_convert", base_path=self.base_audio_path,
                                       target_path=self.target_audio_path, output_path=output_path)
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
        output_path = os.path.join("results", f"voder_ttm_output_{timestamp}.wav")

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
        output_path = os.path.join("results", f"voder_ttm_vc_output_{timestamp}.wav")

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
        self.tts_script_edit.clear()
        self.tts_prompt_edit.clear()

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

def cli_stt_tts_mode():
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
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_stt_tts_{timestamp}.wav")

    success = tts.synthesize(text, output_path)
    if not success:
        print("Error: Synthesis failed")
        return False

    print(f"\n✓ Success! Output saved to: {output_path}")
    return True

def cli_tts_mode():
    print("\n--- TTS Mode ---")
    print("Generate speech from text with voice design (Single mode only)")
    print()

    print("Enter script text (use \\n for new lines):")
    script = input("> ").strip()
    if not script:
        print("Error: No script provided")
        return False

    script = script.replace('\\n', '\n')

    print()
    print("Enter voice prompt (e.g., 'a clean adult male with exciting tone'):")
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
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_tts_{timestamp}.wav")

    success = tts_design.synthesize(script, voice_prompt, output_path)
    if not success:
        print("Error: VoiceDesign synthesis failed")
        return False

    print(f"\n✓ Success! Output saved to: {output_path}")
    return True

def cli_tts_vc_mode():
    print("\n--- TTS+VC Mode ---")
    print("Generate speech from text then convert to target voice (Single mode only)")
    print()

    print("Enter script text (use \\n for new lines):")
    script = input("> ").strip()
    if not script:
        print("Error: No script provided")
        return False

    script = script.replace('\\n', '\n')

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

    print("Generating speech with cloned voice...")
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_tts_vc_{timestamp}.wav")

    success = tts.synthesize(script, output_path)
    if not success:
        print("Error: Synthesis failed")
        return False

    print(f"\n✓ Success! Output saved to: {output_path}")
    return True

def cli_sts_mode():
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

        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"voder_sts_{timestamp}.wav")
        torchaudio.save(output_path, waveform_out, 44100)

        print(f"\n✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for temp_file in [temp_base.name, temp_target.name, temp_output_22k.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def cli_ttm_mode():
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
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_ttm_{timestamp}.wav")

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
    return True

def cli_ttm_vc_mode():
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
    temp_vc_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

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

        print("Resampling TTM output to 22050Hz...")
        import torchaudio
        waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
        if sr_ttm != 22050:
            resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 22050)
            waveform_ttm = resampler_ttm(waveform_ttm)
        torchaudio.save(temp_ttm_22k.name, waveform_ttm, 22050)

        print("Resampling target voice to 22050Hz...")
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        torchaudio.save(temp_target_22k.name, waveform_target, 22050)

        print("Loading Seed-VC model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            return False

        print("Converting voice...")
        vc_success = seed_vc.convert(
            source_path=temp_ttm_22k.name,
            reference_path=temp_target_22k.name,
            output_path=temp_vc_output_22k.name
        )

        if not vc_success:
            print("Error: Voice conversion failed")
            return False

        print("Upsampling output to 44100Hz...")
        waveform_out, sr_out = torchaudio.load(temp_vc_output_22k.name)
        if sr_out != 44100:
            resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
            waveform_out = resampler_out(waveform_out)

        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"voder_ttm_vc_{timestamp}.wav")
        torchaudio.save(output_path, waveform_out, 44100)

        print(f"\n✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output_22k.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def parse_oneline_args(args):
    """Parse one-line command arguments. Returns dict with mode, params, error."""
    if not args:
        return {'error': 'No arguments provided'}

    mode = args[0].lower()
    result = {'mode': mode, 'params': {}, 'error': None}

    i = 1
    while i < len(args):
        keyword = args[i].lower()

        if keyword in ['script', 'voice', 'lyrics', 'styling', 'base', 'target']:
            if i + 1 >= len(args):
                result['error'] = f'Missing value for parameter: {keyword}'
                return result
            value = args[i + 1]
            result['params'][keyword] = value
            i += 2
        else:
            
            try:
                duration = int(keyword)
                result['params']['duration'] = duration
                i += 1
            except ValueError:
                result['error'] = f'Unknown parameter: {keyword}'
                return result

    return result

def validate_oneline_mode(mode_name):
    """Validate mode name. Returns normalized mode or None."""
    valid_modes = ['tts', 'tts+vc', 'sts', 'ttm', 'ttm+vc']

    
    if mode_name.lower() in ['stt+tts', 'stt_tts', 'stttts']:
        return 'stt+tts_rejected'

    if mode_name.lower() in valid_modes:
        return mode_name.lower()

    return None

def show_oneline_usage():
    """Show usage information for one-line commands."""
    print("VODER One-Line Command Usage:")
    print("=" * 60)
    print()
    print("Available modes:")
    print("  tts      - Text-to-Speech")
    print("  tts+vc   - Text-to-Speech + Voice Conversion")
    print("  sts      - Speech-to-Speech (Voice Conversion)")
    print("  ttm      - Text-to-Music")
    print("  ttm+vc   - Text-to-Music + Voice Conversion")
    print()
    print("Note: STT+TTS mode is not available in one-line mode.")
    print("      Use 'tts' mode with your text, or use interactive CLI.")
    print()
    print("Examples:")
    print('  python voder.py tts script "hello world" voice "male voice"')
    print('  python voder.py tts+vc script "hello" target "voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav"')
    print('  python voder.py ttm lyrics "song" styling "pop" 30')
    print('  python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
    print()
    print("Parameters:")
    print("  script   - Text content (use \\n for newlines)")
    print("  voice    - Voice prompt for TTS")
    print("  lyrics   - Song lyrics for TTM")
    print("  styling  - Style prompt for TTM")
    print("  base     - Base audio/video path")
    print("  target   - Target voice path")
    print("  <number> - Duration in seconds (10-300, for TTM modes)")

def execute_oneline_command(parsed):
    """Execute one-line command based on parsed data."""
    mode = parsed['mode']
    params = parsed['params']

    if mode == 'tts':
        return oneline_tts(params)
    elif mode == 'tts+vc':
        return oneline_tts_vc(params)
    elif mode == 'sts':
        return oneline_sts(params)
    elif mode == 'ttm':
        return oneline_ttm(params)
    elif mode == 'ttm+vc':
        return oneline_ttm_vc(params)
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_oneline_usage()
        return False

def oneline_tts(params):
    """Handle TTS one-line command."""
    if 'script' not in params:
        print("Error: TTS mode requires 'script' parameter")
        print('Usage: python voder.py tts script "your text" voice "voice prompt"')
        return False

    if 'voice' not in params:
        print("Error: TTS mode requires 'voice' parameter")
        print('Usage: python voder.py tts script "your text" voice "voice prompt"')
        return False

    script = params['script'].replace('\\n', '\n')
    voice_prompt = params['voice']

    print("Loading Qwen-TTS VoiceDesign model...")
    tts_design = QwenTTSVoiceDesign()
    if tts_design.model is None:
        print("Error: Failed to load VoiceDesign model")
        return False

    print("Generating speech...")
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_tts_{timestamp}.wav")

    success = tts_design.synthesize(script, voice_prompt, output_path)
    if not success:
        print("Error: VoiceDesign synthesis failed")
        return False

    print(f"✓ Success! Output saved to: {output_path}")
    return True

def oneline_tts_vc(params):
    """Handle TTS+VC one-line command."""
    if 'script' not in params:
        print("Error: TTS+VC mode requires 'script' parameter")
        print('Usage: python voder.py tts+vc script "your text" target "voice.wav"')
        return False

    if 'target' not in params:
        print("Error: TTS+VC mode requires 'target' parameter")
        print('Usage: python voder.py tts+vc script "your text" target "voice.wav"')
        return False

    target_path = params['target']
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return False

    script = params['script'].replace('\\n', '\n')

    print("Loading Qwen-TTS model...")
    tts = QwenTTS()
    print("Extracting voice characteristics...")
    success = tts.extract_voice(target_path)
    if not success:
        print("Error: Voice extraction failed")
        return False

    print("Generating speech with cloned voice...")
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_tts_vc_{timestamp}.wav")

    success = tts.synthesize(script, output_path)
    if not success:
        print("Error: Synthesis failed")
        return False

    print(f"✓ Success! Output saved to: {output_path}")
    return True

def oneline_sts(params):
    """Handle STS one-line command."""
    if 'base' not in params:
        print("Error: STS mode requires 'base' parameter")
        print('Usage: python voder.py sts base "input.wav" target "voice.wav"')
        return False

    if 'target' not in params:
        print("Error: STS mode requires 'target' parameter")
        print('Usage: python voder.py sts base "input.wav" target "voice.wav"')
        return False

    base_path = params['base']
    target_path = params['target']

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

        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"voder_sts_{timestamp}.wav")
        torchaudio.save(output_path, waveform_out, 44100)

        print(f"✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for temp_file in [temp_base.name, temp_target.name, temp_output_22k.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def oneline_ttm(params):
    """Handle TTM one-line command."""
    if 'lyrics' not in params:
        print("Error: TTM mode requires 'lyrics' parameter")
        print('Usage: python voder.py ttm lyrics "song lyrics" styling "pop style" 30')
        return False

    if 'styling' not in params:
        print("Error: TTM mode requires 'styling' parameter")
        print('Usage: python voder.py ttm lyrics "song lyrics" styling "pop style" 30')
        return False

    if 'duration' not in params:
        print("Error: TTM mode requires duration (10-300 seconds)")
        print('Usage: python voder.py ttm lyrics "song" styling "pop" 30')
        return False

    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False

    lyrics = params['lyrics'].replace('\\n', '\n')
    style = params['styling'].replace('\\n', '\n')

    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False

    print(f"Generating music ({duration}s duration)...")
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join("results", f"voder_ttm_{timestamp}.wav")

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
    """Handle TTM+VC one-line command."""
    if 'lyrics' not in params:
        print("Error: TTM+VC mode requires 'lyrics' parameter")
        print('Usage: python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
        return False

    if 'styling' not in params:
        print("Error: TTM+VC mode requires 'styling' parameter")
        print('Usage: python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
        return False

    if 'duration' not in params:
        print("Error: TTM+VC mode requires duration (10-300 seconds)")
        print('Usage: python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
        return False

    if 'target' not in params:
        print("Error: TTM+VC mode requires 'target' parameter")
        print('Usage: python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"')
        return False

    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False

    target_path = params['target']
    if not os.path.exists(target_path):
        print(f"Error: Target file not found: {target_path}")
        return False

    lyrics = params['lyrics'].replace('\\n', '\n')
    style = params['styling'].replace('\\n', '\n')

    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper()
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        return False

    temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_ttm_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_target_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_vc_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

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

        print("Resampling TTM output to 22050Hz...")
        import torchaudio
        waveform_ttm, sr_ttm = torchaudio.load(temp_ttm_output.name)
        if sr_ttm != 22050:
            resampler_ttm = torchaudio.transforms.Resample(sr_ttm, 22050)
            waveform_ttm = resampler_ttm(waveform_ttm)
        torchaudio.save(temp_ttm_22k.name, waveform_ttm, 22050)

        print("Resampling target voice to 22050Hz...")
        waveform_target, sr_target = torchaudio.load(target_path)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        torchaudio.save(temp_target_22k.name, waveform_target, 22050)

        print("Loading Seed-VC model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            return False

        print("Converting voice...")
        vc_success = seed_vc.convert(
            source_path=temp_ttm_22k.name,
            reference_path=temp_target_22k.name,
            output_path=temp_vc_output_22k.name
        )

        if not vc_success:
            print("Error: Voice conversion failed")
            return False

        print("Upsampling output to 44100Hz...")
        waveform_out, sr_out = torchaudio.load(temp_vc_output_22k.name)
        if sr_out != 44100:
            resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
            waveform_out = resampler_out(waveform_out)

        os.makedirs("results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("results", f"voder_ttm_vc_{timestamp}.wav")
        torchaudio.save(output_path, waveform_out, 44100)

        print(f"✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for temp_file in [temp_ttm_output.name, temp_ttm_22k.name, temp_target_22k.name, temp_vc_output_22k.name]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

def interactive_cli_mode():
    while True:
        print_banner()

        print("\nSelect Mode:")
        print("1. STT+TTS (Speech-to-Text + Text-to-Speech)")
        print("2. TTS (Text-to-Speech)")
        print("3. TTS+VC (Text-to-Speech + Voice Conversion)")
        print("4. STS (Speech-to-Speech / Voice Conversion)")
        print("5. TTM (Text-to-Music)")
        print("6. TTM+VC (Text-to-Music + Voice Conversion)")

        choice = input("\nEnter your choice (1-6): ").strip()

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
        else:
            print("Invalid choice. Please enter 1-6.")
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
    """Parse and execute one-line command. Returns success boolean."""
    
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