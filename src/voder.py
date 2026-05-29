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
FISH_S2PRO_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "fish_s2pro")
ACESTEP_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "acestep")
SEED_VC_V1_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v1")
SEED_VC_V2_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v2")
WHISPER_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "whisper")
UNISE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "unise")
TANGOFLUX_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "tangoflux")
SVS_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "svs")
VIBEVOICE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "vibevoice_asr")

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
os.makedirs(FISH_S2PRO_DIR, exist_ok=True)
os.makedirs(ACESTEP_DIR, exist_ok=True)
os.makedirs(SEED_VC_V1_DIR, exist_ok=True)
os.makedirs(SEED_VC_V2_DIR, exist_ok=True)
os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(UNISE_DIR, exist_ok=True)
os.makedirs(TANGOFLUX_DIR, exist_ok=True)
os.makedirs(SVS_DIR, exist_ok=True)
os.makedirs(VIBEVOICE_DIR, exist_ok=True)

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
from omegaconf import DictConfig
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
import subprocess
import json
import re
import random

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

class WhisperSTT:
    def __init__(self, model_dir=None, skip_turbo=False):
        self.model_dir = WHISPER_DIR if model_dir is None else model_dir
        self.model = None
        self.checkpoint_path = os.path.join(self.model_dir, "whisper-turbo.pt")
        self.translate_model = None
        self.translate_checkpoint_path = os.path.join(self.model_dir, "whisper-large-v3.pt")
        if not skip_turbo:
            self.ensure_model()

    def _save_checkpoint(self, model, path):
        import torch
        checkpoint = {
            "dims": {
                "n_mels": model.dims.n_mels,
                "n_audio_ctx": model.dims.n_audio_ctx,
                "n_audio_state": model.dims.n_audio_state,
                "n_audio_head": model.dims.n_audio_head,
                "n_audio_layer": model.dims.n_audio_layer,
                "n_vocab": model.dims.n_vocab,
                "n_text_ctx": model.dims.n_text_ctx,
                "n_text_state": model.dims.n_text_state,
                "n_text_head": model.dims.n_text_head,
                "n_text_layer": model.dims.n_text_layer,
            },
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, path)

    def _load_model(self, model_name, checkpoint_path):
        import whisper
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            if os.path.exists(checkpoint_path):
                return whisper.load_model(checkpoint_path)
            else:
                model = whisper.load_model(model_name)
                self._save_checkpoint(model, checkpoint_path)
                return model
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            return None

    def ensure_model(self):
        if self.model is None:
            self.model = self._load_model("large-v3-turbo", self.checkpoint_path)

    def ensure_translate_model(self):
        if self.translate_model is None:
            self.translate_model = self._load_model("large-v3", self.translate_checkpoint_path)

    def transcribe(self, audio_path):
        if self.model is None:
            return None
        try:
            result = self.model.transcribe(audio_path, word_timestamps=True)
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def translate(self, audio_path):
        self.ensure_translate_model()
        if self.translate_model is None:
            return None
        try:
            result = self.translate_model.transcribe(audio_path, task="translate", word_timestamps=True)
            return result
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def cleanup(self):
        self.model = None
        self.translate_model = None

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

    def diarize_full(self, audio_path):
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline(audio_path, min_speakers=1)
            if hasattr(result, 'speaker_diarization'):
                return result
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
            last_assigned_speaker = None
            for t_seg in transcription_segments:
                best_speaker = None
                best_overlap = 0
                for turn in diarization_turns:
                    overlap_start = max(t_seg["start"], turn["start"])
                    overlap_end = min(t_seg["end"], turn["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    if overlap_duration > 0:
                        if t_seg["start"] >= turn["start"] and t_seg["end"] <= turn["end"]:
                            if overlap_duration + 1 > best_overlap:
                                best_speaker = turn["speaker"]
                                best_overlap = overlap_duration + 1
                        elif overlap_duration > best_overlap:
                            best_speaker = turn["speaker"]
                            best_overlap = overlap_duration
                if best_speaker is not None:
                    last_assigned_speaker = best_speaker
                elif last_assigned_speaker is not None:
                    best_speaker = last_assigned_speaker
                if best_speaker is not None:
                    result.append({
                        "speaker": best_speaker,
                        "start": t_seg["start"],
                        "end": t_seg["end"],
                        "text": t_seg["text"]
                    })
            return result
        except Exception as e:
            print(f"Error formatting diarization: {e}")
            return []

def get_system_resources():
    vram_gb = 0
    single_gpu_gb = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_mem = gpu_props.total_memory / (1024 ** 3)
            vram_gb += gpu_mem
            if gpu_mem > single_gpu_gb:
                single_gpu_gb = gpu_mem

    total_sys_gb = 0
    try:
        import psutil
        total_sys_gb = psutil.virtual_memory().total / (1024 ** 3)
        swap = psutil.swap_memory()
        total_sys_gb += swap.total / (1024 ** 3)
    except:
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_total = 0
                swap_total = 0
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])
                    elif line.startswith('SwapTotal:'):
                        swap_total = int(line.split()[1])
                total_sys_gb = (mem_total + swap_total) / (1024 * 1024)
        except:
            pass
    return single_gpu_gb, total_sys_gb

class VibeVoiceASR:
    def __init__(self, model_dir=None):
        self.model_dir = VIBEVOICE_DIR if model_dir is None else model_dir
        self.processor = None
        self.model = None
        self.device = None
        self._loaded = False

    def _check_resources(self):
        single_gpu_gb, total_sys_gb = get_system_resources()
        if single_gpu_gb >= 24.0:
            self.device = torch.device("cuda:0")
            print(f"Single GPU has {single_gpu_gb:.1f} GB - loading entire model on GPU")
            return True
        if total_sys_gb >= 48.0:
            self.device = torch.device("cpu")
            print(f"CPU mode: {total_sys_gb:.1f} GB RAM+Swap/Pagefile available - loading on CPU")
            return True
        return False

    def ensure_model(self):
        if self._loaded:
            return
        try:
            import sys
            asr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asr')
            if asr_path not in sys.path:
                sys.path.insert(0, asr_path)

            os.makedirs(self.model_dir, exist_ok=True)

            if not self._check_resources():
                print("Error: VibeVoice ASR requires 24GB+ VRAM or 48GB+ combined GPU+RAM")
                print("Falling back to Whisper + pyannote")
                return

            hf_token = os.environ.get("HF_TOKEN")
            download_kwargs = {}
            if hf_token:
                download_kwargs["token"] = hf_token

            print("Loading VibeVoice ASR model...")

            from asr.vibevoice_asr_processor import VibeVoiceASRProcessor
            from asr.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

            self.processor = VibeVoiceASRProcessor.from_pretrained(
                "microsoft/VibeVoice-ASR",
                language_model_pretrained_name="Qwen/Qwen2.5-7B",
                **download_kwargs
            )

            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-ASR",
                dtype=model_dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
                cache_dir=self.model_dir,
                **download_kwargs
            )
            self.model.eval()
            self.model.to(self.device)
            self._loaded = True
            print("VibeVoice ASR model loaded successfully")
        except Exception as e:
            print(f"Error loading VibeVoice ASR model: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def transcribe(self, audio_path):
        if not self._loaded:
            self.ensure_model()
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            segments = self.processor.post_process_transcription(generated_text)

            result = []
            for seg in segments:
                raw_text = seg.get("text", seg.get("Content", ""))
                clean = re.sub(r'^\[(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]\s*', '', raw_text, flags=re.IGNORECASE).strip()
                if not clean:
                    continue
                result.append({
                    "start": seg.get("start_time", seg.get("Start", seg.get("Start time", 0))),
                    "end": seg.get("end_time", seg.get("End", seg.get("End time", 0))),
                    "speaker": seg.get("speaker_id", seg.get("Speaker", seg.get("Speaker ID", 0))),
                    "text": clean
                })
            return result
        except Exception as e:
            print(f"VibeVoice ASR transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def transcribe_plain_text(self, audio_path):
        if not self._loaded:
            self.ensure_model()
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
                context_info="Please transcribe this audio without timestamps or speaker labels."
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            try:
                segments = self.processor.post_process_transcription(generated_text)
                if segments:
                    return " ".join(seg.get("text", "") for seg in segments)
            except:
                pass

            return generated_text.strip()
        except Exception as e:
            print(f"VibeVoice ASR transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

def _parse_ref_time_spec(spec):
    if '(' not in spec or not spec.endswith(')'):
        return None, spec
    paren_start = spec.find('(')
    path = spec[paren_start + 1:-1].strip()
    if not path:
        return None, spec
    time_part = spec[:paren_start].strip()
    stem_prefix = None
    if ':' in time_part:
        colon_idx = time_part.find(':')
        after_colon = time_part[colon_idx + 1:]
        if after_colon and after_colon[0].isdigit():
            stem_prefix = time_part[:colon_idx]
            time_part = after_colon
        elif after_colon and after_colon[0] == '/':
            return None, spec
        else:
            before_colon = time_part[:colon_idx]
            if before_colon:
                stem_prefix = before_colon
                time_part = after_colon
    if not time_part:
        if stem_prefix:
            return None, f"{stem_prefix}:{path}"
        return None, path
    segments = time_part.split('/')
    ranges = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if '-' in seg:
            parts = seg.split('-', 1)
            try:
                start = float(parts[0])
                end = float(parts[1])
                if start < 0 or end < 0 or start >= end:
                    return None, spec
                ranges.append((start, end))
            except (ValueError, IndexError):
                return None, spec
        else:
            try:
                start = float(seg)
                if start < 0:
                    return None, spec
                ranges.append((start, None))
            except ValueError:
                return None, spec
    if not ranges:
        return None, spec
    if stem_prefix:
        path = f"{stem_prefix}:{path}"
    return ranges, path

def _extract_ref_segments(audio_path, time_ranges, slot_max, cleanup_list):
    duration = _get_audio_duration(audio_path)
    if duration <= 0:
        return audio_path
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 2:
        wav = wav[:2, :]
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    total_samples = wav.shape[-1]
    if len(time_ranges) == 1:
        start, end = time_ranges[0]
        if end is None:
            end = start + slot_max
        if start > duration:
            print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), adjusting")
            start = max(0, duration - slot_max)
        if end > duration:
            if time_ranges[0][1] is not None:
                print(f"Warning: Time spec end ({time_ranges[0][1]}s) exceeds audio duration ({duration:.1f}s), clamping")
            end = duration
        if end - start < slot_max:
            needed = slot_max - (end - start)
            start = max(0, start - needed)
            if end - start < slot_max:
                end = min(duration, start + slot_max)
        start_sample = int(max(0, start) * sr)
        end_sample = int(min(end, duration) * sr)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        combined = wav[:, start_sample:end_sample]
    else:
        adjusted = []
        for start, end in time_ranges:
            if end is None:
                if start > duration:
                    print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), adjusting")
                    start = max(0, duration - slot_max)
                end = start + slot_max
            else:
                if start > duration:
                    print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), skipping segment")
                    continue
                if end > duration:
                    print(f"Warning: Time spec end ({end}s) exceeds audio duration ({duration:.1f}s), clamping")
                    end = duration
            if start < end:
                adjusted.append((start, end))
        if not adjusted:
            return audio_path
        combined_dur = sum(e - s for s, e in adjusted)
        if combined_dur < slot_max:
            scale = slot_max / combined_dur
            slid = []
            for s, e in adjusted:
                seg_dur = e - s
                new_dur = seg_dur * scale
                mid = (s + e) / 2.0
                ns = mid - new_dur / 2.0
                ne = mid + new_dur / 2.0
                ns = max(0, ne - new_dur)
                ne = min(duration, ns + new_dur)
                ns = max(0, ne - new_dur)
                slid.append((ns, ne))
            adjusted = slid
        extracted = []
        for start, end in adjusted:
            start_sample = int(max(0, start) * sr)
            end_sample = int(min(end, duration) * sr)
            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(start_sample, min(end_sample, total_samples))
            if end_sample > start_sample:
                extracted.append(wav[:, start_sample:end_sample])
        if not extracted:
            return audio_path
        combined = torch.cat(extracted, dim=-1)
    target_samples = int(slot_max * sr)
    if combined.shape[-1] < target_samples:
        reps = math.ceil(target_samples / combined.shape[-1])
        combined = combined.repeat(1, reps)
        combined = combined[:, :target_samples]
    out_path = os.path.join(tempfile.gettempdir(), f"voder_ref_seg_{int(time.time())}_{len(cleanup_list)}.wav")
    torchaudio.save(out_path, combined, sr)
    cleanup_list.append(out_path)
    return out_path

def _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup_list, time_ranges=None, slot_max=30):
    resolved = None
    if not os.path.exists(raw_path) and not is_youtube_url(raw_path):
        print(f"Warning: Audio path not found: {raw_path}, skipping")
        return None
    if is_youtube_url(raw_path):
        print(f"Downloading audio from URL: {raw_path}")
        res, cl = resolve_target_to_audio(raw_path)
        if res is None:
            print("Warning: Could not download audio, skipping")
            return None
        cleanup_list.extend(cl)
        resolved = res
    else:
        r_ext = os.path.splitext(raw_path)[1].lower()
        if r_ext in VIDEO_EXTENSIONS:
            tmp = os.path.join(results_dir, f'_vid_{timestamp}_{len(cleanup_list)}.wav')
            ret = os.system(f'ffmpeg -y -i "{raw_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{tmp}" 2>/dev/null')
            if ret != 0 or not os.path.exists(tmp):
                print("Warning: Failed to extract audio from video, skipping")
                return None
            cleanup_list.append(tmp)
            resolved = tmp
        else:
            valid, msg = validate_audio_file(raw_path)
            if not valid:
                print(f"Warning: Invalid audio file: {msg}, skipping")
                return None
            resolved = raw_path
    if time_ranges:
        resolved = _extract_ref_segments(resolved, time_ranges, slot_max, cleanup_list)
    if sv_type == 'voice':
        print("Extracting vocals via SVS...")
        processed = svs_extract_vocals(resolved)
    elif sv_type == 'music':
        print("Extracting music (removing vocals) via SVS...")
        processed = svs_extract_music(resolved)
    else:
        processed = resolved
    if processed and processed != resolved and processed not in cleanup_list:
        cleanup_list.append(processed)
    return processed

def _compose_refs(ref_entries, results_dir):
    if not ref_entries:
        return None, []
    cleanup = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_entries = len(ref_entries)
    slot_max = 30 // max(1, num_entries)
    has_time_spec = any(len(e) > 2 and e[2] is not None for e in ref_entries)
    processed = []
    for entry in ref_entries:
        sv_type = entry[0]
        raw_path = entry[1]
        tr = entry[2] if len(entry) > 2 else None
        if has_time_spec and tr is None:
            tr = [(0, None)]
        audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup, time_ranges=tr, slot_max=slot_max)
        if audio_path is None:
            continue
        processed.append(audio_path)
    if not processed:
        return None, cleanup
    if len(processed) == 1:
        return processed[0], cleanup
    if has_time_spec:
        tensors = []
        for p in processed:
            wav, sr = torchaudio.load(p)
            if sr != 48000:
                wav = torchaudio.transforms.Resample(sr, 48000)(wav)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2, :]
            tensors.append(wav)
        composed = torch.cat(tensors, dim=-1)
        out_path = os.path.join(results_dir, f'_composed_ref_{timestamp}.wav')
        torchaudio.save(out_path, composed, 48000)
        cleanup.append(out_path)
        return out_path, cleanup
    print(f"Composing {len(processed)} references into 30s composite...")
    tensors = []
    for p in processed:
        wav, sr = torchaudio.load(p)
        if sr != 48000:
            wav = torchaudio.transforms.Resample(sr, 48000)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        tensors.append(wav)
    sr = 48000
    seg10 = 10 * sr
    seg5 = 5 * sr
    composed = None
    if len(tensors) == 2:
        t1, t2 = tensors[0], tensors[1]
        for idx, t in enumerate([t1, t2]):
            if t.shape[-1] < seg10:
                reps = math.ceil(seg10 / t.shape[-1])
                if idx == 0:
                    t1 = t.repeat(1, reps)
                else:
                    t2 = t.repeat(1, reps)
        third1 = t1.shape[-1] // 3
        third2 = t2.shape[-1] // 3
        off1m = random.randint(0, max(0, third1 - seg5))
        off2m = random.randint(0, max(0, third2 - seg5))
        front = t1[:, :seg10]
        mid1 = t1[:, third1 + off1m:third1 + off1m + seg5]
        mid2 = t2[:, third2 + off2m:third2 + off2m + seg5]
        end2_start = max(0, t2.shape[-1] - seg10)
        end2 = t2[:, end2_start:end2_start + seg10]
        composed = torch.cat([front, mid1, mid2, end2], dim=-1)
    else:
        for idx, t in enumerate(tensors):
            if t.shape[-1] < seg10:
                reps = math.ceil(seg10 / t.shape[-1])
                tensors[idx] = t.repeat(1, reps)
        t1, t2, t3 = tensors[0], tensors[1], tensors[2]
        third2 = t2.shape[-1] // 3
        off2 = random.randint(0, max(0, third2 - seg10))
        front = t1[:, :seg10]
        mid = t2[:, third2 + off2:third2 + off2 + seg10]
        end3_start = max(0, t3.shape[-1] - seg10)
        end = t3[:, end3_start:end3_start + seg10]
        composed = torch.cat([front, mid, end], dim=-1)
    out_path = os.path.join(results_dir, f'_composed_ref_{timestamp}.wav')
    torchaudio.save(out_path, composed, sr)
    cleanup.append(out_path)
    return out_path, cleanup

def _compose_sources(source_entries, results_dir):
    if not source_entries:
        return None, []
    if len(source_entries) == 1:
        sv_type, raw_path = source_entries[0]
        cleanup = []
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup)
        return audio_path, cleanup
    cleanup = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    processed = []
    for sv_type, raw_path in source_entries:
        audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup)
        if audio_path is None:
            continue
        processed.append(audio_path)
    if not processed:
        return None, cleanup
    print(f"Composing {len(processed)} sources into composite...")
    tensors = []
    durations = []
    for p in processed:
        wav, sr = torchaudio.load(p)
        if sr != 48000:
            wav = torchaudio.transforms.Resample(sr, 48000)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        tensors.append(wav)
        durations.append(wav.shape[-1] / 48000.0)
    total_dur = sum(durations)
    per_source = total_dur / len(tensors)
    per_source_frames = int(per_source * 48000)
    segments = []
    for t in tensors:
        if t.shape[-1] < per_source_frames:
            reps = math.ceil(per_source_frames / t.shape[-1])
            t = t.repeat(1, reps)
        seg = t[:, :per_source_frames]
        segments.append(seg)
    composed = torch.cat(segments, dim=-1)
    out_path = os.path.join(results_dir, f'_composed_src_{timestamp}.wav')
    torchaudio.save(out_path, composed, 48000)
    cleanup.append(out_path)
    return out_path, cleanup

def _parse_multi_refs(text):
    import re
    matches = re.findall(r'\(([^)]+)\)', text)
    if not matches:
        return None
    return [m.strip() for m in matches if m.strip()]

def _concat_audio_files(file_list, output_path):
    if len(file_list) == 1:
        shutil.copy(file_list[0], output_path)
        return True
    inputs = ' '.join(f'-i "{f}"' for f in file_list)
    filter_parts = ''.join(f'[{i}:a]' for i in range(len(file_list)))
    filter_str = f'{filter_parts}concat=n={len(file_list)}:v=0:a=1[out]'
    cmd = f'ffmpeg -y {inputs} -filter_complex "{filter_str}" -map "[out]" "{output_path}" 2>/dev/null'
    ret = os.system(cmd)
    return ret == 0 and os.path.exists(output_path)

def _extract_target_speaker_from_audio(source_path, target_voice_path, cleanup_list):
    try:
        from unise import UniSEEnhancer
        tse = UniSEEnhancer(UNISE_DIR)
        tse.ensure_model()
        if tse.model is None:
            print("Warning: Could not load TSE model for 'first' pipe, using original audio")
            tse.cleanup()
            del tse
            return source_path
        temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_out.close()
        ok = tse.tse_extract(source_path, target_voice_path, temp_out.name)
        tse.cleanup()
        del tse
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ok and os.path.exists(temp_out.name):
            cleanup_list.append(temp_out.name)
            return temp_out.name
        return source_path
    except Exception as _e:
        print(f"Warning: Target speaker extraction failed: {_e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return source_path

def _resolve_multi_refs(ref_paths, cleanup_list, use_first=False):
    clean_vocals = []
    for ref_path in ref_paths:
        resolved_audio, _cl = resolve_target_to_audio(ref_path.strip())
        if not resolved_audio:
            return None
        cleanup_list.extend(_cl)
        cv = svs_extract_vocals(resolved_audio)
        if cv and cv != resolved_audio:
            cleanup_list.append(cv)
        else:
            cv = resolved_audio
        if resolved_audio not in cleanup_list:
            cleanup_list.append(resolved_audio)
        clean_vocals.append(cv)
    if len(clean_vocals) > 1 and use_first:
        print("Applying 'first' pipe: extracting target speaker from additional references...")
        target_voice = clean_vocals[0]
        for idx in range(1, len(clean_vocals)):
            print(f"  Extracting target speaker from reference {idx + 1}...")
            extracted = _extract_target_speaker_from_audio(clean_vocals[idx], target_voice, cleanup_list)
            clean_vocals[idx] = extracted
    if len(clean_vocals) == 1:
        return clean_vocals[0]
    concat_path = os.path.join(tempfile.gettempdir(), f"voder_multi_ref_{int(time.time())}.wav")
    if _concat_audio_files(clean_vocals, concat_path):
        cleanup_list.append(concat_path)
        return concat_path
    return clean_vocals[0]

def _parse_repaint_pass_spec(spec):
    parts = []
    current = ''
    depth = 0
    for ch in spec:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == '/' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    if current:
        parts.append(current)
    if not parts:
        return None, 'empty pass spec'
    time_part = parts[0]
    time_parts = time_part.split('-')
    if len(time_parts) != 2:
        return None, f'invalid time range format: {time_part}'
    try:
        start_sec = float(time_parts[0].strip())
        end_sec = float(time_parts[1].strip())
    except ValueError:
        return None, f'time values must be numbers: {time_part}'
    if start_sec < 0:
        return None, 'start time cannot be negative'
    if end_sec <= 0:
        return None, 'end time must be greater than 0'
    if start_sec >= end_sec:
        return None, 'start time must be less than end time'
    lyrics = None
    styling = None
    references = []
    bias = None
    j = 1
    while j < len(parts):
        part = parts[j]
        if part.startswith('lyrics(') and part.endswith(')'):
            lyrics = part[7:-1].replace('\\n', '\n')
        elif part.startswith('styling(') and part.endswith(')'):
            styling = part[8:-1].replace('\\n', '\n')
        elif part.startswith('reference-voice(') and part.endswith(')'):
            inner = part[16:-1]
            tr, rp = _parse_ref_time_spec(inner)
            references.append(('voice', rp, tr))
        elif part.startswith('reference-music(') and part.endswith(')'):
            inner = part[15:-1]
            tr, rp = _parse_ref_time_spec(inner)
            references.append(('music', rp, tr))
        elif part.startswith('reference(') and part.endswith(')'):
            inner = part[9:-1]
            tr, rp = _parse_ref_time_spec(inner)
            references.append(('asis', rp, tr))
        elif part == 'bias' and j + 1 < len(parts):
            bias = parts[j + 1]
            j += 1
        j += 1
    if references and len(references) > 3:
        print(f"Warning: repaint pass supports up to 3 references, using first 3")
        references = references[:3]
    return {
        'start': start_sec,
        'end': end_sec,
        'lyrics': lyrics,
        'styling': styling,
        'references': references,
        'bias': bias
    }, None

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

def _parse_sfx_specs(sfx_args, max_duration):
    parsed = []
    for raw in sfx_args:
        raw = raw.strip()
        if not raw:
            continue
        if not raw.startswith('sfx:'):
            return None, f"Invalid SFX spec (must start with sfx:): {raw}"
        body = raw[4:]
        if not body:
            return None, "SFX spec requires a prompt after sfx:"
        parts = body.split('/')
        prompt = parts[0].strip()
        if not prompt:
            return None, "SFX prompt cannot be empty"

        sfx_dur = 5
        sfx_pos = None
        sfx_level = 50

        if len(parts) >= 2:
            dp = parts[1].strip()
            if not dp:
                return None, f"SFX duration-position is empty in: {raw}"
            dp_parts = dp.split('-')
            if len(dp_parts) != 2:
                return None, f"SFX duration-position must be duration-position (e.g. 10-5), got: {dp}"
            dur_str = dp_parts[0].strip()
            pos_str = dp_parts[1].strip()
            if not dur_str or not pos_str:
                return None, f"SFX duration and position must both be numbers, got: {dp}"
            dur_str = dur_str.lstrip('-')
            if not dur_str or not dur_str.isdigit():
                return None, f"SFX duration must be a number, got: {dp_parts[0]}"
            sfx_dur = int(dur_str)
            if sfx_dur < 5:
                sfx_dur = 5
            if sfx_dur > 30:
                print(f"Warning: SFX duration {sfx_dur}s exceeds 30s, clamping to 30")
                sfx_dur = 30
            if not pos_str.isdigit():
                return None, f"SFX position must be a non-negative number, got: {dp_parts[1]}"
            sfx_pos = int(pos_str)
            if sfx_pos < 0:
                return None, f"SFX position cannot be negative: {sfx_pos}"
            if sfx_pos > max_duration:
                return None, f"SFX position {sfx_pos}s exceeds source duration {max_duration:.1f}s"
            if sfx_pos + sfx_dur > max_duration:
                new_dur = max_duration - sfx_pos
                new_dur = max(1, int(new_dur))
                print(f"Warning: SFX at {sfx_pos}s with duration {sfx_dur}s exceeds source duration {max_duration:.1f}s, auto-cutting to {new_dur}s")
                sfx_dur = new_dur

        if len(parts) >= 3:
            lv_str = parts[2].strip()
            if not lv_str:
                return None, f"SFX level is empty in: {raw}"
            lv_str = lv_str.lstrip('-')
            if not lv_str or not lv_str.isdigit():
                return None, f"SFX level must be a number, got: {parts[2]}"
            sfx_level = int(lv_str)
            if sfx_level < 1:
                print(f"Warning: SFX level {sfx_level} is below 1, setting to 1")
                sfx_level = 1
            if sfx_level > 100:
                print(f"Warning: SFX level {sfx_level} exceeds 100, setting to 100")
                sfx_level = 100

        if sfx_pos is None:
            return None, f'SFX spec requires duration-position (e.g. "sfx:thunder/10-5"): {raw}'

        parsed.append({
            'prompt': prompt,
            'duration': sfx_dur,
            'position': sfx_pos,
            'level': sfx_level
        })
    return parsed, None

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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg mixing failed: {result.stderr}")
            return False
        shutil.move(mixed_temp.name, output_path)
        return True
    finally:
        if os.path.exists(mixed_temp.name):
            try:
                os.unlink(mixed_temp.name)
            except:
                pass

def _generate_music_and_mix(ace, music_description, dialogue_path, output_path, music_level_spec=None, reference_audio=None):
    duration = _get_audio_duration(dialogue_path)
    print(f"Dialogue duration: {duration:.2f}s")
    print("Generating background music...")
    music_result = generate_background_music(ace, music_description, duration, reference_audio=reference_audio)
    if music_result is None:
        print("Error: Background music generation failed")
        return False
    music_temp_path, music_temp_dir = music_result
    print("Mixing dialogue with music...")
    success = _mix_dialogue_with_music(dialogue_path, music_temp_path, output_path, music_level_spec)
    if music_temp_dir is not None:
        shutil.rmtree(music_temp_dir, ignore_errors=True)
    return success

def _generate_and_overlay_sfx(source_path, sfx_specs, output_path):
    from tangoflux import TangoFluxGenerator
    sfx_gen = TangoFluxGenerator(TANGOFLUX_DIR)
    sfx_gen.ensure_model()
    if sfx_gen.model is None:
        print("Error: Failed to load TangoFlux SFX model")
        sfx_gen.cleanup()
        del sfx_gen
        return False

    sfx_temp_dir = tempfile.mkdtemp()
    sfx_files = []
    try:
        for idx, spec in enumerate(sfx_specs):
            print(f"  Generating SFX [{idx+1}/{len(sfx_specs)}]: \"{spec['prompt']}\" ({spec['duration']}s at {spec['position']}s, level {spec['level']}%)")
            sfx_wav = os.path.join(sfx_temp_dir, f"sfx_{idx}.wav")
            audio = sfx_gen.generate(spec['prompt'], spec['duration'])
            if audio is None:
                print(f"  Warning: SFX generation failed for \"{spec['prompt']}\", skipping")
                continue
            sfx_gen.save(audio, sfx_wav)
            if not os.path.exists(sfx_wav):
                print(f"  Warning: SFX file not saved for \"{spec['prompt']}\", skipping")
                continue
            sfx_files.append((sfx_wav, spec['position'], spec['level'] / 100.0))

        sfx_gen.cleanup()
        del sfx_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not sfx_files:
            print("Warning: No SFX files generated, copying source as-is")
            shutil.copy2(source_path, output_path)
            return True

        filter_parts = []
        input_idx = 0
        cmd = ['ffmpeg', '-i', source_path]
        for sfx_path, pos, vol in sfx_files:
            cmd.extend(['-i', sfx_path])
            delay_ms = int(pos * 1000)
            label = f"sfx{input_idx}"
            filter_parts.append(f"[{input_idx + 1}:a]adelay={delay_ms}|{delay_ms},volume={vol:.2f}[{label}]")
            input_idx += 1

        mix_inputs = "[0:a]" + "".join(f"[sfx{i}]" for i in range(len(sfx_files)))
        filter_parts.append(f"{mix_inputs}amix=inputs={len(sfx_files) + 1}:duration=first:dropout_transition=0[out]")
        filter_str = ";".join(filter_parts)
        cmd.extend(['-filter_complex', filter_str, '-map', '[out]', '-y', output_path])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: SFX overlay failed: {result.stderr}, copying source as-is")
            shutil.copy2(source_path, output_path)
        return True
    finally:
        try:
            shutil.rmtree(sfx_temp_dir)
        except Exception:
            pass

def _assemble_enhanced_dialogue(dialogue_items, voice_data, tts_design_obj=None, tts_vc_obj=None, vc_voice_data=None, output_path=None, mode='tts', sts_refs=None, use_extreme=False, fish_voice_data=None):
    temp_dir = tempfile.mkdtemp()
    try:
        clips = []
        sfx_generator = None
        design_audio_tracker = {}
        design_cloned_prompts = {}
        sts_vc_obj = None
        if sts_refs:
            sts_vc_obj = SeedVCV2()
            if sts_vc_obj.model is None:
                print("Warning: Seed-VC v2 model failed to load, STS passes will be skipped")
                del sts_vc_obj
                sts_vc_obj = None
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
                    if use_extreme and isinstance(tts_vc_obj, FishTTS) and fish_voice_data and char_lower in fish_voice_data:
                        tts_vc_obj.encoded_refs = fish_voice_data[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                    else:
                        tts_vc_obj.voice_prompt = vc_voice_data[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                    if not success:
                        return False, f"Failed to synthesize line {num}"
                else:
                    if char_lower in design_cloned_prompts and tts_vc_obj is not None:
                        tts_vc_obj.voice_prompt = design_cloned_prompts[char_lower]
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
                        if tts_vc_obj is not None and char_lower not in design_cloned_prompts:
                            if char_lower not in design_audio_tracker:
                                design_audio_tracker[char_lower] = []
                            design_audio_tracker[char_lower].append(raw_file)
                            if len(design_audio_tracker[char_lower]) >= 3:
                                concat_path = os.path.join(temp_dir, f"design_clone_{char_lower}_{int(time.time())}.wav")
                                if _concat_audio_files(design_audio_tracker[char_lower], concat_path):
                                    extracted = svs_extract_vocals(concat_path)
                                    clone_src = extracted if extracted and extracted != concat_path else concat_path
                                    try:
                                        clone_success = tts_vc_obj.extract_voice(clone_src)
                                        if clone_success and tts_vc_obj.voice_prompt is not None:
                                            design_cloned_prompts[char_lower] = tts_vc_obj.voice_prompt
                                    except:
                                        pass
                                    if extracted and extracted != concat_path:
                                        try:
                                            os.unlink(extracted)
                                        except:
                                            pass
                if sts_vc_obj is not None and sts_refs and char_lower in sts_refs:
                    sts_ref_path = sts_refs[char_lower]
                    if os.path.exists(raw_file) and sts_ref_path and os.path.exists(sts_ref_path):
                        sts_out_file = os.path.join(temp_dir, f"sts_{i:03d}.wav")
                        try:
                            svs_raw = svs_extract_vocals(raw_file)
                            vc_source = svs_raw if svs_raw and svs_raw != raw_file else raw_file
                            sts_ok = sts_vc_obj.convert(vc_source, sts_ref_path, sts_out_file)
                            if sts_ok and os.path.exists(sts_out_file):
                                try:
                                    os.unlink(raw_file)
                                except:
                                    pass
                                shutil.move(sts_out_file, raw_file)
                                print(f"  STS pass applied for '{char}' line {num}")
                            else:
                                print(f"  Warning: STS pass failed for '{char}' line {num}, using TTS output")
                            if svs_raw and svs_raw != raw_file:
                                try:
                                    os.unlink(svs_raw)
                                except:
                                    pass
                        except Exception as e:
                            print(f"  Warning: STS pass error for '{char}' line {num}: {e}")
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
        if sts_vc_obj:
            del sts_vc_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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

    def synthesize(self, text, voice_instruct, output_path, language="Auto"):
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

    def synthesize_dialogue(self, dialogue_items, voice_prompts, output_path, language="Auto"):
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

    def synthesize(self, text, output_path, language="Auto"):
        if self.model is None or self.voice_prompt is None:
            return False
        try:
            import soundfile as sf
            import torch
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self.voice_prompt
            )
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"Synthesis error: {e}")
            return False

class FishTTS:
    def __init__(self, model_dir=None):
        self.model_dir = FISH_S2PRO_DIR if model_dir is None else model_dir
        self.model = None
        self.codec = None
        self.tokenizer = None
        self.decode_one_token = None
        self.device = None
        self.dtype = None
        self.encoded_refs = None
        os.makedirs(self.model_dir, exist_ok=True)

    def ensure_model(self):
        if self.model is not None:
            return True
        try:
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            os.makedirs(self.model_dir, exist_ok=True)
            if not os.path.exists(os.path.join(self.model_dir, "config.json")):
                print("Downloading Fish-S2Pro from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="fishaudio/s2-pro",
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
            print("Loading Fish-S2Pro model...")
            from fish_speech.models.text2semantic.inference import init_model, load_codec_model
            self.model, self.decode_one_token = init_model(
                checkpoint_path=self.model_dir,
                device=self.device,
                precision=self.dtype,
                compile=False
            )
            codec_path = os.path.join(self.model_dir, "codec.pth")
            if not os.path.exists(codec_path):
                codec_path = os.path.join(self.model_dir, "codecs", "codec.pth")
            self.codec = load_codec_model(codec_path, self.device, self.dtype)
            from fish_speech.tokenizer import FishTokenizer
            self.tokenizer = FishTokenizer.from_pretrained(self.model_dir)
            return True
        except Exception as e:
            print(f"Error loading Fish-S2Pro: {e}")
            return False

    def encode_voice(self, audio_path, ref_text=None):
        if not self.ensure_model():
            return None
        try:
            import torch
            from fish_speech.models.text2semantic.inference import encode_audio
            prompt_tokens = encode_audio(audio_path, self.codec, self.device)
            self.encoded_refs = {
                "tokens": prompt_tokens.cpu(),
                "text": ref_text or ""
            }
            return True
        except Exception as e:
            print(f"Voice encoding error: {e}")
            return None

    def synthesize(self, text, output_path, temperature=1.0, top_p=0.9, top_k=30):
        if not self.ensure_model():
            return False
        if self.encoded_refs is None:
            print("Error: No voice reference encoded for Fish TTS")
            return False
        try:
            import torch
            import soundfile as sf
            from fish_speech.models.text2semantic.inference import generate_long, decode_to_audio
            from fish_speech.conversation import Conversation, Message
            from fish_speech.content_sequence import TextPart, VQPart
            prompt_tokens = self.encoded_refs["tokens"].to(self.device)
            prompt_text = self.encoded_refs["text"]
            all_audio = []
            for response in generate_long(
                model=self.model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                text=text,
                prompt_text=[prompt_text],
                prompt_tokens=[prompt_tokens],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=0,
            ):
                if response.action == "sample":
                    audio = decode_to_audio(response.codes.to(self.device), self.codec)
                    if audio is not None:
                        all_audio.append(audio.cpu())
            if not all_audio:
                return False
            final_audio = torch.cat(all_audio, dim=-1)
            audio_np = final_audio.squeeze().numpy()
            sf.write(output_path, audio_np, 44100)
            return True
        except Exception as e:
            print(f"Fish TTS synthesis error: {e}")
            return False

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.codec is not None:
            del self.codec
            self.codec = None
        self.decode_one_token = None
        self.tokenizer = None
        self.encoded_refs = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

    def convert(self, source_path, reference_path, output_path, convert_style=False):
        if self.model is None:
            return False
        try:
            generator = self.model.convert_voice_with_streaming(
                source_audio_path=source_path,
                target_audio_path=reference_path,
                diffusion_steps=30,
                length_adjust=1.0,
                intelligebility_cfg_rate=0.7,
                similarity_cfg_rate=0.7,
                top_p=0.9,
                temperature=1.0,
                repetition_penalty=1.0,
                convert_style=convert_style,
                anonymization_only=False,
                device=torch.device(self.device),
                dtype=self.dtype,
                stream_output=True
            )
            full_audio = None
            for _, audio in generator:
                full_audio = audio
            if full_audio is not None:
                save_sr, audio_data = full_audio
                sf.write(output_path, audio_data, save_sr)
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

    def convert(self, source_path, reference_path, output_path, extract_vocals=False):
        if self.model is None:
            return False
        try:
            import librosa
            source_audio = librosa.load(source_path, sr=self.sr)[0]
            actual_reference_path = reference_path

            if extract_vocals:
                print("Extracting clean vocals from target audio...")
                import tempfile as _tf
                temp_vocals = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_vocals.close()
                try:
                    _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
                    if _bs_roformer_lib not in sys.path:
                        sys.path.insert(0, _bs_roformer_lib)
                    _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
                    if _bs_roformer_pkg not in sys.path:
                        sys.path.insert(0, _bs_roformer_pkg)
                    from bs_roformer import BSRoformerSeparator
                    _separator = BSRoformerSeparator(SVS_DIR)
                    _separator.ensure_model(stem='voice')
                    if _separator.vocals_model is not None:
                        _success = _separator.separate(reference_path, 'voice', temp_vocals.name)
                        if _success:
                            actual_reference_path = temp_vocals.name
                            print("Clean vocals extracted from target successfully")
                        else:
                            print("Warning: Vocal extraction failed, using original target")
                    else:
                        print("Warning: Could not load SVS model, using original target")
                    _separator.cleanup()
                    del _separator
                    _separator = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as _e:
                    print(f"Warning: Vocal extraction error: {_e}, using original target")

            ref_audio = librosa.load(actual_reference_path, sr=self.sr)[0]

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
            if extract_vocals and actual_reference_path != reference_path:
                try:
                    os.remove(actual_reference_path)
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"Seed-VC v1 conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

class AceStepWrapper:
    def __init__(self, use_overdose=False, complete_mode=False):
        self.checkpoints_dir = ACESTEP_DIR
        self.handler = None
        self.use_overdose = use_overdose
        self.complete_mode = complete_mode
        if complete_mode:
            self.config_path = "acestep-v15-xl-base"
            self.lm_model = "acestep-5Hz-lm-1.7B"
            self.shift = 1.0
        else:
            self.config_path = "acestep-v15-xl-turbo" if use_overdose else "acestep-v15-turbo"
            self.lm_model = "acestep-5Hz-lm-4B" if use_overdose else "acestep-5Hz-lm-1.7B"
            self.shift = 3.0 if use_overdose else 1.0
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        if self.handler is None:
            if self.complete_mode:
                single_gpu_gb, total_sys_gb = get_system_resources()
                if single_gpu_gb < 32.0 and total_sys_gb < 48.0:
                    print(f"Error: ACE-Step Complete (XL-Base) requires 32GB+ VRAM or 48GB+ combined System Memory (RAM+Swap/Pagefile)")
                    print(f"Detected: {single_gpu_gb:.1f}GB VRAM, {total_sys_gb:.1f}GB System Memory")
                    print("Cannot proceed with complete task.")
                    return
            elif self.use_overdose:
                single_gpu_gb, total_sys_gb = get_system_resources()
                if single_gpu_gb < 32.0 and total_sys_gb < 48.0:
                    print(f"Error: ACE-Step Overdose requires 32GB+ VRAM or 48GB+ combined System Memory (RAM+Swap/Pagefile)")
                    print(f"Detected: {single_gpu_gb:.1f}GB VRAM, {total_sys_gb:.1f}GB System Memory")
                    print("Falling back to Standard ACE-Step model...")
                    self.use_overdose = False
                    self.config_path = "acestep-v15-turbo"
                    self.lm_model = "acestep-5Hz-lm-1.7B"
                    self.shift = 1.0
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from acestep.handler import AceStepHandler
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                print(f"Loading ACE-Step model ({self.config_path})...")
                self.handler = AceStepHandler()
                status, success = self.handler.initialize_service(
                    project_root=self.checkpoints_dir,
                    config_path=self.config_path,
                    device=device
                )
                if not success:
                    print(f"Error initializing ACE-Step: {status}")
                    self.handler = None
            except Exception as e:
                print(f"Error loading ACE-Step model: {e}")
                self.handler = None

    def generate(self, lyrics, style_prompt, output_path, duration=10, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "text2music",
                "shift": self.shift,
            }
            if reference_audio is not None:
                _gen_kwargs["reference_audio"] = reference_audio
            result = self.handler.generate_music(**_gen_kwargs)
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

    def cover(self, src_audio, style_prompt, output_path, cover_strength=0.4, reference_audio=None, lyrics="..."):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": 30,
                "batch_size": 1,
                "task_type": "cover",
                "shift": self.shift,
                "src_audio": src_audio,
                "audio_cover_strength": cover_strength,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
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
            print(f"ACE-Step cover error: {e}")
            return False

    def repaint(self, src_audio, style_prompt, output_path, repaint_start, repaint_end, lyrics="...", cover_strength=0.4, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": 30,
                "batch_size": 1,
                "task_type": "repaint",
                "shift": self.shift,
                "src_audio": src_audio,
                "repainting_start": repaint_start,
                "repainting_end": repaint_end,
                "audio_cover_strength": cover_strength,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
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
            print(f"ACE-Step repaint error: {e}")
            return False

    def complete(self, src_audio, track_classes, output_path, styling=None, duration=None, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = "Complete the input track with " + " | ".join(t.upper() for t in track_classes) + ":"
            _gen_kwargs = {
                "captions": styling if styling else "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "complete",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
                "audio_cover_strength": 0.2,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
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
            print(f"ACE-Step complete error: {e}")
            return False

    def extract(self, src_audio, track_name, output_path, duration=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = f"Extract the {track_name.upper()} track from the audio:"
            _gen_kwargs = {
                "captions": "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "extract",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
            }
            result = self.handler.generate_music(**_gen_kwargs)
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
            print(f"ACE-Step extract error: {e}")
            return False

    def lego(self, src_audio, track_name, output_path, styling=None, duration=None, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = f"Generate the {track_name.upper()} track based on the audio context:"
            _gen_kwargs = {
                "captions": styling if styling else "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "lego",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
                "audio_cover_strength": 0.2,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
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
            print(f"ACE-Step lego error: {e}")
            return False

VALID_ACESTEP_TRACKS = {"woodwinds", "brass", "fx", "synth", "strings", "percussion",
                        "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"}
ACESTEP_INSTRUMENT_TRACKS = {"woodwinds", "brass", "fx", "synth", "strings", "percussion",
                              "keyboard", "guitar", "bass", "drums"}
ACESTEP_VOICE_TRACKS = {"vocals", "backing_vocals"}

def resolve_acestep_tracks(instruments_raw):
    track_classes = []
    use_everything = False
    use_instruments = False
    use_voices = False
    for t in instruments_raw.strip().split():
        tl = t.lower()
        if tl == 'everything':
            use_everything = True
        elif tl == 'instruments':
            use_instruments = True
        elif tl == 'voices':
            use_voices = True
        elif use_everything:
            continue
        else:
            track_classes.append(tl)
    if use_everything:
        return sorted(VALID_ACESTEP_TRACKS), None
    if use_instruments and use_voices:
        return sorted(VALID_ACESTEP_TRACKS), None
    if use_instruments:
        expansion = sorted(ACESTEP_INSTRUMENT_TRACKS)
        track_classes = [t for t in track_classes if t in ACESTEP_VOICE_TRACKS]
        track_classes = expansion + track_classes
    if use_voices:
        expansion = sorted(ACESTEP_VOICE_TRACKS)
        track_classes = [t for t in track_classes if t in ACESTEP_INSTRUMENT_TRACKS]
        track_classes = expansion + track_classes
    track_classes = list(dict.fromkeys(track_classes))
    unknown = [t for t in track_classes if t not in VALID_ACESTEP_TRACKS]
    if unknown:
        return None, unknown
    if not track_classes:
        return None, []
    return track_classes, None

def parse_ref_raw(raw):
    colon_idx = raw.find(':')
    if colon_idx == -1:
        return None, raw.strip()
    prefix = raw[:colon_idx].strip().lower()
    rest = raw[colon_idx + 1:].strip()
    if not rest:
        return None, raw.strip()
    if prefix in VALID_ACESTEP_TRACKS or prefix in ('everything', 'instruments', 'voices'):
        return prefix, rest
    return None, raw.strip()

def generate_background_music(ace_wrapper, music_description, total_duration, progress_callback=None, reference_audio=None):
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
            duration=int(total_duration),
            reference_audio=reference_audio
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
            duration=int(current_duration),
            reference_audio=reference_audio
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

SUPPORTED_TTS_LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian"
}

def _detect_text_language(text):
    import unicodedata
    script_counts = {}
    for ch in text:
        if ch.isspace() or ch in '.,!?;:\'"()-[]{}<>/\\@#$%^&*+=~`|0123456789':
            continue
        try:
            name = unicodedata.name(ch, '')
            if not name:
                continue
            if any(k in name for k in ('CJK', 'HIRAGANA', 'KATAKANA', 'HIRAGANA-KATAKANA', 'HALFWIDTH AND FULLWIDTH FORMS')):
                script_counts.setdefault('cjk', 0)
                script_counts['cjk'] += 1
            elif 'HANGUL' in name:
                script_counts.setdefault('hangul', 0)
                script_counts['hangul'] += 1
            elif 'CYRILLIC' in name:
                script_counts.setdefault('cyrillic', 0)
                script_counts['cyrillic'] += 1
            elif 'ARABIC' in name:
                script_counts.setdefault('arabic', 0)
                script_counts['arabic'] += 1
            elif 'DEVANAGARI' in name:
                script_counts.setdefault('devanagari', 0)
                script_counts['devanagari'] += 1
            elif 'THAI' in name:
                script_counts.setdefault('thai', 0)
                script_counts['thai'] += 1
            elif 'TAMIL' in name:
                script_counts.setdefault('tamil', 0)
                script_counts['tamil'] += 1
            elif 'TELUGU' in name:
                script_counts.setdefault('telugu', 0)
                script_counts['telugu'] += 1
            elif 'BENGALI' in name:
                script_counts.setdefault('bengali', 0)
                script_counts['bengali'] += 1
            elif 'LATIN' in name or name.startswith('FULLWIDTH LATIN'):
                script_counts.setdefault('latin', 0)
                script_counts['latin'] += 1
            elif 'GREEK' in name:
                script_counts.setdefault('greek', 0)
                script_counts['greek'] += 1
            elif 'HEBREW' in name:
                script_counts.setdefault('hebrew', 0)
                script_counts['hebrew'] += 1
            elif 'GEORGIAN' in name:
                script_counts.setdefault('georgian', 0)
                script_counts['georgian'] += 1
            elif 'ARMENIAN' in name:
                script_counts.setdefault('armenian', 0)
                script_counts['armenian'] += 1
            else:
                script_counts.setdefault('other', 0)
                script_counts['other'] += 1
        except Exception:
            continue
    if not script_counts:
        return "en", True
    dominant = max(script_counts, key=script_counts.get)
    qwen_supported_scripts = {'latin', 'cjk', 'hangul', 'cyrillic'}
    if dominant in qwen_supported_scripts:
        if dominant == 'cjk':
            ja_chars = sum(1 for ch in text if any(k in unicodedata.name(ch, '') for k in ('HIRAGANA', 'KATAKANA')))
            if ja_chars > 0:
                return "ja", True
            return "zh", True
        if dominant == 'hangul':
            return "ko", True
        if dominant == 'cyrillic':
            return "ru", True
        return "en", True
    return "other", False

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

def svs_extract_vocals(audio_path):
    try:
        print("Cleaning target audio through SVS voice pipe...")
        import tempfile as _tf
        temp_vocals = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_vocals.close()
        try:
            _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
            if _bs_roformer_lib not in sys.path:
                sys.path.insert(0, _bs_roformer_lib)
            _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
            if _bs_roformer_pkg not in sys.path:
                sys.path.insert(0, _bs_roformer_pkg)
            from bs_roformer import BSRoformerSeparator
            _separator = BSRoformerSeparator(SVS_DIR)
            _separator.ensure_model(stem='voice')
            _result = audio_path
            try:
                if _separator.vocals_model is not None:
                    _success = _separator.separate(audio_path, 'voice', temp_vocals.name)
                    if _success:
                        print("Target vocals cleaned successfully")
                        _result = temp_vocals.name
                    else:
                        print("Warning: SVS vocal extraction failed, using original target")
                else:
                    print("Warning: Could not load SVS model, using original target")
            finally:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _result
        except Exception as _e:
            print(f"Warning: SVS vocal extraction error: {_e}, using original target")
            try:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        try:
            os.unlink(temp_vocals.name)
        except:
            pass
        return audio_path
    except Exception as _e:
        print(f"Warning: SVS error: {_e}, using original target")
        return audio_path

def svs_extract_music(audio_path):
    try:
        print("Cleaning target audio through SVS music pipe...")
        import tempfile as _tf
        temp_music = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_music.close()
        try:
            _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
            if _bs_roformer_lib not in sys.path:
                sys.path.insert(0, _bs_roformer_lib)
            _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
            if _bs_roformer_pkg not in sys.path:
                sys.path.insert(0, _bs_roformer_pkg)
            from bs_roformer import BSRoformerSeparator
            _separator = BSRoformerSeparator(SVS_DIR)
            _separator.ensure_model(stem='music')
            _result = audio_path
            try:
                if _separator.inst_model is not None:
                    _success = _separator.separate(audio_path, 'music', temp_music.name)
                    if _success:
                        print("Target music cleaned successfully")
                        _result = temp_music.name
                    else:
                        print("Warning: SVS music extraction failed, using original target")
                else:
                    print("Warning: Could not load SVS music model, using original target")
            finally:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _result
        except Exception as _e:
            print(f"Warning: SVS music extraction error: {_e}, using original target")
            try:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        try:
            os.unlink(temp_music.name)
        except:
            pass
        return audio_path
    except Exception as _e:
        print(f"Warning: SVS error: {_e}, using original target")
        return audio_path

def ss_extract_speakers(audio_path, use_overdose=False):
    try:
        print("Running SS pipe for speaker separation...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        temp_results = tempfile.mkdtemp()
        outputs = _ss_run_pipeline(
            audio_path, False, temp_results, original_name, timestamp,
            target_path=None, use_overdose=use_overdose
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not outputs:
            try:
                shutil.rmtree(temp_results)
            except:
                pass
            return {}, None
        speaker_clips = {}
        for out_path in outputs:
            fname = os.path.basename(out_path)
            match = re.search(r'speaker(\d+)', fname)
            if match:
                spk_num = str(int(match.group(1)))
                speaker_clips[spk_num] = out_path
        return speaker_clips, temp_results
    except Exception as _e:
        print(f"Warning: SS pipe error: {_e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}, None

def resolve_target_to_audio(path):
    cleanup_files = []
    if is_youtube_url(path):
        print(f"Downloading audio from URL: {path}")
        success_dl, error_msg, audio_path = download_youtube_audio(path)
        if not success_dl:
            print(f"Error: {error_msg}")
            return None, cleanup_files
        cleanup_files.append(audio_path)
        return audio_path, cleanup_files
    if not os.path.exists(path):
        print(f"Error: Target not found: {path}")
        return None, cleanup_files
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        print("Extracting audio from video target...")
        extracted = extract_audio_from_video_cli(path)
        if not extracted:
            print("Error: Could not extract audio from video")
            return None, cleanup_files
        cleanup_files.append(extracted)
        return extracted, cleanup_files
    valid, msg = validate_audio_file(path)
    if not valid:
        print(f"Error: {msg}")
        return None, cleanup_files
    return path, cleanup_files

def download_youtube_video(url, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    try:
        import yt_dlp
        output_path = os.path.join(temp_dir, f"voder_svs_{int(time.time())}.mp4")
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': output_path,
            'merge_output_format': 'mp4',
            'quiet': False,
            'no_warnings': False,
        }
        print(f"Downloading video from: {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return None, "Failed to fetch video information"
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            print(f"Video: {title} ({duration}s)")
            ydl.download([url])
        if os.path.exists(output_path):
            print(f"Video downloaded: {output_path}")
            return output_path, title
        for ext in ['.mp4', '.mkv', '.webm']:
            alt = output_path.replace('.mp4', ext)
            if os.path.exists(alt):
                print(f"Video downloaded: {alt}")
                return alt, title
        return None, "Downloaded file not found"
    except ImportError:
        return None, "yt-dlp not installed"
    except Exception as e:
        return None, f"Download error: {str(e)}"

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

def analyze_dialogue_source(file_path, source_type="audio", use_overdose=False):
    if source_type == "txt":
        return True, None, None, None

    if source_type == "image":
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            return False, "Failed to load EasyOCR model", None, None

        print(f"Extracting text from image: {os.path.basename(file_path)}")
        success, text, error_msg = ocr.extract_text_from_image(file_path)

        ocr.cleanup()
        del ocr
        gc.collect()

        if not success:
            return False, error_msg or "Failed to extract text from image", None, None

        if not text:
            return False, "No text found in image", None, None

        dialogue_items = [(1, 'text', text)]
        return True, None, dialogue_items, None

    if source_type == "youtube":
        print(f"Downloading audio from YouTube...")
        success, error_msg, audio_path = download_youtube_audio(file_path)
        if not success:
            return False, error_msg, None, None

        file_path = audio_path

    audio_path = file_path
    needs_cleanup = False
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(file_path)
        if not audio_path:
            return False, "Failed to extract audio from video", None, None
        needs_cleanup = True
    elif not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        return False, f"Unsupported audio format: {file_path}", None, None

    try:
        if use_overdose:
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper + pyannote")
                asr.cleanup()
                del asr
                use_overdose = False
            else:
                print("Transcribing with VibeVoice ASR...")
                asr_segments = asr.transcribe(audio_path)
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not asr_segments:
                    return False, "VibeVoice ASR transcription returned no segments", None, None

                original_speakers = []
                for seg in asr_segments:
                    speaker = seg["speaker"]
                    if speaker not in original_speakers:
                        original_speakers.append(speaker)

                speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

                if len(original_speakers) == 1:
                    content = " ".join(seg["text"] for seg in asr_segments)
                    dialogue_items = [(1, 'text', content)]
                else:
                    dialogue_items = []
                    current_speaker_num = None
                    current_text_parts = []

                    for seg in asr_segments:
                        speaker_num = speaker_mapping[seg["speaker"]]
                        text = seg["text"]

                        if current_speaker_num is None:
                            current_speaker_num = speaker_num
                            current_text_parts = [text]
                        elif speaker_num == current_speaker_num:
                            current_text_parts.append(text)
                        else:
                            if current_text_parts:
                                content = " ".join(current_text_parts)
                                dialogue_items.append((current_speaker_num, str(current_speaker_num), content))
                            current_speaker_num = speaker_num
                            current_text_parts = [text]

                    if current_text_parts:
                        content = " ".join(current_text_parts)
                        dialogue_items.append((current_speaker_num, str(current_speaker_num), content))

                return True, None, dialogue_items, audio_path

        print("Loading Whisper model...")
        stt = WhisperSTT()
        if stt.model is None:
            return False, "Failed to load Whisper model", None, None

        print("Transcribing audio...")
        result = stt.transcribe(audio_path)

        del stt
        stt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not result:
            return False, "Transcription failed", None, None

        print("Performing speaker diarization...")
        diarization = SpeakerDiarization()

        if diarization.pipeline is None:
            text = result.get("text", "").strip()
            if not text:
                return False, "No text transcribed", None, None
            dialogue_items = [(1, 'text', text)]
            return True, None, dialogue_items, audio_path

        diar_result = diarization.diarize(audio_path)

        formatted_segments = diarization.format_diarization(diar_result, result)

        del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not formatted_segments:
            text = result.get("text", "").strip()
            dialogue_items = [(1, 'text', text)]
            return True, None, dialogue_items, audio_path

        original_speakers = []
        for seg in formatted_segments:
            speaker = seg["speaker"]
            if speaker not in original_speakers:
                original_speakers.append(speaker)

        speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

        if len(original_speakers) == 1:
            content = " ".join(seg["text"] for seg in formatted_segments)
            dialogue_items = [(1, 'text', content)]
        else:
            dialogue_items = []
            current_speaker_num = None
            current_text_parts = []
            current_start_time = None
            last_end_time = None

            for seg in formatted_segments:
                speaker_num = speaker_mapping[seg["speaker"]]
                text = seg["text"]

                if current_speaker_num is None:
                    current_speaker_num = speaker_num
                    current_text_parts = [text]
                    last_end_time = seg["end"]
                elif speaker_num == current_speaker_num:
                    current_text_parts.append(text)
                    last_end_time = seg["end"]
                else:
                    if current_text_parts:
                        content = " ".join(current_text_parts)
                        dialogue_items.append((current_speaker_num, str(current_speaker_num), content))
                    current_speaker_num = speaker_num
                    current_text_parts = [text]
                    current_start_time = seg["start"]
                    last_end_time = seg["end"]

            if current_text_parts:
                content = " ".join(current_text_parts)
                dialogue_items.append((current_speaker_num, str(current_speaker_num), content))

        return True, None, dialogue_items, audio_path

    except Exception as e:
        return False, f"Error analyzing audio: {str(e)}", None, None

def cli_tts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTS Mode ---")

    modify_speech = input("Want to modify speech? (Y/N): ").strip().lower()
    if modify_speech in ['y', 'yes']:
        while True:
            print("\nEnter the path to your audio/video source (file path or YouTube URL):")
            source_path = input("> ").strip()
            if not source_path:
                print("Error: No path provided")
                continue
            break

        _ms_cleanup = []
        audio_path = source_path
        needs_youtube = is_youtube_url(source_path)

        if needs_youtube:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(source_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _ms_cleanup.append(dl_path)
        elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(source_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _ms_cleanup.append(audio_path)
        elif not os.path.exists(source_path):
            print(f"Error: File not found: {source_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _ms_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        ms_overdose = False
        while True:
            overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
            if overdose_input in ['y', 'yes']:
                ms_overdose = True
                break
            elif overdose_input in ['n', 'no']:
                ms_overdose = False
                break
            else:
                print("Please enter Y or N")

        if ms_overdose:
            print("Loading VibeVoice ASR (overdose mode)...")
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                asr.cleanup()
                del asr
                ms_overdose = False
            else:
                try:
                    text = asr.transcribe_plain_text(clean_vocal)
                except Exception as e:
                    print(f"VibeVoice transcription error: {e}")
                    text = ""
                del asr
                asr = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not text or not text.strip():
                    print("Error: No speech detected (VibeVoice)")
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                text = text.strip()
                print(f"\nTranscribed text ({len(text)} chars):")
                display_text = text.replace('\n', '\\n').replace('\r', '\\r')
                print(display_text)
                print()

        if not ms_overdose:
            print("\nLoading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print("Transcribing audio...")
            result = stt.transcribe(clean_vocal)
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not result:
                print("Error: Transcription failed")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            text = result.get("text", "").strip()
            if not text:
                print("Error: No speech detected")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print(f"\nTranscribed text ({len(text)} chars):")
            display_text = text.replace('\n', '\\n').replace('\r', '\\r')
            print(display_text)
            print()

        edited_text = input("Edit text (or press Enter to keep as is): ").strip()
        if edited_text:
            text = edited_text.replace('\\n', '\n')
        if not text:
            print("Error: No text to synthesize")
            for f in _ms_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        use_source = input("Want to use source audio as voice reference? (Y/N): ").strip().lower()
        voice_ref = None
        ms_sts = False
        ms_sts_ref = None
        if use_source in ['y', 'yes']:
            voice_ref = clean_vocal
        else:
            while True:
                print("Enter voice reference path (prefix with sts: for enhanced voice conversion, use (path1)(path2) for multi-ref):")
                ref_path = input("> ").strip()
                if not ref_path:
                    print("Error: No path provided")
                    continue
                break
            if ref_path.lower().startswith('sts:'):
                ms_sts = True
                ref_path = ref_path[4:]
            multi = _parse_multi_refs(ref_path)
            if multi:
                voice_ref = _resolve_multi_refs(multi, _ms_cleanup)
                if not voice_ref:
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
            else:
                resolved_ref, _ref_cl = resolve_target_to_audio(ref_path)
                if not resolved_ref:
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                _ms_cleanup.extend(_ref_cl)
                voice_ref = svs_extract_vocals(resolved_ref)
                if voice_ref and voice_ref != resolved_ref:
                    _ms_cleanup.append(voice_ref)
                if resolved_ref not in _ms_cleanup and resolved_ref != voice_ref:
                    _ms_cleanup.append(resolved_ref)
            if ms_sts and voice_ref:
                ms_sts_ref = voice_ref

        preserve_nonvocals = False
        preserve_input = input("Preserve non-vocals? (Y/N): ").strip().lower()
        if preserve_input in ['y', 'yes']:
            preserve_nonvocals = True

        ms_music_track = None
        if preserve_nonvocals:
            ms_music_track = svs_extract_music(audio_path)
            if ms_music_track and ms_music_track != audio_path:
                _ms_cleanup.append(ms_music_track)
            else:
                ms_music_track = None

        try:
            print("\nLoading Qwen-TTS model...")
            tts = QwenTTS()
            print("Extracting voice characteristics...")
            success = tts.extract_voice(voice_ref)
            if not success:
                print("Error: Voice extraction failed")
                return False
            print("Synthesizing speech...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_tts_ms_{timestamp}.wav")
            success = tts.synthesize(text, output_path)
            if not success:
                print("Error: Synthesis failed")
                return False

            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if ms_sts and ms_sts_ref and os.path.exists(ms_sts_ref):
                print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
                vc = SeedVCV2()
                if vc.model is None:
                    print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                else:
                    svs_out = svs_extract_vocals(output_path)
                    if svs_out and svs_out != output_path:
                        _ms_cleanup.append(svs_out)
                        vc_input = svs_out
                    else:
                        vc_input = output_path
                    try:
                        sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sts_output = os.path.join(results_dir, f"voder_tts_ms_sts_{sts_timestamp}.wav")
                        sts_success = vc.convert(vc_input, ms_sts_ref, sts_output)
                        if sts_success:
                            print(f"✓ STS-converted output saved to: {sts_output}")
                            output_path = sts_output
                        else:
                            print("Warning: STS pass failed, using standard output")
                    finally:
                        del vc
                        vc = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if ms_music_track and os.path.exists(ms_music_track):
                print("\nBlending voice output with music track...")
                blend_timestamp = time.strftime("%Y%m%d_%H%M%S")
                blend_output = os.path.join(results_dir, f"voder_tts_ms_music_{blend_timestamp}.wav")
                blend_cmd = [
                    'ffmpeg', '-i', output_path, '-i', ms_music_track,
                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]',
                    '-map', '[out]', '-y', blend_output
                ]
                blend_result = subprocess.run(blend_cmd, capture_output=True, text=True)
                if blend_result.returncode == 0 and os.path.exists(blend_output):
                    print(f"✓ Blended output saved to: {blend_output}")
                    output_path = blend_output
                else:
                    print("Warning: Music blending failed, voice-only output preserved")

            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for f in _ms_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    use_overdose = False
    overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
    if overdose_input in ['y', 'yes']:
        use_overdose = True

    use_extreme = False
    extreme_input = input("Enable extreme? (Y/N): ").strip().lower()
    if extreme_input in ['y', 'yes']:
        use_extreme = True

    print("\nDo you have a dialogue source file? (audio/video/txt/image)")
    print("Press Y to provide a file, or N to enter manually")
    has_source = input("> ").strip().lower()

    dialogue_items = None
    mode_detected = None
    resolved_audio_path = None

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
                success, error_msg, items, _audio_path = analyze_dialogue_source(file_path, source_type="image", use_overdose=use_overdose)
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
                success, error_msg, items, _audio_path = analyze_dialogue_source(file_path, source_type="youtube", use_overdose=use_overdose)
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'
                if _audio_path:
                    resolved_audio_path = _audio_path

                print(f"\nDetected {len(items)} speaker(s):")
                for idx, speaker_num, content in dialogue_items:
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            else:
                print(f"\nAnalyzing {os.path.basename(file_path)}...")
                success, error_msg, items, _audio_path = analyze_dialogue_source(file_path, source_type="audio", use_overdose=use_overdose)
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'
                if _audio_path:
                    resolved_audio_path = _audio_path

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

        lines = [l.replace('\\n', '\n') for l in lines]

        if mode_detected == 'single':
            script = "\n".join(lines)
            print("Enter voice prompt (or audio/video/URL path to clone a voice, or trained voice name):")
            voice_prompt = input("> ").strip()
            if not voice_prompt:
                print("Error: No voice prompt provided")
                return False
            trained_file = _resolve_voice_ref(voice_prompt)
            if trained_file:
                print(f"Loading trained voice from: {trained_file}")
                voice_items = _load_voice_prompt(trained_file)
                if voice_items is None:
                    print(f"Error: Failed to load trained voice: {trained_file}")
                    return False
                print("Loading Qwen-TTS model...")
                tts = QwenTTS()
                if tts.model is None:
                    print("Error: Failed to load Qwen-TTS model")
                    return False
                tts.voice_prompt = voice_items
                print("Generating speech with trained voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts.synthesize(script, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                print(f"\n✓ Success! Output saved to: {output_path}")
                del tts
                return True
            if os.path.exists(voice_prompt) or is_youtube_url(voice_prompt):
                ref_paths = [voice_prompt]
                while True:
                    more = input("Additional reference? (path/URL, or Enter to finish): ").strip()
                    if not more:
                        break
                    if os.path.exists(more) or is_youtube_url(more):
                        ref_paths.append(more)
                    else:
                        print(f"Warning: File not found: {more}, skipping")
                _cleanup = []
                try:
                    if len(ref_paths) > 1:
                        clean_vocal = _resolve_multi_refs(ref_paths, _cleanup)
                        if not clean_vocal:
                            return False
                    else:
                        resolved_audio, _cl = resolve_target_to_audio(voice_prompt)
                        if not resolved_audio:
                            return False
                        _cleanup.extend(_cl)
                        clean_vocal = svs_extract_vocals(resolved_audio)
                        if clean_vocal and clean_vocal != resolved_audio:
                            _cleanup.append(clean_vocal)
                        if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                            _cleanup.append(resolved_audio)
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    success = tts.extract_voice(clean_vocal)
                    if not success:
                        print("Error: Voice extraction failed")
                        return False
                    print("Generating speech with cloned voice...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
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
                finally:
                    for f in _cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
            else:
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
                text = text.strip().replace('\\n', '\n')
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

        voice_prompts = {}
        target_assignments = {}
        trained_voice_refs = {}
        sts_refs = {}
        _dialogue_cleanup = []
        ss_clips = {}
        ss_temp_dir = None

        if not _is_all_sfx_interactive:
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

                    ss_source_audio = None
                    ss_source_cleanup = []

                    if is_youtube_url(file_path):
                        print(f"Downloading audio from YouTube...")
                        _dl_ok, _dl_err, _dl_path = download_youtube_audio(file_path)
                        if not _dl_ok:
                            print(f"Error: {_dl_err}")
                            retry = input("Try another source? (Y/N): ").strip().lower()
                            if retry not in ['y', 'yes']:
                                return False
                            continue
                        ss_source_audio = _dl_path
                        ss_source_cleanup.append(_dl_path)
                    elif os.path.exists(file_path):
                        ss_source_audio = file_path
                        if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                            print("Extracting audio from video...")
                            ss_source_audio = extract_audio_from_video_cli(file_path)
                            if not ss_source_audio:
                                print("Error: Failed to extract audio from video")
                                retry = input("Try another source? (Y/N): ").strip().lower()
                                if retry not in ['y', 'yes']:
                                    return False
                                continue
                            ss_source_cleanup.append(ss_source_audio)
                    else:
                        print(f"Error: File not found: {file_path}")
                        retry = input("Try another source? (Y/N): ").strip().lower()
                        if retry not in ['y', 'yes']:
                            return False
                        continue

                    print(f"\nExtracting speaker clips via SS pipe...")
                    ss_clips, ss_temp_dir = ss_extract_speakers(ss_source_audio, use_overdose=use_overdose)

                    if ss_clips:
                        print(f"\nExtracted {len(ss_clips)} speaker clip(s) via SS pipe.")
                        for spk_num in sorted(ss_clips.keys(), key=lambda x: int(x)):
                            print(f"  Speaker {spk_num}: {os.path.basename(ss_clips[spk_num])}")
                    else:
                        print("SS pipe returned no speaker clips.")

                    if ss_temp_dir:
                        _dialogue_cleanup.append(ss_temp_dir)
                    for _cf in ss_source_cleanup:
                        _dialogue_cleanup.append(_cf)

                    for i, char_lower in enumerate(sorted_chars):
                        orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                        if char_lower in ss_clips:
                            clip_path = ss_clips[char_lower]
                            clean_vocal = svs_extract_vocals(clip_path)
                            if clean_vocal and clean_vocal != clip_path:
                                _dialogue_cleanup.append(clean_vocal)
                                target_assignments[char_lower] = clean_vocal
                            else:
                                target_assignments[char_lower] = clip_path
                            print(f"  {orig_char} -> speaker {char_lower} (SS pipe)")
                        else:
                            first_path = None
                            ref_paths = []
                            while True:
                                label = f"{orig_char} reference 1" if first_path is None else f"{orig_char} reference (Enter to finish)"
                                path = input(f"{label}: ").strip()
                                if not path:
                                    if first_path is None:
                                        print(f"Warning: At least one reference required for {orig_char}")
                                        continue
                                    break
                                if not os.path.exists(path) and not is_youtube_url(path):
                                    print(f"Warning: File not found: {path}, skipping")
                                    continue
                                if first_path is None:
                                    first_path = path
                                    ref_paths = [path]
                                else:
                                    ref_paths.append(path)
                            if len(ref_paths) > 1:
                                clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                                if not clean_vocal:
                                    return False
                            else:
                                resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
                                if not resolved_audio:
                                    return False
                                _dialogue_cleanup.extend(_cl)
                                clean_vocal = svs_extract_vocals(resolved_audio)
                                if clean_vocal and clean_vocal != resolved_audio:
                                    _dialogue_cleanup.append(clean_vocal)
                            target_assignments[char_lower] = clean_vocal
                            print(f"  {orig_char} -> manual ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")

                    break
            else:
                print(f"\nVoice prompts or audio file paths for {len(chars)} character(s):")
                print("(Enter text for voice prompt, a path/URL to clone a voice, or a trained voice name)")
                for i, char_lower in enumerate(sorted_chars):
                    orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                    prompt = input(f"{orig_char}: ").strip()
                    if not prompt:
                        print(f"Error: No voice prompt or audio path provided for {orig_char}")
                        return False
                    trained_file = _resolve_voice_ref(prompt)
                    if trained_file:
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        trained_voice_refs[char_lower] = trained_file
                        print(f"  {orig_char} -> trained voice ({os.path.basename(trained_file)})")
                    elif os.path.exists(prompt) or is_youtube_url(prompt):
                        ref_paths = [prompt]
                        ref_num = 2
                        while True:
                            more = input(f"{orig_char} reference {ref_num} (Enter to finish): ").strip()
                            if not more:
                                break
                            if os.path.exists(more) or is_youtube_url(more):
                                ref_paths.append(more)
                                ref_num += 1
                            else:
                                print(f"Warning: File not found: {more}, skipping")
                        if len(ref_paths) > 1:
                            clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                            if not clean_vocal:
                                return False
                        else:
                            resolved_audio, _cl = resolve_target_to_audio(prompt)
                            if not resolved_audio:
                                return False
                            _dialogue_cleanup.extend(_cl)
                            clean_vocal = svs_extract_vocals(resolved_audio)
                            if clean_vocal and clean_vocal != resolved_audio:
                                _dialogue_cleanup.append(clean_vocal)
                        target_assignments[char_lower] = clean_vocal
                        print(f"  {orig_char} -> voice clone ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")
                    else:
                        voice_prompts[char_lower] = prompt
                    print(f"Progress: {i+1}/{len(chars)} completed")

        has_tts_chars = len(voice_prompts) > 0
        has_vc_chars = len(target_assignments) > 0 or len(trained_voice_refs) > 0

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
            ref_input = input("Music reference (path/URL, or press Enter to skip): ").strip()
            if ref_input:
                if not os.path.exists(ref_input) and not is_youtube_url(ref_input):
                    print("Error: Music reference not found: " + ref_input)
                    return False
        else:
            ref_input = None

        music_reference_audio = None
        music_ref_cleanup = []
        if ref_input and music_description:
            print("Resolving music reference source...")
            resolved_ref, ref_cl = resolve_target_to_audio(ref_input)
            if not resolved_ref:
                return False
            music_ref_cleanup.extend(ref_cl)
            print("Extracting clean music from reference via SVS...")
            music_reference_audio = svs_extract_music(resolved_ref)
            if music_reference_audio and music_reference_audio != resolved_ref and music_reference_audio not in music_ref_cleanup:
                music_ref_cleanup.append(music_reference_audio)

        try:
            tts_design = None
            if has_tts_chars:
                print("\nLoading Qwen-TTS VoiceDesign model...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False

            tts_obj = None
            vc_voice_prompts = None
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
                for char_lower, trained_file in trained_voice_refs.items():
                    print(f"Loading trained voice for '{char_lower}' from: {trained_file}")
                    voice_items = _load_voice_prompt(trained_file)
                    if voice_items is None:
                        print(f"Error: Failed to load trained voice: {trained_file}")
                        return False
                    vc_voice_prompts[char_lower] = voice_items

            if has_tts_chars and tts_obj is None:
                print("Loading Qwen-TTS model for voice stabilization...")
                tts_obj = QwenTTS()

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

            if has_sfx or has_effects or has_vc_chars or has_tts_chars:
                success, msg = _assemble_enhanced_dialogue(
                    dialogue_items, voice_prompts, tts_design_obj=tts_design,
                    tts_vc_obj=tts_obj, vc_voice_data=vc_voice_prompts,
                    output_path=dialogue_temp.name, mode='tts',
                    sts_refs=sts_refs if sts_refs else None
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
                if tts_obj is not None:
                    del tts_obj
                    tts_obj = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ace = AceStepWrapper(use_overdose=use_overdose)
                if ace.handler is None:
                    print("Error: Failed to load ACE-Step model")
                    return False
                success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec, reference_audio=music_reference_audio)
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
            if tts_obj is not None:
                del tts_obj
                tts_obj = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for f in music_ref_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            for f in _dialogue_cleanup:
                if f and os.path.exists(f):
                    try:
                        if os.path.isdir(f):
                            shutil.rmtree(f)
                        else:
                            os.unlink(f)
                    except:
                        pass
            if 'dialogue_temp' in dir() and os.path.exists(dialogue_temp.name):
                try:
                    os.unlink(dialogue_temp.name)
                except:
                    pass

def cli_stt_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- STT Mode ---")
    print("Speech-to-Text (Transcription)")
    print()

    while True:
        translate_input = input("Translate to English? (Y/N): ").strip().lower()
        if translate_input in ['y', 'yes']:
            enable_translate = True
            break
        elif translate_input in ['n', 'no']:
            enable_translate = False
            break
        else:
            print("Please enter Y or N")

    while True:
        file_path = input("Enter audio/video file path or YouTube URL: ").strip()
        if not file_path:
            print("Error: No path provided")
            continue
        if os.path.exists(file_path) or is_youtube_url(file_path):
            break
        print("Error: File not found or invalid URL")

    while True:
        timestamp_input = input("Keep timestamps? (Y/N): ").strip().lower()
        if timestamp_input in ['y', 'yes']:
            keep_timestamp = True
            break
        elif timestamp_input in ['n', 'no']:
            keep_timestamp = False
            break
        else:
            print("Please enter Y or N")

    while True:
        dialogue_input = input("Enable speaker diarization? (Y/N): ").strip().lower()
        if dialogue_input in ['y', 'yes']:
            enable_dialogue = True
            break
        elif dialogue_input in ['n', 'no']:
            enable_dialogue = False
            break
        else:
            print("Please enter Y or N")

    use_overdose = False
    if not enable_translate:
        while True:
            overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
            if overdose_input in ['y', 'yes']:
                use_overdose = True
                break
            elif overdose_input in ['n', 'no']:
                use_overdose = False
                break
            else:
                print("Please enter Y or N")

    audio_path = file_path
    needs_cleanup = False
    is_youtube = is_youtube_url(file_path)

    if is_youtube:
        print("Downloading audio from YouTube...")
        success_dl, error_msg, audio_path = download_youtube_audio(file_path)
        if not success_dl:
            print(f"Error: {error_msg}")
            return False
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        extracted = extract_audio_from_video_cli(file_path)
        if not extracted:
            print(f"Error: Could not extract audio from {file_path}")
            return False
        audio_path = extracted
        needs_cleanup = True

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)
    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    print("Stage 1: SVS voice isolation (BS-RoFormer)...")
    from bs_roformer import BSRoformerSeparator
    svs_separator = BSRoformerSeparator(SVS_DIR)
    svs_separator.ensure_model(stem='voice')
    if svs_separator.vocals_model is None:
        print("Error: Failed to load BS-RoFormer vocals model")
        svs_separator.cleanup()
        del svs_separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    svs_temp_dir = tempfile.mkdtemp()
    svs_temp = os.path.join(svs_temp_dir, f'_cli_stt_svs_{timestamp}.wav')
    svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
    svs_separator.cleanup()
    del svs_separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if svs_ok and os.path.exists(svs_temp):
        if needs_cleanup and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        audio_path = svs_temp
        needs_cleanup = True
    else:
        print("Warning: SVS voice isolation failed, using original audio")
        shutil.rmtree(svs_temp_dir, ignore_errors=True)

    try:
        if use_overdose and not enable_translate:
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                asr.cleanup()
                del asr
                use_overdose = False

        if use_overdose and not enable_translate:
            print("Transcribing with VibeVoice ASR...")
            asr_segments = asr.transcribe(audio_path)
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not asr_segments:
                print("Error: ASR transcription returned no segments")
                return False

            def format_time_range(start, end):
                def format_single(seconds):
                    if seconds is None:
                        seconds = 0
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 100)
                    return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                return f"[{format_single(start)}-{format_single(end)}]"

            if enable_dialogue:
                original_speakers = []
                for seg in asr_segments:
                    speaker = seg["speaker"]
                    if speaker not in original_speakers:
                        original_speakers.append(speaker)
                speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}
                lines = []
                current_speaker_num = None
                current_text_parts = []
                current_first_time = None
                current_last_time = None
                for seg in asr_segments:
                    speaker_num = speaker_mapping[seg["speaker"]]
                    text = seg.get("text", "")
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
                        if current_text_parts:
                            content_out = " ".join(current_text_parts)
                            if len(original_speakers) == 1:
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content_out}")
                                else:
                                    lines.append(f"text: {content_out}")
                            else:
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                else:
                                    lines.append(f"{current_speaker_num}: {content_out}")
                        current_speaker_num = speaker_num
                        current_text_parts = [text]
                        current_first_time = seg_start
                        current_last_time = seg_end
                if current_text_parts:
                    content_out = " ".join(current_text_parts)
                    if len(original_speakers) == 1:
                        if keep_timestamp:
                            lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content_out}")
                        else:
                            lines.append(f"text: {content_out}")
                    else:
                        if keep_timestamp:
                            lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                        else:
                            lines.append(f"{current_speaker_num}: {content_out}")
                formatted_text = "\n".join(lines)
            elif keep_timestamp:
                lines = []
                for seg in asr_segments:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    text = seg.get("text", "").strip()
                    if text:
                        lines.append(f"{format_time_range(start, end)} text: {text}")
                if lines:
                    formatted_text = "\n".join(lines)
                else:
                    formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
            else:
                formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
        else:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                return False

            if enable_translate and enable_dialogue:
                print("Transcribing audio (for diarization)...")
                original_result = stt.transcribe(audio_path)
                if not original_result:
                    print("Error: Transcription failed")
                    return False

                print("Translating audio to English...")
                result = stt.translate(audio_path)
                if not result:
                    print("Error: Translation failed, using original transcription")
                    result = original_result
                    enable_translate = False

                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif enable_translate:
                print("Translating audio to English...")
                result = stt.translate(audio_path)
                if not result:
                    print("Error: Translation failed")
                    return False

                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("Transcribing audio...")
                result = stt.transcribe(audio_path)
                if not result:
                    print("Error: Transcription failed")
                    return False

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

        if not use_overdose:
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
                    if enable_translate:
                        diarization_segments = diarization.format_diarization(diar_result, original_result)
                    else:
                        diarization_segments = diarization.format_diarization(diar_result, result)

                    formatted_segments = None
                    if diarization_segments:
                        if enable_translate:
                            translated_segments = result.get("segments", [])
                            speaker_time_map = []
                            for ds in diarization_segments:
                                speaker_time_map.append({
                                    "speaker": ds["speaker"],
                                    "start": ds.get("start", 0),
                                    "end": ds.get("end", 0),
                                    "text": ds["text"]
                                })

                            merged_segments = []
                            for ts in translated_segments:
                                ts_start = ts.get("start", 0)
                                ts_end = ts.get("end", 0)
                                ts_text = ts.get("text", "").strip()
                                if not ts_text:
                                    continue
                                best_speaker = None
                                best_overlap = 0
                                for sm in speaker_time_map:
                                    overlap_start = max(ts_start, sm["start"])
                                    overlap_end = min(ts_end, sm["end"])
                                    overlap = max(0, overlap_end - overlap_start)
                                    if overlap > best_overlap:
                                        best_overlap = overlap
                                        best_speaker = sm["speaker"]
                                if best_speaker is not None:
                                    merged_segments.append({
                                        "speaker": best_speaker,
                                        "start": ts_start,
                                        "end": ts_end,
                                        "text": ts_text
                                    })
                            formatted_segments = merged_segments if merged_segments else None
                        else:
                            formatted_segments = diarization_segments

                    if formatted_segments:
                        original_speakers = []
                        for seg in formatted_segments:
                            speaker = seg["speaker"]
                            if speaker not in original_speakers:
                                original_speakers.append(speaker)

                        speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

                        if len(original_speakers) == 1:
                            content_out = " ".join(seg["text"] for seg in formatted_segments)
                            if keep_timestamp:
                                first_time = formatted_segments[0]["start"]
                                last_time = formatted_segments[-1]["end"]
                                formatted_text = f"{format_time_range(first_time, last_time)} text: {content_out}"
                            else:
                                formatted_text = f"text: {content_out}"
                        else:
                            lines = []
                            current_speaker_num = None
                            current_text_parts = []
                            current_first_time = None
                            current_last_time = None

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
                                    if current_text_parts:
                                        content_out = " ".join(current_text_parts)
                                        if keep_timestamp:
                                            lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                        else:
                                            lines.append(f"{current_speaker_num}: {content_out}")
                                    current_speaker_num = speaker_num
                                    current_text_parts = [text]
                                    current_first_time = seg_start
                                    current_last_time = seg_end

                            if current_text_parts:
                                content_out = " ".join(current_text_parts)
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                else:
                                    lines.append(f"{current_speaker_num}: {content_out}")

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

        print("\n" + formatted_text)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if is_youtube:
            base_name = "youtube_stt"
        else:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

        suffix_parts = ["stt"]
        if enable_translate:
            suffix_parts.append("translate")
        if keep_timestamp:
            suffix_parts.append("timestamp")
        if enable_dialogue:
            suffix_parts.append("dialogue")
        suffix = "_".join(suffix_parts)

        output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
        output_path = os.path.join(results_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        print(f"\n\u2713 Success! Output saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        if needs_cleanup and audio_path != file_path and os.path.exists(audio_path):
            try:
                parent_dir = os.path.dirname(audio_path)
                os.unlink(audio_path)
                if os.path.exists(parent_dir) and os.path.basename(parent_dir).startswith('_'):
                    shutil.rmtree(parent_dir, ignore_errors=True)
            except:
                pass
        if is_youtube and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass

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
    base_is_video = os.path.splitext(base_path)[1].lower() in VIDEO_EXTENSIONS
    base_original = base_path
    temp_base_extracted = None
    if base_is_video:
        print("Extracting audio from video...")
        temp_base_extracted = extract_audio_from_video_cli(base_path)
        if not temp_base_extracted:
            print("Error: Could not extract audio from video")
            return False
        base_path = temp_base_extracted
    print()
    target_path = input("Enter target voice audio/video path or URL: ").strip()
    if not target_path:
        print("Error: No target path provided")
        return False
    resolved_target, _target_cleanup = resolve_target_to_audio(target_path)
    if not resolved_target:
        return False
    target_path = resolved_target
    print()
    no_music = False
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
    if not is_music:
        while True:
            nomusic_input = input("Output voice only without music? (Y/N): ").strip().lower()
            if nomusic_input in ['y', 'yes']:
                no_music = True
                break
            elif nomusic_input in ['n', 'no']:
                no_music = False
                break
            else:
                print("Please enter Y or N")
    if is_music:
        if base_is_video:
            print("Error: Base input must be audio for MSTS mode")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        print("Extracting music from source...")
        base_music = svs_extract_music(base_path)
        _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        clean_vocal_target = svs_extract_vocals(target_path)
        _target_cleanup.append(clean_vocal_target)
        print("Resampling inputs to 44100Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 44100:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    else:
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        base_music = None
        if not no_music:
            print("Extracting music from source...")
            base_music = svs_extract_music(base_path)
            _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        clean_vocal_target = svs_extract_vocals(target_path)
        _target_cleanup.append(clean_vocal_target)
        print("\nLoading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Resampling inputs to 22050Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 22050:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 22050)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
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
            temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_output_44k.name, waveform_out, 44100)
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not no_music and base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            elif base_is_video:
                print("Merging converted audio with video...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.mp4")
                ret = os.system(f'ffmpeg -y -i "{base_original}" -i "{temp_output_44k.name}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Error: Failed to merge audio with video")
                    return False
            else:
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_22k.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

def cli_ttm_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTM Mode ---")
    print("Generate music from lyrics and style")
    print()
    use_overdose = False
    while True:
        overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
        if overdose_input in ['y', 'yes']:
            use_overdose = True
            break
        elif overdose_input in ['n', 'no']:
            use_overdose = False
            break
        else:
            print("Please enter Y or N")
    use_vc = False
    while True:
        vc_input = input("Want to clone a voice? (Y/N): ").strip().lower()
        if vc_input in ['y', 'yes']:
            use_vc = True
            break
        elif vc_input in ['n', 'no']:
            use_vc = False
            break
        else:
            print("Please enter Y or N")
    if use_vc:
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
        clone_input = input("Enter source to clone from (audio/video/URL): ").strip()
        if not clone_input:
            print("Error: No clone source provided")
            return False
        while not (os.path.exists(clone_input) or is_youtube_url(clone_input)):
            print(f"Error: Clone source not found: {clone_input}")
            clone_input = input("Enter source to clone from (audio/video/URL): ").strip()
            if not clone_input:
                print("Error: No clone source provided")
                return False
        _vc_cleanup = []
        resolved_audio, cleanup = resolve_target_to_audio(clone_input)
        if resolved_audio is None:
            print("Error: Could not resolve clone source")
            return False
        _vc_cleanup.extend(cleanup)
        clean_vocal = svs_extract_vocals(resolved_audio)
        if clean_vocal != resolved_audio and clean_vocal not in _vc_cleanup:
            _vc_cleanup.append(clean_vocal)
        if resolved_audio not in _vc_cleanup and resolved_audio != clean_vocal:
            _vc_cleanup.append(resolved_audio)
        print("\nLoading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_ttm_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_clone_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
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
            print("Extracting vocals from TTM output...")
            ttm_vocals = svs_extract_vocals(temp_ttm_output.name)
            if ttm_vocals and ttm_vocals != temp_ttm_output.name:
                _vc_cleanup.append(ttm_vocals)
            else:
                ttm_vocals = temp_ttm_output.name
            print("Extracting music from TTM output...")
            ttm_music = svs_extract_music(temp_ttm_output.name)
            if ttm_music and ttm_music != temp_ttm_output.name:
                _vc_cleanup.append(ttm_music)
            else:
                ttm_music = None
            print("Resampling TTM vocals to 44100Hz...")
            waveform_vocals, sr_vocals = torchaudio.load(ttm_vocals)
            if sr_vocals != 44100:
                resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
                waveform_vocals = resampler_vocals(waveform_vocals)
            torchaudio.save(temp_ttm_44k.name, waveform_vocals, 44100)
            print("Resampling clone voice to 44100Hz...")
            waveform_clone, sr_clone = torchaudio.load(clean_vocal)
            if sr_clone != 44100:
                resampler_clone = torchaudio.transforms.Resample(sr_clone, 44100)
                waveform_clone = resampler_clone(waveform_clone)
            torchaudio.save(temp_clone_44k.name, waveform_clone, 44100)
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
                source_path=temp_ttm_44k.name,
                reference_path=temp_clone_44k.name,
                output_path=temp_vc_output.name
            )
            if not vc_success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ttm_music:
                print("Mixing converted vocals with TTM music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_vc_output.name}" -i "{ttm_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_vc_output.name, output_path)
            else:
                print("Saving output...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                shutil.copy(temp_vc_output.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_ttm_output.name, temp_ttm_44k.name, temp_clone_44k.name, temp_vc_output.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
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
    _ttm_cleanup = []
    reference_audio = None
    print()
    ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
    while ref_input:
        if not os.path.exists(ref_input) and not is_youtube_url(ref_input):
            print(f"Error: Reference target not found: {ref_input}")
            ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
            continue
        while True:
            ref_choice = input("Reference type: 1 for voice, 2 for music: ").strip()
            if ref_choice == '1':
                ref_type = 'voice'
                break
            elif ref_choice == '2':
                ref_type = 'music'
                break
            else:
                print("Error: Please enter 1 (voice) or 2 (music)")
        resolved_audio, cleanup = resolve_target_to_audio(ref_input)
        if resolved_audio is None:
            print("Error: Could not resolve reference target")
            ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
            continue
        _ttm_cleanup.extend(cleanup)
        if ref_type == 'voice':
            processed = svs_extract_vocals(resolved_audio)
        else:
            processed = svs_extract_music(resolved_audio)
        if processed != resolved_audio and processed not in _ttm_cleanup:
            _ttm_cleanup.append(processed)
        if resolved_audio not in _ttm_cleanup and resolved_audio != processed:
            _ttm_cleanup.append(resolved_audio)
        reference_audio = processed
        break
    print("\nLoading ACE-Step model...")
    ace_step = AceStepWrapper(use_overdose=use_overdose)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    try:
        print(f"Generating music ({duration}s duration)...")
        if reference_audio:
            print(f"Using reference audio: {reference_audio}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=output_path,
            duration=duration,
            reference_audio=reference_audio
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
    finally:
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def parse_oneline_args(args):
    if not args:
        return {'error': 'No arguments provided'}
    mode = args[0].lower()
    result = {'mode': mode, 'params': {}, 'error': None, 'is_music': False, 'is_mimic': False, 'nomusic': False}
    valid_keywords = ['script', 'voice', 'lyrics', 'styling', 'base', 'target', 'duration', 'timestamp', 'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr', 'reference', 'music']
    i = 1
    current_keyword = None
    result_path = None

    if mode == 'stt':
        file_paths = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower in ['timestamp', 'dialogue', 'translate', 'se', 'overdose']:
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

        if result['params'].get('overdose') and result['params'].get('translate'):
            result['error'] = 'STT overdose cannot be used with translate (ASR does not support translation)'
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
            elif os.path.exists(arg) or is_youtube_url(arg):
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

    if mode == 'svs':
        stem = None
        file_path = None
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
            elif arg_lower in ('voice', 'music', 'both'):
                stem = arg_lower
                i += 1
            elif file_path is None and (os.path.exists(arg) or is_youtube_url(arg)):
                file_path = arg
                i += 1
            else:
                result['error'] = f'Invalid argument: {arg}'
                return result
        if stem is None:
            result['error'] = 'SVS mode requires stem: voice or music'
            return result
        if file_path is None:
            result['error'] = 'SVS mode requires an audio file path'
            return result
        result['params']['stem'] = stem
        result['params']['file_path'] = file_path
        result['params']['result_path'] = result_path
        return result

    if mode == 'ss':
        use_se = False
        file_path = None
        target_path = None
        use_overdose = False
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'se':
                use_se = True
                i += 1
            elif arg_lower == 'overdose':
                use_overdose = True
                i += 1
            elif arg_lower == 'extreme':
                use_extreme = True
                i += 1
            elif arg_lower == 'target':
                if i + 1 < len(args):
                    target_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'target keyword requires a path argument'
                    return result
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif file_path is None and (os.path.exists(arg) or is_youtube_url(arg)):
                file_path = arg
                i += 1
            else:
                result['error'] = f'Invalid argument: {arg}'
                return result
        if file_path is None:
            result['error'] = 'SS mode requires an audio/video file path or URL'
            return result
        result['params']['use_se'] = use_se
        result['params']['file_path'] = file_path
        result['params']['target_path'] = target_path
        result['params']['overdose'] = use_overdose
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

    if mode == 'train':
        sub_type = None
        voice_name = None
        ref_paths = []
        test_script = None
        has_test = False
        use_first = False
        use_extreme = False
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower.startswith('voice:'):
                sub_type = 'voice'
                voice_name = arg[6:].strip()
                i += 1
            elif arg_lower == 'extreme':
                use_extreme = True
                i += 1
            elif arg_lower == 'first':
                use_first = True
                i += 1
            elif arg_lower == 'test':
                has_test = True
                if i + 1 < len(args) and not args[i + 1].lower().startswith('voice:') and not args[i + 1].lower() == 'test' and args[i + 1].lower() != 'first':
                    test_script = args[i + 1]
                    i += 2
                else:
                    test_script = None
                    i += 1
            elif os.path.exists(arg) or is_youtube_url(arg):
                ref_paths.append(arg)
                i += 1
            else:
                ref_paths.append(arg)
                i += 1
        if sub_type != 'voice':
            result['error'] = 'Train mode requires voice: sub-type (e.g., train voice:character-name "path")'
            return result
        if not voice_name:
            result['error'] = 'Train mode requires a character name after voice: (e.g., voice:james)'
            return result
        if not ref_paths:
            result['error'] = 'Train mode requires at least one reference audio path'
            return result
        result['params']['sub_type'] = sub_type
        result['params']['voice_name'] = voice_name
        result['params']['ref_paths'] = ref_paths
        result['params']['has_test'] = has_test
        result['params']['test_script'] = test_script
        result['params']['use_first'] = use_first
        result['params']['extreme'] = use_extreme
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
        elif arg_lower == 'overdose':
            result['params']['overdose'] = True
            i += 1
        elif arg_lower == 'extreme':
            if mode in ('tts', 'sts'):
                result['params']['extreme'] = True
            else:
                print("Warning: 'extreme' keyword is only valid in TTS and STS modes, ignoring")
            i += 1
        elif mode == 'ttm' and arg_lower == 'complete':
            result['params']['complete'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'lego':
            result['params']['lego'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'extract':
            result['params']['extract'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'bgm':
            if i + 1 >= len(args):
                result['error'] = 'bgm requires a source path (audio/video file or URL)'
                return result
            result['params']['bgm'] = True
            result['params']['bgm_source'] = args[i + 1]
            i += 2
        elif mode == 'ttm' and arg_lower == 'voice':
            if 'complete' not in result['params'] and 'lego' not in result['params']:
                result['error'] = 'voice keyword is only valid with complete/lego task'
                return result
            if result['params'].get('use_music'):
                result['error'] = 'voice and music cannot be used together, use one or the other'
                return result
            result['params']['use_vocals'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'music' and 'bgm' not in result['params']:
            if 'complete' not in result['params'] and 'lego' not in result['params']:
                result['error'] = 'music keyword is only valid with complete/lego/bgm task'
                return result
            if result['params'].get('use_vocals'):
                result['error'] = 'voice and music cannot be used together, use one or the other'
                return result
            result['params']['use_music'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'video':
            if 'complete' not in result['params'] and 'bgm' not in result['params']:
                result['error'] = 'video keyword is only valid with complete/bgm task'
                return result
            result['params']['want_video'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'noblend':
            if 'complete' not in result['params']:
                result['error'] = 'noblend keyword is only valid with complete task'
                return result
            result['params']['noblend'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'usrc':
            if 'complete' not in result['params']:
                result['error'] = 'usrc keyword is only valid with complete task'
                return result
            result['params']['use_source'] = True
            i += 1
        elif mode == 'ttm' and (arg.startswith('sfx:') or (arg.startswith('"sfx:') and arg.endswith('"'))):
            if 'bgm' not in result['params'] and 'complete' not in result['params']:
                result['error'] = '"sfx:" specs are only valid with bgm or complete task'
                return result
            if 'complete' in result['params'] and result['params'].get('noblend'):
                result['error'] = '"sfx:" cannot be used with noblend'
                return result
            if 'sfx_specs' not in result['params']:
                result['params']['sfx_specs'] = []
            sfx_val = arg.strip('"') if arg.startswith('"') and arg.endswith('"') else arg
            result['params']['sfx_specs'].append(sfx_val)
            i += 1
        elif mode == 'ttm' and arg_lower == 'stems':
            if 'extract' not in result['params']:
                result['error'] = 'stems keyword is only valid with extract task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'stems keyword requires instruments (e.g., stems "drums bass" or stems "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'add':
            if 'complete' not in result['params']:
                result['error'] = 'add keyword is only valid with complete task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'add keyword requires instruments (e.g., add "drums bass guitar" or add "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'reference':
            if ('complete' not in result['params'] and 'lego' not in result['params']
                and 'is_remix' not in result and 'is_repaint' not in result
                and 'bgm' not in result['params']):
                result['error'] = 'reference keyword is only valid with complete/lego/remix/repaint/bgm task'
                return result
            i += 1
            ref_entries = []
            while i < len(args):
                peek_lower = args[i].lower()
                if peek_lower in ('mix', 'blend', 'result', 'make', 'add', 'overdose',
                                  'complete', 'lego', 'video', 'extract', 'stems', 'only',
                                  'remix', 'repaint', 'bias', 'vc', 'clone', 'noblend', 'usrc',
                                  'script', 'lyrics', 'styling', 'base', 'target', 'duration',
                                  'timestamp', 'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr',
                                  'reference', 'sfx:', 'bgm', 'music'):
                    break
                if peek_lower in ('voice', 'music'):
                    sv_type = peek_lower
                    i += 1
                    if i >= len(args):
                        result['error'] = 'reference requires a path after voice/music'
                        return result
                    tr, rp = _parse_ref_time_spec(args[i])
                    ref_entries.append((sv_type, rp, tr))
                    i += 1
                else:
                    tr, rp = _parse_ref_time_spec(args[i])
                    ref_entries.append(('asis', rp, tr))
                    i += 1
            if not ref_entries:
                result['error'] = 'reference requires at least one path'
                return result
            if len(ref_entries) > 3:
                print("Warning: reference supports up to 3 entries, using first 3")
                ref_entries = ref_entries[:3]
            result['params']['ref_entries'] = ref_entries
        elif mode == 'ttm' and arg_lower == 'make':
            if 'lego' not in result['params']:
                result['error'] = 'make keyword is only valid with lego task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'make keyword requires instruments (e.g., make "drums bass" or make "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'only':
            if 'extract' not in result['params']:
                result['error'] = 'only keyword is only valid with extract task'
                return result
            if result['params'].get('extract_mix'):
                result['error'] = 'only and mix cannot be used together, use one or the other'
                return result
            result['params']['extract_only'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'mix':
            if 'lego' not in result['params'] and 'extract' not in result['params']:
                result['error'] = 'mix keyword is only valid with lego/extract task'
                return result
            if result['params'].get('blend_mode'):
                result['error'] = 'mix and blend cannot be used together, use one or the other'
                return result
            if result['params'].get('extract_only'):
                result['error'] = 'only and mix cannot be used together, use one or the other'
                return result
            if 'extract' in result['params']:
                result['params']['extract_mix'] = True
            else:
                result['params']['mix_mode'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'blend':
            if 'lego' not in result['params']:
                result['error'] = 'blend keyword is only valid with lego task'
                return result
            if result['params'].get('mix_mode'):
                result['error'] = 'mix and blend cannot be used together, use one or the other'
                return result
            result['params']['blend_mode'] = True
            i += 1
        elif arg_lower in valid_keywords:
            current_keyword = arg_lower
            result['params'].setdefault(current_keyword, [])
            i += 1
        elif mode == 'sts' and arg_lower == 'music':
            if result['is_mimic']:
                result['error'] = 'music and mimic cannot be used together'
                return result
            result['is_music'] = True
            current_keyword = None
            i += 1
        elif mode == 'sts' and arg_lower == 'mimic':
            if result['is_music']:
                result['error'] = 'music and mimic cannot be used together'
                return result
            result['is_mimic'] = True
            current_keyword = None
            i += 1
        elif mode == 'sts' and arg_lower == 'nomusic':
            if result['is_music']:
                result['error'] = 'nomusic cannot be used with music'
                return result
            result['nomusic'] = True
            current_keyword = None
            i += 1
        elif mode == 'tts' and arg_lower == 'slc':
            result['params']['slc'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                if peek.lower() != 'music' and peek.lower() != 'overdose' and peek.lower() not in valid_keywords:
                    result['params']['slc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'tts' and arg_lower == 'music' and result['params'].get('slc'):
            result['params']['slc_music'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                if peek.lower() != 'overdose' and peek.lower() not in valid_keywords:
                    result['params']['slc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'tts' and arg_lower == 'svc':
            result['params']['svc'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                if peek.lower() != 'overdose' and peek.lower() not in valid_keywords:
                    result['params']['svc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'ttm' and arg_lower == 'vc':
            result['params']['vc'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'clone':
            if i + 1 >= len(args):
                result['error'] = 'clone requires a source path'
                return result
            _ci = i + 1
            if args[_ci].lower() == 'first':
                result['clone_first'] = True
                _ci += 1
                if _ci >= len(args):
                    result['error'] = 'clone requires a source path after first'
                    return result
            result['clone_path'] = args[_ci]
            i = _ci + 1
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'remix':
            result['is_remix'] = True
            i += 1
            remix_entries = []
            while i < len(args):
                peek_lower = args[i].lower()
                if peek_lower in ('script', 'lyrics', 'styling', 'base', 'target', 'duration', 'timestamp',
                                  'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr',
                                  'complete', 'lego', 'video', 'extract', 'stems', 'only',
                                  'noblend', 'usrc', 'remix', 'repaint', 'bias', 'vc', 'clone',
                                  'reference', 'sfx:', 'add', 'make', 'mix', 'blend', 'result', 'overdose',
                                  'music', 'bgm'):
                    break
                if peek_lower in ('voice', 'music'):
                    sv_type = peek_lower
                    i += 1
                    if i >= len(args):
                        result['error'] = 'remix source requires a path after voice/music'
                        return result
                    remix_entries.append((sv_type, args[i]))
                    i += 1
                else:
                    remix_entries.append(('asis', args[i]))
                    i += 1
            if not remix_entries:
                result['error'] = 'remix requires at least one source path'
                return result
            if len(remix_entries) > 3:
                print("Warning: remix supports up to 3 sources, using first 3")
                remix_entries = remix_entries[:3]
            result['remix_entries'] = remix_entries
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'bias':
            if i + 1 >= len(args):
                result['error'] = 'bias requires a value'
                return result
            result['bias_val'] = args[i + 1]
            i += 2
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'repaint':
            if i + 1 >= len(args):
                result['error'] = 'repaint requires a source path'
                return result
            result['is_repaint'] = True
            _rp_src_idx = i + 1
            if args[_rp_src_idx].lower() in ('voice', 'music'):
                result['repaint_source_prefix'] = args[_rp_src_idx].lower()
                _rp_src_idx += 1
                if _rp_src_idx >= len(args):
                    result['error'] = f'repaint requires a source path after {result["repaint_source_prefix"]}'
                    return result
            result['repaint_path'] = args[_rp_src_idx]
            i = _rp_src_idx + 1
            _mp_pattern = re.compile(r'^\d+\.?\d*-\d+\.?\d*/')
            if i < len(args) and _mp_pattern.match(args[i]):
                result['repaint_multipass'] = []
                while i < len(args) and _mp_pattern.match(args[i]):
                    result['repaint_multipass'].append(args[i])
                    i += 1
            current_keyword = None
        elif mode == 'ttm' and arg.startswith('time:'):
            result['time_range'] = arg[5:]
            i += 1
        elif current_keyword is not None and arg_lower == 'first':
            result['use_first'] = True
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
                if mode == 'ttm' and (result['params'].get('complete') or result['params'].get('lego')
                                   or result['params'].get('extract')):
                    if not result['params'].get('task_source_args'):
                        result['params']['task_source_args'] = [arg]
                        i += 1
                    else:
                        if mode == 'sts':
                            result['error'] = f'invalid parameter "{arg}" only next parameter should be music or mimic or empty'
                        else:
                            result['error'] = f'Unknown parameter: {arg}'
                        return result
                else:
                    try:
                        duration = int(arg)
                        result['params']['duration'] = duration
                        i += 1
                    except ValueError:
                        if mode == 'sts':
                            result['error'] = f'invalid parameter "{arg}" only next parameter should be music or mimic or empty'
                        else:
                            result['error'] = f'Unknown parameter: {arg}'
                        return result

    result['params']['result_path'] = result_path
    if mode == 'ttm' and 'clone_path' in result and not result['params'].get('vc'):
        print("Warning: 'clone' specified without 'vc' flag — clone will be ignored. Use 'vc' to enable voice cloning in TTM mode.")
    return result

def is_num(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

VOICES_DIR = os.path.join(os.getcwd(), "voices")

def _ensure_voices_dir():
    os.makedirs(VOICES_DIR, exist_ok=True)

def _save_voice_prompt(voice_prompt_items, character_name):
    _ensure_voices_dir()
    character_name = character_name.lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"voder_tts_{character_name}_{timestamp}.tts"
    filepath = os.path.join(VOICES_DIR, filename)
    import torch
    from dataclasses import asdict
    payload = {"items": [asdict(it) for it in voice_prompt_items], "character": character_name, "timestamp": timestamp}
    torch.save(payload, filepath)
    return filepath

def _load_voice_prompt(filepath):
    import torch
    from qwen_tts import VoiceClonePromptItem
    if not os.path.exists(filepath):
        return None
    payload = torch.load(filepath, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        return None
    items_raw = payload["items"]
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        return None
    items = []
    for d in items_raw:
        if not isinstance(d, dict):
            return None
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            return None
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk,
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
            ref_text=d.get("ref_text", None),
        ))
    return items

def _find_voice_file(name):
    _ensure_voices_dir()
    if os.path.exists(name) and name.endswith('.tts'):
        return name
    name = name.lower()
    matches = []
    for f in os.listdir(VOICES_DIR):
        if not f.endswith('.tts'):
            continue
        if f.startswith(f"voder_tts_{name}_"):
            matches.append(os.path.join(VOICES_DIR, f))
    if not matches:
        return None
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0]

def _resolve_voice_ref(value):
    if ':' not in value:
        voice_file = _find_voice_file(value.lower())
        if voice_file:
            return voice_file
        return None
    parts = value.split(':', 1)
    first = parts[0].strip()
    second = parts[1].strip().lower()
    if second.endswith('.tts'):
        if os.path.exists(second):
            return second
        voice_file = _find_voice_file(second)
        if voice_file:
            return voice_file
        return None
    voice_file = _find_voice_file(second)
    if voice_file:
        return voice_file
    return None

def _is_trained_voice_ref(value):
    if os.path.exists(value) and value.endswith('.tts'):
        return True
    if _find_voice_file(value.lower()) is not None:
        return True
    if ':' in value:
        parts = value.split(':', 1)
        second = parts[1].strip()
        if second.endswith('.tts'):
            if os.path.exists(second):
                return True
            if _find_voice_file(second.lower()) is not None:
                return True
        if _find_voice_file(second.lower()) is not None:
            return True
    return False

def _save_fish_voice(encoded_refs, character_name, ref_text=None):
    _ensure_voices_dir()
    character_name = character_name.lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"voder_ttse_{character_name}_{timestamp}.ttse"
    filepath = os.path.join(VOICES_DIR, filename)
    import torch
    payload = {
        "tokens": encoded_refs["tokens"],
        "text": ref_text or encoded_refs.get("text", ""),
        "character": character_name,
        "timestamp": timestamp
    }
    torch.save(payload, filepath)
    return filepath

def _load_fish_voice(filepath):
    import torch
    if not os.path.exists(filepath):
        return None
    payload = torch.load(filepath, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "tokens" not in payload:
        return None
    return payload

def _find_fish_voice_file(name):
    _ensure_voices_dir()
    if os.path.exists(name) and name.endswith('.ttse'):
        return name
    name = name.lower()
    matches = []
    for f in os.listdir(VOICES_DIR):
        if not f.endswith('.ttse'):
            continue
        if f.startswith(f"voder_ttse_{name}_"):
            matches.append(os.path.join(VOICES_DIR, f))
    if not matches:
        return None
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0]

def _is_fish_voice_ref(value):
    if os.path.exists(value) and value.endswith('.ttse'):
        return True
    if _find_fish_voice_file(value.lower()) is not None:
        return True
    if ':' in value:
        parts = value.split(':', 1)
        second = parts[1].strip()
        if second.endswith('.ttse'):
            if os.path.exists(second):
                return True
            if _find_fish_voice_file(second.lower()) is not None:
                return True
        if _find_fish_voice_file(second.lower()) is not None:
            return True
    return False

def _resolve_fish_voice_ref(value):
    if ':' not in value:
        voice_file = _find_fish_voice_file(value.lower())
        if voice_file:
            return voice_file
        return None
    parts = value.split(':', 1)
    first = parts[0].strip()
    second = parts[1].strip().lower()
    if second.endswith('.ttse'):
        if os.path.exists(second):
            return second
        voice_file = _find_fish_voice_file(second)
        if voice_file:
            return voice_file
        return None
    voice_file = _find_fish_voice_file(second)
    if voice_file:
        return voice_file
    return None

def _check_voice_extreme_mismatch(voice_path, use_extreme):
    if not voice_path:
        return False
    basename = os.path.basename(voice_path)
    if use_extreme and basename.endswith('.tts'):
        print("Error: .tts voice files are for standard TTS mode, .ttse files are for extreme mode. Use 'voder.py train extreme' to create a .ttse file.")
        return True
    if not use_extreme and basename.endswith('.ttse'):
        print("Error: .ttse voice files are for extreme mode, .tts files are for standard TTS mode. Remove 'extreme' keyword or use a .tts file instead.")
        return True
    return False

def validate_oneline_mode(mode_name):
    valid_modes = ['tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train']
    if mode_name.lower() in valid_modes:
        return mode_name.lower()
    return None

def show_oneline_usage():
    print("VODER One-Line Command Usage:")
    print("=" * 60)
    print()
    print("Available modes:")
    print("  tts      - Text-to-Speech")
    print("  sts      - Speech-to-Speech (Voice Conversion)")
    print("  ttm      - Text-to-Music (use 'vc' flag for voice cloning)")
    print("  stt      - Speech-to-Text (Transcription with optional diarization)")
    print("  se       - Speech Enhancement (denoise, dereverb, restore)")
    print("  sfx      - Sound Effects (text prompt + duration → audio)")
    print("  svs      - Song Voice Separate (extract vocals/music from song)")
    print("  ss       - Speakers Separator (extract all speakers one by one)")
    print("  train    - Train and save voice clones")
    print()
    print("Train examples:")
    print('  python voder.py train voice:james "ref1.wav" "ref2.wav"')
    print('  python voder.py train voice:sarah "ref.wav" test')
    print('  python voder.py train voice:narrator "ref.wav" test "Custom test script here"')
    print()
    print("SVS examples (Song Voice Separate):")
    print('  python voder.py svs voice "path/to/song.mp3"')
    print('  python voder.py svs music "path/to/song.mp3"')
    print('  python voder.py svs voice "path/to/song.mp3" result "output.wav"')
    print('  python voder.py svs music "path/to/song.mp3" result "output.wav"')
    print()
    print("SS examples (Speakers Separator):")
    print('  python voder.py ss "path/to/audio.wav"')
    print('  python voder.py ss "path/to/video.mp4"')
    print('  python voder.py ss "https://youtube.com/watch?v=..."')
    print('  python voder.py ss se "path/to/audio.wav"')
    print()
    print("Single mode examples:")
    print('  python voder.py tts script "hello world" voice "male voice"')
    print('  python voder.py tts script "hello" target "voice.wav"')
    print('  python voder.py tts ocr "path/to/image.png" voice "text: female voice"')
    print('  python voder.py tts ocr "path/to/image.png" target "text: voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav" music')
    print('  python voder.py ttm lyrics "song" styling "pop" 30')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target voice "ref.wav"')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target music "ref.wav"')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target voice "https://youtu.be/..."')
    print('  python voder.py ttm vc lyrics "song" styling "pop" 30 clone "voice.wav"')
    print()
    print("STT examples (Speech-to-Text transcription):")
    print('  python voder.py stt "path/to/audio.wav"')
    print('  python voder.py stt "audio1.wav" "audio2.wav"')
    print('  python voder.py stt "audio.wav" timestamp')
    print('  python voder.py stt "audio.wav" dialogue')
    print('  python voder.py stt "audio.wav" timestamp dialogue')
    print('  python voder.py stt "audio.wav" translate')
    print('  python voder.py stt "audio.wav" translate dialogue')
    print('  python voder.py stt "audio.wav" translate timestamp dialogue')
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
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano"')
    print('  python voder.py tts script "James: Hello" script "sfx: thunder /duration:3" voice "James: deep male" music "soft piano" level "10:20-50"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano" reference "ref_song.mp3"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" music "epic orchestral" reference "https://youtube.com/watch?v=..."')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "chill lo-fi" reference "ref_video.mp4"')
    print('  python voder.py tts overdose script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"')
    print('  python voder.py tts overdose script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav" music "soft piano"')
    print()
    print("Parameters (can appear multiple times):")
    print("  script   - Dialogue line in 'Character: text' format, or plain text for single mode")
    print("  voice    - Voice prompt in 'Character: description' format (TTS)")
    print("  target   - Audio file path in 'Character: path' format (voice clone) or single path (STS)")
    print("  lyrics   - Song lyrics for TTM / remix (single, optional for remix)")
    print("  styling  - Style prompt for TTM (single)")
    print("  base     - Base audio/video path")
    print("  music    - Music flag for STS mode (uses 44.1kHz v1 model)")
    print("  timestamp - Keep Whisper timestamps in output (STT mode)")
    print("  dialogue - Enable speaker diarization (STT mode)")
    print("  translate - Translate to English (STT mode, uses large-v3 model)")
    print("  sound    - Sound effect prompt (SFX mode)")
    print("  duration - Duration in seconds (10-300 for TTM, 1-30 for SFX)")
    print("  steps    - Inference steps (1-100, SFX mode, default: 30)")
    print("  guide    - Guidance scale (1.0-10.0, SFX mode, default: 4.5)")
    print("  music    - Background music description (dialogue/bgm modes)")
    print("  level    - Music volume levels e.g. \"10:20-50 30:60-80\" (dialogue modes) or 0-100 (bgm mode, default: 35)")
    print("  reference - Music reference audio/video path or URL (up to 3 with voice/music prefix, optional time spec: \"start(ref)\" or \"start-end/ref2(ref)\")")
    print("  ocr      - Image file path for OCR text extraction (TTS modes)")
    print("  overdose - Use VibeVoice ASR for dialogue source and enhanced music (TTS/TTM modes)")
    print("  bgm      - Add or replace background music on an audio/video (TTM mode)")
    print("  usrc     - Blend with original source instead of isolated voice/music (complete)")
    print('  "sfx:"   - Sound effect spec for bgm/complete tasks: "sfx:prompt/duration-position/level"')
    print("  <number> - Duration in seconds (10-300, for TTM modes)")
    print()
    print("TTS SLC examples (Speaker Language Conversion):")
    print('  python voder.py tts slc "path/to/audio.wav"')
    print('  python voder.py tts slc music "path/to/audio.wav"')
    print('  python voder.py tts slc "path/to/audio.wav" target "voice_ref.wav"')
    print('  python voder.py tts overdose slc "path/to/audio.wav"')
    print('  python voder.py tts overdose slc music "path/to/audio.wav"')
    print()
    print("TTS SVC examples (Speaker Voice Change):")
    print('  python voder.py tts svc "speech.wav" target "voice_ref.wav"')
    print('  python voder.py tts svc "speech.wav" voice "deep male, authoritative"')
    print('  python voder.py tts overdose svc "speech.wav" target "voice.wav"')
    print('  python voder.py tts svc "speech.wav" target "sts:voice_ref.wav"')
    print('  python voder.py tts svc "speech.wav" target "(ref1.wav)(ref2.wav)(ref3.wav)"')
    print('  python voder.py tts svc "speech.wav" target "sts:(ref1.wav)(ref2.wav)"')
    print()
    print("BGM examples (add/replace background music on audio or video):")
    print('  python voder.py ttm bgm "path/to/audio.wav" music "soft piano"')
    print('  python voder.py ttm bgm "path/to/video.mp4" music "epic orchestral" level 50')
    print('  python voder.py ttm overdose bgm "path/to/audio.wav" music "lo-fi chill" level 25 reference "ref_song.mp3"')
    print('  python voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient synth" level 40')
    print('  python voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"')
    print('  python voder.py ttm bgm "audio.wav" music "piano" "sfx:thunder/10-5/50"')
    print('  python voder.py ttm bgm "audio.wav" "sfx:rain/8-22" "sfx:thunder/10-5/60"')
    print('  python voder.py ttm bgm "audio.wav" music "piano" reference "30-60(ref.wav)"')
    print()
    print("Complete + SFX examples (add instruments and/or sound effects):")
    print('  python voder.py ttm complete "source.wav" add "drums bass" "sfx:thunder/10-5/50"')
    print('  python voder.py ttm complete "source.wav" "sfx:rain/8-22"')
    print('  python voder.py ttm complete video "source.mp4" add "everything" "sfx:boom/12-30/40"')
    print()
    print("Complete with voice/music isolation examples:")
    print('  python voder.py ttm complete voice "song.wav" add "drums bass"')
    print('  python voder.py ttm complete music "song.wav" add "everything"')
    print('  python voder.py ttm complete voice usrc "song.wav" add "drums bass guitar"')
    print('  python voder.py ttm complete music usrc "song.wav" add "everything"')
    print('  python voder.py ttm complete voice "podcast.wav" "sfx:bell/5-10/40"')
    print()
    print("Remix examples (remix a song with new style and optional lyrics):")
    print('  python voder.py ttm remix "song.wav" styling "electronic dance"')
    print('  python voder.py ttm remix "song.wav" lyrics "new words here" styling "pop rock"')
    print('  python voder.py ttm remix voice "song.wav" styling "lo-fi chill"')
    print('  python voder.py ttm remix music "song.wav" lyrics "verse lyrics" styling "jazz"')
    print('  python voder.py ttm remix "song.wav" lyrics "custom lyrics" styling "hip hop" bias 70')
    print('  python voder.py ttm remix "song.wav" styling "ambient" reference "ref_song.mp3"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav"')
    print('  python voder.py ttm remix voice "vocal.wav" music "inst.wav" styling "funk" reference "ref.wav"')
    print('  python voder.py ttm overdose remix "song.wav" lyrics "dreamy verse" styling "synthwave"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference "30-60(ref.wav)"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference voice "0-15/40-55(ref.wav)" music "ref2.wav"')
    print()
    print("Repaint examples (restyle a specific time range of a song):")
    print('  python voder.py ttm repaint "song.wav" time:20-80 styling "more energetic"')
    print('  python voder.py ttm repaint "song.wav" time:20-80 styling "orchestral" bias 80 reference "ref.wav"')
    print('  python voder.py ttm overdose repaint "song.wav" time:20-80 styling "jazz" reference voice "ref.wav"')
    print('  python voder.py ttm repaint voice "song.wav" time:20-80 styling "funk"')
    print('  python voder.py ttm repaint music "song.wav" time:20-80 styling "ambient"')
    print()
    print("Multi-pass repaint examples (multiple edits, each pass builds on the previous):")
    print('  python voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"')
    print('  python voder.py ttm repaint "song.wav" "0-30/styling(funk)/lyrics(new words\\nhere)" "15-30/styling(ambient)/reference(ref.wav)"')
    print('  python voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"')
    print('  python voder.py ttm repaint "song.wav" "20-80/styling(jazz)/reference-voice(30-60(vocals.wav))"')
    print('  python voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"')
    print()
    print('SFX spec format: "sfx:prompt/duration-position/level"')
    print("  prompt    - SFX description text (required)")
    print("  duration  - SFX length 5-30 seconds (clamped, auto-cut if exceeds source)")
    print("  position  - Place at N seconds into source (required, cannot exceed source length)")
    print("  level     - Volume 1-100% (optional, default: 50)")
    print("  Multiple SFX specs can be specified")
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
    if 'is_mimic' in parsed:
        params['is_mimic'] = parsed['is_mimic']
    if 'nomusic' in parsed:
        params['nomusic'] = parsed['nomusic']
    if 'is_remix' in parsed:
        params['is_remix'] = parsed['is_remix']
    if 'remix_entries' in parsed:
        params['remix_entries'] = parsed['remix_entries']
    if 'bias_val' in parsed:
        params['bias_val'] = parsed['bias_val']
    if 'is_repaint' in parsed:
        params['is_repaint'] = parsed['is_repaint']
    if 'repaint_path' in parsed:
        params['repaint_path'] = parsed['repaint_path']
    if 'repaint_source_prefix' in parsed:
        params['repaint_source_prefix'] = parsed['repaint_source_prefix']
    if 'repaint_multipass' in parsed:
        params['repaint_multipass'] = parsed['repaint_multipass']
    if 'time_range' in parsed:
        params['time_range'] = parsed['time_range']
    if 'clone_path' in parsed:
        params['clone_path'] = parsed['clone_path']
    if 'use_first' in parsed:
        params['use_first'] = parsed['use_first']
    if 'clone_first' in parsed:
        params['clone_first'] = parsed['clone_first']

    success = False
    if mode == 'tts':
        success = oneline_tts(params)
    elif mode == 'sts':
        success = oneline_sts(params)
    elif mode == 'ttm':
        if params.get('bgm'):
            success = oneline_ttm_bgm(params)
        elif params.get('complete'):
            success = oneline_ttm_complete(params)
        elif params.get('lego'):
            success = oneline_ttm_lego(params)
        elif params.get('extract'):
            success = oneline_ttm_extract(params)
        else:
            success = oneline_ttm(params)
    elif mode == 'stt':
        success = oneline_stt(params)
    elif mode == 'se':
        success = oneline_se(params)
    elif mode == 'sfx':
        success = oneline_sfx(params)
    elif mode == 'svs':
        success = oneline_svs(params)
    elif mode == 'ss':
        success = oneline_ss(params)
    elif mode == 'train':
        success = oneline_train(params)
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

TRAIN_TEST_SCRIPT = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor. In the distance, you can hear the distant rumble of thunder rolling across the wide open plains, signaling that a summer storm is approaching fast."

def oneline_train(params):
    sub_type = params.get('sub_type')
    voice_name = params.get('voice_name', '').lower()
    ref_paths = params.get('ref_paths', [])
    has_test = params.get('has_test', False)
    test_script = params.get('test_script')
    use_first = params.get('use_first', False)
    use_extreme = params.get('extreme', False)

    if sub_type != 'voice':
        print("Error: Only 'voice' training is supported")
        return False

    _cleanup = []
    try:
        print(f"Training voice '{voice_name}' from {len(ref_paths)} reference(s)...")
        clean_vocal = None
        if len(ref_paths) > 1:
            clean_vocal = _resolve_multi_refs(ref_paths, _cleanup, use_first=use_first)
            if not clean_vocal:
                print("Error: Failed to resolve reference audios")
                return False
        else:
            if use_first:
                print("Warning: 'first' keyword ignored (only one reference provided)")
            resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
            if not resolved_audio:
                print(f"Error: Failed to resolve reference: {ref_paths[0]}")
                return False
            _cleanup.extend(_cl)
            clean_vocal = svs_extract_vocals(resolved_audio)
            if clean_vocal and clean_vocal != resolved_audio:
                _cleanup.append(clean_vocal)
            if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                _cleanup.append(resolved_audio)

        if use_extreme:
            print("Loading Fish-S2Pro model (extreme)...")
            fish_tts = FishTTS()
            if not fish_tts.ensure_model():
                print("Error: Failed to load Fish-S2Pro model")
                return False
            print("Encoding voice reference...")
            success = fish_tts.encode_voice(clean_vocal)
            if not success:
                print("Error: Voice encoding failed")
                return False
            saved_path = _save_fish_voice(fish_tts.encoded_refs, voice_name)
            print(f"Extreme voice '{voice_name}' saved to: {saved_path}")
            if has_test:
                script = test_script if test_script else TRAIN_TEST_SCRIPT
                print(f"Testing trained voice (extreme)...")
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                test_output = os.path.join(results_dir, f"voder_tts_extreme_{voice_name}_test_{timestamp}.wav")
                success = fish_tts.synthesize(script, test_output)
                if success:
                    print(f"Test output saved to: {test_output}")
                else:
                    print("Warning: Test synthesis failed (voice was still saved)")
            fish_tts.cleanup()
        else:
            print("Loading Qwen-TTS model...")
            tts = QwenTTS()
            if tts.model is None:
                print("Error: Failed to load Qwen-TTS model")
                return False

            print("Extracting voice characteristics...")
            success = tts.extract_voice(clean_vocal)
            if not success:
                print("Error: Voice extraction failed")
                return False

            voice_prompt = tts.voice_prompt
            if voice_prompt is None:
                print("Error: Voice prompt extraction returned None")
                return False

            if not isinstance(voice_prompt, list):
                voice_prompt = [voice_prompt]

            saved_path = _save_voice_prompt(voice_prompt, voice_name)
            print(f"Voice '{voice_name}' saved to: {saved_path}")

            if has_test:
                script = test_script if test_script else TRAIN_TEST_SCRIPT
                print(f"Testing trained voice...")
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                test_output = os.path.join(results_dir, f"voder_tts_{voice_name}_test_{timestamp}.wav")
                success = tts.synthesize(script, test_output)
                if success:
                    print(f"Test output saved to: {test_output}")
                else:
                    print("Warning: Test synthesis failed (voice was still saved)")

        return True
    finally:
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def _create_tts_engine(use_extreme=False):
    if use_extreme:
        engine = FishTTS()
        if engine.ensure_model():
            return engine
        print("Warning: Fish-S2Pro failed to load, falling back to Qwen-TTS")
        return QwenTTS()
    return QwenTTS()

def _tts_extract_voice(engine, audio_path, use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        return engine.encode_voice(audio_path)
    return engine.extract_voice(audio_path)

def _tts_synthesize(engine, text, output_path, language="Auto", use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        return engine.synthesize(text, output_path)
    return engine.synthesize(text, output_path, language=language)

def _tts_load_voice(engine, voice_path, use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        payload = _load_fish_voice(voice_path)
        if payload is None:
            return False
        engine.encoded_refs = payload
        return True
    items = _load_voice_prompt(voice_path)
    if items is None:
        return False
    engine.voice_prompt = items
    return True

def _tts_cleanup(engine, use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        engine.cleanup()
    else:
        import gc
        try:
            del engine
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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
    reference_params = params.get('reference', [])
    reference_source = reference_params[0] if reference_params else None
    ocr_param = params.get('ocr', [])
    use_overdose = params.get('overdose', False)
    use_extreme = params.get('extreme', False)

    if params.get('slc'):
        slc_path = params.get('slc_path')
        if not slc_path:
            for t in targets:
                if t:
                    slc_path = t
                    break
        if not slc_path and voices:
            for v in voices:
                if os.path.exists(v) or is_youtube_url(v):
                    slc_path = v
                    break
        if not slc_path:
            for s in scripts:
                if os.path.exists(s) or is_youtube_url(s):
                    slc_path = s
                    break
        if not slc_path:
            print("Error: TTS SLC requires an audio/video source path")
            return False

        _slc_cleanup = []
        audio_path = slc_path
        needs_youtube_dl = is_youtube_url(slc_path)

        if needs_youtube_dl:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(slc_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _slc_cleanup.append(dl_path)
        elif slc_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(slc_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _slc_cleanup.append(audio_path)
        elif not os.path.exists(slc_path):
            print(f"Error: File not found: {slc_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _slc_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        slc_music = params.get('slc_music', False)
        music_track = None
        if slc_music:
            print("Extracting music track via SVS music...")
            music_track = svs_extract_music(audio_path)
            if music_track and music_track != audio_path:
                _slc_cleanup.append(music_track)
            else:
                music_track = None

        print("Loading Whisper model (large-v3)...")
        stt = WhisperSTT(skip_turbo=True)
        stt.ensure_translate_model()
        if stt.translate_model is None:
            print("Error: Failed to load Whisper large-v3 model")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print("Transcribing audio...")
        try:
            result = stt.translate_model.transcribe(clean_vocal, word_timestamps=True)
        except Exception as e:
            print(f"Transcription error: {e}")
            result = None
        if not result:
            print("Error: Transcription failed")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        detected_lang = result.get("language", "en")
        transcribed_text = result.get("text", "").strip()
        if not transcribed_text:
            print("Error: No speech detected in audio")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print(f"Detected language: {detected_lang}")
        print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        tts_lang = "English"
        final_text = transcribed_text

        if detected_lang == "en":
            print("Audio is already in English")
        else:
            print("Translating to English...")
            try:
                trans_result = stt.translate_model.transcribe(clean_vocal, task="translate", word_timestamps=True)
            except Exception as e:
                print(f"Translation error: {e}")
                trans_result = None
            if trans_result and trans_result.get("text", "").strip():
                final_text = trans_result["text"].strip()
                print(f"Translated text: {final_text[:100]}{'...' if len(final_text) > 100 else ''}")
            else:
                print("Warning: Translation failed, using original transcription")

        del stt
        stt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"TTS language: {tts_lang}")
        if use_extreme:
            print("Loading Fish-S2Pro model (extreme)...")
            tts = FishTTS()
            if not tts.ensure_model():
                print("Error: Failed to load Fish-S2Pro model")
                for f in _slc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
        else:
            print("Loading Qwen-TTS model...")
            tts = QwenTTS()
        print("Extracting voice characteristics...")
        success = _tts_extract_voice(tts, clean_vocal, use_extreme=use_extreme)
        if not success:
            print("Error: Voice extraction failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print("Generating speech...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_slc_{timestamp}.wav")
        success = _tts_synthesize(tts, final_text, output_path, language=tts_lang, use_extreme=use_extreme)
        if not success:
            print("Error: Synthesis failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        _tts_cleanup(tts, use_extreme=use_extreme)
        tts = None

        print(f"✓ SLC output saved to: {output_path}")

        if use_overdose:
            print("\nRunning overdose pass (STS v2 non-mimic)...")
            vc = SeedVCV2()
            if vc.model is None:
                print("Warning: Seed-VC v2 model failed to load, skipping overdose pass")
            else:
                svs_out = svs_extract_vocals(output_path)
                if svs_out and svs_out != output_path:
                    _slc_cleanup.append(svs_out)
                    vc_input = svs_out
                else:
                    vc_input = output_path
                try:
                    od_timestamp = time.strftime("%Y%m%d_%H%M%S")
                    od_output = os.path.join(results_dir, f"voder_tts_slc_od_{od_timestamp}.wav")
                    od_success = vc.convert(vc_input, clean_vocal, od_output)
                    if od_success:
                        print(f"✓ Overdose output saved to: {od_output}")
                        output_path = od_output
                    else:
                        print("Warning: Overdose STS pass failed, using standard SLC output")
                finally:
                    del vc
                    vc = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if music_track and os.path.exists(music_track):
            print("\nBlending voice output with music track...")
            blend_timestamp = time.strftime("%Y%m%d_%H%M%S")
            blend_output = os.path.join(results_dir, f"voder_tts_slc_music_{blend_timestamp}.wav")
            blend_cmd = [
                'ffmpeg', '-i', output_path, '-i', music_track,
                '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]',
                '-map', '[out]', '-y', blend_output
            ]
            blend_result = subprocess.run(blend_cmd, capture_output=True, text=True)
            if blend_result.returncode == 0 and os.path.exists(blend_output):
                print(f"✓ Blended output saved to: {blend_output}")
                output_path = blend_output
            else:
                print("Warning: Music blending failed, voice-only output preserved")

        for f in _slc_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return True

    if params.get('svc'):
        svc_path = params.get('svc_path')
        if not svc_path:
            for s in scripts:
                if os.path.exists(s) or is_youtube_url(s):
                    svc_path = s
                    break
        if not svc_path:
            for t in targets:
                if t and not t.lower().startswith('sts:'):
                    if os.path.exists(t) or is_youtube_url(t):
                        svc_path = t
                        break
        if not svc_path and voices:
            for v in voices:
                if v and not v.lower().startswith('sts:'):
                    if os.path.exists(v) or is_youtube_url(v):
                        svc_path = v
                        break
        if not svc_path:
            print("Error: TTS SVC requires an audio/video source path")
            return False

        svc_target = None
        for t in targets:
            if t:
                svc_target = t
                break
        if not svc_target:
            for v in voices:
                if v:
                    svc_target = v
                    break
        if not svc_target:
            print("Error: TTS SVC requires a target voice reference (target or voice parameter)")
            return False

        _svc_cleanup = []
        audio_path = svc_path
        needs_youtube_dl = is_youtube_url(svc_path)

        if needs_youtube_dl:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(svc_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _svc_cleanup.append(dl_path)
        elif svc_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(svc_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _svc_cleanup.append(audio_path)
        elif not os.path.exists(svc_path):
            print(f"Error: File not found: {svc_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _svc_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        if use_overdose:
            print("Loading VibeVoice ASR (overdose mode)...")
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                use_overdose = False
            else:
                try:
                    transcribed_text = asr.transcribe_plain_text(clean_vocal)
                except Exception as e:
                    print(f"VibeVoice transcription error: {e}")
                    transcribed_text = ""
                del asr
                asr = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not transcribed_text or not transcribed_text.strip():
                    print("Error: No speech detected in audio (VibeVoice)")
                    for f in _svc_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                transcribed_text = transcribed_text.strip()
                detected_lang = None
                print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        if not use_overdose:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            stt.ensure_model()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print("Transcribing audio...")
            try:
                result = stt.model.transcribe(clean_vocal, word_timestamps=True)
            except Exception as e:
                print(f"Transcription error: {e}")
                result = None
            if not result:
                print("Error: Transcription failed")
                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            detected_lang = result.get("language", "en")
            transcribed_text = result.get("text", "").strip()
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not transcribed_text:
                print("Error: No speech detected in audio")
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            print(f"Detected language: {detected_lang}")
            print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        tts_lang = "Auto"
        final_text = transcribed_text

        sts_target = svc_target.lower().startswith('sts:')
        target_voice_path = None
        if sts_target:
            sts_ref_raw = svc_target[4:]
            trained_file = _resolve_voice_ref(sts_ref_raw)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                print(f"Loading trained voice for STS pass: {trained_file}")
                if use_extreme:
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        tts.voice_prompt = voice_items
                        use_extreme = False
                else:
                    voice_items = _load_voice_prompt(trained_file)
                    if voice_items is None:
                        print(f"Error: Failed to load trained voice: {trained_file}")
                        return False
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    tts.voice_prompt = voice_items
            else:
                multi = _parse_multi_refs(sts_ref_raw)
                if multi:
                    target_voice_path = _resolve_multi_refs(multi, _svc_cleanup)
                    if not target_voice_path:
                        print("Error: Could not resolve STS target multi-reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                else:
                    resolved_audio, _cl = resolve_target_to_audio(sts_ref_raw)
                    if not resolved_audio:
                        print("Error: Could not resolve STS target reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    _svc_cleanup.extend(_cl)
                    target_voice_path = svs_extract_vocals(resolved_audio)
                    if target_voice_path and target_voice_path != resolved_audio:
                        _svc_cleanup.append(target_voice_path)
                    if resolved_audio not in _svc_cleanup and resolved_audio != target_voice_path:
                        _svc_cleanup.append(resolved_audio)
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                else:
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                print("Extracting voice characteristics from STS target...")
                success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme)
                if not success:
                    print("Error: Voice extraction from STS target failed")
                    del tts
                    tts = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for f in _svc_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
        else:
            trained_file = _resolve_voice_ref(svc_target)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                print(f"Loading trained voice from: {trained_file}")
                if use_extreme:
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        tts.voice_prompt = voice_items
                        use_extreme = False
                else:
                    voice_items = _load_voice_prompt(trained_file)
                    if voice_items is None:
                        print(f"Error: Failed to load trained voice: {trained_file}")
                        return False
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    tts.voice_prompt = voice_items
            else:
                multi = _parse_multi_refs(svc_target)
                if multi:
                    target_voice_path = _resolve_multi_refs(multi, _svc_cleanup)
                    if not target_voice_path:
                        print("Error: Could not resolve target multi-reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    if use_extreme:
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            return False
                    else:
                        print("Loading Qwen-TTS model...")
                        tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme)
                    if not success:
                        print("Error: Voice extraction failed")
                        del tts
                        tts = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                elif os.path.exists(svc_target) or is_youtube_url(svc_target):
                    resolved_audio, _cl = resolve_target_to_audio(svc_target)
                    if not resolved_audio:
                        print("Error: Could not resolve target audio reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    _svc_cleanup.extend(_cl)
                    target_voice_path = svs_extract_vocals(resolved_audio)
                    if target_voice_path and target_voice_path != resolved_audio:
                        _svc_cleanup.append(target_voice_path)
                    if resolved_audio not in _svc_cleanup and resolved_audio != target_voice_path:
                        _svc_cleanup.append(resolved_audio)
                    if use_extreme:
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            return False
                    else:
                        print("Loading Qwen-TTS model...")
                        tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme)
                    if not success:
                        print("Error: Voice extraction failed")
                        del tts
                        tts = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                else:
                    if use_extreme:
                        print("Loading Qwen-TTS VoiceDesign model (extreme: generating placeholder voice)...")
                        tts_design = QwenTTSVoiceDesign()
                        if tts_design.model is None:
                            print("Error: Failed to load VoiceDesign model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        placeholder_text = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor."
                        placeholder_path = os.path.join(tempfile.gettempdir(), f"voder_extreme_placeholder_{int(time.time())}.wav")
                        success = tts_design.synthesize(placeholder_text, svc_target, placeholder_path, language="English")
                        del tts_design
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if not success or not os.path.exists(placeholder_path):
                            print("Error: Failed to generate voice design placeholder for extreme mode")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        _svc_cleanup.append(placeholder_path)
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Encoding voice design audio as Fish reference...")
                        success = tts.encode_voice(placeholder_path)
                        if not success:
                            print("Error: Fish voice encoding failed")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Generating speech (extreme)...")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(results_dir, f"voder_tts_svc_extreme_{timestamp}.wav")
                        success = tts.synthesize(final_text, output_path)
                        if not success:
                            print("Error: Fish TTS synthesis failed")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        _tts_cleanup(tts, use_extreme=True)
                        tts = None
                        print(f"✓ SVC extreme output saved to: {output_path}")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return True
                    else:
                        print(f"Loading Qwen-TTS VoiceDesign model...")
                        tts_design = QwenTTSVoiceDesign()
                        if tts_design.model is None:
                            print("Error: Failed to load VoiceDesign model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Generating speech with voice design...")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(results_dir, f"voder_tts_svc_{timestamp}.wav")
                        success = tts_design.synthesize(final_text, svc_target, output_path, language=tts_lang)
                        if not success:
                            print("Error: VoiceDesign synthesis failed")
                            del tts_design
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        del tts_design
                        print(f"✓ SVC output saved to: {output_path}")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return True

        print("Generating speech...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_svc_{timestamp}.wav")
        success = _tts_synthesize(tts, final_text, output_path, language=tts_lang, use_extreme=use_extreme)
        if not success:
            print("Error: Synthesis failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _svc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        del tts
        tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ SVC output saved to: {output_path}")

        if sts_target:
            print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
            if target_voice_path and os.path.exists(target_voice_path):
                vc = SeedVCV2()
                if vc.model is None:
                    print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                else:
                    svs_out = svs_extract_vocals(output_path)
                    if svs_out and svs_out != output_path:
                        _svc_cleanup.append(svs_out)
                        vc_input = svs_out
                    else:
                        vc_input = output_path
                    try:
                        sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sts_output = os.path.join(results_dir, f"voder_tts_svc_sts_{sts_timestamp}.wav")
                        sts_success = vc.convert(vc_input, target_voice_path, sts_output)
                        if sts_success:
                            print(f"✓ STS-converted output saved to: {sts_output}")
                            output_path = sts_output
                        else:
                            print("Warning: STS voice conversion pass failed, using TTS output")
                    finally:
                        del vc
                        vc = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                print("Warning: No target voice path available for STS pass (trained voice used)")

        for f in _svc_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return True

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
        if not voices and not targets:
            print("Error: Single mode expects one voice or target argument")
            return False
        if music_description:
            print("Warning: Background music is only supported for dialogue mode. Ignoring music parameter.")
        if reference_source:
            print("Warning: Music reference is only supported for dialogue mode. Ignoring reference parameter.")
        script = scripts[0].replace('\\n', '\n')
        if voices:
            voice_value = voices[0]
            trained_file = _resolve_voice_ref(voice_value)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        if tts.model is None:
                            print("Error: Failed to load Qwen-TTS model")
                            return False
                        tts.voice_prompt = voice_items
                        use_extreme = False
                    print("Generating speech with trained voice (extreme)...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(results_dir, f"voder_tts_extreme_{timestamp}.wav")
                    success = _tts_synthesize(tts, script, output_path, use_extreme=use_extreme)
                    if not success:
                        print("Error: Synthesis failed")
                        return False
                    _tts_cleanup(tts, use_extreme=use_extreme)
                    print(f"✓ Success! Output saved to: {output_path}")
                    return True
                print(f"Loading trained voice from: {trained_file}")
                voice_items = _load_voice_prompt(trained_file)
                if voice_items is None:
                    print(f"Error: Failed to load trained voice: {trained_file}")
                    return False
                print("Loading Qwen-TTS model...")
                tts = QwenTTS()
                if tts.model is None:
                    print("Error: Failed to load Qwen-TTS model")
                    return False
                tts.voice_prompt = voice_items
                print("Generating speech with trained voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts.synthesize(script, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                print(f"✓ Success! Output saved to: {output_path}")
                return True
            voice_prompt = voice_value
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
            target_value = targets[0]
            is_sts_target = target_value.lower().startswith('sts:')
            if is_sts_target:
                target_value = target_value[4:]
            use_first = params.get('use_first', False)
            multi = _parse_multi_refs(target_value)
            _cleanup = []
            try:
                if multi:
                    clean_vocal = _resolve_multi_refs(multi, _cleanup, use_first=use_first)
                    if not clean_vocal:
                        return False
                else:
                    if use_first:
                        print("Warning: 'first' keyword ignored (only one reference provided)")
                    resolved_audio, _cl = resolve_target_to_audio(target_value)
                    if not resolved_audio:
                        return False
                    _cleanup.extend(_cl)
                    clean_vocal = svs_extract_vocals(resolved_audio)
                    if clean_vocal and clean_vocal != resolved_audio:
                        _cleanup.append(clean_vocal)
                    if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                        _cleanup.append(resolved_audio)
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                else:
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                print("Extracting voice characteristics...")
                success = _tts_extract_voice(tts, clean_vocal, use_extreme=use_extreme)
                if not success:
                    print("Error: Voice extraction failed")
                    return False
                print("Generating speech with cloned voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{'extreme_' if use_extreme else ''}{timestamp}.wav")
                success = _tts_synthesize(tts, script, output_path, use_extreme=use_extreme)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                del tts
                tts = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_sts_target:
                    print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
                    vc = SeedVCV2()
                    if vc.model is None:
                        print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                    else:
                        svs_out = svs_extract_vocals(output_path)
                        if svs_out and svs_out != output_path:
                            _cleanup.append(svs_out)
                            vc_input = svs_out
                        else:
                            vc_input = output_path
                        try:
                            sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                            sts_output = os.path.join(results_dir, f"voder_tts_sts_{sts_timestamp}.wav")
                            sts_success = vc.convert(vc_input, clean_vocal, sts_output)
                            if sts_success:
                                print(f"✓ STS-converted output saved to: {sts_output}")
                                output_path = sts_output
                            else:
                                print("Warning: STS voice conversion pass failed, using TTS output")
                        finally:
                            del vc
                            vc = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                else:
                    print(f"✓ Success! Output saved to: {output_path}")
                return True
            finally:
                for f in _cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
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
            text = text.strip().replace('\\n', '\n')
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
        trained_voice_refs = {}
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
            trained_file = _resolve_voice_ref(prompt)
            if trained_file:
                trained_voice_refs[char.lower()] = trained_file
            else:
                voice_prompts[char.lower()] = prompt

        try:
            target_assignments = {}
            sts_refs = {}
            all_target_cleanup = []
            use_first = params.get('use_first', False)
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
                is_sts = path.lower().startswith('sts:')
                if is_sts:
                    path = path[4:]
                multi = _parse_multi_refs(path)
                if multi:
                    clean_vocal = _resolve_multi_refs(multi, all_target_cleanup, use_first=use_first)
                    if not clean_vocal:
                        return False
                    target_assignments[char.lower()] = clean_vocal
                    if is_sts:
                        sts_refs[char.lower()] = clean_vocal
                else:
                    resolved_audio, _cleanup = resolve_target_to_audio(path)
                    if not resolved_audio:
                        return False
                    all_target_cleanup.extend(_cleanup)
                    clean_vocal = svs_extract_vocals(resolved_audio)
                    if clean_vocal and clean_vocal != resolved_audio:
                        all_target_cleanup.append(clean_vocal)
                    target_assignments[char.lower()] = clean_vocal
                    if is_sts:
                        sts_ref_path = clean_vocal
                        sts_refs[char.lower()] = sts_ref_path

            overlap = set(voice_prompts.keys()) & (set(target_assignments.keys()) | set(trained_voice_refs.keys()))
            if overlap:
                print(f"Error: Character(s) specified in both voice and target/trained: {', '.join(overlap)}")
                return False

            trained_voice_overlap = set(trained_voice_refs.keys()) & set(target_assignments.keys())
            if trained_voice_overlap:
                print(f"Error: Character(s) specified in both trained voice and target: {', '.join(trained_voice_overlap)}")
                return False

            script_chars = set()
            for _, char, _, _ in dialogue_items:
                if char.lower() != 'sfx':
                    script_chars.add(char.lower())
            all_assigned = set(voice_prompts.keys()) | set(target_assignments.keys()) | set(trained_voice_refs.keys())
            missing = script_chars - all_assigned
            if missing:
                print(f"Error: Missing voice/target for characters: {', '.join(missing)}")
                return False

            has_tts_chars = len(voice_prompts) > 0
            has_vc_chars = len(target_assignments) > 0 or len(trained_voice_refs) > 0

            for char_lower, trained_file in trained_voice_refs.items():
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False

            if music_description and music_description.strip() == "":
                music_description = None

            if music_level_spec and not music_description:
                print("Warning: Level spec ignored (no music description provided)")

            if reference_source and not music_description:
                print("Warning: Music reference ignored (no music description provided)")

            reference_audio = None
            if reference_source and music_description:
                _ref_is_video = False
                _ref_is_link = is_youtube_url(reference_source)
                if not _ref_is_link and os.path.exists(reference_source):
                    _ref_ext = os.path.splitext(reference_source)[1].lower()
                    _ref_is_video = _ref_ext in VIDEO_EXTENSIONS
                if _ref_is_video:
                    print("Reference is a video file, extracting audio...")
                elif _ref_is_link:
                    print("Reference is a URL, downloading audio...")
                else:
                    print("Resolving music reference source...")
                resolved_ref_audio, ref_cleanup = resolve_target_to_audio(reference_source)
                if not resolved_ref_audio:
                    print("Error: Could not resolve music reference source")
                    return False
                print("Cleaning reference through SVS music pipe...")
                reference_audio = svs_extract_music(resolved_ref_audio)
                all_target_cleanup.extend(ref_cleanup)
                if reference_audio and reference_audio != resolved_ref_audio and reference_audio not in all_target_cleanup:
                    all_target_cleanup.append(reference_audio)

            tts_design = None
            if has_tts_chars:
                print("Loading Qwen-TTS VoiceDesign model...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False

            vc_voice_prompts = None
            fish_voice_data = None
            tts_obj = None
            if has_vc_chars:
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts_obj = FishTTS()
                    if not tts_obj.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    fish_voice_data = {}
                    for char_lower, audio_path in target_assignments.items():
                        print(f"Encoding voice for '{char_lower}' (extreme)...")
                        success = tts_obj.encode_voice(audio_path)
                        if not success:
                            print(f"Error: Failed to encode voice from {audio_path}")
                            return False
                        fish_voice_data[char_lower] = {
                            "tokens": tts_obj.encoded_refs["tokens"].clone(),
                            "text": tts_obj.encoded_refs.get("text", "")
                        }
                    for char_lower, trained_file in trained_voice_refs.items():
                        print(f"Loading trained voice for '{char_lower}' (extreme) from: {trained_file}")
                        payload = _load_fish_voice(trained_file)
                        if payload is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        fish_voice_data[char_lower] = payload
                else:
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
                    for char_lower, trained_file in trained_voice_refs.items():
                        print(f"Loading trained voice for '{char_lower}' from: {trained_file}")
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        vc_voice_prompts[char_lower] = voice_items

            if has_tts_chars and tts_obj is None:
                print("Loading Qwen-TTS model for voice stabilization...")
                tts_obj = QwenTTS()

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

            if has_sfx or has_effects or has_vc_chars or has_tts_chars:
                success, msg = _assemble_enhanced_dialogue(
                    dialogue_items, voice_prompts, tts_design_obj=tts_design,
                    tts_vc_obj=tts_obj, vc_voice_data=vc_voice_prompts,
                    output_path=dialogue_temp.name, mode='tts',
                    sts_refs=sts_refs if sts_refs else None,
                    use_extreme=use_extreme, fish_voice_data=fish_voice_data
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
                ace = AceStepWrapper(use_overdose=use_overdose)
                if ace.handler is None:
                    print("Error: Failed to load ACE-Step model")
                    return False
                success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec, reference_audio=reference_audio)
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
        finally:
            for f in all_target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            if 'dialogue_temp' in dir() and os.path.exists(dialogue_temp.name):
                try:
                    os.unlink(dialogue_temp.name)
                except:
                    pass

def oneline_sts(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    is_music = params.get('is_music', False)
    is_mimic = params.get('is_mimic', False)
    no_music = params.get('nomusic', False)

    if 'base' not in params or len(params['base']) != 1:
        print("Error: STS mode requires exactly one 'base' parameter")
        return False
    if 'target' not in params or len(params['target']) != 1:
        print("Error: STS mode requires exactly one 'target' parameter")
        return False
    base_path = params['base'][0]
    target_value = params['target'][0]
    if not os.path.exists(base_path) and not is_youtube_url(base_path):
        print(f"Error: Base file not found: {base_path}")
        return False
    _target_cleanup = []
    use_first = params.get('use_first', False)
    target_multi = _parse_multi_refs(target_value)
    target_pre_cleaned = False
    if target_multi:
        resolved_target = _resolve_multi_refs(target_multi, _target_cleanup, use_first=use_first)
        if not resolved_target:
            return False
        target_pre_cleaned = True
    else:
        if use_first:
            print("Warning: 'first' keyword ignored (only one reference provided)")
        resolved_target, _cl = resolve_target_to_audio(target_value)
        if not resolved_target:
            return False
        _target_cleanup.extend(_cl)
    base_is_video = os.path.splitext(base_path)[1].lower() in VIDEO_EXTENSIONS
    base_original = base_path
    temp_base_extracted = None
    if base_is_video:
        if is_music or is_mimic:
            print("Error: Base input must be audio for this mode")
            return False
        print("Extracting audio from video...")
        temp_base_extracted = os.path.join(tempfile.gettempdir(), f"voder_cli_{int(time.time())}.wav")
        cmd = ['ffmpeg', '-i', base_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', temp_base_extracted]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(temp_base_extracted):
            print("Error: Could not extract audio from video")
            return False
        base_path = temp_base_extracted
    target_path = resolved_target
    if is_music:
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        print("Extracting music from source...")
        base_music = svs_extract_music(base_path)
        _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        if target_pre_cleaned:
            clean_vocal_target = target_path
        else:
            clean_vocal_target = svs_extract_vocals(target_path)
            _target_cleanup.append(clean_vocal_target)
        print("Resampling inputs to 44100Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 44100:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    else:
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        base_music = None
        if not no_music:
            print("Extracting music from source...")
            base_music = svs_extract_music(base_path)
            _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        if target_pre_cleaned:
            clean_vocal_target = target_path
        else:
            clean_vocal_target = svs_extract_vocals(target_path)
            _target_cleanup.append(clean_vocal_target)
        print("Loading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Resampling inputs to 22050Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 22050:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 22050)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_22k.name,
                convert_style=is_mimic
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            print("Upsampling output to 44100Hz...")
            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
            if sr_out != 44100:
                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                waveform_out = resampler_out(waveform_out)
            temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_output_44k.name, waveform_out, 44100)
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not no_music and base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            elif base_is_video:
                print("Merging converted audio with video...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.mp4")
                ret = os.system(f'ffmpeg -y -i "{base_original}" -i "{temp_output_44k.name}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Error: Failed to merge audio with video")
                    return False
            else:
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_22k.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

def oneline_ttm(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    _is_remix = params.get('is_remix', False)
    use_overdose = params.get('overdose', False)
    use_vc = params.get('vc', False)

    if use_vc:
        if _is_remix:
            print("Error: VC cannot be used with remix")
            return False
        if params.get('is_repaint', False):
            print("Error: VC cannot be used with repaint")
            return False
        if 'lyrics' not in params or len(params['lyrics']) != 1:
            print("Error: TTM VC requires exactly one 'lyrics' parameter")
            return False
        if 'styling' not in params or len(params['styling']) != 1:
            print("Error: TTM VC requires exactly one 'styling' parameter")
            return False
        if 'duration' not in params:
            print("Error: TTM VC requires duration (10-300 seconds)")
            return False
        duration = params['duration']
        if not (10 <= duration <= 300):
            print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
            return False
        clone_path = params.get('clone_path')
        if not clone_path:
            print("Error: TTM VC requires clone source path")
            return False
        lyrics = params['lyrics'][0].replace('\\n', '\n')
        style = params['styling'][0].replace('\\n', '\n')
        _vc_cleanup = []
        use_first = params.get('clone_first', False)
        clone_multi = _parse_multi_refs(clone_path)
        clone_pre_cleaned = False
        if clone_multi:
            clean_vocal = _resolve_multi_refs(clone_multi, _vc_cleanup, use_first=use_first)
            if not clean_vocal:
                return False
            clone_pre_cleaned = True
        else:
            if use_first:
                print("Warning: 'first' keyword ignored (only one reference provided)")
            if not os.path.exists(clone_path) and not is_youtube_url(clone_path):
                print(f"Error: Clone source not found: {clone_path}")
                return False
            resolved_audio, cleanup = resolve_target_to_audio(clone_path)
            if resolved_audio is None:
                print("Error: Could not resolve clone source")
                return False
            _vc_cleanup.extend(cleanup)
            clean_vocal = svs_extract_vocals(resolved_audio)
            if clean_vocal != resolved_audio and clean_vocal not in _vc_cleanup:
                _vc_cleanup.append(clean_vocal)
            if resolved_audio not in _vc_cleanup and resolved_audio != clean_vocal:
                _vc_cleanup.append(resolved_audio)
        reference_audio = None
        _target_vals = params.get('target', [])
        if _target_vals:
            if len(_target_vals) >= 2:
                ref_type = _target_vals[0].lower()
                ref_path_raw = _target_vals[1]
                if ref_type not in ('voice', 'music'):
                    ref_type = 'asis'
                    ref_path_raw = _target_vals[0]
            else:
                ref_type = 'asis'
                ref_path_raw = _target_vals[0]
            _vc_tr, ref_path = _parse_ref_time_spec(ref_path_raw)
            if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
                print(f"Error: Reference target not found: {ref_path}")
                for f in _vc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            resolved_ref, ref_cleanup = resolve_target_to_audio(ref_path)
            if resolved_ref is None:
                print("Error: Could not resolve reference target")
                for f in _vc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            _vc_cleanup.extend(ref_cleanup)
            if _vc_tr:
                resolved_ref = _extract_ref_segments(resolved_ref, _vc_tr, 30, _vc_cleanup)
            if ref_type == 'voice':
                processed_ref = svs_extract_vocals(resolved_ref)
            elif ref_type == 'music':
                processed_ref = svs_extract_music(resolved_ref)
            else:
                processed_ref = resolved_ref
            if processed_ref and processed_ref != resolved_ref:
                if processed_ref not in _vc_cleanup:
                    _vc_cleanup.append(processed_ref)
            if resolved_ref not in _vc_cleanup and resolved_ref != processed_ref:
                _vc_cleanup.append(resolved_ref)
            reference_audio = processed_ref
            if ref_type == 'asis':
                print("Using reference audio for music generation (as-is)")
            else:
                print(f"Using reference audio for music generation: {ref_type}")
        print("Loading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_ttm_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_clone_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            print(f"Generating music ({duration}s duration)...")
            if reference_audio:
                print(f"Using reference audio: {reference_audio}")
            success = ace_step.generate(
                lyrics=lyrics,
                style_prompt=style,
                output_path=temp_ttm_output.name,
                duration=duration,
                reference_audio=reference_audio
            )
            if not success:
                print("Error: Music generation failed")
                return False
            print("Extracting vocals from TTM output...")
            ttm_vocals = svs_extract_vocals(temp_ttm_output.name)
            if ttm_vocals and ttm_vocals != temp_ttm_output.name:
                _vc_cleanup.append(ttm_vocals)
            else:
                ttm_vocals = temp_ttm_output.name
            print("Extracting music from TTM output...")
            ttm_music = svs_extract_music(temp_ttm_output.name)
            if ttm_music and ttm_music != temp_ttm_output.name:
                _vc_cleanup.append(ttm_music)
            else:
                ttm_music = None
            print("Resampling TTM vocals to 44100Hz...")
            waveform_vocals, sr_vocals = torchaudio.load(ttm_vocals)
            if sr_vocals != 44100:
                resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
                waveform_vocals = resampler_vocals(waveform_vocals)
            torchaudio.save(temp_ttm_44k.name, waveform_vocals, 44100)
            print("Resampling clone voice to 44100Hz...")
            waveform_clone, sr_clone = torchaudio.load(clean_vocal)
            if sr_clone != 44100:
                resampler_clone = torchaudio.transforms.Resample(sr_clone, 44100)
                waveform_clone = resampler_clone(waveform_clone)
            torchaudio.save(temp_clone_44k.name, waveform_clone, 44100)
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
                source_path=temp_ttm_44k.name,
                reference_path=temp_clone_44k.name,
                output_path=temp_vc_output.name
            )
            if not vc_success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ttm_music:
                print("Mixing converted vocals with TTM music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_vc_output.name}" -i "{ttm_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_vc_output.name, output_path)
            else:
                print("Saving output...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                shutil.copy(temp_vc_output.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_ttm_output.name, temp_ttm_44k.name, temp_clone_44k.name, temp_vc_output.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    if _is_remix:
        _remix_entries = params.get('remix_entries', [])
        if not _remix_entries:
            print("Error: Remix requires at least one source path")
            return False
        if 'styling' not in params or len(params['styling']) != 1:
            print("Error: TTM remix requires 'styling' parameter")
            return False
        style = params['styling'][0].replace('\\n', '\n')
        _remix_lyrics = "..."
        if 'lyrics' in params and len(params['lyrics']) == 1:
            _remix_lyrics = params['lyrics'][0].replace('\\n', '\n')
        _remix_cleanup = []
        resolved_audio, src_cl = _compose_sources(_remix_entries, results_dir)
        if resolved_audio is None:
            print("Error: Could not resolve remix source(s)")
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        _remix_cleanup.extend(src_cl)
        _remix_ref_entries = params.get('ref_entries', [])
        _remix_ref_audio = None
        if _remix_ref_entries:
            _remix_ref_audio, ref_cl = _compose_refs(_remix_ref_entries, results_dir)
            _remix_cleanup.extend(ref_cl)
        _cover_strength = 0.4
        _bias_raw = params.get('bias_val')
        if _bias_raw is not None:
            try:
                _bv = int(_bias_raw)
                if 0 <= _bv <= 100:
                    if _bv == 0 or _bv == 100:
                        _cover_strength = _bv / 100.0
                    elif _bv % 10 == 5:
                        _cover_strength = (_bv - 5) / 100.0
                    else:
                        _cover_strength = (round(_bv / 10) * 10) / 100.0
            except (ValueError, TypeError):
                pass
        original_name = os.path.splitext(os.path.basename(_remix_entries[0][1]))[0]
        original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
        print("Loading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        try:
            print("Generating remix...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_ttm_remix_{original_name}_{timestamp}.wav")
            success = ace_step.cover(
                src_audio=resolved_audio,
                style_prompt=style,
                output_path=output_path,
                cover_strength=_cover_strength,
                reference_audio=_remix_ref_audio,
                lyrics=_remix_lyrics
            )
            if not success:
                print("Error: Remix generation failed")
                return False
            print(f"\n✓ Success! Output saved to: {output_path}")
            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    _is_repaint = params.get('is_repaint', False)
    _repaint_path = params.get('repaint_path')
    _time_range = params.get('time_range')
    _repaint_multipass = params.get('repaint_multipass')
    _repaint_source_prefix = params.get('repaint_source_prefix')

    if _is_repaint:
        if not _repaint_path:
            print("Error: Repaint requires a source path")
            return False
        if not os.path.exists(_repaint_path) and not is_youtube_url(_repaint_path):
            print(f"Error: Repaint source not found: {_repaint_path}")
            return False
        _repaint_cleanup = []
        resolved_audio, cleanup = resolve_target_to_audio(_repaint_path)
        if resolved_audio is None:
            print("Error: Could not resolve repaint source")
            return False
        _repaint_cleanup.extend(cleanup)
        if _repaint_source_prefix:
            _svs_timestamp = time.strftime("%Y%m%d_%H%M%S")
            if _repaint_source_prefix == 'voice':
                _svs_result = svs_extract_vocals(resolved_audio)
            else:
                _svs_result = svs_extract_music(resolved_audio)
            if _svs_result and _svs_result != resolved_audio:
                _repaint_cleanup.append(_svs_result)
                resolved_audio = _svs_result
        if _repaint_multipass:
            _parsed_passes = []
            for _pi, _spec in enumerate(_repaint_multipass):
                _parsed, _err = _parse_repaint_pass_spec(_spec)
                if _err:
                    print(f"Error: Repaint pass {_pi + 1}: {_err}")
                    for f in _repaint_cleanup:
                        if f and os.path.exists(f):
                            try: os.unlink(f)
                            except: pass
                    return False
                _parsed_passes.append(_parsed)
            original_name = os.path.splitext(os.path.basename(_repaint_path))[0]
            original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
            print("Loading ACE-Step model...")
            ace_step = AceStepWrapper(use_overdose=use_overdose)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            _current_source = resolved_audio
            _last_output = None
            _intermediate_files = []
            _total_passes = len(_parsed_passes)
            try:
                for _pi, _pass in enumerate(_parsed_passes):
                    _start_sec = _pass['start']
                    _end_sec = _pass['end']
                    try:
                        import soundfile as sf
                        _audio_info = sf.info(_current_source)
                        _max_duration = _audio_info.duration
                    except Exception:
                        print(f"Error: Could not read audio duration for pass {_pi + 1}")
                        return False
                    if _start_sec > _max_duration:
                        print(f"Error: Pass {_pi + 1}: Start time {_start_sec}s exceeds audio duration {_max_duration:.1f}s")
                        return False
                    if _end_sec > _max_duration:
                        print(f"Pass {_pi + 1}: End time {_end_sec}s exceeds audio duration, clamping to {_max_duration:.1f}s")
                        _end_sec = _max_duration
                    if _start_sec >= _end_sec:
                        print(f"Error: Pass {_pi + 1}: Start time must be less than end time after clamping")
                        return False
                    _pass_style = (_pass.get('styling') or '...')
                    _pass_lyrics = (_pass.get('lyrics') or '...')
                    _cover_strength = 0.4
                    if _pass.get('bias') is not None:
                        try:
                            _bv = int(_pass['bias'])
                            if 0 <= _bv <= 100:
                                if _bv == 0 or _bv == 100:
                                    _cover_strength = _bv / 100.0
                                elif _bv % 10 == 5:
                                    _cover_strength = (_bv - 5) / 100.0
                                else:
                                    _cover_strength = (round(_bv / 10) * 10) / 100.0
                        except (ValueError, TypeError):
                            pass
                    _pass_ref_audio = None
                    if _pass.get('references'):
                        _pass_ref_audio, _pass_ref_cl = _compose_refs(_pass['references'], results_dir)
                        _repaint_cleanup.extend(_pass_ref_cl)
                    _start_int = int(_start_sec)
                    _end_int = int(_end_sec)
                    _pass_num = _pi + 1
                    print(f"Repainting pass {_pass_num}/{_total_passes}: {_start_int}s - {_end_int}s...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if _pi < _total_passes - 1:
                        output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_pass{_pass_num}_{timestamp}.wav")
                    else:
                        output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_{timestamp}.wav")
                    success = ace_step.repaint(
                        src_audio=_current_source,
                        style_prompt=_pass_style,
                        output_path=output_path,
                        repaint_start=_start_sec,
                        repaint_end=_end_sec,
                        lyrics=_pass_lyrics,
                        cover_strength=_cover_strength,
                        reference_audio=_pass_ref_audio
                    )
                    if not success:
                        print(f"Error: Repaint pass {_pass_num} failed")
                        return False
                    if _last_output:
                        _intermediate_files.append(_last_output)
                    _last_output = output_path
                    _current_source = output_path
                for f in _intermediate_files:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                print(f"\n✓ Success! Output saved to: {_last_output}")
                del ace_step
                ace_step = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            finally:
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
        else:
            if _time_range is None:
                print("Error: Repaint requires time range (e.g., time:20-80)")
                return False
            if 'styling' not in params or len(params['styling']) != 1:
                print("Error: TTM repaint requires 'styling' parameter")
                return False
            _time_parts = _time_range.split('-')
            if len(_time_parts) != 2:
                print(f"Error: Invalid time format '{_time_range}', expected time:start-end")
                return False
            try:
                _start_sec = float(_time_parts[0].strip())
                _end_sec = float(_time_parts[1].strip())
            except ValueError:
                print(f"Error: Time values must be numbers, got '{_time_range}'")
                return False
            if _start_sec < 0:
                print("Error: Start time cannot be negative")
                return False
            if _end_sec <= 0:
                print("Error: End time must be greater than 0")
                return False
            if _start_sec == _end_sec:
                print("Error: Start and end time cannot be the same")
                return False
            if _start_sec >= _end_sec:
                print("Error: Start time must be less than end time")
                return False
            style = params['styling'][0].replace('\\n', '\n')
            _lyrics_content = "..."
            if 'lyrics' in params and len(params['lyrics']) == 1:
                _lyrics_content = params['lyrics'][0].replace('\\n', '\n')
            _rp_ref_entries = params.get('ref_entries', [])
            _repaint_ref_audio = None
            if _rp_ref_entries:
                _repaint_ref_audio, ref_cl = _compose_refs(_rp_ref_entries, results_dir)
                _repaint_cleanup.extend(ref_cl)
            try:
                import soundfile as sf
                _audio_info = sf.info(resolved_audio)
                _max_duration = _audio_info.duration
            except Exception:
                print("Error: Could not read audio duration")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            if _start_sec > _max_duration:
                print(f"Error: Start time {_start_sec}s exceeds audio duration {_max_duration:.1f}s")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            if _end_sec > _max_duration:
                print(f"End time {_end_sec}s exceeds audio duration, clamping to {_max_duration:.1f}s")
                _end_sec = _max_duration
            _cover_strength = 0.4
            _bias_raw = params.get('bias_val')
            if _bias_raw is not None:
                try:
                    _bv = int(_bias_raw)
                    if 0 <= _bv <= 100:
                        if _bv == 0 or _bv == 100:
                            _cover_strength = _bv / 100.0
                        elif _bv % 10 == 5:
                            _cover_strength = (_bv - 5) / 100.0
                        else:
                            _cover_strength = (round(_bv / 10) * 10) / 100.0
                except (ValueError, TypeError):
                    pass
            original_name = os.path.splitext(os.path.basename(_repaint_path))[0]
            original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
            print("Loading ACE-Step model...")
            ace_step = AceStepWrapper(use_overdose=use_overdose)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            try:
                _start_int = int(_start_sec)
                _end_int = int(_end_sec)
                print(f"Repainting {_start_int}s - {_end_int}s...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_{timestamp}.wav")
                success = ace_step.repaint(
                    src_audio=resolved_audio,
                    style_prompt=style,
                    output_path=output_path,
                    repaint_start=_start_sec,
                    repaint_end=_end_sec,
                    lyrics=_lyrics_content,
                    cover_strength=_cover_strength,
                    reference_audio=_repaint_ref_audio
                )
                if not success:
                    print("Error: Repaint generation failed")
                    return False
                print(f"\n✓ Success! Output saved to: {output_path}")
                del ace_step
                ace_step = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            finally:
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass

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
    _ttm_cleanup = []
    reference_audio = None
    _target_vals = params.get('target', [])
    if _target_vals:
        if len(_target_vals) >= 2:
            ref_type = _target_vals[0].lower()
            ref_path_raw = _target_vals[1]
            if ref_type not in ('voice', 'music'):
                ref_type = 'asis'
                ref_path_raw = _target_vals[0]
        else:
            ref_type = 'asis'
            ref_path_raw = _target_vals[0]
        _ttm_tr, ref_path = _parse_ref_time_spec(ref_path_raw)
        if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
            print(f"Error: Reference target not found: {ref_path}")
            return False
        resolved_audio, cleanup = resolve_target_to_audio(ref_path)
        if resolved_audio is None:
            print("Error: Could not resolve reference target")
            return False
        _ttm_cleanup.extend(cleanup)
        if _ttm_tr:
            resolved_audio = _extract_ref_segments(resolved_audio, _ttm_tr, 30, _ttm_cleanup)
        if ref_type == 'voice':
            processed = svs_extract_vocals(resolved_audio)
        elif ref_type == 'music':
            processed = svs_extract_music(resolved_audio)
        else:
            processed = resolved_audio
        if processed != resolved_audio and processed not in _ttm_cleanup:
            _ttm_cleanup.append(processed)
        if resolved_audio not in _ttm_cleanup and resolved_audio != processed:
            _ttm_cleanup.append(resolved_audio)
        reference_audio = processed
    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper(use_overdose=use_overdose)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    try:
        print(f"Generating music ({duration}s duration)...")
        if reference_audio:
            print(f"Using reference audio: {reference_audio}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=output_path,
            duration=duration,
            reference_audio=reference_audio
        )
        if not success:
            print("Error: Music generation failed")
            return False
        print(f"\n✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_complete(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    use_vocals = params.get('use_vocals', False)
    use_music = params.get('use_music', False)
    use_source = params.get('use_source', False)
    want_video = params.get('want_video', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Complete task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    sfx_specs_raw = params.get('sfx_specs', [])
    has_instruments = bool(instruments_raw)
    has_sfx = bool(sfx_specs_raw)

    if not has_instruments and not has_sfx:
        print('Error: Complete task requires instruments and/or "sfx:" specs (e.g., add "drums bass" or "sfx:thunder/10-5")')
        return False

    track_classes = None
    if has_instruments:
        track_classes, unknown = resolve_acestep_tracks(instruments_raw)
        if unknown is not None and len(unknown) > 0:
            print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
            print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
            print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
            return False
        if track_classes is None:
            print("Error: No valid tracks specified")
            return False

    noblend = params.get('noblend', False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    video_path = None
    downloaded_video = None
    is_video_source = False

    ext = os.path.splitext(source_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        is_video_source = True

    if is_youtube_url(source_path):
        if want_video:
            print(f"Downloading video from URL: {source_path}")
            downloaded_video, video_title = download_youtube_video(source_path, results_dir)
            if downloaded_video is None:
                print(f"Error: {video_title}")
                return False
            video_path = downloaded_video
            is_video_source = True
            original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
            temp_audio = os.path.join(results_dir, f'_ttm_complete_dl_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from downloaded video")
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            print(f"Downloading audio from URL: {source_path}")
            resolved, cleanup = resolve_target_to_audio(source_path)
            if resolved is None:
                print("Error: Could not download audio from URL")
                return False
            _cleanup.extend(cleanup)
            source_audio = resolved
    elif is_video_source:
        video_path = source_path
        temp_audio = os.path.join(results_dir, f'_ttm_complete_vid_{timestamp}.wav')
        ret = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
        if ret != 0 or not os.path.exists(temp_audio):
            print("Error: Failed to extract audio from video")
            return False
        source_audio = temp_audio
        _cleanup.append(temp_audio)
    else:
        valid, msg = validate_audio_file(source_path)
        if not valid:
            print(f"Error: {msg}")
            return False
        source_audio = source_path

    source_duration = _get_audio_duration(source_audio)

    sfx_specs = None
    if sfx_specs_raw:
        sfx_specs, sfx_err = _parse_sfx_specs(sfx_specs_raw, source_duration)
        if sfx_err:
            print(f"Error: {sfx_err}")
            return False

    actual_source = source_audio
    if use_vocals:
        print("Extracting vocals via SVS...")
        vocals = svs_extract_vocals(source_audio)
        if vocals != source_audio and vocals not in _cleanup:
            _cleanup.append(vocals)
        actual_source = vocals
    elif use_music:
        print("Extracting music (removing vocals) via SVS...")
        music = svs_extract_music(source_audio)
        if music != source_audio and music not in _cleanup:
            _cleanup.append(music)
        actual_source = music
    else:
        print("Using source audio as-is")

    if use_source and not use_vocals and not use_music:
        print("Warning: usrc has no effect without voice or music (only one source to blend with)")
        use_source = False

    reference_audio = None
    _ref_entries = params.get('ref_entries', [])
    if _ref_entries:
        reference_audio, ref_cl = _compose_refs(_ref_entries, results_dir)
        _cleanup.extend(ref_cl)

    try:
        output_ext = '.wav'
        if want_video and video_path:
            output_ext = '.mp4'
        elif want_video and not video_path:
            print("Warning: 'video' specified but source is an audio file (not video). Outputting as WAV.")
        _noblend_tag = '_noblend_' if noblend else ''
        _usrc_tag = '_usrc_' if use_source else ''
        output_filename = f'voder_ttm_complete_{original_name}{_noblend_tag}{_usrc_tag}{timestamp}{output_ext}'
        output_path = os.path.join(results_dir, output_filename)

        blended_path = actual_source

        if has_instruments:
            print(f"Tracks to add: {', '.join(track_classes)}")
            print("Loading ACE-Step XL-Base model (complete task)...")
            print("Note: Complete task uses the base model (50 inference steps), this may take a while...")

            ace_step = AceStepWrapper(use_overdose=True, complete_mode=True)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model for complete task")
                for f in _cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            temp_gen_wav = os.path.join(results_dir, f'_ttm_complete_gen_{timestamp}.wav')
            print(f"Completing track (adding {len(track_classes)} instruments)...")
            _styling = None
            if 'styling' in params and len(params['styling']) == 1:
                _styling = params['styling'][0].replace('\\n', '\n')
            success = ace_step.complete(
                src_audio=actual_source,
                track_classes=track_classes,
                output_path=temp_gen_wav,
                styling=_styling,
                reference_audio=reference_audio
            )

            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not success or not os.path.exists(temp_gen_wav):
                if os.path.exists(temp_gen_wav):
                    try:
                        os.unlink(temp_gen_wav)
                    except:
                        pass
                print("Error: Complete generation failed")
                return False

            if noblend:
                blended_path = temp_gen_wav
                _cleanup.append(temp_gen_wav)
            else:
                blend_source = source_audio if use_source else actual_source
                if use_source:
                    print("Blending completed audio with original source (usrc)...")
                else:
                    print("Blending completed audio with source...")
                temp_blend_wav = os.path.join(results_dir, f'_ttm_complete_blend_{timestamp}.wav')
                ret = os.system(f'ffmpeg -y -i "{temp_gen_wav}" -i "{blend_source}" -filter_complex amix=inputs=2:duration=longest "{temp_blend_wav}" 2>/dev/null')
                if os.path.exists(temp_gen_wav):
                    try:
                        os.unlink(temp_gen_wav)
                    except:
                        pass
                if ret == 0 and os.path.exists(temp_blend_wav):
                    blended_path = temp_blend_wav
                    _cleanup.append(temp_blend_wav)
                else:
                    print("Warning: Blend failed, using generated audio as-is")
                    blended_path = temp_gen_wav
                    _cleanup.append(temp_gen_wav)
                    if os.path.exists(temp_blend_wav):
                        try:
                            os.unlink(temp_blend_wav)
                        except:
                            pass

        if sfx_specs:
            print(f"Applying {len(sfx_specs)} SFX overlay(s)...")
            sfx_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sfx_temp.close()
            sfx_ok = _generate_and_overlay_sfx(blended_path, sfx_specs, sfx_temp.name)
            if sfx_ok and os.path.exists(sfx_temp.name):
                blended_path = sfx_temp.name
                _cleanup.append(sfx_temp.name)
            else:
                if os.path.exists(sfx_temp.name):
                    try:
                        os.unlink(sfx_temp.name)
                    except:
                        pass
                print("Warning: SFX overlay failed, using audio without SFX")

        if want_video and video_path:
            print("Merging audio with video...")
            mux_cmd = ['ffmpeg', '-i', video_path, '-i', blended_path,
                        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                        '-shortest', '-y', output_path]
            mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
            if mux_result.returncode != 0:
                print(f"Error: Video muxing failed: {mux_result.stderr}")
                return False
        else:
            shutil.copy2(blended_path, output_path)

        print(f"\nSuccess! Output saved to: {output_path}")
        return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        if downloaded_video and os.path.exists(downloaded_video):
            try:
                os.remove(downloaded_video)
            except:
                pass

def oneline_ttm_lego(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    use_vocals = params.get('use_vocals', False)
    use_music = params.get('use_music', False)
    mix_mode = params.get('mix_mode', False)
    blend_mode = params.get('blend_mode', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Lego task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    if not instruments_raw:
        print('Error: Lego task requires instruments (e.g., make "drums bass" or make "everything")')
        return False

    track_classes, unknown = resolve_acestep_tracks(instruments_raw)
    if unknown is not None and len(unknown) > 0:
        print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
        print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
        print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
        return False
    if track_classes is None:
        print("Error: No valid tracks specified")
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    if is_youtube_url(source_path):
        print(f"Downloading audio from URL: {source_path}")
        resolved, cleanup = resolve_target_to_audio(source_path)
        if resolved is None:
            print("Error: Could not download audio from URL")
            return False
        _cleanup.extend(cleanup)
        source_audio = resolved
    else:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            temp_audio = os.path.join(results_dir, f'_lego_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{source_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from video")
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            valid, msg = validate_audio_file(source_path)
            if not valid:
                print(f"Error: {msg}")
                return False
            source_audio = source_path

    actual_source = source_audio
    if use_vocals:
        print("Extracting vocals via SVS...")
        vocals = svs_extract_vocals(source_audio)
        if vocals != source_audio and vocals not in _cleanup:
            _cleanup.append(vocals)
        actual_source = vocals
    elif use_music:
        print("Extracting music (removing vocals) via SVS...")
        music = svs_extract_music(source_audio)
        if music != source_audio and music not in _cleanup:
            _cleanup.append(music)
        actual_source = music
    else:
        print("Using source audio as-is")

    _ref_entries = params.get('ref_entries', [])
    track_set = set(track_classes)
    stem_refs = {}
    fallback_ref = None
    if _ref_entries:
        ref_cache = {}
        num_ref_entries = len(_ref_entries)
        lego_slot_max = 30 // max(1, num_ref_entries)
        for entry in _ref_entries:
            sv_type = entry[0]
            raw = entry[1]
            lego_tr = entry[2] if len(entry) > 2 else None
            stem_name, ref_path = parse_ref_raw(raw)
            cache_key = (ref_path, sv_type)
            if cache_key not in ref_cache:
                if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
                    print(f"Warning: Reference not found: {ref_path}, skipping")
                    continue
                if is_youtube_url(ref_path):
                    print(f"Downloading reference audio from URL: {ref_path}")
                    resolved_ref, ref_cl = resolve_target_to_audio(ref_path)
                    if resolved_ref is None:
                        print(f"Warning: Could not download reference, skipping")
                        continue
                    _cleanup.extend(ref_cl)
                    ref_audio = resolved_ref
                else:
                    r_ext = os.path.splitext(ref_path)[1].lower()
                    if r_ext in VIDEO_EXTENSIONS:
                        ref_temp = os.path.join(results_dir, f'_lego_ref_vid_{timestamp}.wav')
                        ret = os.system(f'ffmpeg -y -i "{ref_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{ref_temp}" 2>/dev/null')
                        if ret != 0 or not os.path.exists(ref_temp):
                            print("Warning: Failed to extract audio from reference video, skipping")
                            continue
                        ref_audio = ref_temp
                        _cleanup.append(ref_temp)
                    else:
                        valid, msg = validate_audio_file(ref_path)
                        if not valid:
                            print(f"Warning: Invalid reference file: {msg}")
                            continue
                        ref_audio = ref_path
                if lego_tr:
                    ref_audio = _extract_ref_segments(ref_audio, lego_tr, lego_slot_max, _cleanup)
                if sv_type == 'voice':
                    ref_processed = svs_extract_vocals(ref_audio)
                elif sv_type == 'music':
                    ref_processed = svs_extract_music(ref_audio)
                else:
                    ref_processed = ref_audio
                if ref_processed and ref_processed != ref_audio:
                    if ref_processed not in _cleanup:
                        _cleanup.append(ref_processed)
                if ref_audio not in _cleanup and ref_audio != ref_processed:
                    _cleanup.append(ref_audio)
                ref_cache[cache_key] = ref_processed
            resolved_audio = ref_cache[cache_key]
            if stem_name is None:
                fallback_ref = resolved_audio
            else:
                if stem_name == 'everything':
                    stems = sorted(VALID_ACESTEP_TRACKS)
                elif stem_name == 'instruments':
                    stems = sorted(ACESTEP_INSTRUMENT_TRACKS)
                elif stem_name == 'voices':
                    stems = sorted(ACESTEP_VOICE_TRACKS)
                elif stem_name in track_set:
                    stems = [stem_name]
                else:
                    continue
                for s in stems:
                    if s in track_set:
                        stem_refs[s] = resolved_audio
        if stem_refs or fallback_ref is not None:
            refd = list(stem_refs.keys())
            if fallback_ref is not None:
                unrefd = [t for t in track_classes if t not in stem_refs]
                if unrefd:
                    print(f"References loaded: {len(refd)} specific, fallback for {len(unrefd)} more")
                else:
                    print(f"References loaded: {len(refd)} specific")
            else:
                print(f"References loaded: {len(refd)} specific, no fallback")

    print(f"Tracks to generate ({len(track_classes)}): {', '.join(track_classes)}")
    print("Loading ACE-Step XL-Base model (lego task)...")

    ace_step = AceStepWrapper(use_overdose=False, complete_mode=True)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model for lego task")
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False

    _styling = None
    if 'styling' in params and len(params['styling']) == 1:
        _styling = params['styling'][0].replace('\\n', '\n')

    try:
        generated_files = []
        all_succeeded = True
        for idx, track in enumerate(track_classes):
            track_ref = stem_refs.get(track, fallback_ref)
            if track_ref:
                print(f"\n[{idx+1}/{len(track_classes)}] Generating {track} (with reference)...")
            else:
                print(f"\n[{idx+1}/{len(track_classes)}] Generating {track}...")
            temp_output = os.path.join(results_dir, f'_lego_tmp_{track}_{timestamp}.wav')
            success = ace_step.lego(
                src_audio=actual_source,
                track_name=track,
                output_path=temp_output,
                styling=_styling,
                reference_audio=track_ref
            )
            if success:
                generated_files.append(temp_output)
                print(f"  {track} generated successfully")
            else:
                print(f"  Failed to generate {track}")
                all_succeeded = False
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass

        if not generated_files:
            print("Error: No tracks were generated successfully")
            return False

        if not all_succeeded:
            print(f"Warning: {len(track_classes) - len(generated_files)}/{len(track_classes)} tracks failed to generate")

        if mix_mode or blend_mode:
            mode_label = "blend" if blend_mode else "mix"
            print(f"\n{mode_label.capitalize()}ing {len(generated_files)} tracks...")
            mix_output = os.path.join(results_dir, f'_lego_{mode_label}_raw_{timestamp}.wav')
            input_list = " ".join(f'-i "{f}"' for f in generated_files)
            ret = os.system(f'ffmpeg -y {input_list} -filter_complex amix=inputs={len(generated_files)}:duration=longest "{mix_output}" 2>/dev/null')
            if ret != 0 or not os.path.exists(mix_output):
                print(f"Error: Failed to {mode_label} tracks")
                for f in generated_files:
                    if os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                return False

            if blend_mode:
                blend_output = os.path.join(results_dir, f'_lego_blend_src_{timestamp}.wav')
                ret = os.system(f'ffmpeg -y -i "{mix_output}" -i "{actual_source}" -filter_complex amix=inputs=2:duration=longest "{blend_output}" 2>/dev/null')
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                if ret != 0 or not os.path.exists(blend_output):
                    print("Error: Failed to blend mixed tracks with source")
                    for f in generated_files:
                        if os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    if os.path.exists(blend_output):
                        try:
                            os.unlink(blend_output)
                        except:
                            pass
                    return False
                output_filename = f'voder_ttm_lego_blend_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(blend_output, output_path)
            else:
                output_filename = f'voder_ttm_lego_mix_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(mix_output, output_path)

            for f in generated_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

            print(f"\nSuccess! Output saved to: {output_path}")
            return True
        else:
            exported = []
            for idx, f in enumerate(generated_files):
                track = track_classes[idx]
                output_filename = f'voder_ttm_lego_{track}_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(f, output_path)
                exported.append(output_path)

            print(f"\nSuccess! {len(exported)} track(s) exported:")
            for p in exported:
                print(f"  {p}")
            return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step
        ace_step = None
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_extract(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    extract_mix = params.get('extract_mix', False)
    extract_only = params.get('extract_only', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Extract task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    if not instruments_raw:
        print('Error: Extract task requires stems (e.g., stems "drums bass" or stems "everything")')
        return False

    track_classes, unknown = resolve_acestep_tracks(instruments_raw)
    if unknown is not None and len(unknown) > 0:
        print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
        print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
        print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
        return False
    if track_classes is None:
        print("Error: No valid tracks specified")
        return False

    if extract_only:
        if len(track_classes) >= 12:
            print("Error: 'only' cannot be used with 'everything' or all 12 stems (nothing would remain)")
            return False
        specified_set = set(track_classes)
        all_tracks = sorted(VALID_ACESTEP_TRACKS)
        track_classes = [t for t in all_tracks if t not in specified_set]
        removed_names = sorted(specified_set)
        print(f"Only mode: removing {', '.join(removed_names)}, extracting {len(track_classes)} remaining stems")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    if is_youtube_url(source_path):
        print(f"Downloading audio from URL: {source_path}")
        resolved, cleanup = resolve_target_to_audio(source_path)
        if resolved is None:
            print("Error: Could not download audio from URL")
            return False
        _cleanup.extend(cleanup)
        source_audio = resolved
    else:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            temp_audio = os.path.join(results_dir, f'_extract_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{source_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from video")
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            valid, msg = validate_audio_file(source_path)
            if not valid:
                print(f"Error: {msg}")
                return False
            source_audio = source_path

    print(f"Tracks to extract ({len(track_classes)}): {', '.join(track_classes)}")
    print("Loading ACE-Step XL-Base model (extract task)...")

    ace_step = AceStepWrapper(use_overdose=False, complete_mode=True)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model for extract task")
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False

    try:
        generated_files = []
        all_succeeded = True
        for idx, track in enumerate(track_classes):
            print(f"\n[{idx+1}/{len(track_classes)}] Extracting {track}...")
            temp_output = os.path.join(results_dir, f'_extract_tmp_{track}_{timestamp}.wav')
            success = ace_step.extract(
                src_audio=source_audio,
                track_name=track,
                output_path=temp_output
            )
            if success:
                generated_files.append(temp_output)
                print(f"  {track} extracted successfully")
            else:
                print(f"  Failed to extract {track}")
                all_succeeded = False
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass

        if not generated_files:
            print("Error: No tracks were extracted successfully")
            return False

        if not all_succeeded:
            print(f"Warning: {len(track_classes) - len(generated_files)}/{len(track_classes)} tracks failed to extract")

        if extract_mix or extract_only:
            mode_label = "only" if extract_only else "mix"
            print(f"\nMixing {len(generated_files)} extracted tracks ({mode_label})...")
            mix_output = os.path.join(results_dir, f'_extract_{mode_label}_raw_{timestamp}.wav')
            input_list = " ".join(f'-i "{f}"' for f in generated_files)
            ret = os.system(f'ffmpeg -y {input_list} -filter_complex amix=inputs={len(generated_files)}:duration=longest "{mix_output}" 2>/dev/null')
            if ret != 0 or not os.path.exists(mix_output):
                print(f"Error: Failed to mix extracted tracks")
                for f in generated_files:
                    if os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                return False

            if extract_only:
                output_filename = f'voder_ttm_extract_{original_name}_{timestamp}.wav'
            else:
                output_filename = f'voder_ttm_extract_mix_{original_name}_{timestamp}.wav'
            output_path = os.path.join(results_dir, output_filename)
            shutil.move(mix_output, output_path)

            for f in generated_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

            print(f"\nSuccess! Output saved to: {output_path}")
            return True
        else:
            exported = []
            for idx, f in enumerate(generated_files):
                track = track_classes[idx]
                output_filename = f'voder_ttm_extract_{track}_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(f, output_path)
                exported.append(output_path)

            print(f"\nSuccess! {len(exported)} track(s) extracted:")
            for p in exported:
                print(f"  {p}")
            return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step
        ace_step = None
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_bgm(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    bgm_source = params.get('bgm_source', '')
    if not bgm_source:
        print("Error: bgm requires a source path (audio/video file or URL)")
        return False

    music_params_list = params.get('music', [])
    music_description = None
    if music_params_list:
        music_description = music_params_list[-1]
        if music_description:
            music_description = music_description.strip()

    sfx_specs_raw = params.get('sfx_specs', [])

    if not music_description and not sfx_specs_raw:
        print('Error: bgm requires a music description and/or "sfx:" specs')
        return False

    level = 35
    level_list = params.get('level', [])
    if level_list:
        try:
            lv = int(level_list[-1])
            if lv < 0 or lv > 100:
                print("Error: level must be between 0 and 100")
                return False
            level = lv
        except (ValueError, TypeError):
            print("Error: level must be a number between 0 and 100")
            return False

    use_overdose = params.get('overdose', False)
    want_video = params.get('want_video', False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cleanup_files = []
    original_video_path = None
    downloaded_video = None
    video_title = None

    try:
        is_link = is_youtube_url(bgm_source)
        is_video_file = False
        if not is_link and os.path.exists(bgm_source):
            ext = os.path.splitext(bgm_source)[1].lower()
            is_video_file = ext in VIDEO_EXTENSIONS
            if is_video_file:
                original_video_path = bgm_source

        if is_link and want_video:
            print(f"Downloading video from URL: {bgm_source}")
            downloaded_video, video_title = download_youtube_video(bgm_source, results_dir)
            if downloaded_video is None:
                print(f"Error: {video_title}")
                return False
            original_video_path = downloaded_video
            temp_audio = os.path.join(results_dir, f'_bgm_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{downloaded_video}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from downloaded video")
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False
            source_audio = temp_audio
            cleanup_files.append(temp_audio)
        elif want_video and not original_video_path:
            print("Warning: 'video' specified but source is an audio file (not video). Outputting as WAV.")
            print(f"Resolving source: {bgm_source}")
            source_audio, source_cleanup = resolve_target_to_audio(bgm_source)
            if source_audio is None:
                print("Error: Could not resolve source to audio")
                return False
            cleanup_files.extend(source_cleanup)
        else:
            print(f"Resolving source: {bgm_source}")
            source_audio, source_cleanup = resolve_target_to_audio(bgm_source)
            if source_audio is None:
                print("Error: Could not resolve source to audio")
                return False
            cleanup_files.extend(source_cleanup)

        source_duration = _get_audio_duration(source_audio)

        sfx_specs = None
        if sfx_specs_raw:
            sfx_specs, sfx_err = _parse_sfx_specs(sfx_specs_raw, source_duration)
            if sfx_err:
                print(f"Error: {sfx_err}")
                return False

        print("Cleaning source audio through SVS voice pipe...")
        clean_voice = svs_extract_vocals(source_audio)
        if clean_voice != source_audio:
            cleanup_files.append(clean_voice)

        voice_duration = _get_audio_duration(clean_voice)
        print(f"Clean voice duration: {voice_duration:.2f}s")

        reference_audio = None
        _bgm_ref_entries = params.get('ref_entries', [])
        if _bgm_ref_entries:
            reference_audio, ref_cl = _compose_refs(_bgm_ref_entries, results_dir)
            cleanup_files.extend(ref_cl)

        mixed_path = None

        if music_description:
            print("Loading ACE-Step model...")
            ace = AceStepWrapper(use_overdose=use_overdose)
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False

            print(f"Generating background music (description: \"{music_description}\")...")
            music_result = generate_background_music(ace, music_description, voice_duration, reference_audio=reference_audio)
            del ace
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if music_result is None:
                print("Error: Background music generation failed")
                return False
            music_temp_path, music_temp_dir = music_result

            vol = level / 100.0
            print(f"Mixing clean voice with music (volume: {level}%)...")
            mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            mixed_temp.close()
            cmd = [
                'ffmpeg', '-i', clean_voice, '-i', music_temp_path,
                '-filter_complex', f'[1:a]volume={vol:.2f}[music];[0:a][music]amix=inputs=2:duration=longest',
                '-y', mixed_temp.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if music_temp_dir is not None:
                shutil.rmtree(music_temp_dir, ignore_errors=True)
            if result.returncode != 0:
                print(f"Error: FFmpeg mixing failed: {result.stderr}")
                try:
                    os.unlink(mixed_temp.name)
                except:
                    pass
                return False
            mixed_path = mixed_temp.name
            cleanup_files.append(mixed_path)
        else:
            mixed_path = clean_voice

        if sfx_specs:
            print(f"Applying {len(sfx_specs)} SFX overlay(s)...")
            sfx_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sfx_temp.close()
            sfx_ok = _generate_and_overlay_sfx(mixed_path, sfx_specs, sfx_temp.name)
            if sfx_ok and os.path.exists(sfx_temp.name):
                mixed_path = sfx_temp.name
                cleanup_files.append(sfx_temp.name)
            else:
                if os.path.exists(sfx_temp.name):
                    try:
                        os.unlink(sfx_temp.name)
                    except:
                        pass
                print("Warning: SFX overlay failed, using audio without SFX")

        if original_video_path and os.path.exists(original_video_path):
            if downloaded_video and video_title:
                name = video_title.replace(' ', '_').replace('/', '_')[:50]
            else:
                name = os.path.splitext(os.path.basename(original_video_path))[0]
            out_ext = os.path.splitext(original_video_path)[1]
            if not out_ext:
                out_ext = '.mp4'
            output_path = os.path.join(results_dir, f"voder_ttm_bgm_{name}_{timestamp}{out_ext}")
            print("Muxing mixed audio into video...")
            final_temp = tempfile.NamedTemporaryFile(suffix=out_ext, delete=False)
            final_temp.close()
            mux_cmd = [
                'ffmpeg', '-i', original_video_path, '-i', mixed_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', final_temp.name
            ]
            mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
            if mux_result.returncode != 0:
                print(f"Error: Video muxing failed: {mux_result.stderr}")
                try:
                    os.unlink(final_temp.name)
                except:
                    pass
                return False
            shutil.move(final_temp.name, output_path)
        else:
            if is_link:
                name = "audio"
            else:
                name = os.path.splitext(os.path.basename(bgm_source))[0]
                if not name:
                    name = "audio"
            output_path = os.path.join(results_dir, f"voder_ttm_bgm_{name}_{timestamp}.wav")
            shutil.copy2(mixed_path, output_path)

        print(f"✓ Success! Output saved to: {output_path}")
        return True
    finally:
        if downloaded_video and os.path.exists(downloaded_video):
            try:
                os.remove(downloaded_video)
            except:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        for f in cleanup_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_stt(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])
    keep_timestamp = params.get('timestamp', False)
    enable_dialogue = params.get('dialogue', False)
    enable_translate = params.get('translate', False)
    use_overdose = params.get('overdose', False)

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

        do_translate = enable_translate
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

                print(f"\u2713 Success! Output saved to: {output_path}")
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

        use_se = params.get('se', False)

        bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
        if bs_roformer_lib not in sys.path:
            sys.path.insert(0, bs_roformer_lib)
        bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
        if bs_roformer_pkg not in sys.path:
            sys.path.insert(0, bs_roformer_pkg)

        svs_temp = None
        if not is_image:
            print("Stage 1: SVS voice isolation (BS-RoFormer)...")
            from bs_roformer import BSRoformerSeparator
            svs_separator = BSRoformerSeparator(SVS_DIR)
            svs_separator.ensure_model(stem='voice')
            if svs_separator.vocals_model is None:
                print("Error: Failed to load BS-RoFormer vocals model")
                svs_separator.cleanup()
                del svs_separator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            svs_temp_dir = tempfile.mkdtemp()
            svs_temp = os.path.join(svs_temp_dir, f'_stt_svs_{timestamp}.wav')
            svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
            svs_separator.cleanup()
            del svs_separator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if svs_ok and os.path.exists(svs_temp):
                audio_path = svs_temp
            else:
                print("Warning: SVS voice isolation failed, using original audio")
                if svs_temp_dir:
                    shutil.rmtree(svs_temp_dir, ignore_errors=True)

        se_temp = None
        if use_se and not is_image:
            print("Stage 2: Speech Enhancement (UniSE SE)...")
            from unise import UniSEEnhancer
            se_enhancer = UniSEEnhancer(UNISE_DIR)
            se_enhancer.ensure_model()
            if se_enhancer.model is None:
                print("Error: Failed to load UniSE SE model")
                se_enhancer.cleanup()
                del se_enhancer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if svs_temp and os.path.exists(svs_temp):
                    shutil.rmtree(os.path.dirname(svs_temp), ignore_errors=True)
                continue
            se_temp_dir = tempfile.mkdtemp()
            se_temp = os.path.join(se_temp_dir, f'_stt_se_{timestamp}.wav')
            se_ok = se_enhancer.enhance(audio_path, se_temp)
            se_enhancer.cleanup()
            del se_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if se_ok and os.path.exists(se_temp):
                audio_path = se_temp
            else:
                print("Warning: Speech Enhancement failed, using previous audio")
                if se_temp_dir:
                    shutil.rmtree(se_temp_dir, ignore_errors=True)

        try:
            if use_overdose and not do_translate and not is_image:
                asr = VibeVoiceASR()
                asr.ensure_model()
                if asr.model is None:
                    print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                    asr.cleanup()
                    del asr
                    use_overdose = False

            if use_overdose and not do_translate and not is_image:
                print("Transcribing with VibeVoice ASR...")
                asr_segments = asr.transcribe(audio_path)
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not asr_segments:
                    print(f"Error: ASR transcription returned no segments for {file_path}")
                    continue

                def format_time_range(start, end):
                    def format_single(seconds):
                        if seconds is None:
                            seconds = 0
                        minutes = int(seconds // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds % 1) * 100)
                        return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                    return f"[{format_single(start)}-{format_single(end)}]"

                if enable_dialogue:
                    original_speakers = []
                    for seg in asr_segments:
                        speaker = seg["speaker"]
                        if speaker not in original_speakers:
                            original_speakers.append(speaker)
                    speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}
                    lines = []
                    current_speaker_num = None
                    current_text_parts = []
                    current_first_time = None
                    current_last_time = None
                    for seg in asr_segments:
                        speaker_num = speaker_mapping[seg["speaker"]]
                        text = seg.get("text", "")
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
                            if current_text_parts:
                                content = " ".join(current_text_parts)
                                if len(original_speakers) == 1:
                                    if keep_timestamp:
                                        lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content}")
                                    else:
                                        lines.append(f"text: {content}")
                                else:
                                    if keep_timestamp:
                                        lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                    else:
                                        lines.append(f"{current_speaker_num}: {content}")
                            current_speaker_num = speaker_num
                            current_text_parts = [text]
                            current_first_time = seg_start
                            current_last_time = seg_end
                    if current_text_parts:
                        content = " ".join(current_text_parts)
                        if len(original_speakers) == 1:
                            if keep_timestamp:
                                lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content}")
                            else:
                                lines.append(f"text: {content}")
                        else:
                            if keep_timestamp:
                                lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                            else:
                                lines.append(f"{current_speaker_num}: {content}")
                    formatted_text = "\n".join(lines)
                elif keep_timestamp:
                    lines = []
                    for seg in asr_segments:
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "").strip()
                        if text:
                            lines.append(f"{format_time_range(start, end)} text: {text}")
                    if lines:
                        formatted_text = "\n".join(lines)
                    else:
                        formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
                else:
                    formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
            else:
                print("Loading Whisper model...")
                stt = WhisperSTT()
                if stt.model is None:
                    print("Error: Failed to load Whisper model")
                    continue

                if do_translate and enable_dialogue:
                    print("Transcribing audio (for diarization)...")
                    original_result = stt.transcribe(audio_path)
                    if not original_result:
                        print(f"Error: Transcription failed for {file_path}")
                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    print("Translating audio to English...")
                    result = stt.translate(audio_path)
                    if not result:
                        print("Error: Translation failed, using original transcription")
                        result = original_result
                        do_translate = False

                    del stt
                    stt = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                elif do_translate:
                    print("Translating audio to English...")
                    result = stt.translate(audio_path)
                    if not result:
                        print(f"Error: Translation failed for {file_path}")
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
                else:
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

            if not use_overdose:
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
                        if do_translate:
                            diarization_segments = diarization.format_diarization(diar_result, original_result)
                        else:
                            diarization_segments = diarization.format_diarization(diar_result, result)

                        formatted_segments = None
                        if diarization_segments:
                            if do_translate:
                                translated_segments = result.get("segments", [])
                                speaker_time_map = []
                                for ds in diarization_segments:
                                    speaker_time_map.append({
                                        "speaker": ds["speaker"],
                                        "start": ds.get("start", 0),
                                        "end": ds.get("end", 0),
                                        "text": ds["text"]
                                    })

                                merged_segments = []
                                for ts in translated_segments:
                                    ts_start = ts.get("start", 0)
                                    ts_end = ts.get("end", 0)
                                    ts_text = ts.get("text", "").strip()
                                    if not ts_text:
                                        continue
                                    best_speaker = None
                                    best_overlap = 0
                                    for sm in speaker_time_map:
                                        overlap_start = max(ts_start, sm["start"])
                                        overlap_end = min(ts_end, sm["end"])
                                        overlap = max(0, overlap_end - overlap_start)
                                        if overlap > best_overlap:
                                            best_overlap = overlap
                                            best_speaker = sm["speaker"]
                                    if best_speaker is not None:
                                        merged_segments.append({
                                            "speaker": best_speaker,
                                            "start": ts_start,
                                            "end": ts_end,
                                            "text": ts_text
                                        })
                                formatted_segments = merged_segments if merged_segments else None
                            else:
                                formatted_segments = diarization_segments

                        if formatted_segments:
                            original_speakers = []
                            for seg in formatted_segments:
                                speaker = seg["speaker"]
                                if speaker not in original_speakers:
                                    original_speakers.append(speaker)

                            speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

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
                                        if current_text_parts:
                                            content = " ".join(current_text_parts)
                                            if keep_timestamp:
                                                lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                            else:
                                                lines.append(f"{current_speaker_num}: {content}")
                                        current_speaker_num = speaker_num
                                        current_text_parts = [text]
                                        current_first_time = seg_start
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
            if do_translate:
                suffix_parts.append("translate")
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
                    parent_dir = os.path.dirname(audio_path)
                    os.unlink(audio_path)
                    if os.path.exists(parent_dir) and os.path.basename(parent_dir).startswith('_'):
                        shutil.rmtree(parent_dir, ignore_errors=True)
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

    resolved_files = []
    _se_cleanup = []
    for file_path in files:
        if is_youtube_url(file_path):
            print(f"Downloading video from URL: {file_path}")
            dl_path, dl_err = download_youtube_video(file_path)
            if not dl_path:
                print(f"Error: {dl_err}")
                for f in _se_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            _se_cleanup.append(dl_path)
            resolved_files.append(dl_path)
        elif os.path.exists(file_path):
            resolved_files.append(file_path)
        else:
            print(f"Error: File not found: {file_path}")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

    print("Loading UniSE Speech Enhancement model...")
    from unise import UniSEEnhancer
    enhancer = UniSEEnhancer(UNISE_DIR)
    enhancer.ensure_model()
    if enhancer.model is None:
        print("Error: Failed to load UniSE model")
        for f in _se_cleanup:
            if f and os.path.exists(f):
                try: os.unlink(f)
                except: pass
        return False

    success_count = 0
    for file_path in resolved_files:
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

    for f in _se_cleanup:
        if f and os.path.exists(f):
            try: os.unlink(f)
            except: pass

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
    return success_count > 0

def _ss_resolve_input(file_path, results_dir, timestamp):
    audio_path = None
    cleanup_list = []
    original_name = None
    is_url = is_youtube_url(file_path)

    if is_url:
        print(f"Downloading audio from URL...")
        success_dl, error_msg, downloaded_path = download_youtube_audio(file_path)
        if not success_dl:
            return None, None, None, cleanup_list, f"Download failed: {error_msg}"
        audio_path = downloaded_path
        cleanup_list.append(audio_path)
        original_name = "download"
    elif os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        original_name = os.path.splitext(os.path.basename(file_path))[0][:50]
        if is_video:
            temp_audio = os.path.join(results_dir, f'_ss_input_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{file_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                return None, None, None, cleanup_list, "Failed to extract audio from video"
            audio_path = temp_audio
            cleanup_list.append(audio_path)
        else:
            try:
                torchaudio.load(file_path)
                audio_path = file_path
            except Exception:
                return None, None, None, cleanup_list, f"Could not read audio file: {file_path}"
    else:
        return None, None, None, cleanup_list, f"File not found: {file_path}"

    return audio_path, original_name, is_url, cleanup_list, None

def _ss_run_pipeline(audio_path, use_se, results_dir, original_name, timestamp, target_path=None, use_overdose=False):
    all_outputs = []
    temp_dirs = []

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)
    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    print("Stage 1: SVS voice isolation (BS-RoFormer)...")
    from bs_roformer import BSRoformerSeparator
    svs_separator = BSRoformerSeparator(SVS_DIR)
    svs_separator.ensure_model(stem='voice')
    if svs_separator.vocals_model is None:
        print("Error: Failed to load BS-RoFormer vocals model")
        svs_separator.cleanup()
        del svs_separator
        return None

    svs_temp_dir = tempfile.mkdtemp()
    temp_dirs.append(svs_temp_dir)
    svs_temp = os.path.join(svs_temp_dir, f'_ss_svs_{timestamp}.wav')
    svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
    svs_separator.cleanup()
    del svs_separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not svs_ok or not os.path.exists(svs_temp):
        print("Error: SVS voice isolation failed")
        return None

    clean_source = svs_temp

    if target_path and os.path.exists(target_path):
        print("Stage 2: Target-based extraction (UniSE TSE)...")
        from unise import UniSEEnhancer
        tse_enhancer = UniSEEnhancer(UNISE_DIR)
        tse_enhancer.ensure_model()
        if tse_enhancer.model is None:
            print("Error: Failed to load UniSE TSE model")
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        output_filename = f"voder_ss_{original_name}_{timestamp}_extracted.wav"
        tse_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(tse_temp_dir)
        tse_temp_path = os.path.join(tse_temp_dir, output_filename)

        print(f"  Extracting target voice from source using reference...")
        tse_ok = tse_enhancer.tse_extract(clean_source, target_path, tse_temp_path)
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_se and tse_ok and os.path.exists(tse_temp_path):
            print("Applying Speech Enhancement to extracted voice...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp = os.path.join(tse_temp_dir, f"se_{output_filename}")
                se_ok = se_enh.enhance(tse_temp_path, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, tse_temp_path)
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if tse_ok and os.path.exists(tse_temp_path):
            final_path = os.path.join(results_dir, output_filename)
            shutil.copy2(tse_temp_path, final_path)
            all_outputs.append(final_path)
            print(f"  Extracted voice saved to: {final_path}")
        else:
            print(f"  Warning: TSE extraction failed for target voice")

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        return all_outputs if all_outputs else None

    if use_overdose:
        print("Stage 2: Transcription + Speaker Diarization (VibeVoice ASR)...")
        asr = VibeVoiceASR()
        asr.ensure_model()
        if asr.model is None:
            print("Error: Failed to load VibeVoice ASR model")
            asr.cleanup()
            del asr
            return None

        asr_segments = asr.transcribe(clean_source)

        if not asr_segments:
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Error: VibeVoice ASR transcription returned no segments")
            return None

        formatted = asr_segments
    else:
        print("Stage 2: Speaker Diarization (pyannote)...")
        diarization = SpeakerDiarization()
        if diarization.pipeline is None:
            print("Error: Speaker diarization model not available (HF_TOKEN required)")
            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        diar_full = diarization.diarize_full(clean_source)
        if diar_full is None:
            print("Error: Speaker diarization failed")
            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        if hasattr(diar_full, 'exclusive_speaker_diarization'):
            exclusive_diar = diar_full.exclusive_speaker_diarization
            inclusive_diar = diar_full.speaker_diarization
        else:
            exclusive_diar = diar_full
            inclusive_diar = diar_full

        formatted = []
        for turn in inclusive_diar.itertracks(yield_label=True):
            segment, track, speaker = turn
            formatted.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker,
                "text": ""
            })

        exclusive_segments = {}
        for turn in exclusive_diar.itertracks(yield_label=True):
            segment, track, speaker = turn
            if speaker not in exclusive_segments:
                exclusive_segments[speaker] = []
            exclusive_segments[speaker].append({
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end) - float(segment.start)
            })

        for spk in exclusive_segments:
            exclusive_segments[spk].sort(key=lambda x: x["duration"], reverse=True)

    if not formatted:
        print("Error: No speaker segments found")
        return None

    speaker_segments = {}
    for seg in formatted:
        spk = seg["speaker"]
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})

    for spk in speaker_segments:
        segs = speaker_segments[spk]
        segs.sort(key=lambda x: x["start"])
        merged = []
        for s in segs:
            if merged and s["start"] - merged[-1]["end"] < 0.3:
                merged[-1]["end"] = s["end"]
                merged[-1]["text"] += " " + s["text"]
            else:
                merged.append({"start": s["start"], "end": s["end"], "text": s["text"]})
        speaker_segments[spk] = merged

    first_speaker_order = sorted(speaker_segments.keys(), key=lambda spk: speaker_segments[spk][0]["start"])
    sorted_speakers = first_speaker_order
    num_speakers = len(sorted_speakers)
    print(f"Detected {num_speakers} speaker(s)")

    if num_speakers < 2:
        print("Only one speaker detected. No separation needed.")
        if use_overdose:
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for spk in sorted_speakers:
            segs = speaker_segments[spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            dur = longest["end"] - longest["start"]
            print(f"  Speaker 1: {len(segs)} segments, longest: {dur:.1f}s")
        output_filename = f"voder_ss_{original_name}_{timestamp}_speaker1.wav"
        single_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(single_temp_dir)
        single_temp = os.path.join(single_temp_dir, output_filename)
        shutil.copy2(clean_source, single_temp)

        if use_se:
            print("Applying Speech Enhancement to extracted voice...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp = os.path.join(single_temp_dir, f"se_{output_filename}")
                se_ok = se_enh.enhance(single_temp, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, single_temp)
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_path = os.path.join(results_dir, output_filename)
        shutil.copy2(single_temp, final_path)
        all_outputs.append(final_path)
        print(f"Output saved to: {final_path}")

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        return all_outputs

    speaker_to_num = {}
    for idx, spk in enumerate(sorted_speakers, 1):
        speaker_to_num[spk] = idx

    for spk in sorted_speakers:
        segs = speaker_segments[spk]
        longest = max(segs, key=lambda x: x["end"] - x["start"])
        dur = longest["end"] - longest["start"]
        print(f"  Speaker {speaker_to_num[spk]}: {len(segs)} segments, longest: {dur:.1f}s")

    if use_overdose:
        print("Stage 3: Target Speaker Extraction (UniSE TSE)...")
        from unise import UniSEEnhancer
        tse_enhancer = UniSEEnhancer(UNISE_DIR)
        tse_enhancer.ensure_model()
        if tse_enhancer.model is None:
            print("Error: Failed to load UniSE TSE model")
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        tse_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(tse_temp_dir)
        speaker_temp_files = {}

        for spk in sorted_speakers:
            spk_num = speaker_to_num[spk]
            segs = speaker_segments[spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            start_t = longest["start"]
            dur_t = longest["end"] - longest["start"]

            if dur_t > 5.0:
                mid = start_t + dur_t / 2.0
                start_t = mid - 2.5
                dur_t = 5.0
                if start_t < 0:
                    start_t = 0.0

            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass0.wav")
            cmd = [
                'ffmpeg', '-i', clean_source,
                '-ss', str(start_t),
                '-t', str(dur_t),
                '-ar', '16000', '-ac', '1',
                '-y', enroll_clip
            ]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0 or not os.path.exists(enroll_clip):
                print(f"  Warning: Failed to cut enrollment clip for speaker {spk_num}, skipping")
                continue

            current_enroll = enroll_clip
            current_source = clean_source
            max_passes = 3
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None

            for pass_idx in range(1, max_passes + 1):
                pass_output = os.path.join(tse_temp_dir, f"spk{spk_num}_pass{pass_idx}.wav")
                print(f"  Speaker {spk_num} — Pass {pass_idx}: extracting with enrollment from {os.path.basename(current_enroll)}...")

                tse_ok = tse_enhancer.tse_extract(current_source, current_enroll, pass_output)
                if not tse_ok or not os.path.exists(pass_output):
                    print(f"  Warning: TSE extraction failed for speaker {spk_num} pass {pass_idx}")
                    if last_good_pass:
                        shutil.copy2(last_good_pass, speaker_temp)
                    break

                last_good_pass = pass_output

                recheck = asr.transcribe(pass_output)
                if recheck is None:
                    print(f"  Speaker {spk_num} — VibeVoice re-check failed, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                recheck_speakers = set()
                for seg in recheck:
                    recheck_speakers.add(seg.get("speaker"))

                if len(recheck_speakers) <= 1:
                    print(f"  Speaker {spk_num} — Clean! VibeVoice confirms single speaker after pass {pass_idx}")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                print(f"  Speaker {spk_num} — Still {len(recheck_speakers)} speakers detected, refining...")

                longest_seg = max(recheck, key=lambda x: x["end"] - x["start"])
                ls = longest_seg["start"]
                ld = longest_seg["end"] - longest_seg["start"]

                if ld > 5.0:
                    mid = ls + ld / 2.0
                    ls = mid - 2.5
                    ld = 5.0
                    if ls < 0:
                        ls = 0.0

                next_enroll = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass{pass_idx}.wav")
                cmd = [
                    'ffmpeg', '-i', pass_output,
                    '-ss', str(ls),
                    '-t', str(ld),
                    '-ar', '16000', '-ac', '1',
                    '-y', next_enroll
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(next_enroll):
                    print(f"  Speaker {spk_num} — Failed to cut refined enrollment, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                current_enroll = next_enroll
                current_source = clean_source

                if pass_idx == max_passes:
                    print(f"  Speaker {spk_num} — Max passes reached, using final result")
                    shutil.copy2(pass_output, speaker_temp)

            if os.path.exists(speaker_temp):
                speaker_temp_files[spk_num] = (speaker_temp, output_filename)
            else:
                print(f"  Warning: No output for speaker {spk_num}")

        asr.cleanup()
        del asr
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not speaker_temp_files:
            print("Error: Failed to extract any speakers")
            for td in temp_dirs:
                try:
                    shutil.rmtree(td)
                except Exception:
                    pass
            return None

        if use_se and speaker_temp_files:
            print("Applying Speech Enhancement to extracted voices...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp_dir = tempfile.mkdtemp()
                for spk_num, (temp_f, fname) in speaker_temp_files.items():
                    se_tmp = os.path.join(se_tmp_dir, f"se_{fname}")
                    se_ok = se_enh.enhance(temp_f, se_tmp)
                    if se_ok and os.path.exists(se_tmp):
                        shutil.copy2(se_tmp, temp_f)
                try:
                    shutil.rmtree(se_tmp_dir)
                except Exception:
                    pass
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for spk_num, (temp_f, fname) in speaker_temp_files.items():
            final_path = os.path.join(results_dir, fname)
            shutil.copy2(temp_f, final_path)
            all_outputs.append(final_path)

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        print(f"\n{'=' * 60}")
        print(f"Separated {len(all_outputs)} speaker(s) successfully:")
        for p in all_outputs:
            print(f"  {os.path.basename(p)}")

        return all_outputs

    else:
        print("Stage 3: Target Speaker Extraction (UniSE TSE)...")
        from unise import UniSEEnhancer
        tse_enhancer = UniSEEnhancer(UNISE_DIR)
        tse_enhancer.ensure_model()
        if tse_enhancer.model is None:
            print("Error: Failed to load UniSE TSE model")
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        tse_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(tse_temp_dir)
        speaker_temp_files = {}

        for spk in sorted_speakers:
            spk_num = speaker_to_num[spk]
            clean_segs = exclusive_segments.get(spk, [])

            enroll_parts = []
            collected = 0.0
            target_enroll = 5.0

            for seg in clean_segs:
                if collected >= target_enroll:
                    break
                remaining = target_enroll - collected
                take_dur = min(seg["duration"], remaining)
                enroll_parts.append({
                    "start": seg["start"],
                    "duration": take_dur
                })
                collected += take_dur

            if not enroll_parts:
                segs = speaker_segments[spk]
                longest = max(segs, key=lambda x: x["end"] - x["start"])
                start_t = longest["start"]
                dur_t = longest["end"] - longest["start"]
                if dur_t > 5.0:
                    mid = start_t + dur_t / 2.0
                    start_t = mid - 2.5
                    dur_t = 5.0
                    if start_t < 0:
                        start_t = 0.0
                enroll_parts.append({"start": start_t, "duration": dur_t})
                collected = dur_t

            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass0.wav")

            if len(enroll_parts) == 1:
                part = enroll_parts[0]
                cmd = [
                    'ffmpeg', '-i', clean_source,
                    '-ss', str(part["start"]),
                    '-t', str(part["duration"]),
                    '-ar', '16000', '-ac', '1',
                    '-y', enroll_clip
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Warning: Failed to cut enrollment clip for speaker {spk_num}, skipping")
                    continue
            else:
                part_files = []
                for pi, part in enumerate(enroll_parts):
                    part_file = os.path.join(tse_temp_dir, f"enroll_{spk_num}_part{pi}.wav")
                    cmd = [
                        'ffmpeg', '-i', clean_source,
                        '-ss', str(part["start"]),
                        '-t', str(part["duration"]),
                        '-ar', '16000', '-ac', '1',
                        '-y', part_file
                    ]
                    ret = subprocess.run(cmd, capture_output=True, text=True)
                    if ret.returncode != 0 or not os.path.exists(part_file):
                        continue
                    part_files.append(part_file)

                if not part_files:
                    print(f"  Warning: Failed to cut enrollment clips for speaker {spk_num}, skipping")
                    continue

                concat_list = os.path.join(tse_temp_dir, f"enroll_{spk_num}_concat.txt")
                with open(concat_list, 'w') as f:
                    for pf in part_files:
                        f.write(f"file '{pf}'\n")

                cmd = [
                    'ffmpeg', '-f', 'concat', '-safe', '0',
                    '-i', concat_list,
                    '-ar', '16000', '-ac', '1',
                    '-y', enroll_clip
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Warning: Failed to concatenate enrollment for speaker {spk_num}, skipping")
                    continue

            current_enroll = enroll_clip
            current_source = clean_source
            max_passes = 3
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None

            for pass_idx in range(1, max_passes + 1):
                pass_output = os.path.join(tse_temp_dir, f"spk{spk_num}_pass{pass_idx}.wav")
                print(f"  Speaker {spk_num} — Pass {pass_idx}: extracting...")

                tse_ok = tse_enhancer.tse_extract(current_source, current_enroll, pass_output)
                if not tse_ok or not os.path.exists(pass_output):
                    print(f"  Warning: TSE extraction failed for speaker {spk_num} pass {pass_idx}")
                    if last_good_pass:
                        shutil.copy2(last_good_pass, speaker_temp)
                    break

                last_good_pass = pass_output

                recheck = diarization.diarize_full(pass_output)
                if recheck is None:
                    print(f"  Speaker {spk_num} — Re-check failed, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                if hasattr(recheck, 'exclusive_speaker_diarization'):
                    recheck_excl = recheck.exclusive_speaker_diarization
                else:
                    recheck_excl = recheck

                recheck_speakers = set()
                for turn in recheck_excl.itertracks(yield_label=True):
                    _, _, speaker = turn
                    recheck_speakers.add(speaker)

                if len(recheck_speakers) <= 1:
                    print(f"  Speaker {spk_num} — Clean! Single speaker confirmed after pass {pass_idx}")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                print(f"  Speaker {spk_num} — Still {len(recheck_speakers)} speakers detected, refining...")

                recheck_segs = []
                for turn in recheck_excl.itertracks(yield_label=True):
                    segment, _, speaker = turn
                    dur = float(segment.end) - float(segment.start)
                    recheck_segs.append({"start": float(segment.start), "duration": dur, "speaker": speaker})

                recheck_segs.sort(key=lambda x: x["duration"], reverse=True)

                best_seg = recheck_segs[0]
                ls = best_seg["start"]
                ld = best_seg["duration"]

                if ld > 5.0:
                    mid = ls + ld / 2.0
                    ls = mid - 2.5
                    ld = 5.0
                    if ls < 0:
                        ls = 0.0

                next_enroll = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass{pass_idx}.wav")
                cmd = [
                    'ffmpeg', '-i', pass_output,
                    '-ss', str(ls),
                    '-t', str(ld),
                    '-ar', '16000', '-ac', '1',
                    '-y', next_enroll
                ]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(next_enroll):
                    print(f"  Speaker {spk_num} — Failed to cut refined enrollment, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                current_enroll = next_enroll
                current_source = clean_source

                if pass_idx == max_passes:
                    print(f"  Speaker {spk_num} — Max passes reached, using final result")
                    shutil.copy2(pass_output, speaker_temp)

            if os.path.exists(speaker_temp):
                speaker_temp_files[spk_num] = (speaker_temp, output_filename)
            else:
                print(f"  Warning: No output for speaker {spk_num}")

        diarization.pipeline = None
        del diarization
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not speaker_temp_files:
            print("Error: Failed to extract any speakers")
            for td in temp_dirs:
                try:
                    shutil.rmtree(td)
                except Exception:
                    pass
            return None

        if use_se and speaker_temp_files:
            print("Applying Speech Enhancement to extracted voices...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp_dir = tempfile.mkdtemp()
                for spk_num, (temp_f, fname) in speaker_temp_files.items():
                    se_tmp = os.path.join(se_tmp_dir, f"se_{fname}")
                    se_ok = se_enh.enhance(temp_f, se_tmp)
                    if se_ok and os.path.exists(se_tmp):
                        shutil.copy2(se_tmp, temp_f)
                try:
                    shutil.rmtree(se_tmp_dir)
                except Exception:
                    pass
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for spk_num, (temp_f, fname) in speaker_temp_files.items():
            final_path = os.path.join(results_dir, fname)
            shutil.copy2(temp_f, final_path)
            all_outputs.append(final_path)

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        print(f"\n{'=' * 60}")
        print(f"Separated {len(all_outputs)} speaker(s) successfully:")
        for p in all_outputs:
            print(f"  {os.path.basename(p)}")

        return all_outputs

def oneline_ss(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = params.get('file_path', '')
    use_se = params.get('use_se', False)
    target_path = params.get('target_path')
    use_overdose = params.get('overdose', False)

    if not file_path:
        print("Error: SS mode requires an audio/video file path or URL")
        return False

    if target_path and not os.path.exists(target_path) and not is_youtube_url(target_path):
        print(f"Error: Target file not found or invalid: {target_path}")
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("VODER SS - Speakers Separator")
    print("=" * 60)

    target_audio = target_path
    if target_path:
        if is_youtube_url(target_path):
            print("Downloading target audio from URL...")
            success_dl, error_msg, target_audio = download_youtube_audio(target_path)
            if not success_dl:
                print(f"Error: Target download failed: {error_msg}")
                return False
        elif target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from target video...")
            extracted = extract_audio_from_video_cli(target_path)
            if extracted:
                target_audio = extracted
            else:
                print("Error: Could not extract audio from target video")
                return False

    audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
    if err:
        print(f"Error: {err}")
        if target_audio and target_audio != target_path and os.path.exists(target_audio):
            try:
                os.unlink(target_audio)
            except:
                pass
        return False

    try:
        pipeline_outputs = _ss_run_pipeline(audio_path, use_se, results_dir, original_name, timestamp, target_audio, use_overdose)
        if pipeline_outputs is None:
            print("SS pipeline failed")
            return False
        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return False
    finally:
        for f in cleanup_list:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        if target_audio and target_audio != target_path and os.path.exists(target_audio):
            try:
                os.unlink(target_audio)
            except:
                pass

def cli_ss_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SS Mode ---")
    print("Speakers Separator - extract all speakers from audio one by one")
    print("Pipeline: SVS voice isolation -> STT + diarization -> TSE extraction")
    print("Supports: audio files, video files, YouTube/TikTok URLs")
    print()

    while True:
        file_path = input("Enter audio/video file path or URL: ").strip()
        if not file_path:
            print("Error: File path cannot be empty. Please try again.")
            continue
        if is_youtube_url(file_path):
            break
        if os.path.exists(file_path):
            try:
                torchaudio.load(file_path)
                break
            except Exception:
                video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
                if os.path.splitext(file_path)[1].lower() in video_exts:
                    break
                print("Error: Could not read audio file. Please try again.")
        else:
            print(f"Error: File not found: {file_path}")

    target_path = None
    while True:
        target_input = input("Enter target voice path (audio/video/URL, or Enter to skip): ").strip()
        if not target_input:
            break
        if is_youtube_url(target_input):
            target_path = target_input
            break
        if os.path.exists(target_input):
            target_path = target_input
            break
        print("Error: File not found or invalid path")

    use_overdose = False
    if not target_path:
        while True:
            overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
            if overdose_input in ['y', 'yes']:
                use_overdose = True
                break
            elif overdose_input in ['n', 'no']:
                use_overdose = False
                break
            else:
                print("Please enter Y or N")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print()

    if target_path:
        target_audio = target_path
        if is_youtube_url(target_path):
            print("Downloading target audio from URL...")
            success_dl, error_msg, target_audio = download_youtube_audio(target_path)
            if not success_dl:
                print(f"Warning: Target download failed, using path as-is: {error_msg}")
                target_audio = target_path
        elif target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from target video...")
            extracted = extract_audio_from_video_cli(target_path)
            if extracted:
                target_audio = extracted
            else:
                print("Warning: Could not extract audio from target video")

        audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
        if err:
            print(f"Error: {err}")
            if target_audio and target_audio != target_path and os.path.exists(target_audio):
                try:
                    os.unlink(target_audio)
                except:
                    pass
            return False

        try:
            pipeline_outputs = _ss_run_pipeline(audio_path, False, results_dir, original_name, timestamp, target_audio, use_overdose)
            if pipeline_outputs is None:
                print("SS pipeline failed")
                return False
            return True
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            return False
        finally:
            for f in cleanup_list:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
            if target_audio and target_audio != target_path and os.path.exists(target_audio):
                try:
                    os.unlink(target_audio)
                except:
                    pass
    else:
        audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
        if err:
            print(f"Error: {err}")
            return False

        try:
            pipeline_outputs = _ss_run_pipeline(audio_path, False, results_dir, original_name, timestamp, None, use_overdose)
            if pipeline_outputs is None:
                print("SS pipeline failed")
                return False
            return True
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            return False
        finally:
            for f in cleanup_list:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass

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

def oneline_svs(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    stem = params.get('stem', 'voice')
    file_path = params.get('file_path', '')

    if not file_path:
        print('Error: SVS mode requires an audio file path or URL')
        return False

    is_url = is_youtube_url(file_path)
    if not is_url and not os.path.exists(file_path):
        print(f'Error: File not found: {file_path}')
        return False

    stems_to_run = ['voice', 'music'] if stem == 'both' else [stem]
    stem_labels = {'voice': 'vocals', 'music': 'instruments', 'both': 'vocals and instruments'}
    print(f'Song Voice Separate - extracting {stem_labels.get(stem, stem)}')
    print(f'  Input: {file_path}')
    print('Loading BS-RoFormer Resurrection model...')

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)

    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    from bs_roformer import BSRoformerSeparator
    separator = BSRoformerSeparator(SVS_DIR)
    for s in stems_to_run:
        separator.ensure_model(stem=s)
    if 'voice' in stems_to_run and separator.vocals_model is None:
        print('Error: Failed to load vocals model')
        return False
    if 'music' in stems_to_run and separator.inst_model is None:
        print('Error: Failed to load instrumental model')
        return False

    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
        downloaded_video = None
        actual_file_path = file_path

        if is_url:
            downloaded_video, video_title = download_youtube_video(file_path, results_dir)
            if downloaded_video is None:
                print(f'Error: {video_title}')
                return False
            actual_file_path = downloaded_video
            original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
            is_video = True
        else:
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            input_ext = os.path.splitext(file_path)[1].lower()
            is_video = input_ext in video_exts

        temp_audio = None
        if is_video:
            print('Video detected, extracting audio...')
            temp_audio = os.path.join(results_dir, f'_svs_temp_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print('Error: Failed to extract audio from video')
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False

        audio_source = temp_audio if is_video else actual_file_path
        output_paths = []
        all_ok = True
        for s in stems_to_run:
            suffix = 'vocals' if s == 'voice' else 'instruments'
            output_filename = f'voder_svs_{original_name}_{timestamp}_{suffix}.mp4' if is_video else f'voder_svs_{original_name}_{timestamp}_{suffix}.wav'
            output_path = os.path.join(results_dir, output_filename)

            if is_video:
                temp_wav = os.path.join(results_dir, f'_svs_temp_{timestamp}_{suffix}.wav')
                success = separator.separate(audio_source, s, temp_wav)
                if success:
                    print(f'Merging {suffix} back into video...')
                    ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -i "{temp_wav}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                    if ret != 0 or not os.path.exists(output_path):
                        print(f'Error: Failed to merge {suffix} with video')
                        success = False
                        all_ok = False
                    else:
                        os.remove(temp_wav)
                        output_paths.append(output_path)
                else:
                    all_ok = False
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
            else:
                success = separator.separate(audio_source, s, output_path)
                if success:
                    output_paths.append(output_path)
                else:
                    all_ok = False

        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if downloaded_video and os.path.exists(downloaded_video):
            os.remove(downloaded_video)

        if output_paths:
            print(f'\nSuccess! {len(output_paths)} file(s) saved:')
            for p in output_paths:
                print(f'  {p}')
            return True
        else:
            print('Error: All separations failed')
            return False
    except Exception as e:
        traceback.print_exc()
        print(f'Error: {e}')
        return False
    finally:
        separator.cleanup()
        del separator
        separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

def cli_svs_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SVS Mode ---")
    print("Song Voice Separate - extract vocals or instrumental from a song")
    print("Model: BS-RoFormer Resurrection (best single-pass vocals SDR + instrumental SDR)")
    print()

    while True:
        file_path = input("Enter song, video file path, or YouTube/TikTok URL: ").strip()
        if not file_path:
            print("Error: File path cannot be empty. Please try again.")
            continue
        if is_youtube_url(file_path):
            break
        if os.path.exists(file_path):
            try:
                torchaudio.load(file_path)
                break
            except Exception:
                video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
                if os.path.splitext(file_path)[1].lower() in video_exts:
                    break
                print("Error: Could not read audio file. Please try again.")
        else:
            print(f"Error: File not found: {file_path}")

    while True:
        choice = input("Separate what? 1: Extract voice  2: Extract music: ").strip()
        if choice == '1':
            stem = 'voice'
            break
        elif choice == '2':
            stem = 'music'
            break
        else:
            print("Error: Please enter 1 or 2.")

    stem_label = 'vocals' if stem == 'voice' else 'instruments'
    if is_youtube_url(file_path):
        print(f"\nExtracting {stem_label} from: {file_path}")
    else:
        print(f"\nExtracting {stem_label} from: {os.path.basename(file_path)}")
    print("Loading BS-RoFormer Resurrection model (first run downloads ~390MB)...")

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)

    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    from bs_roformer import BSRoformerSeparator
    separator = BSRoformerSeparator(SVS_DIR)
    separator.ensure_model(stem=stem)
    if stem == 'voice' and separator.vocals_model is None:
        print("Error: Failed to load vocals model")
        return False
    if stem == 'music' and separator.inst_model is None:
        print("Error: Failed to load instrumental model")
        return False

    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        suffix = 'vocals' if stem == 'voice' else 'instruments'

        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
        downloaded_video = None
        actual_file_path = file_path
        is_url = is_youtube_url(file_path)

        if is_url:
            downloaded_video, video_title = download_youtube_video(file_path, results_dir)
            if downloaded_video is None:
                print(f'Error: {video_title}')
                return False
            actual_file_path = downloaded_video
            original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
            is_video = True
        else:
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            input_ext = os.path.splitext(file_path)[1].lower()
            is_video = input_ext in video_exts

        output_filename = f'voder_svs_{original_name}_{timestamp}_{suffix}.mp4' if is_video else f'voder_svs_{original_name}_{timestamp}_{suffix}.wav'
        output_path = os.path.join(results_dir, output_filename)

        temp_audio = None
        if is_video:
            print('Video detected, extracting audio...')
            temp_audio = os.path.join(results_dir, f'_svs_temp_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print('Error: Failed to extract audio from video')
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False

        if is_video:
            temp_wav = os.path.join(results_dir, f'_svs_temp_{timestamp}_{suffix}.wav')
            success = separator.separate(temp_audio, stem, temp_wav)
            if success:
                print('Merging separated audio back into video...')
                ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -i "{temp_wav}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print('Error: Failed to merge audio with video')
                    success = False
                else:
                    os.remove(temp_wav)
                    if temp_audio and os.path.exists(temp_audio):
                        os.remove(temp_audio)
            else:
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)
        else:
            success = separator.separate(actual_file_path, stem, output_path)

        if downloaded_video and os.path.exists(downloaded_video):
            os.remove(downloaded_video)

        if success:
            print(f'\nSuccess! Output saved to: {output_path}')
        else:
            print('Error: Separation failed')
        return success
    except Exception as e:
        traceback.print_exc()
        print(f'Error: {e}')
        return False
    finally:
        separator.cleanup()
        del separator
        separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def interactive_cli_mode():
    while True:
        print_banner()
        print("\nSelect Mode:")
        print("1. TTS (Text-to-Speech)")
        print("2. STS (Speech-to-Speech / Voice Conversion)")
        print("3. TTM (Text-to-Music)")
        print("4. SE (Speech Enhancement)")
        print("5. SFX (Sound Effects Generation)")
        print("6. SVS (Song Voice Separate)")
        print("7. STT (Speech-to-Text)")
        print("8. SS (Speakers Separator)")
        choice = input("\nEnter your choice (1-8): ").strip()
        success = False
        if choice == '1':
            success = cli_tts_mode()
        elif choice == '2':
            success = cli_sts_mode()
        elif choice == '3':
            success = cli_ttm_mode()
        elif choice == '4':
            success = cli_se_mode()
        elif choice == '5':
            success = cli_sfx_mode()
        elif choice == '6':
            success = cli_svs_mode()
        elif choice == '7':
            success = cli_stt_mode()
        elif choice == '8':
            success = cli_ss_mode()
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
    if mode is None:
        print(f"Error: Invalid mode '{parsed['mode']}'")
        show_oneline_usage()
        return False
    parsed['mode'] = mode
    return execute_oneline_command(parsed)



if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "cli" and len(sys.argv) == 2:
            interactive_cli_mode()
            sys.exit(0)
        arg_offset = 1
        if sys.argv[1] == "cli":
            arg_offset = 2
        if len(sys.argv) > arg_offset:
            result = parse_and_execute_oneline(sys.argv[arg_offset:])
            sys.exit(0 if result else 1)
    import gui
    gui.launch()
