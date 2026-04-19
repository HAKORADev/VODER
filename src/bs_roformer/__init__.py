import os
import sys
import time
import gc
import math
import yaml
import numpy as np
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import warnings

warnings.filterwarnings("ignore")

HUGGINGFACE_REPO = "pcunwa/BS-Roformer-Resurrection"

VOCALS_CKPT = "BS-Roformer-Resurrection.ckpt"
VOCALS_CONFIG = "BS-Roformer-Resurrection-Config.yaml"
INST_CKPT = "BS-Roformer-Resurrection-Inst.ckpt"
INST_CONFIG = "BS-Roformer-Resurrection-Inst-Config.yaml"

BEST_OVERLAP_VOCALS = 2
BEST_OVERLAP_INST = 2


class BSRoformerSeparator:
    """BS-RoFormer Resurrection model for music source separation (vocals/instrumental)."""

    def __init__(self, model_dir=None):
        self.model_dir = model_dir
        self.vocals_model = None
        self.inst_model = None
        self.vocals_config = None
        self.inst_config = None

    def _get_windowing_array(self, window_size, fade_size):
        # Crossfade window for overlap-add — matches the official demix implementation
        fade_in = torch.linspace(0, 1, fade_size)
        fade_out = torch.linspace(1, 0, fade_size)
        window = torch.ones(window_size)
        window[-fade_size:] = fade_out
        window[:fade_size] = fade_in
        return window

    def ensure_model(self, stem=None):
        """Download model files if missing, and load only the needed model.

        Args:
            stem: 'voice' or 'music'. If None, both models are loaded.
        """
        if self.model_dir is None:
            print("Error: model_dir not set")
            return False

        os.makedirs(self.model_dir, exist_ok=True)
        bs_roformer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib")
        if bs_roformer_path not in sys.path:
            sys.path.insert(0, bs_roformer_path)

        print("Checking BS-RoFormer Resurrection model files...")

        vocals_ckpt_path = os.path.join(self.model_dir, VOCALS_CKPT)
        vocals_config_path = os.path.join(self.model_dir, VOCALS_CONFIG)
        inst_ckpt_path = os.path.join(self.model_dir, INST_CKPT)
        inst_config_path = os.path.join(self.model_dir, INST_CONFIG)

        # Always download all files so both models are available
        missing = []
        if not os.path.exists(vocals_ckpt_path):
            missing.append(VOCALS_CKPT)
        if not os.path.exists(vocals_config_path):
            missing.append(VOCALS_CONFIG)
        if not os.path.exists(inst_ckpt_path):
            missing.append(INST_CKPT)
        if not os.path.exists(inst_config_path):
            missing.append(INST_CONFIG)

        if missing:
            print(f"Downloading missing model files: {', '.join(missing)}")
            success = self._download_model(missing)
            if not success:
                print("Error: Failed to download model files")
                return False

        try:
            # Only load the model that's needed for the requested stem
            if stem == "voice":
                if self.vocals_model is None:
                    print("Loading vocals model...")
                    self.vocals_config = self._load_config(vocals_config_path)
                    self.vocals_model = self._load_model(self.vocals_config, vocals_ckpt_path)
                    if self.vocals_model is None:
                        print("Error: Failed to load vocals model")
                        return False
                    print("Vocals model loaded successfully")
            elif stem == "music":
                if self.inst_model is None:
                    print("Loading instrumental model...")
                    self.inst_config = self._load_config(inst_config_path)
                    self.inst_model = self._load_model(self.inst_config, inst_ckpt_path)
                    if self.inst_model is None:
                        print("Error: Failed to load instrumental model")
                        return False
                    print("Instrumental model loaded successfully")
            else:
                # Load both
                if self.vocals_model is None:
                    print("Loading vocals model...")
                    self.vocals_config = self._load_config(vocals_config_path)
                    self.vocals_model = self._load_model(self.vocals_config, vocals_ckpt_path)
                    if self.vocals_model is None:
                        print("Error: Failed to load vocals model")
                        return False
                    print("Vocals model loaded successfully")

                if self.inst_model is None:
                    print("Loading instrumental model...")
                    self.inst_config = self._load_config(inst_config_path)
                    self.inst_model = self._load_model(self.inst_config, inst_ckpt_path)
                    if self.inst_model is None:
                        print("Error: Failed to load instrumental model")
                        self.vocals_model = None
                        return False
                    print("Instrumental model loaded successfully")

            return True

        except Exception as e:
            print(f"Error loading models: {e}")
            self.vocals_model = None
            self.inst_model = None
            return False

    def _download_model(self, missing_files):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("Error: huggingface_hub not installed. Run: pip install huggingface_hub")
            return False

        for filename in missing_files:
            try:
                print(f"  Downloading {filename}...")
                hf_hub_download(
                    repo_id=HUGGINGFACE_REPO,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
                print(f"  Downloaded {filename}")
            except Exception as e:
                print(f"  Error downloading {filename}: {e}")
                return False

        return True

    def _load_config(self, config_path):
        try:
            from ml_collections import ConfigDict
            with open(config_path, 'r') as f:
                config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
            return config
        except Exception as e:
            print(f"Error loading config {config_path}: {e}")
            return None

    def _load_model(self, config, ckpt_path):
        try:
            from models.bs_roformer.bs_roformer import BSRoformer
        except ImportError:
            try:
                import importlib
                spec = importlib.util.find_spec("models.bs_roformer.bs_roformer")
                if spec is None:
                    print("BS-RoFormer model code not found. Make sure the lib/ directory contains the model code.")
                    return None
            except Exception:
                print("BS-RoFormer model code not found.")
                return None

        try:
            from models.bs_roformer.bs_roformer import BSRoformer
            model = BSRoformer(**dict(config.model))
            checkpoint = torch.load(ckpt_path, weights_only=False, map_location="cpu")
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            return model
        except Exception as e:
            print(f"Error initializing model: {e}")
            return None

    def _demix(self, config, model, mix, device):
        """Core separation logic — ported from ZFTurbo demix() for bs_roformer.

        The BSRoformer model handles STFT/ISTFT internally. forward() signature:
            forward(raw_audio, target=None, active_stem_ids=None, return_loss_breakdown=False)
        Input: (batch, channels, time) -> Output: (batch, stems, channels, time)
        With num_stems=1, output shape is (batch, 1, 2, time).
        """
        if 'chunk_size' in config.inference:
            chunk_size = config.inference.chunk_size
        else:
            chunk_size = config.audio.chunk_size
        num_overlap = config.inference.num_overlap
        batch_size = config.inference.batch_size

        fade_size = chunk_size // 10
        step = chunk_size // num_overlap
        border = chunk_size - step
        length_init = mix.shape[-1]
        windowing_array = self._get_windowing_array(chunk_size, fade_size)

        # Add reflect padding for edge artifacts (same as official demix)
        mix_tensor = torch.tensor(mix, dtype=torch.float32)
        if length_init > 2 * border and border > 0:
            mix_tensor = nn.functional.pad(mix_tensor, (border, border), mode="reflect")

        target_instrument = config.training.target_instrument

        use_amp = getattr(config.training, 'use_amp', True)

        model.eval()
        with torch.cuda.amp.autocast(enabled=use_amp):
            with torch.inference_mode():
                result = torch.zeros(1, mix_tensor.shape[0], mix_tensor.shape[1], dtype=torch.float32)
                counter = torch.zeros(1, mix_tensor.shape[0], mix_tensor.shape[1], dtype=torch.float32)

                i = 0
                batch_data = []
                batch_locations = []
                total_chunks = math.ceil(mix_tensor.shape[1] / step)
                chunk_count = 0

                while i < mix_tensor.shape[1]:
                    part = mix_tensor[:, i:i + chunk_size]
                    chunk_len = part.shape[-1]
                    if chunk_len > chunk_size // 2:
                        pad_mode = "reflect"
                    else:
                        pad_mode = "constant"
                    part = nn.functional.pad(part, (0, chunk_size - chunk_len), mode=pad_mode, value=0)

                    batch_data.append(part)
                    batch_locations.append((i, chunk_len))
                    i += step

                    if len(batch_data) >= batch_size or i >= mix_tensor.shape[1]:
                        arr = torch.stack(batch_data, dim=0).to(device)
                        x = model(arr)

                        # x shape: (batch, stems, channels, time) — with num_stems=1 it's (batch, 1, channels, time)
                        if x.shape[1] == 1:
                            x = x[:, 0]  # squeeze stems dim -> (batch, channels, time)

                        window = windowing_array.clone()
                        if i - step == 0:
                            window[:fade_size] = 1
                        elif i >= mix_tensor.shape[1]:
                            window[-fade_size:] = 1

                        for j, (start, seg_len) in enumerate(batch_locations):
                            result[:, :, start:start + seg_len] += x[j, :, :seg_len].cpu() * window[:seg_len]
                            counter[:, :, start:start + seg_len] += window[:seg_len]

                        batch_data.clear()
                        batch_locations.clear()

                        chunk_count += 1
                        if chunk_count % 5 == 0 or i >= mix_tensor.shape[1]:
                            progress = min(chunk_count / total_chunks * 100, 100)
                            print(f"  Progress: {progress:.0f}%")

                estimated_source = result / counter
                estimated_source = estimated_source.cpu().numpy()
                np.nan_to_num(estimated_source, copy=False, nan=0.0)

                # Remove reflect padding
                if length_init > 2 * border and border > 0:
                    estimated_source = estimated_source[:, :, border:-border]

                # Shape is (1, channels, time) -> squeeze batch -> (channels, time)
                estimated_source = estimated_source[0]

                waveforms = {target_instrument: estimated_source}
                return waveforms

    def separate(self, input_path, stem, output_path, device=None):
        """Separate vocals or instrumental from an audio file.

        Args:
            input_path: Path to input audio file
            stem: 'voice' or 'music'
            output_path: Path to save the output WAV file
            device: 'cpu', 'cuda', or None (auto-detect)

        Returns:
            True on success, False on failure
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if stem not in ("voice", "music"):
            print("Error: stem must be 'voice' or 'music'")
            return False

        if stem == "voice" and self.vocals_model is None:
            print("Error: Vocals model not loaded")
            return False
        if stem == "music" and self.inst_model is None:
            print("Error: Instrumental model not loaded")
            return False

        if not os.path.exists(input_path):
            print(f"Error: File not found: {input_path}")
            return False

        print(f"Loading audio: {input_path}")
        try:
            mix, sr = librosa.load(input_path, sr=44100, mono=False)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return False

        if len(mix.shape) == 1:
            mix = np.stack([mix, mix])

        print(f"Audio: {mix.shape[-1] / sr:.1f}s, stereo")
        print(f"Device: {device}")
        print(f"Separating {stem}...")

        t0 = time.time()

        if stem == "voice":
            model = self.vocals_model
            config = self.vocals_config
        else:
            model = self.inst_model
            config = self.inst_config

        model = model.to(device)
        model.eval()

        try:
            waveforms = self._demix(config, model, mix, device)
        except Exception as e:
            print(f"Error during separation: {e}")
            model = model.to("cpu")
            return False

        model = model.to("cpu")

        target = config.training.target_instrument
        result = waveforms[target]

        if result.shape[0] == 1 and mix.shape[0] == 2:
            result = np.concatenate([result, result], axis=0)
        elif result.shape[0] == 1:
            pass
        elif result.shape[0] > 2:
            result = result[:2]

        peak = float(np.abs(result).max())
        if peak > 1.0:
            result = result / peak

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, result.T, sr)

        elapsed = time.time() - t0
        print(f"Done in {elapsed:.1f}s")
        print(f"Output: {output_path}")
        return True

    def cleanup(self):
        if self.vocals_model is not None:
            del self.vocals_model
        if self.inst_model is not None:
            del self.inst_model
        self.vocals_model = None
        self.inst_model = None
        self.vocals_config = None
        self.inst_config = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
