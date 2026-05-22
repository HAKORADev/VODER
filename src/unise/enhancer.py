import os
import sys
import math
import torch
import numpy as np
import soundfile as sf
import tempfile
import subprocess
from pathlib import Path


class UniSEEnhancer:
    def __init__(self, model_cache_dir):
        """
        model_cache_dir: where to store/find downloaded model weights
        """
        self.model_cache_dir = model_cache_dir
        self.codec_dir = os.path.join(model_cache_dir, "codec")
        self.ckpt_path = os.path.join(model_cache_dir, "unise.ckpt")
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def ensure_model(self):
        """Download model if not cached, then load it."""
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs(self.codec_dir, exist_ok=True)

        self._download_biocodec()
        self._download_unise_ckpt()
        self._load_model()

    def _download_biocodec(self):
        bicodec_model_path = os.path.join(self.codec_dir, "BiCodec", "model.safetensors")
        if os.path.exists(bicodec_model_path):
            return

        try:
            from huggingface_hub import hf_hub_download

            print("Downloading BiCodec weights from HuggingFace...")

            required_files = [
                "BiCodec/model.safetensors",
                "BiCodec/config.yaml",
                "config.yaml",
                "wav2vec2-large-xlsr-53/pytorch_model.bin",
                "wav2vec2-large-xlsr-53/config.json",
                "wav2vec2-large-xlsr-53/preprocessor_config.json",
            ]

            for repo_file in required_files:
                local_path = os.path.join(self.codec_dir, repo_file)
                if not os.path.exists(local_path):
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    hf_hub_download(
                        repo_id="SparkAudio/Spark-TTS-0.5B",
                        filename=repo_file,
                        local_dir=self.codec_dir,
                    )

            print("BiCodec weights downloaded.")
        except Exception as e:
            print(f"Error downloading BiCodec: {e}")

    def _download_unise_ckpt(self):
        """Download UniSE checkpoint from HuggingFace if not present."""
        if os.path.exists(self.ckpt_path):
            return

        try:
            from huggingface_hub import hf_hub_download
            import shutil

            print("Downloading UniSE model weights from HuggingFace...")
            from huggingface_hub import list_repo_files

            files = list_repo_files("QuarkAudio/QuarkAudio-UniSE")
            ckpt_files = [f for f in files if f.endswith('.ckpt')]

            if ckpt_files:
                downloaded = hf_hub_download(
                    repo_id="QuarkAudio/QuarkAudio-UniSE",
                    filename=ckpt_files[0],
                    local_dir=self.model_cache_dir,
                )
                target = os.path.join(self.model_cache_dir, "unise.ckpt")
                if downloaded != target:
                    shutil.move(downloaded, target)

            print("UniSE model weights downloaded.")
        except Exception as e:
            print(f"Error downloading UniSE checkpoint: {e}")

    def _load_model(self):
        """Load the UniSE model."""
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            from unise.model import Model

            config = {
                'codec_ckpt_dir': self.codec_dir,
                'ckpt_path': self.ckpt_path,
                'stft_config': {
                    'hop_length': 320,
                    'win_length': 640,
                    'n_fft': 640,
                    'n_mels': 80,
                },
                'llm_config': {
                    'num_tasks': 3,
                    'task_map': {'se': 0, 'tse': 1, 'rtse': 2},
                    'feats_dim': 768,
                    'llm_base_config': {
                        'cond_dim': 80,
                        'global_size': 4096,
                        'semantic_size': 8192,
                        'hidden_size': 512,
                        'num_layers': 12,
                        'num_attention_heads': 8,
                        'dropout_p': 0.1,
                        'max_position_embeddings': 4096,
                        'label_smoothing': 0.1,
                        'conformer_params': {
                            'num_layers': 6,
                            'dim': 512,
                            'heads': 8,
                            'dim_head': 64,
                            'depthwise_conv_kernel_size': 31,
                            'ff_mult': 4,
                            'dropout': 0.1,
                            'qk_norm': None,
                            'pe_attn_head': 1,
                        },
                    },
                },
            }

            print("Loading UniSE model...")
            self.model = Model(config=config)

            ckpt = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt['state_dict']

            remapped = {}
            for k, v in state_dict.items():
                if k.startswith('dnn.layers.') or k.startswith('dnn.norm.') or k.startswith('dnn.rotary_emb.'):
                    remapped['dnn.llm.' + k[4:]] = v
                else:
                    remapped[k] = v
            self.model.load_state_dict(remapped, strict=False)
            self.model = self.model.to(self.device)
            self.model.eval()
            print("UniSE model loaded successfully.")
        except Exception as e:
            print(f"Error loading UniSE model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    @torch.inference_mode()
    def enhance(self, audio_path, output_path):
        """
        Enhance a single audio file.
        Input: audio_path (any format supported by soundfile/torchaudio)
        Output: output_path (16kHz WAV)
        """
        if self.model is None:
            return False

        import torchaudio

        # Load and resample to 16kHz mono
        wav, sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        wav = wav.squeeze(0)  # (T,)

        src = wav.unsqueeze(0).to(self.device)  # (1, T)
        length = src.size(-1)
        seg_len = 5 * 16000

        # Pad and segment with circular wrap
        pad_len = math.ceil(length / seg_len) * seg_len - length
        if pad_len > 0:
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], mode='wrap')
        else:
            seg_src = src.cpu().numpy()
        seg_src = torch.from_numpy(seg_src.copy()).to(self.device)
        seg_src = seg_src.reshape(-1, seg_len)

        # Normalize per segment
        max_val = src.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-8)
        seg_src = seg_src / max_val

        # Extract features
        mix_mel = self.model.stft_logmel(seg_src)
        mix_feats = self.model.extract_semantic_features(seg_src)

        # Generate
        global_ids, semantic_ids = self.model.dnn.generate(
            task_name='se',
            enroll_mel=None,
            enroll_feats=None,
            mix_mel=mix_mel,
            mix_feats=mix_feats,
            do_sample=False,
        )

        # Detokenize
        est = self.model.tokenizer.detokenize(
            global_ids.unsqueeze(1), semantic_ids
        ).squeeze(1)
        est = est.reshape(-1)[:length].cpu().numpy()

        # Save
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, est, 16000)
        return True

    def enhance_video(self, video_path, output_path):
        """
        Enhance audio in a video file.
        Extract audio -> enhance -> replace audio in video -> save.
        """
        # Extract audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio.close()
        enhanced_audio_path = None

        try:
            cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', temp_audio.name]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(temp_audio.name) or os.path.getsize(temp_audio.name) == 0:
                print("Error: Could not extract audio from video")
                return False

            # Enhance audio
            enhanced_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            enhanced_audio.close()
            enhanced_audio_path = enhanced_audio.name

            success = self.enhance(temp_audio.name, enhanced_audio_path)
            if not success:
                return False

            # Merge enhanced audio back into video
            cmd = [
                'ffmpeg', '-i', video_path, '-i', enhanced_audio_path,
                '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                '-map', '0:v:0', '-map', '1:a:0',
                '-y', output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if os.path.exists(output_path):
                return True
            else:
                print(f"Error merging audio: {result.stderr}")
                return False
        finally:
            for f in [temp_audio.name, enhanced_audio_path]:
                if f and os.path.exists(f):
                    os.remove(f)

    @torch.inference_mode()
    def tse_extract(self, mixture_path, enroll_path, output_path):
        if self.model is None:
            return False

        import torchaudio

        mix_wav, mix_sr = torchaudio.load(mixture_path)
        if mix_wav.shape[0] > 1:
            mix_wav = mix_wav.mean(dim=0, keepdim=True)
        if mix_sr != 16000:
            mix_wav = torchaudio.transforms.Resample(mix_sr, 16000)(mix_wav)
        mix_wav = mix_wav.squeeze(0)
        src = mix_wav.unsqueeze(0).to(self.device)
        length = src.size(-1)

        seg_len = 5 * 16000
        pad_len = math.ceil(length / seg_len) * seg_len - length
        if pad_len > 0:
            seg_src = np.pad(src.cpu().numpy(), [(0, 0), (0, pad_len)], mode='wrap')
        else:
            seg_src = src.cpu().numpy()
        seg_src = torch.from_numpy(seg_src.copy()).to(self.device)
        seg_src = seg_src.reshape(-1, seg_len)

        enroll_wav, enroll_sr = torchaudio.load(enroll_path)
        if enroll_wav.shape[0] > 1:
            enroll_wav = enroll_wav.mean(dim=0, keepdim=True)
        if enroll_sr != 16000:
            enroll_wav = torchaudio.transforms.Resample(enroll_sr, 16000)(enroll_wav)
        enroll_wav = enroll_wav.squeeze(0).unsqueeze(0).to(self.device)

        enroll_mel = self.model.stft_logmel(enroll_wav)
        enroll_feats = self.model.extract_semantic_features(enroll_wav)
        enroll_mel = torch.cat([enroll_mel for _ in range(seg_src.size(0))], dim=0)
        enroll_feats = torch.cat([enroll_feats for _ in range(seg_src.size(0))], dim=0)

        mix_mel = self.model.stft_logmel(seg_src)
        mix_feats = self.model.extract_semantic_features(seg_src)

        global_ids, semantic_ids = self.model.dnn.generate(
            task_name='tse',
            enroll_mel=enroll_mel,
            enroll_feats=enroll_feats,
            mix_mel=mix_mel,
            mix_feats=mix_feats,
            do_sample=False,
        )

        est = self.model.tokenizer.detokenize(
            global_ids.unsqueeze(1), semantic_ids
        ).squeeze(1)
        est = est.reshape(-1)[:length].cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        sf.write(output_path, est, 16000)
        return True

    def cleanup(self):
        self.model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
