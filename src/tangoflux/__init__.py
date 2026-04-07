import os
import json
import torch
import torchaudio
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from diffusers import AutoencoderOobleck
from .model import TangoFlux


class TangoFluxGenerator:
    """
    Text-to-audio generation using TangoFlux model.
    Generates sound effects from text prompts with configurable duration.
    """

    HF_REPO = "declare-lab/TangoFlux"
    SAMPLE_RATE = 44100
    DEFAULT_STEPS = 30
    DEFAULT_GUIDANCE = 4.5

    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.vae = None

    def ensure_model(self):
        if self.model is not None:
            return
        os.makedirs(self.model_dir, exist_ok=True)

        print("Loading TangoFlux model (this may take a while on first run)...")
        try:
            self.vae = AutoencoderOobleck()

            paths = snapshot_download(repo_id=self.HF_REPO, cache_dir=self.model_dir)
            vae_weights = load_file("{}/vae.safetensors".format(paths))
            self.vae.load_state_dict(vae_weights)

            weights = load_file("{}/tangoflux.safetensors".format(paths))

            with open("{}/config.json".format(paths), "r") as f:
                config = json.load(f)

            self.model = TangoFlux(config)
            self.model.load_state_dict(weights, strict=False)

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.vae.to(device)
            self.model.to(device)
            print("TangoFlux model loaded successfully.")
        except Exception as e:
            print(f"Error loading TangoFlux model: {e}")
            self.model = None
            self.vae = None

    def generate(self, prompt, duration, steps=None, guidance_scale=None):
        if self.model is None:
            return None
        if steps is None:
            steps = self.DEFAULT_STEPS
        if guidance_scale is None:
            guidance_scale = self.DEFAULT_GUIDANCE

        duration = max(1, min(30, int(duration)))

        with torch.no_grad():
            latents = self.model.inference_flow(
                prompt,
                duration=duration,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
            )
            wave = self.vae.decode(latents.transpose(2, 1)).sample.cpu()[0]

        waveform_end = int(duration * self.SAMPLE_RATE)
        wave = wave[:, :waveform_end]
        return wave

    def save(self, audio, output_path):
        if audio is None:
            return False
        try:
            torchaudio.save(output_path, audio, sample_rate=self.SAMPLE_RATE)
            return True
        except Exception as e:
            print(f"Error saving audio: {e}")
            return False

    def cleanup(self):
        if self.model is not None:
            del self.model
        if self.vae is not None:
            del self.vae
        self.model = None
        self.vae = None
