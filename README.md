# VODER - Voice Blender

<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="256" height="256"/>
</p>

**VODER** is a Local, Free, Offline, professional-grade voice processing and transformation tool that enables seamless conversion between speech, text, and music. Built for creators, developers, and audio professionals, VODER delivers **high-quality synthesis, voice cloning, transcription, music generation, sound effects, and speech enhancement** capabilities through an intuitive interface.

🚀 **Ready in Colab:** [Open VODER in Google Colab](https://colab.research.google.com/drive/1hditIfW9JzusNcFhlHFoclCIIsNiRFNk?usp=sharing)

🤖 **For AI agents and automated tools:** See [Bots.md](Bots.md)

---

## Quick Start

### Run from Source
```bash
# Clone the repository
git clone https://github.com/HAKORADev/VODER.git
cd VODER

# Install dependencies
pip install -r requirements.txt

# IMPORTANT: After installing requirements, upgrade protobuf to avoid compatibility issues
pip install --upgrade protobuf==5.29.6

# Launch GUI
python src/voder.py

# Or use CLI mode
python src/voder.py cli
```

### Run in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hditIfW9JzusNcFhlHFoclCIIsNiRFNk?usp=sharing)

Open the link, connect to a runtime, and press **Run All** (or run cells one by one until the last one). Once execution completes, VODER is ready to use directly in your browser — no installation required.

### Installation Requirements
```bash
# Install FFmpeg (required for audio processing)
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Additional system dependencies (required for new features)
# Linux: sudo apt install sox
# macOS: brew install sox
# yt-dlp: pip install yt-dlp
```

> **New Dependencies (v04/08/2026 update):** VODER now requires `yt-dlp` (for YouTube/Bilibili/TikTok URL support), `easyocr` and `onnxruntime` (for image text extraction), `lightning` (for pyannote model loading), `sox` (for audio manipulation), `einx`, `x-transformers`, `safetensors`, `soxr` (for UniSE speech enhancement), `tqdm`/`packaging`, `rotary_embedding_torch`, `beartype`, and `ml_collections` (for BS-RoFormer vocal/music separation), and `huggingface-hub==0.34.0` (pinned for model download compatibility). These are included in `requirements.txt` — simply run `pip install -r requirements.txt` after pulling the latest version.

> **New Model Directories:** VODER now downloads additional models for BS-RoFormer (vocal/music separation) and VibeVoice ASR (advanced transcription). Ensure sufficient disk space is available — model files are cached in the standard Hugging Face cache directory.

---

## Core Capabilities

### 🎤 **10 Processing Modes**

VODER offers ten distinct voice processing modes, each designed for specific audio transformation needs:

| Mode | Description | Input | Output |
|------|-------------|-------|--------|
| **STT+TTS** | Speech-to-Text then Text-to-Speech | Audio | Audio |
| **TTS** | Text-to-Speech with Voice Design & Cloning | Text + Optional Reference | Audio |
| **STS** | Speech-to-Speech (Voice Conversion) | Audio/Video + Reference | Audio/Video |
| **TTM** | Text-to-Music Generation & Manipulation | Text + Audio | Audio |
| **STT** | Speech-to-Text (Transcription & Translation) | Audio / Video / Image / URL | Text |
| **SE** | Speech Enhancement (Denoise/Dereverb) | Audio / Video | Audio / Video |
| **SFX** | Sound Effects Generation | Text | Audio |
| **SVS** | Song Voice Separate (Vocal/Music Isolation) | Audio / Video / URL | Audio |
| **SLC** | Speaker Language Conversion | Audio / URL | Audio |
| **SS** | Speakers Separator | Audio / Video | Audio per speaker |

> **Note:** `tts+vc` and `ttm+vc` are no longer available as standalone modes. Voice cloning in TTS is handled via the `target` parameter, and voice conversion in TTM is handled via the `vc` flag. Use `tts` and `ttm` respectively.

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 4-6 cores |
| RAM | 12GB+ system memory |
| GPU (CUDA) | Optional (CPU-only operation supported) |
| VRAM | 4GB minimum (6GB recommended, 16GB for best performance) |
| Storage | SSD recommended |

**Note:** VODER runs entirely on CPU. No GPU is required for any mode. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

### SVS Mode Requirements

SVS mode requires the BS-RoFormer Resurrection model, which is downloaded automatically on first use. The model adds approximately **1.5GB** to disk storage in the Hugging Face model cache.

### SS / VibeVoice ASR Requirements

The SS mode (Speakers Separator) and STT overdose mode use Microsoft VibeVoice ASR, which has significant memory requirements:

- **VRAM:** 24GB+ GPU VRAM recommended, or
- **RAM:** 48GB+ system memory for CPU/offload operation
- If VibeVoice ASR cannot load due to insufficient resources, SS falls back to Whisper + pyannote speaker diarization

### ACE-Step Overdose / Complete Requirements

TTM mode with overdose or complete quality tiers uses larger ACE-Step models:

- **VRAM:** 32GB+ GPU VRAM recommended, or
- **RAM:** 48GB+ system memory for CPU/offload operation

### Speaker Diarization Requirements

Speaker diarization (STT with diarization or multi-speaker analysis) adds additional memory requirements:

- **RAM:** Expect approximately **2–3GB more** system memory when using speaker diarization, as the pyannote model loads alongside the transcription pipeline
- **HF_TOKEN:** The pyannote speaker-diarization-community-1 model requires a **Hugging Face access token** with accepted terms of use. Set the `HF_TOKEN` environment variable before running:
  ```bash
  export HF_TOKEN="hf_your_token_here"
  ```
  You can obtain a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) after accepting the model's license on its Hugging Face page.

### Recommended Requirements

VODER is designed to maximize output quality rather than speed. Meeting the minimum requirements ensures reliable operation — the focus is on achieving professional-grade audio results, not processing benchmarks. More RAM allows for longer audio generation and more complex workflows. For the best experience with all features (including speaker diarization, speech enhancement, VibeVoice ASR, and BS-RoFormer separation), **32GB+ RAM** is recommended.

---

## Documentation

- **[READ.md](READ.md)** — Detailed mode descriptions, CLI examples, notes, and technical deep-dives
- **[Guide.md](Guide.md)** — Comprehensive usage guide, technical implementation, and creative techniques
- **[COMMAND_CATALOG.md](COMMAND_CATALOG.md)** — Complete oneline command reference — every mode, flag, keyword, and syntax with examples and a Quick Jump table
- **[CHANGELOG.md](CHANGELOG.md)** — Development history and version changes
- **[Bots.md](Bots.md)** — Guidelines for AI agents and automated systems
- **[voder-skill.md](voder-skill.md)** — Direct Agent skill
- **[Languages.md](Languages.md)** — Supported languages across all components, auto‑detection capabilities, and language configuration

---

## Version Information

**Note:** VODER does not maintain PyPI packages or pre-built binaries. Running from source ensures access to the most recent features and improvements.

---

## Contributing

VODER is open-source (AGPL-3.0 License) and welcomes contributions:

- New voice processing modes
- Additional model integrations
- UI/UX improvements
- Performance optimizations
- Documentation and translations
- Bug reports and feature requests

Please submit pull requests or issues via GitHub.

---

## License

AGPL v3.0 License — See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

Built with appreciation for the open-source AI voice synthesis community and the amazing models that power VODER.
