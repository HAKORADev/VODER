# VODER - Voice Blender

<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="128" height="128"/>
</p>

**VODER** is a Local, Free, Offline, professional-grade voice processing and transformation tool that enables seamless conversion between speech, text, and music. Built for creators, developers, and audio professionals, VODER delivers **high-quality synthesis, voice cloning, transcription, and music generation** capabilities through an intuitive interface.

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

> **New Dependencies (v04/02/2026 update):** VODER now requires `yt-dlp` (for YouTube/Bilibili/TikTok URL support), `easyocr` and `onnxruntime` (for image text extraction), `lightning` (for pyannote model loading), and `sox` (for audio manipulation). These are included in `requirements.txt` — simply run `pip install -r requirements.txt` after pulling the latest version.

---

## Core Capabilities

### 🎤 **7 Processing Modes**

VODER offers seven distinct voice processing modes, each designed for specific audio transformation needs:

| Mode | Description | Input | Output |
|------|-------------|-------|--------|
| **STT+TTS** | Speech-to-Text then Text-to-Speech | Audio | Audio |
| **TTS** | Text-to-Speech with Voice Design | Text | Audio |
| **TTS+VC** | Text-to-Speech + Voice Cloning | Text + Reference | Audio |
| **STS** | Speech-to-Speech (Voice Conversion) | Audio + Reference | Audio |
| **TTM** | Text-to-Music Generation | Text | Audio |
| **TTM+VC** | Text-to-Music + Voice Conversion | Text + Reference | Audio |
| **STT** | Speech-to-Text (Transcription) | Audio / Video / Image / URL | Text |

**MSTS (Music-STS):** STS mode now supports musical inputs. When processing songs or musical audio, select "musical inputs?" to use the Seed-VC v1 model (44.1kHz) instead of the standard v2 model (22.05kHz), providing better voice conversion quality for music content.

### STT Mode (Speech-to-Text)

STT is a **standalone transcription mode** available as a one-line CLI command. It transcribes audio, video, images, or YouTube URLs into plain text with optional enhancements.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.)
- Image files containing text (PNG, JPG, etc.) — text is extracted via OCR before transcription
- YouTube / Bilibili / TikTok URLs — downloaded and processed automatically

**Features:**
- Clean text transcription output
- Optional **timestamps** for word-level or segment-level timing
- Optional **dialogue mode** that detects and formats multi-speaker conversations
- Optional **speaker diarization** that identifies individual speakers by name/label
- **Batch processing** — pass multiple files/URLs in a single command to process them all at once
- Results saved to a specified output file or printed to the terminal

**Quick Examples:**
```bash
# Basic transcription
python src/voder.py stt "audio.wav"

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With dialogue formatting
python src/voder.py stt "audio.wav" dialogue

# Batch processing
python src/voder.py stt "audio1.wav" "audio2.wav"

# Transcribe a YouTube video directly
python src/voder.py stt "https://youtube.com/watch?v=..."

# Save TTS output to a specific file
python src/voder.py tts script "Hello" voice "male" result "/path/to/output.txt"
```

---

### 🎭 **Dialogue System**

VODER features a powerful **row-based dialogue editor** designed for creating multi-speaker audio content such as podcasts, AI news broadcasts, audiobooks, and conversational content. This system enables script-based generation where multiple characters speak with distinct voices in a cohesive narrative flow.

**GUI Dialogue Input:**
- Each line is a separate row with **Character** and **Dialogue** fields.
- New rows are added automatically when you fill the last row.
- First row has no delete button; subsequent rows can be deleted individually.
- Voice prompts or audio assignments appear dynamically for every character found in the script.

**Optional Background Music:**
- When generating dialogue (TTS or TTS+VC mode), VODER can automatically **add ambient background music** that matches the length of the spoken audio.
- A clean dialog appears before processing, asking: *"Enter music description (or press Skip):"*
- If a description is provided (e.g., `"soft piano, cinematic strings"`), VODER:
  - Generates music via ACE-Step using the description as style prompt and `"..."` as empty lyrics.
  - Automatically fits the music duration to the exact length of the dialogue.
  - Mixes the music at **35% volume** relative to the dialogue.
  - Cleans up temporary files and saves the final result with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`).
- If the user skips, processing proceeds normally without music.

This feature is available in both **GUI** and **CLI** modes (interactive and one‑line). It is **only triggered for dialogue scripts** (i.e., more than one line, or a single line containing a colon).

**Example Script (conceptual):**
```plaintext
James: Welcome to our podcast! Today we'll explore AI advances.
Sarah: Thanks James! I'm excited to discuss my latest research.
James: Let's dive in. First, tell us about neural networks.
```

**Key Features:**
- Multi-character script support with real-time character extraction
- Individual voice prompts for each character (TTS mode)
- Reference audio assignment per character via dropdown numbers (TTS+VC mode)
- **Optional background music** – automatically generated, duration‑fitted, volume‑controlled
- Automatic audio concatenation with proper pacing
- Ideal for podcasts, news segments, interviews, and storytelling

The dialogue system is available in both **TTS** (Voice Design) and **TTS+VC** (Voice Cloning) modes, allowing you to create voices either through descriptive prompts or by cloning from real audio samples.

---

### 🧠 **Intelligent Source Analysis**

VODER now supports **cross-platform source input** — a unified input pipeline that accepts audio, video, images, and URLs across multiple processing modes. This enables powerful new workflows:

- **YouTube / Bilibili / TikTok URL Support:** Paste a video URL directly as input in STT, STT+TTS, and dialogue modes. VODER automatically downloads the audio track and processes it — no manual downloading or conversion required.
- **Image Text Extraction (OCR):** Feed image files (PNG, JPG, etc.) as input. VODER uses EasyOCR to extract embedded text, which is then processed as dialogue script content. This works in STT, TTS, and TTS+VC modes — enabling workflows like "photo of a script → spoken audio."
- **Automatic Voice Clip Extraction:** When processing multi-speaker audio (e.g., a podcast recording), VODER can automatically identify and extract individual speaker segments. This replaces the previous manual approach of splitting audio files.
- **Speaker Diarization:** Powered by pyannote, VODER identifies who spoke when in multi-speaker audio. Each speaker is labeled consistently, and the diarization output can be combined with transcription for fully annotated results.

> **Multi-Speaker Input — Now Supported!** Previous versions of VODER required manually separating multi-speaker audio before processing. With the new Intelligent Source Analysis system, VODER can now accept multi-speaker audio directly. The speaker diarization pipeline automatically identifies speakers, extracts their voice clips, and makes them available for voice cloning and transcription. See [Guide.md](Guide.md) for the updated workflow.

---

### 🔧 **AI Model Integration**

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Conversion:** [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — Seed-VC for speech-to-speech transformation
- **Music Generation:** [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) — ACE-Step for lyrics-to-music synthesis
- **Speaker Diarization:** [pyannote/speaker-diarization-community-1](https://github.com/pyannote/pyannote-audio) — pyannote for identifying and labeling individual speakers in multi-speaker audio
- **Image Text Extraction:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) — EasyOCR for extracting text from images, enabling image-to-speech workflows

---

## Usage Guide

### GUI Mode

1. Launch: `python src/voder.py`
2. Select mode from dropdown (7 available modes)
3. Load input files based on mode:
   - **STT+TTS:** Load base audio (content), then load target audio (voice)
   - **STT:** Load audio, video, image, or enter a URL for transcription
   - **TTS:** Enter dialogue row‑by‑row in the script area, and fill the automatically generated voice prompts for each character  
     **Optional:** Before generation, a dialog will ask if you want background music; enter a description or press Skip.
   - **TTS+VC:** Enter dialogue rows, load voice reference audio files (each assigned a number), then assign each character an audio number via dropdown  
     **Optional:** The same background music dialog appears before generation.
   - **STS:** Load base audio and target voice audio
   - **TTM:** Enter lyrics and style prompt
   - **TTM+VC:** Enter lyrics, style prompt, and load target voice audio
4. Click **"Generate"** (TTS/TTS+VC/TTM/TTM+VC) or **"Patch"** (STT+TTS/STS) or **"Transcribe"** (STT)
5. Listen to output and save results

### CLI Mode (Interactive)
```bash
python src/voder.py cli
```
The interactive CLI now supports full dialogue creation:
- Enter multiple lines (empty line to finish).
- Lines without a colon → **single mode** (one text, one voice prompt/audio).
- Lines with colon (`Character: text`) → **dialogue mode**.
- VODER will ask for a voice prompt (TTS) or audio file path (TTS+VC) for each character, in order.
- **After** collecting all voice prompts/assignments, you will be asked:  
  `Add background music? (y/N):`  
  If you answer `y` or `yes`, you can enter a music description.  
  Leaving the description blank or entering empty skips the music.

### One-Line Commands
One‑line commands now support **dialogue mode** through multiple values per parameter, as well as the optional **`music`** parameter for background music.

**Single mode examples:**
```bash
python src/voder.py tts script "Hello world" voice "female, cheerful"
python src/voder.py tts+vc script "Hello" target "voice.wav"
python src/voder.py sts base "input.wav" target "voice.wav"
python src/voder.py ttm lyrics "Verse 1: ..." styling "upbeat pop" 30
python src/voder.py ttm+vc lyrics "..." styling "pop" duration 30 target "voice.wav"
```

**STT mode examples:**
```bash
# Basic transcription
python src/voder.py stt "audio.wav"

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With dialogue formatting
python src/voder.py stt "audio.wav" dialogue

# Batch processing — multiple files in one command
python src/voder.py stt "audio1.wav" "audio2.wav"

# Transcribe a YouTube video directly
python src/voder.py stt "https://youtube.com/watch?v=..."

# Save TTS output to a specific file
python src/voder.py tts script "Hello" voice "male" result "/path/to/output.txt"
```

**Dialogue mode examples (TTS):**
```bash
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic"
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic" music "soft piano, cinematic"
```

**Dialogue mode examples (TTS+VC):**
```bash
python src/voder.py tts+vc script "James: Let's start with AI." "Sarah: I've been working on this for years." target "James: /path/to/james_voice.wav" "Sarah: /path/to/sarah_voice.wav"
python src/voder.py tts+vc script "James: Let's start with AI." "Sarah: I've been working on this for years." target "James: /path/to/james_voice.wav" "Sarah: /path/to/sarah_voice.wav" music "ambient electronic, chill"
```

**Note:** STT+TTS mode is not available in one-line CLI because it requires interactive text editing.
If the `music` parameter is supplied in single‑mode (plain text without colon), it is ignored with a warning.

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

### Speaker Diarization Requirements

Speaker diarization (STT with diarization or multi-speaker analysis) adds additional memory requirements:

- **RAM:** Expect approximately **2–3GB more** system memory when using speaker diarization, as the pyannote model loads alongside the transcription pipeline
- **HF_TOKEN:** The pyannote speaker-diarization-community-1 model requires a **Hugging Face access token** with accepted terms of use. Set the `HF_TOKEN` environment variable before running:
  ```bash
  export HF_TOKEN="hf_your_token_here"
  ```
  You can obtain a token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) after accepting the model's license on its Hugging Face page.

### Recommended Requirements

VODER is designed to maximize output quality rather than speed. Meeting the minimum requirements ensures reliable operation — the focus is on achieving professional-grade audio results, not processing benchmarks. More RAM allows for longer audio generation and more complex workflows. For the best experience with all features (including speaker diarization), **16GB+ RAM** is recommended.

---

## Technical Highlights

- **Unified Audio Pipeline:** Seven processing modes in a single interface eliminates the need for multiple tools
- **Centralized Model Management:** A unified model management system handles loading, caching, and offloading of all AI models — ensuring efficient resource usage and preventing memory accumulation across multi-step workflows
- **Intelligent Dialogue Editor:** Row‑based script input with automatic character tracking and per‑voice assignment
- **State-of-the-Art Models:** Production-quality models from leading AI research organizations
- **Voice Cloning:** Extract and replicate voice characteristics from reference audio samples
- **Music Generation:** Lyrics-to-music synthesis with style control and voice conversion
- **Cross-Modal Transformation:** Speech-to-speech, text-to-speech, speech-to-text, and text-to-text conversions
- **Cross-Platform Source Input:** Unified input pipeline accepts audio files, video files, images, and URLs (YouTube, Bilibili, TikTok) — no manual format conversion required
- **Automatic Speaker Identification:** Multi-speaker audio is automatically segmented and labeled using pyannote speaker diarization, with individual voice clips extracted for downstream processing
- **Speaker Diarization with Word-Level Alignment:** Combines Whisper transcription with pyannote diarization to produce speaker-labeled, timestamped transcripts with per-word speaker attribution
- **MSTS (Music-STS):** STS mode supports musical inputs using Seed-VC v1 at 44.1kHz for better music voice conversion
- **Memory Optimisation:** Models are now explicitly offloaded after each operation to prevent memory accumulation in session-based workflows
- **Background Music for Dialogue:** Automatically generated, duration‑fitted, volume‑controlled ambient music – a unique enhancement for narrated content

---

## Documentation

- **[Guide.md](Guide.md)** — Detailed usage guide, technical implementation, and creative techniques
- **[CHANGELOG.md](CHANGELOG.md)** — Development history and version changes
- **[Bots.md](Bots.md)** — Guidelines for AI agents and automated systems

---

## Version Information

**Note:** VODER does not maintain PyPI packages or pre-built binaries. Running from source ensures access to the most recent features and improvements.

---

## Contributing

VODER is open-source (MIT License) and welcomes contributions:

- New voice processing modes
- Additional model integrations
- UI/UX improvements
- Performance optimizations
- Documentation and translations
- Bug reports and feature requests

Please submit pull requests or issues via GitHub.

---

## License

MIT License — See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

Built with appreciation for the open-source AI voice synthesis community and the amazing models that power VODER.
