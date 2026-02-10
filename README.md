# VODER - Voice Blender

<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="128" height="128"/>
</p>

**VODER** is a professional-grade voice processing and transformation tool that enables seamless conversion between speech, text, and music. Built for creators, developers, and audio professionals, VODER delivers **high-quality synthesis, voice cloning, and music generation** capabilities through an intuitive interface.

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

# Launch GUI
python src/voder.py

# Or use CLI mode
python src/voder.py cli
```

### Installation Requirements
```bash
# Install FFmpeg (required for audio processing)
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Core Capabilities

### 🎤 **6 Processing Modes**

VODER offers six distinct voice processing modes, each designed for specific audio transformation needs:

| Mode | Description | Input | Output |
|------|-------------|-------|--------|
| **STT+TTS** | Speech-to-Text then Text-to-Speech | Audio | Audio |
| **TTS** | Text-to-Speech with Voice Design | Text | Audio |
| **TTS+VC** | Text-to-Speech + Voice Cloning | Text + Reference | Audio |
| **STS** | Speech-to-Speech (Voice Conversion) | Audio + Reference | Audio |
| **TTM** | Text-to-Music Generation | Text | Audio |
| **TTM+VC** | Text-to-Music + Voice Conversion | Text + Reference | Audio |

---

### 🎭 **Dialogue System**

VODER features a powerful dialogue system designed for creating multi-speaker audio content such as podcasts, AI news broadcasts, audiobooks, and conversational content. This feature enables script-based generation where multiple characters speak with distinct voices in a cohesive narrative flow.

**Dialogue Format:**
```plaintext
1:James: "Welcome to our podcast! Today we'll explore AI advances."
2:Sarah: "Thanks James! I'm excited to discuss my latest research."
3:James: "Let's dive in. First, tell us about neural networks."
```

**Key Features:**
- Multi-character script support with sequential line numbering
- Individual voice prompts for each character
- Reference audio-based voice cloning per speaker
- Automatic audio concatenation with proper pacing
- Ideal for podcasts, news segments, interviews, and storytelling

The dialogue system is available in both TTS (Voice Design) and TTS+VC (Voice Cloning) modes, allowing you to create voices either through descriptive prompts or by cloning from real audio samples.

---

### 🔧 **AI Model Integration**

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Conversion:** [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — Seed-VC for speech-to-speech transformation
- **Music Generation:** [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) — ACE-Step for lyrics-to-music synthesis

---

## Usage Guide

### GUI Mode

1. Launch: `python src/voder.py`
2. Select mode from dropdown (6 available modes)
3. Load input files based on mode:
   - **STT+TTS:** Load base audio (content), then load target audio (voice)
   - **TTS:** Enter script text and voice prompt (supports dialogue mode)
   - **TTS+VC:** Enter script text and load voice reference audio (supports dialogue mode)
   - **STS:** Load base audio and target voice audio
   - **TTM:** Enter lyrics and style prompt
   - **TTM+VC:** Enter lyrics, style prompt, and load target voice audio
4. Click "Patch" to process
5. Listen to output and save results

### CLI Mode (Interactive)
```bash
python src/voder.py cli
```

### One-Line Commands
```bash
# Text-to-Speech
python src/voder.py tts script "Hello world" voice "female, cheerful"

# Text-to-Speech with Voice Cloning
python src/voder.py tts+vc script "Hello" target "voice.wav"

# Speech-to-Speech Voice Conversion
python src/voder.py sts base "input.wav" target "voice.wav"

# Text-to-Music
python src/voder.py ttm lyrics "Verse 1: ..." styling "upbeat pop" 30

# Text-to-Music with Voice Conversion
python src voder.py ttm+vc lyrics "..." styling "pop" 30 target "voice.wav"
```

**Note:** STT+TTS mode is not available in one-line CLI because it requires interactive text editing.

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| CPU | 4-6 cores |
| GPU | 8GB+ VRAM (NVIDIA) |
| RAM | 16GB system memory |
| Storage | SSD recommended |

### Recommended Requirements

VODER is designed to maximize output quality rather than speed. Meeting the minimum requirements ensures reliable operation — the focus is on achieving professional-grade audio results, not processing benchmarks. Higher VRAM allows for longer audio generation and more complex workflows.

---

## Technical Highlights

- **Unified Audio Pipeline:** Six processing modes in a single interface eliminates the need for multiple tools
- **State-of-the-Art Models:** Production-quality models from leading AI research organizations
- **Dialogue Scripting:** Multi-speaker content creation with character-specific voice assignments
- **Voice Cloning:** Extract and replicate voice characteristics from reference audio samples
- **Music Generation:** Lyrics-to-music synthesis with style control and voice conversion
- **Cross-Modal Transformation:** Speech-to-speech, text-to-speech, and speech-to-text conversions

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
