# VODER - Voice Blender

<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="128" height="128"/>
</p>

**VODER** is a professional-grade voice processing and transformation tool that enables seamless conversion between speech, text, and music. Built for creators, developers, and audio professionals, VODER delivers **high-quality synthesis, voice cloning, and music generation** capabilities through an intuitive interface.

### NEW!:
-**Regardless** of all these drama, the tool actually works very perfectly now, it is STABLE NOW!

🤖 **For AI agents and automated tools:** See [Bots.md](Bots.md)

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

There is no "recommended" configuration in the traditional sense. This is not a video game where higher frame rates provide a better experience. The goal is simply to avoid running out of memory (OOM) during processing. Any system meeting the minimum requirements will work — the focus is on functionality, not performance benchmarks.

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

#Notes:
- TTS, TTS+VC Supports dialouge mode
- Dialogues: a feature to write scripts to make full podcasts or AI-News
- in TTS+VC you can clone real Human voices
- Dialogues are GUI-exclusive feature!

### **AI Model Integration**

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Conversion:** [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — Seed-VC for speech-to-speech transformation
- **Music Generation:** [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) — ACE-Step for lyrics-to-music synthesis

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

---

## Installation Requirements

```bash
# Install FFmpeg (required for audio processing)
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

---

## Usage Guide

### GUI Mode

1. Launch: `python src/voder.py`
2. Select mode from dropdown (6 available modes)
3. Load input files based on mode:
   - **STT+TTS:** Load base audio (content), then load target audio (voice)
   - **TTS:** Enter script text and voice prompt
   - **TTS+VC:** Enter script text and load voice reference audio
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
python src/voder.py ttm+vc lyrics "..." styling "pop" 30 target "voice.wav"
```
#Note:
- STT+TTS not in one-line cli because it requires interaction

---

## Documentation

- **[Guide.md](Guide.md)** — Detailed usage guide, technical implementation, and creative techniques
- **[CHANGELOG.md](CHANGELOG.md)** — Development history and changes
- **[Bots.md](Bots.md)** — Guidelines for AI agents and automated systems

---

## Notes

### Testing Limitations

This project was not and may never be tested as thoroughly as it should be because I do not have the required computing power to properly validate all features. I am doing the best I can with the resources available to me. If you can or wish to help make this project better and bigger, please reach out on X: [@HAKORAdev](https://x.com/HAKORAdev)

### Project Vision

The goal of VODER is to be a local, free, open-source alternative to commercial voice synthesis platforms like ElevenLabs. Real-life limitations (hardware, resources, time) have slowed development, but I kept working on it anyway because waiting for someone else to build it would have meant it might never exist at all.

### Quality Expectations

This project may not and likely will not reach the same quality level as my other projects. The reason is simple: I was never able to run, test, or properly validate VODER due to hardware limitations. I am an independent developer with no funding — no corporate sponsors, no sugar daddy, just passion and limited resources.

### Why VODER Exists

I created VODER because no existing tool (including ComfyUI or other workflow-based solutions) provides all six audio processing capabilities in a single, unified interface. I took the first step by building this tool. Perhaps someone else will finish what I started, or perhaps the community will help evolve it further. Either way, having this tool exist is better than having nothing at all.

### Model Configuration

VODER uses hardcoded default models because they represent the best available quality. Smaller, quantized, or "fast" models produce significantly worse results — to the point where not using the tool would be preferable to using degraded models. That said, if you have the technical capability and wish to modify the code to use different models or configurations, the source code is available for you to do so.

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

**Resources:** [GitHub Issues](https://github.com/HAKORADev/VODER/issues)
