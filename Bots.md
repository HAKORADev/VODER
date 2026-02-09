# VODER - Bot & AI Agent Usage Guide

This document provides instructions for AI agents, bots, and automated systems on how to use VODER for voice processing tasks. AI agents typically operate in headless environments without continuous terminal access, so this guide focuses on one-liner commands and processing patterns.

## Quick Start

```bash
# Clone and install
git clone https://github.com/HAKORADev/VODER.git && cd VODER
pip install -r requirements.txt

# Run processing modes (one-liner)
python voder.py tts script "Hello world" voice "male voice"      # Text-to-Speech
python voder.py tts+vc script "Hello" target "voice.wav"         # TTS + Voice Clone
python voder.py sts base "input.wav" target "voice.wav"          # Voice Conversion (requires GPU)
python voder.py ttm lyrics "song lyrics" styling "pop" 30        # Text-to-Music
python voder.py ttm+vc lyrics "song" styling "pop" 30 target "voice.wav"  # TTM + Voice Clone (requires GPU)
```

## Installation

### Python Dependencies

```bash
pip install -r requirements.txt
```

All Python dependencies are installed from the requirements.txt file in the VODER root directory.

### FFmpeg Installation

**FFmpeg is REQUIRED for audio/video processing.** FFmpeg is not a Python package and must be installed separately on your system.

**Windows (winget):**
```powershell
winget install FFmpeg
```

**Windows (manual):**
```powershell
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract to C:\ffmpeg
# Add C:\ffmpeg\bin to system PATH
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (apt):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**Verify FFmpeg Installation:**
```bash
ffmpeg -version
```

## Processing Modes

VODER supports multiple processing modes. STT+TTS mode is only available in interactive CLI and GUI because it requires user interaction to review and edit the transcribed text before synthesis.

### Text-to-Speech (tts)

Generate speech from text with voice design using Qwen3-TTS VoiceDesign model.

```bash
python voder.py tts script "text here" voice "voice description"
```

**Parameters:**
- `script` - Text to synthesize (use \\n for newlines)
- `voice` - Voice prompt describing the desired voice

**Example:**
```bash
python voder.py tts script "Hello, this is a test." voice "clean adult male"
```

**GPU Requirement:** This mode works on CPU without GPU.

### Text-to-Speech + Voice Clone (tts+vc)

Generate speech from text then clone it to target voice using Qwen3-TTS Base model.

```bash
python voder.py tts+vc script "text here" target "voice_reference.wav"
```

**Parameters:**
- `script` - Text to synthesize
- `target` - Path to voice reference audio file

**Example:**
```bash
python voder.py tts+vc script "Hello everyone" target "speaker1.wav"
```

**Note:** This mode uses Qwen3-TTS Base for voice cloning, not Seed-VC.

**GPU Requirement:** This mode works on CPU without GPU.

### Speech-to-Speech / Voice Conversion (sts)

Convert voice from base audio to target voice without changing content using Seed-VC.

```bash
python voder.py sts base "source_audio.wav" target "voice_reference.wav"
```

**Parameters:**
- `base` - Path to source audio (supports .wav, .mp3, .mp4, .avi, .mov, .mkv)
- `target` - Path to target voice reference audio

**Example:**
```bash
python voder.py sts base "speech.wav" target "celebrity_voice.wav"
```

**GPU Requirement:** This mode requires NVIDIA GPU with minimum 8GB VRAM (Seed-VC only works on GPU).

### Text-to-Music (ttm)

Generate music from lyrics and style prompt using ACE-Step.

```bash
python voder.py ttm lyrics "song lyrics" styling "style description" duration_seconds
```

**Parameters:**
- `lyrics` - Song lyrics (use \\n for newlines)
- `styling` - Style prompt describing the music style
- `duration` - Duration in seconds (10-300)

**Example:**
```bash
python voder.py ttm lyrics "Verse 1:\nWalking down the street" styling "upbeat pop" 30
```

**GPU Requirement:** This mode works on CPU without GPU.

### Text-to-Music + Voice Clone (ttm+vc)

Generate music using ACE-Step then apply voice conversion using Seed-VC.

```bash
python voder.py ttm+vc lyrics "song lyrics" styling "style" duration target "voice.wav"
```

**Parameters:**
- `lyrics` - Song lyrics
- `styling` - Style prompt
- `duration` - Duration in seconds (10-300)
- `target` - Voice reference audio path

**Example:**
```bash
python voder.py ttm+vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" 30 target "singer.wav"
```

**GPU Requirement:** This mode requires NVIDIA GPU with minimum 8GB VRAM (Seed-VC only works on GPU).

### STT+TTS Mode (Interactive CLI Only)

This mode transcribes audio then synthesizes with voice cloning. It is only available in interactive CLI mode because it requires user interaction to review and edit the transcribed text before synthesis.

```bash
python voder.py cli
# Then select option 1 for STT+TTS
```

**Note:** STT+TTS is not available in one-liner mode due to the interactive text editing requirement.

## CLI vs GUI Feature Comparison

### CLI-Only Features

- One-liner command execution
- Batch processing with command chaining
- Headless operation
- Interactive CLI mode with text editing (STT+TTS available here)

### GUI-Only Features

- **Dialogue Mode**: Interactive conversational exchanges for TTS and TTS+VC
- Real-time preview of processing status
- Interactive parameter adjustment

### Shared Features

- Processing modes (TTS, TTS+VC, STS, TTM)
- Parameter customization
- Output file generation

## GPU Requirements

### Modes Requiring GPU

The following modes require NVIDIA GPU with minimum 8GB VRAM because they use Seed-VC:

- **STS** (Speech-to-Speech Voice Conversion)
- **TTM+VC** (Text-to-Music + Voice Conversion)

Seed-VC only works on NVIDIA GPUs and cannot run on CPU or non-NVIDIA graphics cards.

### Modes Working Without GPU

The following modes work on CPU-only systems:

- **TTS** (Text-to-Speech with Voice Design)
- **TTS+VC** (Text-to-Speech with Voice Clone using Qwen3-TTS Base)
- **TTM** (Text-to-Music)

Processing will be significantly slower without GPU acceleration.

### Recommended GPU

An NVIDIA RTX 3060 with 8GB+ VRAM is sufficient for all VODER processing modes including Seed-VC operations.

## Output Location

All results are automatically exported to the `results/` folder located next to `voder.py` in the VODER directory.

```bash
VODER/
  voder.py
  results/
    voder_tts_20260210_120000.wav
    voder_sts_20260210_120100.wav
    ...
```

## Troubleshooting

### Issue: Voice conversion fails immediately

**Cause**: No NVIDIA GPU detected or insufficient VRAM

**Solution**: Verify NVIDIA GPU is installed and has at least 8GB VRAM. Voice conversion modes (STS, TTM+VC) require GPU.

### Issue: Out of memory errors

**Cause**: Model too large for available GPU memory

**Solution**: Process shorter audio segments or reduce duration for TTM modes

### Issue: Module not found errors

**Cause**: Python dependencies not installed

**Solution**: Run `pip install -r requirements.txt`

### Issue: FFmpeg not found errors

**Cause**: FFmpeg not installed or not in system PATH

**Solution**: Install FFmpeg separately (see FFmpeg Installation section)

### Issue: STT+TTS mode not working

**Cause**: STT+TTS is not available in one-liner mode because it requires interactive text editing

**Solution**: Use interactive CLI mode with `python voder.py cli` or use GUI

### Issue: Slow processing

**Cause**: Running on CPU without GPU acceleration (for STS or TTM+VC modes)

**Solution**: Use NVIDIA GPU with 8GB+ VRAM for acceleration, or use TTS, TTS+VC, or TTM modes which work on CPU

## Summary for AI Agents

1. **Python installation**: `pip install -r requirements.txt`
2. **FFmpeg installation**: Required system package (not Python)
3. **One-liner syntax**: `python voder.py mode param "value" param "value"`
4. **GPU required for**: STS and TTM+VC modes only (Seed-VC)
5. **CPU-only modes**: TTS, TTS+VC, TTM (no GPU needed)
6. **Dialogue mode is GUI-only**: Not available via CLI
7. **STT+TTS is CLI-only**: Interactive text editing required, use `python voder.py cli`
8. **Output location**: Results saved to `results/` folder next to voder.py
9. **Available modes**: tts, tts+vc, sts, ttm, ttm+vc
10. **TTM duration**: 10-300 seconds
11. **Video support**: VODER supports video input (auto audio extraction via FFmpeg)
