# VODER - Bot & AI Agent Usage Guide

This document provides comprehensive instructions for AI agents, bots, and automated systems on how to effectively use VODER for voice processing tasks. AI agents typically operate in headless environments without continuous terminal access, so this guide focuses on one-liner commands and batch processing patterns.

## Table of Contents

1. [Purpose](#purpose)
2. [Quick Start for AI Agents](#quick-start-for-ai-agents)
3. [Installation](#installation)
4. [FFmpeg Setup](#ffmpeg-setup)
5. [One-Liner Command Patterns](#one-liner-command-patterns)
6. [Command Reference](#command-reference)
7. [CLI vs GUI Feature Comparison](#cli-vs-gui-feature-comparison)
8. [GPU Requirements](#gpu-requirements)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)
11. [Example Workflows](#example-workflows)

---

## Purpose

VODER is a professional-grade voice processing tool that enables seamless conversion between speech, text, and music. For AI agents operating in automated pipelines, VODER offers:

- **Unified Audio Pipeline**: Six processing modes in a single interface
- **CLI-First Design**: All core features accessible via command line
- **No GUI Required**: Runs entirely in headless terminals
- **Dialogue Support**: Multi-speaker script generation (GUI-exclusive)
- **Music Generation**: Lyrics-to-music synthesis with voice conversion
- **Voice Cloning**: Extract and replicate voice characteristics from reference audio

---

## Quick Start for AI Agents

AI agents typically cannot maintain interactive terminal sessions. Use the following pattern:

```bash
# Clone the repository
git clone https://github.com/HAKORADev/VODER.git && cd VODER

# Install dependencies (one-liner)
pip install -r requirements.txt

# Process files immediately (one-liner)
python src/voder.py tts script "Hello world" voice "male voice"

# Chain multiple operations
python src/voder.py tts script "Hello" voice "female" && python src/voder.py tts script "World" voice "male"
```

---

## Installation

### Python Dependencies

Install all required packages in a single command:

```bash
pip install -r requirements.txt
```

**Package explanations:**

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework for neural network models |
| `torchaudio` | Audio loading and processing |
| `transformers` | HuggingFace model integration |
| `PyQt5` | GUI framework (required for full functionality) |
| `omegaconf` | Configuration management |
| `hydra-core` | Configuration framework |
| `huggingface_hub` | Model download and caching |
| `soundfile` | Audio file I/O operations |

### Verify Installation

```bash
python -c "import torch; import torchaudio; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## FFmpeg Setup

**⚠️ CRITICAL: FFmpeg is REQUIRED for audio processing and video input support.**

FFmpeg handles audio concatenation, resampling, and video audio extraction. Without FFmpeg in your system PATH, audio processing may fail or produce degraded results.

### Install FFmpeg

**Windows (winget):**
```powershell
winget install FFmpeg
```

**Windows (manual):**
```powershell
# Download from https://www.gyan.dev/ffmpeg/builds/
# Extract to C:\ffmpeg
# Add C:\ffmpeg\bin to system PATH
setx PATH "%PATH%;C:\ffmpeg\bin" /M
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Linux (apt):**
```bash
sudo apt update && sudo apt install ffmpeg
```

### Verify FFmpeg Installation

```bash
ffmpeg -version
```

### Automated FFmpeg Download (Linux/macOS)

```bash
# Download and install FFmpeg if not present
if ! command -v ffmpeg &> /dev/null; then
    cd /tmp
    wget https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.tar.xz
    tar -xf ffmpeg-release-essentials.tar.xz
    sudo cp ffmpeg-*/*/bin/ffmpeg /usr/local/bin/
    sudo cp ffmpeg-*/*/bin/ffprobe /usr/local/bin/
    rm -rf ffmpeg-*
fi
```

---

## One-Liner Command Patterns

AI agents can chain commands using `&&` or `;` in shell environments.

### Basic One-Liner Pattern

```bash
python src/voder.py <mode> param "value" param "value"
```

### Command Chaining Examples

**Multiple TTS operations:**

```bash
python src/voder.py tts script "Part one" voice "male" && python src/voder.py tts script "Part two" voice "female"
```

**Voice conversion pipeline:**

```bash
python src/voder.py sts base "input.wav" target "voice1.wav" && python src/voder.py sts base "output.wav" target "voice2.wav"
```

**Music generation with batch processing:**

```bash
python src/voder.py ttm lyrics "Verse 1:..." styling "pop" 30 && python src/voder.py ttm lyrics "Chorus:..." styling "rock" 30
```

### Interactive Mode (Not Recommended for Bots)

Interactive CLI mode (`python src/voder.py cli`) requires continuous terminal input and is not suitable for AI agents. Use direct one-liner commands instead.

---

## Command Reference

### Syntax

```bash
python src/voder.py <mode> [parameters]
```

### Mode Options

| Mode | Description | GPU Required | One-Liner |
|------|-------------|--------------|-----------|
| `tts` | Text-to-Speech with Voice Design | No | ✅ Yes |
| `tts+vc` | Text-to-Speech + Voice Cloning | No | ✅ Yes |
| `sts` | Speech-to-Speech (Voice Conversion) | Yes | ✅ Yes |
| `ttm` | Text-to-Music Generation | No | ✅ Yes |
| `ttm+vc` | Text-to-Music + Voice Conversion | Yes | ✅ Yes |
| `stt+tts` | Speech-to-Text + TTS | No | ❌ Interactive Only |

### Text-to-Speech (tts)

Generate speech from text with voice design using Qwen3-TTS VoiceDesign model.

```bash
python src/voder.py tts script "text here" voice "voice description"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (use `\n` for newlines) | Yes |
| `voice` | Voice prompt describing the desired voice | Yes |

**Example:**
```bash
python src/voder.py tts script "Hello, this is a test of the VODER system." voice "clean adult male with warm tone"
```

**Voice Prompt Examples:**

| Voice Type | Prompt |
|------------|--------|
| Male | "adult male, deep voice, clear pronunciation" |
| Female | "adult female, soft voice, friendly tone" |
| Energetic | "young adult, excited tone, fast speech" |
| Narrator | "middle-aged, authoritative, slow pace" |

### Text-to-Speech + Voice Clone (tts+vc)

Generate speech from text then clone it to target voice using Qwen3-TTS Base model.

```bash
python src/voder.py tts+vc script "text here" target "voice_reference.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize | Yes |
| `target` | Path to voice reference audio file | Yes |

**Example:**
```bash
python src/voder.py tts+vc script "Welcome to our podcast episode." target "speaker_reference.wav"
```

**Voice Reference Requirements:**
- Format: WAV (recommended), MP3 supported
- Duration: 5-30 seconds optimal
- Quality: Clear speech, minimal background noise
- Content: Single speaker, continuous speech

### Speech-to-Speech / Voice Conversion (sts)

Convert voice from base audio to target voice without changing content using Seed-VC v2.

```bash
python src/voder.py sts base "source_audio.wav" target "voice_reference.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `base` | Path to source audio or video | Yes |
| `target` | Path to target voice reference audio | Yes |

**Supported Input Formats:**
- Audio: WAV, MP3, FLAC, OGG
- Video: MP4, AVI, MOV, MKV (audio auto-extracted)

**Example:**
```bash
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav"
```

### Text-to-Music (ttm)

Generate music from lyrics and style prompt using ACE-Step.

```bash
python src/voder.py ttm lyrics "song lyrics" styling "style description" duration_seconds
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `\n` for newlines) | Yes |
| `styling` | Style prompt describing the music | Yes |
| `duration` | Duration in seconds (10-300) | Yes |

**Example:**
```bash
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street" styling "upbeat pop with female vocals" 30
```

**Style Prompt Examples:**

| Genre | Prompt |
|-------|--------|
| Pop | "upbeat pop, catchy melody, modern production" |
| Rock | "electric guitar, driving drums, powerful vocals" |
| Ballad | "piano accompaniment, emotional, slow tempo" |
| Electronic | "synthesizer, dance beat, energetic" |

### Text-to-Music + Voice Clone (ttm+vc)

Generate music using ACE-Step then apply voice conversion using Seed-VC.

```bash
python src/voder.py ttm+vc lyrics "song lyrics" styling "style" duration target "voice.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics | Yes |
| `styling` | Style prompt | Yes |
| `duration` | Duration in seconds (10-300) | Yes |
| `target` | Voice reference audio path | Yes |

**Example:**
```bash
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" 30 target "singer_reference.wav"
```

---

## CLI vs GUI Feature Comparison

VODER has features exclusive to each mode. Understanding these differences helps AI agents choose the right approach.

### CLI-Only Features

These features are only available via command line:

| Feature | Description |
|---------|-------------|
| **One-Liner Execution** | Single command processing |
| **Batch Processing** | Chain multiple commands with `&&` |
| **Headless Operation** | No GUI required, fully automated |
| **Direct Mode Access** | All five modes available directly |

### GUI-Only Features

These features require the graphical interface:

| Feature | Description |
|---------|-------------|
| **Dialogue Mode** | Multi-speaker script generation with character-specific voices |
| **Real-time Preview** | Watch waveform visualization during processing |
| **Interactive Script Editing** | Edit transcribed text before synthesis |
| **Voice Reference Management** | Visual audio file selection and organization |
| **Progress Visualization** | Detailed progress bar with status messages |

### Shared Features

Available in both CLI and GUI:

- Text-to-Speech (TTS)
- Text-to-Speech + Voice Cloning (TTS+VC)
- Speech-to-Speech (STS)
- Text-to-Music (TTM)
- Text-to-Music + Voice Conversion (TTM+VC)
- Output file generation (WAV format)
- Parameter customization

---

## GPU Requirements

### Modes Requiring GPU

The following modes require NVIDIA GPU with minimum 8GB VRAM because they use Seed-VC:

| Mode | Description | Minimum VRAM |
|------|-------------|--------------|
| `sts` | Speech-to-Speech Voice Conversion | 8GB |
| `ttm+vc` | Text-to-Music + Voice Conversion | 8GB |

**Seed-VC only works on NVIDIA GPUs and cannot run on CPU or non-NVIDIA graphics cards.**

### Modes Working Without GPU

The following modes work on CPU-only systems:

| Mode | Description | Performance |
|------|-------------|-------------|
| `tts` | Text-to-Speech with Voice Design | Slower but functional |
| `tts+vc` | Text-to-Speech with Voice Clone | Slower but functional |
| `ttm` | Text-to-Music | Slower but functional |

Processing will be significantly slower without GPU acceleration.

### Recommended GPU Configuration

| GPU Model | VRAM | Capability |
|-----------|------|------------|
| RTX 3060 | 8GB | Minimum for all modes |
| RTX 4070 | 12GB | Recommended, faster processing |
| RTX 4090 | 24GB | Optimal for production use |

### Verify GPU Availability

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## Limitations

### CLI Mode Limitations

1. **No Dialogue Mode**: Multi-speaker script generation requires GUI interaction
2. **No Interactive Editing**: Cannot review/edit transcribed text (STT+TTS unavailable)
3. **No Visual Feedback**: Cannot see waveform visualization during processing
4. **Single Script Mode**: TTS only supports single-voice scripts, not multi-character dialogues

### GUI Mode Limitations

1. **No Batch Processing**: Must process files one at a time manually
2. **No Command Chaining**: Cannot chain multiple operations
3. **Display Required**: Requires X11/Wayland display server
4. **Interactive Only**: Cannot run fully automated pipelines

### FFmpeg Dependencies

1. **Video Input Requires FFmpeg**: Without FFmpeg, video file audio extraction fails
2. **Audio Resampling Requires FFmpeg**: Sample rate conversion needs FFmpeg
3. **Audio Concatenation Requires FFmpeg**: Dialogue segment joining needs FFmpeg

### Model Download Requirements

1. **First Run Downloads Models**: Initial run downloads models from HuggingFace (GB-sized)
2. **HuggingFace Token May Be Required**: Some models require authentication
3. **Local Cache Location**: Models cached in `./models/` and `./checkpoints/` directories

---

## Troubleshooting

### Issue: Voice conversion fails immediately

**Cause**: No NVIDIA GPU detected or insufficient VRAM

**Solution**: Verify NVIDIA GPU is installed and has at least 8GB VRAM:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Issue: Out of memory errors

**Cause**: Model too large for available GPU memory

**Solution**: 
- Reduce TTM duration (shorter audio = less memory)
- Process shorter audio segments for STS
- Use TTS modes instead of voice conversion modes

### Issue: Module not found errors

**Cause**: Python dependencies not installed

**Solution**: Run `pip install -r requirements.txt`

### Issue: FFmpeg not found errors

**Cause**: FFmpeg not installed or not in system PATH

**Solution**: Install FFmpeg separately (see FFmpeg Setup section)

### Issue: STT+TTS mode not working

**Cause**: STT+TTS is not available in one-liner mode because it requires interactive text editing

**Solution**: Use interactive CLI mode with `python src/voder.py cli` or use GUI

### Issue: Slow processing

**Cause**: Running on CPU without GPU acceleration (for STS or TTM+VC modes)

**Solution**: Use NVIDIA GPU with 8GB+ VRAM for acceleration, or use TTS, TTS+VC, or TTM modes which work on CPU

### Issue: HuggingFace model download fails

**Cause**: Network issues or authentication required

**Solution**:
1. Check internet connection
2. Add HuggingFace token to `HF_TOKEN.txt` file
3. Retry after clearing cache: `rm -rf ./models ./checkpoints`

### Issue: Voice cloning produces poor results

**Cause**: Poor quality reference audio

**Solution**: Use high-quality reference audio:
- 5-30 seconds duration
- Clear speech, minimal background noise
- Single speaker, no music
- Consistent volume levels

### Justification: VODER Has No Known Systemic Issues

VODER is a mature tool with all modes fully operational. When issues occur, they are almost always due to:

1. **Missing Python libraries**: Solved by `pip install -r requirements.txt`
2. **FFmpeg not in PATH**: Solved by FFmpeg installation
3. **Insufficient GPU VRAM**: Use modes that work on CPU or upgrade GPU
4. **Poor reference audio quality**: Use clear, single-speaker audio samples
5. **Model download failures**: Check network or add HuggingFace token

VODER handles all internal error cases gracefully with clear error messages.

---

## Example Workflows

### Workflow 1: Voice Cloning Pipeline

```bash
# Setup
cd /workspace
git clone https://github.com/HAKORADev/VODER.git
cd VODER
pip install -r requirements.txt

# Create output directory
mkdir -p results

# Generate speech with cloned voice
python src/voder.py tts+vc script "Welcome to our weekly podcast episode." target "host_voice.wav" && \
python src/voder.py tts+vc script "Today we'll discuss the latest in AI technology." target "guest_voice.wav" && \
python src/voder.py tts+vc script "Let's begin with our first topic." target "host_voice.wav"

# Results are in results/ directory
ls results/
```

### Workflow 2: Voice Conversion with Video Input

```bash
# Install FFmpeg if needed
command -v ffmpeg || (sudo apt update && sudo apt install ffmpeg)

# Process video input (audio auto-extracted)
python src/voder.py sts base "presentation.mp4" target "narrator_voice.wav"

# Output saved to results/
ls results/voder_sts_*.wav
```

### Workflow 3: Music Generation with Voice Conversion

```bash
# Generate music with synthesized vocals
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street\nFeeling the rhythm in my feet" styling "upbeat pop with female vocals" 30

# Generate music with cloned vocals
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment\nEverything feels right" styling "rock ballad" 30 target "singer_reference.wav"

# Move results
mv results/*.wav /path/to/final/output/
```

### Workflow 4: Multi-Speaker Dialogue (GUI Required)

Note: Multi-speaker dialogue requires GUI mode for script editing and character management.

```bash
# Launch GUI for dialogue creation
python src/voder.py
# Select TTS or TTS+VC mode
# Enter dialogue script with character names
# Assign voice references per character
# Generate multi-speaker output
```

### Workflow 5: Single Command with All Parameters

```bash
python src/voder.py tts script "Hello world, this is VODER speaking." voice "clean adult male with professional tone"
```

This generates speech using Qwen3-TTS VoiceDesign model with the specified voice characteristics.

---

## Summary for AI Agents

1. **Always use one-liner commands**: `python src/voder.py <mode> [params]`
2. **Install dependencies first**: `pip install -r requirements.txt`
3. **Install FFmpeg**: Required for audio processing and video input
4. **GPU required for**: STS and TTM+VC modes only (Seed-VC)
5. **CPU-only modes**: TTS, TTS+VC, TTM (no GPU needed)
6. **Dialogue mode is GUI-only**: Not available via CLI
7. **STT+TTS is CLI-interactive only**: Use `python src/voder.py cli`
8. **Output location**: Results saved to `results/` directory
9. **Available modes**: tts, tts+vc, sts, ttm, ttm+vc
10. **TTM duration**: 10-300 seconds
11. **Video support**: VODER supports video input (auto audio extraction via FFmpeg)
12. **HuggingFace token**: Add to `HF_TOKEN.txt` for gated models

---

**For questions or issues, visit: https://github.com/HAKORADev/VODER**
