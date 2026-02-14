# VODER - Bot & AI Agent Usage Guide

This document provides comprehensive instructions for AI agents, bots, and automated systems on how to effectively use VODER for voice processing tasks. AI agents typically operate in headless environments without continuous terminal access, so this guide focuses on one‑liner commands and batch processing patterns.

## Table of Contents

1. [Purpose](#purpose)
2. [Quick Start for AI Agents](#quick-start-for-ai-agents)
3. [Installation](#installation)
4. [FFmpeg Setup](#ffmpeg-setup)
5. [One‑Liner Command Patterns](#one-liner-command-patterns)
6. [Command Reference](#command-reference)
7. [CLI vs GUI Feature Comparison](#cli-vs-gui-feature-comparison)
8. [GPU Requirements](#gpu-requirements)
9. [Limitations](#limitations)
10. [Troubleshooting](#troubleshooting)
11. [Example Workflows](#example-workflows)

---

## Purpose

VODER is a professional‑grade voice processing tool that enables seamless conversion between speech, text, and music. For AI agents operating in automated pipelines, VODER offers:

- **Unified Audio Pipeline**: Six processing modes in a single interface
- **CLI‑First Design**: All core features accessible via command line
- **No GUI Required**: Runs entirely in headless terminals
- **Full Dialogue Support**: Multi‑speaker script generation **now available in CLI** (both interactive and one‑liner)
- **Optional Background Music for Dialogue**: Automatically generated, duration‑fitted ambient music with 35% volume – available in both interactive and one‑liner CLI
- **Music Generation**: Lyrics‑to‑music synthesis with voice conversion
- **Voice Cloning**: Extract and replicate voice characteristics from reference audio

---

## Quick Start for AI Agents

AI agents typically cannot maintain interactive terminal sessions. Use the following pattern:

```bash
# Clone the repository
git clone https://github.com/HAKORADev/VODER.git && cd VODER

# Install dependencies (one‑liner)
pip install -r requirements.txt

# IMPORTANT: Upgrade protobuf to avoid compatibility issues
pip install --upgrade protobuf==5.29.6

# Process files immediately (one‑liner)
python src/voder.py tts script "Hello world" voice "male voice"

# Chain multiple operations
python src/voder.py tts script "Hello" voice "female" && python src/voder.py tts script "World" voice "male"
```

**For dialogue mode** (multiple characters), use multiple values per parameter. **To add background music**, include the `music` parameter with a description:

```bash
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic" music "soft piano, cinematic"
```

---

## Installation

### Python Dependencies

Install all required packages in a single command:

```bash
pip install -r requirements.txt
```

**IMPORTANT: After installing requirements, upgrade protobuf to avoid compatibility issues:**

```bash
pip install --upgrade protobuf==5.29.6
```

**Package explanations:**

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning framework for neural network models |
| `torchaudio` | Audio loading and processing |
| `transformers` | HuggingFace model integration |
| `PyQt5` | GUI framework (required only for GUI mode) |
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

## One‑Liner Command Patterns

AI agents can chain commands using `&&` or `;` in shell environments.

### Basic One‑Liner Pattern

```bash
python src/voder.py <mode> param "value" param "value"
```

### Dialogue Mode with Optional Background Music (One‑Liner)

Dialogue is supported in **TTS** and **TTS+VC** modes using multiple values per parameter. **Background music is optional** and only available in dialogue mode (not single mode).

- For **TTS**: supply one or more `script` lines, one or more `voice` lines (in the same character order), and optionally one `music` parameter.
- For **TTS+VC**: supply one or more `script` lines, one or more `target` file paths (in the same character order), and optionally one `music` parameter.

```bash
python src/voder.py tts script "James: Hello, I'm James." "Sarah: Hi James, I'm Sarah." voice "James: deep male voice, calm" "Sarah: young female voice, cheerful" music "ambient electronic, chill"

python src/voder.py tts+vc script "James: Let's start the meeting." "Sarah: I've prepared the slides." target "James: /path/to/james.wav" "Sarah: /path/to/sarah.wav" music "soft piano, strings"
```

**What happens when `music` is supplied:**
- VODER synthesises all dialogue segments and concatenates them.
- It then measures the exact duration of the combined dialogue.
- A music track is generated using ACE‑Step with:
  - Lyrics: `"..."` (empty placeholder)
  - Style: the value of the `music` parameter
  - Duration: exactly the dialogue length
- The music is mixed at **35% volume** relative to the dialogue.
- The final file is saved with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`).
- If `music` is omitted or set to an empty string (`music ""`), no background music is added.

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

### Interactive Mode (Also Supports Dialogue & Background Music)

Interactive CLI mode (`python src/voder.py cli`) allows you to enter multiple lines of script (empty line to finish) and automatically detects single vs. dialogue mode. It then prompts you for voice prompts (TTS) or audio file paths (TTS+VC) for each character. **After** all prompts are collected, you will be asked:

```
Add background music? (y/N):
```

If you answer `y` or `yes`, you can enter a music description. Leaving the description blank or pressing Enter without input skips the music. This mode is **not recommended for fully automated bots**, but can be used in semi‑automated workflows.

---

## Command Reference

### Syntax

```bash
python src/voder.py <mode> [parameters]
```

### Mode Options

| Mode | Description | GPU Required | One‑Liner |
|------|-------------|--------------|-----------|
| `tts` | Text‑to‑Speech with Voice Design | No | ✅ Yes (single & dialogue + optional music) |
| `tts+vc` | Text‑to‑Speech + Voice Cloning | No | ✅ Yes (single & dialogue + optional music) |
| `sts` | Speech‑to‑Speech (Voice Conversion) | Yes | ✅ Yes (single only) |
| `ttm` | Text‑to‑Music Generation | No | ✅ Yes (single only) |
| `ttm+vc` | Text‑to‑Music + Voice Conversion | Yes | ✅ Yes (single only) |
| `stt+tts` | Speech‑to‑Text + TTS | No | ❌ Interactive Only |

### Text‑to‑Speech (tts)

Generate speech from text using Qwen3‑TTS VoiceDesign model.  
**Supports both single and dialogue modes. Dialogue mode supports optional background music.**

**Single mode:**
```bash
python src/voder.py tts script "text here" voice "voice description"
```

**Dialogue mode (no music):**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2"
```

**Dialogue mode with background music:**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2" music "description of background music"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (single mode) OR `Character: text` (dialogue mode) | Yes |
| `voice` | Voice prompt (single mode) OR `Character: prompt` (dialogue mode) | Yes |
| `music` | Description for automatically generated background music (dialogue only) | No |

**Voice Prompt Examples:**

| Voice Type | Prompt |
|------------|--------|
| Male | "adult male, deep voice, clear pronunciation" |
| Female | "adult female, soft voice, friendly tone" |
| Energetic | "young adult, excited tone, fast speech" |
| Narrator | "middle‑aged, authoritative, slow pace" |

**Music Description Examples:**

| Mood | Description |
|------|-------------|
| Cinematic | "soft piano, cinematic strings, emotional" |
| Ambient | "ambient electronic, chill, atmospheric" |
| Corporate | "corporate background, professional, subtle" |
| Fantasy | "orchestral fantasy, magical, adventurous" |

### Text‑to‑Speech + Voice Clone (tts+vc)

Generate speech from text then clone it to target voice using Qwen3‑TTS Base model.  
**Supports both single and dialogue modes. Dialogue mode supports optional background music.**

**Single mode:**
```bash
python src/voder.py tts+vc script "text here" target "voice_reference.wav"
```

**Dialogue mode (no music):**
```bash
python src/voder.py tts+vc script "Character1: line1" "Character2: line2" target "Character1: /path/to/ref1.wav" "Character2: /path/to/ref2.wav"
```

**Dialogue mode with background music:**
```bash
python src/voder.py tts+vc script "Character1: line1" "Character2: line2" target "Character1: /path/to/ref1.wav" "Character2: /path/to/ref2.wav" music "description of background music"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (single) OR `Character: text` (dialogue) | Yes |
| `target` | Path to voice reference audio (single) OR `Character: path` (dialogue) | Yes |
| `music` | Description for automatically generated background music (dialogue only) | No |

**Voice Reference Requirements:**
- Format: WAV (recommended), MP3 supported
- Duration: 5‑30 seconds optimal
- Quality: Clear speech, minimal background noise
- Content: Single speaker, continuous speech

### Speech‑to‑Speech / Voice Conversion (sts)

Convert voice from base audio to target voice without changing content using Seed‑VC v2.

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
- Video: MP4, AVI, MOV, MKV (audio auto‑extracted)

**Example:**
```bash
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav"
```

### Text‑to‑Music (ttm)

Generate music from lyrics and style prompt using ACE‑Step.

```bash
python src/voder.py ttm lyrics "song lyrics" styling "style description" duration_seconds
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `\n` for newlines) | Yes |
| `styling` | Style prompt describing the music | Yes |
| `duration` | Duration in seconds (10‑300) | Yes |

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

### Text‑to‑Music + Voice Clone (ttm+vc)

Generate music using ACE‑Step then apply voice conversion using Seed‑VC.

```bash
python src/voder.py ttm+vc lyrics "song lyrics" styling "style" duration 30 target "voice.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics | Yes |
| `styling` | Style prompt | Yes |
| `duration` | Duration in seconds (10-300) | Yes |
| `target` | Voice reference audio path | Yes |

**Memory optimisation:** This mode automatically releases the ACE‑Step model from GPU memory before loading Seed‑VC, reducing peak VRAM usage.

**Example:**
```bash
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" duration 30 target "singer_reference.wav"
```

---

## CLI vs GUI Feature Comparison

VODER offers different experiences depending on the interface. Understanding these differences helps AI agents choose the right approach.

### CLI‑Only Features

| Feature | Description |
|---------|-------------|
| **One‑Liner Execution** | Single command processing |
| **Batch Processing** | Chain multiple commands with `&&` |
| **Headless Operation** | No GUI required, fully automated |
| **Direct Mode Access** | All five modes available directly |
| **Music Parameter** | One‑liner background music addition (dialogue only) |

### GUI‑Only Features

| Feature | Description |
|---------|-------------|
| **Row‑Based Visual Script Editor** | Interactive table for entering character/dialogue lines |
| **Real‑time Waveform Preview** | Watch waveform visualization during processing |
| **Audio List Management** | Visual drag‑and‑drop reference audio organization |
| **Progress Bar & Status Updates** | Detailed visual feedback |
| **Interactive Segment Selection** | Click on transcribed segments to edit text |
| **Background Music Dialog** | Clean modal dialog asking for music description before generation |

### Shared Features

Available in **both** CLI and GUI:

| Feature | CLI Implementation | GUI Implementation |
|---------|-------------------|-------------------|
| **Text‑to‑Speech (TTS)** | One‑liner with `script`/`voice` + optional `music` | Row‑based script + voice prompt fields + optional music dialog |
| **TTS+VC (Voice Cloning)** | One‑liner with `script`/`target` + optional `music` | Row‑based script + audio number dropdowns + optional music dialog |
| **Dialogue Mode** | ✅ Repeated parameters or interactive input + optional `music` | ✅ Visual script editor with character tracking + music prompt |
| **Background Music** | ✅ `music` parameter (one‑liner) or interactive yes/no | ✅ Modal dialog before generation |
| **STS / TTM / TTM+VC** | ✅ One‑liner commands | ✅ Dedicated panels |
| **Output File Generation** | ✅ Saved to `results/` | ✅ Saved to `results/` |
| **Parameter Customisation** | ✅ Duration, prompts, etc. | ✅ Duration, prompts, etc. |

**Important:** Dialogue mode and optional background music are **fully supported in CLI** for both TTS and TTS+VC, using either one‑liner repeated parameters or interactive multi‑line input.

---

## GPU Requirements

### All Modes Run on CPU

VODER operates entirely on CPU. No GPU is required for any mode. This makes VODER accessible to users without NVIDIA graphics hardware. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

### Memory Requirements by Mode

| Mode | RAM Required | GPU (CUDA) | VRAM | Notes |
|------|--------------|-------------|------|-------|
| TTS, TTS+VC (no music) | 12GB | Optional | 4GB (minimum, GTX 1060 Ti) | 8GB base + 4GB (Qwen) |
| TTS, TTS+VC (with music) | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| STT+TTS | 12GB | Optional | 4GB (minimum, GTX 1060 Ti) | 8GB base + 4GB (Qwen) |
| STS | 13GB | Optional | 14GB | 8GB base + 5GB (Seed-VC) |
| TTM | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| TTM+VC | 23GB | Optional | 16GB | 8GB base + 15GB (ACE) |

### VRAM Guidelines

| VRAM | Performance | Suitable Modes |
|------|-------------|----------------|
| No GPU (CPU only) | Slow | All modes work on CPU |
| 4GB | Usable | TTS, TTS+VC (no music), STT+TTS |
| 6GB | Minimum | TTS, TTS+VC (no music), STT+TTS |
| 14GB | Mid-range | STS, all TTS modes |
| 15-16GB | Recommended | TTS+VC with music, TTM, TTM+VC |
| 24GB | Maximum (RTX 4090) | All modes at full speed |
| T4 (16GB) | Server-grade | All modes (not consumer GPU) |

**Note:** The T4 GPU has 16GB VRAM but is a server-grade GPU, not a typical consumer card like GTX 1660 Super.

### Modes Requiring More Memory

The following modes require approximately 23GB RAM due to the ACE-Step model:

- **TTM** (Text-to-Music)
- **TTM+VC** (Text-to-Music + Voice Conversion)
- **TTS** with background music
- **TTS+VC** with background music

### Modes Working With Less Memory

The following modes work with approximately 12-13GB RAM:

- **TTS** (Text-to-Speech) - 12GB
- **TTS+VC** (TTS + Voice Cloning) - 12GB
- **STT+TTS** (Speech-to-Text + TTS) - 12GB
- **STS** (Speech-to-Speech) - 13GB

### Verify System Memory

```bash
# Check available RAM
free -h
```

---

## Limitations

### CLI Mode Limitations

1. **No Real‑time Preview**: Cannot see waveform during processing
2. **No Visual Audio Management**: Cannot drag‑and‑drop reference files
3. **STT+TTS Unavailable in One‑Liner**: Speech‑to‑text + TTS requires interactive text editing (available in `python src/voder.py cli` interactive mode, but not one‑liner)
4. **Single Mode for STS/TTM/TTM+VC**: These modes do not support multi‑speaker dialogue in CLI
5. **Music only for Dialogue**: `music` parameter is ignored in single mode

### GUI Mode Limitations

1. **No Batch Processing**: Must process files one at a time manually
2. **No Command Chaining**: Cannot chain multiple operations
3. **Display Required**: Requires X11/Wayland display server
4. **Interactive Only**: Cannot run fully automated pipelines

### FFmpeg Dependencies

1. **Video Input Requires FFmpeg**: Without FFmpeg, video file audio extraction fails
2. **Audio Resampling Requires FFmpeg**: Sample rate conversion needs FFmpeg
3. **Audio Concatenation Requires FFmpeg**: Dialogue segment joining needs FFmpeg
4. **Music Mixing Requires FFmpeg**: Dialogue + background music mixing needs FFmpeg

### Model Download Requirements

1. **First Run Downloads Models**: Initial run downloads models from HuggingFace (GB‑sized)
2. **HuggingFace Token May Be Required**: Some models require authentication
3. **Local Cache Location**: Models cached in `./models/` and `./checkpoints/` directories

---

## Troubleshooting

### Issue: Voice conversion fails immediately

**Cause**: Insufficient system memory

**Solution**: Verify you have enough RAM:

```bash
# Check available RAM
free -h
```

For STS mode, ensure at least 13GB RAM is available. For TTM/TTM+VC or TTS/TTS+VC with music, ensure at least 23GB RAM is available.

### Issue: Out of memory errors

**Cause**: Model too large for available system memory

**Solution**:
- For TTS/TTS+VC without music: Ensure at least 12GB RAM available
- For TTS/TTS+VC with music, TTM, or TTM+VC: Ensure at least 23GB RAM available
- For STS: Ensure at least 13GB RAM available
- Reduce TTM duration (shorter audio = less memory)
- Process shorter audio segments for STS
- Use TTS modes instead of voice conversion modes

### Issue: Module not found errors

**Cause**: Python dependencies not installed

**Solution**: Run `pip install -r requirements.txt`

### Issue: FFmpeg not found errors

**Cause**: FFmpeg not installed or not in system PATH

**Solution**: Install FFmpeg separately (see FFmpeg Setup section)

### Issue: STT+TTS mode not working in one‑liner

**Cause**: STT+TTS requires interactive text editing and is only available in interactive CLI or GUI

**Solution**: Use interactive CLI with `python src/voder.py cli` and select mode 1, or use GUI

### Issue: Dialogue mode not working in one‑liner

**Cause**: Missing or incorrectly ordered `voice`/`target` parameters

**Solution**: Ensure each character in the script has a corresponding `voice` (or `target`) entry **in the same character order**. Example:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep voice" "Sarah: cheerful voice"
```

### Issue: Background music not added even though `music` parameter is supplied

**Cause**: The command is not in dialogue mode (i.e., all `script` parameters are plain text without colon). Background music is only available for dialogue scripts.

**Solution**: Use `Character: text` format for at least one script parameter, or use multiple script lines with colon format.

### Issue: Music generation is slow or fails

**Cause**: ACE‑Step model loading time; insufficient resources; empty music description

**Solution**: 
- Ensure `music` description is not empty.
- Use CPU mode if GPU not available (slower but works).
- Reduce dialogue length (shorter music duration = faster generation).

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

**Solution**: Use high‑quality reference audio:
- 5‑30 seconds duration
- Clear speech, minimal background noise
- Single speaker, no music
- Consistent volume levels

### Justification: VODER Has No Known Systemic Issues

VODER is a mature tool with all modes fully operational. When issues occur, they are almost always due to:

1. **Missing Python libraries**: Solved by `pip install -r requirements.txt`
2. **FFmpeg not in PATH**: Solved by FFmpeg installation
3. **Insufficient GPU VRAM**: Use modes that work on CPU or upgrade GPU
4. **Poor reference audio quality**: Use clear, single‑speaker audio samples
5. **Model download failures**: Check network or add HuggingFace token
6. **Misuse of music parameter**: Only valid in dialogue mode

VODER handles all internal error cases gracefully with clear error messages.

---

## Example Workflows

### Workflow 1: Voice Cloning Pipeline (Single Speaker)

```bash
# Setup
cd /workspace
git clone https://github.com/HAKORADev/VODER.git
cd VODER
pip install -r requirements.txt
pip install --upgrade protobuf==5.29.6

# Create output directory
mkdir -p results

# Generate speech with cloned voice
python src/voder.py tts+vc script "Welcome to our weekly podcast episode." target "host_voice.wav" && \
python src/voder.py tts+vc script "Today we'll discuss the latest in AI technology." target "guest_voice.wav" && \
python src/voder.py tts+vc script "Let's begin with our first topic." target "host_voice.wav"

# Results are in results/ directory
ls results/
```

### Workflow 2: Dialogue Generation with Voice Design + Background Music (CLI One‑Liner)

```bash
python src/voder.py tts script "Narrator: Once upon a time, in a digital realm," "Alice: I wonder what secrets this code holds." "Bob: Let's find out together!" voice "Narrator: calm male voice, slow and measured" "Alice: bright female voice, curious" "Bob: enthusiastic male voice, friendly" music "orchestral fantasy, magical, adventurous"
```

### Workflow 3: Dialogue Generation with Voice Cloning + Background Music (CLI One‑Liner)

```bash
python src/voder.py tts+vc script "James: Welcome to our podcast!" "Sarah: Thanks for having me, James." "James: So, Sarah, tell us about your work." target "James: /voices/james_reference.wav" "Sarah: /voices/sarah_reference.wav" "James: /voices/james_reference.wav" music "soft piano, cinematic strings"
```

### Workflow 4: Voice Conversion with Video Input

```bash
# Install FFmpeg if needed
command -v ffmpeg || (sudo apt update && sudo apt install ffmpeg)

# Process video input (audio auto‑extracted)
python src/voder.py sts base "presentation.mp4" target "narrator_voice.wav"

# Output saved to results/
ls results/voder_sts_*.wav
```

### Workflow 5: Music Generation with Voice Conversion

```bash
# Generate music with synthesized vocals
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street\nFeeling the rhythm in my feet" styling "upbeat pop with female vocals" 30

# Generate music with cloned vocals
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment\nEverything feels right" styling "rock ballad" 30 target "singer_reference.wav"

# Move results
mv results/*.wav /path/to/final/output/
```

### Workflow 6: Interactive Dialogue Creation with Background Music (Semi‑Automated)

For complex scripts where you want to decide about music interactively:

```bash
python src/voder.py cli
# Select option 2 (TTS) or 3 (TTS+VC)
# Enter multiple lines of dialogue (empty line to finish)
# VODER will automatically prompt you for voice prompts / audio paths per character
# Then it will ask: "Add background music? (y/N):"
# Answer y and enter a description, or just press Enter to skip
```

---

## Summary for AI Agents

1. **Always use one‑liner commands**: `python src/voder.py <mode> [params]`
2. **Dialogue + optional background music** is now available in CLI for TTS and TTS+VC:
   - Use multiple values per parameter (e.g., `script "line1" "line2"`)
   - Add `music "description"` to automatically generate background music.
3. **Install dependencies first**: `pip install -r requirements.txt`
4. **Upgrade protobuf**: `pip install --upgrade protobuf==5.29.6` (avoids compatibility issues)
5. **Install FFmpeg**: Required for audio processing, video input, and music mixing
6. **RAM required**: 12GB minimum (23GB for TTM, TTM+VC, or TTS/TTS+VC with music)
7. **GPU/VRAM**: Optional - 4GB minimum (6GB recommended, 16GB for best performance)
8. **CPU-only operation**: All modes run on CPU, no GPU required
9. **STT+TTS is CLI‑interactive only**: Use `python src/voder.py cli`
10. **Output location**: Results saved to `results/` directory
11. **Available modes**: tts, tts+vc, sts, ttm, ttm+vc
12. **TTM duration**: 10‑300 seconds (use `duration n` or just `n` at the end)
13. **Video support**: VODER supports video input (auto audio extraction via FFmpeg)
14. **HuggingFace token**: Add to `HF_TOKEN.txt` for gated models
15. **Music parameter behaviour**: Only effective for dialogue scripts; ignored otherwise; empty string treated as skip

---

**For questions or issues, visit: https://github.com/HAKORADev/VODER**
