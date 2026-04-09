# VODER - Bot & AI Agent Usage Guide

This document provides comprehensive instructions for AI agents, bots, and automated systems on how to effectively use VODER for voice processing tasks. AI agents typically operate in headless environments without continuous terminal access, so this guide focuses on one‑liner commands and batch processing patterns.

## Table of Contents

1. [Purpose](#purpose)
2. [Quick Start for AI Agents](#quick-start-for-ai-agents)
3. [Installation](#installation)
4. [FFmpeg Setup](#ffmpeg-setup)
5. [HF_TOKEN Setup for AI Agents](#hf_token-setup-for-ai-agents)
6. [One‑Liner Command Patterns](#one-liner-command-patterns)
7. [Command Reference](#command-reference)
8. [New CLI Features](#new-cli-features)
9. [CLI vs GUI Feature Comparison](#cli-vs-gui-feature-comparison)
10. [GPU Requirements](#gpu-requirements)
11. [Limitations](#limitations)
12. [Troubleshooting](#troubleshooting)
13. [Example Workflows](#example-workflows)

---

## Purpose

VODER is a professional‑grade voice processing tool that enables seamless conversion between speech, text, and music. For AI agents operating in automated pipelines, VODER offers:

- **Unified Audio Pipeline**: Nine processing modes in a single interface
- **CLI‑First Design**: All core features accessible via command line
- **No GUI Required**: Runs entirely in headless terminals
- **Full Dialogue Support**: Multi‑speaker script generation **now available in CLI** (both interactive and one‑liner)
- **Script Directives**: Per-line control over timing, volume, and duration
- **SFX Integration**: Embed sound effects directly in dialogue scripts
- **Optional Background Music for Dialogue**: Automatically generated, duration‑fitted ambient music with configurable volume levels
- **Music Generation**: Lyrics‑to‑music synthesis with voice conversion
- **Sound Effects Generation**: Text-to-audio synthesis for custom sound design
- **Speech Enhancement**: Denoise, dereverberate, and restore speech audio
- **Voice Cloning**: Extract and replicate voice characteristics from reference audio
- **Standalone STT**: Transcribe audio, video, images, and YouTube URLs to text
- **Speaker Diarization**: Identify and label individual speakers in multi‑speaker audio
- **Image OCR**: Extract text from images as dialogue input for TTS processing
- **YouTube/Video Download**: Process audio from YouTube, Bilibili, and TikTok URLs directly
- **Automatic Voice Extraction**: Extract individual voice clips from multi‑speaker sources for cloning
- **Result Routing**: Copy results to any filesystem path using the `result` parameter

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

# Transcribe audio to text (STT)
python src/voder.py stt "audio.wav" timestamp dialogue result "/output/transcript.txt"

# Enhance speech audio (SE)
python src/voder.py se "noisy_audio.wav" result "/output/enhanced.wav"

# Generate sound effects (SFX)
python src/voder.py sfx sound "thunder rumbling" duration 10 result "/output/thunder.wav"

# Chain multiple operations
python src/voder.py tts script "Hello" voice "female" && python src/voder.py tts script "World" voice "male"
```

**For dialogue mode** (multiple characters), use multiple values per parameter. **To add background music**, include the `music` parameter with a description:

```bash
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic" music "soft piano, cinematic"
```

**With SFX lines embedded:**
```bash
python src/voder.py tts script "James: Hello" "sfx: door bell /duration:3" "Sarah: Hi there!" voice "James: male" "Sarah: female" music "ambient" level "0:30-60:50"
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
| `yt-dlp` | YouTube, Bilibili, and TikTok video/audio download |
| `easyocr` | Image text extraction (OCR) for processing images as dialogue input |
| `lightning` | PyTorch Lightning backend required by pyannote for speaker diarization |
| `sox` | Audio processing utilities (resampling, format conversion, channel manipulation) |
| `einx` | Required for UniSE speech enhancement model |
| `x-transformers` | Required for UniSE speech enhancement model |
| `safetensors` | Required for TangoFlux and UniSE model loading |
| `soxr` | High-quality audio resampling |
| `tqdm` | Progress bars |
| `packaging` | Version handling |

**Note:** `pyannote.audio` is bundled locally in `src/libs/pyannote` and does not require a separate pip install. However, a HuggingFace token is required for speaker diarization (see [HF_TOKEN Setup for AI Agents](#hf_token-setup-for-ai-agents)).

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

## HF_TOKEN Setup for AI Agents

**⚠️ REQUIRED for speaker diarization in STT mode.**

The pyannote speaker diarization model is gated on HuggingFace and requires authentication. Without a valid token, the `dialogue` flag in STT mode will fail.

### Step 1: Accept Model Conditions

Visit the following HuggingFace model pages and accept the user agreement for each:

1. **Speaker Diarization**: https://huggingface.co/pyannote/speaker-diarization-community-1
2. **Segmentation**: https://huggingface.co/pyannote/segmentation-3.0

You must be logged into a HuggingFace account and click "Accept" on each model's page.

### Step 2: Create the Token File

```bash
# Create HF_TOKEN.txt in the VODER root directory
echo "hf_your_token_here" > HF_TOKEN.txt
```

The token is read automatically from `HF_TOKEN.txt` when diarization is requested.

### Environment Variable Fallback

If `HF_TOKEN.txt` does not exist, VODER checks for the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="hf_your_token_here"
python src/voder.py stt "meeting.wav" dialogue
```

### Verify Token

```bash
# Test that the token is valid
python -c "from huggingface_hub import HfFolder; print('Token set:', bool(HfFolder.get_token()))"
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

- For **TTS**: supply one or more `script` lines, one or more `voice` lines (in the same character order), and optionally one `music` parameter and one `level` parameter.
- For **TTS+VC**: supply one or more `script` lines, one or more `target` file paths (in the same character order), and optionally one `music` parameter and one `level` parameter.

```bash
python src/voder.py tts script "James: Hello, I'm James." "Sarah: Hi James, I'm Sarah." voice "James: deep male voice, calm" "Sarah: young female voice, cheerful" music "ambient electronic, chill"

python src/voder.py tts+vc script "James: Let's start the meeting." "Sarah: I've prepared the slides." target "James: /path/to/james.wav" "Sarah: /path/to/sarah.wav" music "soft piano, strings" level "40"
```

**With SFX lines embedded:**
```bash
python src/voder.py tts script "Narrator: Once upon a time" "sfx: magical chime /duration:5 /level:50" "Narrator: In a faraway land..." voice "Narrator: deep storytelling voice" music "fantasy orchestral"
```

**What happens when `music` is supplied:**
- VODER synthesises all dialogue segments and concatenates them.
- It then measures the exact duration of the combined dialogue.
- A music track is generated using ACE‑Step with:
  - Lyrics: `"..."` (empty placeholder)
  - Style: the value of the `music` parameter
  - Duration: exactly the dialogue length
- The music is mixed at **configurable volume** (default 35%, adjustable via `level` parameter).
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
python src/voder.py ttm lyrics "Verse 1:..." styling "pop" duration 30 && python src/voder.py ttm lyrics "Chorus:..." styling "rock" duration 30
```

**Speech enhancement pipeline:**

```bash
python src/voder.py se "noisy_recording.wav" result "/clean/recording.wav"
```

**Sound effects generation:**

```bash
python src/voder.py sfx sound "rain on tin roof" duration 15 result "/sfx/rain.wav"
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
| `tts` | Text‑to‑Speech with Voice Design | No | ✅ Yes (single & dialogue + optional music + SFX support) |
| `tts+vc` | Text‑to‑Speech + Voice Cloning | No | ✅ Yes (single & dialogue + optional music + SFX support) |
| `sts` | Speech‑to‑Speech (Voice Conversion) | No | ✅ Yes (single only) |
| `ttm` | Text‑to‑Music Generation | No | ✅ Yes (single only) |
| `ttm+vc` | Text‑to‑Music + Voice Conversion | No | ✅ Yes (single only) |
| `stt` | Speech‑to‑Text Transcription | No | ✅ Yes (single, batch, timestamps, diarization, URLs) |
| `stt+tts` | Speech‑to‑Text + TTS | No | ❌ Interactive Only |
| `se` | Speech Enhancement (Denoise/Dereverb) | No | ✅ Yes |
| `sfx` | Sound Effects Generation | No | ✅ Yes |

### Text‑to‑Speech (tts)

Generate speech from text using Qwen3‑TTS VoiceDesign model.  
**Supports both single and dialogue modes. Dialogue mode supports optional background music and SFX lines.**

**Single mode:**
```bash
python src/voder.py tts script "text here" voice "voice description"
```

**OCR input (image to narration):**
```bash
python src/voder.py tts ocr "path/to/image.png" voice "text: professional male narrator"

python src/voder.py tts ocr "script_screenshot.jpg" voice "text: warm female voice"
```

**Dialogue mode (no music):**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2"
```

**Dialogue mode with background music:**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2" music "description of background music"
```

**Dialogue mode with music volume control:**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt" "Character2: voice prompt" music "soft piano" level "0:30-60:50"
```

**Dialogue with SFX lines:**
```bash
python src/voder.py tts script "James: Hello" "sfx: door bell /duration:3 /level:60" "Sarah: Hi!" voice "James: male" "Sarah: female" music "ambient"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (single mode) OR `Character: text` (dialogue mode) OR `sfx: description /duration:nn` (SFX lines) | Yes |
| `voice` | Voice prompt (single mode) OR `Character: prompt` (dialogue mode) for generated voices | Yes (unless all scripts are SFX lines or using target) |
| `target` | Path to voice reference (single) OR `Character: path` (dialogue) for cloned voices — can mix with `voice` | No (but required if no `voice` for non-SFX lines) |
| `music` | Description for automatically generated background music (dialogue only) | No |
| `level` | Music volume levels e.g. `"10:20-50 30:60-80"` (dialogue modes, default: 35%) | No |

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

**Level Specification Examples:**

| Format | Meaning |
|--------|---------|
| `"35"` | Constant 35% volume throughout |
| `"50"` | Constant 50% volume throughout |
| `"0:20-60:50"` | 20% at 0 seconds, 50% at 60 seconds |
| `"0:30-30:50+10"` | Fade from 30% to 50% over 10 seconds starting at 0s |

**Cross-use Feature (Mixing Generated and Cloned Voices):**

Both TTS and TTS+VC one-line modes support mixing generated and cloned voices in the same dialogue. Use `voice "Character: prompt"` for generated voices and `target "Character: path"` for cloned voices:

```bash
# TTS mode with mixed voices: James uses generated, Sarah uses cloned
python src/voder.py tts script "James: Hello!" "Sarah: Hi there!" voice "James: deep male voice" target "Sarah: /path/to/sarah_voice.wav"

# TTS+VC mode with mixed voices: James uses cloned, Sarah uses generated
python src/voder.py tts+vc script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james_voice.wav" voice "Sarah: bright female voice"
```

**Important:** A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

### Text‑to‑Speech + Voice Clone (tts+vc)

Generate speech from text then clone it to target voice using Qwen3‑TTS Base model.  
**Supports both single and dialogue modes. Dialogue mode supports optional background music and SFX lines.**

**Single mode:**
```bash
python src/voder.py tts+vc script "text here" target "voice_reference.wav"
```

**OCR input (image to narration with voice clone):**
```bash
python src/voder.py tts+vc ocr "path/to/image.png" target "text: voice_reference.wav"

python src/voder.py tts+vc ocr "subtitle_image.jpg" target "text: speaker_clone.wav"
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
| `script` | Text to synthesize (single) OR `Character: text` (dialogue) OR `sfx: description /duration:nn` (SFX lines) | Yes |
| `target` | Path to voice reference audio (single) OR `Character: path` (dialogue) for cloned voices | Yes (unless all scripts are SFX lines or using voice) |
| `voice` | Voice prompt (single) OR `Character: prompt` (dialogue) for generated voices — can mix with `target` | No (but required if no `target` for non-SFX lines) |
| `music` | Description for automatically generated background music (dialogue only) | No |
| `level` | Music volume levels (dialogue modes, default: 35%) | No |

**Voice Reference Requirements:**
- Format: WAV (recommended), MP3 supported
- Duration: 5‑30 seconds optimal
- Quality: Clear speech, minimal background noise
- Content: Single speaker, continuous speech

### Speech‑to‑Speech / Voice Conversion (sts)

Convert voice from base audio to target voice without changing content using Seed‑VC v2. **MSTS (Music-STS)**: For musical inputs, add the `music` keyword to use Seed‑VC v1 at 44.1kHz for better quality.

```bash
python src/voder.py sts base "source_audio.wav" target "voice_reference.wav"

python src/voder.py sts base "song.wav" target "voice_reference.wav" music
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `base` | Path to source audio or video | Yes |
| `target` | Path to target voice reference audio | Yes |
| `music` | Use Seed-VC v1 (44.1kHz) for musical inputs | No |
| `mimic` | Transfer accent and speaking style from target voice | No |

**Supported Input Formats:**
- Audio: WAV, MP3, FLAC, OGG
- Video: MP4, AVI, MOV, MKV (audio auto‑extracted)

**MSTS Example:**
```bash
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" music
```

**Mimic Example (Style Transfer):**
```bash
python src/voder.py sts base "source_audio.wav" target "character_voice.wav" mimic
```
**Note:** `mimic` and `music` cannot be used together.

### Text‑to‑Music (ttm)

Generate music from lyrics and style prompt using ACE‑Step. Supports instrumental-only generation with empty lyrics.

```bash
python src/voder.py ttm lyrics "song lyrics" styling "style description" duration 30
```

**Instrumental music (no vocals):**
```bash
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill" duration 60
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `"..."` for instrumental only) | Yes |
| `styling` | Style prompt describing the music | Yes |
| `duration` | Duration in seconds (10‑300) | Yes |

**Example:**
```bash
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street" styling "upbeat pop with female vocals" duration 30

# Instrumental
python src/voder.py ttm lyrics "..." styling "cinematic orchestral, dramatic" duration 90
```

**Style Prompt Examples:**

| Genre | Prompt |
|-------|--------|
| Pop | "upbeat pop, catchy melody, modern production" |
| Rock | "electric guitar, driving drums, powerful vocals" |
| Ballad | "piano accompaniment, emotional, slow tempo" |
| Electronic | "synthesizer, dance beat, energetic" |
| Instrumental | "ambient electronic, atmospheric, no vocals" |

### Text‑to‑Music + Voice Clone (ttm+vc)

Generate music using ACE‑Step then apply voice conversion using Seed‑VC.

```bash
python src/voder.py ttm+vc lyrics "song lyrics" styling "style" duration 30 target "voice.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `"..."` for instrumental) | Yes |
| `styling` | Style prompt | Yes |
| `duration` | Duration in seconds (10-300) | Yes |
| `target` | Voice reference audio path | Yes |

**Memory optimisation:** This mode automatically releases the ACE‑Step model from GPU memory before loading Seed‑VC, reducing peak VRAM usage.

**Example:**
```bash
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" duration 30 target "singer_reference.wav"
```

### Speech‑to‑Text (stt)

Transcribe audio, video, images, or YouTube URLs to text using Whisper. Supports timestamps, speaker diarization, batch processing, and automatic result routing.

**Basic transcription:**
```bash
python src/voder.py stt "audio.wav"
```

**Transcription with timestamps:**
```bash
python src/voder.py stt "audio.wav" timestamp
```

**Transcription with speaker diarization (dialogue format):**
```bash
python src/voder.py stt "audio.wav" dialogue
```

**Full transcription with timestamps, diarization, and result routing:**
```bash
python src/voder.py stt "audio.wav" timestamp dialogue result "/output/transcript.txt"
```

**Transcribe a YouTube video:**
```bash
python src/voder.py stt "https://www.youtube.com/watch?v=VIDEO_ID" timestamp dialogue
```

**Transcribe an image (OCR):**
```bash
python src/voder.py stt "screenshot.png"
```

**Batch transcription (multiple files):**
```bash
python src/voder.py stt "file1.wav" "file2.mp3" "file3.mp4" timestamp result "/output/batch/"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `files` | One or more input file paths or URLs (positional arguments after `stt`) | Yes |
| `timestamp` | Include word‑level timestamps in the output | No |
| `dialogue` | Enable speaker diarization (requires HF_TOKEN) | No |
| `result` | Copy result file(s) to the specified path (file or directory) | No |

**Supported Input Formats:**
- **Audio**: WAV, MP3, FLAC, OGG, AAC, M4A, WMA
- **Video**: MP4, AVI, MOV, MKV, WebM (audio auto‑extracted via FFmpeg)
- **Images**: PNG, JPG, JPEG, BMP, TIFF (text extracted via EasyOCR)
- **YouTube**: Direct YouTube URL (audio downloaded via yt-dlp)
- **Bilibili**: Direct Bilibili URL (audio downloaded via yt-dlp)
- **TikTok**: Direct TikTok URL (audio downloaded via yt-dlp)

**Output Format:**

The transcription result is saved as a `.txt` file in the `results/` directory. The content varies by flags used:

| Flags Used | Output Format |
|------------|---------------|
| (none) | Plain text transcript |
| `timestamp` | Timestamped transcript with `[MM:SS.mmm → MM:SS.mmm]` segments |
| `dialogue` | Speaker‑labelled dialogue format (`Speaker 1: ...`, `Speaker 2: ...`) |
| `timestamp dialogue` | Both timestamps and speaker labels combined |

**Diarization Output Example:**
```
[00:00.000 → 00:03.500] Speaker 1: Welcome everyone to today's meeting.
[00:03.500 → 00:07.200] Speaker 2: Thank you, let's get started with the agenda.
[00:07.200 → 00:12.100] Speaker 1: First item, we need to review the quarterly results.
```

**Batch Processing Notes:**
- When multiple input files are provided, each is transcribed independently.
- If `result` points to a directory, all output files are copied there.
- If `result` points to a file path, only the last result is copied (use directory for batch).

### Speech Enhancement (se)

Enhance speech audio by denoising, dereverberating, and restoring clarity. Uses UniSE model from Alibaba's unified-audio project.

**Basic enhancement:**
```bash
python src/voder.py se "noisy_audio.wav"
```

**Enhance audio from video:**
```bash
python src/voder.py se "recording.mp4"
```

**Save to specific location:**
```bash
python src/voder.py se "audio.wav" result "/output/enhanced.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `file` | Input audio or video file path (positional) | Yes |
| `result` | Copy result to the specified path | No |

**Supported Input Formats:**
- **Audio**: WAV, MP3, FLAC, OGG, AAC, M4A
- **Video**: MP4, AVI, MOV, MKV (audio auto‑extracted)

**Important Notes:**
- **Not for music**: SE is optimized for speech only; music may be degraded
- **16kHz output**: Enhanced audio is output at 16kHz sample rate
- **Best for**: Noisy recordings, distant microphones, room echo removal

**Output Example:**
```
Loading UniSE Speech Enhancement model...
Enhancing audio...
✓ Success! Output saved to: results/voder_se_20260408_120000.wav
```

### Sound Effects Generation (sfx)

Generate custom sound effects from text descriptions using TangoFlux model.

**Basic sound effect:**
```bash
python src/voder.py sfx sound "thunder rumbling in the distance" duration 10
```

**With quality parameters:**
```bash
python src/voder.py sfx sound "rain on a tin roof" duration 15 steps 50 guide 3.5
```

**Save to specific location:**
```bash
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"
```

**Parameters:**

| Parameter | Description | Range | Default | Required |
|-----------|-------------|-------|---------|----------|
| `sound` | Text description of the sound effect | Any text | — | Yes |
| `duration` | Duration in seconds | 1-30 | — | Yes |
| `steps` | Inference steps (quality vs speed) | 1-100 | 30 | No |
| `guide` | Guidance scale (prompt adherence) | 1.0-10.0 | 4.5 | No |
| `result` | Output file path | Any path | — | No |

**Sound Prompt Tips:**

| Sound Type | Example Prompts |
|------------|-----------------|
| Nature | "heavy rain on a tin roof with distant thunder" |
| Impacts | "deep punchy kick drum impact with reverb tail" |
| Ambient | "busy coffee shop atmosphere with clinking cups" |
| Transitions | "swoosh whoosh transition with rising pitch" |
| Mechanical | "old car engine starting and idling roughly" |
| Sci-fi | "futuristic laser blast with digital distortion" |

**Step/Guide Guidelines:**

| Parameter | Low Value | Default | High Value |
|-----------|-----------|---------|------------|
| `steps` | 10-20 (fast, basic) | 30 (good) | 50-100 (slow, best quality) |
| `guide` | 1.0-2.0 (creative) | 4.5 (balanced) | 7.0-10.0 (strict adherence) |

---

## New CLI Features

The following features were added in the 04/08/2026 major update and are available to AI agents via CLI.

### Script Directives (Per-Line Control)

Append directives to dialogue lines for fine-grained control:

| Directive | Format | Description |
|-----------|--------|-------------|
| `/time:nn` | `/time:5` | Position at 5 seconds from start |
| `/time:nn-nn` | `/time:10-3` | Position at 10s, cut 3s from end |
| `/time:nn+nn` | `/time:5+2` | Position at 5s, cut 2s from start |
| `/time:nn-nn+nn` | `/time:10-3+2` | Position at 10s, cut 3s from end, cut 2s from start |
| `/level:0-100` | `/level:75` | Volume level for this line (default: 100) |
| `/duration:1-30` | `/duration:10` | Duration for SFX lines (required for `sfx:`) |

**Example:**
```bash
python src/voder.py tts script "James: Welcome! /time:0 /level:100" "sfx: intro music /duration:5 /level:40 /time:0" "Sarah: Hello everyone! /time:6" voice "James: male" "Sarah: female"
```

### SFX Lines in Dialogue

Use `sfx:` as the character name to embed sound effects:

```bash
python src/voder.py tts script "James: Hello" "sfx: door bell /duration:3 /level:60" "Sarah: Who's there?" voice "James: male" "Sarah: female"
```

### Cross-use Feature (Mix Generated and Cloned Voices)

Both TTS and TTS+VC one-line modes support mixing `voice` and `target` parameters in the same dialogue:

```bash
# TTS mode: James generated, Sarah cloned
python src/voder.py tts script "James: Hello!" "Sarah: Hi!" voice "James: male" target "Sarah: /path/to/sarah.wav"

# TTS+VC mode: James cloned, Sarah generated
python src/voder.py tts+vc script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james.wav" voice "Sarah: female"
```

**Note:** A character cannot have both `voice` and `target` — each character must use one or the other.

### Universal: `result` Parameter

Copy the generated result file to any filesystem path. Works with **all modes** (tts, tts+vc, sts, ttm, ttm+vc, stt, se, sfx).

```bash
# Copy TTS result to a specific directory
python src/voder.py tts script "Hello world" voice "male voice" result "/mnt/shared/output/"

# Copy STT transcript to a specific file
python src/voder.py stt "meeting.wav" timestamp result "/data/transcripts/meeting.txt"

# Copy music result
python src/voder.py ttm lyrics "Hello world" styling "pop" duration 30 result "/mnt/shared/music/"

# Copy speech enhancement result
python src/voder.py se "noisy.wav" result "/clean/audio.wav"

# Copy sound effect result
python src/voder.py sfx sound "thunder" duration 10 result "/sfx/thunder.wav"
```

- If the path ends with `/`, it is treated as a directory (created if needed).
- If the path does not end with `/`, it is treated as a file path (parent directories created if needed).
- The original result is always saved in `results/`; `result` creates an additional copy.

### STT‑Only: `timestamp` Flag

Include word‑level timestamps in STT transcription output.

```bash
python src/voder.py stt "interview.wav" timestamp
```

### STT‑Only: `dialogue` Flag

Enable speaker diarization to identify and label individual speakers. Requires HF_TOKEN (see [HF_TOKEN Setup for AI Agents](#hf_token-setup-for-ai-agents)).

```bash
python src/voder.py stt "meeting.wav" dialogue
```

### YouTube URL Input

Pass a YouTube, Bilibili, or TikTok URL directly as input for STT transcription or dialogue analysis. Audio is downloaded automatically via yt-dlp.

```bash
# Transcribe a YouTube video
python src/voder.py stt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" timestamp

# Transcribe with diarization
python src/voder.py stt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" dialogue

# Also works with Bilibili and TikTok
python src/voder.py stt "https://www.bilibili.com/video/BV1xx411c7mD" timestamp
```

### Image File Input

Pass an image file (PNG, JPG, etc.) to STT mode. Text is extracted via EasyOCR and returned as a plain text transcript. This enables using images of scripts, subtitles, or screenshots as dialogue input.

```bash
# Extract text from an image
python src/voder.py stt "script_screenshot.png"

# Extract text and save to a specific path
python src/voder.py stt "subtitle_image.jpg" result "/output/extracted_text.txt"
```

### Batch File Processing (STT)

Provide multiple input files to STT mode for batch transcription. All files are processed sequentially.

```bash
# Transcribe multiple audio files
python src/voder.py stt "episode1.wav" "episode2.wav" "episode3.wav" timestamp

# Batch with result routing to a directory
python src/voder.py stt "meeting1.wav" "meeting2.wav" dialogue result "/data/transcripts/"
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
| **Direct Mode Access** | All nine modes available directly |
| **Music Parameter** | One‑liner background music addition (dialogue only) |
| **Level Parameter** | Configurable music volume with time-based segments |
| **Script Directives** | Per-line timing, volume, and duration control |
| **SFX in Dialogue** | Embed sound effects via `sfx:` character |
| **Cross-use Feature** | Mix generated (`voice`) and cloned (`target`) voices in same dialogue |
| **STT Transcription** | Speech-to-text with timestamps, diarization, and batch processing |
| **YouTube/URL Input** | Direct transcription from YouTube, Bilibili, TikTok URLs |
| **Image OCR Input** | Text extraction from images via EasyOCR |
| **Result Routing** | Copy output to arbitrary filesystem paths with `result` parameter |
| **Speech Enhancement** | Denoise, dereverberate, restore speech audio |
| **Sound Effects Generation** | Text-to-audio synthesis with configurable parameters |

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
| **Text‑to‑Speech (TTS)** | One‑liner with `script`/`voice` + optional `music`/`level` | Row‑based script + voice prompt fields + optional music dialog |
| **TTS+VC (Voice Cloning)** | One‑liner with `script`/`target` + optional `music`/`level` | Row‑based script + audio number dropdowns + optional music dialog |
| **Dialogue Mode** | ✅ Repeated parameters or interactive input + optional `music`/`level` | ✅ Visual script editor with character tracking + music prompt |
| **SFX Lines** | ✅ `sfx: description /duration:nn` in script | ✅ `sfx` character in dialogue rows |
| **Script Directives** | ✅ `/time:`, `/level:`, `/duration:` in dialogue lines | ✅ Same directive syntax in dialogue rows |
| **Background Music** | ✅ `music` parameter (one‑liner) or interactive yes/no | ✅ Modal dialog before generation |
| **STS / TTM / TTM+VC** | ✅ One‑liner commands | ✅ Dedicated panels |
| **STT (Speech-to-Text)** | ✅ One‑liner with optional `timestamp`, `dialogue`, `result` | ✅ Dedicated panel |
| **SE (Speech Enhancement)** | ✅ One‑liner command | ✅ Dedicated panel |
| **SFX (Sound Effects)** | ✅ One‑liner with `sound`, `duration`, `steps`, `guide` | ✅ Dedicated panel |
| **Output File Generation** | ✅ Saved to `results/` | ✅ Saved to `results/` |
| **Parameter Customisation** | ✅ Duration, prompts, etc. | ✅ Duration, prompts, etc. |

**Important:** Dialogue mode, script directives, SFX lines, and optional background music are **fully supported in CLI** for both TTS and TTS+VC, using either one‑liner repeated parameters or interactive multi‑line input.

---

## GPU Requirements

### All Modes Run on CPU

VODER operates entirely on CPU. No GPU is required for any mode. This makes VODER accessible to users without NVIDIA graphics hardware. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

### Memory Requirements by Mode

| Mode | RAM Required | GPU (CUDA) | VRAM | Notes |
|------|--------------|-------------|------|-------|
| TTS, TTS+VC (no music) | 12GB | Optional | 4GB (minimum, GTX 1060) | 8GB base + 4GB (Qwen) |
| TTS, TTS+VC (with music) | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| STT | 12GB | Optional | 4GB (minimum) | 8GB base + 4GB (Whisper) |
| STT + Diarization | 15GB | Optional | 4GB (minimum) | +3GB (Pyannote) |
| STT+TTS | 12GB | Optional | 4GB (minimum, GTX 1060) | 8GB base + 4GB (Qwen) |
| STS | 13GB | Optional | 14GB | 8GB base + 5GB (Seed-VC) |
| TTM | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| TTM+VC | 23GB | Optional | 16GB | 8GB base + 15GB (ACE) |
| SE | 11GB | Optional | 4GB | 8GB base + 2-3GB (UniSE) |
| SFX | 12GB | Optional | 4GB | 8GB base + 3-4GB (TangoFlux) |

### VRAM Guidelines

| VRAM | Performance | Suitable Modes |
|------|-------------|----------------|
| No GPU (CPU only) | Slow | All modes work on CPU |
| 4GB | Usable | TTS, TTS+VC (no music), STT, STT+TTS, SE, SFX |
| 6GB | Minimum | TTS, TTS+VC (no music), STT, STT+TTS, SE, SFX |
| 14GB | Mid-range | STS, all TTS modes, SE, SFX |
| 15-16GB | Recommended | TTS+VC with music, TTM, TTM+VC, all modes |
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

The following modes work with approximately 11-13GB RAM:

- **TTS** (Text-to-Speech) - 12GB
- **TTS+VC** (TTS + Voice Cloning) - 12GB
- **STT** (Speech-to-Text) - 12GB
- **STT+TTS** (Speech-to-Text + TTS) - 12GB
- **STS** (Speech-to-Speech) - 13GB
- **SE** (Speech Enhancement) - 11GB
- **SFX** (Sound Effects) - 12GB

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

### STT Mode Limitations

1. **STT+TTS is Interactive Only**: Combining transcription with re‑synthesis requires interactive CLI or GUI
2. **Diarization Requires HF_TOKEN**: Speaker diarization needs a valid HuggingFace token with accepted model conditions
3. **YouTube Download Requires Internet**: URL-based transcription needs network access and yt-dlp installed
4. **Image OCR Accuracy Varies**: Text extraction quality depends on image resolution, font clarity, and language support
5. **Speaker Diarization Accuracy Varies**: Best results with clear audio, minimal background noise, and ≤4 speakers; overlapping speech and noisy environments reduce accuracy

### SE Mode Limitations

1. **Speech Only**: Not designed for music enhancement
2. **16kHz Output**: Output sample rate is fixed at 16kHz
3. **Cannot Recover Missing Data**: Severely corrupted audio has restoration limits

### SFX Mode Limitations

1. **Duration Limit**: Maximum 30 seconds per sound effect
2. **Prompt Sensitivity**: Results vary based on prompt quality
3. **No Multi-file Batch**: Each SFX command generates one sound effect

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
- For STT with diarization: Ensure at least 15GB RAM available
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

### Issue: SFX line missing duration error

**Cause**: SFX lines require `/duration:nn` directive

**Solution**: Add duration to SFX lines:
```bash
python src/voder.py tts script "James: Hello" "sfx: thunder /duration:5" voice "James: male"
```

### Issue: Speech enhancement degrades music

**Cause**: SE mode is designed for speech only

**Solution**: Don't use SE mode on music content

### Issue: Sound effect doesn't match prompt

**Cause**: Insufficient guidance or prompt quality

**Solution**: 
- Increase `guide` parameter (try 7-10)
- Make prompts more descriptive
- Increase `steps` for better quality

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

### Issue: Pyannote speaker diarization fails

**Cause**: Missing or invalid HF_TOKEN, model conditions not accepted, or torchaudio compatibility issues

**Solution**:
1. Ensure `HF_TOKEN.txt` exists in the VODER root directory with a valid token
2. Visit https://huggingface.co/pyannote/speaker-diarization-community-1 and accept the model conditions
3. Visit https://huggingface.co/pyannote/segmentation-3.0 and accept the model conditions
4. Verify the token has read access: `python -c "from huggingface_hub import HfFolder; print(HfFolder.get_token())"`
5. Ensure `torchaudio` is compatible: `pip install torchaudio --upgrade`
6. If using CUDA, verify torchaudio CUDA version matches PyTorch: `python -c "import torchaudio; print(torchaudio.__version__)"`

### Issue: YouTube download fails

**Cause**: Network connectivity issues, video unavailable or region‑locked, or yt-dlp not installed

**Solution**:
1. Verify internet connectivity: `curl -I https://www.youtube.com`
2. Ensure yt-dlp is installed: `pip install yt-dlp` and upgrade: `pip install --upgrade yt-dlp`
3. Verify the URL is correct and the video is publicly accessible
4. Try downloading manually first: `yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"`
5. If the video is age‑restricted or private, it cannot be processed

### Issue: EasyOCR not extracting text from images

**Cause**: Poor image quality, unsupported format, or text too small/blurry

**Solution**:
1. Ensure the image is a supported format (PNG, JPG, JPEG, BMP, TIFF)
2. Use higher resolution images (at least 300 DPI for scanned documents)
3. Ensure text is clearly visible with good contrast
4. Crop the image to the text region if possible
5. Pre-process: increase contrast and convert to grayscale for better results

### Issue: STT diarization producing poor results

**Cause**: Overlapping speech, excessive background noise, too many speakers, or poor audio quality

**Solution**:
1. Use audio with minimal background noise
2. For best results, limit to 2‑4 speakers
3. Avoid audio with frequent overlapping speech
4. Pre-process audio with noise reduction if possible
5. Ensure audio sample rate is at least 16kHz
6. For YouTube URLs, download quality may vary — consider downloading manually with yt-dlp first

### Justification: VODER Has No Known Systemic Issues

VODER is a mature tool with all modes fully operational. When issues occur, they are almost always due to:

1. **Missing Python libraries**: Solved by `pip install -r requirements.txt`
2. **FFmpeg not in PATH**: Solved by FFmpeg installation
3. **Insufficient GPU VRAM**: Use modes that work on CPU or upgrade GPU
4. **Poor reference audio quality**: Use clear, single‑speaker audio samples
5. **Model download failures**: Check network or add HuggingFace token
6. **Misuse of music parameter**: Only valid in dialogue mode
7. **Missing HF_TOKEN for diarization**: Create `HF_TOKEN.txt` with valid token
8. **yt-dlp not installed**: Install via `pip install yt-dlp`
9. **Poor OCR input quality**: Use high‑resolution, high‑contrast images
10. **SE used on music**: SE is speech-only
11. **SFX missing duration**: Add `/duration:nn` to SFX lines

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

### Workflow 3: Dialogue Generation with Voice Cloning + Background Music + SFX (CLI One‑Liner)

```bash
python src/voder.py tts+vc script "James: Welcome to our podcast!" "sfx: intro jingle /duration:5 /level:60" "Sarah: Thanks for having me, James." "James: So, Sarah, tell us about your work." target "James: /voices/james_reference.wav" "Sarah: /voices/sarah_reference.wav" music "soft piano, cinematic strings" level "0:30-60:50"
```

### Workflow 4: Speech Enhancement Pipeline

```bash
# Enhance a noisy recording
python src/voder.py se "noisy_podcast.wav" result "/output/clean_podcast.wav"

# Enhance audio from video
python src/voder.py se "interview_recording.mp4" result "/output/enhanced_interview.wav"
```

### Workflow 5: Sound Effects Generation

```bash
# Generate individual sound effects
python src/voder.py sfx sound "rain on a tin roof with distant thunder" duration 15 result "/sfx/rain_thunder.wav"
python src/voder.py sfx sound "footsteps on gravel approaching then receding" duration 10 steps 50 result "/sfx/footsteps.wav"
python src/voder.py sfx sound "magical chime with reverb tail" duration 5 guide 3.0 result "/sfx/chime.wav"
```

### Workflow 6: Voice Conversion with Video Input

```bash
# Install FFmpeg if needed
command -v ffmpeg || (sudo apt update && sudo apt install ffmpeg)

# Process video input (audio auto‑extracted)
python src/voder.py sts base "presentation.mp4" target "narrator_voice.wav"

# Output saved to results/
ls results/voder_sts_*.wav
```

### Workflow 7: Music Generation with Voice Conversion

```bash
# Generate instrumental music
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill, atmospheric" duration 60 result "/music/ambient.wav"

# Generate music with cloned vocals
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment\nEverything feels right" styling "rock ballad" duration 30 target "singer_reference.wav"

# Move results
mv results/*.wav /path/to/final/output/
```

### Workflow 8: Interactive Dialogue Creation with Background Music (Semi‑Automated)

For complex scripts where you want to decide about music interactively:

```bash
python src/voder.py cli
# Select option 2 (TTS) or 3 (TTS+VC)
# Enter multiple lines of dialogue (empty line to finish)
# Include SFX lines with: sfx: description /duration:nn
# VODER will automatically prompt you for voice prompts / audio paths per character
# Then it will ask: "Add background music? (y/N):"
# Answer y and enter a description, or just press Enter to skip
```

### Workflow 9: Batch Transcription with Timestamps and Diarization

```bash
# Setup HF_TOKEN for diarization
echo "hf_your_token_here" > HF_TOKEN.txt

# Transcribe multiple meeting recordings with timestamps and speaker labels
python src/voder.py stt "meeting_2026-01.wav" "meeting_2026-02.wav" "meeting_2026-03.wav" timestamp dialogue result "/data/transcripts/"

# Results copied to /data/transcripts/
ls /data/transcripts/
```

### Workflow 10: YouTube Video Transcription

```bash
# Transcribe a YouTube video with timestamps
python src/voder.py stt "https://www.youtube.com/watch?v=dQw4w9WgXcQ" timestamp result "/output/youtube_transcript.txt"

# Transcribe with speaker diarization (interview/podcast)
echo "hf_your_token_here" > HF_TOKEN.txt
python src/voder.py stt "https://www.youtube.com/watch?v=EXAMPLE_ID" dialogue result "/output/interview.txt"
```

### Workflow 11: Image Text Extraction to Dialogue

```bash
# Extract text from a script screenshot
python src/voder.py stt "script_page.png" result "/output/extracted_script.txt"

# Extract text from a subtitle image and use as TTS input
python src/voder.py stt "subtitles.jpg" result "/output/subtitle_text.txt"

# Chain: extract from image, then use extracted text for TTS
python src/voder.py stt "notes.png" result "/tmp/notes.txt" && \
SCRIPT=$(cat /tmp/notes.txt) && \
python src/voder.py tts script "$SCRIPT" voice "professional narrator"
```

### Workflow 12: Complete Audio Production Pipeline

```bash
# 1. Enhance source audio
python src/voder.py se "raw_recording.wav" result "/clean/audio.wav"

# 2. Transcribe with diarization
python src/voder.py stt "/clean/audio.wav" timestamp dialogue result "/script/dialogue.txt"

# 3. Generate dialogue with music and SFX (using extracted script as reference)
python src/voder.py tts+vc \
  script "Host: Welcome to the show!" \
  script "sfx: applause /duration:5 /level:50" \
  script "Guest: Thanks for having me!" \
  target "Host: /voices/host.wav" \
  target "Guest: /voices/guest.wav" \
  music "upbeat podcast intro" \
  level "0:30-60:40" \
  result "/final/episode.wav"
```
