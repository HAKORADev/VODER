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

- **Unified Audio Pipeline**: Eight processing modes in a single interface
- **CLI‑First Design**: All core features accessible via command line
- **No GUI Required**: Runs entirely in headless terminals
- **Full Dialogue Support**: Multi‑speaker script generation **now available in CLI** (both interactive and one‑liner)
- **Script Directives**: Per-line control over timing, volume, and duration
- **SFX Integration**: Embed sound effects directly in dialogue scripts
- **Optional Background Music for Dialogue**: Automatically generated, duration‑fitted ambient music with configurable volume levels and optional reference audio for style guidance
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
- **Song Voice Separation (SVS)**: Separate vocals from music using BS‑RoFormer
- **Speaker Language Conversion (SLC)**: Translate speech to English while preserving speaker voice (TTS sub‑task: `tts slc`, `tts slc music` for music preservation)
- **Speaker Voice Change (SVC)**: Transcribe single-speaker audio and re-synthesize with a different voice (TTS sub-task: `tts svc "path" target "voice_ref"`)
- **Speakers Separator (SS)**: Extract individual speakers from multi‑speaker audio
- **Translation in STT**: Translate transcribed speech to English automatically
- **Overdose Quality Mode**: Enhanced transcription, dialogue source analysis, and music generation using VibeVoice ASR
- **Video I/O**: Video input with automatic audio extraction; video output with replaced audio (STS)

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

# Separate vocals from music (SVS)
python src/voder.py svs "song.mp3" voice result "/output/vocals.wav"

# Translate speech to English (SLC via TTS)
python src/voder.py tts slc "spanish_audio.wav" result "/output/english.wav"

# Re-synthesize speech with a different voice (SVC via TTS)
python src/voder.py tts svc "speech.wav" target "voice_ref.wav" result "/output/new_voice.wav"

# Separate speakers (SS)
python src/voder.py ss "meeting.wav"

# STT with translation
python src/voder.py stt "french_audio.wav" translate timestamp result "/output/translated.txt"

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
| `rotary_embedding_torch` | Rotary embeddings for BS-RoFormer SVS model |
| `beartype` | Runtime type checking for BS-RoFormer |
| `ml_collections` | Configuration collections for BS-RoFormer |

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

Dialogue is supported in **TTS** mode using multiple values per parameter. **Background music is optional** and only available in dialogue mode (not single mode).

- For **TTS** with generated voices: supply one or more `script` lines, one or more `voice` lines (in the same character order), and optionally one `music` parameter and one `level` parameter.
- For **TTS** with cloned voices: supply one or more `script` lines, one or more `target` file paths (in the same character order), and optionally one `music` parameter and one `level` parameter.

```bash
python src/voder.py tts script "James: Hello, I'm James." "Sarah: Hi James, I'm Sarah." voice "James: deep male voice, calm" "Sarah: young female voice, cheerful" music "ambient electronic, chill"

python src/voder.py tts script "James: Let's start the meeting." "Sarah: I've prepared the slides." target "James: /path/to/james.wav" "Sarah: /path/to/sarah.wav" music "soft piano, strings" level "40"
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

Interactive CLI mode (`python src/voder.py cli`) presents a menu of 8 options (1‑8). In TTS mode, it allows you to enter multiple lines of script (empty line to finish) and automatically detects single vs. dialogue mode. It then prompts you for voice prompts or audio file paths (for voice cloning) for each character. When you provide audio, video, or a URL as input instead of text, the interactive flow offers a "modify speech? (Y/N)" prompt — if accepted, it runs SVS voice isolation → Whisper transcription → editable text → voice choice (source or custom) → Qwen‑TTS synthesis. **After** all prompts are collected, you will be asked:

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
| `tts` | Text‑to‑Speech with Voice Design & Voice Cloning (via `target`), SLC sub‑task (`tts slc`), SVC sub‑task (`tts svc`), optional `overdose` for VibeVoice ASR and enhanced music | No | ✅ Yes (single & dialogue + optional music + SFX + overdose + SLC + SVC support) |
| `tts+vc` | Text‑to‑Speech + Voice Cloning — **REMOVED** (use `tts` with `target`) | No | ❌ No longer accepted |
| `sts` | Speech‑to‑Speech (Voice Conversion) with video I/O & auto vocal extraction | No | ✅ Yes (single only) |
| `ttm` | Text‑to‑Music Generation with sub‑tasks (`complete`, `lego`, `extract`, `remix`, `repaint`, `bgm`), `vc` flag, SFX overlay (`bgm`/`complete`), three‑tier ACE‑Step | No | ✅ Yes (single only) |
| `ttm+vc` | Text‑to‑Music + Voice Conversion — **REMOVED** (use `ttm vc` with `clone`) | No | ❌ No longer accepted |
| `stt` | Speech‑to‑Text Transcription with translation, overdose, video/URL input | No | ✅ Yes (single, batch, timestamps, diarization, URLs) |
| `se` | Speech Enhancement (Denoise/Dereverb) | No | ✅ Yes |
| `sfx` | Sound Effects Generation | No | ✅ Yes |
| `svs` | Song Voice Separation (BS‑RoFormer) | No | ✅ Yes |
| `ss` | Speakers Separator (Multi‑Speaker Extraction) | No | ✅ Yes |

### Text‑to‑Speech (tts)

Generate speech from text using Qwen3‑TTS VoiceDesign model.
**Supports both single and dialogue modes. Dialogue mode supports optional background music and SFX lines.**
**Voice cloning is available via the `target` parameter — supply a voice reference audio path to clone that voice. Multi-reference cloning is supported using parenthesized format: `(path1)(path2)(path3)`. Add the `first` keyword before the references (`target first "(path1)(path2)(path3)"`) to extract only the first reference's speaker from all others via TSE before compiling.**
**SLC (Speaker Language Conversion) is available as a sub‑task: `tts slc "path.wav"`, `tts slc music "path.wav"`, `tts overdose slc "path.wav"`, `tts overdose slc music "path.wav"`. Always translates to English using Whisper large-v3. Supports audio files, video files, and YouTube/URL input with automatic SVS voice isolation on source. The `music` flag preserves non-vocals by extracting and blending the instrumental track.**
**SVC (Speaker Voice Change) is available as a sub‑task: `tts svc "path.wav" target "voice_ref.wav"`. Transcribes single-speaker audio and re-synthesizes it with a different target voice. The pipeline runs SVS voice isolation → Whisper/VibeVoice transcription → Qwen‑TTS synthesis with the specified target voice. Add the `overdose` flag to use VibeVoice ASR instead of Whisper. Supports the `sts:` prefix on `target` to route through STS v2 (Seed‑VC) for voice conversion instead of Qwen‑TTS, preserving more of the original prosody.**

**In interactive CLI mode, providing audio/video/URL as input triggers a "modify speech? (Y/N)" prompt that runs the STT+TTS flow: SVS voice isolation → Whisper transcription → edit text → choose voice (source or custom) → Qwen‑TTS synthesis.**

**Single mode (voice design):**
```bash
python src/voder.py tts script "text here" voice "voice description"
```

**Single mode (voice cloning via target):**
```bash
python src/voder.py tts script "text here" target "voice_reference.wav"
```

**Single mode (multi-reference cloning):**
```bash
python src/voder.py tts script "text here" target "(voice1.wav)(voice2.wav)(voice3.wav)"
```

**Single mode (multi-reference cloning with `first` — extract only first reference's speaker from all others):**
```bash
python src/voder.py tts script "text here" target first "(voice1.wav)(voice2.wav)(voice3.wav)"
```

**Single mode with language parameter:**
```bash
python src/voder.py tts script "Hola mundo" voice "female, warm" language "Spanish"
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

# With reference audio for style guidance
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2" music "description of background music" reference "path/to/style_ref.wav"

# With video file as reference for style guidance
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2" music "description of background music" reference "path/to/style_ref.mp4"

# With YouTube URL as reference for style guidance
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt for char1" "Character2: voice prompt for char2" music "description of background music" reference "https://youtube.com/watch?v=..."
```

**Dialogue mode with music volume control:**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt" "Character2: voice prompt" music "soft piano" level "0:30-60:50"
```

**Dialogue with trained voices:**
```bash
# Use a trained voice by name (latest .tts from voices/ directory)
python src/voder.py tts script "Hello world" voice "my-character"

# Use a specific trained .tts file
python src/voder.py tts script "Hello world" voice "my-character:path/to/file.tts"

# Use a trained voice for a different character name
python src/voder.py tts script "Hello world" voice "my-char:another-name"

# Dialogue with trained voices
python src/voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: hero" voice "Sarah: heroine"

# Mix trained and described voices
python src/voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: hero" voice "Sarah: cheerful female"
```

**Newline support in scripts:**
```bash
# Use \n for line breaks in script text
python src/voder.py tts script "James: First line\nSecond line" voice "James: deep male"

# Newline in dialogue
python src/voder.py tts script "Narrator: Chapter one\nThe beginning" voice "Narrator: deep male"
```

**Dialogue with SFX lines:**
```bash
python src/voder.py tts script "James: Hello" "sfx: door bell /duration:3 /level:60" "Sarah: Hi!" voice "James: male" "Sarah: female" music "ambient"
```

**TTS with overdose mode:**
```bash
# Overdose mode: uses VibeVoice ASR for dialogue source analysis and voice clip extraction (when using audio as dialogue source)
python src/voder.py tts overdose script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"

# Overdose mode with voice cloning and background music (music uses ACE-Step XL turbo)
python src/voder.py tts overdose script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav" music "soft piano"
```

When `overdose` is used:
- Dialogue source analysis uses **VibeVoice ASR** instead of Whisper + pyannote for higher accuracy transcription and speaker identification
- Voice clip extraction from multi-speaker audio uses VibeVoice ASR segments with automatic overlap trimming (removes first 2s and last 3s from longest segment to avoid cross-speaker overlap)
- Background music generation uses **ACE-Step XL turbo** for enhanced quality
- Requires 24GB+ VRAM or 48GB+ combined system memory for VibeVoice ASR

**SLC (Speaker Language Conversion) — TTS Sub‑task:**

Translate speech from any language to English while preserving the speaker's voice. SLC is now a TTS sub‑task accessed via `tts slc`. Always translates to English using Whisper large-v3 (not turbo). Supports audio files, video files, and YouTube/URL input. SVS voice isolation runs automatically on the source before processing.

**Translate to English:**
```bash
python src/voder.py tts slc "spanish_speech.wav"
```

**Translate with music preservation (blend non-vocals back):**
```bash
python src/voder.py tts slc music "speech.wav" result "/output.wav"
```

**Translate with overdose quality:**
```bash
python src/voder.py tts overdose slc "speech.wav" result "/output.wav"
```

**Overdose + music preservation:**
```bash
python src/voder.py tts overdose slc music "speech.wav" result "/output.wav"
```

**From YouTube/Video URL:**
```bash
python src/voder.py tts slc "https://youtube.com/watch?v=..." result "/output/english.wav"
```

**SLC Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `slc` | Invoke SLC sub‑task within TTS mode (always translates to English) | Yes (for SLC) |
| `music` | Preserve non-vocals — extract instrumental and blend with voice output | No |
| `file` | Input audio/video/URL (positional after `slc` or `slc music`) | Yes |
| `result` | Output path | No |

**SLC Limitations:**
- Translation quality depends on Whisper large-v3's transcription accuracy for the source language
- Output is always English (Whisper can only translate to English)
- Voice-music sync may vary when using the `music` flag; this is inherent to the approach

> **Note:** The standalone `slc` mode has been merged into TTS. The old `slc` command is no longer accepted — use `tts slc` instead. The `translate` keyword is no longer needed; SLC always translates to English.

#### SVC (Speaker Voice Change)

Transcribe single-speaker audio and re-synthesize it with a different voice. SVC is a TTS sub‑task accessed via `tts svc`. The pipeline isolates vocals via SVS, transcribes with Whisper (or VibeVoice with `overdose`), then synthesizes new speech using Qwen‑TTS with the specified target voice. When the `sts:` prefix is used on `target`, an additional Seed‑VC v2 non‑mimic pass is applied after Qwen‑TTS synthesis for higher voice fidelity to the target reference. Multi-reference targets are supported: `target "(ref1.wav)(ref2.wav)(ref3.wav)"` concatenates multiple references for richer voice extraction.

**SVC Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `svc` | Invoke SVC sub‑task within TTS mode (positional path after `svc`) | Yes (for SVC) |
| `target` / `voice` | Voice reference for re-synthesis. Use `target "path"` for a voice clone reference or `voice "description"` for a generated voice | Yes |
| `overdose` | Use VibeVoice ASR instead of Whisper for transcription | No |

**SVC Pipeline:**
1. SVS voice isolation on source audio
2. Whisper / VibeVoice transcription of isolated vocals
3. Qwen‑TTS synthesis with the specified target voice
4. (Optional) If `target` uses `sts:` prefix → additional Seed‑VC v2 non‑mimic pass after Qwen‑TTS synthesis for enhanced voice fidelity

**Basic SVC:**
```bash
python src/voder.py tts svc "speech.wav" target "voice_ref.wav" result "/output/new_voice.wav"
```

**SVC with overdose quality:**
```bash
python src/voder.py tts overdose svc "speech.wav" target "voice_ref.wav" result "/output/new_voice.wav"
```

**SVC with sts: prefix (STS v2 voice conversion):**
```bash
python src/voder.py tts svc "speech.wav" target "sts:voice_ref.wav" result "/output/new_voice.wav"
```

**SVC with generated voice:**
```bash
python src/voder.py tts svc "speech.wav" voice "deep male voice, authoritative" result "/output/new_voice.wav"
```

**SVC with multi-reference target:**
```bash
python src/voder.py tts svc "speech.wav" target "(ref1.wav)(ref2.wav)(ref3.wav)" result "/output/new_voice.wav"
```

**SVC with STS pass and multi-reference:**
```bash
python src/voder.py tts svc "speech.wav" target "sts:(ref1.wav)(ref2.wav)" result "/output/new_voice.wav"
```

**Output naming:** SVC outputs are saved as `voder_tts_svc_*.wav` (or the path specified by `result`).

#### STS Voice Pass (`sts:` Prefix)

The `sts:` prefix on a `target` reference applies an additional Seed‑VC v2 non‑mimic voice conversion pass after the standard Qwen‑TTS cloning synthesis. This produces a more faithful voice match by running the TTS output through Seed‑VC v2 using the `sts:` reference as the target voice, preserving more of the target's voice characteristics while keeping the speech content intact. The `sts:` prefix works in single mode, dialogue mode, SVC sub‑task, and interactive modify speech.

**Where `sts:` works:**
- **Single TTS**: `target "sts:voice_ref.wav"` — clones via Qwen‑TTS then Seed‑VC v2 pass for enhanced fidelity
- **Dialogue TTS**: `target "Character:sts:voice_ref.wav"` — each character's lines are synthesized then converted via Seed‑VC v2
- **SVC**: `tts svc "source.wav" target "sts:voice_ref.wav"` — transcribe then apply Seed‑VC v2 pass after synthesis
- **Modify Speech**: When entering a custom voice reference, prefix with `sts:` to apply the Seed‑VC v2 pass after Qwen‑TTS synthesis

**Single mode with sts: prefix:**
```bash
python src/voder.py tts script "Hello world" target "sts:voice_ref.wav"
```

**Dialogue mode with sts: prefix:**
```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" target "James:sts:james_ref.wav" "Sarah:sarah_ref.wav"
```

**SVC with sts: prefix:**
```bash
python src/voder.py tts svc "speech.wav" target "sts:voice_ref.wav" result "/output/new_voice.wav"
```

**Voice Stabilization:**

VoiceDesign characters in dialogue mode automatically get their voice stabilized. After 3 script lines, the outputs are concatenated, SVS-cleaned, and fed to Qwen3-TTS Base for voice extraction. All subsequent lines use the cloned voice instead of VoiceDesign, eliminating vocal drift in long dialogues. This happens automatically with no configuration needed.

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (single mode) OR `Character: text` (dialogue mode) OR `sfx: description /duration:nn` (SFX lines) | Yes |
| `voice` | Voice prompt (single mode) OR `Character: prompt` (dialogue mode) for generated voices. Also accepts trained voice names/paths: `"character-name"`, `"character-name:path/to/file.tts"`, or `"character-name:another-name"` | Yes (unless all scripts are SFX lines or using target) |
| `target` | Path to voice reference (single) OR `Character: path` (dialogue) for cloned voices — can mix with `voice`. Supports `sts:` prefix (e.g. `"sts:voice_ref.wav"`) to route through STS v2 (Seed‑VC) for voice conversion instead of Qwen‑TTS. Multi-reference format: `(path1)(path2)(path3)` concatenates multiple references into one. Add `first` keyword before the refs (`target first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling | No (but required if no `voice` for non-SFX lines) |
| `music` | Description for automatically generated background music (dialogue only) | No |
| `level` | Music volume levels e.g. `"10:20-50 30:60-80"` (dialogue modes, default: 35%) | No |
| `reference` | Reference audio/video/URL for dialogue background music style guidance (processed via SVS music pipe to extract clean instrumental) | No |
| `overdose` | Use VibeVoice ASR for dialogue source/voice clip extraction and ACE-Step XL turbo for background music (TTS mode) | No |
| `language` | Output language for speech synthesis (e.g., `"Spanish"`, `"English"`) | No |
| `slc` | Invoke SLC sub‑task for speech translation to English (use `tts slc "path.wav"`, `tts slc music "path.wav"` for music preservation) | No |
| `svc` | Invoke SVC sub‑task for speaker voice change (use `tts svc "path.wav" target "voice_ref.wav"`) | No |

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

TTS one-line mode supports mixing generated and cloned voices in the same dialogue. Use `voice "Character: prompt"` for generated voices and `target "Character: path"` for cloned voices:

```bash
# TTS mode with mixed voices: James uses generated, Sarah uses cloned
python src/voder.py tts script "James: Hello!" "Sarah: Hi there!" voice "James: deep male voice" target "Sarah: /path/to/sarah_voice.wav"

# TTS mode with mixed voices: James uses cloned, Sarah uses generated
python src/voder.py tts script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james_voice.wav" voice "Sarah: bright female voice"

# TTS mode with multi-reference cloning per character
python src/voder.py tts script "James: Hello!" target "James:(voice1.wav)(voice2.wav)(voice3.wav)"

# TTS mode with multi-reference cloning + first keyword (extract only first ref's speaker from all others)
python src/voder.py tts script "James: Hello!" target first "James:(voice1.wav)(voice2.wav)(voice3.wav)"
```

**Important:** A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

> **Note:** The `tts+vc` mode has been fully merged into TTS. The old `tts+vc` command is no longer accepted — use `tts` with `target` instead.

### Voice Training (train voice)

Train a voice clone from reference audio and save it as a `.tts` file for later reuse. Oneline-only command.

```bash
# Train a voice from a single reference
python src/voder.py train voice:character-name "path/to/reference.wav"

# Train from multiple references (SVS-cleaned and concatenated)
python src/voder.py train voice:character-name "ref1.wav" "ref2.wav" "ref3.wav"

# Train with first keyword (extract only first ref's speaker from all others via TSE)
python src/voder.py train voice:character-name first "ref1.wav" "ref2.wav" "ref3.wav"

# Train with test sample (hardcoded 30+ second script)
python src/voder.py train voice:character-name "ref.wav" test

# Train with custom test script
python src/voder.py train voice:character-name "ref.wav" test "Custom test script text"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `voice:name` | Character name for the trained voice (used to reference it later) | Yes |
| `"path1" "path2" ...` | One or more audio file paths for reference audio | Yes |
| `first` | Extract only the first reference's speaker from all others via TSE before compiling (oneline only) | No |
| `test` | Generate a test sample using a hardcoded 30+ second script | No |
| `test "script"` | Generate a test sample using a custom test script | No |

**Output:** Trained voices are saved as `voder_tts_character-name_timestamp.tts` in the `voices/` directory.

**Using Trained Voices:** See the `voice` parameter in the TTS section above for syntax (`"character-name"`, `"character-name:path.tts"`, `"character-name:another-name"`).

### Text‑to‑Speech + Voice Clone (merged into TTS)

> **Note:** The `tts+vc` mode has been fully merged into TTS. The old command is no longer accepted. See [Text‑to‑Speech (tts)](#text-to-speech-tts) for the unified documentation.

Generate speech from text then clone it to target voice using Qwen3‑TTS Base model.  
**Supports both single and dialogue modes. Dialogue mode supports optional background music and SFX lines.**

**Single mode:**
```bash
python src/voder.py tts script "text here" target "voice_reference.wav"
```

**OCR input (image to narration with voice clone):**
```bash
python src/voder.py tts ocr "path/to/image.png" target "text: voice_reference.wav"

python src/voder.py tts ocr "subtitle_image.jpg" target "text: speaker_clone.wav"
```

**Dialogue mode (no music):**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" target "Character1: /path/to/ref1.wav" "Character2: /path/to/ref2.wav"
```

**Dialogue mode with background music:**
```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" target "Character1: /path/to/ref1.wav" "Character2: /path/to/ref2.wav" music "description of background music"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `script` | Text to synthesize (single) OR `Character: text` (dialogue) OR `sfx: description /duration:nn` (SFX lines) | Yes |
| `target` | Path to voice reference audio (single) OR `Character: path` (dialogue) for cloned voices | Yes (unless all scripts are SFX lines or using voice) |
| `voice` | Voice prompt (single) OR `Character: prompt` (dialogue) for generated voices — can mix with `target` | No (but required if no `target` for non-SFX lines) |
| `music` | Description for automatically generated background music (dialogue only) | No |
| `level` | Music volume levels (dialogue modes, default: 35%) | No |
| `reference` | Reference audio/video/URL for dialogue background music style guidance (processed via SVS music pipe to extract clean instrumental) | No |
| `overdose` | Use VibeVoice ASR for dialogue source/voice clip extraction and ACE-Step XL turbo for background music (TTS mode) | No |

**Voice Reference Requirements:**
- Format: WAV (recommended), MP3 supported
- Duration: 5‑30 seconds optimal
- Quality: Clear speech, minimal background noise
- Content: Single speaker, continuous speech

### Speech‑to‑Speech / Voice Conversion (sts)

Convert voice from base audio to target voice without changing content using Seed‑VC v2. **MSTS (Music-STS)**: For musical inputs, add the `music` keyword to use Seed‑VC v1 at 44.1kHz for better quality. **Supports video input/output**: when a video file is provided as `base`, audio is auto‑extracted, processed, and re‑muxed into an `.mp4` output. **Automatic vocal extraction**: vocals are automatically extracted from both the source and the `target` before voice conversion. Source music is separated and mixed back after conversion. **`nomusic` flag**: outputs converted voice only without mixing back the source music.

```bash
python src/voder.py sts base "source_audio.wav" target "voice_reference.wav"

python src/voder.py sts base "song.wav" target "voice_reference.wav" music

python src/voder.py sts base "song.wav" target "voice_reference.wav" nomusic
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `base` | Path to source audio or video | Yes |
| `target` | Path to target voice reference audio. Multi-reference format: `(path1)(path2)(path3)` concatenates multiple references into one. Add `first` keyword before the refs (`target first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling | Yes |
| `music` | Use Seed-VC v1 (44.1kHz) for musical inputs | No |
| `mimic` | Transfer accent and speaking style from target voice | No |
| `nomusic` | Output converted voice only (no music recombination) | No |

**Supported Input Formats:**
- Audio: WAV, MP3, FLAC, OGG
- Video: MP4, AVI, MOV, MKV (audio auto‑extracted)

**MSTS Example:**
```bash
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" music
```

**Video I/O Example:**
```bash
# From video input (auto-extracts audio, outputs .mp4 with replaced audio)
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" result "/output/output.mp4"
```

**Mimic Example (Style Transfer):**
```bash
python src/voder.py sts base "source_audio.wav" target "character_voice.wav" mimic
```

**Multi-Reference Target Example:**
```bash
python src/voder.py sts base "source.wav" target "(voice1.wav)(voice2.wav)(voice3.wav)"
```

**Multi-Reference Target with `first` (extract only first ref's speaker from all others):**
```bash
python src/voder.py sts base "source.wav" target first "(voice1.wav)(voice2.wav)(voice3.wav)"
```
**Note:** `mimic` and `music` cannot be used together. `nomusic` cannot be used with `music`.

### Text‑to‑Music (ttm)

Generate music from lyrics and style prompt using ACE‑Step (three‑tier architecture). Supports instrumental-only generation with empty lyrics. **Sub‑tasks**: `complete` (add missing tracks), `lego` (build individual instrument tracks), `extract` (extract specific tracks), `remix` (style transfer / cover with `bias` control and optional `lyrics`), `repaint` (restyle a specific time range; multi-pass for sequential edits building on each previous result). **SFX overlay**: `bgm` and `complete` sub‑tasks support `sfx:` specs to overlay generated sound effects onto the output. **Voice conversion**: add `vc` flag **before** `lyrics`/`styling`/`duration` and use `clone` for voice reference. **Overdose quality**: add `overdose` flag for enhanced output quality.

**Flags and modifiers** (can be combined):
- `vc` — enable voice conversion after music generation
- `overdose` — use enhanced quality (three‑tier ACE‑Step)
- `vc` and `remix` are mutually exclusive
- `vc` and `repaint` are mutually exclusive

**Generate music (standard):**
```bash
python src/voder.py ttm lyrics "song lyrics" styling "style description" duration 30
```

**Generate music with overdose quality:**
```bash
python src/voder.py ttm lyrics "song lyrics" styling "pop" duration 30 overdose
```

**Generate music with voice conversion (vc flag BEFORE lyrics, use `clone` for voice ref):**
```bash
python src/voder.py ttm vc lyrics "song lyrics" styling "pop" duration 30 clone "voice.wav"
```

**TTM VC with multi-reference clone:**
```bash
python src/voder.py ttm vc lyrics "song lyrics" styling "pop" duration 30 clone "(voice1.wav)(voice2.wav)(voice3.wav)"
```

**TTM VC with multi-reference clone + `first` (extract only first ref's speaker from all others):**
```bash
python src/voder.py ttm vc lyrics "song lyrics" styling "pop" duration 30 clone first "(voice1.wav)(voice2.wav)(voice3.wav)"
```

**Maximum TTM VC command (with overdose, target music ref, and result):**
```bash
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"
```

**Instrumental music (no vocals):**
```bash
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill" duration 60
```

**Complete: add missing tracks to existing audio:**
```bash
python src/voder.py ttm complete "input.wav" add "drums bass" voice result "/output/completed.wav"
```

**Complete with styling prompt:**
```bash
python src/voder.py ttm complete "input.wav" add "drums bass" styling "dramatic cinematic" reference "ref.wav" result "/output/completed.wav"
```

**Complete with noblend (generated instruments only, no mixing with original):**
```bash
python src/voder.py ttm complete noblend "input.wav" add "drums bass" result "/output/instruments_only.wav"
```

**Complete with SFX overlay:**
```bash
python src/voder.py ttm complete "input.wav" add "drums bass" sfx "thunder rumbling/10-5/60" result "/output/completed_sfx.wav"
```

**Complete with multiple SFX overlays:**
```bash
python src/voder.py ttm complete "input.wav" add "drums bass" sfx "thunder rumbling/10-5/60" sfx "rain on roof/15-0/30" result "/output/completed_multi_sfx.wav"
```

**Complete with SFX only (no add, music model not loaded):**
```bash
python src/voder.py ttm complete "input.wav" sfx "doorbell/5-12/50" result "/output/complete_sfx_only.wav"
```

**Complete with voice isolation (SVS pre-extract vocals):**
```bash
python src/voder.py ttm complete voice "song.wav" add "drums bass" result "/output/voice_completed.wav"
```

**Complete with music isolation (SVS pre-extract instruments):**
```bash
python src/voder.py ttm complete music "song.wav" add "everything" result "/output/music_completed.wav"
```

**Complete with voice isolation + usrc (blend with original source):**
```bash
python src/voder.py ttm complete voice usrc "song.wav" add "drums bass guitar" result "/output/voice_usrc_completed.wav"
```

**Complete with music isolation + usrc (blend with original source):**
```bash
python src/voder.py ttm complete music usrc "song.wav" add "everything" result "/output/music_usrc_completed.wav"
```

**Lego: build individual instrument tracks:**
```bash
python src/voder.py ttm lego "input.wav" make "drums bass" mix result "/output/lego.wav"
```

**Lego with styling prompt:**
```bash
python src/voder.py ttm lego "input.wav" make "drums bass" styling "jazz trio" mix result "/output/lego.wav"
```

**Extract: extract specific tracks from audio:**
```bash
python src/voder.py ttm extract "input.wav" stems "vocals drums" result "/output/extracted.wav"
```

**Remix: style transfer (cover) on existing audio with bias control:**
```bash
python src/voder.py ttm remix "input.wav" styling "jazz" result "/output/remix.wav"
```

**Remix with custom lyrics (optional lyrics for new vocal content):**
```bash
python src/voder.py ttm remix "input.wav" lyrics "new verse words" styling "jazz" result "/output/remix.wav"
```

**Remix with bias control (0‑100, default 40):**
```bash
python src/voder.py ttm remix "input.wav" styling "jazz" bias 70 result "/output/remix.wav"
```

**Remix with reference (voice extraction from reference for guidance):**
```bash
python src/voder.py ttm remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"
```

**Remix with reference (music extraction from reference):**
```bash
python src/voder.py ttm remix "input.wav" styling "jazz" reference music "ref.wav" result "/output/remix.wav"
```

**Remix with reference (used as-is):**
```bash
python src/voder.py ttm remix "input.wav" styling "jazz" reference "ref.wav" result "/output/remix.wav"
```

**Overdose remix with reference:**
```bash
python src/voder.py ttm overdose remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"
```

**Overdose remix with lyrics:**
```bash
python src/voder.py ttm overdose remix "input.wav" lyrics "dreamy verse lines" styling "synthwave" result "/output/remix.wav"
```

**Remix vocals only (SVS pre-extract vocals from source):**
```bash
python src/voder.py ttm remix voice "song.wav" styling "soulful R&B" result "/output/voice_remix.wav"
```

**Remix music only (SVS pre-extract instruments from source):**
```bash
python src/voder.py ttm remix music "song.wav" styling "electronic synth" result "/output/music_remix.wav"
```

**Overdose remix with voice isolation:**
```bash
python src/voder.py ttm overdose remix voice "song.wav" styling "cinematic orchestral" result "/output/voice_od_remix.wav"
```

**Multi-source remix (compose vocals from one song + instruments from another):**
```bash
python src/voder.py ttm remix voice "vocals.wav" music "instruments.wav" styling "funk" bias 60 result "/output/multi_src_remix.wav"
```

**Multi-reference remix (2 references composed into 30s composite):**
```bash
python src/voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav" result "/output/multi_ref_remix.wav"
```

**Multi-reference remix (3 references):**
```bash
python src/voder.py ttm remix "song.wav" styling "rock" reference "ref1.wav" voice "ref2.wav" music "ref3.wav" result "/output/multi_ref3_remix.wav"
```

**Repaint: restyle a specific time range of existing audio:**
```bash
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"
```

**Repaint with voice/music isolation on source:**
```bash
python src/voder.py ttm repaint voice "source.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"
python src/voder.py ttm repaint music "source.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"
```

**Repaint with bias and reference:**
```bash
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" bias 60 reference voice "ref.wav" result "/output/repainted.wav"
```

**Overdose repaint with reference:**
```bash
python src/voder.py ttm overdose repaint "source.wav" time:20-80 styling "more energetic" reference music "ref.wav" result "/output/repainted.wav"
```

**Repaint with multi-reference (up to 3):**
```bash
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" reference voice "ref1.wav" music "ref2.wav" result "/output/repainted.wav"
```

**Multi-pass repaint (multiple edits, each pass builds on the previous result):**
```bash
# Two passes: restyle 20-80s as orchestral, then restyle 10-30s of that result as jazz with bias 70
python src/voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"

# Two passes with lyrics and reference
python src/voder.py ttm repaint "song.wav" "0-30/styling(funk)/lyrics(new words\nhere)" "15-30/styling(ambient)/reference(ref.wav)"

# Overdose multi-pass with per-pass references
python src/voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"

# Multi-pass with voice isolation on source
python src/voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"
```

**BGM (Replace Background Music):**
```bash
# Standard quality (ACE-Step turbo 1.5)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30

# Overdose quality (ACE-Step XL 1.5 turbo)
python src/voder.py ttm overdose bgm "video.mp4" music "cinematic orchestral" level 50

# With reference for style guidance
python src/voder.py ttm bgm "recording.wav" music "jazz lounge" level 35 reference "style_ref.wav"

# From YouTube URL (audio only)
python src/voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient chill" level 25 result "/output/new_bgm.wav"

# From YouTube URL with video output (downloads video, replaces bgm, outputs .mp4)
python src/voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"

# BGM with SFX overlay (music + sound effects)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30 sfx "thunder rumbling/10-5/60"

# BGM with multiple SFX overlays
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30 sfx "thunder rumbling/10-5/60" sfx "rain on roof/15-20/30"

# BGM with SFX only (no music, SFX overlaid directly on clean voice)
python src/voder.py ttm bgm "podcast.wav" sfx "doorbell/5-12/50" result "/output/bgm_sfx_only.wav"
```

**BGM Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `bgm "source"` | Source audio/video/URL to replace music in | Yes |
| `music "description"` | Description for new background music | No (required unless `sfx:` specs provided) |
| `level <0-100>` | Music volume level (default: 35) | No |
| `sfx "prompt/duration-position/level"` | SFX overlay spec: prompt (required), duration 5-30s (auto-clamped), position in seconds (required, non-negative, cannot exceed source duration), level 1-100% (optional, default: 50). Multiple `sfx:` specs allowed. | No |
| `video` | Preserve video output (downloads video from URL, merges back to .mp4) | No |
| `reference "path"` | Reference audio/video/URL for style guidance (up to 3 entries with `voice`/`music` prefix; multiple refs composed into 30s composite) | No |
| `overdose` | Use ACE-Step XL 1.5 turbo for enhanced quality | No |

**BGM Rules:**
- Cannot be combined with `vc`, `remix`, `repaint`, `complete`, `lego`, or `extract`
- Source supports audio files, video files, and URLs
- `music` is optional if `sfx:` specs are provided; at least one of `music` or `sfx:` must be present
- When `sfx:` is provided without `music`, SFX are overlaid directly on the clean voice source
- When both `music` and `sfx:` are provided, SFX are overlaid after BGM mixing
- `video` flag: when source is a YouTube URL, downloads the video file (not just audio) and merges the result back into .mp4. For local video files, video output is automatic (no flag needed). If `video` is used with an audio source, outputs .wav with a warning.
- Reference supports audio files, video files, and URLs; up to 3 references composed into 30s composite
- Normal uses ACE-Step turbo 1.5; overdose uses ACE-Step XL 1.5 turbo
- SFX are generated by TangoFlux; ACE-Step is offloaded first to free GPU memory
- Output naming: `voder_ttm_bgm_{original-name}_{timestamp}.wav` (audio) or `.mp4` (video)

**Instrument tracks:** The 12 available tracks are: `woodwinds`, `brass`, `fx`, `synth`, `strings`, `percussion`, `keyboard`, `guitar`, `bass`, `drums`, `backing_vocals`, `vocals`. Shortcuts: `everything` = all 12 tracks, `instruments` = 10 non‑voice tracks, `voices` = `vocals` + `backing_vocals`.

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `"..."` for instrumental only; also optional for `remix` to guide new vocal content) | Yes (generate mode) |
| `styling` | Style prompt describing the music | Yes (generate mode), optional (`complete`/`lego`) |
| `noblend` | Output generated instruments only without blending with original audio (`complete` only) | No |
| `duration` | Duration in seconds (10‑300) | Yes (generate mode) |
| `sfx "prompt/duration-position/level"` | SFX overlay spec for `bgm`/`complete` only: prompt (required), duration 5-30s (auto-clamped), position in seconds (required, non-negative, cannot exceed source duration), level 1-100% (optional, default: 50). Minus signs stripped from duration/level, invalid values = error, level clamped with warnings. Multiple `sfx:` specs allowed. SFX generated by TangoFlux; ACE-Step offloaded first. | No |
| `vc` | Enable voice conversion (place **before** lyrics/styling/duration) | No |
| `clone` | Voice reference audio path (required when `vc` is set). Multi-reference format: `(path1)(path2)(path3)` concatenates multiple references into one. Add `first` keyword before the refs (`clone first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling | Yes (when vc is set) |
| `target` | Optional music reference (`target voice "ref.wav"` or `target music "ref.wav"`) | No |
| `bias` | Style transfer strength for `remix`/`repaint` (0‑100, default 40) | No |
| `reference` | Reference audio for `remix`/`repaint`/`bgm`/`complete`/`lego` guidance (up to 3 entries with `voice`/`music` prefix; `reference voice "path"`, `reference music "path"`, or `reference "path"` for as-is; accepts audio, video, and URLs; multiple refs composed into 30s composite) | No |
| `repaint` | Source audio for section repaint; optional `voice`/`music` prefix for SVS isolation (`repaint voice "path"` or `repaint music "path"`) | Required for repaint sub-task |
| `"start-end/styling(...)/..."` | Multi-pass repaint spec (quoted string): time range `start-end` required; optional `/styling(text)`, `/lyrics(text)`, `/reference-voice(path)`, `/reference-music(path)`, `/reference(path)` (up to 3 per pass), `/bias/nn`. Multiple pass specs = multiple sequential repaint passes, each using previous result as source. | No (multi-pass mode) |
| `video` | Preserve video output for `complete`/`bgm` (downloads video from URL, merges back to .mp4) | No |
| `overdose` | Use enhanced quality mode (three‑tier ACE‑Step) | No |
| `result` | Output file path | No |

**Sub-task Reference:**

| Sub-task | CLI Keyword | Description |
|----------|------------|-------------|
| complete | `complete "source.wav" add "drums bass" voice` | Add missing tracks to existing audio (optional `styling` prompt, optional `noblend` flag, optional `sfx:` specs for SFX overlay; `add` is optional if `sfx:` specs provided; `sfx:` cannot be used with `noblend`; if only `sfx:` with no `add`, music model not loaded; SFX overlaid after blend) |
| lego | `lego "source.wav" make "drums bass" mix` | Build individual instrument tracks (optional `styling` prompt) |
| extract | `extract "source.wav" stems "vocals drums"` | Extract specific tracks |
| remix | `remix "source.wav" styling "jazz"` | Style transfer (cover) with bias control, optional lyrics, multi-source (up to 3) and multi-reference (up to 3) |
| repaint | `repaint "source.wav" time:20-80 styling "..."` | Restyle a specific time range; optional `voice`/`music` prefix on source for SVS isolation; multi-pass mode with `"start-end/styling(...)/lyrics(...)/reference-voice(path)/reference-music(path)/reference(path)/bias/nn"` quoted pass specs (each pass uses previous result as source) |
| bgm | `bgm "source.wav" music "description" level 30` | Replace background music in audio/video (optional `video` flag for URL→video output, optional `reference`, optional `sfx:` specs for SFX overlay; `music` is optional if `sfx:` specs provided; SFX overlaid after BGM mixing or directly on clean voice) |

**Examples:**
```bash
# Standard generation
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street" styling "upbeat pop with female vocals" duration 30

# Instrumental
python src/voder.py ttm lyrics "..." styling "cinematic orchestral, dramatic" duration 90

# With overdose quality
python src/voder.py ttm lyrics "Verse:\nLyrics" styling "epic rock" duration 45
python src/voder.py ttm overdose lyrics "Verse:\nLyrics" styling "epic rock" duration 45

# With voice conversion (vc flag before lyrics, clone for voice ref)
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" duration 30 clone "singer_reference.wav"

# Overdose + VC combined
python src/voder.py ttm overdose vc lyrics "Chorus:\nWe are the champions" styling "stadium rock" duration 30 clone "singer.wav"
```

**SFX Overlay Syntax (BGM and Complete sub‑tasks only):**

The `sfx:` parameter overlays generated sound effects onto the output audio. Syntax: `sfx "prompt/duration-position/level"`

| Component | Format | Rules |
|-----------|--------|-------|
| `prompt` | Text description of the sound effect | Required. Any descriptive text. |
| `duration` | Integer, 5-30 seconds | Required. Auto-clamped to 5-30 range. Minus signs are stripped. Invalid (non-numeric) values produce an error. |
| `position` | Integer, non-negative seconds | Required. Place at N seconds into source audio. Cannot exceed source duration. Minus signs are stripped. Invalid values produce an error. |
| `level` | Integer, 1-100% | Optional, default: 50. Clamped to 1-100 with warnings. Minus signs are stripped. Invalid values produce an error. |

- Multiple `sfx:` specs are allowed; each generates and overlays a separate SFX
- SFX are generated by TangoFlux; ACE-Step is offloaded from GPU first to free memory
- In **BGM**: `music` is optional if `sfx:` specs are provided; SFX overlaid after BGM mixing, or directly on clean voice if no `music`
- In **Complete**: `add` is optional if `sfx:` specs are provided; `sfx:` cannot be used with `noblend`; if only `sfx:` with no `add`, the music model is not loaded; SFX overlaid after blend

**SFX Overlay Examples:**

| Spec | Meaning |
|------|---------|
| `sfx "thunder rumbling/10-5/60"` | Thunder sound, 10s duration, at 5s position, 60% volume |
| `sfx "doorbell/5-12"` | Doorbell sound, 5s duration, at 12s position, 50% volume (default) |
| `sfx "rain on roof/15-0/30"` | Rain sound, 15s duration, at start (0s), 30% volume |
| `sfx "alarm/30-60/80"` | Alarm sound, 30s duration (clamped from max), at 60s, 80% volume |

**Style Prompt Examples:**

| Genre | Prompt |
|-------|--------|
| Pop | "upbeat pop, catchy melody, modern production" |
| Rock | "electric guitar, driving drums, powerful vocals" |
| Ballad | "piano accompaniment, emotional, slow tempo" |
| Electronic | "synthesizer, dance beat, energetic" |
| Instrumental | "ambient electronic, atmospheric, no vocals" |

> **Note:** The `ttm+vc` mode has been fully merged into TTM. The old command is no longer accepted — use `ttm vc` with `clone` for voice conversion. The `target` parameter in `ttm vc` mode is reserved for optional music references (`target voice` / `target music`).

### Text‑to‑Music + Voice Clone (merged into TTM)

> **Note:** The `ttm+vc` mode has been fully merged into TTM. The old command is no longer accepted. For the full feature set (including `clone` syntax, `remix`, `repaint`, `lego`, `target` for music references), use `ttm vc` directly. See [Text‑to‑Music (ttm)](#text-to-music-ttm) for the unified documentation.

Generate music using ACE‑Step then apply voice conversion using Seed‑VC.

```bash
python src/voder.py ttm vc lyrics "song lyrics" styling "style" duration 30 clone "voice.wav"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `lyrics` | Song lyrics (use `"..."` for instrumental) | Yes |
| `styling` | Style prompt | Yes |
| `duration` | Duration in seconds (10-300) | Yes |
| `clone` | Voice reference audio path for voice conversion. Multi-reference format: `(path1)(path2)(path3)` concatenates multiple references into one | Yes |

**Note:** The `target` parameter in `ttm vc` mode is reserved for optional music references (`target voice "ref.wav"` or `target music "ref.wav"`). Use `clone` for voice conversion references.

**Memory optimisation:** This mode automatically releases the ACE‑Step model from GPU memory before loading Seed‑VC, reducing peak VRAM usage.

**Example:**
```bash
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "rock ballad" duration 30 clone "singer_reference.wav"
```

### Speech‑to‑Text (stt)

Transcribe audio, video, images, or YouTube URLs to text using Whisper. Supports timestamps, speaker diarization, batch processing, translation, overdose quality, and automatic result routing. **Translation**: add `translate` flag to translate transcribed speech to English. **Overdose**: add `overdose` flag for enhanced transcription quality using VibeVoice ASR. **SVS pre‑cleanup**: SVS vocal separation runs automatically before transcription to improve accuracy. **Note:** `overdose` and `translate` are mutually exclusive.

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

**Transcription with translation to English:**
```bash
python src/voder.py stt "french_audio.wav" translate timestamp result "/output/translated.txt"
```

**Transcription with overdose quality:**
```bash
python src/voder.py stt "meeting.wav" overdose dialogue
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
| `translate` | Translate transcribed speech to English | No |
| `overdose` | Use enhanced transcription quality (VibeVoice ASR) | No |
| `result` | Copy result file(s) to the specified path (file or directory) | No |

**Important:** `overdose` and `translate` are mutually exclusive and cannot be used together.

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
| `translate` | Plain text transcript translated to English |
| `translate timestamp` | Timestamped transcript translated to English |
| `overdose` | Enhanced accuracy plain text transcript (VibeVoice ASR) |
| `overdose dialogue` | Enhanced accuracy with speaker labels |

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

### Song Voice Separate (svs)

Separate vocals from music or extract instrumental tracks using the BS‑RoFormer model.

**Extract vocals:**
```bash
python src/voder.py svs "song.mp3" voice
```

**Extract instrumental:**
```bash
python src/voder.py svs "song.mp3" music
```

**Extract both (voice first, then music):**
```bash
python src/voder.py svs "song.mp3" both
```

**With result routing:**
```bash
python src/voder.py svs "song.mp3" voice result "/output/vocals.wav"
```

**From YouTube URL:**
```bash
python src/voder.py svs "https://youtube.com/watch?v=..." voice
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `file` | Input audio/video/URL | Yes |
| `voice` | Extract vocals stem | Yes* |
| `music` | Extract music/instrumental stem | Yes* |
| `both` | Extract both stems sequentially (voice first, then music) | Yes* |
| `result` | Output path | No |

*Any one of `voice`, `music`, or `both` required.

### Speaker Language Conversion (merged into TTS)

> **Note:** The standalone `slc` mode has been merged into TTS as a sub‑task. The old `slc` command is no longer accepted. Use `tts slc` instead. See [SLC (Speaker Language Conversion) — TTS Sub‑task](#slc-speaker-language-conversion--tts-sub-task) for the unified documentation.

### Speakers Separator (ss)

Extract individual speakers from multi‑speaker audio using VibeVoice ASR. Each speaker's audio is saved as a separate file.

**Separate speakers from a meeting:**
```bash
python src/voder.py ss "meeting.wav"
```

**Separate speakers with result routing:**
```bash
python src/voder.py ss "podcast.mp4" result "/output/speakers/"
```

**Overdose mode (uses VibeVoice ASR instead of Whisper+Pyannote):**
```bash
python src/voder.py ss "meeting.wav" overdose

python src/voder.py ss "meeting.wav" overdose result "/output/speakers/"
```

**Parameters:**

| Parameter | Description | Required |
|-----------|-------------|----------|
| `file` | Input audio or video file path (positional) | Yes |
| `result` | Output directory for separated speaker files | No |
| `overdose` | Use VibeVoice ASR instead of Whisper+Pyannote for enhanced separation | No |

**Note:** Requires 24GB+ VRAM or 48GB+ system memory for VibeVoice ASR model.

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

Both TTS one-line mode supports mixing `voice` and `target` parameters in the same dialogue:

```bash
# TTS mode: James generated, Sarah cloned
python src/voder.py tts script "James: Hello!" "Sarah: Hi!" voice "James: male" target "Sarah: /path/to/sarah.wav"

# TTS mode with mixed voices: James cloned, Sarah generated
python src/voder.py tts script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james.wav" voice "Sarah: female"
```

**Note:** A character cannot have both `voice` and `target` — each character must use one or the other.

### Universal: `result` Parameter

Copy the generated result file to any filesystem path. Works with **all modes** (tts, sts, ttm, stt, se, sfx, svs, ss) and sub‑tasks (`tts slc`).

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

# Copy SVS result
python src/voder.py svs "song.mp3" voice result "/output/vocals.wav"

# Copy SLC result
python src/voder.py tts slc "speech.wav" result "/output/english.wav"

# Copy SS result
python src/voder.py ss "meeting.wav" result "/output/speakers/"
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

### STT‑Only: `translate` Flag

Translate transcribed speech to English automatically.

```bash
python src/voder.py stt "french_audio.wav" translate
```

### STT‑Only: `overdose` Flag

Use enhanced transcription quality with VibeVoice ASR.

```bash
python src/voder.py stt "meeting.wav" overdose dialogue
```

**Note:** `overdose` and `translate` are mutually exclusive.

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
| **Direct Mode Access** | All eight modes available directly |
| **Music Parameter** | One‑liner background music addition (dialogue only) |
| **Level Parameter** | Configurable music volume with time-based segments |
| **Script Directives** | Per-line timing, volume, and duration control |
| **SFX in Dialogue** | Embed sound effects via `sfx:` character |
| **Cross-use Feature** | Mix generated (`voice`) and cloned (`target`) voices in same dialogue |
| **STT Transcription** | Speech-to-text with timestamps, diarization, translation, and batch processing |
| **STT Overdose Mode** | Enhanced transcription quality using VibeVoice ASR |
| **TTS Overdose Mode** | VibeVoice ASR for dialogue source analysis + enhanced music generation using ACE-Step XL turbo |
| **STT Translation** | Automatic translation of transcribed speech to English |
| **YouTube/URL Input** | Direct transcription from YouTube, Bilibili, TikTok URLs |
| **Image OCR Input** | Text extraction from images via EasyOCR |
| **Result Routing** | Copy output to arbitrary filesystem paths with `result` parameter |
| **Speech Enhancement** | Denoise, dereverberate, restore speech audio |
| **Sound Effects Generation** | Text-to-audio synthesis with configurable parameters |
| **Song Voice Separation** | Separate vocals from music using BS‑RoFormer |
| **Speaker Language Conversion** | Translate speech to English preserving voice (TTS sub‑task: `tts slc`, `tts slc music` for music preservation) |
| **Speakers Separator** | Extract individual speakers from multi‑speaker audio |
| **Video I/O** | Video input with audio extraction; video output with replaced audio (STS) |
| **TTM Sub-tasks** | Complete, lego, extract, remix, and repaint sub-tasks for music processing |
| **TTM SFX Overlay** | Overlay generated sound effects on `bgm` and `complete` sub‑task outputs via `sfx:` specs |

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
| **Text‑to‑Speech (TTS)** | One‑liner with `script`/`voice`/`target` + optional `music`/`level`/`language` | Row‑based script + voice prompt fields + optional music dialog |
| **Voice Cloning (TTS with `target`)** | One‑liner with `script`/`target` + optional `music`/`level` | Row‑based script + audio number dropdowns + optional music dialog |
| **Dialogue Mode** | ✅ Repeated parameters or interactive input + optional `music`/`level` | ✅ Visual script editor with character tracking + music prompt |
| **SFX Lines** | ✅ `sfx: description /duration:nn` in script | ✅ `sfx` character in dialogue rows |
| **Script Directives** | ✅ `/time:`, `/level:`, `/duration:` in dialogue lines | ✅ Same directive syntax in dialogue rows |
| **Background Music** | ✅ `music` parameter (one‑liner) or interactive yes/no | ✅ Modal dialog before generation |
| **STS / TTM / TTM vc** | ✅ One‑liner commands | ✅ Dedicated panels |
| **STT (Speech-to-Text)** | ✅ One‑liner with optional `timestamp`, `dialogue`, `translate`, `overdose`, `result` | ✅ Dedicated panel |
| **SE (Speech Enhancement)** | ✅ One‑liner command | ✅ Dedicated panel |
| **SFX (Sound Effects)** | ✅ One‑liner with `sound`, `duration`, `steps`, `guide` | ✅ Dedicated panel |
| **SVS (Song Voice Separate)** | ✅ One‑liner command | ✅ Dedicated panel |
| **SLC (Speaker Language Conversion)** | ✅ One‑liner command (`tts slc`) | ✅ Dedicated panel |
| **SS (Speakers Separator)** | ✅ One‑liner command | ✅ Dedicated panel |
| **Output File Generation** | ✅ Saved to `results/` | ✅ Saved to `results/` |
| **Parameter Customisation** | ✅ Duration, prompts, etc. | ✅ Duration, prompts, etc. |

**Important:** Dialogue mode, script directives, SFX lines, and optional background music are **fully supported in CLI** for TTS (both voice design and voice cloning via `target`), using either one‑liner repeated parameters or interactive multi‑line input.

---

## GPU Requirements

### All Modes Run on CPU

VODER operates entirely on CPU. No GPU is required for any mode. This makes VODER accessible to users without NVIDIA graphics hardware. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

### Memory Requirements by Mode

| Mode | RAM Required | GPU (CUDA) | VRAM | Notes |
|------|--------------|-------------|------|-------|
| TTS (no music) | 12GB | Optional | 4GB (minimum, GTX 1060) | 8GB base + 4GB (Qwen) |
| TTS (with music) | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| TTS + Overdose | 48GB | Optional | 24GB VRAM or 48GB RAM | 8GB base + ~40GB (VibeVoice ASR) + 15GB (ACE XL, if music) |
| STT | 12GB | Optional | 4GB (minimum) | 8GB base + 4GB (Whisper) |
| STT + Diarization | 15GB | Optional | 4GB (minimum) | +3GB (Pyannote) |
| STT + Overdose | 48GB | Optional | 24GB VRAM or 48GB RAM | 8GB base + ~40GB (VibeVoice ASR) |
| STS | 13GB | Optional | 14GB | 8GB base + 5GB (Seed-VC) |
| TTM | 23GB | Optional | 15GB (recommended, RTX 3080 or 16GB GPU) | 8GB base + 15GB (ACE) |
| TTM + Overdose | 48GB | Optional | 32GB VRAM or 48GB RAM | 8GB base + ~40GB (VibeVoice ASR + three‑tier ACE‑Step) |
| TTM + Complete | 48GB | Optional | 32GB VRAM or 48GB RAM | 8GB base + ~40GB (VibeVoice ASR + three‑tier ACE‑Step) |
| TTM vc | 23GB | Optional | 16GB | 8GB base + 15GB (ACE) |
| SE | 11GB | Optional | 4GB | 8GB base + 2-3GB (UniSE) |
| SFX | 12GB | Optional | 4GB | 8GB base + 3-4GB (TangoFlux) |
| SVS | 14GB | Optional | 6-7GB (additional for BS-RoFormer) | 8GB base + 2-3GB (BS-RoFormer) |
| TTS + SLC | 18GB | Optional | 4GB | 8GB base + ~3GB (SVS) + ~3GB (Whisper large-v3) + 4GB (Qwen, shared with TTS) |
| SS | 48GB | Optional | 24GB VRAM or 48GB RAM | 8GB base + ~40GB (VibeVoice ASR) |

### VRAM Guidelines

| VRAM | Performance | Suitable Modes |
|------|-------------|----------------|
| No GPU (CPU only) | Slow | All modes work on CPU |
| 4GB | Usable | TTS (no music), STT, SE, SFX, TTS SLC |
| 6GB | Minimum | TTS (no music), STT, SE, SFX, TTS SLC, SVS |
| 14GB | Mid-range | STS, all TTS modes, SE, SFX, SVS, TTS SLC |
| 15-16GB | Recommended | TTS with music, TTM, TTM vc, all standard modes |
| 24GB | High (RTX 4090) | All modes including SS, STT overdose, TTS overdose, TTM overdose |
| 32GB | Professional | All modes at full speed including TTM overdose/complete |
| 48GB | Server-grade | All modes including SS, overdose modes on GPU |
| T4 (16GB) | Server-grade | All standard modes (not consumer GPU) |

**Note:** The T4 GPU has 16GB VRAM but is a server-grade GPU, not a typical consumer card like GTX 1660 Super.

### Modes Requiring More Memory

The following modes require approximately 23GB RAM due to the ACE-Step model:

- **TTM** (Text-to-Music)
- **TTM vc** (TTM with voice conversion)
- **TTS** with background music

The following modes require approximately 48GB RAM (or 24GB+ VRAM) for VibeVoice ASR:

- **STT** with `overdose` flag
- **TTS** with `overdose` flag
- **TTM** with `overdose` flag
- **TTM** `complete` sub-task
- **SS** (Speakers Separator)

### Modes Working With Less Memory

The following modes work with approximately 11-13GB RAM:

- **TTS** (Text-to-Speech) - 12GB
- **TTS** (Text-to-Speech, including voice cloning via `target`) - 12GB
- **TTS SLC** (Speaker Language Conversion via `tts slc`) - 18GB
- **STT** (Speech-to-Text) - 12GB
- **STS** (Speech-to-Speech) - 13GB
- **SE** (Speech Enhancement) - 11GB
- **SFX** (Sound Effects) - 12GB
- **SVS** (Song Voice Separate) - 14GB

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
3. **Single Mode for STS/TTM/TTM vc**: These modes do not support multi‑speaker dialogue in CLI
4. **Music only for Dialogue**: `music` parameter is ignored in single mode

### STT Mode Limitations

1. **Diarization Requires HF_TOKEN**: Speaker diarization needs a valid HuggingFace token with accepted model conditions
2. **YouTube Download Requires Internet**: URL-based transcription needs network access and yt-dlp installed
3. **Image OCR Accuracy Varies**: Text extraction quality depends on image resolution, font clarity, and language support
4. **Speaker Diarization Accuracy Varies**: Best results with clear audio, minimal background noise, and ≤4 speakers; overlapping speech and noisy environments reduce accuracy
5. **Overdose Not Available with Translation**: The `overdose` and `translate` flags are mutually exclusive and cannot be used together

### SVS Mode Limitations

1. **Limited to Vocal/Music Separation**: SVS separates only vocals from music/instrumental tracks; other stem types (drums, bass, etc.) are not supported
2. **Source Quality Dependent**: Separation quality depends heavily on the source audio mix and clarity of the original recording

### SLC Mode Limitations (TTS Sub‑task)

1. **Translation Quality Depends on Whisper large-v3**: Translation accuracy depends on Whisper large-v3's transcription accuracy for the source language
2. **English Only**: Output is always English (Whisper can only translate to English, not between arbitrary language pairs)
3. **Music Sync**: When using the `music` flag, voice-music synchronization may vary; this is inherent to the approach

### SS Mode Limitations

1. **High Hardware Requirements**: Requires 24GB+ VRAM or 48GB+ system memory for the VibeVoice ASR model
2. **Speaker Count**: Best results with 2-6 speakers; performance degrades with higher speaker counts

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

For STS mode, ensure at least 13GB RAM is available. For TTM or TTS with music, ensure at least 23GB RAM is available.

### Issue: Out of memory errors

**Cause**: Model too large for available system memory

**Solution**:
- For TTS without music: Ensure at least 12GB RAM available
- For TTS with music, TTM, or TTM vc: Ensure at least 23GB RAM available
- For STS: Ensure at least 13GB RAM available
- For STT with diarization: Ensure at least 15GB RAM available
- For STT with overdose, TTM with overdose, TTM complete, or SS: Ensure at least 48GB RAM available (or 24GB+ VRAM)
- Reduce TTM duration (shorter audio = less memory)
- Process shorter audio segments for STS
- Use TTS modes instead of voice conversion modes

### Issue: Module not found errors

**Cause**: Python dependencies not installed

**Solution**: Run `pip install -r requirements.txt`

### Issue: FFmpeg not found errors

**Cause**: FFmpeg not installed or not in system PATH

**Solution**: Install FFmpeg separately (see FFmpeg Setup section)

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

**Cause**: Running on CPU without GPU acceleration (for STS or TTM vc modes)

**Solution**: Use NVIDIA GPU with 8GB+ VRAM for acceleration, or use TTS or TTM modes which work on CPU

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

### Issue: SVS separation quality is poor

**Cause**: Source audio has complex mix, low quality, or poorly separated stems

**Solution**:
1. Use higher quality source audio (320kbps MP3 or lossless)
2. Ensure source is properly mixed (vocals not buried in effects)
3. Pre-process with speech enhancement for better vocal extraction

### Issue: SLC translation produces incorrect output

**Cause**: Source language not well supported by Whisper large-v3, or unclear speech

**Solution**:
1. Ensure source audio is clear with minimal background noise
2. Pre-process with speech enhancement (`se` mode) before SLC
3. Use shorter audio segments for better accuracy
4. SLC always uses Whisper large-v3 (not turbo) for maximum accuracy
5. Note: SLC is now a TTS sub‑task — use `tts slc "path.wav"` instead of the old standalone `slc` command

### Issue: SS fails with out of memory

**Cause**: VibeVoice ASR requires 24GB+ VRAM or 48GB+ system memory

**Solution**:
1. Verify available memory: `free -h` (need 48GB+ RAM) or `nvidia-smi` (need 24GB+ VRAM)
2. Process shorter audio segments to reduce memory pressure
3. Use STT with `dialogue` flag as an alternative for speaker identification

### Issue: TTM overdose or complete sub-task fails

**Cause**: VibeVoice ASR + three‑tier ACE‑Step require significant memory

**Solution**:
1. Ensure 32GB+ VRAM or 48GB+ system memory available
2. Use standard TTM mode (without overdose) if resources are limited
3. Reduce duration for overdose mode to lower memory requirements

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
12. **SS/overdose insufficient memory**: Ensure 48GB+ RAM or 24GB+ VRAM
13. **STT overdose + translate conflict**: These flags are mutually exclusive (STT mode only; SLC always translates to English)

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
python src/voder.py tts script "Welcome to our weekly podcast episode." target "host_voice.wav" && \
python src/voder.py tts script "Today we'll discuss the latest in AI technology." target "guest_voice.wav" && \
python src/voder.py tts script "Let's begin with our first topic." target "host_voice.wav"

# Results are in results/ directory
ls results/
```

### Workflow 2: Dialogue Generation with Voice Design + Background Music (CLI One‑Liner)

```bash
python src/voder.py tts script "Narrator: Once upon a time, in a digital realm," "Alice: I wonder what secrets this code holds." "Bob: Let's find out together!" voice "Narrator: calm male voice, slow and measured" "Alice: bright female voice, curious" "Bob: enthusiastic male voice, friendly" music "orchestral fantasy, magical, adventurous"
```

### Workflow 3: Dialogue Generation with Voice Cloning + Background Music + SFX (CLI One‑Liner)

```bash
python src/voder.py tts script "James: Welcome to our podcast!" "sfx: intro jingle /duration:5 /level:60" "Sarah: Thanks for having me, James." "James: So, Sarah, tell us about your work." target "James: /voices/james_reference.wav" "Sarah: /voices/sarah_reference.wav" music "soft piano, cinematic strings" level "0:30-60:50"
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

# Process video input (audio auto‑extracted, output as .mp4 with replaced audio)
python src/voder.py sts base "presentation.mp4" target "narrator_voice.wav" result "/output/output.mp4"

# Output saved to results/
ls results/voder_sts_*.wav
```

### Workflow 7: Music Generation with Voice Conversion

```bash
# Generate instrumental music
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill, atmospheric" duration 60 result "/music/ambient.wav"

# Generate music with cloned vocals
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment\nEverything feels right" styling "rock ballad" duration 30 clone "singer_reference.wav"

# Move results
mv results/*.wav /path/to/final/output/
```

### Workflow 8: Interactive Dialogue Creation with Background Music (Semi‑Automated)

For complex scripts where you want to decide about music interactively:

```bash
python src/voder.py cli
# Select option 1 (TTS) — voice cloning and SLC are now part of TTS
# Enter multiple lines of dialogue (empty line to finish)
# Include SFX lines with: sfx: description /duration:nn
# Or provide audio/video/URL to trigger the "modify speech?" flow
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
python src/voder.py tts \
  script "Host: Welcome to the show!" \
  script "sfx: applause /duration:5 /level:50" \
  script "Guest: Thanks for having me!" \
  target "Host: /voices/host.wav" \
  target "Guest: /voices/guest.wav" \
  music "upbeat podcast intro" \
  level "0:30-60:40" \
  result "/final/episode.wav"
```

### Workflow 13: Song Vocal Separation and Re-synthesis

```bash
# 1. Separate vocals from a song
python src/voder.py svs "original_song.mp3" voice result "/output/vocals.wav"

# 2. Separate instrumental
python src/voder.py svs "original_song.mp3" music result "/output/instrumental.wav"

# 3. Enhance extracted vocals
python src/voder.py se "/output/vocals.wav" result "/output/clean_vocals.wav"

# 4. Convert vocals to a different voice
python src/voder.py sts base "/output/clean_vocals.wav" target "new_singer.wav" result "/output/converted_vocals.wav"
```

### Workflow 14: Speech Translation Pipeline

```bash
# Translate Spanish speech to English
python src/voder.py tts slc "spanish_podcast.wav" result "/output/english_version.wav"

# Translate from a YouTube video
python src/voder.py tts slc "https://youtube.com/watch?v=EXAMPLE" result "/output/translated.wav"

# Chain: translate then transcribe
python src/voder.py tts slc "german_speech.wav" result "/output/english.wav" && \
python src/voder.py stt "/output/english.wav" timestamp result "/output/english_transcript.txt"
```

### Workflow 15: Multi-Speaker Separation and Processing

```bash
# 1. Separate speakers from a meeting recording
python src/voder.py ss "meeting.wav" result "/output/speakers/"

# 2. Transcribe each separated speaker individually
python src/voder.py stt "/output/speaker_1.wav" timestamp result "/output/speaker_1.txt"
python src/voder.py stt "/output/speaker_2.wav" timestamp result "/output/speaker_2.txt"

# 3. Enhance each speaker's audio
python src/voder.py se "/output/speaker_1.wav" result "/output/enhanced_speaker_1.wav"
python src/voder.py se "/output/speaker_2.wav" result "/output/enhanced_speaker_2.wav"
```

### Workflow 16: Music Production with Overdose Quality

```bash
# Generate music with enhanced quality
python src/voder.py ttm lyrics "Verse 1:\nWalking through the rain\nFeeling no pain" styling "melancholic indie rock" duration 60 overdose result "/music/high_quality.wav"

# Generate with voice conversion and overdose (vc flag before lyrics, clone for voice ref)
python src/voder.py ttm overdose vc lyrics "Chorus:\nWe'll rise again" styling "epic orchestral" duration 30 clone "singer.wav" result "/music/epic_cover.wav"
```
