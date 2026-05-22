# VODER — Detailed Reference

> This document contains detailed mode descriptions, CLI examples, technical notes, and usage guides for each of VODER's ten processing modes. For a quick overview, see [README.md](README.md).

---

## Technical Setup Notes

### Dependencies

VODER requires several system and Python dependencies:

- **FFmpeg** — Required for audio processing, concatenation, resampling, and video audio extraction.
  ```bash
  # Windows: winget install FFmpeg
  # macOS:   brew install ffmpeg
  # Linux:   sudo apt install ffmpeg
  ```
- **SoX** — Required for audio manipulation (`sox`).
  ```bash
  # macOS: brew install sox
  # Linux: sudo apt install sox
  ```
- **yt-dlp** — Required for YouTube/Bilibili/TikTok URL support (`pip install yt-dlp`).
- **protobuf** — After installing requirements, upgrade to avoid compatibility issues:
  ```bash
  pip install --upgrade protobuf==5.29.6
  ```

All Python dependencies are listed in `requirements.txt`. Run `pip install -r requirements.txt` after cloning.

### Model Directories

VODER downloads and caches models automatically on first use. Models are stored centrally under `src/models/` — see [Guide.md](Guide.md) for the full directory structure. Key additions:

- **BS-RoFormer** (vocal/music separation) — downloaded on first SVS use (~1.5GB)
- **VibeVoice ASR** (advanced transcription) — downloaded on first SS/overdose STT use
- Ensure sufficient disk space is available for model files

### Mode History

> `tts+vc` and `ttm+vc` are no longer standalone modes. Voice cloning in TTS is handled via the `target` parameter, and voice conversion in TTM is handled via the `vc` flag. Use `tts` and `ttm` respectively.

---

## Table of Contents

- [1. STT+TTS Mode](#1-stttts-mode)
- [2. TTS Mode](#2-tts-mode)
  - [2.1 Voice Design & Cloning](#21-voice-design--cloning)
  - [2.2 Dialogue System](#22-dialogue-system)
  - [2.3 Cross-Use Feature](#23-cross-use-feature)
  - [2.4 Background Music](#24-background-music)
- [3. STS Mode](#3-sts-mode)
  - [3.1 MSTS (Music-STS)](#31-msts-music-sts)
- [4. TTM Mode](#4-ttm-mode)
  - [4.1 Sub-Tasks](#41-sub-tasks)
  - [4.2 Quality Tiers](#42-quality-tiers)
  - [4.3 Voice Conversion in TTM](#43-voice-conversion-in-ttm)
  - [4.4 Instrument Tracks](#44-instrument-tracks)
- [5. STT Mode](#5-stt-mode)
  - [5.1 Features](#51-features)
  - [5.2 CLI Examples](#52-cli-examples)
- [6. SE Mode](#6-se-mode)
- [7. SFX Mode](#7-sfx-mode)
- [8. SVS Mode](#8-svs-mode)
- [9. SLC Mode](#9-slc-mode)
- [10. SS Mode](#10-ss-mode)
- [Intelligent Source Analysis](#intelligent-source-analysis)
- [AI Model Integration](#ai-model-integration)
- [Usage Guide](#usage-guide)
  - [GUI Mode](#gui-mode)
  - [CLI Mode (Interactive)](#cli-mode-interactive)
  - [One-Line Commands](#one-line-commands)
- [Technical Highlights](#technical-highlights)

---

## 1. STT+TTS Mode

Speech-to-Text then Text-to-Speech — a two-step pipeline that transcribes audio and then re-synthesizes it. This mode is available only in GUI and interactive CLI because it involves interactive text editing between the transcription and synthesis steps.

**Workflow:**
1. Load base audio file
2. VODER transcribes the audio to text
3. Edit the transcribed text as needed
4. Load a target voice reference (optional, for voice cloning)
5. VODER synthesizes the edited text into speech

**GUI Steps:** Load base audio (content), then load target audio (voice). Click **"Patch"** to start.

**Note:** STT+TTS mode is not available in one-line CLI because it requires interactive text editing between the transcription and synthesis steps.

---

## 2. TTS Mode

Text-to-Speech with Voice Design and Cloning. TTS is VODER's most feature-rich mode, supporting single-line synthesis, multi-character dialogue, voice cloning, cross-use mixing, embedded sound effects, script directives, and optional background music.

**Supported Inputs:**
- Text (single line or multi-line dialogue script)
- Image files (PNG, JPG, etc.) — text extracted via OCR and processed as dialogue content
- YouTube URLs accepted as voice cloning references

**Key Parameters:**
- `script` — Text or dialogue lines to synthesize
- `voice` — Voice design prompt per character (e.g., `"James: deep male voice, authoritative"`)
- `target` — Voice cloning reference audio per character (e.g., `"James: /path/to/james.wav"`)
- `language` — Output language for TTS synthesis
- `music` — Background music style description (dialogue only)
- `level` — Background music volume control (dialogue only)
- `ocr` — Image file path to extract text from via EasyOCR

### 2.1 Voice Design & Cloning

TTS supports two approaches for voice creation, and they can be mixed in the same dialogue:

**Voice Design (via `voice` parameter):**
Describe the voice you want using natural language. VODER generates a matching voice from the prompt.
```bash
python src/voder.py tts script "Hello world" voice "female, cheerful"
python src/voder.py tts script "Hello world" voice "deep male voice, authoritative"
```

**Voice Cloning (via `target` parameter):**
Provide a reference audio file and VODER replicates the speaker's voice characteristics.
```bash
python src/voder.py tts script "Hello" target "voice.wav"
python src/voder.py tts script "Hello" target "https://youtube.com/watch?v=..."
```

**Note:** A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

### 2.2 Dialogue System

VODER features a powerful **row-based dialogue editor** designed for creating multi-speaker audio content such as podcasts, AI news broadcasts, audiobooks, and conversational content. This system enables script-based generation where multiple characters speak with distinct voices in a cohesive narrative flow.

**GUI Dialogue Input:**
- Each line is a separate row with **Character** and **Dialogue** fields.
- New rows are added automatically when you fill the last row.
- First row has no delete button; subsequent rows can be deleted individually.
- Voice prompts or audio assignments appear dynamically for every character found in the script.

**Script Directives (Per-Line):**

VODER now supports powerful directives that can be appended to any dialogue line for fine-grained control:

| Directive | Format | Description |
|-----------|--------|-------------|
| `/time:nn` | `/time:5` | Position this line at 5 seconds from the start |
| `/time:nn-nn` | `/time:10-3` | Position at 10s, cut 3s from end |
| `/time:nn+nn` | `/time:5+2` | Position at 5s, cut 2s from start |
| `/time:nn-nn+nn` | `/time:10-3+2` | Position at 10s, cut 3s from end, cut 2s from start |
| `/level:0-100` | `/level:75` | Set volume level for this line (default: 100) |
| `/duration:1-30` | `/duration:10` | Duration for SFX lines (required for `sfx:` character) |

**SFX Lines in Dialogue:**

You can embed sound effects directly in dialogue scripts using the special `sfx:` character:

```plaintext
James: Welcome to our podcast!
sfx: door creaking open /duration:3
Sarah: Hello everyone, glad to be here.
sfx: gentle background music /duration:10 /level:30
James: Let's dive into today's topic.
```

**SFX Line Requirements:**
- Character field must be `sfx` (case-insensitive)
- `/duration:nn` directive is **required** (1-30 seconds)
- Optional `/level:0-100` to control volume

### 2.3 Cross-Use Feature

TTS one-line mode supports mixing generated and cloned voices in the same dialogue. Use `voice` for generated voices and `target` for cloned voices:

```bash
# James uses a generated voice, Sarah uses a cloned voice
python src/voder.py tts script "James: Hello!" "Sarah: Hi there!" voice "James: deep male voice" target "Sarah: /path/to/sarah_voice.wav"

# James uses a cloned voice, Sarah uses a generated voice
python src/voder.py tts script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james_voice.wav" voice "Sarah: bright female voice"
```

### 2.4 Background Music

When generating dialogue (TTS mode), VODER can automatically **add ambient background music** that matches the length of the spoken audio.

**GUI:** A clean dialog appears before processing, asking: *"Enter music description (or press Skip):"*

**CLI:** Use the `music` parameter with a style description.

If a description is provided (e.g., `"soft piano, cinematic strings"`), VODER:
- Generates music via ACE-Step using the description as style prompt and `"..."` as empty lyrics.
- Automatically fits the music duration to the exact length of the dialogue.
- Mixes the music at **35% volume** relative to the dialogue (configurable via `level` parameter).
- Cleans up temporary files and saves the final result with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`).

**Optional `reference` parameter:** When `reference "path"` is provided alongside `music`, the reference audio is first processed through the SVS music pipe (BS-RoFormer) to extract clean instrumental music, which is then passed to ACE-Step as stylistic guidance. This ensures the generated background music matches the style of a specific existing track.

```bash
# With reference for style guidance
python src/voder.py tts script "James: Hello" voice "James: male" music "soft piano" reference "path/to/ref.wav"
```

If the user skips, processing proceeds normally without music.

**Music Volume Level Control (`level` parameter):**

```bash
# Constant volume at 50%
python src/voder.py tts script "James: Hello" voice "James: male" music "soft piano" level "50"

# Time-based segments (from 0s: 20%, at 30s: fade to 50%)
python src/voder.py tts script "James: Hello" voice "James: male" music "soft piano" level "0:20-30:50"

# With fade transitions
python src/voder.py tts script "James: Hello" voice "James: male" music "cinematic" level "0:20-30:50+60"
```

**Level Format:**
- `"volume"` — Constant volume percentage (e.g., `"35"` for 35%)
- `"start:vol-end:vol"` — Volume at start time, different volume at end time
- `"start:from-to+fade"` — Fade from volume to another over specified duration

**Example Script with music and SFX:**
```plaintext
James: Welcome to our podcast! Today we'll explore AI advances.
Sarah: Thanks James! I'm excited to discuss my latest research.
sfx: keyboard typing /duration:5 /level:40
James: Let's dive in. First, tell us about neural networks.
```

This feature is available in both **GUI** and **CLI** modes (interactive and one‑line). It is **only triggered for dialogue scripts** (i.e., more than one line, or a single line containing a colon).

---

## 3. STS Mode

Speech-to-Speech (Voice Conversion). STS transforms the voice in an audio or video file to match a target speaker's voice using Seed-VC.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.) — audio is extracted, converted, then re-attached

**Key Parameters:**
- `base` — Source audio/video file
- `target` — Reference voice audio
- `mimic` — Enable mimic mode for closer voice matching
- `musical` — Use Seed-VC v1 model (44.1kHz) for musical inputs

**Quick Examples:**
```bash
python src/voder.py sts base "input.wav" target "voice.wav"
python src/voder.py sts base "source.wav" target "reference.wav" mimic
```

### 3.1 MSTS (Music-STS)

STS mode now supports musical inputs. When processing songs or musical audio, select "musical inputs?" to use the Seed-VC v1 model (44.1kHz) instead of the standard v2 model (22.05kHz), providing better voice conversion quality for music content.

**Additional STS Features:**
- **Video I/O** — feed a video file directly and receive a video with the converted voice audio track.
- **Automatic Vocal Extraction** — when a target reference contains mixed audio (vocals + music), STS automatically extracts clean vocals via BS-RoFormer before voice conversion.

---

## 4. TTM Mode

Text-to-Music Generation & Manipulation. TTM synthesizes music from lyrics and style descriptions using ACE-Step, with support for voice conversion, sub-tasks, and a three-tier quality system.

**Key Parameters:**
- `lyrics` — Song lyrics text
- `styling` — Style/mood description (e.g., "upbeat pop", "cinematic orchestral")
- `duration` — Output duration in seconds
- `target` — Voice cloning reference audio (for `vc` mode)
- `clone` — Clone reference for voice conversion
- `vc` — Enable voice conversion flag
- `result` — Output file path

**Quick Examples:**
```bash
# Basic music generation
python src/voder.py ttm lyrics "Verse 1:\nWalking down the empty street\nFeeling the rhythm in my feet" styling "upbeat pop" duration 30

# Music with voice conversion
python src/voder.py ttm vc lyrics "..." styling "pop" duration 30 clone "voice.wav"

# TTM with overdose quality
python src/voder.py ttm overdose lyrics "..." styling "pop" duration 30

# TTM with overdose + voice conversion
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"

# Remix with reference (voice extraction for guidance)
python src/voder.py ttm remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"

# Remix with reference (music extraction)
python src/voder.py ttm remix "input.wav" styling "jazz" reference music "ref.wav" result "/output/remix.wav"

# Remix with reference (used as-is)
python src/voder.py ttm remix "input.wav" styling "jazz" reference "ref.wav" result "/output/remix.wav"

# Overdose remix with reference
python src/voder.py ttm overdose remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"

# Repaint with reference
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" reference voice "ref.wav" result "/output/repainted.wav"

# Overdose repaint with reference
python src/voder.py ttm overdose repaint "source.wav" time:20-80 styling "more energetic" reference music "ref.wav" result "/output/repainted.wav"

# BGM: Replace background music in existing audio/video
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30
python src/voder.py ttm overdose bgm "video.mp4" music "cinematic orchestral" level 50
python src/voder.py ttm bgm "recording.wav" music "jazz lounge" level 35 reference "style_ref.wav"
```

### 4.1 Sub-Tasks

TTM supports six sub-tasks for different music manipulation workflows:

| Sub-Task | Description |
|----------|-------------|
| **complete** | Full lyrics-to-music synthesis with all instrument tracks |
| **lego** | Generate music building blocks that can be recombined |
| **extract** | Extract and isolate individual elements from existing music |
| **remix** | Create a remix version of an existing track (supports `reference` for additional guidance) |
| **repaint** | Re-style or regenerate elements of an existing track (supports `reference` for additional guidance) |
| **bgm** | Replace background music in existing audio/video — strips current music, generates new bgm, mixes at configurable volume |

### 4.2 Quality Tiers

TTM uses a three-tier ACE-Step quality system:

| Tier | Model | Quality | Resource Usage |
|------|-------|---------|----------------|
| **standard** | ACE-Step (default) | Standard | Lower |
| **overdose** | ACE-Step XL-Turbo | High | Higher (32GB+ VRAM or 48GB+ RAM) |
| **complete** | ACE-Step XL-Base | Maximum | Highest (32GB+ VRAM or 48GB+ RAM) |

Use the `overdose` keyword before `lyrics` to activate overdose quality, or `complete` for maximum quality.

### 4.3 Voice Conversion in TTM

TTM supports voice conversion through the `vc` flag. When enabled, you can clone a singer's voice from a reference audio file and apply it to the generated music:

```bash
# Voice conversion with standard quality
python src/voder.py ttm vc lyrics "..." styling "pop" duration 30 clone "voice.wav"

# Voice conversion with overdose quality
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"
```

### 4.4 Instrument Tracks

TTM can output up to **12 individual instrument tracks** in addition to the mixed audio, allowing for fine-grained post-production control over each instrument in the generated music.

---

## 5. STT Mode

STT is a **standalone transcription mode** available as a one-line CLI command. It transcribes audio, video, images, or YouTube URLs into plain text with optional enhancements.

### 5.1 Features

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.)
- Image files containing text (PNG, JPG, etc.) — text is extracted via OCR before transcription
- YouTube / Bilibili / TikTok URLs — downloaded and processed automatically

**Capabilities:**
- Clean text transcription output
- **Translation** — translate audio to English from any of Whisper's 99 supported languages
- **Overdose mode** — use Microsoft VibeVoice ASR instead of Whisper for enhanced transcription accuracy and speaker diarization
- **Pre-cleanup via SVS** — optionally isolate vocals from mixed audio before transcription for cleaner results
- Optional **timestamps** for word-level or segment-level timing
- Optional **dialogue mode** that detects and formats multi-speaker conversations
- Optional **speaker diarization** that identifies individual speakers by name/label
- **Batch processing** — pass multiple files/URLs in a single command to process them all at once
- Results saved to a specified output file or printed to the terminal

### 5.2 CLI Examples

```bash
# Basic transcription
python src/voder.py stt "audio.wav"

# Translate audio to English
python src/voder.py stt "audio.wav" translate

# Transcribe a YouTube video directly
python src/voder.py stt "https://youtube.com/watch?v=..."

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With dialogue formatting
python src/voder.py stt "audio.wav" dialogue

# With overdose mode (VibeVoice ASR for enhanced accuracy)
python src/voder.py stt "audio.wav" overdose

# Batch processing — multiple files in one command
python src/voder.py stt "audio1.wav" "audio2.wav"

# Save output to a specific file
python src/voder.py stt "audio.wav" result "/path/to/output.txt"
```

---

## 6. SE Mode

SE (Speech Enhancement) is a standalone mode for improving audio quality by removing noise, reducing reverberation, and restoring speech clarity.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.) — audio is extracted automatically

**Features:**
- Denoising — removes background noise and artifacts
- Dereverberation — reduces room echo and reverb effects
- Speech restoration — enhances clarity and intelligibility
- Outputs at 16kHz sample rate (optimized for speech)
- **Not designed for musical enhancement** — use for speech content only

**Quick Examples:**
```bash
# Basic speech enhancement
python src/voder.py se "noisy_audio.wav"

# Enhance audio from video
python src/voder.py se "recording.mp4"

# Save to specific location
python src/voder.py se "audio.wav" result "/path/to/enhanced.wav"
```

**CLI Usage:**
```bash
# Interactive mode
python src/voder.py cli
# Select option 7 (SE)

# One-liner mode
python src/voder.py se "audio_file.wav" result "/output/enhanced.wav"
```

---

## 7. SFX Mode

SFX (Sound Effects) is a standalone mode for generating custom sound effects from text descriptions.

**Features:**
- Text-to-audio generation for any sound effect
- Configurable duration (1-30 seconds)
- Adjustable inference steps (1-100, default 30)
- Adjustable guidance scale (1.0-10.0, default 4.5)
- 44.1kHz output quality

**Quick Examples:**
```bash
# Generate a simple sound effect (default 10 seconds)
python src/voder.py sfx sound "thunder rumbling in the distance"

# Specify duration
python src/voder.py sfx sound "rain on a tin roof" duration 15

# Adjust quality parameters
python src/voder.py sfx sound "explosion with debris" duration 5 steps 50 guide 3.5

# Save to specific location
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"
```

**Parameters:**
| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `sound` | Text description of the sound effect | Any text | Required |
| `duration` | Duration in seconds | 1-30 | Required |
| `steps` | Inference steps (quality vs speed) | 1-100 | 30 |
| `guide` | Guidance scale (adherence to prompt) | 1.0-10.0 | 4.5 |
| `result` | Output file path | Any path | Optional |

**Sound Prompt Tips:**
- Be descriptive but concise
- Include environmental context (e.g., "in a forest", "in a small room")
- Specify intensity (e.g., "distant", "loud", "faint")
- Combine multiple elements (e.g., "thunder with heavy rain")

---

## 8. SVS Mode

SVS isolates vocals from music or extracts instrumental tracks from songs using BS-RoFormer Resurrection.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.) — audio is extracted automatically
- YouTube URLs — downloaded and processed automatically

**Features:**
- Vocal isolation — extracts clean vocals from mixed audio
- Music extraction — extracts instrumental tracks
- Used internally by STS for automatic vocal extraction from target references
- Used internally by STT for pre-cleanup vocal isolation
- Used internally by TTS for vocal extraction from voice cloning targets

**Quick Examples:**
```bash
# Extract vocals from a song
python src/voder.py svs "song.mp3" voice

# Extract instrumental from a song
python src/voder.py svs "song.mp3" music

# Extract both stems (voice first, then music)
python src/voder.py svs "song.mp3" both

# Process a YouTube URL
python src/voder.py svs "https://youtube.com/watch?v=..." voice

# Save to specific location
python src/voder.py svs "audio_file.wav" voice result "/output/vocals.wav"
```

**CLI Usage:**
```bash
# Interactive mode
python src/voder.py cli
# Select SVS from menu

# One-liner mode
python src/voder.py svs "audio_file.wav" voice result "/output/vocals.wav"
```

---

## 9. SLC Mode

SLC translates speech from one language to another while preserving the speaker's voice identity. It leverages Whisper's translation capability (supporting 99 languages) and Qwen3-TTS for resynthesis.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- YouTube URLs — downloaded and processed automatically

**Features:**
- Translates to English from any of Whisper's 99 supported languages
- Preserves original speaker's voice, tone, and delivery style
- Without target parameter: translates to English with same original voice
- With target reference: can change speaker voice while translating
- Preserving original language (if TTS-supported) with different target: voice change that can match or surpass STS quality

**Quick Examples:**
```bash
# Translate to English preserving speaker voice
python src/voder.py slc "spanish_speech.wav" result "/output/english.wav"

# Translate with different voice reference
python src/voder.py slc "speech.wav" target "voice_ref.wav" result "/output.wav"
```

---

## 10. SS Mode

SS extracts individual speaker audio from multi-speaker recordings using VibeVoice ASR for speaker identification and segmentation.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.)

**Features:**
- Automatic speaker identification and separation
- Produces separate audio files for each detected speaker
- Provides speaker-labeled transcript with timestamps
- Requires 24GB+ VRAM or 48GB+ system memory
- Falls back to Whisper + pyannote if VibeVoice ASR cannot load

**Quick Examples:**
```bash
# Separate speakers from a recording
python src/voder.py ss "meeting.wav"
```

---

## Intelligent Source Analysis

VODER supports **cross-platform source input** — a unified input pipeline that accepts audio, video, images, and URLs across multiple processing modes. This enables powerful new workflows:

- **YouTube / Bilibili / TikTok URL Support:** Paste a video URL directly as input in STT, STT+TTS, SVS, SLC, and dialogue modes. VODER automatically downloads the audio track and processes it — no manual downloading or conversion required.
- **Image Text Extraction (OCR):** Feed image files (PNG, JPG, etc.) as input. VODER uses EasyOCR to extract embedded text, which is then processed as dialogue script content. This works in STT, TTS, and TTS modes — enabling workflows like "photo of a script → spoken audio."
- **Automatic Voice Clip Extraction:** When processing multi-speaker audio (e.g., a podcast recording), VODER can automatically identify and extract individual speaker segments. This replaces the previous manual approach of splitting audio files.
- **Speaker Diarization:** Powered by pyannote, VODER identifies who spoke when in multi-speaker audio. Each speaker is labeled consistently, and the diarization output can be combined with transcription for fully annotated results.

> **Multi-Speaker Input — Now Supported!** Previous versions of VODER required manually separating multi-speaker audio before processing. With the new Intelligent Source Analysis system, VODER can now accept multi-speaker audio directly. The speaker diarization pipeline automatically identifies speakers, extracts their voice clips, and makes them available for voice cloning and transcription. See [Guide.md](Guide.md) for the updated workflow.

---

## AI Model Integration

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription and translation
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Conversion:** [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — Seed-VC for speech-to-speech transformation
- **Music Generation:** [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) — ACE-Step for lyrics-to-music synthesis
- **Sound Effects:** [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) — TangoFlux for text-to-audio generation
- **Speech Enhancement:** [alibaba/unified-audio](https://github.com/alibaba/unified-audio) — UniSE for denoising, dereverberation, and speech restoration
- **Voice Separation:** [BS-RoFormer Resurrection](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) — BS-RoFormer for vocal/music isolation
- **Advanced ASR:** [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) — VibeVoice ASR for speaker diarization, transcription, and overdose mode
- **Speaker Diarization:** [pyannote/speaker-diarization-community-1](https://github.com/pyannote/pyannote-audio) — pyannote for identifying and labeling individual speakers in multi-speaker audio
- **Image Text Extraction:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) — EasyOCR for extracting text from images, enabling image-to-speech workflows

---

## Usage Guide

### GUI Mode

1. Launch: `python src/voder.py`
2. Select mode from dropdown (10 available modes)
3. Load input files based on mode:
   - **STT+TTS:** Load base audio (content), then load target audio (voice)
   - **STT:** Load audio, video, image, or enter a URL for transcription
   - **TTS:** Enter dialogue row‑by‑row in the script area, and fill the automatically generated voice prompts for each character. Use the `target` field for voice cloning from a reference audio file, or leave blank for voice design from a text prompt. Optionally set a `language` parameter for TTS output language. YouTube URLs are accepted as voice prompts for cloning.
     **Optional:** Before generation, a dialog will ask if you want background music; enter a description or press Skip.
   - **STS:** Load base audio/video and target voice audio. Video input is accepted and video output is produced automatically. When a target contains mixed audio, vocals are extracted via BS-RoFormer.
   - **TTM:** Enter lyrics and style prompt. Supports sub-tasks (complete, lego, extract, remix, repaint) and a three-tier ACE-Step quality system (standard, overdose, complete). Use the `vc` flag for voice conversion with a clone audio reference. Outputs up to 12 instrument tracks.
   - **SE:** Load audio or video file for enhancement
   - **SFX:** Enter a text description of the desired sound effect
   - **SVS:** Load audio, video, or enter a YouTube URL for vocal/music isolation
   - **SLC:** Load audio or enter a YouTube URL for language conversion
   - **SS:** Load audio or video for speaker separation
4. Click **"Generate"** (TTS/TTM) or **"Patch"** (STT+TTS/STS) or **"Transcribe"** (STT) or **"Enhance"** (SE) or **"Separate"** (SVS/SS) or **"Convert"** (SLC)
5. Listen to output and save results

### CLI Mode (Interactive)

```bash
python src/voder.py cli
```

The interactive CLI now supports full dialogue creation:
- Enter multiple lines (empty line to finish).
- Lines without a colon → **single mode** (one text, one voice prompt/audio).
- Lines with colon (`Character: text`) → **dialogue mode**.
- Use `sfx: description /duration:nn` for embedded sound effects.
- VODER will ask for a voice prompt or audio file path for each character, in order. Use `target` to clone from a reference audio file.
- **After** collecting all voice prompts/assignments, you will be asked:
  `Add background music? (y/N):`
  If you answer `y` or `yes`, you can enter a music description and optionally a level specification.
  Leaving the description blank or entering empty skips the music.

### One-Line Commands

One‑line commands support **dialogue mode** through multiple values per parameter, as well as the optional **`music`** and **`level`** parameters for background music.

**TTS — Single mode:**
```bash
python src/voder.py tts script "Hello world" voice "female, cheerful"
python src/voder.py tts script "Hello" target "voice.wav"
python src/voder.py tts ocr "path/to/image.png" voice "text: female voice"
python src/voder.py tts ocr "path/to/image.png" target "text: voice.wav"
```

**TTS — Dialogue mode:**
```bash
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic"
python src/voder.py tts script "James: Welcome to the show!" "Sarah: Glad to be here." voice "James: deep male voice, authoritative" "Sarah: bright female voice, energetic" music "soft piano, cinematic"
python src/voder.py tts script "James: Hello" "sfx: door bell /duration:3" voice "James: deep male" music "ambient" level "0:30-60:50"
```

**TTS — Dialogue with voice cloning (target parameter):**
```bash
python src/voder.py tts script "James: Let's start with AI." "Sarah: I've been working on this for years." target "James: /path/to/james_voice.wav" "Sarah: /path/to/sarah_voice.wav"
python src/voder.py tts script "James: Let's start with AI." "Sarah: I've been working on this for years." target "James: /path/to/james_voice.wav" "Sarah: /path/to/sarah_voice.wav" music "ambient electronic, chill" level "40"
```

**STS mode:**
```bash
python src/voder.py sts base "input.wav" target "voice.wav"
python src/voder.py sts base "source.wav" target "reference.wav" mimic
```

**TTM mode:**
```bash
python src/voder.py ttm lyrics "Verse 1:\nWalking down the empty street\nFeeling the rhythm in my feet" styling "upbeat pop" duration 30
python src/voder.py ttm vc lyrics "..." styling "pop" duration 30 clone "voice.wav"
python src/voder.py ttm overdose lyrics "..." styling "pop" duration 30
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"
```

**STT mode:**
```bash
# Basic transcription
python src/voder.py stt "audio.wav"

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With dialogue formatting
python src/voder.py stt "audio.wav" dialogue

# Translate audio to English
python src/voder.py stt "audio.wav" translate

# With overdose mode (VibeVoice ASR)
python src/voder.py stt "audio.wav" overdose

# Batch processing — multiple files in one command
python src/voder.py stt "audio1.wav" "audio2.wav"

# Transcribe a YouTube video directly
python src/voder.py stt "https://youtube.com/watch?v=..."

# Save output to a specific file
python src/voder.py stt "audio.wav" result "/path/to/output.txt"
```

**SVS mode:**
```bash
python src/voder.py svs "song.mp3" voice
python src/voder.py svs "song.mp3" music
python src/voder.py svs "song.mp3" both
python src/voder.py svs "https://youtube.com/watch?v=..." voice
python src/voder.py svs "audio_file.wav" voice result "/output/vocals.wav"
```

**SLC mode:**
```bash
python src/voder.py slc "spanish_speech.wav" result "/output/english.wav"
python src/voder.py slc "speech.wav" target "voice_ref.wav" result "/output.wav"
```

**SS mode:**
```bash
python src/voder.py ss "meeting.wav"
```

**SE mode:**
```bash
python src/voder.py se "noisy_audio.wav"
python src/voder.py se "recording.mp4"
python src/voder.py se "audio.wav" result "/path/to/enhanced.wav"
```

**SFX mode:**
```bash
python src/voder.py sfx sound "rain on a tin roof" duration 10
python src/voder.py sfx sound "thunder rumbling" duration 5 steps 50 guide 3.5
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"
```

**Notes:**
- STT+TTS mode is not available in one-line CLI because it requires interactive text editing.
- If the `music` parameter is supplied in single‑mode (plain text without colon), it is ignored with a warning.
- A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

---

## Technical Highlights

- **Unified Audio Pipeline:** Ten processing modes in a single interface eliminates the need for multiple tools
- **Centralized Model Management:** A unified model management system handles loading, caching, and offloading of all AI models — ensuring efficient resource usage and preventing memory accumulation across multi-step workflows
- **Intelligent Dialogue Editor:** Row‑based script input with automatic character tracking and per‑voice assignment
- **Script Directives:** Per-line control over timing, volume, and duration for precise audio production
- **SFX Integration:** Embed sound effects directly in dialogue scripts using the special `sfx:` character
- **State-of-the-Art Models:** Production-quality models from leading AI research organizations
- **Voice Cloning:** Extract and replicate voice characteristics from reference audio samples via the `target` parameter in TTS mode
- **Music Generation:** Lyrics-to-music synthesis with style control, voice conversion (`vc` flag), sub-tasks (complete, lego, extract, remix, repaint, bgm), and a three-tier ACE-Step quality system (standard, overdose, complete) with up to 12 instrument tracks
- **Sound Effects Generation:** Text-to-audio synthesis for custom sound design
- **Speech Enhancement:** Denoise, dereverberate, and restore speech audio
- **Vocal/Music Separation:** BS-RoFormer integration for automatic vocal extraction — used internally by STS (target cleanup), STT (pre-cleanup isolation), TTS (voice cloning target cleanup), and available as a standalone SVS mode
- **Cross-Modal Transformation:** Speech-to-speech, text-to-speech, speech-to-text, text-to-text, and speaker language conversion
- **Cross-Platform Source Input:** Unified input pipeline accepts audio files, video files, images, and URLs (YouTube, Bilibili, TikTok) across multiple modes — no manual format conversion required
- **VibeVoice ASR:** Microsoft VibeVoice for overdose transcription, speaker diarization, and speaker separation with automatic fallback to Whisper + pyannote
- **Language Parameter in TTS:** Specify output language for TTS synthesis
- **Translation Capability in STT:** Translate audio to English from any of Whisper's 99 supported languages
- **Video I/O for STS Mode:** Feed video files directly into STS and receive video output with converted voice audio
- **YouTube URL Expansion:** YouTube/Bilibili/TikTok URL support expanded across STT, STT+TTS, SVS, and SLC modes
- **Automatic Speaker Identification:** Multi-speaker audio is automatically segmented and labeled using pyannote speaker diarization, with individual voice clips extracted for downstream processing
- **Speaker Diarization with Word-Level Alignment:** Combines Whisper transcription with pyannote diarization to produce speaker-labeled, timestamped transcripts with per-word speaker attribution
- **MSTS (Music-STS):** STS mode supports musical inputs using Seed-VC v1 at 44.1kHz for better music voice conversion
- **Memory Optimisation:** Models are now explicitly offloaded after each operation to prevent memory accumulation in session-based workflows
- **Background Music for Dialogue:** Automatically generated, duration‑fitted, volume‑controlled ambient music with time-based level adjustments
