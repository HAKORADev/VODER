# VODER — Detailed Reference

> This document contains detailed mode descriptions, CLI examples, technical notes, and usage guides for each of VODER's eight processing modes. For a quick overview, see [README.md](README.md).

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
>
> `stt+tts` and `slc` are no longer standalone modes. STT+TTS is now integrated into TTS interactive mode as a "modify speech?" prompt. SLC is now a TTS oneline sub-task. Use `tts` for both.

---

## Table of Contents

- [1. TTS Mode](#1-tts-mode)
  - [1.1 Voice Design & Cloning](#11-voice-design--cloning)
  - [1.2 Dialogue System](#12-dialogue-system)
  - [1.3 Cross-Use Feature](#13-cross-use-feature)
  - [1.4 Background Music](#14-background-music)
  - [1.5 SLC (Speech Language Conversion)](#15-slc-speech-language-conversion)
  - [1.6 Modify Speech (STT+TTS)](#16-modify-speech-stttts)
- [2. STS Mode](#2-sts-mode)
  - [2.1 MSTS (Music-STS)](#21-msts-music-sts)
- [3. TTM Mode](#3-ttm-mode)
  - [3.1 Sub-Tasks](#31-sub-tasks)
  - [3.2 Quality Tiers](#32-quality-tiers)
  - [3.3 Voice Conversion in TTM](#33-voice-conversion-in-ttm)
  - [3.4 Instrument Tracks](#34-instrument-tracks)
- [4. STT Mode](#4-stt-mode)
  - [4.1 Features](#41-features)
  - [4.2 CLI Examples](#42-cli-examples)
- [5. SE Mode](#5-se-mode)
- [6. SFX Mode](#6-sfx-mode)
- [7. SVS Mode](#7-svs-mode)
- [8. SS Mode](#8-ss-mode)
- [Intelligent Source Analysis](#intelligent-source-analysis)
- [AI Model Integration](#ai-model-integration)
- [Usage Guide](#usage-guide)
  - [GUI Mode](#gui-mode)
  - [CLI Mode (Interactive)](#cli-mode-interactive)
  - [One-Line Commands](#one-line-commands)
- [Technical Highlights](#technical-highlights)

---

## 1. TTS Mode

Text-to-Speech with Voice Design and Cloning. TTS is VODER's most feature-rich mode, supporting single-line synthesis, multi-character dialogue, voice cloning, cross-use mixing, embedded sound effects, script directives, optional background music, speech language conversion (SLC), and an integrated modify-speech pipeline.

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

### 1.1 Voice Design & Cloning

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

### 1.2 Dialogue System

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

### 1.3 Cross-Use Feature

TTS one-line mode supports mixing generated and cloned voices in the same dialogue. Use `voice` for generated voices and `target` for cloned voices:

```bash
# James uses a generated voice, Sarah uses a cloned voice
python src/voder.py tts script "James: Hello!" "Sarah: Hi there!" voice "James: deep male voice" target "Sarah: /path/to/sarah_voice.wav"

# James uses a cloned voice, Sarah uses a generated voice
python src/voder.py tts script "James: Welcome!" "Sarah: Thanks!" target "James: /path/to/james_voice.wav" voice "Sarah: bright female voice"
```

### 1.4 Background Music

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

### 1.5 SLC (Speech Language Conversion)

SLC translates speech from one language to another while preserving the speaker's voice identity. It is now a TTS oneline sub-task, leveraging Whisper's translation capability (supporting 99 languages) and Qwen3-TTS for resynthesis.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- YouTube URLs — downloaded and processed automatically

**Features:**
- Same-language resynthesis — re-synthesize speech preserving the original voice and language
- Translation to English — translate from any of Whisper's 99 supported languages while preserving the speaker's voice, tone, and delivery style
- Optional overdose mode — runs an STS v2 non-mimic pass after TTS output for better voice preservation

**CLI Examples:**
```bash
# Same-language resynthesis (preserve voice, same language)
python src/voder.py tts slc "speech.wav"

# Translate to English preserving speaker voice
python src/voder.py tts slc translate "spanish_speech.wav"

# Overdose mode: STS v2 non-mimic pass after TTS for better voice preservation
python src/voder.py tts overdose slc translate "speech.wav"
```

### 1.6 Modify Speech (STT+TTS)

TTS interactive mode includes an integrated modify-speech pipeline. When launching TTS interactively, the first prompt asks **"modify speech? (Y/N)"** — answering yes initiates the following workflow:

1. Provide an audio file, video file, or URL
2. SVS voice isolation — extracts clean vocals from the source
3. Whisper transcription — transcribes the isolated speech to text
4. Edit the transcribed text as needed
5. Choose a voice — use the source voice or specify a custom voice reference
6. Qwen-TTS synthesis — generates the final audio from the edited text

This feature is available only in GUI and interactive CLI because it involves interactive text editing between the transcription and synthesis steps.

---

## 2. STS Mode

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

### 2.1 MSTS (Music-STS)

STS mode now supports musical inputs. When processing songs or musical audio, select "musical inputs?" to use the Seed-VC v1 model (44.1kHz) instead of the standard v2 model (22.05kHz), providing better voice conversion quality for music content.

**Additional STS Features:**
- **Video I/O** — feed a video file directly and receive a video with the converted voice audio track.
- **Automatic Vocal Extraction** — when a target reference contains mixed audio (vocals + music), STS automatically extracts clean vocals via BS-RoFormer before voice conversion.

---

## 3. TTM Mode

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

### 3.1 Sub-Tasks

TTM supports six sub-tasks for different music manipulation workflows:

| Sub-Task | Description |
|----------|-------------|
| **complete** | Full lyrics-to-music synthesis with all instrument tracks |
| **lego** | Generate music building blocks that can be recombined |
| **extract** | Extract and isolate individual elements from existing music |
| **remix** | Create a remix version of an existing track (supports `reference` for additional guidance) |
| **repaint** | Re-style or regenerate elements of an existing track (supports `reference` for additional guidance) |
| **bgm** | Replace background music in existing audio/video — strips current music, generates new bgm, mixes at configurable volume |

### 3.2 Quality Tiers

TTM uses a three-tier ACE-Step quality system:

| Tier | Model | Quality | Resource Usage |
|------|-------|---------|----------------|
| **standard** | ACE-Step (default) | Standard | Lower |
| **overdose** | ACE-Step XL-Turbo | High | Higher (32GB+ VRAM or 48GB+ RAM) |
| **complete** | ACE-Step XL-Base | Maximum | Highest (32GB+ VRAM or 48GB+ RAM) |

Use the `overdose` keyword before `lyrics` to activate overdose quality, or `complete` for maximum quality.

### 3.3 Voice Conversion in TTM

TTM supports voice conversion through the `vc` flag. When enabled, you can clone a singer's voice from a reference audio file and apply it to the generated music:

```bash
# Voice conversion with standard quality
python src/voder.py ttm vc lyrics "..." styling "pop" duration 30 clone "voice.wav"

# Voice conversion with overdose quality
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"
```

### 3.4 Instrument Tracks

TTM can output up to **12 individual instrument tracks** in addition to the mixed audio, allowing for fine-grained post-production control over each instrument in the generated music.

---

## 4. STT Mode

STT is a **standalone transcription mode** available as a one-line CLI command. It transcribes audio, video, images, or YouTube URLs into plain text with optional enhancements.

### 4.1 Features

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

### 4.2 CLI Examples

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

## 5. SE Mode

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
# Select option 5 (SE)

# One-liner mode
python src/voder.py se "audio_file.wav" result "/output/enhanced.wav"
```

---

## 6. SFX Mode

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

## 7. SVS Mode

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

## 8. SS Mode

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

- **YouTube / Bilibili / TikTok URL Support:** Paste a video URL directly as input in STT, SVS, SLC (via TTS), and dialogue modes. VODER automatically downloads the audio track and processes it — no manual downloading or conversion required.
- **Image Text Extraction (OCR):** Feed image files (PNG, JPG, etc.) as input. VODER uses EasyOCR to extract embedded text, which is then processed as dialogue script content. This works in STT, TTS, and TTS modes — enabling workflows like "photo of a script → spoken audio."
- **Automatic Voice Clip Extraction:** When processing multi-speaker audio (e.g., a podcast recording), VODER can automatically identify and extract individual speaker segments. This replaces the previous manual approach of splitting audio files.
- **Speaker Diarization:** Powered by pyannote, VODER identifies who spoke when in multi-speaker audio. Each speaker is labeled consistently, and the diarization output can be combined with transcription for fully annotated results.

> **Multi-Speaker Input — Now Supported!** Previous versions of VODER required manually separating multi-speaker audio before processing. With the new Intelligent Source Analysis system, VODER can now accept multi-speaker audio directly. The speaker diarization pipeline automatically identifies speakers, extracts their voice clips, and makes them available for voice cloning and transcription. See [Guide.md](Guide.md) for the updated workflow.

---

## AI Model Integration

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription and translation
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Synthesis (Extreme):** [FishAudio/S2-Pro](https://huggingface.co/fishaudio/s2-pro) — Fish Audio S2-Pro for higher quality cloning, 80+ language support, and emotion/tone/effect tags via `[tag]` syntax (activated with `extreme` keyword)
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
2. Select mode from dropdown (8 available modes)
3. Load input files based on mode:
   - **TTS:** Enter dialogue row‑by‑row in the script area, and fill the automatically generated voice prompts for each character. Use the `target` field for voice cloning from a reference audio file, or leave blank for voice design from a text prompt. Optionally set a `language` parameter for TTS output language. YouTube URLs are accepted as voice prompts for cloning.
     **Optional:** Before generation, a dialog will ask if you want background music; enter a description or press Skip.
     **Modify Speech:** At the start, a prompt asks "modify speech? (Y/N)" — answer yes to load audio/video/URL, transcribe, edit text, choose voice, and re-synthesize.
   - **STS:** Load base audio/video and target voice audio. Video input is accepted and video output is produced automatically. When a target contains mixed audio, vocals are extracted via BS-RoFormer.
   - **TTM:** Enter lyrics and style prompt. Supports sub-tasks (complete, lego, extract, remix, repaint) and a three-tier ACE-Step quality system (standard, overdose, complete). Use the `vc` flag for voice conversion with a clone audio reference. Outputs up to 12 instrument tracks.
   - **STT:** Load audio, video, image, or enter a URL for transcription
   - **SE:** Load audio or video file for enhancement
   - **SFX:** Enter a text description of the desired sound effect
   - **SVS:** Load audio, video, or enter a YouTube URL for vocal/music isolation
   - **SS:** Load audio or video for speaker separation
4. Click **"Generate"** (TTS/TTM) or **"Patch"** (STS) or **"Transcribe"** (STT) or **"Enhance"** (SE) or **"Separate"** (SVS/SS)
5. Listen to output and save results

### CLI Mode (Interactive)

```bash
python src/voder.py cli
```

The interactive CLI presents 8 options (1–8). When TTS is selected:

- **First prompt:** `modify speech? (Y/N):` — answer `y` or `yes` to load audio/video/URL, transcribe with Whisper (after SVS voice isolation), edit text, choose voice (source or custom), and synthesize with Qwen-TTS. Answer `n` or press Enter to proceed with normal TTS input.
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

**TTS — SLC (Speech Language Conversion):**
```bash
# Same-language resynthesis
python src/voder.py tts slc "speech.wav"

# Translate to English preserving speaker voice
python src/voder.py tts slc translate "spanish_speech.wav"

# Overdose mode for better voice preservation
python src/voder.py tts overdose slc translate "speech.wav"
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
- If the `music` parameter is supplied in single‑mode (plain text without colon), it is ignored with a warning.
- A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

---

## Technical Highlights

- **Unified Audio Pipeline:** Eight processing modes in a single interface eliminates the need for multiple tools
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
- **Speech Language Conversion (SLC):** Integrated into TTS as an oneline sub-task — translate speech to English or re-synthesize in the same language while preserving the speaker's voice, with optional overdose mode for enhanced voice fidelity
- **Modify Speech (STT+TTS):** Integrated into TTS interactive mode — transcribe, edit, and re-synthesize speech with source or custom voice selection
- **Cross-Modal Transformation:** Speech-to-speech, text-to-speech, speech-to-text, text-to-text, and speech language conversion
- **Cross-Platform Source Input:** Unified input pipeline accepts audio files, video files, images, and URLs (YouTube, Bilibili, TikTok) across multiple modes — no manual format conversion required
- **VibeVoice ASR:** Microsoft VibeVoice for overdose transcription, speaker diarization, and speaker separation with automatic fallback to Whisper + pyannote
- **Language Parameter in TTS:** Specify output language for TTS synthesis
- **Translation Capability in STT:** Translate audio to English from any of Whisper's 99 supported languages
- **Video I/O for STS Mode:** Feed video files directly into STS and receive video output with converted voice audio
- **YouTube URL Expansion:** YouTube/Bilibili/TikTok URL support expanded across STT, SVS, SLC (via TTS), and dialogue modes
- **Automatic Speaker Identification:** Multi-speaker audio is automatically segmented and labeled using pyannote speaker diarization, with individual voice clips extracted for downstream processing
- **Speaker Diarization with Word-Level Alignment:** Combines Whisper transcription with pyannote diarization to produce speaker-labeled, timestamped transcripts with per-word speaker attribution
- **MSTS (Music-STS):** STS mode supports musical inputs using Seed-VC v1 at 44.1kHz for better music voice conversion
- **Memory Optimisation:** Models are now explicitly offloaded after each operation to prevent memory accumulation in session-based workflows
- **Background Music for Dialogue:** Automatically generated, duration‑fitted, volume‑controlled ambient music with time-based level adjustments
