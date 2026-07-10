# VODER — Detailed Reference

> This document contains detailed mode descriptions, CLI examples, technical notes, and usage guides for each of VODER's eight processing modes, followed by reference for VODER's task-layer features (voice training, side-quests, and chains). For a quick overview, see [README.md](../README.md).

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
- **yt-dlp** — Required for URL support (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter, Reddit) (`pip install yt-dlp`).
- **gallery-dl** — Required for image downloads from Reddit, Instagram, X/Twitter, and other image platforms (`pip install gallery-dl`).
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

### VADAR Model Setup

VADAR is the natural-language AI agent (`python voder.py vadar "<request>"` or interactive CLI option 10). Unlike the other models, VADAR's model is **not downloaded automatically** on first use — you download it with a single command (see below) before running VADAR for the first time.

**Model:** `OpenYourMind/gemma-4-12B-it-abliterated-uncensored` — Gemma 4 12B, abliterated uncensored variant. A multimodal model (text + image + audio + video) with the architecture class `Gemma4UnifiedForConditionalGeneration`.

**Where it goes:** `src/models/checkpoints/vadar/` (the `VADAR_MODEL_DIR` constant in `voder.py`).

**How to download:**

The model loading / downloading / caching logic lives in `src/voder.py` (not in the VADAR package). The automatic download subcommand uses `huggingface_hub.snapshot_download` to pull the entire model repo (~24GB) into `src/models/checkpoints/vadar/`:

```bash
python voder.py vadar "hello"
```

You do **not** need to manually pip install anything beyond `requirements.txt` — `torch`, `transformers`, `psutil`, and `huggingface_hub` are all listed there already.

**Equivalent manual download (only if you prefer the `huggingface-cli`):**

```bash
# Make sure huggingface_hub is installed (it's in requirements.txt)
pip install -U "huggingface_hub[cli]"

# Download the entire model repo into src/models/checkpoints/vadar/
huggingface-cli download OpenYourMind/gemma-4-12B-it-abliterated-uncensored \
  --local-dir src/models/checkpoints/vadar
```

Place every file directly inside `src/models/checkpoints/vadar/` — do not nest them in a sub-folder. The loader requires at least one `.safetensors` or `.bin` weights file to start.

**Dependencies:** VADAR loads the model via `AutoModelForMultimodalLM` and `AutoProcessor` from `transformers`, with `torch` as the backend. The system prompt builder also uses `psutil` to read CPU/RAM specs. All three packages — `torch`, `transformers`, `psutil` — are already in `requirements.txt`, so a normal `pip install -r requirements.txt` covers them.

**Hardware:** the model is 12B parameters. With `bfloat16` on a CUDA GPU you need roughly 24 GB VRAM for comfortable inference; on CPU it runs in `float32` (slow but functional) and needs roughly 48 GB system RAM. VADAR auto-detects: if `torch.cuda.is_available()` is true, it uses `bfloat16` + `device_map="auto"`; otherwise it falls back to CPU `float32`.

**What happens if the model is missing:** `vadar` checks the directory at startup. If the directory doesn't exist or contains no `.safetensors` / `.bin` files, VADAR prints setup instructions (mentioning the `python voder.py vadar "hello"` command and the target path) and exits without running. No error traceback — just a clean message telling you what to do.

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
  - [1.6 Dub (Video/Audio Dubbing)](#16-dub-videoaudio-dubbing)
  - [1.7 Modify Speech (STT+TTS)](#17-modify-speech-stttts)
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
- [Tasks & Features (beyond the 8 modes)](#tasks--features-beyond-the-8-modes)
  - [Voice Training (`train`)](#voice-training-train)
  - [Side-Quests (`quest`)](#side-quests-quest)
    - [download](#download)
    - [noframes](#noframes)
  - [Chains (`chains`)](#chains-chains)
- [VADAR (`vadar`) — the natural-language agent](#vadar-vadar--the-natural-language-agent)
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
- URLs from any supported platform accepted as voice cloning references

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
- URLs from any supported platform — downloaded and processed automatically

**Features:**
- Same-language resynthesis — re-synthesize speech preserving the original voice and language
- Translation to English — translate from any of Whisper's 99 supported languages while preserving the speaker's voice, tone, and delivery style
- Any-to-any translation — translate between any of 76 languages using the `translate (source-target)` syntax with TranslateGemma 12B
- Optional overdose mode — runs an STS v2 non-mimic pass after TTS output for better voice preservation

**CLI Examples:**
```bash
# Same-language resynthesis (preserve voice, same language)
python src/voder.py tts slc "speech.wav"

# Translate to English preserving speaker voice
python src/voder.py tts slc translate "spanish_speech.wav"

# Translate to Arabic with TranslateGemma
python src/voder.py tts slc translate "(auto-ar)" "english_speech.wav"

# Shorthand: (ar) is equivalent to (auto-ar)
python src/voder.py tts slc translate "(ar)" "english_speech.wav"

# Overdose mode: STS v2 non-mimic pass after TTS for better voice preservation
python src/voder.py tts overdose slc translate "speech.wav"
```

### 1.6 Dub (Video/Audio Dubbing)

Dub translates and replaces speech in a video or audio file while preserving the original speaker's voice and the background music. It is the deepest TTS sub-task, combining SVS separation, VibeVoice ASR with audio events, TranslateGemma translation, Fish S2 Pro TTS with voice cloning, per-segment speed adjustment, and timeline-based assembly.

**Supported Inputs:**
- Video files (MP4, MKV, AVI, etc.)
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- URLs from any supported platform — downloaded and processed automatically

**Features:**
- Auto-translate to English by default (no `translate` keyword needed)
- Any-to-any translation via `translate (source-target)` syntax with TranslateGemma
- Per-segment TTS generation and speed adjustment for near-perfect timing alignment
- Audio event preservation — silence, music, and noise segments are detected and left untouched
- Background music preservation via SVS music separation
- Optional subtitle burning with the `subtitle` keyword
- Multi-speaker detection with per-speaker voice cloning

**Pipeline:**
1. SVS voice + music isolation (BS-RoFormer)
2. VibeVoice ASR transcription with audio events (speech segments + non-speech markers)
3. TranslateGemma per-segment translation (with timing context for concise output)
4. Fish S2 Pro TTS per segment (voice cloning from source, short segments avoid drift)
5. Per-segment speed adjustment (match original segment duration)
6. Timeline assembly (overlay each segment at its original position on a silent base)
7. Mix with music track (preserve background music)
8. Video mux or subtitle burn

**CLI Examples:**
```bash
# Dub video to English (default)
python src/voder.py tts dub "video.mp4"

# Dub video to Japanese
python src/voder.py tts dub translate "(auto-ja)" "video.mp4"

# Shorthand: (ja) is equivalent to (auto-ja)
python src/voder.py tts dub translate "(ja)" "video.mp4"

# Dub video with translated subtitles burned on
python src/voder.py tts dub subtitle "video.mp4"

# Dub video to Arabic with subtitles
python src/voder.py tts dub translate "(auto-ar)" subtitle "video.mp4"

# Dub audio file (no video output)
python src/voder.py tts dub "speech.wav"

# Dub YouTube video to French (audio downloaded by default → WAV output)
python src/voder.py tts dub translate "(auto-fr)" "https://youtube.com/watch?v=..."

# Dub YouTube video with `video` keyword — video downloaded → MP4 output
python src/voder.py tts dub video "https://youtube.com/watch?v=..."
```

### 1.7 Modify Speech (STT+TTS)

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

STT is a **standalone transcription mode** available as a one-line CLI command. It transcribes audio, video, images, or platform URLs into plain text with optional enhancements.

### 4.1 Features

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.)
- Image files containing text (PNG, JPG, etc.) — text is extracted via OCR before transcription
- URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter) — downloaded and processed automatically

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

SE (Sound Enhancement) is a standalone mode for improving audio quality through denoising, dereverberation, restoration, and super-resolution.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.) — audio is extracted automatically

**Sub-Modes:**

| Command | Pipeline | Output |
|---------|----------|--------|
| `se "path"` | UniSE enhancement | 16kHz WAV |
| `se voice "path"` | SVS voice → UniSE on vocals | 16kHz WAV |
| `se voice blend "path"` | SVS voice+music → UniSE on voice → blend | 48kHz WAV |
| `se sr "path"` | AudioSR super-resolution (basic model) | 48kHz WAV |
| `se sr music "path"` | SVS music → AudioSR (basic model) | 48kHz WAV |
| `se sr music blend "path"` | SVS voice+music → AudioSR on music + UniSE on voice → blend | 48kHz WAV |
| `se sr voice "path"` | SVS voice → AudioSR (speech model) on vocals | 48kHz WAV |
| `se sr voice blend "path"` | SVS voice+music → AudioSR speech on vocals → blend with music | 48kHz WAV |
| `se sr voice music "path"` | SVS → AudioSR speech on vocals + basic on music → auto-blend | 48kHz WAV |

**Features:**
- Denoising — removes background noise and artifacts
- Dereverberation — reduces room echo and reverb effects
- Speech restoration — enhances clarity and intelligibility
- Super-resolution — upscales low-sample-rate audio to 48kHz via AudioSR
- Voice extraction — SVS isolates vocals before enhancement
- Blend — mixes enhanced vocals with original/upsampled music

**Quick Examples:**
```bash
# Basic enhancement
python src/voder.py se "noisy_audio.wav"

# Enhance extracted voice only
python src/voder.py se voice "song.wav"

# Enhance voice and blend with music
python src/voder.py se voice blend "song.wav"

# Super-resolution upsample to 48kHz
python src/voder.py se sr "low_quality.wav"

# Upsample music and blend with enhanced voice
python src/voder.py se sr music blend "song.wav"

# Voice super-resolution with speech model
python src/voder.py se sr voice "vocals.wav"

# Full SR: speech on vocals + basic on music
python src/voder.py se sr voice music "song.wav"

# Save to specific location
python src/voder.py se "audio.wav" result "/path/to/enhanced.wav"
```

**CLI Usage:**
```bash
# Interactive mode
python src/voder.py cli
# Select option 4 (SE)

# One-liner mode
python src/voder.py se voice blend "audio_file.wav" result "/output/enhanced.wav"
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
- URLs from any supported platform — downloaded and processed automatically

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

# Process a YouTube URL (audio downloaded by default → WAV output)
python src/voder.py svs "https://youtube.com/watch?v=..." voice

# Process a YouTube URL with `video` keyword — video downloaded → MP4 output (one per stem)
python src/voder.py svs "https://youtube.com/watch?v=..." voice video

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

SS extracts individual speaker audio from multi-speaker recordings using VibeVoice ASR for speaker identification and segmentation. With the `blend` keyword, each speaker's audio is mixed with the original non-vocals (instrumental/background) track. With the `video` keyword, separated audio is muxed with the original video to produce MP4 output.

**Supported Inputs:**
- Audio files (WAV, MP3, FLAC, OGG, etc.)
- Video files (MP4, MKV, AVI, etc.)
- URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter)

**Features:**
- Automatic speaker identification and separation
- Produces separate audio files for each detected speaker
- Provides speaker-labeled transcript with timestamps
- `blend` flag: blend each speaker with non-vocals (preserves background audio)
- `video` flag: mux separated audio with original video for MP4 output
- `se` flag: apply sound enhancement before separation
- `overdose` flag: use VibeVoice ASR for better accuracy
- `target` flag: extract a specific speaker using a voice reference
- Requires 24GB+ VRAM or 48GB+ system memory
- Falls back to Whisper + pyannote if VibeVoice ASR cannot load

**Quick Examples:**
```bash
# Separate speakers from a recording
python src/voder.py ss "meeting.wav"

# With blend (each speaker + non-vocals)
python src/voder.py ss blend "vlog.wav"

# With video output
python src/voder.py ss video "interview.mp4"

# Target extraction with video output
python src/voder.py ss target "ref.wav" video "interview.mp4"

# Full pipeline
python src/voder.py ss overdose se blend video "vlog.mp4"
```

---

## Tasks & Features (beyond the 8 modes)

The eight modes above (TTS, STS, TTM, STT, SE, SFX, SVS, SS) are VODER's main processing engine. On top of them, three task-layer features are available as oneline commands: `train` (save reusable voice clones), `quest` (side-quests — lightweight utility tasks), and `chains` (user-defined pipelines of voder oneline tasks). Sitting above all of them is **VADAR**, a natural-language AI agent that can call any of the modes or features on your behalf — see [VADAR (`vadar`) — the natural-language agent](#vadar-vadar--the-natural-language-agent) below.

### Voice Training (`train`)

Train a voice clone from reference audio and save it as a `.tts` (standard, Qwen3-TTS) or `.ttse` (extreme, Fish S2-Pro) file in the `voices/` directory for later reuse in TTS. Oneline-only command — not available in interactive CLI or GUI. Standard mode uses Qwen3-TTS Base; extreme mode (`train extreme voice:name`) uses Fish Audio S2-Pro.

```bash
# Train a voice (standard Qwen3-TTS, saves .tts)
python src/voder.py train voice:narrator "narrator_ref.wav"

# Train from multiple references
python src/voder.py train voice:hero "hero_clip1.wav" "hero_clip2.wav" "hero_clip3.wav"

# Train extreme voice (Fish S2-Pro, saves .ttse)
python src/voder.py train extreme voice:narrator "narrator_ref.wav"

# Train with a test sample after training
python src/voder.py train voice:narrator "ref.wav" test
python src/voder.py train voice:narrator "ref.wav" test "Custom test script"
```

Once trained, the voice can be referenced in TTS via `voice "narrator"` (latest `.tts` from `voices/`) or `voice "narrator:path/to/file.tts"` (specific file). `.tts` files only work without `extreme`; `.ttse` files only work with `extreme`.

### Side-Quests (`quest`)

Side-quests are lightweight utility tasks that live outside the voder engine but are still useful in a voice-processing workflow. They are designed to grow over time as more quests are added. Each quest is implemented as a small class registered in a side-quest registry, so future quests can be added without touching the rest of the codebase.

**Supported Inputs:**
- `download` — URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter, Reddit) + experimental `public_net` for other sites. Audio (default), video (`video` keyword), or image (`image` keyword via gallery-dl).
- `noframes` — local video files only (refuses URLs and audio-only files)

#### download

Downloads a URL (or copies a local file) into `results/downloads/{audios,videos,images}/`. Audio is the default; the optional `video` keyword switches to a full video download; the optional `image` keyword downloads images via gallery-dl.

Supported platforms: YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter, Reddit. Experimental `public_net` support for other sites — attempted via yt-dlp/gallery-dl with a warning. Downloads that fail without cookies are automatically retried with Chrome → Brave → Edge cookies.

```bash
# Download a YouTube URL as audio (MP3, default)
python src/voder.py quest download "https://youtube.com/watch?v=..."

# Download the same URL as a full video (MP4)
python src/voder.py quest download video "https://youtube.com/watch?v=..."

# Download an image (or image gallery) from Reddit/Instagram/X via gallery-dl
python src/voder.py quest download image "https://reddit.com/r/.../comments/..."

# Copy a local audio/video file to results/downloads/ with the quest naming scheme
python src/voder.py quest download "/path/to/local.wav"
python src/voder.py quest download "/path/to/local.mp4"

# Save the result to a specific path
python src/voder.py quest download "https://youtube.com/watch?v=..." result "./out.mp3"
python src/voder.py quest download video "https://youtube.com/watch?v=..." result "./out.mp4"
python src/voder.py quest download image "https://reddit.com/..." result "./out.jpg"
```

**Output naming:** `voder_quest_download_<original-name>_<timestamp>.<ext>`

- For platform URLs, `<original-name>` is derived from the platform video ID (YouTube video ID, TikTok video ID, Bilibili BV id, Instagram reel id, Facebook video id, Twitter status id, Snapchat spotlight id, Reddit post id — sanitized to safe filename characters).
- For local files, `<original-name>` is the file's stem (without extension).
- Extension matches the downloaded/copied file (`.mp3` for audio, `.mp4` for video, etc.).

#### noframes

Extracts the audio track from a **local video file**. This quest deliberately refuses URLs and audio-only files — it is strictly a "video → audio" extractor for files you already have on disk. Use `quest download` if you need to fetch a URL first.

```bash
# Extract audio from a local MP4
python src/voder.py quest noframes "video.mp4"

# Save the result to a specific path
python src/voder.py quest noframes "video.mp4" result "./out.wav"
```

**Output naming:** `voder_quest_noframes_<original-name>_<timestamp>.wav`

- Output is always WAV (PCM 16-bit, 44.1 kHz, stereo) extracted via FFmpeg.
- Refuses inputs whose extension is not a video format (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.wmv`, `.m4v`).
- Refuses URLs — provide a local file path only.

#### Adding More Quests

Side-quests are registered in a `SIDE_QUESTS` registry. Each quest subclasses `SideQuest` and implements `parse(args)` and `execute(parsed, results_dir, timestamp, result_path=None)`. New quests can be added without changing the quest dispatcher or parser.

### Chains (`chains`)

Chains let the user compose their own pipelines out of voder's existing oneline tasks. A chain is a named voder command; its output is captured to a temp directory and indexed under the chain name. Later chains can reference earlier chain names as input paths — voder substitutes the chain name with the temp file path before running the later chain. The **last** non-empty chain's output is exported to `results/`; all intermediate outputs live in `temp_chains/` so they don't pollute the results folder.

**Command format:**
```
python src/voder.py chains "name1" <voder command...> / "name2" <voder command that references "name1"> / "name3" <voder command that references "name1" and/or "name2"> / ...
```

- Use ` / ` (space slash space) to separate chains. The slash must be its own argv element — do not attach it to neighbouring arguments.
- Each chain **starts** with a quoted name (or any single token; quotes are optional but recommended, especially if the name contains spaces).
- The rest of the chain's args are a normal voder oneline command (e.g., `tts script "hi" voice "male"`, `svs voice "song.wav"`, `se "vocals"`, …).
- Inside a later chain, any argument that exactly matches a previous chain name is replaced with that chain's output path. If the argument does not match a chain name, it is treated normally (as a path, URL, or whatever the command expects).
- Intermediate chain outputs are stored in `temp_chains/` with names like `voder_chain_<safe_name>_<timestamp>.<ext>`.
- The **last** non-empty chain's output stays in `results/` (or `voices/` for `train` chains) — that's the user-visible result.

**Validation rules:**

- **Duplicate chain names** are an error and stop the pipeline immediately. Two chains cannot share the same name.
- **Empty chains** (a name with no command following it) are **skipped**. Their names are NOT marked as used, so the same name can be reused later in the same chains command. Example: `"a" / "b" / "a" tts script "hi"` is valid — the first two are empty, and the third (non-empty) chain claims the name `a`.
- **Trailing empty chains** at the end are ignored, just like empty chains in the middle.
- If **all** chains are empty, the pipeline returns an error ("no valid chains to execute").

**Quick Examples:**

```bash
# Generate a song, isolate its vocals, then convert them to a different voice
python src/voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"

# Isolate vocals, enhance them, transcribe the result
python src/voder.py chains "vocals" svs voice "song.wav" / "enhanced" se voice "vocals" / "text" stt "enhanced" timestamp

# Train a voice from a chain's output, then use it to speak
python src/voder.py chains "vocal" svs voice "song.wav" / "trained" train voice:singer "vocal" / "spoken" tts script "Hello world" voice "singer"

# A chain that downloads audio from a URL, then transcribes it
python src/voder.py chains "audio" quest download "https://youtube.com/watch?v=..." / "text" stt "audio" timestamp

# Empty chains are skipped and their names remain reusable — this is valid:
python src/voder.py chains "skip1" / "skip2" / "real" tts script "hi" voice "male"

# Duplicate names are an error and stop the pipeline:
# python src/voder.py chains "a" tts script "one" / "a" tts script "two"   # ERROR
```

**Notes:**

- Chain names are matched exactly (case-sensitive) against command arguments. If a chain name happens to look like a file path or URL, it still wins — voder checks chain names first.
- For multi-output commands (e.g., `svs both`, `ss`, TTM with stems), only the **latest** file produced by the chain is exposed as the chain's output. If you need multiple outputs, run separate chains.
- The `result "<path>"` keyword works as usual on the whole `chains` command — it copies the **final** chain's output to the given path.

---

## VADAR (`vadar`) — the natural-language agent

VADAR is the natural-language layer that sits on top of the 8 modes and 3 task-layer features. Instead of typing `tts script "..." voice "..."` or composing a `chains` command, you describe the task in plain English and VADAR figures out which VODER commands to run, in what order, and reads their outputs to verify the result.

**Two ways to invoke:**

```bash
# Oneline — single natural-language request
python src/voder.py vadar "Generate a 30-second upbeat pop song about rain, then isolate its vocals"
python src/voder.py vadar "Make a slowed+reverb version of song.wav with extra bass" result "./slowed.wav"

# Interactive CLI — multi-turn chat
python src/voder.py cli
# pick option 10: VADAR (AI agent — talk naturally, it decides what to run)
```

**How VADAR works:**

- VADAR is powered by Gemma 4 12B (abliterated uncensored variant from `OpenYourMind/gemma-4-12B-it-abliterated-uncensored`). The model is loaded locally via `AutoModelForMultimodalLM` + `AutoProcessor` from `transformers`. Model files go in `src/models/checkpoints/vadar/` — they download automatically on first run. See [VADAR Model Setup](#vadar-model-setup) above.
- Each request triggers VADAR's **agent loop**: think → decide → reply → act → eval → reply. VADAR can iterate the loop multiple times for complex tasks.
- An **act** is a VODER oneline command VADAR runs (e.g., `ttm lyrics "..." styling "pop" 30`, `quest download "..."`, `svs voice "..."`). Each act has a unique title in the session, and VADAR can read its output using the `read` tool.
- VADAR emits special EOS tokens to signal state transitions: `<EOS_REPLY>` ends a reply (user can respond), `<EOS_ACT>` triggers act execution, `<EOS_DONE>` signals task completion.
- **No network access, no system shell.** VADAR can only run VODER oneline commands and read files inside the VODER project directory (plus paths the user explicitly provides).
- Knowledge cutoff is approximately mid-2025.

**VADAR's tools:**

| Tool | Description |
|------|-------------|
| `look <path\|url>` | Analyze an image file (returns a description of what VADAR sees). |
| `listen <path\|url> [HH:MM:SS-HH:MM:SS]` | Analyze audio. Without a range: total length + summary. With a range: that segment. |
| `watch <path\|url> [HH:MM:SS-HH:MM:SS]` | Analyze video. Same semantics as `listen`. |
| `read <path\|act_title> [start-end]` | Read text or a previous act's output. Without a range: total line count + first 100 lines. With `start-end` line range: those lines. |
| `list [type] [path]` | List files. Type: `videos`, `images`, `audios`, `texts`, `others`, `all`, or `.ext`. |
| `search <query> path <path> [formats <ext1,ext2,...>]` | Search for files containing the query in their name. |
| `memory_read <vadar\|user> <id>` | Read a memory file (VADAR's or the user's). |
| `memory_write <vadar\|user> <content>` | Create a new memory file. |
| `memory_edit <vadar\|user> <id> <content>` | Edit an existing memory file. |
| `memory_delete <vadar\|user> <id>` | Delete a memory file (must have read it first). |
| `calculate <code>` | Run Python code using whitelisted libraries (default: `math` — extendable via `src/voders/vadars/supported_libs.txt`). |

**Session, memory, and personality:**

- Every VADAR invocation creates a session directory at `src/voders/vadars/sessions/<timestamp>_<type>/` (`<type>` is `oneline` or `interactive`). The session holds `inputs.txt`, `outputs.txt`, `acts.txt`, `log.txt`, and `context.txt`.
- VADAR has a sliding context window (~8192 tokens, 95% retention on overflow — the oldest 5% of non-system messages are dropped when the window fills; the system prompt is always preserved).
- Persistent memories live in `src/voders/vadars/memories/vadar/` (VADAR's own) and `src/voders/vadars/memories/user/` (about the user) as numbered `.txt` files.
- VADAR's personality is defined in `src/voders/vadars/about/` — `personality.md` (shipped with content), `custom-vadar.md` (empty by default — you write your own VADAR traits like "I am supportive and funny"), `user.md` (empty by default — you write things about yourself like "my name is John"), and `how-to-respond.md` (shipped with content) — all written in the first person ("I"). These are loaded into the system prompt at the start of every session.
- Config: `src/voders/vadars/config.json (ping_time field)` (default 15s — how long VADAR waits before checking in on a silent user) and `src/voders/vadars/supported_libs.txt` (default `math` — the whitelist for the `calculate` tool).

**Brotherhood:** VADAR is part of the VODER brotherhood alongside Eval (evaluates plans before reply and results after acts) and Summarizer (condenses long outputs into summaries VADAR can work with).

See [Guide.md](Guide.md) for the full VADAR user guide, and [COMMAND_CATALOG.md](COMMAND_CATALOG.md) §11 for the complete syntax reference.

---

## Intelligent Source Analysis

VODER supports **cross-platform source input** — a unified input pipeline that accepts audio, video, images, and URLs across multiple processing modes. This enables powerful new workflows:

- **Universal URL Support:** Paste a video URL from YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, or X/Twitter directly as input in STT, SVS, SLC (via TTS), and dialogue modes. VODER's two-step detection first checks the URL shape (host + path patterns per platform) to reject channel pages, profiles, playlists, and photo posts offline; then runs yt-dlp with `download=False` to verify the link actually resolves to a downloadable video stream before downloading. Once verified, the audio track is downloaded automatically — no manual downloading or conversion required.
- **Image Text Extraction (OCR):** Feed image files (PNG, JPG, etc.) as input. VODER uses EasyOCR to extract embedded text, which is then processed as dialogue script content. This works in STT, TTS, and TTS modes — enabling workflows like "photo of a script → spoken audio."
- **Automatic Voice Clip Extraction:** When processing multi-speaker audio (e.g., a podcast recording), VODER can automatically identify and extract individual speaker segments. This replaces the previous manual approach of splitting audio files.
- **Speaker Diarization:** Powered by pyannote, VODER identifies who spoke when in multi-speaker audio. Each speaker is labeled consistently, and the diarization output can be combined with transcription for fully annotated results.

> **Multi-Speaker Input — Now Supported!** Previous versions of VODER required manually separating multi-speaker audio before processing. With the new Intelligent Source Analysis system, VODER can now accept multi-speaker audio directly. The speaker diarization pipeline automatically identifies speakers, extracts their voice clips, and makes them available for voice cloning and transcription. See [Guide.md](Guide.md) for the updated workflow.

---

## AI Model Integration

VODER leverages state-of-the-art open-source models for professional-grade audio processing:

- **Speech Recognition:** [openai/whisper](https://github.com/openai/whisper) — Whisper for accurate audio transcription and translation
- **Voice Synthesis:** [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — Qwen3-TTS for natural text-to-speech
- **Voice Synthesis (Extreme):** [FishAudio/S2-Pro](https://huggingface.co/fishaudio/s2-pro) — Fish Audio S2-Pro for higher quality cloning, 80+ language support, and voice effects via `[tag]` syntax (activated with `extreme` keyword)
- **Voice Conversion:** [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) — Seed-VC for speech-to-speech transformation
- **Music Generation:** [ace-step/ACE-Step-1.5](https://github.com/ace-step/ACE-Step-1.5) — ACE-Step for lyrics-to-music synthesis
- **Sound Effects:** [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) — TangoFlux for text-to-audio generation
- **Sound Enhancement:** [alibaba/unified-audio](https://github.com/alibaba/unified-audio) + [versatile_audio_super_resolution](https://github.com/haoheliu/versatile_audio_super_resolution) — UniSE for denoising/dereverb, AudioSR for super-resolution
- **Voice Separation:** [BS-RoFormer Resurrection](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) — BS-RoFormer for vocal/music isolation
- **Advanced ASR:** [Microsoft VibeVoice](https://github.com/microsoft/VibeVoice) — VibeVoice ASR for speaker diarization, transcription, and overdose mode
- **Any-to-Any Translation:** [Google TranslateGemma 12B](https://huggingface.co/google/translategemma-12b-it) — TranslateGemma for translation between 76 languages, decoupled from ASR engine
- **Speaker Diarization:** [pyannote/speaker-diarization-community-1](https://github.com/pyannote/pyannote-audio) — pyannote for identifying and labeling individual speakers in multi-speaker audio
- **Image Text Extraction:** [EasyOCR](https://github.com/JaidedAI/EasyOCR) — EasyOCR for extracting text from images, enabling image-to-speech workflows
- **VADAR AI Agent (Heavy/Overdose):** [OpenYourMind/gemma-4-12B-it-abliterated-uncensored](https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored) — Gemma 4 12B (abliterated uncensored variant) for the heavy VADAR natural-language agent. Multimodal (text + image + audio + video), loaded locally via `AutoModelForMultimodalLM` + `AutoProcessor` from `transformers`. The model downloads automatically on first run (uses `huggingface_hub.snapshot_download` to pull files into `src/models/checkpoints/heavy_vadar/`) — see [VADAR Model Setup](#vadar-model-setup) above.
- **VADAR AI Agent (Lite):** [Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit](https://huggingface.co/Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit) — SuperGemma 4 12B (abliterated, GGUF Q4_K_M 4-bit quantized) for the lite VADAR natural-language agent. Text-only (no multimodal), loaded via `llama-cpp-python` (llama.cpp). The GGUF model + chat template download automatically on first run (uses `huggingface_hub.hf_hub_download` to pull files into `src/models/checkpoints/lite_vadar/`). Runs on 16GB RAM / 4 CPU cores / any T4 GPU.

---

## Usage Guide

### GUI Mode

1. Launch: `python src/voder.py gui`
2. Select mode from dropdown (8 available modes)
3. Load input files based on mode:
   - **TTS:** Enter dialogue row‑by‑row in the script area, and fill the automatically generated voice prompts for each character. Use the `target` field for voice cloning from a reference audio file, or leave blank for voice design from a text prompt. Optionally set a `language` parameter for TTS output language. URLs from any supported platform are accepted as voice prompts for cloning.
     **Optional:** Before generation, a dialog will ask if you want background music; enter a description or press Skip.
     **Modify Speech:** At the start, a prompt asks "modify speech? (Y/N)" — answer yes to load audio/video/URL, transcribe, edit text, choose voice, and re-synthesize.
   - **STS:** Load base audio/video and target voice audio. Video input is accepted and video output is produced automatically. When a target contains mixed audio, vocals are extracted via BS-RoFormer.
   - **TTM:** Enter lyrics and style prompt. Supports sub-tasks (complete, lego, extract, remix, repaint) and a three-tier ACE-Step quality system (standard, overdose, complete). Use the `vc` flag for voice conversion with a clone audio reference. Outputs up to 12 instrument tracks.
   - **STT:** Load audio, video, image, or enter a URL for transcription
   - **SE:** Load audio or video file for enhancement
   - **SFX:** Enter a text description of the desired sound effect
   - **SVS:** Load audio, video, or enter a platform URL for vocal/music isolation
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

**TTS — Dub (Video/Audio Dubbing):**
```bash
# Dub video to English (default)
python src/voder.py tts dub "video.mp4"

# Dub video to Japanese with TranslateGemma
python src/voder.py tts dub translate "(auto-ja)" "video.mp4"

# Shorthand: (ja) is equivalent to (auto-ja)
python src/voder.py tts dub translate "(ja)" "video.mp4"

# Dub video with translated subtitles
python src/voder.py tts dub subtitle "video.mp4"

# Dub video to Arabic with subtitles
python src/voder.py tts dub translate "(auto-ar)" subtitle "video.mp4"

# Dub audio file
python src/voder.py tts dub "speech.wav"
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

# Translate to any language with TranslateGemma
python src/voder.py stt "audio.wav" translate "(auto-ja)"
python src/voder.py stt "audio.wav" translate "(ar-fr)"

# Shorthand: (ja) is equivalent to (auto-ja)
python src/voder.py stt "audio.wav" translate "(ja)"

# Overdose + translate to Japanese
python src/voder.py stt "audio.wav" overdose translate "(auto-ja)"

# Subtitle a video with translated text
python src/voder.py stt overdose subtitle translate "(auto-en)" "video.mp4"

# Shorthand: (en) is equivalent to (auto-en)
python src/voder.py stt overdose subtitle translate "(en)" "video.mp4"

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
python src/voder.py se voice blend "song.wav"
python src/voder.py se sr "low_quality.wav"
python src/voder.py se sr music blend "song.wav"
python src/voder.py se sr voice "vocals.wav"
python src/voder.py se sr voice music "song.wav"
python src/voder.py se "audio.wav" result "/path/to/enhanced.wav"
```

**SFX mode:**
```bash
python src/voder.py sfx sound "rain on a tin roof" duration 10
python src/voder.py sfx sound "thunder rumbling" duration 5 steps 50 guide 3.5
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"
```

**Side-Quests (`quest`):**
```bash
# Download a URL as audio (default)
python src/voder.py quest download "https://youtube.com/watch?v=..."

# Download a URL as video
python src/voder.py quest download video "https://youtube.com/watch?v=..."

# Extract audio from a local video file (refuses URLs and audio files)
python src/voder.py quest noframes "video.mp4"

# With result path
python src/voder.py quest download "https://youtube.com/watch?v=..." result "./out.mp3"
python src/voder.py quest noframes "video.mp4" result "./out.wav"
```

**Chains (`chains`):**
```bash
# Generate a song → isolate vocals → voice-convert them
python src/voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"

# Isolate vocals → enhance → transcribe
python src/voder.py chains "vocals" svs voice "song.wav" / "enhanced" se voice "vocals" / "text" stt "enhanced" timestamp

# Download audio → transcribe it
python src/voder.py chains "audio" quest download "https://youtube.com/watch?v=..." / "text" stt "audio" timestamp

# Empty chains are skipped (names remain reusable); duplicate names are an error
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
- **Sound Enhancement:** Denoise, dereverb, restore, and super-resolve audio (voice, sr, sr music sub-modes)
- **Vocal/Music Separation:** BS-RoFormer integration for automatic vocal extraction — used internally by STS (target cleanup), STT (pre-cleanup isolation), TTS (voice cloning target cleanup), and available as a standalone SVS mode
- **Speech Language Conversion (SLC):** Integrated into TTS as an oneline sub-task — translate speech to English or any of 76 languages via TranslateGemma, or re-synthesize in the same language while preserving the speaker's voice, with optional overdose mode for enhanced voice fidelity
- **Video/Audio Dubbing (TTS Dub):** Translate and replace speech in videos while preserving the original speaker's voice and background music — per-segment TTS generation, speed adjustment, and timeline-based assembly for near-perfect timing alignment
- **Any-to-Any Translation (TranslateGemma):** Translate between any of 76 languages using the `translate (source-target)` syntax, decoupled from the ASR engine — works with Whisper, VibeVoice ASR, SLC, dub, and subtitle modes
- **Modify Speech (STT+TTS):** Integrated into TTS interactive mode — transcribe, edit, and re-synthesize speech with source or custom voice selection
- **Side-Quests (`quest`):** Lightweight utility tasks, grouped by category in the `quest` listing (run `quest` with no args to see the live tree). `download` (URL → audio/video file in `results/`) stands alone at the top; the other 17 quests live under the **Media Manipulation** category, split into three sub-categories — **Sound Effects** (bassboost, fade, loudnorm, pitch, reverb, soundlevel, speed), **Audio Editing** (cut, merge, mix, remove, reverse, silence), and **Format & File** (compress, convert, glue, noframes). Categorization is defined externally in `src/voders/quests_categories.py`. New quests can be added to the registry without touching the dispatcher.
- **VADAR AI Agent (`vadar`):** A natural-language agent powered by Gemma 4 12B (abliterated uncensored variant). Describe a task in plain English — VADAR thinks, decides, replies, and acts, running any VODER oneline command (or chain of commands) on your behalf. Has its own tools (`look` / `listen` / `watch` for image / audio / video analysis, `read` / `list` / `search` for files and prior act outputs, `memory_read` / `memory_write` / `memory_edit` / `memory_delete` for persistent memory, `calculate` for Python with whitelisted libraries), a sliding context window (~8192 tokens, 95% retention on overflow), and a configurable personality in `src/voders/vadars/about/`. Each session is logged under `src/voders/vadars/sessions/<timestamp>_<type>/` with separate `inputs.txt`, `outputs.txt`, `acts.txt`, `log.txt`, and `context.txt` files. Memories live in `src/voders/vadars/memories/vadar/` and `src/voders/vadars/memories/user/`. Part of the VODER brotherhood alongside Eval (plan/result evaluator) and Summarizer (long-output condenser). No network access, no system shell — only VODER project paths + user-provided paths. The model downloads automatically on first run; see [VADAR Model Setup](#vadar-model-setup) above.
- **Chains (`chains`):** User-defined pipelines that wire any number of voder tasks together — each chain is named, its output is captured to `temp_chains/`, and later chains can reference earlier chain names as input paths. The last non-empty chain's output reaches `results/`. Empty chains are skipped (names remain reusable); duplicate names are an error.
- **Cross-Modal Transformation:** Speech-to-speech, text-to-speech, speech-to-text, text-to-text, and speech language conversion
- **Cross-Platform Source Input:** Unified input pipeline accepts audio files, video files, images, and URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter) across multiple modes — no manual format conversion required
- **VibeVoice ASR:** Microsoft VibeVoice for overdose transcription, speaker diarization, and speaker separation with automatic fallback to Whisper + pyannote
- **Language Parameter in TTS:** Specify output language for TTS synthesis
- **Translation Capability in STT:** Translate audio to English from any of Whisper's 99 supported languages
- **Video I/O for STS Mode:** Feed video files directly into STS and receive video output with converted voice audio
- **Universal URL Expansion:** URL support expanded across STT, SVS, SLC (via TTS), and dialogue modes for YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, and X/Twitter
- **Automatic Speaker Identification:** Multi-speaker audio is automatically segmented and labeled using pyannote speaker diarization, with individual voice clips extracted for downstream processing
- **Speaker Diarization with Word-Level Alignment:** Combines Whisper transcription with pyannote diarization to produce speaker-labeled, timestamped transcripts with per-word speaker attribution
- **MSTS (Music-STS):** STS mode supports musical inputs using Seed-VC v1 at 44.1kHz for better music voice conversion
- **Memory Optimisation:** Models are now explicitly offloaded after each operation to prevent memory accumulation in session-based workflows
- **Background Music for Dialogue:** Automatically generated, duration‑fitted, volume‑controlled ambient music with time-based level adjustments
