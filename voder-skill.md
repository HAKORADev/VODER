# VODER Skill for AI Agents

## Overview

VODER is a professional-grade voice processing tool that provides **8 distinct audio transformation modes** in a unified CLI interface. This skill enables AI agents to leverage VODER's full potential for complex audio processing workflows that would be impossible or extremely difficult without this knowledge.

The eight modes are: **TTS** (Text-to-Speech with optional voice cloning, SLC sub-task, and interactive modify-speech), **STS** (Speech-to-Speech voice conversion), **TTM** (Text-to-Music with optional voice cloning), **STT** (Speech-to-Text transcription), **SE** (Speech Enhancement), **SFX** (Sound Effects generation), **SVS** (Source/Track Vocal Separation), and **SS** (Speaker Separation).

> **Note**: SLC (Spoken Language Conversion / Dubbing) is now a TTS oneline sub-task (`tts slc`), not a standalone mode. STT+TTS (transcribe → edit → resynthesize) is now integrated into TTS interactive mode as a "modify speech?" prompt, not a standalone mode.

**Core Philosophy**: VODER prioritizes **quality over speed**. There are no "fast" or "degraded" model options. The tool uses the best available models (Whisper large-v3-turbo / large-v3, Qwen3-TTS, Seed-VC, ACE-Step XL-Turbo / XL-Base / 1.5, BS-RoFormer Resurrection, VibeVoice ASR, Pyannote, UniSE, TangoFlux) to produce professional-quality output.

---

# SECTION 1: UNDERSTANDING THE ARCHITECTURE

## What VODER Actually Is

VODER is not a single AI model — it is an **orchestration layer** that coordinates multiple state-of-the-art AI models to perform audio transformations. Understanding this architecture is crucial for combining features effectively.

### The Model Stack

| Model | Purpose | Used In Modes |
|-------|---------|---------------|
| **Whisper large-v3-turbo** | Fast speech-to-text transcription | STT, Dialogue Source Analysis |
| **Whisper large-v3** | High-accuracy transcription + translation to English | STT with `translate` flag, TTS `slc` sub-task (translation step) |
| **Qwen3-TTS VoiceDesign** | Generate speech from voice descriptions | TTS (voice design path) |
| **Qwen3-TTS Base** | Text-to-speech with built-in voice cloning | TTS (voice clone path via `target`), TTS `slc` sub-task (resynthesis step), TTS interactive modify-speech |
| **Seed-VC v2** | Voice conversion (22.05kHz speech) | STS, TTM with `vc` flag, TTS `slc overdose` (non-mimic pass) |
| **Seed-VC v1** | Voice conversion (44.1kHz music) | MSTS (music voice conversion) |
| **ACE-Step XL-Turbo** | Enhanced music generation (highest quality) | TTM with `overdose` flag, TTS with `overdose` + `music` |
| **ACE-Step XL-Base** | Music generation (complete-mode sub-tasks) | TTM (`complete`, `extract`, `lego`) |
| **ACE-Step 1.5** | Music generation (legacy / background music) | TTM (default), Background Music (dialogue `music` param) |
| **BS-RoFormer Resurrection** | Vocal/music separation (stem extraction) | SVS, STS (auto vocal extraction), STT (pre-cleanup), TTS (voice clone cleanup, SLC voice isolation), TTM `bgm` (strip music + reference cleanup) |
| **VibeVoice ASR** | Advanced ASR with native speaker diarization | STT with `overdose` flag, TTS with `overdose` flag, SS |
| **Pyannote** | Speaker diarization (who spoke when) | STT with `dialogue` flag |
| **EasyOCR** | Text extraction from images | STT with image input |
| **UniSE** | Speech enhancement/denoising | SE |
| **TangoFlux** | Text-to-audio sound effects | SFX, TTM `bgm`/`complete` (SFX overlay) |

### How Modes Relate to Each Other

```
INPUT TYPES:
┌──────────────────────────────────────────────────────────────────┐
│ Text ──────────────────► TTS, TTM, SFX                            │
│ Audio ─────────────────► STS, STT, SE, SVS, SS                  │
│ Audio/Video/URL ───────► TTS slc (language conversion sub-task) │
│ Video ─────────────────► STS, STT, SE, SVS, SS (auto-extract)   │
│ Image ─────────────────► STT (OCR text extraction)              │
│ YouTube/URL ───────────► STT, STS, TTM, SVS (auto-dl)          │
└──────────────────────────────────────────────────────────────────┘

OUTPUT TYPES:
┌──────────────────────────────────────────────────────────────────┐
│ Audio Output: TTS, STS, TTM, SE, SFX, SVS                       │
│ Audio Stems:  SVS (voice + instrumental)                         │
│ Audio Files:  SS (per-speaker segments)                          │
│ Text Output:  STT, SS (transcript)                               │
│ Interactive:  TTS interactive modify-speech (requires text edit) │
└──────────────────────────────────────────────────────────────────┘
```

### The Pipeline Flow

Understanding how data flows through VODER helps you chain operations:

```
TEXT INPUT PATH (TTS - Voice Design):
Text + Voice Description → Qwen3-TTS VoiceDesign → [Speech with Designed Voice]

TEXT INPUT PATH (TTS - Voice Cloning):
Text + Reference Audio → Qwen3-TTS Base (extract voice embedding → synthesize with clone) → [Speech with Cloned Voice]

AUDIO INPUT PATH (Voice Conversion - STS):
Source Audio + Target Voice Audio → Seed-VC → [Converted Audio]

AUDIO INPUT PATH (Transcription):
Audio → Whisper (large-v3-turbo) → [Transcript Text]

AUDIO INPUT PATH (Translation):
Audio → Whisper large-v3 → [English Transcript Text]

AUDIO INPUT PATH (Overdose Transcription):
Audio → VibeVoice ASR → [Transcript with Native Diarization]

MUSIC GENERATION PATH (Standard):
Lyrics + Style → ACE-Step 1.5 → [Music]

MUSIC GENERATION PATH (Overdose):
Lyrics + Style → ACE-Step XL-Turbo → [Enhanced Quality Music]

MUSIC GENERATION PATH (Voice Clone):
Lyrics + Style → ACE-Step → [Music with Vocals] → Seed-VC Voice Clone → Final Music

ENHANCEMENT PATH:
Degraded Audio → UniSE → [Clean Audio at 16kHz]

SEPARATION PATH (SVS):
Mixed Audio → BS-RoFormer → [Vocals] + [Instrumental]

LANGUAGE CONVERSION PATH (TTS SLC Sub-Task):
Source Audio/Video/URL → SVS Voice Isolation → Whisper Translate → English Text → Qwen3-TTS (with voice ref) → Translated Audio
[With overdose: → Seed-VC v2 non-mimic pass for better voice preservation]

SPEAKER SEPARATION PATH (SS):
Multi-Speaker Audio → VibeVoice ASR → Speaker Segments → Individual Audio Files

BGM REPLACEMENT PATH (TTM BGM):
Source Audio/Video → SVS Voice Pipe (strip music) → Detect Duration → ACE-Step (generate new bgm) → Mix at level → [SFX overlay via TangoFlux] → [Re-mux if video]

SFX OVERLAY PATH (TTM BGM/Complete):
Source Audio → [BGM mixing or instrument blending] → TangoFlux (generate SFX) → Overlay at position/level → [Output]
```

---

## How Parameters Work Together

### Parameter Types

VODER uses three types of parameters:

| Type | Description | Examples |
|------|-------------|----------|
| **Positional** | Mode name comes first, input files follow | `stt "audio.wav"` |
| **Named** | Key-value pairs with space separation | `voice "male"` `duration 30` |
| **Flags** | Standalone keywords that enable features | `timestamp` `dialogue` `music` `translate` `overdose` `mimic` `vc` `nomusic` |

### Parameter Multiplicity

Some parameters accept **multiple values** (dialogue mode), others accept **single values**:

| Parameter | Single Value | Multiple Values | Mode |
|-----------|--------------|-----------------|------|
| `script` | `"Hello world"` | `"James: Hello" "Sarah: Hi"` | TTS |
| `voice` | `"male voice"` | `"James: male" "Sarah: female"` | TTS |
| `target` | `"voice.wav"` | `"James: james.wav" "Sarah: sarah.wav"` | TTS, STS |
| `music` | `"ambient"` | (single only) | TTS (dialogue) |
| `level` | `"35"` | (single only) | TTS (dialogue) |
| `reference` | `"ref.wav"` / `"ref.mp4"` / URL | (single only) | TTS (dialogue bgm) |
| `lyrics` | `"..."` | (single only) | TTM |
| `styling` | `"pop"` | (single only) | TTM |
| `stem` | `"voice"` | (single only) | SVS |
| `sfx:` | `"thunder/10-5/70"` | `"thunder/10-5/70" "rain/30-0/25"` | TTM (bgm/complete) |
| `sound` | `"rain"` | (single only) | SFX |
| `steps` | `30` | (single only) | SFX |
| `guide` | `4.5` | (single only) | SFX |

### Parameter Order Rules

1. **Mode comes first**: `tts`, `stt`, `sts`, `ttm`, `svs`, `ss`, etc. Sub-tasks follow mode: `tts slc`, `tts overdose slc`
2. **Required parameters follow**: `script`, `voice`, `target`, `base`, `lyrics`, `styling`, etc.
3. **Optional parameters come after**: `music`, `level`, `result`, `vc`, `stem`, `task`, etc.
4. **Flags can appear anywhere after mode**: `timestamp`, `dialogue`, `music` (STS), `mimic` (STS), `nomusic` (STS), `translate` (STT), `overdose` (STT, TTM, TTS), `vc` (TTM)

---

# SECTION 2: COMPLETE ONE-LINE CLI COMMANDS CATALOG

## Catalog Navigation

| Mode | Section | Input Type | Output Type | One-Liner Support |
|------|---------|------------|-------------|-------------------|
| TTS | 2.1 | Text [ + Audio ] | Audio | ✅ Full (single + dialogue, voice cloning via `target`, trained voices via `voice`, SLC sub-task via `slc`) |
| Voice Training | 2.1a | Audio | .tts file | ✅ Full (oneline only) |
| STS | 2.2 | Audio/Video + Audio | Audio/Video | ✅ Single only |
| TTM | 2.3 | Text [ + Audio ] | Audio | ✅ Single only (voice cloning via `vc` + `clone`) |
| STT | 2.4 | Audio/Video/Image/URL | Text | ✅ Full (single + batch) |
| SE | 2.5 | Audio/Video | Audio/Video | ✅ Full |
| SFX | 2.6 | Text | Audio | ✅ Full |
| SVS | 2.7 | Audio/Video/URL | Audio (stems) | ✅ Full |
| SS | 2.8 | Audio/Video/URL | Audio + Text | ✅ Full |

> **Note**: `tts+vc` and `ttm+vc` are no longer accepted as commands and will produce an error. Use `tts` with `target` for voice cloning, and `ttm` with `vc` + `clone` for voice conversion in TTM. SLC is now a TTS sub-task (`tts slc`), not a standalone mode. STT+TTS is now integrated into TTS interactive mode.

---

## 2.1 TTS (Text-to-Speech with Voice Design & Voice Cloning)

### What It Is
TTS mode generates human-like speech from text input. It supports two synthesis paths in a single unified mode:

1. **Voice Design** (`voice` parameter): Creates voices from scratch based on natural language descriptions using Qwen3-TTS VoiceDesign. You can describe voices that don't exist in any database — a "weathered old sailor with a gravelly voice" or a "cheerful AI assistant with a slight metallic quality."

2. **Voice Cloning** (`target` parameter): Generates speech that sounds like a specific real person from a reference audio file using Qwen3-TTS Base's built-in cloning capability. The reference audio can be a recording of anyone (with ethical consent), and the output will match their voice characteristics.

Both paths can be **mixed in the same dialogue** using the cross-use feature — some characters designed, others cloned.

> **Note**: The old `tts+vc` command is no longer accepted. Use `tts` with the `target` parameter instead.

### How It Works

**Voice Design Path:**
1. **Voice Prompt Interpretation**: The model parses your voice description to extract characteristics (age, gender, tone, pace, accent)
2. **Speech Synthesis**: Text is converted to mel-spectrograms based on the voice characteristics
3. **Audio Generation**: Spectrograms are converted to waveform audio

**Voice Cloning Path (IMPORTANT: Uses Qwen3-TTS Base Built-in Cloning):**

**Voice cloning does NOT use Seed-VC**. It uses **Qwen3-TTS Base's built-in voice cloning capability**:

1. **Voice Embedding Extraction**: Qwen3-TTS Base's `create_voice_clone_prompt()` method analyzes the reference audio and extracts a voice embedding (x-vector) using `x_vector_only_mode=True`
2. **Direct Synthesis with Clone**: The `generate_voice_clone()` method synthesizes the text **directly with the cloned voice characteristics embedded** — this is NOT a two-step process (synthesis then conversion), but a single integrated process
3. **Consistency Optimization**: In dialogue mode, the voice embedding is extracted **once per character** at the start and reused for all their lines

**Why This Matters**: Unlike a two-stage process (synthesize → convert), Qwen3-TTS Base's integrated cloning produces more natural results because the voice characteristics are considered during the entire synthesis process, not applied as a transformation afterward.

**Shared Path:**
4. **Optional Music Addition**: If `music` parameter is provided, ACE-Step generates background music that matches the dialogue duration

### TTS Overdose Mode

When the `overdose` flag is added to TTS, it activates an enhanced processing pipeline for dialogue source analysis and voice clip extraction:

- **VibeVoice ASR instead of Whisper+Pyannote**: TTS overdose uses VibeVoice ASR for dialogue source analysis and voice clip extraction, providing superior speaker diarization with native speaker identification — no separate Pyannote step needed
- **Safer voice clip extraction**: Voice clips extracted with overdose trim the first 2 seconds and last 3 seconds from the longest segment per speaker, avoiding cross-speaker overlap at segment boundaries and producing cleaner reference audio for cloning
- **ACE-Step XL Turbo for music**: When `music` is also specified alongside `overdose`, ACE-Step XL Turbo is used instead of the standard ACE-Step 1.5 for background music generation, producing higher quality music

```bash
# TTS overdose with voice design
python src/voder.py tts overdose script "James: Hello" "Sarah: Hi" voice "James: deep male" "Sarah: cheerful female"

# TTS overdose with voice cloning and music (XL Turbo for bgm)
python src/voder.py tts overdose script "A: line" "B: line" target "A: ref.wav" "B: ref.wav" music "ambient"
```

### When to Use Voice Design vs Voice Cloning

| Scenario | Voice Design (`voice`) | Voice Cloning (`target`) |
|----------|----------------------|------------------------|
| Fictional characters | ✅ Ideal | ❌ No reference exists |
| Brand-consistent content | ✅ If voice profile defined | ✅ If reference available |
| Localization | ✅ Possible | ✅ Better — preserves identity |
| Accessibility | ❌ No reference | ✅ Use familiar voice |
| Podcast/narration | ✅ Full control | ✅ Match existing host |
| Testing/prototyping | ✅ Fast iteration | ❌ Need reference first |

### Command Catalog

#### Single Mode (One Speaker) — Voice Design
```bash
# Minimal command
python src/voder.py tts script "Your text here" voice "voice description"

# With output routing
python src/voder.py tts script "Your text here" voice "voice description" result "/output/file.wav"

# Full command with music
python src/voder.py tts script "Your text here" voice "voice description" music "music description" level "volume" result "/output/file.wav"

# OCR input (image to narration)
python src/voder.py tts ocr "path/to/image.png" voice "text: professional male narrator"

python src/voder.py tts ocr "script_screenshot.jpg" voice "text: warm female voice"
```

#### Single Mode (One Speaker) — Voice Cloning
```bash
# Voice cloning with target parameter
python src/voder.py tts script "Your text here" target "voice_reference.wav"

# Multi-reference cloning (concatenates references into composite)
python src/voder.py tts script "Your text here" target "(voice1.wav)(voice2.wav)(voice3.wav)"

# Multi-reference cloning with first keyword (extract only first ref's speaker from all others via TSE)
python src/voder.py tts script "Your text here" target first "(voice1.wav)(voice2.wav)(voice3.wav)"

# With output routing
python src/voder.py tts script "Your text here" target "voice_reference.wav" result "/output/file.wav"

# OCR input with voice clone
python src/voder.py tts ocr "path/to/image.png" target "text: voice_reference.wav"
```

#### Dialogue Mode (Multiple Speakers) — Voice Design
```bash
# Two characters
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice prompt1" "Character2: voice prompt2"

# Three+ characters
python src/voder.py tts script "A: line" "B: line" "C: line" voice "A: prompt" "B: prompt" "C: prompt"

# Dialogue with background music
python src/voder.py tts script "A: line1" "B: line2" voice "A: prompt1" "B: prompt2" music "ambient description"

# Dialogue with music and volume control
python src/voder.py tts script "A: line1" "B: line2" voice "A: prompt1" "B: prompt2" music "ambient description" level "35"

# Dialogue with SFX lines embedded
python src/voder.py tts script "A: Hello" "sfx: door bell /duration:3" "B: Who's there?" voice "A: male" "B: female"

# Full dialogue command with all features
python src/voder.py tts script "A: Welcome /time:0" "sfx: intro /duration:5 /level:40 /time:0" "B: Hello! /time:6" voice "A: deep male" "B: bright female" music "soft ambient" level "0:30-60:20" result "/output/podcast.wav"

# Dialogue with background music and reference for style guidance
python src/voder.py tts script "A: line1" "B: line2" voice "A: prompt" "B: prompt" music "ambient" reference "style_ref.wav"

# Dialogue with background music and video reference for style guidance
python src/voder.py tts script "A: line1" "B: line2" voice "A: prompt" "B: prompt" music "ambient" reference "style_ref.mp4"

# Dialogue with background music and YouTube URL reference for style guidance
python src/voder.py tts script "A: line1" "B: line2" voice "A: prompt" "B: prompt" music "ambient" reference "https://youtube.com/watch?v=..."
```

#### Dialogue Mode (Multiple Speakers) — Voice Cloning
```bash
# Two characters with cloned voices
python src/voder.py tts script "James: line1" "Sarah: line2" target "James: /path/to/james.wav" "Sarah: /path/to/sarah.wav"

# With background music
python src/voder.py tts script "J: Hello" "S: Hi" target "J: james.wav" "S: sarah.wav" music "jazz background" level "30"
```

#### Dialogue Mode — Cross-Use (Mix Designed + Cloned)
```bash
# Mix designed and cloned voices in the same dialogue
python src/voder.py tts script \
  "James: Welcome to our podcast!" \
  "Sarah: Thanks for having me!" \
  voice "James: deep male voice, authoritative" \
  target "Sarah: /path/to/sarah_voice_reference.wav"

# Cross-use: James cloned, Sarah designed
python src/voder.py tts script \
  "James: Let me share my screen." \
  "Sarah: Go ahead, I'm ready." \
  target "James: /path/to/james_voice.wav" \
  voice "Sarah: bright female voice, enthusiastic"

# Three characters: mixed approach
python src/voder.py tts script \
  "Host: Welcome to the debate!" \
  "Guest1: Thank you for having me." \
  "Guest2: Pleasure to be here." \
  voice "Host: professional broadcaster, neutral accent" \
  target "Guest1: /path/to/guest1.wav" "Guest2: /path/to/guest2.wav"
```

### Parameter Reference

| Parameter | Required | Purpose | Single Mode | Dialogue Mode |
|-----------|----------|---------|-------------|---------------|
| `script` | Yes | Text to synthesize | Single text string | Multiple `"Char: text"` strings |
| `voice` | Yes* | Voice description or trained voice | Single prompt or trained voice name | `"Char: prompt"` per character or `"Char: trained-name"` |
| `target` | No* | Voice reference file | Single path or multi-ref `(path1)(path2)` | `"Char: /path/to/file.wav"` or `"Char:(ref1.wav)(ref2.wav)"` |
| `music` | No | Background music style | Ignored | Single description |
| `level` | No | Music volume | Ignored | Volume specification |
| `reference` | No | Reference audio/video/URL for bgm style guidance | Ignored | Single path (processed via SVS music pipe to extract clean instrumental) |
| `overdose` | No | Use VibeVoice ASR for source analysis + safer voice clip extraction; XL Turbo for bgm when `music` is set | Ignored | Flag only |
| `result` | No | Output destination | Path | Path |

*Either `voice` or `target` required for non-SFX lines. Can mix both using cross-use feature. If `target` is provided without `voice`, voice cloning path is used automatically. The `voice` parameter also accepts trained voice references — see **Trained Voice Usage** below.

### Trained Voice Usage in TTS

When using the `voice` parameter, a trained voice name or path can be used instead of a voice description. When a trained voice is used, Qwen3-TTS Base (voice cloning) is used instead of VoiceDesign.

| Syntax | Behavior |
|--------|----------|
| `voice "character-name"` | Uses the latest `.tts` file with that name from `voices/` |
| `voice "character-name:path/to/file.tts"` | Uses a specific `.tts` file |
| `voice "character-name:another-name"` | Uses the latest `.tts` file for `another-name` from `voices/` |

This works in both oneline and interactive CLI modes.

```bash
# Single mode with trained voice
python src/voder.py tts script "Hello world" voice "my-character"

# Dialogue with trained voices
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: hero" voice "Sarah: heroine"

# Mix trained and described voices
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: hero" voice "Sarah: cheerful female"

# Specific .tts file
python src/voder.py tts script "Hello" voice "narrator:voices/voder_tts_narrator_20260101.tts"
```

### Newline Support in TTS Scripts

Use `\n` in script text for actual newlines. Works in both oneline and interactive CLI modes.

```bash
# Newline in dialogue script
python src/voder.py tts script "James: First line\nSecond line" voice "James: deep male"

# Newline in single mode
python src/voder.py tts script "First paragraph\nSecond paragraph" voice "professional narrator"
```

### SLC Sub-Task (Spoken Language Conversion / Dubbing)

SLC (Spoken Language Conversion) is now a TTS oneline sub-task that translates spoken content from one language to another and re-synthesizes it with speech. It combines Whisper's translation capability with Qwen3-TTS's voice synthesis to produce **dubbed audio** — the content is translated but the voice character is preserved (or replaced).

**How It Works:**
1. **Source Input**: Accepts audio files, video files, and YouTube/URL sources
2. **SVS Voice Isolation**: BS-RoFormer isolates the voice from the source (handles mixed audio/video)
3. **Translation**: Whisper large-v3 transcribes and translates the isolated voice to English
4. **Voice Extraction** (if no `target`): The source audio's voice characteristics are analyzed
5. **Re-Synthesis**: Qwen3-TTS Base synthesizes the English text with the extracted or target voice
6. **Overdose Post-Processing** (optional): When `tts overdose slc` is used, Seed-VC v2 runs a non-mimic pass after TTS output for better voice preservation

**Two Modes of Operation:**

| Mode | Parameter | Result |
|------|-----------|--------|
| **Same-Voice Translation** | No `target` | Translated in the original speaker's voice |
| **Different-Voice Translation** | With `target` | Translated in a different person's voice |

**Language Preservation Trick:** To preserve specific words or phrases in the original language (e.g., names, technical terms, cultural expressions), wrap them in `{original}` markers within the translation. The translator preserves text inside `{ }` braces without translation.

**Command Catalog:**

```bash
# Same-voice translation (preserves original speaker's voice)
python src/voder.py tts slc "foreign_speech.wav"

# Different-voice translation
python src/voder.py tts slc "foreign_speech.wav" target "english_voice.wav"

# From video file (audio auto-extracted)
python src/voder.py tts slc "foreign_movie.mp4" target "dub_actor.wav" result "/output/dubbed.mp4"

# From YouTube URL
python src/voder.py tts slc "https://www.youtube.com/watch?v=VIDEO_ID"

# With output routing
python src/voder.py tts slc "spanish_interview.mp3" target "narrator.wav" result "/output/english_version.wav"

# Same-voice with language preservation
python src/voder.py tts slc "japanese_speech.wav" result "/output/english_dub.wav"

# Overdose SLC — runs STS v2 non-mimic pass after TTS for better voice preservation
python src/voder.py tts overdose slc "foreign_speech.wav"

# Overdose SLC with custom target voice
python src/voder.py tts overdose slc "foreign_speech.wav" target "voice_ref.wav"

# Translate only (just get the English text, no synthesis)
python src/voder.py tts slc translate "foreign_speech.wav"

# Overdose SLC with translate
python src/voder.py tts overdose slc translate "foreign_speech.wav"
```

**SLC Parameter Reference:**

| Parameter | Required | Purpose | Default |
|-----------|----------|---------|----------|
| `target` | No | Voice reference for dubbing voice | (uses source speaker's voice) |
| `translate` | No | Translate only (no synthesis, output English text) | Off |
| `overdose` | No | Runs STS v2 non-mimic pass after TTS for better voice preservation | Off |
| `result` | No | Output destination | Auto-generated |

**Limitations:**
- Source language is auto-detected; output is always **English**
- Same-voice quality depends on how distinct the source voice features are
- Very short audio segments (< 3 seconds) may produce lower quality voice matching
- Heavy background noise reduces voice extraction accuracy

### Interactive Modify-Speech (formerly STT+TTS)

In TTS interactive mode (via `python src/voder.py cli`), you can provide audio, video, or a URL and choose to **modify speech**. This replaces the old standalone STT+TTS mode. The flow is:

1. **Provide audio/video/URL input**
2. **SVS Isolation**: BS-RoFormer isolates the voice from the source
3. **Whisper Transcription**: The isolated voice is transcribed to text
4. **Edit Text**: You review and edit the transcription (fix errors, change words, modify content)
5. **Choose Voice**: Use the source speaker's voice or provide a custom voice reference
6. **Qwen-TTS Synthesis**: The edited text is synthesized with the chosen voice

```bash
# Interactive mode
python src/voder.py cli
# Then select TTS from the menu
# When prompted "modify speech? (Y/N)", choose Y
# Provide audio/video/URL source
# Edit the transcribed text
# Choose voice (source or custom)
```

### Voice Stabilization

VoiceDesign characters in dialogue mode automatically get their voice stabilized. After 3 script lines, the outputs are concatenated, SVS-cleaned, and fed to Qwen3-TTS Base for voice extraction. All subsequent lines use the cloned voice instead of VoiceDesign, eliminating vocal drift in long dialogues. This happens automatically — no configuration is needed.

---

## 2.1a Voice Training (train voice)

Train a Qwen3-TTS Base voice clone from reference audio and save it as a `.tts` file for later reuse. Oneline-only command.

**Command Syntax:**

```bash
python src/voder.py train voice:character-name "path1" "path2" ...
```

- `character-name` is the name used to reference the trained voice later
- One or more audio file paths provide the reference audio for training
- Multiple paths are SVS-cleaned individually and concatenated into a composite before voice extraction
- Add the `first` keyword before the paths to extract only the first reference's speaker from all others via TSE: `train voice:name first "ref1.wav" "ref2.wav"` — the first reference identifies the target speaker, and TSE extraction pulls that speaker's voice from the remaining references before compiling
- The trained voice is saved as `voder_tts_character-name_timestamp.tts` in the `voices/` directory

**Optional Test Sample:**

- Add `test` at the end to generate a test sample using a hardcoded 30+ second script:
  ```bash
  python src/voder.py train voice:my-character "ref1.wav" "ref2.wav" test
  ```

- Add `test "custom script"` to use a custom test script:
  ```bash
  python src/voder.py train voice:my-character "ref1.wav" test "Custom test script for verification"
  ```

**Examples:**

```bash
# Train from single reference
python src/voder.py train voice:narrator "narrator_ref.wav"

# Train from multiple references
python src/voder.py train voice:hero "hero_clip1.wav" "hero_clip2.wav" "hero_clip3.wav"

# Train with test sample
python src/voder.py train voice:narrator "ref.wav" test

# Train with custom test script
python src/voder.py train voice:narrator "ref.wav" test "The quick brown fox jumps over the lazy dog."
```

### Voice Prompt Syntax

Voice prompts are natural language descriptions. The model extracts semantic meaning, so order doesn't matter:

```
"adult male, deep voice, authoritative tone, British accent, measured pace"
"young female, energetic, fast-paced, cheerful, American accent"
"elderly male, gravelly voice, slow and deliberate, storytelling quality"
```

**Effective Elements to Include:**
- **Age**: young adult, middle-aged, elderly
- **Gender**: male, female, androgynous
- **Tone**: warm, cold, friendly, authoritative, dramatic
- **Pace**: fast-paced, measured, slow, deliberate
- **Quality**: clear, gravelly, breathy, resonant
- **Accent**: British, American, Southern, neutral
- **Context**: professional, casual, broadcast, conversational

### Reference Audio Requirements (Voice Cloning)

| Factor | Requirement | Why |
|--------|-------------|-----|
| **Duration** | 10-30 seconds optimal | Enough data for voice extraction; longer doesn't help |
| **Quality** | Clear, minimal noise | Noise interferes with voice feature extraction |
| **Content** | Continuous speech | Silence or music doesn't contribute voice data |
| **Speaker** | Single speaker only | Mixed speakers confuse the extraction |
| **Format** | WAV preferred, MP3 supported | WAV preserves audio fidelity |

> **Pro Tip**: Run noisy reference audio through SE (Speech Enhancement) before using for voice cloning. VODER can also use BS-RoFormer to extract clean vocals from a mixed recording before cloning.

### Voice Consistency in Dialogue
VODER extracts voice characteristics **once per character** at the start of dialogue processing. This means:
- All lines from "James" use the same extracted voice profile
- No variation between the 1st and 10th line of the same character
- Professional-quality consistency throughout long dialogues

**Voice Stabilization:** VoiceDesign characters in dialogue mode automatically get their voice stabilized after 3 lines. The outputs are concatenated, SVS-cleaned, and fed to Qwen3-TTS Base for voice extraction. All subsequent lines use the cloned voice instead of VoiceDesign, eliminating vocal drift. This is automatic — no configuration needed.

---

## 2.2 STS (Speech-to-Speech Voice Conversion)

### What It Is
STS mode transforms the **voice** in source audio to sound like a different person, while preserving **everything else** — the words, emotion, timing, prosody, pauses, and delivery style. Only the speaker identity changes.

STS supports audio **and video** input/output. When given a video file, the audio track is extracted, processed, and optionally re-attached to the video.

### How It Works
1. **Source Separation**: BS-RoFormer extracts vocals and music from source audio; only vocals go to the VC model
2. **Target Extraction**: Clean vocals are extracted from the target reference
3. **Content Extraction**: Seed-VC extracts the linguistic and prosodic content from source vocals (what was said, how it was said)
4. **Voice Extraction**: The target voice reference is analyzed for speaker characteristics
5. **Voice Transfer**: The content is re-synthesized with the target voice characteristics
6. **Recombination**: Converted vocals are mixed back with the source music (unless `nomusic` is used)
7. **Output Assembly**: Audio is written (or re-attached to video container)
8. **Sample Rate Handling**: v2 model outputs at 22.05kHz (speech), v1 at 44.1kHz (music)

### Auto Vocal Extraction
VODER automatically runs BS‑RoFormer vocal isolation on both the source and target audio. For the source, vocals are separated so the VC model processes only the voice — producing cleaner conversion — and the instrumental is extracted separately for recombination after conversion (unless `nomusic` is used). For the target, clean vocals are extracted to improve cloning quality. If SVS extraction fails, the original audio is used as a fallback. This is particularly useful for:
- Converting vocals in mixed audio without degrading the music
- Cleaning up recordings before conversion
- Ensuring the VC model receives the cleanest possible voice input

### STS vs TTS: When to Use Which

| Scenario | Use STS When... | Use TTS When... |
|----------|-----------------|-----------------|
| Input | You have audio you want to preserve | You have text you want to speak |
| Delivery | You want to keep original emotion/timing | You want fresh synthesis |
| Content | Content is fixed (what was said) | You can edit the text |
| Source | Performance matters (acting, singing) | Text-only workflow |

### Command Catalog

#### Standard Voice Conversion (Speech)
```bash
# Basic command
python src/voder.py sts base "source_audio.wav" target "voice_reference.wav"

# With output routing
python src/voder.py sts base "source.wav" target "voice.wav" result "/output/converted.wav"

# From video file (audio auto-extracted, output re-attached to video)
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" result "/output/output.mp4"

# Audio-only output from video
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" result "/output/output.wav"
```

#### MSTS (Music Voice Conversion)
```bash
# For songs/musical content - uses 44.1kHz model
python src/voder.py sts base "song.wav" target "singer_voice.wav" music

# Convert singing voice in a song
python src/voder.py sts base "original_song.wav" target "new_singer.wav" music result "/output/cover.wav"

# From video (music video voice conversion)
python src/voder.py sts base "music_video.mp4" target "new_singer.wav" music result "/output/cover.mp4"
```

#### Mimic (Style Transfer)
```bash
# Transfer voice timbre AND accent/emotion/style from target
python src/voder.py sts base "source.wav" target "character.wav" mimic

# This is invalid - mimic and music cannot be combined
python src/voder.py sts base "source.wav" target "reference.wav" mimic music
# Error: music and mimic cannot be used together
```

**Mimic Language Quality Note**: When using `mimic` for cross-language voice conversion (e.g., converting Spanish speech to an English speaker's voice), quality may vary. Mimic transfers timbre and style but does not translate content. For language conversion, use `tts slc` instead.

#### nomusic (Voice-Only Output)
```bash
# Output only the converted voice without mixing back source music
python src/voder.py sts base "song.wav" target "singer.wav" nomusic

# This is invalid - nomusic and music cannot be combined
python src/voder.py sts base "song.wav" target "singer.wav" nomusic music
# Error: nomusic cannot be used with music
```

#### Multi-Reference Target (Oneline Only)
```bash
# Multiple voice references concatenated for richer cloning
python src/voder.py sts base "source.wav" target "(voice1.wav)(voice2.wav)(voice3.wav)"

# Multi-reference with MSTS
python src/voder.py sts base "song.wav" target "(singer1.wav)(singer2.wav)" music

# Multi-reference with first keyword (extract only first ref's speaker from all others via TSE)
python src/voder.py sts base "source.wav" target first "(voice1.wav)(voice2.wav)(voice3.wav)"
```

### Model Selection

| Flag | Model | Sample Rate | Use Case |
|------|-------|-------------|----------|
| (none) | Seed-VC v2 | 22.05kHz | Speech, podcasts, interviews (music auto-recombined) |
| `music` | Seed-VC v1 | 44.1kHz | Songs, musical content, singing |
| `nomusic` | Seed-VC v2 | 22.05kHz | Voice-only output (no music recombination) |
| `mimic` | Seed-VC v2 (AR+CFM) | 22.05kHz | Style transfer (voice + accent + delivery) |

### Video I/O Support

| Input | Output | Behavior |
|-------|--------|----------|
| Audio (WAV, MP3, FLAC) | Audio (WAV) | Standard processing |
| Video (MP4, MKV, AVI) | Audio (WAV) | Audio extracted, processed, output as audio |
| Video (MP4, MKV, AVI) | Video (MP4) | Audio extracted, processed, re-attached to original video |

> **Tip**: The output format (audio vs video) is determined by the `result` file extension. Use `.wav` for audio-only, `.mp4` for video output.

---

## 2.3 TTM (Text-to-Music Generation with Optional Voice Cloning)

### What It Is
TTM mode generates **complete musical compositions** from lyrics and style descriptions using the ACE-Step model family. The model creates both the instrumental arrangement AND the vocal performance. You provide lyrics, describe the musical style, specify duration, and receive a fully produced song.

With the `vc` flag and `clone` parameter, TTM also supports **voice cloning** — generating music where the vocalist sounds like a specific real person.

> **Note**: The old `ttm+vc` command is no longer accepted. Use `ttm vc` with `clone "path"` for voice clone source. The `target` parameter is reserved for optional music references (`target voice "path"` / `target music "path"`).

### How It Works
1. **Lyrics Processing**: Lyrics are parsed into vocal melody and rhythm
2. **Style Interpretation**: Style prompt guides instrumentation, genre, mood, tempo
3. **Music Generation**: ACE-Step model creates aligned instrumental and vocal tracks
4. **Duration Matching**: Output is stretched/compressed to hit target duration
5. **Optional Voice Conversion** (with `vc` flag): ACE-Step is offloaded from memory, Seed-VC converts the vocal track to match reference voice, converted vocals are mixed back with instrumental

### Three-Tier ACE-Step System

TTM mode automatically selects the best ACE-Step model based on the task and flags:

| Tier | Model | Quality | Speed | When Used |
|------|-------|---------|-------|-----------|
| **Turbo** | ACE-Step XL-Turbo | Highest | Slowest | `overdose` flag — maximum quality generation |
| **Base** | ACE-Step XL-Base | High | Medium | `complete`, `extract`, `lego` sub-tasks |
| **Legacy** | ACE-Step 1.5 | Standard | Fastest | Default (no `task` specified), background music in dialogue |

### Sub-Tasks

TTM supports multiple sub-tasks via the `task` parameter:

| Sub-Task | CLI Keyword | Description | Model |
|----------|------------|-------------|-------|
| **Complete** | `complete` | Add missing tracks to existing audio; supports optional `styling` prompt, `noblend` flag, `voice`/`music` isolation, `usrc` blend source, and `sfx:` overlay specs | XL-Base (+ SVS if voice/music + TangoFlux for SFX) |
| **Lego** | `lego` | Build/generate individual instrument tracks; supports optional `styling` prompt | XL-Base |
| **Extract** | `extract` | Extract individual tracks from audio | XL-Base |
| **Remix** | `remix` | Style transfer (cover) with bias control; supports `voice`/`music` source isolation per entry (up to 3 sources), optional `lyrics` for new vocal content, and `reference` for additional guidance (up to 3 references composed into 30s composite) | XL-Turbo (overdose) or Legacy (+ SVS if voice/music) |
| **Repaint** | `repaint` | Restyle a specific time range of a song; supports `reference` for additional guidance; optional `voice`/`music` prefix on source for SVS isolation; multi-pass mode for sequential edits building on each previous result | XL-Turbo (overdose) or Legacy (+ SVS if voice/music prefix) |
| **BGM** | `bgm` | Replace background music in existing audio/video; strips music, generates new bgm, mixes at level; supports `video` flag, `reference`, and `sfx:` overlay specs | 1.5 Turbo (standard) or XL-Turbo (overdose) (+ TangoFlux for SFX) |
| **Overdose** | (flag) | Maximum quality full generation | XL-Turbo |

### 12 Instrument Tracks

The 12 available tracks are used by `lego`, `extract`, and `complete` sub-tasks:

| Track | Name | Description |
|-------|------|-------------|
| `drums` | Drums | Drum kit, percussion backbone |
| `bass` | Bass | Bass guitar, synth bass, upright bass |
| `guitar` | Guitar | Electric guitar (lead/rhythm) |
| `keyboard` | Keyboard | Piano, organ, synthesizer keys |
| `strings` | Strings | Violin, cello, string ensemble |
| `brass` | Brass | Trumpet, trombone, horn section |
| `woodwinds` | Woodwinds | Flute, clarinet, saxophone |
| `synth` | Synthesizer | Synth leads, pads, arpeggios |
| `percussion` | Percussion | Hand percussion, shakers, congas |
| `fx` | FX / Sound Design | Sound effects, textures, atmospheric elements |
| `vocals` | Vocals | Lead vocal track |
| `backing_vocals` | Backing Vocals | Background vocals, harmonies |

**Shortcuts**:
- `everything` = all 12 tracks
- `instruments` = first 10 (drums through fx, non-voice)
- `voices` = `vocals` + `backing_vocals`

### Unique Capability: Instrumental-Only
Using `"..."` as lyrics generates **purely instrumental music** with no vocals. This is how background music for dialogue is created internally.

### Command Catalog

#### Standard Song Generation
```bash
# With lyrics (song with vocals) — uses ACE-Step 1.5 (legacy, fast)
python src/voder.py ttm lyrics "Verse 1:\nLyrics here\n\nChorus:\nChorus lyrics" styling "pop, upbeat, female vocals" duration 60

# Instrumental only (no vocals)
python src/voder.py ttm lyrics "..." styling "cinematic orchestral, dramatic" duration 90

# With output routing
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill" duration 120 result "/output/background.wav"

# Short jingle
python src/voder.py ttm lyrics "..." styling "upbeat corporate, bright" duration 15 result "/output/jingle.wav"
```

#### Overdose Mode (Maximum Quality)
```bash
# Overdose: highest quality generation using ACE-Step XL-Turbo
python src/voder.py ttm lyrics "Verse 1:\nLyrics here\n\nChorus:\nChorus lyrics" styling "pop, upbeat, female vocals" duration 60 overdose

# Overdose instrumental
python src/voder.py ttm lyrics "..." styling "cinematic orchestral, dramatic" duration 90 overdose result "/output/high_quality.wav"
```

#### Sub-Task Commands
```bash
# Complete: add missing tracks to existing audio
python src/voder.py ttm complete "base_track.wav" add "drums bass" styling "rock ballad" result "/output/completed.wav"

# Complete with noblend (generated instruments only, no blending with original)
python src/voder.py ttm complete noblend "base_track.wav" add "drums bass" result "/output/instruments_only.wav"

# Complete with SFX overlay only (no instrument addition, ACE-Step not loaded)
python src/voder.py ttm complete "voiceover.wav" sfx:"thunder rumble/10-5/60" sfx:"rain patter/30-0/30" result "/output/atmosphere.wav"

# Complete with instruments and SFX overlay
python src/voder.py ttm complete "base_track.wav" add "drums bass" sfx:"cymbal crash/5-15/70" result "/output/enhanced.wav"

# Complete with SFX overlay is invalid with noblend
# python src/voder.py ttm complete noblend "base.wav" sfx:"boom/5-0/50"
# Error: sfx: cannot be used with noblend

# Complete with voice isolation (SVS pre-extract vocals, blend with vocals)
python src/voder.py ttm complete voice "song.wav" add "drums bass" result "/output/voice_completed.wav"

# Complete with music isolation (SVS pre-extract instruments, blend with instruments)
python src/voder.py ttm complete music "song.wav" add "everything" result "/output/music_completed.wav"

# Complete with voice + usrc (blend with original source instead of isolated vocals)
python src/voder.py ttm complete voice usrc "song.wav" add "drums bass guitar" result "/output/voice_usrc_completed.wav"

# Complete with music + usrc (blend with original source instead of isolated instruments)
python src/voder.py ttm complete music usrc "song.wav" add "everything" result "/output/music_usrc_completed.wav"

# Lego: build/generate individual instrument tracks
python src/voder.py ttm lego "..." make "drums bass strings" styling "jazz trio" duration 120 result "/output/stems.wav"

# Extract: extract individual tracks from audio
python src/voder.py ttm extract "existing_song.wav" stems "vocals drums bass" result "/output/extracted/"

# Remix: style transfer (cover) with bias control
python src/voder.py ttm remix "input.wav" styling "jazz" bias 40 result "/output/remix.wav"

# Remix with custom lyrics (optional lyrics for new vocal content)
python src/voder.py ttm remix "input.wav" lyrics "new verse words" styling "jazz" result "/output/remix.wav"

# Remix with reference (voice extraction from reference for guidance)
python src/voder.py ttm remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"

# Remix with reference (music extraction from reference)
python src/voder.py ttm remix "input.wav" styling "jazz" reference music "ref.wav" result "/output/remix.wav"

# Remix with reference (used as-is, no extraction)
python src/voder.py ttm remix "input.wav" styling "jazz" reference "ref.wav" result "/output/remix.wav"

# Overdose remix with reference
python src/voder.py ttm overdose remix "input.wav" styling "jazz" reference voice "ref.wav" result "/output/remix.wav"

# Overdose remix with lyrics
python src/voder.py ttm overdose remix "input.wav" lyrics "dreamy verse lines" styling "synthwave" result "/output/remix.wav"

# Remix vocals only (SVS pre-extract vocals from source)
python src/voder.py ttm remix voice "song.wav" styling "soulful R&B" result "/output/voice_remix.wav"

# Remix music only (SVS pre-extract instruments from source)
python src/voder.py ttm remix music "song.wav" styling "electronic synth" result "/output/music_remix.wav"

# Overdose remix with voice isolation
python src/voder.py ttm overdose remix voice "song.wav" styling "cinematic orchestral" result "/output/voice_od_remix.wav"

# Multi-source remix (vocals from one song + instruments from another)
python src/voder.py ttm remix voice "vocals.wav" music "instruments.wav" styling "funk" bias 60 result "/output/multi_remix.wav"

# Multi-reference remix (2 references composed into 30s composite)
python src/voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav" result "/output/remix.wav"

# Multi-reference remix (3 references)
python src/voder.py ttm remix "song.wav" styling "rock" reference "ref1.wav" voice "ref2.wav" music "ref3.wav" result "/output/remix.wav"

# Repaint: restyle a specific time range of a song
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"

# Repaint with voice/music isolation on source
python src/voder.py ttm repaint voice "source.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"
python src/voder.py ttm repaint music "source.wav" time:20-80 styling "ambient" result "/output/repainted.wav"

# Repaint with reference (voice extraction from reference for guidance)
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" reference voice "ref.wav" result "/output/repainted.wav"

# Repaint with reference (used as-is)
python src/voder.py ttm repaint "source.wav" time:20-80 styling "more energetic" reference "ref.wav" result "/output/repainted.wav"

# Overdose repaint with reference
python src/voder.py ttm overdose repaint "source.wav" time:20-80 styling "more energetic" reference music "ref.wav" result "/output/repainted.wav"

# Multi-pass repaint (each pass builds on the previous result)
python src/voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"
python src/voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"
python src/voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"
```

#### Voice Cloning (VC)
```bash
# Generate song with cloned vocalist (vc flag + clone)
python src/voder.py ttm vc lyrics "Verse 1:\nMy lyrics here" styling "rock ballad, emotional" duration 60 clone "singer_reference.wav"

# Instrumental backing + cloned voice
python src/voder.py ttm vc lyrics "..." styling "acoustic guitar backing" duration 180 clone "voice.wav" result "/output/backing.wav"

# With output routing
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "pop anthem" duration 45 clone "artist.wav" result "/output/song.wav"

# With optional music reference
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "pop" duration 30 clone "singer.wav" target music "backing_track.wav"

# With multi-reference clone (oneline only)
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "pop" duration 30 clone "(voice1.wav)(voice2.wav)(voice3.wav)"

# With multi-reference clone + first keyword (extract only first ref's speaker from all others via TSE)
python src/voder.py ttm vc lyrics "Chorus:\nThis is our moment" styling "pop" duration 30 clone first "(voice1.wav)(voice2.wav)(voice3.wav)"
```

#### Maximum TTM VC Command
```bash
python src/voder.py ttm overdose vc lyrics "content" styling "prompt" duration 20 clone "path/link" target music "path/link" result "path"
```

#### BGM Sub-Task (Replace Background Music)
```bash
# Replace background music (standard quality, ACE-Step 1.5 Turbo)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30

# Replace background music (overdose quality, ACE-Step XL-Turbo)
python src/voder.py ttm overdose bgm "video.mp4" music "cinematic orchestral" level 50

# Replace background music with reference for style guidance
python src/voder.py ttm bgm "podcast.wav" music "upbeat electronic" level 35 reference "style_ref.wav"

# From YouTube URL with audio-only output
python src/voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient chill" level 25 result "/output/new_bgm.wav"

# From YouTube URL with video output (downloads video, replaces bgm, outputs .mp4)
python src/voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"

# BGM with SFX overlay (music + sound effects)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient" level 30 sfx:"doorbell/5-10/60" result "/output/podcast_sfx.wav"

# BGM with multiple SFX overlays
python src/voder.py ttm bgm "narration.wav" music "cinematic" level 40 sfx:"thunder rumble/10-5/70" sfx:"rain patter/30-0/25" result "/output/dramatic.wav"

# BGM with SFX overlay only (no music, SFX overlaid on clean voice)
python src/voder.py ttm bgm "voiceover.wav" sfx:"ocean waves/30-0/30" sfx:"seagull call/5-15/50" result "/output/ambient.wav"

# BGM with music, reference, and SFX
python src/voder.py ttm bgm "interview.wav" music "lo-fi beats" level 25 reference "ref.wav" sfx:"keyboard typing/10-30/20" result "/output/full.wav"
```

**BGM Pipeline:** Source → SVS voice pipe (strip existing music) → detect duration → [ACE-Step generate new bgm in 250-300s chunks if music provided] → [optional SVS music pipe on reference] → mix at level → [TangoFlux generate SFX → overlay SFX at position/level] → re-mux to video if needed

**BGM Output Naming:** `voder_ttm_bgm_{original-name}_{timestamp}.wav` (audio) or `.mp4` (video)

**BGM Key Rules:**
- `bgm` cannot be combined with `vc`, `remix`, `repaint`, `complete`, `lego`, or `extract`
- Source supports audio, video, and URL inputs
- `video` flag: when source is a YouTube URL, downloads the video file (not just audio) and merges the result back into .mp4. For local video files, video output is automatic (no flag needed). If `video` is used with an audio source, outputs .wav with a warning.
- Reference supports audio files, video files, and URLs — always processed through SVS music pipe for clean instrumental
- Normal uses ACE-Step turbo 1.5; overdose uses ACE-Step XL 1.5 turbo
- Default volume level is 35
- `music` is required unless `sfx:` specs are provided; if only `sfx:` specs are present, SFX is overlaid directly on the clean voice track (no BGM generation)
- `sfx:` overlay specs are overlaid after BGM mixing (or directly on clean voice if no music); SFX generated by TangoFlux with ACE-Step offloaded first
- Multiple `sfx:` specs are allowed; each generates and overlays an independent sound effect

### SFX Overlay Spec Format

SFX overlay specs allow sound effects to be overlaid onto the output of `bgm` and `complete` sub-tasks. Each spec follows the format:

```
sfx:"prompt/duration-position/level"
```

| Component | Required | Description | Validation |
|-----------|----------|-------------|------------|
| `prompt` | Yes | SFX description text (e.g., "thunder rumble", "doorbell") | Must be non-empty |
| `duration` | Yes | SFX length in seconds (5-30) | Auto-clamped: <5 -> 5, >30 -> 30 with warning; minus sign stripped; invalid -> error |
| `position` | Yes | Place SFX at N seconds into source | Non-negative; cannot exceed source duration; invalid -> error |
| `level` | No | Volume 1-100% | Default: 50; minus sign stripped; <1 -> warning -> 1; >100 -> warning -> 100; invalid -> error |

**Parsing rules:**
- The spec is split by `/` into components: `prompt/duration-position/level`
- `duration` and `position` are separated by `-` (dash) in the second component
- `level` is the optional third component
- Multiple `sfx:` specs can be specified independently

**Examples:**
```
sfx:"thunder rumble/10-5/70"        # 10s thunder at 5s in, volume 70%
sfx:"doorbell/5-30"                # 5s doorbell at 30s in, volume 50% (default)
sfx:"rain patter/30-0/25"          # 30s rain at start, volume 25%
sfx:"ocean waves/30-0/30"          # 30s ocean at start (clamped from 60 -> 30 with warning)
sfx:"click/5-10/150"               # 5s click at 10s, volume 100% (clamped from 150 with warning)
```

**Behavior by sub-task:**
- **BGM**: SFX is overlaid after BGM mixing (or directly on clean voice if no `music` provided); `music` becomes optional when `sfx:` specs are present; ACE-Step is offloaded before TangoFlux loads
- **Complete**: SFX is overlaid after the blend step; `add` becomes optional when `sfx:` specs are present; `sfx:` cannot be used with `noblend`; if only `sfx:` (no `add`), the music model (ACE-Step) is not loaded; ACE-Step is offloaded before TangoFlux loads

**Model handling:** SFX is generated by the TangoFlux model. When SFX overlay is triggered, ACE-Step is offloaded from memory before TangoFlux loads, ensuring efficient memory usage.

### Lyrics Format
```
Verse 1:
First line of verse
Second line of verse

Chorus:
Chorus lyrics here
More chorus lyrics

Verse 2:
Second verse content

Bridge:
Bridge section lyrics

Outro:
Final lines
```

### Style Prompt Guidelines

| Element | Examples |
|---------|----------|
| **Genre** | pop, rock, electronic, jazz, classical, hip-hop, folk |
| **Mood** | upbeat, melancholic, dramatic, peaceful, energetic |
| **Instrumentation** | piano and strings, heavy guitars, synthesizer, acoustic guitar |
| **Tempo** | slow ballad, mid-tempo, fast-paced |
| **Vocals** | female vocals, male vocals, choir, no vocals |

### Duration Considerations

| Duration | Best For | Quality |
|----------|----------|---------|
| 10-30s | Jingles, transitions, intros | Very consistent |
| 30-60s | Verses, choruses | Consistent |
| 60-120s | Complete short songs | Generally consistent |
| 120-300s | Full compositions | May have variation |

### Memory Optimization (Voice Clone Path)
The automatic model offloading between ACE-Step and Seed-VC stages means voice clone mode uses **less peak memory** than running TTM and STS separately.

### Parameter Reference

| Parameter | Required | Purpose | Default |
|-----------|----------|---------|---------|
| `lyrics` | Yes* | Song lyrics or `"..."` for instrumental; also optional for `remix` to guide new vocal content | — |
| `styling` | Yes** | Musical style description (optional for `complete`/`lego` sub-tasks) | — |
| `duration` | Yes** | Target duration in seconds | — |
| `clone` | No* | Voice clone source path (required when `vc` is set). Multi-reference format: `(path1)(path2)` concatenates multiple references into one. Add `first` keyword (`clone first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling | — |
| `target` | No | Music reference audio (optional, with type prefix: `target voice "path"` or `target music "path"`) | — |
| `remix` | No | Source audio for remix style transfer | — |
| `repaint` | No | Source audio for section repaint; optional `voice`/`music` prefix for SVS isolation | — |
| `time:start-end` | No† | Time range (for repaint, single-pass mode required) | — |
| `"start-end/styling(...)/..."` | No | Multi-pass repaint spec (quoted): time range required; optional `/styling(text)`, `/lyrics(text)`, `/reference-voice(path)`, `/reference-music(path)`, `/reference(path)` (up to 3 per pass), `/bias/nn`. Multiple pass specs = multiple sequential repaint passes. | — |
| `bias` | No | Cover strength 0-100 (for remix/repaint) | 40 |
| `reference` | No | Reference audio for remix/repaint/bgm guidance (up to 3 entries with `voice`/`music` prefix; `reference voice "path"`, `reference music "path"`, or `reference "path"` for as-is; multiple refs composed into 30s composite) | — |
| `complete` | No | Complete sub-task flag | Off |
| `lego` | No | Lego sub-task flag | Off |
| `extract` | No | Extract sub-task flag | Off |
| `add` | No*** | Instrument list for complete (e.g., `add "drums bass"`); optional if `sfx:` specs provided | All instruments |
| `make` | No | Instrument list for lego (e.g., `make "drums bass"`) | All instruments |
| `stems` | No | Instrument list for extract (e.g., `stems "vocals drums"`) | All stems |
| `only` | No | Extract single track only (no mix) | Off |
| `mix` | No | Mix extracted lego/extract tracks back | Off |
| `blend` | No | Blend mode for lego | — |
| `voice` | No | Use vocals category (for complete/lego) | Off |
| `music` | No | Use instruments category (for complete/lego) | Off |
| `video` | No | Output video (for complete/bgm) | Off |
| `noblend` | No | Output generated instruments only without blending with original (complete only); cannot be used with `sfx:` | Off |
| `bgm` | No | Replace background music in source (audio/video/URL) | — |
| `level` | No | Music volume for bgm sub-task (0-100) | 35 |
| `sfx:` | No | SFX overlay spec for bgm/complete (e.g., `sfx:"description/duration-position/level"`); multiple allowed; see SFX Overlay Spec section | — |
| `reference` | No | Reference audio/video/URL for remix/repaint/bgm guidance | — |
| `vc` | No | Enable voice cloning on vocalist | Off |
| `overdose` | No | Use XL-Turbo for maximum quality | Off |
| `result` | No | Output destination | Auto-generated |

*`clone` required when `vc` is set. `target` is optional music reference only.
**Required for generation tasks (default, overdose).
***`add` is required for `complete` unless `sfx:` specs are provided. If only `sfx:` is used (no `add`), ACE-Step is not loaded.
†Required when using `repaint` sub-task in single-pass mode (not needed for multi-pass mode where time range is embedded in pass specs).

---

## 2.4 STT (Speech-to-Text Transcription)

### What It Is
STT mode converts audio, video, images, and URLs into text. It uses Whisper for transcription and can optionally translate to English, identify **who spoke when** using Pyannote speaker diarization, or use VibeVoice ASR for advanced transcription with native diarization. This is the only mode that produces **text output** as its primary output (SS also produces text as a secondary output).

### How It Works
1. **Input Processing**: Audio extracted from video; text extracted from images via OCR; URLs downloaded via yt-dlp
2. **Pre-Cleanup** (optional): BS-RoFormer can separate vocals from music/noise before transcription for cleaner results
3. **Transcription**: Whisper transcribes with word-level timestamps (or VibeVoice ASR with `overdose`)
4. **Translation** (optional): With `translate` flag, Whisper large-v3 translates non-English audio to English
5. **Optional Diarization**: Pyannote identifies speaker segments (or VibeVoice with `overdose`)
6. **Alignment**: Transcription and diarization are aligned using three-tier overlap matching
7. **Output**: Text file saved to results/ directory

### Input Flexibility

| Input Type | How It's Processed |
|------------|-------------------|
| Audio file (WAV, MP3, FLAC, etc.) | Direct transcription |
| Video file (MP4, MKV, AVI, etc.) | Audio track extracted, then transcribed |
| Image file (PNG, JPG, etc.) | Text extracted via EasyOCR |
| YouTube URL | Audio downloaded via yt-dlp, then transcribed |
| Bilibili URL | Audio downloaded via yt-dlp, then transcribed |
| TikTok URL | Audio downloaded via yt-dlp, then transcribed |

### Flags: translate and overdose

STT supports two advanced flags that **cannot be used together** (mutually exclusive):

| Flag | Model Used | What It Does |
|------|-----------|--------------|
| `translate` | Whisper large-v3 | Transcribes AND translates non-English audio to English text |
| `overdose` | VibeVoice ASR | Advanced transcription with native speaker diarization built-in |

**`translate` flag**: Uses Whisper large-v3 (not turbo) for maximum translation accuracy. The output is English text regardless of the source language. Useful for subtitling foreign content, translating meetings, or processing multilingual media.

**`overdose` flag**: Uses VibeVoice ASR which provides superior transcription quality with **native speaker diarization** — no separate Pyannote step needed. Ideal for challenging audio (multiple speakers, overlapping speech, noisy environments). Note: `overdose` implies diarization; the `dialogue` flag is redundant and ignored when `overdose` is active.

**Mutual Exclusivity**: `overdose` and `translate` cannot be combined. If both are specified, an error is raised. Choose one based on your need:
- Need English text from foreign audio? → `translate`
- Need best transcription + speaker labels? → `overdose`
- Need standard transcription? → Neither flag (uses Whisper large-v3-turbo)

### SVS Pre-Cleanup
For audio with significant background music or noise, STT can internally use BS-RoFormer (SVS mode) to extract clean vocals before transcription. This is triggered automatically when the audio is detected to have high noise/music content, or can be manually invoked by running SVS first:
```bash
# Manual pre-cleanup: separate vocals, then transcribe
python src/voder.py svs "noisy_recording.wav" stem voice result "/clean/vocals.wav"
python src/voder.py stt "/clean/vocals.wav" timestamp
```

### Command Catalog

#### Basic Transcription
```bash
# Single audio file
python src/voder.py stt "audio.wav"

# Video file (audio auto-extracted)
python src/voder.py stt "video.mp4"

# Image file (OCR text extraction)
python src/voder.py stt "screenshot.png"

# YouTube URL
python src/voder.py stt "https://www.youtube.com/watch?v=VIDEO_ID"

# Bilibili URL
python src/voder.py stt "https://www.bilibili.com/video/BV1xx411c7mD"

# TikTok URL
python src/voder.py stt "https://www.tiktok.com/@user/video/123456789"
```

#### With Timestamps
```bash
python src/voder.py stt "audio.wav" timestamp
```

#### With Speaker Diarization
```bash
python src/voder.py stt "audio.wav" dialogue
```

#### With Translation
```bash
# Translate non-English audio to English
python src/voder.py stt "spanish_interview.mp3" translate

# Translate with timestamps
python src/voder.py stt "french_meeting.wav" translate timestamp

# Translate YouTube video
python src/voder.py stt "https://youtube.com/watch?v=VIDEO_ID" translate result "/output/english_transcript.txt"
```

#### With Overdose (VibeVoice ASR)
```bash
# Advanced transcription with native diarization
python src/voder.py stt "noisy_meeting.wav" overdose

# Overdose with timestamps
python src/voder.py stt "podcast_episode.wav" overdose timestamp

# Overdose for YouTube
python src/voder.py stt "https://youtube.com/watch?v=VIDEO_ID" overdose result "/output/overdose_transcript.txt"
```

#### Full Transcription
```bash
python src/voder.py stt "audio.wav" timestamp dialogue result "/output/transcript.txt"
```

#### Batch Processing
```bash
# Multiple files
python src/voder.py stt "file1.wav" "file2.mp3" "file3.mp4"

# Batch with timestamps and diarization
python src/voder.py stt "meeting1.wav" "meeting2.wav" timestamp dialogue result "/output/transcripts/"

# Batch with translation
python src/voder.py stt "spanish_ep1.wav" "spanish_ep2.wav" translate result "/output/translations/"
```

### Output Format Variations

| Flags | Output Format | Example |
|-------|---------------|---------|
| (none) | Plain text | `Hello everyone welcome to today's meeting` |
| `timestamp` | Timestamped segments | `[00:00.000 → 00:03.500] Hello everyone` |
| `dialogue` | Speaker-labeled | `Speaker 1: Hello everyone` |
| `timestamp dialogue` | Combined | `[00:00.000 → 00:03.500] Speaker 1: Hello everyone` |
| `translate` | English text | `Hello everyone welcome to today's meeting` (translated) |
| `translate timestamp` | Translated + timestamps | `[00:00.000 → 00:03.500] Hello everyone` |
| `overdose` | Enhanced + speaker-labeled | `Speaker 1 (00:00): Hello everyone` |
| `overdose timestamp` | Enhanced + timestamps + speakers | `[00:00.000 → 00:03.500] Speaker 1: Hello everyone` |

### HF_TOKEN Requirement
Speaker diarization (`dialogue` flag) requires:
1. HuggingFace account
2. Token from https://huggingface.co/settings/tokens
3. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-community-1
4. Token in `HF_TOKEN.txt` file or `HF_TOKEN` environment variable

> **Note**: `overdose` flag uses VibeVoice ASR and does NOT require HF_TOKEN for diarization. Use `overdose` if you don't have a HF_TOKEN but still need speaker identification.

---

## 2.5 SE (Speech Enhancement)

### What It Is
SE mode improves audio quality by removing noise, reducing reverberation, and restoring speech clarity. It's designed specifically for **speech content** — not music.

### How It Works
1. **Audio Analysis**: UniSE model separates speech from noise/reverb
2. **Noise Reduction**: Background noise is suppressed
3. **Dereverberation**: Room echo and reverb are reduced
4. **Restoration**: Speech frequencies are enhanced for clarity
5. **Output**: Clean audio at 16kHz sample rate

### What It Does NOT Do
- Cannot recover severely corrupted audio
- Not designed for music (will degrade musical content)
- Cannot fix very low sample rate recordings
- Cannot restore missing frequencies

### Command Catalog

```bash
# Basic enhancement
python src/voder.py se "noisy_audio.wav"

# From video file (enhanced audio re-attached to video)
python src/voder.py se "recording.mp4"

# Audio-only output from video
python src/voder.py se "recording.mp4" result "/output/clean.wav"

# With output routing
python src/voder.py se "audio.wav" result "/output/clean.wav"

# Enhance before using for voice cloning
python src/voder.py se "noisy_reference.wav" result "/clean/reference.wav"
```

### Best Use Cases
- Noisy meeting recordings
- Distant microphone recordings
- Room echo removal
- Pre-processing before voice cloning
- Cleaning up field recordings

---

## 2.6 SFX (Sound Effects Generation)

### What It Is
SFX mode generates custom sound effects from text descriptions using TangoFlux. Any sound you can describe, you can generate — natural sounds, mechanical sounds, ambient environments, impacts, transitions, sci-fi effects.

### How It Works
1. **Text Encoding**: The sound description is encoded into a semantic representation
2. **Diffusion Process**: Audio is generated through iterative denoising
3. **Duration Control**: Output is trimmed/looped to match requested duration
4. **Quality Scaling**: More steps = higher quality but slower generation

### Command Catalog

```bash
# Basic sound effect
python src/voder.py sfx sound "thunder rumbling in the distance" duration 10

# With quality parameters
python src/voder.py sfx sound "rain on a tin roof" duration 15 steps 50 guide 3.5

# With output routing
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"

# Short transition sound
python src/voder.py sfx sound "swoosh transition" duration 2 steps 20 result "/sfx/swoosh.wav"

# Ambient environment
python src/voder.py sfx sound "busy coffee shop with clinking cups and muffled conversations" duration 30 result "/sfx/cafe.wav"
```

### Parameter Reference

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `sound` | any text | required | Description of the sound |
| `duration` | 1-30 | required | Length in seconds |
| `steps` | 1-100 | 30 | Higher = better quality, slower |
| `guide` | 1.0-10.0 | 4.5 | Higher = stricter adherence to prompt |
| `result` | path | optional | Output destination |

### Sound Prompt Tips

| Sound Type | Prompt Strategy |
|------------|-----------------|
| Natural | Include environment: "rain on metal roof in a forest" |
| Impacts | Specify intensity and reverb: "heavy punch impact with long reverb tail" |
| Ambient | Layer elements: "forest at night with crickets and distant owl" |
| Transitions | Describe movement: "whoosh from left to right" |
| Mechanical | Include rhythm: "old clock ticking steadily" |
| Sci-fi | Mix familiar and unfamiliar: "futuristic laser with digital distortion" |

---

## 2.7 SVS (Source/Track Vocal Separation)

### What It Is
SVS mode separates mixed audio into individual stems using BS-RoFormer Resurrection. The most common separation is **vocals vs instrumental**, but the model can also separate other track components. SVS is also used **internally** by other modes: STS (auto vocal extraction before voice conversion), STT (pre-cleanup before transcription), and TTS (voice clone cleanup to extract clean vocals from mixed reference audio).

### How It Works
1. **Input Processing**: Audio extracted from video if needed; URLs downloaded via yt-dlp
2. **Stem Separation**: BS-RoFormer Resurrection analyzes the audio spectrogram and separates it into requested stems
3. **Output**: Individual stem files saved (or merged based on request)

### Supported Stems

| Stem Value | Output | Description |
|-----------|--------|-------------|
| `voice` | Vocal track | Isolated vocals, singing, speech |
| `music` | Instrumental track | Everything except vocals |
| `both` | Two files (sequential) | Extracts voice stem first, then music stem |

### YouTube URL Support
SVS can directly download and process audio from YouTube, Bilibili, and TikTok URLs. The audio is downloaded via yt-dlp and then separated.

### Command Catalog

```bash
# Separate vocals from instrumental (outputs both stems)
python src/voder.py svs "mixed_audio.wav"

# Get only the vocal stem
python src/voder.py svs "mixed_audio.wav" stem voice

# Get only the instrumental stem
python src/voder.py svs "mixed_audio.wav" stem music

# Extract both stems sequentially (voice first, then music)
python src/voder.py svs "mixed_audio.wav" stem both

# From video file (audio auto-extracted)
python src/voder.py svs "music_video.mp4" stem voice result "/output/vocals.wav"

# From YouTube URL
python src/voder.py svs "https://www.youtube.com/watch?v=VIDEO_ID"

# From YouTube with specific stem and output routing
python src/voder.py svs "https://www.youtube.com/watch?v=VIDEO_ID" stem music result "/output/instrumental.wav"

# With output routing
python src/voder.py svs "song.wav" result "/output/separated/"

# Batch processing
python src/voder.py svs "song1.wav" "song2.mp3" "song3.flac" result "/output/stems/"
```

### Parameter Reference

| Parameter | Required | Purpose | Default |
|-----------|----------|---------|---------|
| `stem` | No | Which stem to extract: `voice`, `music` | Both stems |
| `result` | No | Output destination | Auto-generated |

### Internal Usage by Other Modes

| Mode | How SVS Is Used |
|------|-----------------|
| STS | Extracts vocals and music from source (vocals for conversion, music for recombination), and clean vocals from target |
| STT | Pre-cleanup: separates vocals for cleaner transcription of music-heavy audio |
| TTS | When `target` reference contains background noise/music, extracts clean voice. Multi-reference targets (`(path1)(path2)`) are individually cleaned then concatenated. SLC sub-task: isolates voice from source audio/video/URL before translation |

### Best Use Cases
- Creating karaoke tracks (extract instrumental from songs)
- Isolating vocals for voice cloning reference
- Pre-cleaning audio before STT transcription
- Extracting acapella for remixing
- Podcast noise removal (separate speech from background)

---

## 2.8 SS (Speaker Separation)

### What It Is
SS mode takes multi-speaker audio and separates it into **individual audio files** — one per identified speaker, along with a full transcript. It uses VibeVoice ASR which provides **native speaker diarization** — identifying who spoke when and extracting each speaker's segments into separate files.

### How It Works
1. **Input Processing**: Audio extracted from video if needed; URLs downloaded via yt-dlp
2. **Speaker Identification**: VibeVoice ASR analyzes the audio and identifies distinct speakers
3. **Segment Extraction**: Each speaker's segments are extracted and concatenated into individual files
4. **Transcript Generation**: A full transcript with speaker labels and timestamps is generated
5. **Output**: Individual speaker audio files + combined transcript text file

### VibeVoice ASR Requirements

| Requirement | Details |
|------------|---------|
| **Model** | VibeVoice ASR (bundled with VODER) |
| **HF_TOKEN** | Not required (VibeVoice handles diarization natively) |
| **Audio Quality** | Clearer audio produces better separation |
| **Minimum Speakers** | 2 (single-speaker audio is returned as-is) |
| **Maximum Speakers** | No hard limit, but accuracy decreases beyond ~8 speakers |

### Fallback Behavior

If VibeVoice ASR fails or is unavailable, SS falls back to a two-step process:

1. **Pyannote** performs speaker diarization (identifies who spoke when)
2. **Audio segmentation** extracts each speaker's segments based on diarization timestamps

The fallback requires `HF_TOKEN` for Pyannote (see STT section for setup instructions). The fallback produces slightly lower quality segmentation because Pyannote only provides timestamps (not VibeVoice's enhanced speaker embeddings).

### Command Catalog

```bash
# Basic speaker separation
python src/voder.py ss "multi_speaker_audio.wav"

# From video file (audio auto-extracted)
python src/voder.py ss "panel_discussion.mp4" result "/output/speakers/"

# From YouTube URL
python src/voder.py ss "https://www.youtube.com/watch?v=VIDEO_ID"

# With output routing
python src/voder.py ss "podcast_episode.wav" result "/output/separated/"

# With timestamp flag (adds timestamps to transcript)
python src/voder.py ss "interview.wav" timestamp result "/output/interview/"

# Batch processing
python src/voder.py ss "ep1.wav" "ep2.wav" "ep3.wav" result "/output/all_episodes/"

# With overdose (VibeVoice ASR for higher quality)
python src/voder.py ss "meeting.wav" overdose
python src/voder.py ss "podcast.mp4" overdose result "/output/speakers/"
```

### Output Structure

When `result` is `/output/separated/`, SS creates:
```
/output/separated/
├── transcript.txt          # Full transcript with speaker labels
├── speaker_0.wav           # Speaker 0's audio segments concatenated
├── speaker_1.wav           # Speaker 1's audio segments concatenated
├── speaker_2.wav           # Speaker 2's audio segments concatenated
└── ...
```

The transcript format:
```
[00:00.000 → 00:05.200] Speaker 0: Welcome everyone to today's discussion.
[00:05.500 → 00:08.300] Speaker 1: Thank you for having me.
[00:09.000 → 00:15.100] Speaker 0: Let's start with the first topic.
[00:15.500 → 00:22.800] Speaker 2: I have some thoughts on that.
```

### Parameter Reference

| Parameter | Required | Purpose | Default |
|-----------|----------|---------|---------|
| `timestamp` | No | Include timestamps in transcript | Off (speaker labels only) |
| `result` | No | Output directory | Auto-generated |

### Best Use Cases
- Separating podcast guests for individual processing
- Extracting individual speaker audio for voice cloning references
- Pre-processing interviews before transcription
- Creating speaker-specific training data
- Analyzing multi-speaker recordings

---

# SECTION 3: SCRIPT DIRECTIVES SYSTEM

## What Script Directives Are

Script directives are special commands embedded **inside dialogue lines** that control how that specific line is processed. They allow fine-grained control over timing, volume, and duration at the **per-line level**.

## Why They Exist

Without directives, all dialogue lines are:
- Concatenated sequentially (no gaps)
- At uniform volume (100%)
- With duration determined by text length

Directives break these constraints, enabling:
- **Overlapping audio** (multiple lines at same time position)
- **Volume variation** (background lines at lower volume)
- **SFX duration control** (sound effects have fixed duration)
- **Audio layering** (SFX playing under speech)

## Directive Reference

| Directive | Format | Purpose | Applies To |
|-----------|--------|---------|------------|
| `/time:nn` | `/time:5` | Position line at 5 seconds from start | All lines |
| `/time:nn-nn` | `/time:10-3` | Position at 10s, cut 3s from end | All lines |
| `/time:nn+nn` | `/time:5+2` | Position at 5s, cut 2s from start | All lines |
| `/time:nn-nn+nn` | `/time:10-3+2` | Position at 10s, cut 3s from end AND cut 2s from start | All lines |
| `/level:0-100` | `/level:75` | Volume percentage for this line | All lines |
| `/duration:1-30` | `/duration:10` | Duration in seconds | SFX lines (required) |

## How Time Positioning Works

```
Without /time:              With /time:
┌────────────────────┐      ┌────────────────────┐
│ Line 1 (plays now) │      │ Line 1 /time:0     │
│ Line 2 (after 1)   │      │ Line 2 /time:0     │ ← overlaps with Line 1
│ Line 3 (after 2)   │      │ Line 3 /time:5     │ ← starts at 5 seconds
└────────────────────┘      └────────────────────┘
   Sequential                  Controlled positioning
```

## Deep Dive: /time: Syntax and Cutting

The `/time:` directive uses a flexible syntax that combines three operations in any order:

### Syntax Breakdown

```
/time:<position>[-<cut_from_end>][+<cut_from_start>]
```

- **Position (plain number)**: When the line should start (in seconds from the beginning of the output)
- **-nn (minus prefix)**: Cut this many seconds from the END of the generated audio
- **+nn (plus prefix)**: Cut this many seconds from the START (beginning) of the generated audio

### Understanding Cut Direction

The cutting terminology can be confusing. Here's how to think about it:

- **`-nn` (cut from end)**: Removes audio from the tail. Think of it as "trim off the last N seconds"
- **`+nn` (cut from start)**: Removes audio from the head. Think of it as "skip the first N seconds"

### Visual Examples

```
Original generated audio (10 seconds total):
┌────────────────────────────────────┐
│ 0s        5s        10s           │
│ [=========AUDIO CONTENT=========] │
└────────────────────────────────────┘

/time:5-3 (start at 5s, cut 3s from end):
              ┌──────────────┐
              │ 5s      7s   │  (plays 0s-7s of original, positioned at 5s in output)
              │ [=========]  │  (last 3 seconds removed)
              └──────────────┘

/time:5+2 (start at 5s, cut 2s from start):
              ┌──────────────────────┐
              │ 5s              13s  │
              │   [=============]    │  (first 2 seconds skipped, plays 2s-10s of original)
              └──────────────────────┘

/time:5-3+2 (start at 5s, cut 3s from end AND 2s from start):
              ┌────────────┐
              │ 5s     10s │
              │   [====]   │  (first 2s and last 3s removed, plays 2s-7s of original)
              └────────────┘
```

### Why Use Combined Cutting?

**Scenario 1: Remove intro/outro padding**
- Generated audio often has a slight intro breath or outro silence
- `/time:0-1+0.5` removes the half-second intro breath and 1-second outro tail

**Scenario 2: Tight dialogue timing**
- Two speakers' lines should slightly overlap for natural conversation flow
- Line 1: `"A: Hello there!" /time:0-0.5` (trim tail to make room)
- Line 2: `"B: Hi!" /time:1.5` (starts before Line 1 fully ends, creating overlap)

**Scenario 3: SFX that's too long**
- Generated SFX might be 10 seconds but you only need the middle section
- `"sfx: engine revving /duration:10 /time:0-2+1"` keeps seconds 1-8 (removes 1s intro, 2s outro)

### Practical Command Examples with Advanced Cutting

```bash
# Podcast intro: music fades in under host speech
python src/voder.py tts script \
  "sfx: upbeat podcast intro theme /duration:15 /level:40 /time:0-2" \
  "Host: Welcome back to the show! /time:2" \
  voice "Host: warm male voice"
# The SFX has its last 2 seconds trimmed so the transition feels cleaner

# Dialogue overlap for natural conversation
python src/voder.py tts script \
  "Alice: I was thinking about what you said... /time:0-0.8" \
  "Bob: And? /time:3.5" \
  "Alice: I think you're right. /time:4.5" \
  voice "Alice: female, thoughtful" "Bob: male, curious"
# Alice's first line is trimmed at the end, Bob's response starts before she fully finishes

# SFX with precise timing - remove intro breath and outro decay
python src/voder.py tts script \
  "sfx: thunder rumble /duration:8 /level:60 /time:5-2+1" \
  "Narrator: The storm was approaching. /time:0" \
  voice "Narrator: deep voice"
# Thunder starts at 5s mark, but we remove 1s intro and 2s outro, keeping the "meat" of the sound
```

## Command Examples

### Basic Time Positioning
```bash
python src/voder.py tts script \
  "Host: Welcome to the show! /time:0" \
  "sfx: intro music /duration:10 /level:40 /time:0" \
  "Host: Today we have a special guest. /time:10" \
  voice "Host: male broadcaster"
```

### Volume Control for Background Elements
```bash
python src/voder.py tts script \
  "Narrator: The scene opens on a quiet street. /level:100" \
  "sfx: distant traffic /duration:20 /level:20" \
  "Narrator: A car approaches slowly. /level:100" \
  "sfx: car engine /duration:5 /level:40" \
  voice "Narrator: deep male voice"
```

### Complex Layering
```bash
python src/voder.py tts script \
  "sfx: rain and thunder /duration:60 /level:30 /time:0" \
  "Character: What a terrible night... /time:5 /level:90" \
  "sfx: door creaking /duration:3 /level:50 /time:10" \
  "Character: Who's there? /time:13 /level:100" \
  voice "Character: nervous male voice" \
  music "tense atmospheric horror" level "25"
```

---

# SECTION 4: SFX LINES IN DIALOGUE

## What SFX Lines Are

SFX lines are a special type of dialogue line where the "character" is `sfx:` (case-insensitive). Instead of speech synthesis, VODER generates a sound effect matching the description.

## Why This Integration Matters

Before SFX lines, you had to:
1. Generate dialogue audio
2. Generate SFX audio separately
3. Use audio editing software to mix them
4. Manually align timing and adjust volumes

With SFX lines, everything happens in **one command** — VODER generates speech and SFX, positions them correctly, adjusts volumes, and produces the final mixed output.

## Syntax

```
"sfx: sound description /duration:nn /level:nn"
```

**Required:**
- Character must be `sfx:` (case-insensitive)
- `/duration:nn` must be present (1-30 seconds)

**Optional:**
- `/level:nn` for volume (0-100, default 100)
- `/time:nn` for positioning

## Command Examples

### Simple SFX Insertion
```bash
python src/voder.py tts script \
  "James: Hello, who's at the door?" \
  "sfx: door bell ringing /duration:3" \
  "Sarah: That must be the pizza!" \
  voice "James: male" "Sarah: female"
```

### SFX with Volume Control
```bash
python src/voder.py tts script \
  "Narrator: The forest was alive with sounds." \
  "sfx: birds chirping and rustling leaves /duration:15 /level:30" \
  "Narrator: But something else was watching." \
  voice "Narrator: deep male storytelling voice"
```

### SFX with Time Positioning (Layering)
```bash
python src/voder.py tts script \
  "sfx: ambient cafe noise /duration:60 /level:25 /time:0" \
  "Barista: What can I get you today? /time:5" \
  "Customer: I'll have a large coffee, please. /time:8" \
  "sfx: coffee machine grinding /duration:5 /level:40 /time:12" \
  "Barista: Coming right up! /time:18" \
  voice "Barista: cheerful female" "Customer: casual male"
```

---

# SECTION 5: CROSS-USE FEATURE

## What Cross-Use Is

Cross-use allows mixing **generated voices** (via `voice` parameter) and **cloned voices** (via `target` parameter) in the **same dialogue**. This works in TTS mode (which now includes voice cloning via `target`).

## Why This Matters

Before the TTS merge, cross-use required switching between TTS and TTS+VC modes. Now everything is in one mode:

- Some characters with designed voices (`voice`), others with cloned voices (`target`)
- Perfect for scenarios where you have reference audio for some speakers but not others
- Mix known voices with new character voices

## Rules

1. Each character must use EITHER `voice` OR `target`, not both
2. Character names must match between script and parameter
3. Case-insensitive matching (James = james = JAMES)

## Command Examples

### One Generated, One Cloned
```bash
python src/voder.py tts script \
  "James: Welcome to our podcast!" \
  "Sarah: Thanks for having me!" \
  voice "James: deep male voice, authoritative" \
  target "Sarah: /path/to/sarah_voice_reference.wav"
```

### TTS Voice Cloning Syntax (formerly TTS+VC, now merged into TTS)
```bash
# This now works in tts mode too — cross-use is the default behavior
python src/voder.py tts script \
  "James: Let me share my screen." \
  "Sarah: Go ahead, I'm ready." \
  target "James: /path/to/james_voice.wav" \
  voice "Sarah: bright female voice, enthusiastic"
```

### Three Characters: Mixed Approach
```bash
python src/voder.py tts script \
  "Host: Welcome to the debate!" \
  "Guest1: Thank you for having me." \
  "Guest2: Pleasure to be here." \
  voice "Host: professional broadcaster, neutral accent" \
  target "Guest1: /path/to/guest1.wav" "Guest2: /path/to/guest2.wav"
```

---

# SECTION 6: BACKGROUND MUSIC SYSTEM

## What Background Music Is

When using `music` parameter in dialogue mode, VODER automatically:
1. Generates all dialogue segments
2. Measures total dialogue duration
3. Creates music matching that exact duration
4. Mixes music at specified volume level
5. Outputs final file with `_m` suffix

## How It Works Internally

```
Dialogue Lines → Speech Synthesis → Concatenation → Duration Measurement
                                                          ↓
Music Description → ACE-Step 1.5 (lyrics: "...") → Duration-Matched Music
                                                          ↓
                                     Mix (Dialogue + Music at Level %)
                                                          ↓
                                          Final Output (_m suffix)
```

## Why Use Empty Lyrics

The `music` parameter internally uses `lyrics "..."` for ACE-Step, which tells the model to generate **instrumental-only music** with no vocals. This is specifically designed for background/ambient use. The legacy ACE-Step 1.5 model is used for background music generation because it is faster and sufficient for ambient/background quality.

## Level Parameter Syntax

| Format | Meaning | Use Case |
|--------|---------|----------|
| `"35"` | Constant 35% volume | Simple ambient background |
| `"50"` | Constant 50% volume | More prominent music |
| `"0:30-60:50"` | 30% at 0s, 50% at 60s | Fade in over time |
| `"0:50-30:20+10"` | Fade from 50% to 20% over 10s starting at 0s | Intro fade out |

## Command Examples

### Simple Background Music
```bash
python src/voder.py tts script \
  "Host: Welcome to our show!" \
  "Guest: Great to be here!" \
  voice "Host: male" "Guest: female" \
  music "soft jazz background"
```

### With Volume Control
```bash
python src/voder.py tts script \
  "A: Let's discuss the topic." \
  "B: I have some thoughts." \
  voice "A: male" "B: female" \
  music "ambient electronic, chill" \
  level "25"
```

### Time-Based Volume Changes
```bash
python src/voder.py tts script \
  "Intro: Welcome to the podcast!" \
  "Host: Today we'll explore..." \
  voice "Intro: energetic" "Host: professional" \
  music "upbeat intro music" \
  level "0:50-30:20"
```
# Music louder at start (50%), fades to quieter (20%) by 30 seconds

---

# SECTION 7: FEATURE COMBOS & ORDER RULES

## Understanding Feature Compatibility

Not all features work together. This section maps out exactly what combinations are possible and in what order parameters should appear.

## Mode-Feature Compatibility Matrix

| Feature | TTS | STS | TTM | STT | SE | SFX | SVS | SS |
|---------|-----|-----|-----|-----|-----|-----|-----|-----|
| Single mode | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dialogue mode | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SLC sub-task | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `voice` param | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `target` param | ✅ | ✅ | ✅† | ❌ | ❌ | ❌ | ❌ | ❌ |
| Cross-use | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `music` param | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `level` param | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SFX lines | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Script directives | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `timestamp` flag | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `dialogue` flag | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `translate` flag | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| `overdose` flag | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ |
| `clone` param | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `mimic` flag | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `vc` flag | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `music` flag (STS) | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `task` param (TTM) | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `stems` param | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `stem` param (SVS) | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| `steps` param | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `guide` param | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `result` param | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

*`clone` for TTM requires `vc` flag.
†`target` in TTM is optional music reference only (use `target voice` or `target music` prefix).

## Flag Exclusivity Rules

| Rule | Modes Affected | Details |
|------|---------------|---------|
| `overdose` XOR `translate` | STT | Cannot use both; choose one based on need |
| `overdose` XOR `dialogue` | STT | `overdose` includes native diarization; `dialogue` is redundant |
| `mimic` XOR `music` | STS | Cannot transfer style and switch to music model simultaneously |
| `remix` XOR `vc` | TTM | Remix and voice cloning are mutually exclusive |
| `sfx:` XOR `noblend` | TTM (complete) | SFX overlay cannot be used with noblend; noblend outputs generated instruments only, incompatible with overlay |

## Valid Parameter Orders

### TTS Mode
```
python src/voder.py tts [overdose] script "text" [script "text2" ...] [voice "prompt" [voice "prompt2" ...]] [target "path" [target "Char: path2" ...]] [music "description"] [level "spec"] [result "path"]
python src/voder.py tts slc [translate] "source_audio.wav" [target "voice.wav"] [result "path"]
python src/voder.py tts overdose slc [translate] "source_audio.wav" [target "voice.wav"] [result "path"]
```

### STS Mode
```
python src/voder.py sts base "source.wav" [source2.wav ...] target "voice.wav" [music] [mimic] [result "path"]
```

### TTM Mode
```
python src/voder.py ttm [lyrics "lyrics text"] styling "style prompt" duration N [vc] [clone "path"] [target music "path"] [overdose] [result "path"]
python src/voder.py ttm complete "source.wav" [add "instruments"] [noblend] [sfx:"prompt/duration-position/level" ...] styling "style" [result "path"]
python src/voder.py ttm lego "..." [make "instruments"] styling "style" duration N [result "path"]
python src/voder.py ttm extract "source.wav" [stems "instruments"] [result "path"]
python src/voder.py ttm remix [voice/music] "source.wav" [voice/music "source2.wav"] [lyrics "lyrics text"] styling "style" [bias N] [reference [voice/music] "ref.wav" [voice/music "ref2.wav"]] [result "path"]
python src/voder.py ttm repaint [voice/music] "source.wav" time:start-end styling "style" [bias N] [reference [voice/music] "ref.wav"] [result "path"]
python src/voder.py ttm repaint [voice/music] "source.wav" "start-end/styling(style)[/lyrics(text)][/reference-voice(path)][/reference-music(path)][/reference(path)][/bias/nn]" ["start-end/styling(style)/..." ...]
python src/voder.py ttm bgm "source.wav" [music "description"] level N [reference "path"] [sfx:"prompt/duration-position/level" ...] [video] [result "path"]
```

### STT Mode
```
python src/voder.py stt "file1" ["file2" ...] [timestamp] [dialogue] [translate] [overdose] [result "path"]
```

### SE Mode
```
python src/voder.py se "input.wav" [result "path"]
```

### SFX Mode
```
python src/voder.py sfx sound "description" duration N [steps N] [guide N.N] [result "path"]
```

### SVS Mode
```
python src/voder.py svs "input.wav" [stem voice|music] [result "path"]
```

### SS Mode
```
python src/voder.py ss "input.wav" ["input2.wav" ...] [timestamp] [result "path"]
```

## Feature Combo Catalog

### Combo 1: Dialogue + SFX + Background Music (Full Production)
**Mode**: TTS
**Features**: Dialogue mode + SFX lines + music param + level param
```bash
python src/voder.py tts script \
  "sfx: intro jingle /duration:5 /level:50 /time:0" \
  "Host: Welcome to our show!" \
  "sfx: applause /duration:3 /level:40 /time:3" \
  "Guest: Thanks for having me!" \
  voice "Host: male broadcaster" "Guest: female, enthusiastic" \
  music "upbeat podcast intro music" \
  level "0:50-30:30"
```

### Combo 2: Dialogue + Cross-Use + Background Music
**Mode**: TTS
**Features**: Dialogue mode + voice + target (cross-use) + music
```bash
python src/voder.py tts script \
  "James: Let's start the interview." \
  "Sarah: I'm ready when you are." \
  target "James: /path/to/james_voice.wav" \
  voice "Sarah: bright female voice" \
  music "soft ambient electronic"
```

### Combo 3: STT with Timestamps + Diarization + Result Routing
**Mode**: STT
**Features**: timestamp + dialogue + result
```bash
python src/voder.py stt "podcast_episode.wav" timestamp dialogue result "/output/transcripts/episode1.txt"
```

### Combo 4: STT with Translation
**Mode**: STT
**Features**: translate + timestamp + result
```bash
python src/voder.py stt "spanish_interview.mp3" translate timestamp result "/output/english_translation.txt"
```

### Combo 5: STT with Overdose
**Mode**: STT
**Features**: overdose + timestamp + result
```bash
python src/voder.py stt "noisy_panel.wav" overdose timestamp result "/output/overdose_transcript.txt"
```

### Combo 6: Batch STT with All Features
**Mode**: STT
**Features**: Multiple files + timestamp + dialogue + result
```bash
python src/voder.py stt "ep1.wav" "ep2.wav" "ep3.wav" timestamp dialogue result "/output/transcripts/"
```

### Combo 7: YouTube Transcription with Full Analysis
**Mode**: STT
**Features**: URL input + timestamp + dialogue
```bash
python src/voder.py stt "https://youtube.com/watch?v=VIDEO_ID" timestamp dialogue result "/output/video_transcript.txt"
```

### Combo 8: MSTS for Song Cover (Video I/O)
**Mode**: STS
**Features**: music flag + result (video output)
```bash
python src/voder.py sts base "original_song.mp4" target "new_singer_voice.wav" music result "/output/cover.mp4"
```

### Combo 9: TTM + Voice Clone (formerly TTM+VC)
**Mode**: TTM
**Features**: lyrics + styling + duration + vc + clone
```bash
python src/voder.py ttm vc lyrics "Verse 1:\nMy custom lyrics\n\nChorus:\nChorus text" styling "pop ballad, emotional" duration 90 clone "artist_voice.wav" result "/output/custom_song.wav"
```

### Combo 10: TTM Overdose (Maximum Quality)
**Mode**: TTM
**Features**: lyrics + styling + duration + overdose
```bash
# TTM overdose (highest quality music generation)
python src/voder.py ttm overdose lyrics "Verse:\nAmazing lyrics" styling "epic orchestral, cinematic" duration 120 result "/output/high_quality.wav"

# TTM overdose with voice cloning
python src/voder.py ttm overdose vc lyrics "Chorus:\nWe are one" styling "stadium rock" duration 30 clone "singer.wav"
```

### Combo 11: TTM Lego (Instrument Stems)
**Mode**: TTM
**Features**: lego + make + styling + duration
```bash
python src/voder.py ttm lego "..." make "keyboard bass drums saxophone" styling "jazz combo" duration 180 result "/output/jazz_stems.wav"
```

### Combo 12: SVS Pre-Cleanup + STT
**Mode**: SVS then STT (two commands)
**Features**: Vocal separation + transcription
```bash
python src/voder.py svs "noisy_recording.wav" stem voice result "/clean/vocals.wav"
python src/voder.py stt "/clean/vocals.wav" timestamp result "/output/clean_transcript.txt"
```

### Combo 13: SE Pre-processing + TTS Voice Cloning
**Mode**: SE then TTS (two commands)
**Features**: Enhancement + voice cloning
```bash
python src/voder.py se "noisy_reference.wav" result "/clean/reference.wav"
python src/voder.py tts script "Hello, this is a voice clone test." target "/clean/reference.wav" result "/output/cloned_speech.wav"
```

### Combo 14: SVS + STS (Clean Vocal Extraction + Voice Conversion)
**Mode**: SVS then STS (two commands)
**Features**: Vocal separation + voice conversion
```bash
python src/voder.py svs "mixed_song.wav" stem voice result "/clean/vocals.wav"
python src/voder.py sts base "/clean/vocals.wav" target "new_singer.wav" result "/output/converted.wav"
```

### Combo 15: SS + TTS (Speaker Separation + Re-synthesis)
**Mode**: SS then TTS (two commands)
**Features**: Speaker separation + voice cloning per speaker
```bash
python src/voder.py ss "interview.wav" result "/output/speakers/"
# Then clone each speaker's voice:
python src/voder.py tts script "Speaker 0's lines here..." target "/output/speakers/speaker_0.wav" voice "text: professional narrator" result "/output/narration.wav"
```

### Combo 16: SLC (Language Dubbing)
**Mode**: TTS SLC sub-task
**Features**: Foreign audio/video/URL + optional target voice
```bash
# Same-voice dubbing (preserves original speaker's voice)
python src/voder.py tts slc "french_interview.wav" result "/output/english_dub.wav"

# Different-voice dubbing
python src/voder.py tts slc "spanish_podcast.mp3" target "english_narrator.wav" result "/output/english_version.wav"

# From YouTube
python src/voder.py tts slc "https://youtube.com/watch?v=VIDEO_ID"

# Overdose SLC for better voice preservation
python src/voder.py tts overdose slc "foreign_speech.wav"
```

### Combo 17: Image-to-Audio Pipeline
**Mode**: STT then TTS (two commands)
**Features**: Image OCR + text-to-speech
```bash
python src/voder.py stt "script_screenshot.png" result "/output/extracted_text.txt"
# Parse the text file, then:
python src/voder.py tts script "[extracted text content]" voice "professional narrator" result "/output/audio.wav"
```

### Combo 18: Full Podcast Episode Production
**Mode**: TTS
**Features**: Dialogue + SFX + directives + music + level + result
```bash
python src/voder.py tts script \
  "sfx: podcast intro with music /duration:10 /level:60 /time:0" \
  "Host: Welcome to Tech Talk, episode forty-two! /time:0 /level:100" \
  "sfx: transition swoosh /duration:2 /level:40 /time:10" \
  "Host: Today we're diving deep into AI. /time:12" \
  "Guest: Excited to share my research! /time:18" \
  "sfx: typing on keyboard /duration:5 /level:25 /time:25" \
  "Host: Let's start with the basics. /time:30" \
  voice "Host: adult male, warm conversational, podcast style" "Guest: adult female, academic, clear pronunciation" \
  music "soft lo-fi beats, chill, minimal" \
  level "0:30-60:25-180:15" \
  result "/output/episode42.wav"
```

### Combo 19: TTS Overdose (Enhanced Dialogue Analysis + Music)
**Mode**: TTS
**Features**: overdose + voice cloning + music (XL Turbo)
```bash
# TTS overdose with voice design and enhanced music
python src/voder.py tts overdose script "James: Welcome to the show" "Sarah: Great to be here" voice "James: deep male" "Sarah: cheerful female" music "cinematic ambient" level "30" result "/output/podcast_hd.wav"

# TTS overdose with voice cloning from reference audio
python src/voder.py tts overdose script "Host: Let's dive in" "Guest: Absolutely" target "Host: host_ref.wav" "Guest: guest_ref.wav" music "soft jazz" result "/output/interview_hd.wav"
```

---

# SECTION 8: MEMORY REQUIREMENTS & SYSTEM PLANNING

## Memory by Mode

| Mode | RAM | VRAM (if GPU) | Notes |
|------|-----|---------------|-------|
| TTS — voice design (single/dialogue) | 12GB | 4GB | Qwen VoiceDesign model |
| TTS — voice clone (single/dialogue) | 12GB | 4GB | Qwen Base model |
| TTS + music | 23GB | 15-16GB | Adds ACE-Step 1.5 |
| TTS + overdose | 14GB | 4GB | Adds VibeVoice ASR for source analysis |
| TTS + overdose + music | 30GB | 22-24GB | VibeVoice ASR + ACE-Step XL Turbo |
| STS | 13GB | 14GB | Seed-VC (+ BS-RoFormer if auto-extract) |
| STS + video I/O | 13GB | 14GB | Same as STS, FFmpeg for muxing |
| TTM (legacy/1.5) | 23GB | 15-16GB | ACE-Step 1.5 |
| TTM (XL-Base sub-tasks) | 26GB | 18-20GB | ACE-Step XL-Base |
| TTM (XL-Turbo overdose) | 30GB | 22-24GB | ACE-Step XL-Turbo |
| TTM + vc (voice clone) | 26GB | 18-20GB | Auto-offloads between stages |
| STT | 12GB | N/A (CPU) | Whisper large-v3-turbo |
| STT + translate | 12GB | N/A (CPU) | Whisper large-v3 |
| STT + overdose | 14GB | N/A (CPU) | VibeVoice ASR |
| STT + diarization | 15GB | N/A (CPU) | Whisper + Pyannote |
| SE | 11GB | 4GB | UniSE |
| SFX | 12GB | 4GB | TangoFlux |
| SVS | 14GB | 8GB | BS-RoFormer |
| TTS slc | 14GB | 4GB | Whisper + Qwen3-TTS (+ SVS for voice isolation) |
| TTS slc overdose | 18GB | 12GB | Whisper + Qwen3-TTS + Seed-VC v2 (non-mimic) |
| SS | 14GB | N/A (CPU) | VibeVoice ASR |

## Planning Complex Workflows

### Workflow Memory Budget
When chaining operations, you don't need to sum all requirements — models are offloaded between operations. Plan for the **peak memory of the most demanding step**.

### Example: Podcast Production Pipeline
```
Step 1: STT (12GB peak) → offloaded
Step 2: TTS voice clone with music (23GB peak) → offloaded
Step 3: Done

Total memory needed: 23GB (not 35GB)
```

### Example: Song Cover Pipeline
```
Step 1: SE (11GB peak) → offloaded
Step 2: TTM + vc (26GB peak) → offloaded
Step 3: Done

Total memory needed: 26GB
```

### Example: Foreign Film Dubbing Pipeline
```
Step 1: SVS — extract vocals (14GB peak) → offloaded
Step 2: TTS slc — dub to English (14GB peak) → offloaded
Step 3: Done

Total memory needed: 14GB
```

### Example: Multi-Speaker Analysis Pipeline
```
Step 1: SS — separate speakers (14GB peak) → offloaded
Step 2: SVS — clean each speaker (14GB peak per file) → offloaded
Step 3: STT — transcribe each (12GB peak per file) → offloaded
Step 4: Done

Total memory needed: 14GB
```

---

# SECTION 9: TROUBLESHOOTING

| Issue | Cause | Solution |
|-------|-------|----------|
| Out of memory | Insufficient RAM/VRAM | Check requirements table; close other apps |
| FFmpeg not found | Missing system dependency | Install FFmpeg to PATH |
| Slow processing | CPU-only operation | Normal for CPU; GPU speeds up certain modes |
| Diarization fails | Missing/invalid HF_TOKEN | Set up HF_TOKEN.txt with valid token |
| YouTube download fails | Network/availability | Check video exists and is public |
| Poor voice cloning | Bad reference audio | Use 10-30s clear speech, single speaker; run SE first |
| SFX quality issues | Insufficient steps | Increase steps parameter |
| Music doesn't generate | Single mode used | music only works in dialogue mode |
| SFX line ignored | Missing /duration | Add /duration:nn directive |
| Cross-use conflict | Both voice and target for same character | Use one or the other per character |
| `overdose` + `translate` error | Mutually exclusive flags | Use one or the other, not both |
| SS fallback to Pyannote | VibeVoice unavailable | Install VibeVoice model; or set up HF_TOKEN for fallback |
| TTS slc poor voice match | Noisy source audio | Run SE or SVS on source before tts slc |
| SVS incomplete separation | Very mixed audio | Try SE first to clean up, then SVS |
| TTM overdose too slow | XL-Turbo is resource-intensive | Use standard TTM (1.5) for faster results |
| Video output has no audio | FFmpeg muxing issue | Ensure FFmpeg is installed and in PATH |
| TTM lego missing stems | Invalid stem names | Use only the 12 supported instrument track names |

---

# SECTION 10: PRO TIPS

1. **Enhance before cloning**: Run SE on noisy reference audio before using for voice cloning
2. **Separate before cloning**: Run SVS to extract clean vocals from mixed reference audio
3. **Test with short samples**: Generate 5-10 second tests before full production
4. **Layer with time positioning**: Use `/time:0` for overlapping SFX and speech
5. **Fade background music**: Use level `"0:50-30:20"` for intro-to-content transitions
6. **Batch STT for efficiency**: Process multiple files in one command
7. **Auto-clone for testing**: Use same file for STT analysis and voice reference to test pipeline
8. **MSTS for songs**: Always use `music` flag when converting singing voice
9. **Instrumental TTM**: Use `lyrics "..."` for backing tracks
10. **Result routing**: Always use `result` for automated workflows
11. **Check memory first**: Ensure 23GB RAM for any workflow involving music; 30GB for overdose
12. **Use overdose for final output**: Generate with standard TTM for testing, switch to overdose for final production
13. **SS before STT for multi-speaker**: Run SS first to identify and separate speakers, then transcribe individually for cleaner results
14. **SVS before STS for mixed audio**: Auto vocal extraction in STS handles most cases, but manual SVS → STS gives more control
15. **TTS slc preserves speaker identity**: For dubbing, use no-target mode (`tts slc "source.wav"`) to keep the original speaker's voice in English
16. **TTS is unified now**: Don't think in terms of TTS vs TTS+VC — just use `tts` with `voice` or `target` (or both via cross-use)
17. **TTM is unified now**: Don't think in terms of TTM vs TTM+VC — just use `ttm` with `vc` + `clone` when you need voice cloning
18. **Legos for custom arrangements**: Use `lego` with specific `make` stems to build custom instrumental arrangements
19. **Extract for remixing**: Use `extract` to pull individual stems from existing songs
20. **Remix for style transfer**: Use `remix` with `styling` and `bias` to create cover versions with adjustable style strength; add `lyrics` to guide new vocal content; use multi-source (up to 3) for creative composite sources and multi-reference (up to 3) for diverse style guidance
21. **Repaint for section editing**: Use `repaint` with `time:start-end` to restyle specific sections of a song; add `voice`/`music` prefix to isolate source components; use multi-pass mode (`"start-end/styling(...)/..."`) for sequential edits that build on each previous result
22. **Overdose XOR translate**: Remember these STT flags are mutually exclusive — pick based on whether you need translation or enhanced transcription
23. **TTS overdose for cleaner cloning**: Use `overdose` flag with TTS when doing voice cloning from dialogue sources — the 2s/3s trim on extracted voice clips avoids cross-speaker contamination and produces cleaner reference audio
24. **TTS overdose + music for premium output**: Combining `overdose` with `music` in TTS gives you both superior voice clip extraction (VibeVoice ASR) and higher quality background music (ACE-Step XL Turbo)

---

*This skill provides comprehensive understanding of VODER's architecture, complete CLI command catalog for all 10 modes, feature compatibility rules, and combo possibilities. AI agents can use this knowledge to construct complex audio processing workflows that would be impossible without deep understanding of how the tool works.*
