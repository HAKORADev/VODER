# VODER Skill for AI Agents

## Overview

VODER is a professional-grade voice processing tool that provides **9 distinct audio transformation modes** in a unified CLI interface. This skill enables AI agents to leverage VODER's full potential for complex audio processing workflows that would be impossible or extremely difficult without this knowledge.

**Core Philosophy**: VODER prioritizes **quality over speed**. There are no "fast" or "degraded" model options. The tool uses the best available models (Whisper large-v3-turbo, Qwen3-TTS, Seed-VC, ACE-Step, Pyannote, UniSE, TangoFlux) to produce professional-quality output.

---

# SECTION 1: UNDERSTANDING THE ARCHITECTURE

## What VODER Actually Is

VODER is not a single AI model — it is an **orchestration layer** that coordinates multiple state-of-the-art AI models to perform audio transformations. Understanding this architecture is crucial for combining features effectively.

### The Model Stack

| Model | Purpose | Used In Modes |
|-------|---------|---------------|
| **Whisper large-v3-turbo** | Speech-to-text transcription | STT, STT+TTS, Dialogue Source Analysis |
| **Qwen3-TTS VoiceDesign** | Generate speech from voice descriptions | TTS |
| **Qwen3-TTS Base** | Text-to-speech with built-in voice cloning | TTS+VC, STT+TTS |
| **Seed-VC v2** | Voice conversion (22.05kHz speech) | STS, TTM+VC |
| **Seed-VC v1** | Voice conversion (44.1kHz music) | MSTS (music voice conversion) |
| **ACE-Step 1.5** | Music generation from lyrics/style | TTM, TTM+VC, Background Music |
| **Pyannote** | Speaker diarization (who spoke when) | STT with `dialogue` flag |
| **EasyOCR** | Text extraction from images | STT with image input |
| **UniSE** | Speech enhancement/denoising | SE |
| **TangoFlux** | Text-to-audio sound effects | SFX |

### How Modes Relate to Each Other

```
INPUT TYPES:
┌─────────────────────────────────────────────────────────────────┐
│ Text ──────────────────► TTS, TTS+VC, TTM, TTM+VC, SFX         │
│ Audio ─────────────────► STS, STT, STT+TTS, SE                 │
│ Video ─────────────────► STS, STT, SE (auto-extract audio)     │
│ Image ─────────────────► STT (OCR text extraction)             │
│ YouTube/URL ───────────► STT (auto-download + transcribe)      │
└─────────────────────────────────────────────────────────────────┘

OUTPUT TYPES:
┌─────────────────────────────────────────────────────────────────┐
│ Audio Output: TTS, TTS+VC, STS, TTM, TTM+VC, SE, SFX           │
│ Text Output:  STT                                               │
│ Interactive:  STT+TTS (requires text editing step)              │
└─────────────────────────────────────────────────────────────────┘
```

### The Pipeline Flow

Understanding how data flows through VODER helps you chain operations:

```
TEXT INPUT PATH (TTS - Voice Design):
Text + Voice Description → Qwen3-TTS VoiceDesign → [Speech with Designed Voice]

TEXT INPUT PATH (TTS+VC - Voice Cloning):
Text + Reference Audio → Qwen3-TTS Base (extract voice embedding → synthesize with clone) → [Speech with Cloned Voice]

AUDIO INPUT PATH (Voice Conversion - STS):
Source Audio + Target Voice Audio → Seed-VC → [Converted Audio]

AUDIO INPUT PATH (Transcription):
Audio → Whisper → [Transcript Text]

MUSIC GENERATION PATH:
Lyrics + Style → ACE-Step → [Music with Vocals] → [Optional: Seed-VC Voice Clone] → Final Music

ENHANCEMENT PATH:
Degraded Audio → UniSE → [Clean Audio at 16kHz]
```

---

## How Parameters Work Together

### Parameter Types

VODER uses three types of parameters:

| Type | Description | Examples |
|------|-------------|----------|
| **Positional** | Mode name comes first, input files follow | `stt "audio.wav"` |
| **Named** | Key-value pairs with space separation | `voice "male"` `duration 30` |
| **Flags** | Standalone keywords that enable features | `timestamp` `dialogue` `music` |

### Parameter Multiplicity

Some parameters accept **multiple values** (dialogue mode), others accept **single values**:

| Parameter | Single Value | Multiple Values | Mode |
|-----------|--------------|-----------------|------|
| `script` | `"Hello world"` | `"James: Hello" "Sarah: Hi"` | TTS, TTS+VC |
| `voice` | `"male voice"` | `"James: male" "Sarah: female"` | TTS |
| `target` | `"voice.wav"` | `"James: james.wav" "Sarah: sarah.wav"` | TTS+VC |
| `music` | `"ambient"` | (single only) | TTS, TTS+VC |
| `level` | `"35"` | (single only) | TTS, TTS+VC |

### Parameter Order Rules

1. **Mode comes first**: `tts`, `stt`, `sts`, etc.
2. **Required parameters follow**: `script`, `voice`, `target`, etc.
3. **Optional parameters come after**: `music`, `level`, `result`
4. **Flags can appear anywhere after mode**: `timestamp`, `dialogue`, `music` (for STS)

---

# SECTION 2: COMPLETE ONE-LINE CLI COMMANDS CATALOG

## Catalog Navigation

| Mode | Section | Input Type | Output Type | One-Liner Support |
|------|---------|------------|-------------|-------------------|
| TTS | 2.1 | Text | Audio | ✅ Full (single + dialogue) |
| TTS+VC | 2.2 | Text + Audio | Audio | ✅ Full (single + dialogue) |
| STS | 2.3 | Audio + Audio | Audio | ✅ Single only |
| TTM | 2.4 | Text | Audio | ✅ Single only |
| TTM+VC | 2.5 | Text + Audio | Audio | ✅ Single only |
| STT | 2.6 | Audio/Video/Image/URL | Text | ✅ Full (single + batch) |
| SE | 2.7 | Audio/Video | Audio | ✅ Full |
| SFX | 2.8 | Text | Audio | ✅ Full |
| STT+TTS | 2.9 | Audio + Audio | Audio | ❌ Interactive only |

---

## 2.1 TTS (Text-to-Speech with Voice Design)

### What It Is
TTS mode generates human-like speech from text input. Unlike traditional TTS systems that use pre-recorded voices, VODER's Qwen3-TTS VoiceDesign model **creates voices from scratch** based on natural language descriptions. This means you can describe voices that don't exist in any database — a "weathered old sailor with a gravelly voice" or a "cheerful AI assistant with a slight metallic quality."

### How It Works
1. **Voice Prompt Interpretation**: The model parses your voice description to extract characteristics (age, gender, tone, pace, accent)
2. **Speech Synthesis**: Text is converted to mel-spectrograms based on the voice characteristics
3. **Audio Generation**: Spectrograms are converted to waveform audio
4. **Optional Music Addition**: If `music` parameter is provided, ACE-Step generates background music that matches the dialogue duration

### Command Catalog

#### Single Mode (One Speaker)
```bash
# Minimal command
python src/voder.py tts script "Your text here" voice "voice description"

# With output routing
python src/voder.py tts script "Your text here" voice "voice description" result "/output/file.wav"

# Full command
python src/voder.py tts script "Your text here" voice "voice description" music "music description" level "volume" result "/output/file.wav"

# OCR input (image to narration)
python src/voder.py tts ocr "path/to/image.png" voice "text: professional male narrator"

python src/voder.py tts ocr "script_screenshot.jpg" voice "text: warm female voice"
```

#### Dialogue Mode (Multiple Speakers)
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
```

### Parameter Reference

| Parameter | Required | Purpose | Single Mode | Dialogue Mode |
|-----------|----------|---------|-------------|---------------|
| `script` | Yes | Text to synthesize | Single text string | Multiple `"Char: text"` strings |
| `voice` | Yes* | Voice description | Single prompt | `"Char: prompt"` per character |
| `target` | No* | Voice reference file | Single path | `"Char: /path/to/file.wav"` |
| `music` | No | Background music style | Ignored | Single description |
| `level` | No | Music volume | Ignored | Volume specification |
| `result` | No | Output destination | Path | Path |

*Either `voice` or `target` required for non-SFX lines. Can mix both using cross-use feature.

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

---

## 2.2 TTS+VC (Text-to-Speech + Voice Cloning)

### What It Is
TTS+VC mode generates speech from text that sounds like a specific voice from a reference audio file. This is **voice cloning** — the ability to make synthesized speech sound like a real person. The reference audio can be a recording of anyone (with ethical consent), and the output will match their voice characteristics.

### How It Works (IMPORTANT: Uses Qwen3-TTS Base Built-in Cloning)

**TTS+VC does NOT use Seed-VC**. It uses **Qwen3-TTS Base's built-in voice cloning capability**:

1. **Voice Embedding Extraction**: Qwen3-TTS Base's `create_voice_clone_prompt()` method analyzes the reference audio and extracts a voice embedding (x-vector) using `x_vector_only_mode=True`
2. **Direct Synthesis with Clone**: The `generate_voice_clone()` method synthesizes the text **directly with the cloned voice characteristics embedded** — this is NOT a two-step process (synthesis then conversion), but a single integrated process
3. **Consistency Optimization**: In dialogue mode, the voice embedding is extracted **once per character** at the start and reused for all their lines

**Why This Matters**: Unlike a two-stage process (synthesize → convert), Qwen3-TTS Base's integrated cloning produces more natural results because the voice characteristics are considered during the entire synthesis process, not applied as a transformation afterward.

### Why Use TTS+VC Instead of TTS
- **Consistent branding**: Use the same voice across all content
- **Specific person**: Clone a particular voice (podcast host, character, celebrity with permission)
- **Localization**: Maintain voice identity while changing language
- **Accessibility**: Create content in a familiar voice

### Command Catalog

#### Single Mode
```bash
# Minimal command
python src/voder.py tts+vc script "Your text here" target "voice_reference.wav"

# With output routing
python src/voder.py tts+vc script "Your text here" target "voice_reference.wav" result "/output/file.wav"

# OCR input (image to narration with voice clone)
python src/voder.py tts+vc ocr "path/to/image.png" target "text: voice_reference.wav"

python src/voder.py tts+vc ocr "subtitle_image.jpg" target "text: speaker_clone.wav"
```

#### Dialogue Mode
```bash
# Two characters with cloned voices
python src/voder.py tts+vc script "James: line1" "Sarah: line2" target "James: /path/to/james.wav" "Sarah: /path/to/sarah.wav"

# With background music
python src/voder.py tts+vc script "J: Hello" "S: Hi" target "J: james.wav" "S: sarah.wav" music "jazz background" level "30"

# Cross-use: Mix cloned and generated voices
python src/voder.py tts+vc script "J: Hello" "S: Hi" target "J: james.wav" voice "S: bright female voice"
```

### Reference Audio Requirements

| Factor | Requirement | Why |
|--------|-------------|-----|
| **Duration** | 10-30 seconds optimal | Enough data for voice extraction; longer doesn't help |
| **Quality** | Clear, minimal noise | Noise interferes with voice feature extraction |
| **Content** | Continuous speech | Silence or music doesn't contribute voice data |
| **Speaker** | Single speaker only | Mixed speakers confuse the extraction |
| **Format** | WAV preferred, MP3 supported | WAV preserves audio fidelity |

### Voice Consistency in Dialogue
VODER extracts voice characteristics **once per character** at the start of dialogue processing. This means:
- All lines from "James" use the same extracted voice profile
- No variation between the 1st and 10th line of the same character
- Professional-quality consistency throughout long dialogues

---

## 2.3 STS (Speech-to-Speech Voice Conversion)

### What It Is
STS mode transforms the **voice** in source audio to sound like a different person, while preserving **everything else** — the words, emotion, timing, prosody, pauses, and delivery style. Only the speaker identity changes.

### How It Works
1. **Content Extraction**: Seed-VC extracts the linguistic and prosodic content from source audio (what was said, how it was said)
2. **Voice Extraction**: The target voice reference is analyzed for speaker characteristics
3. **Voice Transfer**: The content is re-synthesized with the target voice characteristics
4. **Sample Rate Handling**: v2 model outputs at 22.05kHz (speech), v1 at 44.1kHz (music)

### STS vs TTS+VC: When to Use Which

| Scenario | Use STS When... | Use TTS+VC When... |
|----------|-----------------|---------------------|
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

# From video file (audio auto-extracted)
python src/voder.py sts base "presentation.mp4" target "voice_actor.wav" result "/output/output.wav"
```

#### MSTS (Music Voice Conversion)
```bash
# For songs/musical content - uses 44.1kHz model
python src/voder.py sts base "song.wav" target "singer_voice.wav" music

# Convert singing voice in a song
python src/voder.py sts base "original_song.wav" target "new_singer.wav" music result "/output/cover.wav"
```

### Model Selection

| Flag | Model | Sample Rate | Use Case |
|------|-------|-------------|----------|
| (none) | Seed-VC v2 | 22.05kHz | Speech, podcasts, interviews |
| `music` | Seed-VC v1 | 44.1kHz | Songs, musical content, singing |

---

## 2.4 TTM (Text-to-Music Generation)

### What It Is
TTM mode generates **complete musical compositions** from lyrics and style descriptions using ACE-Step. The model creates both the instrumental arrangement AND the vocal performance. You provide lyrics, describe the musical style, specify duration, and receive a fully produced song.

### How It Works
1. **Lyrics Processing**: Lyrics are parsed into vocal melody and rhythm
2. **Style Interpretation**: Style prompt guides instrumentation, genre, mood, tempo
3. **Music Generation**: ACE-Step creates aligned instrumental and vocal tracks
4. **Duration Matching**: Output is stretched/compressed to hit target duration

### Unique Capability: Instrumental-Only
Using `"..."` as lyrics generates **purely instrumental music** with no vocals. This is how background music for dialogue is created internally.

### Command Catalog

```bash
# With lyrics (song with vocals)
python src/voder.py ttm lyrics "Verse 1:\nLyrics here\n\nChorus:\nChorus lyrics" styling "pop, upbeat, female vocals" duration 60

# Instrumental only (no vocals)
python src/voder.py ttm lyrics "..." styling "cinematic orchestral, dramatic" duration 90

# With output routing
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill" duration 120 result "/output/background.wav"

# Short jingle
python src/voder.py ttm lyrics "..." styling "upbeat corporate, bright" duration 15 result "/output/jingle.wav"
```

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

---

## 2.5 TTM+VC (Text-to-Music + Voice Conversion)

### What It Is
TTM+VC generates music (like TTM) and then converts the vocalist's voice to match a reference. This lets you create songs with a **specific singer's voice** without that person ever recording the song.

### How It Works
1. **Music Generation**: ACE-Step creates music with default AI vocals
2. **Model Swap**: ACE-Step is offloaded from memory
3. **Voice Conversion**: Seed-VC converts the vocal track to match reference voice
4. **Mixing**: Converted vocals are mixed back with instrumental

### Memory Optimization
The automatic model offloading between stages means this mode uses **less peak memory** than running TTM and STS separately.

### Command Catalog

```bash
# Generate song with cloned vocalist
python src/voder.py ttm+vc lyrics "Verse 1:\nMy lyrics here" styling "rock ballad, emotional" duration 60 target "singer_reference.wav"

# Instrumental backing + cloned voice
python src/voder.py ttm+vc lyrics "..." styling "acoustic guitar backing" duration 180 target "voice.wav" result "/output/backing.wav"

# With output routing
python src/voder.py ttm+vc lyrics "Chorus:\nThis is our moment" styling "pop anthem" duration 45 target "artist.wav" result "/output/song.wav"
```

---

## 2.6 STT (Speech-to-Text Transcription)

### What It Is
STT mode converts audio, video, images, and URLs into text. It uses Whisper for transcription and can optionally identify **who spoke when** using Pyannote speaker diarization. This is the only mode that produces **text output** rather than audio.

### How It Works
1. **Input Processing**: Audio extracted from video; text extracted from images via OCR; URLs downloaded via yt-dlp
2. **Transcription**: Whisper transcribes with word-level timestamps
3. **Optional Diarization**: Pyannote identifies speaker segments
4. **Alignment**: Transcription and diarization are aligned using three-tier overlap matching
5. **Output**: Text file saved to results/ directory

### Input Flexibility

| Input Type | How It's Processed |
|------------|-------------------|
| Audio file (WAV, MP3, FLAC, etc.) | Direct transcription |
| Video file (MP4, MKV, AVI, etc.) | Audio track extracted, then transcribed |
| Image file (PNG, JPG, etc.) | Text extracted via EasyOCR |
| YouTube URL | Audio downloaded via yt-dlp, then transcribed |
| Bilibili URL | Audio downloaded via yt-dlp, then transcribed |
| TikTok URL | Audio downloaded via yt-dlp, then transcribed |

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
```

### Output Format Variations

| Flags | Output Format | Example |
|-------|---------------|---------|
| (none) | Plain text | `Hello everyone welcome to today's meeting` |
| `timestamp` | Timestamped segments | `[00:00.000 → 00:03.500] Hello everyone` |
| `dialogue` | Speaker-labeled | `Speaker 1: Hello everyone` |
| `timestamp dialogue` | Combined | `[00:00.000 → 00:03.500] Speaker 1: Hello everyone` |

### HF_TOKEN Requirement
Speaker diarization (`dialogue` flag) requires:
1. HuggingFace account
2. Token from https://huggingface.co/settings/tokens
3. Accept conditions at https://huggingface.co/pyannote/speaker-diarization-community-1
4. Token in `HF_TOKEN.txt` file or `HF_TOKEN` environment variable

---

## 2.7 SE (Speech Enhancement)

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

# From video file
python src/voder.py se "recording.mp4"

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

## 2.8 SFX (Sound Effects Generation)

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

## 2.9 STT+TTS (Speech-to-Text + Synthesis)

### What It Is
STT+TTS mode transcribes audio to text, allows editing of the text, then re-synthesizes with a target voice. This enables **content modification** while maintaining the general structure of the original.

### Why It's Interactive Only
The text editing step requires user interaction. You must:
1. Review the transcription
2. Edit the text (fix errors, change words, modify content)
3. Approve for synthesis

### Command
```bash
# Interactive mode only
python src/voder.py cli
# Then select STT+TTS from the menu
```

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

Cross-use allows mixing **generated voices** (via `voice` parameter) and **cloned voices** (via `target` parameter) in the **same dialogue**. This works in both TTS and TTS+VC modes.

## Why This Matters

Without cross-use:
- TTS mode: ALL characters must use generated voices
- TTS+VC mode: ALL characters must use cloned voices

With cross-use:
- Some characters generated, others cloned
- Perfect for scenarios where you have reference audio for some speakers but not others
- Mix known voices with new character voices

## Rules

1. Each character must use EITHER `voice` OR `target`, not both
2. Character names must match between script and parameter
3. Case-insensitive matching (James = james = JAMES)

## Command Examples

### TTS Mode: One Generated, One Cloned
```bash
python src/voder.py tts script \
  "James: Welcome to our podcast!" \
  "Sarah: Thanks for having me!" \
  voice "James: deep male voice, authoritative" \
  target "Sarah: /path/to/sarah_voice_reference.wav"
```

### TTS+VC Mode: One Cloned, One Generated
```bash
python src/voder.py tts+vc script \
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
Music Description → ACE-Step (lyrics: "...") → Duration-Matched Music
                                                          ↓
                                     Mix (Dialogue + Music at Level %)
                                                          ↓
                                          Final Output (_m suffix)
```

## Why Use Empty Lyrics

The `music` parameter internally uses `lyrics "..."` for ACE-Step, which tells the model to generate **instrumental-only music** with no vocals. This is specifically designed for background/ambient use.

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

| Feature | TTS | TTS+VC | STS | TTM | TTM+VC | STT | SE | SFX |
|---------|-----|--------|-----|-----|--------|-----|-----|-----|
| Single mode | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dialogue mode | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `voice` param | ✅ | ✅* | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `target` param | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ |
| Cross-use | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `music` param | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `level` param | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SFX lines | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Script directives | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `timestamp` flag | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `dialogue` flag | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| `result` param | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `music` flag (STS) | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| `steps` param | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |
| `guide` param | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

*In TTS+VC, `voice` is for generated voices; `target` for cloned voices

## Valid Parameter Orders

### TTS Mode
```
python src/voder.py tts script "text" [script "text2" ...] voice "prompt" [voice "prompt2" ...] [target "Char: path" ...] [music "description"] [level "spec"] [result "path"]
```

### TTS+VC Mode
```
python src/voder.py tts+vc script "text" [script "text2" ...] target "path" [target "Char: path2" ...] [voice "Char: prompt" ...] [music "description"] [level "spec"] [result "path"]
```

### STS Mode
```
python src/voder.py sts base "source.wav" target "voice.wav" [music] [result "path"]
```

### TTM Mode
```
python src/voder.py ttm lyrics "lyrics text" styling "style prompt" duration N [result "path"]
```

### TTM+VC Mode
```
python src/voder.py ttm+vc lyrics "lyrics" styling "style" duration N target "voice.wav" [result "path"]
```

### STT Mode
```
python src/voder.py stt "file1" ["file2" ...] [timestamp] [dialogue] [result "path"]
```

### SE Mode
```
python src/voder.py se "input.wav" [result "path"]
```

### SFX Mode
```
python src/voder.py sfx sound "description" duration N [steps N] [guide N.N] [result "path"]
```

## Feature Combo Catalog

### Combo 1: Dialogue + SFX + Background Music (Full Production)
**Mode**: TTS or TTS+VC
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

### Combo 2: Dialogue + Cross-use + Background Music
**Mode**: TTS or TTS+VC
**Features**: Dialogue mode + voice + target (cross-use) + music
```bash
python src/voder.py tts+vc script \
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

### Combo 4: Batch STT with All Features
**Mode**: STT
**Features**: Multiple files + timestamp + dialogue + result
```bash
python src/voder.py stt "ep1.wav" "ep2.wav" "ep3.wav" timestamp dialogue result "/output/transcripts/"
```

### Combo 5: YouTube Transcription with Full Analysis
**Mode**: STT
**Features**: URL input + timestamp + dialogue
```bash
python src/voder.py stt "https://youtube.com/watch?v=VIDEO_ID" timestamp dialogue result "/output/video_transcript.txt"
```

### Combo 6: MSTS for Song Cover
**Mode**: STS
**Features**: music flag + result
```bash
python src/voder.py sts base "original_song.wav" target "new_singer_voice.wav" music result "/output/cover.wav"
```

### Combo 7: TTM+VC for Custom Song with Specific Voice
**Mode**: TTM+VC
**Features**: lyrics + styling + duration + target
```bash
python src/voder.py ttm+vc lyrics "Verse 1:\nMy custom lyrics\n\nChorus:\nChorus text" styling "pop ballad, emotional" duration 90 target "artist_voice.wav" result "/output/custom_song.wav"
```

### Combo 8: SE Pre-processing + TTS+VC
**Mode**: SE then TTS+VC (two commands)
**Features**: Enhancement + voice cloning
```bash
python src/voder.py se "noisy_reference.wav" result "/clean/reference.wav"
python src/voder.py tts+vc script "Hello, this is a voice clone test." target "/clean/reference.wav" result "/output/cloned_speech.wav"
```

### Combo 9: Image-to-Audio Pipeline
**Mode**: STT then TTS (two commands)
**Features**: Image OCR + text-to-speech
```bash
python src/voder.py stt "script_screenshot.png" result "/output/extracted_text.txt"
# Parse the text file, then:
python src/voder.py tts script "[extracted text content]" voice "professional narrator" result "/output/audio.wav"
```

### Combo 10: Full Podcast Episode Production
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

---

# SECTION 8: MEMORY REQUIREMENTS & SYSTEM PLANNING

## Memory by Mode

| Mode | RAM | VRAM (if GPU) | Notes |
|------|-----|---------------|-------|
| TTS (single/dialogue) | 12GB | 4GB | Qwen model |
| TTS + music | 23GB | 15-16GB | Adds ACE model |
| TTS+VC | 12GB | 4GB | Qwen + Seed-VC |
| TTS+VC + music | 23GB | 15-16GB | Full stack |
| STS | 13GB | 14GB | Seed-VC alone |
| TTM | 23GB | 15-16GB | ACE model |
| TTM+VC | 23GB | 16GB | Auto-offloads between stages |
| STT | 12GB | N/A (CPU) | Whisper |
| STT + diarization | 15GB | N/A (CPU) | Whisper + Pyannote |
| SE | 11GB | 4GB | UniSE |
| SFX | 12GB | 4GB | TangoFlux |

## Planning Complex Workflows

### Workflow Memory Budget
When chaining operations, you don't need to sum all requirements — models are offloaded between operations. Plan for the **peak memory of the most demanding step**.

### Example: Podcast Production Pipeline
```
Step 1: STT (15GB peak) → offloaded
Step 2: TTS+VC with music (23GB peak) → offloaded
Step 3: Done

Total memory needed: 23GB (not 38GB)
```

### Example: Song Cover Pipeline
```
Step 1: SE (11GB peak) → offloaded
Step 2: TTM+VC (23GB peak) → offloaded
Step 3: Done

Total memory needed: 23GB
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
| Poor voice cloning | Bad reference audio | Use 10-30s clear speech, single speaker |
| SFX quality issues | Insufficient steps | Increase steps parameter |
| Music doesn't generate | Single mode used | music only works in dialogue mode |
| SFX line ignored | Missing /duration | Add /duration:nn directive |
| Cross-use conflict | Both voice and target for same character | Use one or the other per character |

---

# SECTION 10: PRO TIPS

1. **Enhance before cloning**: Run SE on noisy reference audio before using for voice cloning
2. **Test with short samples**: Generate 5-10 second tests before full production
3. **Layer with time positioning**: Use `/time:0` for overlapping SFX and speech
4. **Fade background music**: Use level `"0:50-30:20"` for intro-to-content transitions
5. **Batch STT for efficiency**: Process multiple files in one command
6. **Auto-clone for testing**: Use same file for STT analysis and voice reference to test pipeline
7. **MSTS for songs**: Always use `music` flag when converting singing voice
8. **Instrumental TTM**: Use `lyrics "..."` for backing tracks
9. **Result routing**: Always use `result` for automated workflows
10. **Check memory first**: Ensure 23GB RAM for any workflow involving music

---

*This skill provides comprehensive understanding of VODER's architecture, complete CLI command catalog, feature compatibility rules, and combo possibilities. AI agents can use this knowledge to construct complex audio processing workflows that would be impossible without deep understanding of how the tool works.*
