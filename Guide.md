# VODER Technical Guide

## Table of Contents

- [Introduction & Vision](#introduction--vision)
- [The Philosophy: Quality Over Speed](#the-philosophy-quality-over-speed)
- [Why Hardcoded Models?](#why-hardcoded-models)
  - [The Quality Imperative](#the-quality-imperative)
  - [Custom Model Support](#custom-model-support)
  - [Custom Versions](#custom-versions)
- [Centralized Model Management](#centralized-model-management)
- [Processing Modes Deep Dive](#processing-modes-deep-dive)
  - [STT: Speech-to-Text](#stt-speech-to-text)
  - [TTS: Text-to-Speech](#tts-text-to-speech)
  - [TTS+VC: Text-to-Speech + Voice Cloning](#ttsvc-text-to-speech--voice-cloning)
  - [STS: Speech-to-Speech Voice Conversion](#sts-speech-to-speech-voice-conversion)
  - [TTM: Text-to-Music](#ttm-text-to-music)
  - [TTM+VC: Text-to-Music + Voice Conversion](#ttmvc-text-to-music--voice-conversion)
  - [STT+TTS: Speech-to-Text + Synthesis](#stttts-speech-to-text--synthesis)
- [Speaker Diarization](#speaker-diarization)
  - [What It Is](#what-it-is)
  - [How It Works](#how-it-works-2)
  - [Three-Tier Alignment System](#three-tier-alignment-system)
  - [Post-Processing](#post-processing)
  - [HF_TOKEN Requirement](#hf_token-requirement)
  - [Where It's Available](#where-its-available)
  - [Diarization Tips](#diarization-tips)
- [Image Text Extraction (EasyOCR)](#image-text-extraction-easyocr)
  - [Supported Formats](#supported-formats)
  - [How It Integrates](#how-it-integrates)
- [YouTube & Video Platform Support](#youtube--video-platform-support)
  - [Supported Platforms](#supported-platforms)
  - [How It Works](#how-it-works-3)
  - [Cross-Mode Integration](#cross-mode-integration)
  - [Error Handling & Fallbacks](#error-handling--fallbacks)
- [Voice Clip Extraction](#voice-clip-extraction)
  - [What It Does](#what-it-does)
  - [How It Works](#how-it-works-4)
  - [Integration with TTS+VC](#integration-with-ttsvc)
  - [YouTube URL Support](#youtube-url-support)
- [The Dialogue System](#the-dialogue-system)
  - [What Dialogue Mode Is](#what-dialogue-mode-is)
  - [How It Works](#how-it-works)
  - [Dialogue Source Analysis](#dialogue-source-analysis)
  - [Dialogue Input in GUI](#dialogue-input-in-gui)
  - [Dialogue Input in CLI](#dialogue-input-in-cli)
    - [Interactive CLI Dialogue](#interactive-cli-dialogue)
    - [One‑Liner Dialogue](#one-liner-dialogue)
  - [Voice Prompt Configuration](#voice-prompt-configuration)
  - [Optional Background Music for Dialogue](#optional-background-music-for-dialogue)
    - [How It Works](#how-it-works-1)
    - [GUI Workflow](#gui-workflow)
    - [Interactive CLI Workflow](#interactive-cli-workflow)
    - [One‑Liner CLI Workflow](#one-liner-cli-workflow)
    - [Technical Implementation](#technical-implementation)
- [Tips & Tricks](#tips--tricks)
  - [Getting Better Results](#getting-better-results)
  - [Multi-Speaker Scenarios](#multi-speaker-scenarios)
  - [Using Same Audio Source](#using-same-audio-source)
  - [Voice Cloning Best Practices](#voice-cloning-best-practices)
  - [Background Music Best Practices](#background-music-best-practices)
  - [Diarization Best Practices](#diarization-best-practices)
  - [YouTube Download Tips](#youtube-download-tips)
  - [OCR Accuracy Tips](#ocr-accuracy-tips)
  - [Voice Clip Extraction Best Practices](#voice-clip-extraction-best-practices)
- [Version Information](#version-information)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Introduction & Vision

VODER is a professional‑grade voice processing tool that brings together seven distinct audio transformation capabilities in a single, unified interface. Unlike tools that force you to jump between multiple applications for different voice‑related tasks, VODER provides everything from standalone transcription to text‑to‑speech synthesis to music generation under one roof.

**What VODER Actually Does:**

At its core, VODER orchestrates state‑of‑the‑art AI models to perform voice‑related transformations. It can transcribe speech to text with speaker identification, generate speech from text using either designed voices or cloned references, transform one voice into another while preserving content, and create music from lyrics with optional voice conversion for the vocalist. It can download and analyze content directly from YouTube and other video platforms, extract voice clips from multi‑speaker audio for use as cloning references, and even read text from images using optical character recognition. This isn't about chasing the fastest processing times or highest frame rates — it's about achieving professional‑quality results that actually sound good.

**Why VODER Exists:**

The voice synthesis market is dominated by expensive commercial platforms that charge per character or per month. ElevenLabs, OpenAI, and others offer powerful capabilities, but at costs that add up quickly for creators, developers, and businesses alike. More importantly, no existing open‑source solution offered all seven processing capabilities in a unified interface. You could find separate tools for TTS, voice conversion, and music generation, but none that worked together seamlessly — and certainly none that could pull a video from YouTube, identify the speakers, extract voice references, and generate a complete dialogue.

VODER was built to fill this gap. The goal from day one was to create a local, free, open‑source alternative that doesn't compromise on quality. Is it perfect? No software is. But it works, it keeps improving, and it provides genuine utility without subscription fees or usage limits.

**What Makes VODER Different:**

Most voice processing tools focus on a single use case. VODER takes a different approach — it treats voice and audio processing as a unified problem space. The same interface that generates speech from text can also convert that speech between voices, and the same voice cloning technology can apply to both speech and singing. The same transcription engine that powers speech‑to‑text also drives speaker diarization for multi‑speaker analysis. This integration enables workflows that would otherwise require multiple tools and significant manual effort.

---

## The Philosophy: Quality Over Speed

### We Don't Chase FPS

This is worth emphasizing because it's fundamental to VODER's design philosophy. There are no "recommended requirements" in the traditional sense. This isn't a video game where higher frame rates give you a better experience. The only metric that matters is avoiding one thing: Out Of Memory (OOM) errors.

When we say "minimum requirements" with 8GB VRAM, that's not a performance target — it's a reliability floor. If you have exactly 8GB, VODER will work. If you have 12GB, it won't process things twice as fast. It just means you have more headroom for longer audio files or more complex operations. The quality remains the same because we're not offering quality presets that sacrifice output fidelity for speed.

**Why We Don't Offer Fast Modes:**

Every other tool on the market offers "fast" or "efficient" variants of their models. Smaller models, quantized weights, reduced quality settings. We explicitly chose not to include these options. Here's why: a degraded model produces output that is genuinely worse, not just faster to generate. If you're using voice synthesis for content creation, professional work, or anything where quality matters, you'd be better off not using the tool at all than using a degraded version.

Think of it like photography. You can have a cheap smartphone camera that takes pictures instantly, or you can use a professional camera that requires proper technique and takes slightly longer. The smartphone photo is "faster" but the professional camera photo is objectively better quality. VODER is the professional camera of voice processing tools.

**The OOM Reality:**

Some operations require significant memory. Voice conversion models, especially, need to load multiple neural network components and maintain activations throughout the processing pipeline. If you try to process a 10‑minute audio file and run out of VRAM, the solution isn't to use a smaller model — it's to process shorter segments. VODER doesn't offer shortcuts that compromise quality because shortcuts in AI almost always mean worse output.

**System Requirements Explained:**

When we list minimum requirements, we're being honest about what actually works. All VODER modes run on CPU — no GPU is required. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

| Mode | Base Memory | Additional | Total RAM | GPU (CUDA) | VRAM |
|------|--------------|------------|-----------|------------|------|
| STT (standalone) | 8GB | +4GB (Whisper) | 12GB | CPU only | N/A |
| STT + Diarization | 8GB | +4GB (Whisper) +2-3GB (Pyannote) | 15GB | CPU only | N/A |
| TTS, TTS+VC (no music) | 8GB | +4GB (Qwen) | 12GB | Optional | 4GB (GTX 1060) |
| TTS, TTS+VC (with music) | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| STT+TTS | 8GB | +4GB (Qwen) | 12GB | Optional | 4GB (GTX 1060) |
| STS | 8GB | +5GB (Seed-VC) | 13GB | Optional | 14GB |
| TTM | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| TTM+VC | 8GB | +15GB (ACE) | 23GB | Optional | 16GB |

- **CPU**: 4-6 cores minimum for model loading and non-GPU operations
- **RAM**: 12GB minimum for basic modes (STT, TTS, STT+TTS), 15GB for STT with diarization, 23GB for ACE-related modes (TTM, TTM+VC, or TTS/TTS+VC with music)
- **GPU (CUDA)**: Optional - all modes work on CPU. GPU acceleration significantly speeds up STS, TTM, and TTM+VC modes
- **VRAM**: 4GB minimum (6GB recommended, 16GB for best performance with music modes). STT and diarization modes are CPU-only and require no GPU.
- **Storage**: SSD recommended for model downloads and result saving

**VRAM Guidelines:**

| VRAM | Performance Level | Suitable Modes |
|------|-------------------|----------------|
| No GPU (CPU only) | Slow | All modes (STT, STT+diarization, OCR included) |
| 4GB | Usable | TTS, TTS+VC (no music), STT+TTS |
| 6GB | Minimum | TTS, TTS+VC (no music), STT+TTS |
| 14GB | Mid-range | STS, all TTS modes |
| 15-16GB | Recommended | TTS+VC with music, TTM, TTM+VC |
| 24GB | Maximum | All modes at full speed (RTX 4090) |
| T4 (16GB) | Server-grade | All modes (not typical consumer GPU) |

These aren't arbitrary numbers. They're based on actual testing of the models VODER uses.

---

## Why Hardcoded Models?

VODER uses hardcoded default models. This isn't an accident or a limitation — it's a deliberate design choice made for quality reasons.

### The Quality Imperative

The models VODER uses were selected because they represent the best available quality in their respective categories. Qwen3‑TTS for text‑to‑speech, Seed‑VC v2 for voice conversion, ACE‑Step for music generation, Whisper for speech‑to‑text, Pyannote for speaker diarization, EasyOCR for image text extraction — these aren't arbitrary choices. They're the result of evaluating multiple alternatives and selecting the ones that produce the best results.

Smaller models exist. Quantized variants exist. "Fast" versions exist. We deliberately don't use them because they produce noticeably worse output. A smaller TTS model sounds less natural, has more artifacts, and fails on complex text. A quantized voice conversion model loses the subtle characteristics that make voice cloning convincing. Using degraded models would undermine the entire purpose of having VODER exist.

**The HF_TOKEN.txt File:**

You'll find a file called `HF_TOKEN.txt` in the VODER directory. This file serves two important purposes:

1. It allows VODER to access gated model repositories (such as Pyannote's speaker diarization pipeline on HuggingFace).
2. It allows advanced users to modify model configurations if they really want to.

The file contains instructions for getting your HuggingFace token. If you provide a valid token, VODER will use it for gated model repositories — **this is required for speaker diarization to function**. See the [Speaker Diarization](#speaker-diarization) section for details on setting up your token.

**We Do Not Recommend Changing Models:**

This needs to be stated clearly. The hardcoded models are there because they're the best options available. If you have technical expertise and want to experiment with different model configurations, the capability exists. But VODER is optimized for its default configuration, and deviation from these defaults may produce worse results or cause errors.

Think of it like a restaurant that only serves one dish. They chose that dish because it's the best thing they can make. You can ask them to make something else, but it won't be as good as their specialty. VODER's specialty is orchestrating these specific models together — that's what it does best.

### Custom Versions

If someone creates a modified version of VODER with different model configurations, that's exactly what it is: a modified version. Custom configurations won't be supported in the main VODER documentation or issue tracker because the main project only guarantees quality for its default configuration.

For those interested in exploring custom model configurations, we'll maintain a separate document (CUSTOM_VERSIONS.md) where community‑contributed modifications can be documented. These are not official VODER builds, but if you want to share your experiments with different models or configurations, that file provides a place to do so.

---

## Centralized Model Management

VODER now uses a centralized model storage system under `src/models/`. This is a structural improvement that eliminates the problem of model files being scattered across different directories.

**Directory Structure:**

```
src/models/
├── whisper/           # Whisper STT model (whisper-turbo.pt)
├── qwen_tts_voice_design/  # Qwen3-TTS VoiceDesign model
├── qwen_tts_base/     # Qwen3-TTS Base model
├── seed_vc/           # Seed-VC voice conversion models
├── ace_step/          # ACE-Step music generation models
├── pyannote/          # Pyannote diarization pipeline
└── easyocr/           # EasyOCR models and weights
```

**HuggingFace Cache Redirection:**

Some models (particularly Pyannote and EasyOCR) are downloaded through HuggingFace. VODER sets the `HF_HOME` and `TRANSFORMERS_CACHE` environment variables to point to the `src/models/` directory. This means:

- All HuggingFace downloads go into the centralized directory
- Models aren't scattered in `~/.cache/huggingface/` or other system directories
- You can see exactly what's downloaded and how much space it uses
- Cleaning up is as simple as deleting `src/models/`

**Auto-Creation at Startup:**

All model subdirectories are automatically created when VODER starts. You don't need to manually create any directories. If a directory doesn't exist, it's created before any model loading begins.

**Why This Matters:**

Previously, model files could end up in multiple locations depending on how they were downloaded — some in the project root, some in system cache directories, some in user home directories. This made it difficult to:

- Track total disk usage for VODER
- Clean up after uninstalling
- Move VODER to a different drive
- Share installations across machines

The centralized system solves all of these problems. Everything VODER needs lives under `src/models/`, making the installation self‑contained and predictable.

---

## Processing Modes Deep Dive

### STT: Speech-to-Text

**What It Does:**

STT (Speech‑to‑Text) is a standalone transcription mode that converts audio, video, and images into text. It uses Whisper to transcribe speech with word‑level timestamps, and can optionally identify individual speakers using Pyannote diarization. It can even download and transcribe content directly from YouTube URLs.

This is VODER's first mode that doesn't produce audio output — its output is a text file.

**How It Works:**

1. **Input Handling**: VODER accepts multiple input types:
   - **Audio files** (WAV, MP3, FLAC, OGG, M4A, etc.)
   - **Video files** (MP4, MKV, AVI, MOV, etc.) — audio track is extracted automatically
   - **Image files** (PNG, JPG, JPEG, BMP, TIFF) — text is extracted via EasyOCR
   - **YouTube/URLs** — audio is downloaded via yt-dlp before transcription
2. **Transcription**: Whisper loads the audio and produces a transcript with word‑level timestamps
3. **Optional Timestamps**: The `timestamp` flag adds formatted timestamps to the output
4. **Optional Diarization**: The `dialogue` flag runs Pyannote speaker diarization and attributes each segment to a speaker
5. **Output**: Results are saved as `.txt` files in the `results/` directory

**Batch Processing:**

STT mode supports processing multiple files in a single command. When you provide multiple input paths (or a directory), VODER processes each file sequentially and produces a separate output text file for each.

**Output File Naming:**

| Input Type | Output Naming |
|------------|---------------|
| Audio file (`podcast.mp3`) | `voder_stt_podcast.txt` |
| Audio with timestamps | `voder_stt_podcast_timestamp.txt` |
| Audio with diarization | `voder_stt_podcast_dialogue.txt` |
| Audio with both | `voder_stt_podcast_timestamp_dialogue.txt` |
| YouTube URL | `voder_stt_<video_id>.txt` |
| Image file (`slide.png`) | `voder_stt_slide.txt` |

The base filename is derived from the input filename (without extension). For YouTube URLs, the video ID is used.

**CLI Usage:**

```bash
# Basic transcription
python src/voder.py stt result /path/to/audio.mp3

# With timestamps
python src/voder.py stt result /path/to/audio.mp3 timestamp

# With speaker diarization
python src/voder.py stt result /path/to/audio.mp3 dialogue

# With both timestamps and diarization
python src/voder.py stt result /path/to/audio.mp3 timestamp dialogue

# Transcribe a YouTube video
python src/voder.py stt result "https://www.youtube.com/watch?v=VIDEO_ID" timestamp dialogue

# Batch process multiple files
python src/voder.py stt result /path/to/file1.mp3 /path/to/file2.wav /path/to/image.png

# Interactive CLI
python src/voder.py cli
# Select mode 1 (STT), then follow prompts
```

**Best For:**

- Transcribing podcasts, interviews, and meetings
- Creating subtitles or captions for video content
- Content analysis and text mining
- Accessibility — making audio content available to deaf/hard‑of‑hearing users
- Extracting text from images (screenshots, slides, scanned documents)
- Generating dialogue scripts from existing multi‑speaker audio
- Preparing voice reference clips for TTS+VC dialogue mode

**Technical Notes:**

STT mode is entirely CPU‑based. No GPU is required. Whisper Turbo provides an excellent balance of speed and accuracy. Processing time depends on audio length — approximately 1x real‑time on a modern CPU (a 10‑minute file takes about 10 minutes to transcribe).

When the `dialogue` flag is used, Pyannote's speaker diarization pipeline runs after Whisper transcription. The two outputs are aligned using a three‑tier system (see [Speaker Diarization](#speaker-diarization) for details).

**Memory Requirements:** STT requires approximately 12GB RAM (8GB base + ~4GB for Whisper model). With diarization enabled, it requires approximately 15GB RAM (8GB base + ~4GB Whisper + ~2-3GB Pyannote).

---

### TTS: Text-to-Speech

**What It Does:**

TTS generates speech from text using Qwen3‑TTS VoiceDesign. You provide a text script and a voice prompt describing the desired voice characteristics, and VODER produces audio of that voice saying that text.

**How It Works:**

The VoiceDesign model interprets natural language descriptions to generate appropriate voice characteristics. Unlike traditional TTS systems that use pre‑recorded voice samples, VoiceDesign creates voices from scratch based on your description. This makes it incredibly flexible — you can describe voices that don't exist in any database.

**Why It's Like That:**

VoiceDesign exists because not everyone wants to clone an existing voice. Sometimes you need a generic voice for narration, or you want to create a character voice that doesn't correspond to any real person. The descriptive approach provides infinite flexibility without requiring reference audio files.

**Optional Background Music (Dialogue Only):**

When using TTS in **dialogue mode** (multiple speakers, script lines containing a colon), you can optionally add automatically generated background music. After the dialogue is synthesized, VODER generates a music track using ACE‑Step with empty lyrics `"..."` and a duration matching the exact length of the dialogue. The music is mixed at **35% volume** relative to the dialogue, creating a subtle ambient bed. The final file is saved with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`). This feature is available in GUI (via a clean modal dialog), interactive CLI (prompt after voice prompts), and one‑liner CLI (optional `music` parameter). See [Optional Background Music for Dialogue](#optional-background-music-for-dialogue) for full details.

**Best For:**

- Narration and voiceover work
- Creating character voices for content
- Situations where you don't have reference audio
- Rapid prototyping of voice concepts
- Generating multiple voice variations for comparison
- **Dialogue with ambient soundtrack** (podcasts, storytelling)

**Voice Prompt Examples:**

| Desired Voice | Example Prompt |
|---------------|----------------|
| Professional male | "adult male, deep voice, clear pronunciation, professional tone" |
| Warm female | "adult female, warm tone, gentle, conversational" |
| Energetic young | "young adult, energetic, fast‑paced, enthusiastic" |
| News anchor | "middle‑aged, authoritative, measured pace, broadcasting quality" |
| Storytelling | "deep narrative voice, expressive, dramatic pauses" |

**Technical Notes:**

TTS mode works on CPU without GPU acceleration. Processing time scales with text length, not with prompt complexity. The VoiceDesign model interprets prompts at generation time, so more detailed prompts give the model more information to work with but don't significantly affect processing time.

**Memory Requirements:** TTS requires approximately 12GB RAM (8GB base + 4GB for Qwen model).

---

### TTS+VC: Text-to-Speech + Voice Cloning

**What It Does:**

TTS+VC generates speech from text and then applies voice cloning to match a reference voice. The text is synthesized using Qwen3‑TTS Base, and the output is transformed to sound like the voice in your reference audio.

**How It Works:**

The process happens in two stages. First, Qwen3‑TTS Base generates speech from your text using its default voice characteristics. Then, the voice cloning system extracts distinctive features from your reference audio and applies them to the generated speech. The result is your text spoken by a voice that matches your reference.

**Why It's Like That:**

Voice cloning opens possibilities that pure TTS can't match. You can clone a specific person's voice and use it consistently across all your content. You can match voices between different speakers in a dialogue. You can create synthetic content that sounds like real people (with appropriate consent and ethical considerations).

**Voice Clip Extraction Integration:**

When using TTS+VC with the interactive CLI, you now have the option to automatically extract voice reference clips from a multi‑speaker audio file. Instead of manually finding and providing reference audio for each character, VODER can:

1. Download audio from a YouTube URL (or accept a local file)
2. Run Whisper + Pyannote to identify speakers and their segments
3. Extract the longest segment per speaker as a voice reference clip
4. Feed those clips directly into the TTS+VC dialogue pipeline

This eliminates the manual step of finding clean reference audio for each speaker. See [Voice Clip Extraction](#voice-clip-extraction) for full details.

**Optional Background Music (Dialogue Only):**

Just like in TTS mode, when TTS+VC is used in **dialogue mode** you can optionally add automatically generated background music. The music is generated **after** all dialogue lines have been synthesized, concatenated, and voice‑cloned. It uses the same ACE‑Step process (empty lyrics, auto‑duration, 35% volume) and the same output naming (`_m` suffix). The feature is accessible through the same GUI dialog, interactive CLI prompt, and one‑liner `music` parameter. This allows you to create fully produced podcast episodes, narrated stories, or interview segments with ambient background music — all in a single operation.

**Best For:**

- Consistent voice branding across content
- Dialogue with cloned character voices
- Matching voice characteristics between speakers
- Creating content in a voice you don't have but can record
- Localization while preserving original voice characteristics
- **Produced dialogue with background ambience**

**Reference Audio Requirements:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 10‑30 seconds optimal |
| Quality | Clear audio, minimal background noise |
| Content | Continuous speech, not singing or silence |
| Speakers | Single speaker only |
| Format | WAV preferred, MP3 supported |

**Single vs Dialogue Mode:**

In **single mode** (one reference file), the entire script uses that voice. In **dialogue mode** (multiple reference files), each character in a dialogue script is assigned a different reference audio. This is the foundation of VODER's dialogue system, and it is available in **both GUI and CLI**.

**Voice Consistency in Dialogue:**

VODER extracts voice characteristics **once per character** in dialogue mode, rather than re‑extracting for each line. This ensures consistent voice quality throughout the dialogue. If a character speaks multiple lines (e.g., 5 lines for "James"), the voice prompt is extracted once and reused for all lines of that character. This eliminates variations that occurred when re-extracting voice for each line, providing stable and professional-quality voice cloning across entire dialogues.

**Technical Notes:**

TTS+VC works on CPU without GPU. The voice cloning happens during synthesis, not as a post‑processing step, which ensures the cloned voice characteristics are integrated throughout the generated speech rather than applied superficially.

**Memory Requirements:** TTS+VC requires approximately 12GB RAM (8GB base + 4GB for Qwen model). If using background music, it requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### STS: Speech-to-Speech Voice Conversion

**What It Does:**

STS (Speech‑to‑Speech) transforms source audio to sound like a target voice while preserving the original content, emotion, timing, and prosody. The speaker changes, but everything they say remains exactly the same.

**MSTS (Music-STS):**

STS now supports musical inputs via the **MSTS** feature. When converting voice in songs or musical audio, use the `music` parameter to switch to Seed‑VC v1 (44.1kHz) instead of the standard v2 model (22.05kHz). This provides better voice conversion quality for music content because v1 is optimized for higher sample rates and musical waveforms.

- **GUI**: A dialog asks "musical inputs?" with Yes/No buttons before processing
- **Interactive CLI**: After entering base and target paths, prompted "Are the inputs musical? (Y/N):"
- **One-line CLI**: Add `music` keyword at the end: `voder.py sts path/base path/target music`
- **Output**: MSTS outputs use `voder_m_sts_timestamp.wav` naming; standard STS uses `voder_sts_timestamp.wav`

**How It Works:**

Seed‑VC v2 analyzes both the source and target audio to extract content representations and voice characteristics. It then synthesizes new audio that combines the source content with the target voice. This isn't simple audio manipulation — it's neural voice conversion that genuinely reconstructs the speech in a different voice.

**Why It's Like That:**

Voice conversion serves specific use cases that TTS and TTS+VC can't handle. You might have archival audio that needs voice preservation but content modification. You might want to maintain the exact delivery and emotion of a performance while changing the voice. Voice conversion preserves paralinguistic features that text‑to‑speech can't reproduce.

**Best For:**

- Preserving delivery while changing voice
- Content modification in existing audio
- Voice anonymization or de‑identification
- Consistent voice application across multiple recordings
- Archival content republishing with voice updates

**Input Considerations:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 5‑60 seconds optimal per segment |
| Content | Clear speech, minimal background music |
| Quality | Studio quality preferred, phone quality works but loses detail |
| Format | WAV or high‑bitrate MP3 |

**Technical Notes:**

STS runs on CPU without GPU. Input audio is automatically resampled to 22050 Hz for model processing, and output is resampled to 44100 Hz for playback.

**Memory Requirements:** STS requires approximately 13GB RAM (8GB base + 5GB for Seed-VC model).

---

### TTM: Text-to-Music

**What It Does:**

TTM (Text‑to‑Music) generates original music from lyrics and a style prompt using ACE‑Step. You provide song lyrics, describe the desired musical style, and specify duration — VODER creates original music with vocals matching your lyrics.

**How It Works:**

ACE‑Step interprets your lyrics as vocal content and your style prompt as musical direction. It generates both the instrumental arrangement and the vocal performance, synchronized to your specified duration. The lyrics become the vocal melody, and the style prompt guides the instrumentation, genre, and mood.

**Why It's Like That:**

Music generation from lyrics is distinct from instrumental generation because vocals add a layer of complexity. The lyrics must be converted to actual singing, which requires understanding of melody, rhythm, and phonetics. ACE‑Step handles this by treating lyrics as both content and guidance for the vocal generation pipeline.

**Note on Background Music:**

The same ACE‑Step engine is used to generate background music for dialogue. In that context, the lyrics are set to `"..."` (a placeholder for empty vocals), and the style prompt is taken from the user's music description. This yields purely instrumental music suitable for ambient use.

**Best For:**

- Creating original background music with vocals
- Song prototyping and demo creation
- Content needing custom music with lyrics
- Experimental music creation
- Rapid music visualization from lyrics

**Lyrics Format:**

```
Verse 1:
Walking down the empty street
Feeling the rhythm in my feet
The city lights are shining bright
Guiding me through the night

Chorus:
This is our moment, this is our time
Everything's gonna be just fine
Dancing under the moonlight
Everything feels so right
```

**Style Prompt Examples:**

| Genre/Mood | Example Prompt |
|------------|----------------|
| Upbeat pop | "upbeat pop, catchy melody, modern production, female vocals" |
| Rock ballad | "electric guitar, driving drums, powerful vocals, emotional" |
| Electronic dance | "synthesizer, dance beat, energetic, electronic production" |
| Acoustic folk | "acoustic guitar, gentle arrangement, folk style, warm vocals" |

**Duration Considerations:**

| Duration | Use Case |
|----------|----------|
| 10‑30 seconds | Short clips, transitions, soundbites |
| 30‑60 seconds | Full verses or choruses |
| 60‑120 seconds | Complete short songs |
| 120‑300 seconds | Full compositions with multiple sections |

Shorter durations are more reliable and consistent. Very long durations may produce variable results depending on the complexity of lyrics and style combination.

**Technical Notes:**

TTM works on CPU without GPU. Processing time scales primarily with duration rather than lyrics length. The style prompt complexity doesn't significantly affect processing time but does affect the musical output characteristics.

**Memory Requirements:** TTM requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### TTM+VC: Text-to-Music + Voice Conversion

**What It Does:**

TTM+VC generates music from lyrics and style (same as TTM) and then applies voice conversion to change the vocalist's voice. This combines music generation with voice cloning for the singing voice.

**How It Works:**

The pipeline is straightforward: first generate the music with ACE‑Step (TTM stage), then apply Seed‑VC voice conversion to the vocal track (VC stage). The generated music's vocals are transformed to match your reference voice while preserving the melody, timing, and musical characteristics.

**Memory Optimisation:**

VODER explicitly offloads models from memory after each operation completes. This applies to all modes in both GUI and interactive CLI:

- **GUI Mode**: ProcessingThread calls cleanup() after finishing, releasing all loaded models (STT, TTS, TTS+VC, STS, TTM)
- **Interactive CLI**: Each mode (TTS, TTS+VC, STS, STT+TTS, TTM, TTM+VC) offloads models before returning
- **Pattern Applied**: `del model`, `gc.collect()`, `torch.cuda.empty_cache()`

This prevents memory accumulation when performing multiple operations in a single session, making VODER more reliable for batch processing workflows.

**Why It's Like That:**

Sometimes the generated vocals from ACE‑Step don't match the specific voice you need. TTM+VC allows you to generate music efficiently with default vocals, then swap in a cloned voice. This is particularly useful for consistent voice branding in music content or when you need a specific singer's voice in your generated music.

**Best For:**

- Music with specific vocalist voice
- Consistent voice across multiple generated tracks
- Voice‑preserving music modifications
- Professional music production workflows
- Content requiring both music generation and voice cloning

**Technical Notes:**

TTM+VC runs on CPU. This is a composite mode that chains TTM and STS operations, so it inherits the memory requirements of both stages. Longer durations increase the chance of issues.

**Memory Requirements:** TTM+VC requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### STT+TTS: Speech-to-Text + Synthesis

**What It Does:**

STT+TTS transcribes audio to text using Whisper, allows you to edit the transcribed content, and then synthesizes the edited text with a target voice. This enables voice modification while preserving the original delivery characteristics.

**How It Works:**

The transcription stage converts speech to text with word‑level timestamps. You can review and modify the transcribed text before synthesis. The synthesis stage then reads your (possibly edited) text and produces audio in the target voice. This preserves the timing and delivery structure from the original audio if you don't modify the text significantly.

**Why It's Like That:**

This mode is for when you have existing audio content that needs voice transformation. By transcribing, editing, and resynthesizing, you can change what someone says while keeping the general timing and delivery. It's not a simple voice conversion — it's a reconstructive process that allows complete content modification.

**Best For:**

- Changing content in existing audio
- Fixing transcription errors automatically
- Localizing content into different languages
- Creating fictional dialogue from real voice samples
- Voice modification with full control over content

**Interactive Nature:**

STT+TTS requires user interaction for text editing, which is why it's only available in interactive CLI mode and GUI mode. The one‑liner mode cannot accommodate this workflow. You must either use `python src/voder.py cli` and select the STT+TTS option, or use the GUI for full visual feedback.

**Multi‑Speaker Note:**

If your base audio contains multiple speakers, Whisper will transcribe all of them. The synthesis will use a single target voice for the entire text. If you need per‑speaker voice cloning, use the dialogue system with speaker diarization instead (see [Dialogue Source Analysis](#dialogue-source-analysis)).

**Technical Notes:**

STT+TTS works on CPU without GPU for the Whisper transcription stage. Voice cloning in the synthesis stage also works on CPU. This makes it accessible for users without NVIDIA graphics hardware.

**Memory Requirements:** STT+TTS requires approximately 12GB RAM (8GB base + 4GB for Qwen model).

---

## Speaker Diarization

### What It Is

Speaker diarization is the process of automatically identifying and separating who said what in an audio recording. VODER uses Pyannote, a state‑of‑the‑art diarization pipeline, combined with Whisper's word‑level timestamps to produce detailed, speaker‑attributed transcripts.

Instead of a flat transcript that reads like a wall of text, diarization produces output like this:

```
[00:00.000 → 00:05.230] SPEAKER_00: Welcome to today's podcast.
[00:05.500 → 00:09.800] SPEAKER_01: Thanks for having me, great to be here.
[00:10.100 → 00:16.400] SPEAKER_00: Let's dive right in. What made you start this project?
```

This is invaluable for analyzing interviews, meetings, podcasts, and any content with multiple speakers.

### How It Works

The diarization pipeline runs in two stages:

1. **Pyannote Segmentation**: The audio is analyzed by Pyannote's speaker embedding and segmentation model. This produces time‑based segments, each labeled with a speaker ID (SPEAKER_00, SPEAKER_01, etc.). Pyannote identifies how many speakers are present and where each speaker's turns begin and end.

2. **Whisper Alignment**: Whisper transcribes the full audio with word‑level timestamps. Each word gets a start and end time. VODER then aligns Whisper's word timestamps with Pyannote's speaker segments to determine which speaker said each word.

The result is a word‑level transcript where every word is attributed to a specific speaker.

### Three-Tier Alignment System

Aligning Whisper words to Pyannote segments isn't always straightforward — timing differences between the two models can cause edge cases. VODER uses a three‑tier alignment strategy to handle this:

**Tier 1: Contained**

If a Whisper word's start and end times fall entirely within a Pyannote speaker segment, the word is assigned to that speaker. This is the most reliable case and covers the vast majority of words.

**Tier 2: Best Overlap**

If a word isn't fully contained within any segment (it straddles a boundary), VODER calculates the overlap duration between the word and each candidate speaker segment. The word is assigned to the speaker with the longest overlap. This handles most boundary cases correctly.

**Tier 3: Nearest Neighbor**

In rare cases where a word has no overlap with any segment (e.g., it falls in a gap between segments), VODER assigns it to the speaker of the nearest preceding segment. This prevents "orphan" words that have no speaker attribution.

### Post-Processing

After initial alignment, two post‑processing steps improve quality:

**Nearest-Speaker Fallback:**

Any remaining unattributed words (words that somehow escaped all three alignment tiers) are assigned to the closest speaker segment. This ensures every word in the transcript has a speaker label.

**Short Utterance Merging:**

Very short speaker segments (e.g., a 0.3‑second fragment attributed to SPEAKER_01 surrounded by SPEAKER_00 segments) are often diarization artifacts rather than genuine speaker changes. VODER merges short segments into their neighboring speaker to reduce false speaker switches. This produces cleaner, more readable output.

### HF_TOKEN Requirement

Pyannote's models are hosted on HuggingFace behind a gated access agreement. To use diarization, you must:

1. Visit [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the user agreement
2. Visit [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the user agreement
3. Create a HuggingFace access token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Add your token to `src/HF_TOKEN.txt` (one line, just the token string)

Without a valid token, diarization will fail with an authentication error. See [Troubleshooting](#troubleshooting--common-issues) for common token issues.

### Where It's Available

Diarization is integrated into multiple VODER features:

| Feature | How Diarization Is Used |
|---------|------------------------|
| **STT mode** (`dialogue` flag) | Produces speaker‑attributed transcript as a text file |
| **Dialogue source analysis** | Analyzes multi‑speaker audio to generate a dialogue script for TTS+VC |
| **Voice clip extraction** | Identifies speakers and selects the best reference clip per speaker |

### Diarization Tips

**For Best Results:**

- Use clear audio with minimal background noise
- Ensure speakers have distinct voices (different pitch, timbre, or accent)
- Avoid music playing underneath speech
- Two to four speakers work best; more than six may reduce accuracy
- Longer recordings (60+ seconds) give Pyannote more data to distinguish speakers

**Known Limitations:**

- Overlapping speech may be attributed to only one speaker
- Very similar voices (e.g., identical twins) may be confused
- Heavy background noise degrades diarization accuracy
- The number of speakers is estimated automatically and may be wrong for very short clips

---

## Image Text Extraction (EasyOCR)

VODER can extract text from images using EasyOCR. This is useful when your source material contains visual text — screenshots, presentation slides, scanned documents, or photos of signs and labels.

### Supported Formats

| Format | Extensions |
|--------|-----------|
| JPEG | `.jpg`, `.jpeg` |
| PNG | `.png` |
| BMP | `.bmp` |
| TIFF | `.tiff`, `.tif` |
| WebP | `.webp` |

### How It Integrates

EasyOCR is available in two contexts:

**1. STT Mode:**

When you pass an image file as input to STT mode, VODER automatically detects it as an image (rather than audio or video) and runs EasyOCR instead of Whisper. The extracted text is saved to a `.txt` file, just like audio transcription output.

```bash
python src/voder.py stt result /path/to/screenshot.png
# Output: results/voder_stt_screenshot.txt
```

**2. Dialogue Source Analysis:**

When using dialogue source analysis (e.g., in TTS+VC interactive CLI), if you provide an image file as the source, VODER extracts the text via OCR and then proceeds to analyze it for dialogue content. Text formatted with character prefixes (like "James: Hello") is parsed into a dialogue script automatically.

**Technical Notes:**

EasyOCR runs entirely on CPU — no GPU is needed. It supports 80+ languages including English, Chinese, Japanese, Korean, and most European languages. Language detection is automatic; no configuration is needed.

Memory usage for EasyOCR is minimal (a few hundred MB) on top of VODER's base requirements. The OCR models are stored in `src/models/easyocr/` as part of the centralized model management system.

---

## YouTube & Video Platform Support

VODER can download audio directly from YouTube and other video platforms, then process it with any mode that accepts audio input. This eliminates the manual step of downloading files with a separate tool.

### Supported Platforms

| Platform | URL Pattern | Notes |
|----------|-------------|-------|
| YouTube | `youtube.com/watch?v=...`, `youtu.be/...` | Full support, best quality |
| Bilibili | `bilibili.com/video/...` | Full support |
| TikTok | `tiktok.com/@user/video/...` | Full support, short clips |

VODER uses **yt-dlp** for all downloads. yt-dlp is a mature, actively maintained tool that supports hundreds of video platforms beyond those listed. If a platform works with yt-dlp, it likely works with VODER.

### How It Works

1. **URL Detection**: VODER checks if the input path starts with `http://` or `https://`. If so, it's treated as a URL.
2. **Audio Download**: yt-dlp downloads the best available audio stream and saves it as a temporary file in VODER's temp directory.
3. **Processing**: The downloaded audio is passed to whatever mode you selected (STT, dialogue analysis, voice clip extraction, etc.).
4. **Cleanup**: The temporary download file is deleted after processing. Only the final result is kept.

### Cross-Mode Integration

YouTube URLs work as input in all of these contexts:

| Mode | What Happens |
|------|--------------|
| **STT** | Downloads audio, transcribes with Whisper |
| **STT + dialogue** | Downloads audio, transcribes with diarization |
| **Dialogue source analysis** | Downloads audio, runs STT + diarization, generates dialogue script |
| **Voice clip extraction** | Downloads audio, identifies speakers, extracts clips |
| **STS** | Downloads audio, uses as base or target for voice conversion |

### Error Handling & Fallbacks

- **Invalid URL**: VODER detects non‑video URLs early and shows a clear error
- **Private/age‑restricted videos**: yt-dlp may fail; VODER shows the download error with a suggestion to use a local file instead
- **Geo‑blocked content**: Download will fail; use a VPN or download manually
- **yt-dlp not installed**: VODER checks for yt-dlp at startup and warns if it's missing. Install it with `pip install yt-dlp` or your system package manager
- **Large files**: Very long videos may take a long time to download. VODER has no file size limit, but be mindful of disk space and processing time

---

## Voice Clip Extraction

### What It Does

Voice clip extraction automatically finds the best voice reference clip for each speaker in a multi‑speaker audio file. Instead of manually cutting and selecting audio segments for each speaker, VODER does it for you.

This feature is designed to feed directly into TTS+VC dialogue mode — once clips are extracted, they're available as voice references for dialogue generation.

### How It Works

1. **Input**: Provide a multi‑speaker audio file (or YouTube URL)
2. **Speaker Identification**: Whisper transcribes the audio with word‑level timestamps; Pyannote identifies speaker segments
3. **Segment Selection**: For each speaker, VODER finds the longest continuous segment. Longer segments provide better voice characteristics for cloning.
4. **Extraction**: FFmpeg extracts each selected segment as a separate WAV file
5. **Output**: One clip per speaker, saved to the results directory

**Output Naming:**

```
voder_clip_SPEAKER_00_timestamp.wav
voder_clip_SPEAKER_01_timestamp.wav
voder_clip_SPEAKER_02_timestamp.wav
```

The SPEAKER IDs correspond to the order speakers first appear in the audio.

### Integration with TTS+VC

Voice clip extraction is most powerful when used as part of the TTS+VC interactive CLI workflow. After extracting clips, VODER can immediately use them as voice references for dialogue generation:

```
Interactive CLI (TTS+VC) workflow:

1. Enter dialogue script lines
2. VODER asks: "Provide audio reference for each speaker, or provide a multi-speaker source for automatic extraction"
3. You provide a YouTube URL or local multi-speaker file
4. VODER extracts clips and assigns them to speakers
5. Dialogue is generated with the extracted voice references
```

This creates a complete pipeline: from raw multi‑speaker audio to a fully produced dialogue with cloned voices, all in one interactive session.

### YouTube URL Support

You can provide a YouTube URL as the input for voice clip extraction. VODER downloads the audio, runs diarization, extracts clips — all automatically. This means you can clone voices from YouTube videos without manually downloading anything:

```bash
# In interactive TTS+VC mode, when asked for audio references:
> https://www.youtube.com/watch?v=VIDEO_ID
```

Or extract clips standalone:

```bash
python src/voder.py stt result "https://www.youtube.com/watch?v=VIDEO_ID" dialogue
```

**Best Practices:**

- Choose audio where each speaker has at least 10‑15 seconds of continuous speech
- Avoid segments with background music or sound effects
- Two to four speakers works best for diarization accuracy
- If possible, use audio where speakers take turns (not constant overlap)

---

## The Dialogue System

### What Dialogue Mode Is

VODER's dialogue system enables multi‑speaker script generation. You write a script with multiple characters, assign voice references to each character, and VODER generates a cohesive audio track where each line is spoken by the appropriate voice.

**What It Is NOT:**

Despite how it might seem, dialogue mode is not AI systems conversing with each other. There are no neural networks having conversations. Each line is synthesized independently, one after another, using the specified voice reference. The "conversation" effect is achieved through:

- Sequential processing of script lines in order
- Voice routing that matches characters to their assigned samples
- FFmpeg concatenation that preserves timing and flow
- Independent synthesis of each line with consistent voice characteristics

It's automation, not artificial conversation intelligence.

**Dialogue is Now Available in CLI:**

Earlier versions of VODER restricted dialogue creation to the GUI. **This is no longer the case.** Dialogue mode is fully supported in **both GUI and CLI**, including one‑liner commands and interactive CLI input. All references to "dialogue is GUI‑only" in older documentation are outdated.

### How It Works

The dialogue processing pipeline follows these stages:

1. **Parse Script**: Extract dialogue items with sequence number, character name, and text
2. **Parse Voice Prompts**: Build character‑to‑audio‑reference mapping
3. **Validate**: Ensure every character has a voice reference
4. **Temporary Files**: Create temporary directory for segment audio files
5. **Iterate Lines**: For each dialogue line:
   - Load corresponding voice reference audio
   - Extract voice characteristics from reference
   - Synthesize the line text using that voice
   - Save segment to temporary file
6. **Concatenate**: Use FFmpeg to combine all segments into one file
7. **Optional Background Music**: If requested, generate and mix music
8. **Clean Up**: Remove temporary files
9. **Export**: Save final dialogue to results folder

### Dialogue Source Analysis

VODER can now **analyze existing multi‑speaker content to generate dialogue scripts automatically**. This overcomes the previous limitation where multi‑speaker input wasn't supported. You can now start with a raw interview, podcast, or meeting recording and let VODER figure out who said what — then use that script for dialogue generation with cloned voices.

**Previously**, the "Why Not Multi-Speaker Input?" section explained that VODER couldn't accept multi‑speaker audio because speaker separation was too unreliable. **That limitation has been overcome.** With Pyannote diarization and Whisper word‑level alignment, VODER can now identify speakers and attribute speech to them with high accuracy.

**How Dialogue Source Analysis Works:**

VODER accepts several types of source material:

**1. Text Files:**

Provide a text file containing dialogue. VODER reads it and auto‑formats it into a dialogue script. If the text already follows the `Character: text` format, it's parsed directly. If it's plain text without character labels, it's treated as single‑speaker input.

```bash
# Text file with character labels (auto-parsed into dialogue)
python src/voder.py tts+vc result /path/to/script.txt target "Character1: /ref1.wav" "Character2: /ref2.wav"
```

**2. Audio/Video Files:**

Provide an audio or video file with multiple speakers. VODER runs Whisper + Pyannote diarization to transcribe the content and identify speakers. The result is a speaker‑attributed dialogue script ready for TTS+VC processing.

```bash
# Analyze audio and generate dialogue
python src/voder.py tts+vc result /path/to/interview.mp3
# VODER will prompt for voice references per detected speaker
```

**3. Image Files:**

Provide an image containing text (a screenshot of a script, a photo of a document, etc.). VODER uses EasyOCR to extract the text, then parses it as a dialogue script if it contains character labels.

```bash
# Extract text from image and use as dialogue source
python src/voder.py tts+vc result /path/to/screenshot.png
```

**4. YouTube URLs:**

Provide a YouTube, Bilibili, or TikTok URL. VODER downloads the audio, then runs the same STT + diarization pipeline as local audio files.

```bash
# Analyze YouTube video and generate dialogue
python src/voder.py tts+vc result "https://www.youtube.com/watch?v=VIDEO_ID"
```

**What This Enables:**

The combination of diarization + dialogue source analysis creates a powerful new workflow:

1. Start with any multi‑speaker source (YouTube video, interview recording, podcast)
2. VODER identifies speakers and transcribes their dialogue
3. You can optionally edit the generated script
4. VODER extracts voice clips per speaker (or you provide your own)
5. VODER generates a complete dialogue with cloned voices

This pipeline can recreate entire conversations with different voices, translate dialogues while preserving speaker identity, or produce dubbed versions of existing content.

---

### Dialogue Input in GUI

VODER's GUI now uses a **row‑based dialogue editor** instead of a free‑text script box. This design makes character assignment explicit and reduces parsing errors.

**Script Entry:**

- Each line is a separate row containing **Character** and **Dialogue** fields.
- New rows are automatically added when you fill the last row.
- The first row has no delete button; subsequent rows can be deleted individually.
- The GUI automatically tracks which characters appear in the script and displays a corresponding voice prompt area for each character.

**Voice Prompt Assignment:**

- In **TTS mode** (Voice Design): each character gets a `QLineEdit` where you type a natural‑language voice description.
- In **TTS+VC mode** (Voice Cloning): each character gets a `QComboBox` dropdown listing the numbers of the audio files you have loaded. You simply select the number corresponding to the reference audio you want for that character.

**Audio Reference Management (TTS+VC):**

- Use the **"Add Audio"** button to load reference files.
- Each loaded file is assigned a unique number (1, 2, 3…).
- You can play or delete any file from the list at any time.
- When you add or delete files, the dropdowns in the voice prompt area are automatically updated with the current set of numbers.

**Example GUI Workflow (TTS+VC):**

1. Switch to TTS+VC mode.
2. In the script area, add rows:  
   `James: Welcome to the podcast.`  
   `Sarah: Thanks for having me.`  
   `James: Let's talk about AI.`
3. Load three audio files: `james_voice.wav`, `sarah_voice.wav`, `james_voice.wav` (again for the second James line).
4. The voice prompt area automatically shows rows for **james** and **sarah** with dropdowns containing `1, 2, 3`.
5. Assign:  
   `james` → `1`  
   `sarah` → `2`  
   `james` → `3` (or you can reuse `1` if you prefer the same reference)
6. (Optional) A dialog will appear asking if you want background music — see [Optional Background Music for Dialogue](#optional-background-music-for-dialogue).
7. Click **Generate**. VODER synthesizes each line with the appropriate cloned voice, concatenates them, and (if requested) mixes with background music.

**Why This Design:**

- Eliminates format errors (no more `1:James: "text"` syntax mistakes).
- Makes character‑to‑audio assignment visual and immediate.
- Prevents mismatches between script characters and available references.
- Enables quick auditioning of different voice assignments.

---

### Dialogue Input in CLI

VODER now provides two ways to create dialogue in the command line: **interactive** and **one‑liner**.

#### Interactive CLI Dialogue

Run `python src/voder.py cli`, select mode 2 (TTS) or 3 (TTS+VC). You will be prompted to enter script lines. Enter one line per row, using the format:

```
Character: text
```

Type your lines, press Enter after each, and leave an empty line to finish. VODER automatically detects dialogue mode (because lines contain `:`). It then asks for voice prompts (TTS) or audio file paths (TTS+VC) **for each character that appeared**, in sorted order.

**Example (TTS):**

```
$ python src/voder.py cli
...
Select Mode: 2
Enter script lines. Use format 'Character: text' for dialogue, or plain text for single speech.
Empty line finishes script entry.
> James: Welcome to the show.
> Sarah: Glad to be here.
> 
Voice prompts for 2 character(s):
james: deep male voice, authoritative
sarah: bright female voice, cheerful
```

**After** collecting all voice prompts/assignments, you will be asked:

```
Add background music? (y/N):
```

If you answer `y` or `yes`, you can enter a music description. Leaving the description blank or pressing Enter without input skips the music. VODER then generates the full dialogue (with or without background music).

**Example (TTS+VC):**

```
$ python src/voder.py cli
...
Select Mode: 3
Enter script lines...
> Narrator: Once upon a time...
> Alice: I wonder what this does.
> Bob: Let's find out.
> 
Audio file paths for 3 character(s):
narrator: /voices/narrator.wav
alice: /voices/alice.wav
bob: /voices/bob.wav
Add background music? (y/N): y
Music description: soft piano, cinematic strings
```

**Why Interactive CLI Dialogue Exists:**

- Users who prefer terminal workflows can now create full multi‑speaker content without launching the GUI.
- The interactive prompts ensure that every character receives a valid voice reference before processing begins.
- It bridges the gap between full automation (one‑liner) and visual interfaces.
- The optional music prompt fits naturally into this interactive flow.

#### One‑Liner Dialogue

One‑liner commands now support dialogue through **multiple values per parameter**. This is the recommended method for automated scripts and AI agents.

**Syntax for TTS dialogue:**

```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice description" "Character2: voice description"
```

**Syntax for TTS+VC dialogue:**

```bash
python src/voder.py tts+vc script "Character1: line1" "Character2: line2" target "Character1: /path/to/reference1.wav" "Character2: /path/to/reference2.wav"
```

**Syntax for STT transcription:**

```bash
# Basic
python src/voder.py stt result /path/to/audio.mp3

# With timestamps
python src/voder.py stt result /path/to/audio.mp3 timestamp

# With speaker diarization
python src/voder.py stt result /path/to/audio.mp3 dialogue

# With both
python src/voder.py stt result /path/to/audio.mp3 timestamp dialogue

# YouTube URL as input
python src/voder.py stt result "https://www.youtube.com/watch?v=VIDEO_ID" dialogue

# Image file as input
python src/voder.py stt result /path/to/screenshot.png
```

**Optional Background Music in One‑Liner:**

To add background music, simply include a `music` parameter with your description:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep male" "Sarah: cheerful female" music "soft piano, cinematic"
```

If the `music` parameter is supplied but the script is **not** in dialogue mode (i.e., no colon in any `script` parameter), it is ignored with a warning. If the `music` parameter is present but its value is an empty string (`music ""`), it is treated as if no music was requested.

**Important Rules:**

- The order of `script` values must match the dialogue line order.
- The order of `voice`/`target` values must match the character order (first appearance in script).
- For single‑speaker scripts, you may omit the colon in both script and voice/target; the system will treat it as single mode and ignore any `music` parameter.
- You can also use explicit keyword repetition if preferred (backward compatible).

**Examples:**

```bash
python src/voder.py tts script "James: Hello, Sarah." "Sarah: Hi James, how are you?" "James: I'm great, thanks for asking!" voice "James: deep male, warm" "Sarah: young female, cheerful" music "ambient electronic, chill"

python src/voder.py tts+vc script "Host: Welcome to the podcast." "Guest: Thanks for having me." "Host: So, tell us about your work." target "Host: /voices/host.wav" "Guest: /voices/guest.wav" "Host: /voices/host.wav" music "soft piano, strings"
```

**Validation:**

If any character in the script does not have a matching voice/target entry, VODER will reject the command with a clear error message listing the missing characters.

---

### Voice Prompt Configuration

The mapping between characters and their voice references (audio file numbers or file paths) is handled differently in GUI and CLI, but the underlying concept is the same.

#### In GUI (TTS+VC)

- Audio files are loaded into a list and automatically numbered.
- The voice prompt area provides a dropdown per character showing all available numbers.
- You select the number that corresponds to the desired reference file.

#### In GUI (TTS)

- The voice prompt area provides a text field per character.
- You type a natural‑language description (e.g., "warm female narrator").

#### In CLI (TTS)

- For dialogue, you supply `voice "Character: description"` entries.
- For single mode, you supply `voice "description"` (no colon).

#### In CLI (TTS+VC)

- For dialogue, you supply `target "Character: /path/to/audio.wav"` entries.
- For single mode, you supply `target "/path/to/audio.wav"` (no colon).

**No More Numbered Prompts in GUI:**

Older versions of VODER required you to write prompts like `James:1` in a text box. This is **no longer used in the GUI**. The dropdown system eliminates syntax errors and makes voice assignment explicit.

---

### Optional Background Music for Dialogue

VODER includes a unique feature that automatically generates and mixes ambient background music into dialogue scripts. This is **only available for dialogue mode** (i.e., when the script contains at least one line with a colon) and works for both **TTS** and **TTS+VC** modes.

#### How It Works

1. **Dialogue Synthesis** – VODER first generates all dialogue segments, concatenates them into a single audio file using FFmpeg, and saves it temporarily.
2. **Duration Measurement** – The exact duration of the dialogue audio is calculated (using `torchaudio.info`).
3. **Music Generation** – The ACE‑Step model is loaded (if not already) and used to generate a music track with:
   - **Lyrics**: `"..."` (a placeholder that yields pure instrumental music)
   - **Style prompt**: the description provided by the user (e.g., `"soft piano, cinematic strings"`)
   - **Duration**: exactly the length of the dialogue audio (rounded to nearest whole second)
4. **Volume Adjustment** – The music track is reduced to **35% of its original volume** using FFmpeg's `volume=0.35` filter. This level has been empirically chosen to provide a noticeable but non‑intrusive ambient bed.
5. **Mixing** – The attenuated music is mixed with the dialogue using FFmpeg's `amix` filter, which sums the two streams and preserves the longer duration (the music is generated to match exactly, so both durations are equal).
6. **Memory Management** – After dialogue synthesis, the Qwen‑TTS model is released from memory. After music generation, the ACE‑Step handler is explicitly deleted and, if CUDA is available, `torch.cuda.empty_cache()` is called. This reduces peak VRAM usage and makes the feature viable on 8GB cards.
7. **File Cleanup** – Both the temporary dialogue file and the temporary music file are deleted. Only the final mixed file remains in the `results/` directory.
8. **Output Naming** – The output file is named with an `_m` suffix, e.g., `voder_tts_dialogue_20250212_143022_m.wav`. This makes it immediately clear that the file contains background music.

#### GUI Workflow

When you click **Generate** in TTS or TTS+VC mode **and** the script contains at least one line with a colon (i.e., dialogue mode), VODER displays a clean modal dialog before any processing begins:

<p align="center">
  <i>Background Music Dialog</i><br>
  <code>Enter music description (or press Skip):</code><br>
  <code>[ OK ] [ Skip ]</code>
</p>

- **OK**: If you enter a non‑empty description and click OK, VODER will proceed with music generation as described above. If the description is empty, a warning is shown and you are returned to the dialog.
- **Skip**: Clicking Skip bypasses music generation entirely.

The dialog is styled consistently with the rest of VODER's GUI and respects the same color scheme and font choices.

#### Interactive CLI Workflow

In interactive CLI mode (TTS or TTS+VC), after you have entered all script lines and provided all voice prompts/audio paths, VODER asks:

```
Add background music? (y/N):
```

- If you type `y` or `yes`, it then prompts:
  ```
  Music description:
  ```
  Enter your description (e.g., `soft piano, cinematic`). If you press Enter without typing anything, the description is considered empty and VODER **skips** music generation (no warning; it's treated as a normal skip).
- If you type anything else (or just press Enter), music is skipped.

This flow is natural, non‑intrusive, and requires only one extra decision point.

#### One‑Liner CLI Workflow

For one‑liner commands, the `music` parameter is used:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep" "Sarah: bright" music "soft piano"
```

- If the `music` parameter is **present and its value is non‑empty**, background music is generated.
- If the `music` parameter is **present but its value is an empty string** (`music ""`), it is ignored (no music).
- If the `music` parameter is **absent**, no music is generated.
- If the script is **not** in dialogue mode (i.e., all `script` parameters are plain text without colon), the `music` parameter is ignored and a warning is printed.

This design allows automated scripts to optionally include music without breaking existing workflows.

#### Technical Implementation

The feature is implemented in `ProcessingThread` with two new modes:

- `tts_voice_design_dialogue` – handles TTS dialogue + optional music
- `tts_vc_dialogue` – handles TTS+VC dialogue + optional music

Both modes follow the same pattern:

```python
# 1. Generate dialogue audio
dialogue_temp = synthesize_and_concat(...)

# 2. If music_description is not None:
if music_description:
    # 3. Get dialogue duration
    duration = get_audio_duration(dialogue_temp)
    # 4. Generate music with ACE‑Step
    music_temp = ace.generate(lyrics="...", style_prompt=music_description, duration=duration)
    # 5. Mix at 35% volume using FFmpeg
    mixed_temp = ffmpeg_mix(dialogue_temp, music_temp, volume=0.35)
    # 6. Replace output with mixed file
    os.replace(mixed_temp, output_path)
    # 7. Clean up temporary files
    os.unlink(dialogue_temp)
    os.unlink(music_temp)
else:
    os.replace(dialogue_temp, output_path)
```

**Why 35%?** This value was determined through listening tests: at 35% relative volume, the music is clearly audible but does not compete with the spoken word for attention. Higher volumes (>40%) begin to mask speech; lower volumes (<30%) become too subtle. The value is hardcoded for consistency – there is no user‑adjustable volume control, because that would introduce another variable and complicate the user experience. If you need different mixing levels, you can always post‑process the output with an external audio editor.

**Why `"..."` as lyrics?** ACE‑Step requires a non‑empty lyrics string. Using three dots `...` is a conventional placeholder that reliably produces instrumental music with no discernible vocals. It has been tested across many style prompts and consistently yields the desired ambient track.

**Why auto‑fit duration?** Manually specifying a duration would create two problems: (1) the user would need to know the exact dialogue length in advance, and (2) the music would either be cut off or fade out before the dialogue ends. By auto‑fitting, VODER guarantees that the music plays for the entire dialogue and stops exactly when the speech ends. This creates a polished, professional feel.

**Memory optimisation:** The dialogue generation stage loads either Qwen3‑TTS VoiceDesign or Qwen3‑TTS Base. After the dialogue file is written, these models are allowed to be garbage‑collected. When music is requested, ACE‑Step is loaded, used, and then explicitly deleted with `del self.ace_tt` followed by `torch.cuda.empty_cache()`. This frees GPU memory before the next operation (which is none, since mixing is done via FFmpeg on CPU). This careful management makes the feature usable even on 8GB GPUs.

**File naming:** The `_m` suffix is added to the base filename. This is a simple, visible indicator that the file contains background music. It also prevents accidental overwriting of the non‑music version if you generate both variants.

**Cleanup:** All temporary files (individual dialogue segments, concatenated dialogue, generated music) are deleted. Only the final output file remains in `results/`. This keeps your working directory tidy and avoids accumulating gigabytes of intermediate audio.

---

## Tips & Tricks

### Getting Better Results

**For STT (Speech-to-Text):**

- Use audio with minimal background noise for best transcription accuracy
- The `timestamp` flag is useful for creating subtitles or navigating long recordings
- The `dialogue` flag is essential for multi‑speaker content — without it, all text is attributed to a single speaker
- Batch processing multiple files is more efficient than running them one at a time
- For YouTube videos, the download quality can affect transcription — prefer higher‑quality sources

**For TTS (Voice Design):**

- Be specific in voice prompts — "warm adult female" is better than just "female"
- Include pacing hints if you want specific rhythm — "slow, deliberate" or "fast, energetic"
- Mention the use case if relevant — "podcast host" or "news broadcast voice"
- Experiment with variations — small prompt changes can significantly affect output

**For TTS+VC (Voice Cloning):**

- Use 10‑30 seconds of clean reference audio
- Avoid background music or noise in reference
- Ensure consistent volume throughout reference
- Single continuous speech is better than multiple short clips
- The reference voice quality directly affects clone quality

**For STS (Voice Conversion):**

- Base and target should have similar audio characteristics
- If base is phone‑quality, target should also be phone‑quality
- Very short clips (under 2 seconds) may not convert well
- Very long clips (over 5 minutes) may cause memory issues
- Clear speech converts better than expressive/emphatic speech

**For TTM (Music Generation):**

- Structure lyrics with verse/chorus markers for better organization
- Keep lyrics simple for more coherent results
- Style prompts work best when specific — "80s synthpop" is better than "good"
- Shorter durations (30‑60 seconds) are more reliable
- Complex lyrics with unusual structures may produce inconsistent results

**For TTM+VC:**

- Because this mode chains two models, it uses more VRAM than either alone
- The memory optimisation now frees ACE‑Step before loading Seed‑VC, reducing OOM risk
- If you encounter memory issues, try reducing duration or processing shorter segments

### Multi-Speaker Scenarios

**Use Dialogue Mode for Manual Scripts:**

If you have a script and voice references ready, dialogue mode is the way to go. See [Dialogue Input in GUI](#dialogue-input-in-gui) and [Dialogue Input in CLI](#dialogue-input-in-cli) for details.

**Use Dialogue Source Analysis for Existing Audio:**

If you have an existing recording (podcast, interview, meeting) and want to recreate it with different voices, use dialogue source analysis. VODER will identify the speakers, transcribe their speech, and generate a dialogue script that you can then produce with cloned voices.

**Dialogue Planning:**

1. Write your script with character names (or let VODER generate it from source analysis)
2. Gather reference audio for each character (10‑30 seconds each, or use voice clip extraction)
3. In GUI: load references, assign via dropdowns
4. In CLI: provide references via repeated `target` parameters or interactively
5. (Optional) Decide if you want background music
6. Generate dialogue in one operation
7. Review and iterate if needed

**Character Consistency:**

Once you've assigned a voice reference to a character, keep using the same reference for that character throughout the project. Changing references mid‑dialogue creates inconsistent results.

### Using Same Audio Source

**The Modification Trick:**

For STT+TTS mode, if you use the same audio file as both base (content) and target (voice), you get voice modification. The transcribed text becomes editable, and the synthesis uses the same voice characteristics from the original audio. This allows you to:

- Change words or phrases while keeping the original voice
- Fix awkward phrasing while maintaining voice consistency
- Localize content while preserving original voice characteristics
- Create fictional quotes from real voice samples

**When This Works Best:**

- Reference audio is clean and of good quality
- You want minimal change to the overall delivery
- You're making small edits, not rewriting entire passages
- The original voice has clear, consistent characteristics

### Voice Cloning Best Practices

**Reference Audio Quality Hierarchy:**

| Quality | Characteristics | Result |
|--------|-----------------|--------|
| Excellent | Studio recording, no noise, consistent volume | Best clone quality |
| Good | Clean recording, minimal background, consistent | Good clone quality |
| Acceptable | Some background, slight inconsistencies | Acceptable quality |
| Poor | Heavy noise, compression artifacts, inconsistent | Poor clone quality |

**What to Avoid in Reference Audio:**

- Background music or sounds
- Multiple speakers (even briefly)
- Extreme volume variations
- Phone‑quality or highly compressed audio
- Emotional extremes that distort voice characteristics
- Audio that has been heavily processed or filtered

**The 10‑30 Second Sweet Spot:**

Reference audio between 10 and 30 seconds produces the best results. Shorter references may not capture enough voice characteristics. Longer references don't significantly improve quality and take longer to process.

### Background Music Best Practices

**When to Use It:**

Background music enhances dialogue when used tastefully. It's particularly effective for:

- Podcast intros and outros
- Narrative storytelling (audiobooks, guided meditations)
- Cinematic dialogue scenes
- Interview segments with ambient backing
- Educational content that benefits from a relaxed mood

**When to Skip It:**

Not every dialogue needs music. Consider skipping if:

- The content is informational/dry (music can be distracting)
- You plan to add music later in post‑production
- The dialogue itself is the primary focus (e.g., news reading)
- You're testing or iterating rapidly (music generation adds time)

**Choosing a Music Description:**

The style prompt for background music should match the mood of the content. Some guidelines:

| Mood | Description Example |
|------|---------------------|
| Relaxed, thoughtful | "soft piano, gentle strings, ambient" |
| Energetic, upbeat | "upbeat electronic, modern production" |
| Mysterious, suspenseful | "dark ambient, low drone, cinematic" |
| Inspirational, uplifting | "orchestral, emotional, building crescendo" |
| Corporate, professional | "corporate background, subtle, professional" |

**Avoid** overly complex or specific descriptions like "solo violin in D minor, arpeggios, with reverb" – ACE‑Step works better with broader genre/mood cues.

**Duration Handling:**

The music is always exactly as long as the dialogue. This means:

- If your dialogue is 42.7 seconds, the music will be 42 seconds (rounded to nearest whole second).
- No fade‑out is applied; the music stops abruptly when the dialogue ends. This is intentional – if you need a fade, you can add it later.
- Very short dialogue (<10 seconds) may still generate music, but ACE‑Step performs best with durations ≥10 seconds.

**Volume Level:**

The fixed 35% volume has been carefully chosen. If you find it too loud or too soft, you can adjust it with an external audio editor. We do not provide a user‑adjustable volume slider because it would add complexity and likely be misused (e.g., set to 100% and then complain that the music overpowers the speech).

**Performance Impact:**

Generating background music adds approximately 10‑20 seconds of processing time for a 30‑second dialogue (on a modern CPU). On GPU, it is faster. This is usually negligible compared to the time saved by not having to manually find, edit, and mix a music track.

**File Management:**

All temporary files are deleted. Only the final `.wav` file (with the `_m` suffix) remains in `results/`. If you need both the dialogue‑only and music‑mixed versions, generate twice (once without music, once with). The naming convention prevents accidental overwrites.

### Diarization Best Practices

**Getting Good Diarization Results:**

- **Clear audio is essential** — diarization accuracy drops sharply with noisy recordings. If you have a choice, use the cleanest audio source available.
- **Distinct speakers help** — two male voices of similar pitch and cadence are harder to separate than a male and female voice. The more acoustically different the speakers, the better the results.
- **Longer recordings are better** — give Pyannote at least 30 seconds of audio. Short clips (<10 seconds) may not contain enough speaker turns for accurate identification.
- **Avoid constant overlap** — if speakers talk over each other continuously, Pyannote may struggle. The best results come from turn‑based conversations.
- **Check speaker count** — if the recording has 2 speakers but Pyannote detects 3, it may be splitting one speaker's turns (possibly due to background noise variation). Review the output and correct if needed.

### YouTube Download Tips

- **Use stable URLs** — avoid URLs with expiring tokens or session IDs
- **Short videos download faster** — a 5‑minute clip is practical; a 2‑hour lecture will take significant time to download and process
- **yt-dlp updates** — if downloads start failing, try updating yt-dlp: `pip install -U yt-dlp`
- **Quality** — yt-dlp downloads the best available audio quality by default. Higher quality means better transcription and diarization results
- **Region restrictions** — some videos are geo‑blocked. If a download fails, try using a VPN or download the video manually with another tool

### OCR Accuracy Tips

**Getting Good OCR Results from Images:**

- **Use clear, high‑resolution images** — blurry or low‑resolution images produce poor OCR results
- **Good lighting matters** — evenly lit text without harsh shadows or glare is ideal
- **Straight text works best** — perspective distortion (text at an angle) reduces accuracy
- **Simple backgrounds** — text on a plain background is easier to read than text on complex images
- **Common fonts** — standard fonts like Arial, Helvetica, and Times New Roman are recognized more reliably
- **Supported languages** — EasyOCR supports 80+ languages. English is the most reliable. For CJK languages (Chinese, Japanese, Korean), ensure the text is large enough for the character set
- **Post‑processing** — always review OCR output. It may contain minor errors (e.g., "rn" read as "m", "0" read as "O"). This is normal for any OCR system

### Voice Clip Extraction Best Practices

**Choosing Good Source Audio:**

- Each speaker should have at least 10‑15 seconds of continuous speech somewhere in the recording. The extraction picks the **longest** segment per speaker, so longer stretches of solo speech produce better reference clips.
- Avoid using source audio where a speaker's longest segment contains background music, sound effects, or other speakers faintly in the background.
- If a speaker's longest segment is only 3‑5 seconds, consider finding a different source recording where they speak for longer. Short clips may not capture enough voice characteristics for convincing cloning.
- Recordings with clear turn‑taking (one person speaks, then the other responds) work better than recordings with frequent interruptions or overlapping speech.

---

## Version Information

**Timestamp‑Based Versioning:**

VODER uses timestamp‑based versioning rather than semantic versioning (v1.2.3, etc.). Each build is identified by its creation timestamp in YYYYMMDD_HHMMSS format. This approach reflects VODER's development philosophy — continuous improvement rather than numbered releases.

**Why Not Semantic Versioning:**

Traditional semantic versioning implies discrete releases with specific feature sets and bug fixes between versions. VODER development doesn't follow that pattern. Changes are made when they're ready, tested, and merged. A user downloading VODER today gets the absolute latest version with all improvements since the last commit.

**Version Tracking:**

- Each commit to the main branch gets a timestamp
- The CHANGELOG.md documents significant changes with dates
- No numbered releases means no "latest stable version" confusion
- Everyone always uses the current development version

**No PyPI Package:**

Unlike IMDER, VODER is not distributed via PyPI. Running from source is the only way to use VODER. This ensures:

- Always access to latest features
- No version compatibility issues
- Direct access to development version
- Transparency in what's running

---

## Troubleshooting & Common Issues

### STS/TTM+VC Fails Immediately

**Cause:** No NVIDIA GPU detected or insufficient VRAM

**Solution:**
```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Check VRAM
nvidia-smi
```

If no GPU is detected, Seed‑VC modes cannot work. These modes require NVIDIA GPU with minimum 8GB VRAM.

### Out of Memory Errors

**Cause:** Model too large for available memory

**Solution:**
- Process shorter audio segments
- Reduce TTM duration (shorter music = less memory)
- Close other GPU‑intensive applications
- Ensure no other processes are using GPU memory

### Insufficient Memory for STT + Diarization

**Cause:** Running both Whisper (~4GB) and Pyannote (~2-3GB) requires approximately 15GB total RAM

**Solution:**
- Ensure your system has at least 15GB RAM available
- Close other memory‑intensive applications
- If you only need transcription without speaker identification, omit the `dialogue` flag (reduces memory to ~12GB)
- Process shorter audio files

### FFmpeg Not Found

**Cause:** FFmpeg not installed or not in system PATH

**Solution:**
```bash
# Verify FFmpeg installation
ffmpeg -version

# Install if needed
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### HuggingFace Model Download Fails

**Cause:** Network issues or gated repository access

**Solution:**
1. Check internet connection
2. Add HuggingFace token to `src/HF_TOKEN.txt` for gated models
3. Clear cache and retry:
   ```bash
   rm -rf src/models/
   python src/voder.py
   ```

### Pyannote / HF_TOKEN Issues

**Cause:** Missing or invalid HuggingFace token for Pyannote's gated models

**Solution:**
1. Visit [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and accept the user agreement
2. Visit [https://huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and accept the user agreement
3. Create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — a "Read" token is sufficient
4. Place the token in `src/HF_TOKEN.txt` (just the token string, nothing else on the line)
5. Ensure the file has no trailing whitespace or newline characters
6. If you see a `401 Unauthorized` or `403 Forbidden` error, double‑check that you've accepted the agreements for both models

### yt-dlp Download Fails

**Cause:** yt-dlp not installed, outdated, or the video is unavailable

**Solution:**
```bash
# Install yt-dlp
pip install yt-dlp

# Update to latest version
pip install -U yt-dlp

# Test manually
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID" -x --audio-format wav -o test_audio.%(ext)s
```

If the video is private, age‑restricted, or geo‑blocked, yt-dlp cannot download it. Download the audio manually using a browser extension or alternative tool, then provide the local file to VODER.

### EasyOCR Errors

**Cause:** EasyOCR not installed, or unsupported image format

**Solution:**
```bash
# Install EasyOCR
pip install easyocr

# Verify installation
python -c "import easyocr; print(easyocr.Reader(['en']))"
```

If you get a `Pillow` error, ensure Pillow is installed: `pip install Pillow`. If OCR produces garbled output, try using a higher‑resolution image or pre‑processing the image to improve contrast.

### Diarization Quality Issues

**Cause:** Poor audio quality, too many speakers, or very similar voices

**Solution:**
- Use the cleanest audio source available
- If possible, limit to 2‑4 speakers
- Ensure speakers have distinct voice characteristics
- Remove background music before running diarization
- For recordings with heavy noise, consider pre‑processing with noise reduction
- Review the output and manually correct speaker labels if needed

### Voice Cloning Produces Poor Results

**Cause:** Poor quality reference audio

**Solution:** Use high‑quality reference audio:
- 10‑30 seconds duration
- Clear speech, minimal background noise
- Single speaker, no music
- Consistent volume levels
- No post‑processing or effects

### Dialogue Character Not Found (GUI)

**Cause:** Character name mismatch between script and voice prompt assignments

**Solution:** The GUI automatically tracks characters from the script rows and displays prompts for each. If a character is missing from the voice prompt area, it means no row in the script uses that character. Ensure the character name is spelled consistently across all rows.

### Dialogue Character Not Found (CLI)

**Cause:** Missing `voice` or `target` entry for a character in one‑liner mode

**Solution:** Check that every character that appears in `script` parameters also appears exactly once in the corresponding `voice`/`target` parameters, with the same spelling (case‑insensitive). Example:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep voice" "Sarah: cheerful voice"
```

### GUI: Audio Dropdowns Not Appearing

**Cause:** TTS+VC mode requires at least one audio file loaded

**Solution:** Click **"Add Audio"** and load at least one reference file. The dropdowns will populate automatically.

### Background Music Not Added (GUI)

**Cause:** You pressed Skip or left the description empty

**Solution:** In the dialog, enter a non‑empty description and click OK. If you accidentally skipped, you must regenerate with the correct option.

### Background Music Not Added (One‑Liner)

**Cause:** The `music` parameter was omitted, its value was empty, or the script was not in dialogue mode

**Solution:** Ensure:
- At least one `script` parameter contains a colon (`Character: text`)
- You include `music "description"` with a non‑empty string
- You are in TTS or TTS+VC mode

### Background Music Generation Fails

**Cause:** ACE‑Step model not loaded, insufficient resources, or invalid music description

**Solution:**
- Check that you have sufficient RAM (16GB recommended)
- Try a simpler music description (e.g., "piano")
- Verify that FFmpeg is installed and in PATH
- If using GPU, ensure you have at least 8GB VRAM (or use CPU – slower but works)

### Music Volume Seems Off

**Cause:** Subjective perception; 35% is a fixed default

**Solution:** If you consistently find the volume too high or too low, you can post‑process the output file with an audio editor. For automated workflows, you could add an FFmpeg command after generation to adjust the volume further.

### Quality Issues with TTM

**Cause:** Complex lyrics or ambitious style prompts

**Solution:**
- Simplify lyrics structure
- Use more conventional style descriptions
- Try shorter durations first
- Start with well‑known genres ("pop", "rock") before experimenting

### Mode-Specific Reference

| Mode | Common Issue | Solution |
|------|--------------|----------|
| STT | Poor transcription | Use cleaner audio, reduce background noise |
| STT + dialogue | Bad speaker identification | Ensure distinct voices, minimal noise, 30+ seconds of audio |
| STT + dialogue | HF_TOKEN error | Accept Pyannote agreements, add valid token to HF_TOKEN.txt |
| STT (YouTube) | Download fails | Update yt-dlp, check URL, try local download |
| STT (image) | OCR errors | Use higher‑resolution images, improve lighting |
| Voice clip extraction | Short or poor clips | Use longer source recordings, cleaner audio |
| STT+TTS | Multi‑speaker confusion | Use dialogue source analysis instead |
| TTS | Unnatural voice | More detailed prompts |
| TTS+VC | Clone quality issues | Better reference audio |
| STS | Conversion fails | Shorter input, check VRAM |
| TTM | Inconsistent music | Shorter duration, simpler lyrics |
| TTM+VC | Out of memory | Memory optimisation already helps; try shorter duration |
| Dialogue (any) | Missing character assignment | Ensure every character has a voice prompt/audio path |
| Dialogue (music) | Music not generated | Use non‑empty description, ensure dialogue mode |

---

## Final Notes

VODER is a tool built for creators, developers, and audio professionals who need professional‑grade voice processing without subscription fees or usage limits. It prioritizes quality over speed, simplicity over complexity, and utility over marketing.

All seven processing modes work reliably. The "problematic modes" designation from earlier versions is outdated — Seed‑VC v2 has proven stable across the use cases VODER supports. If you encounter issues, they're more likely to be related to resource constraints or input quality than mode‑specific bugs.

**Dialogue is now everywhere.** The GUI provides a visual, error‑free script editor with dropdown voice assignments. The CLI offers both interactive and one‑liner dialogue creation. Dialogue can be optionally enhanced with automatically generated, duration‑fitted background music. And with dialogue source analysis, you can now start from existing multi‑speaker recordings and let VODER generate the script for you.

**Multi‑speaker input is no longer a limitation.** The previous "Why Not Multi-Speaker Input?" section explained the technical challenges. With Pyannote diarization and Whisper word‑level alignment, those challenges have been overcome. VODER can now accept multi‑speaker audio, identify who said what, extract voice clips, and generate complete dialogue productions — all from a single source file or URL.

**STT mode brings standalone transcription.** VODER isn't just about generating audio anymore — it can also transcribe it. With timestamps, speaker diarization, batch processing, and YouTube support, STT mode is a full‑featured transcription tool.

**Background music completes the pipeline.** With it, VODER transforms from a voice processor into a complete audio production workstation. Podcasters can generate entire episodes with music beds. Storytellers can add cinematic ambience. Educators can create engaging narrated content. All of this is possible because we integrated the music generation model we already had (ACE‑Step) into the dialogue pipeline in a thoughtful, user‑friendly way.

**Choose the interface that fits your workflow.** If you love visual interaction, use the GUI. If you live in the terminal, use the interactive CLI. If you're an AI agent or need to automate thousands of generations, use the one‑liner. Every interface has full access to dialogue, voice cloning, background music, transcription, diarization, and voice clip extraction.

**Remember:** Quality over speed. Use dialogue mode for multi‑speaker content. Reference audio quality matters. Music descriptions should match the mood. Ensure your HF_TOKEN is set for diarization. And when in doubt, start with simpler configurations before experimenting with advanced workflows.

For questions, issues, or collaboration opportunities, visit the GitHub repository or reach out through (X)[https://x.com/HAKORAdev].

---

*VODER — They say what you want them to say.*
