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
    - [TTS SLC: Speaker Language Conversion](#tts-slc-speaker-language-conversion)
    - [TTS SVC: Speaker Voice Change](#tts-svc-speaker-voice-change)
    - [TTS Modify Speech (STT+TTS)](#tts-modify-speech-stttts)
  - [Voice Training](#voice-training)
  - [STS: Speech-to-Speech Voice Conversion](#sts-speech-to-speech-voice-conversion)
  - [TTM: Text-to-Music](#ttm-text-to-music)
  - [SE: Speech Enhancement](#se-speech-enhancement)
  - [SFX: Sound Effects Generation](#sfx-sound-effects-generation)
  - [SVS: Song Voice Separate](#svs-song-voice-separate)
  - [SS: Speakers Separator](#ss-speakers-separator)
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
  - [Script Directives](#script-directives)
    - [Time Positioning](#time-positioning)
    - [Volume Level Control](#volume-level-control)
    - [Duration for SFX](#duration-for-sfx)
  - [SFX Lines in Dialogue](#sfx-lines-in-dialogue)
  - [Optional Background Music for Dialogue](#optional-background-music-for-dialogue)
    - [How It Works](#how-it-works-1)
    - [GUI Workflow](#gui-workflow)
    - [Interactive CLI Workflow](#interactive-cli-workflow)
    - [One‑Liner CLI Workflow](#one-liner-cli-workflow)
    - [Music Volume Level Control](#music-volume-level-control)
    - [Technical Implementation](#technical-implementation)
- [TTM Mode: Instrumental Option](#ttm-mode-instrumental-option)
  - [Creating Instrumental Music](#creating-instrumental-music)
  - [Contextual Lyrics](#contextual-lyrics)
- [Tips & Tricks](#tips--tricks)
  - [Getting Better Results](#getting-better-results)
  - [Multi-Speaker Scenarios](#multi-speaker-scenarios)
  - [Using Same Audio Source (Auto-Clone Trick)](#using-same-audio-source-auto-clone-trick)
  - [Voice Cloning Best Practices](#voice-cloning-best-practices)
  - [Background Music Best Practices](#background-music-best-practices)
  - [Diarization Best Practices](#diarization-best-practices)
  - [YouTube Download Tips](#youtube-download-tips)
  - [OCR Accuracy Tips](#ocr-accuracy-tips)
  - [Voice Clip Extraction Best Practices](#voice-clip-extraction-best-practices)
  - [Sound Effects Best Practices](#sound-effects-best-practices)
  - [Speech Enhancement Best Practices](#speech-enhancement-best-practices)
  - [SLC Tricks: Music Preservation & Voice Fidelity](#slc-tricks-music-preservation--voice-fidelity)
  - [STS Mimic Language Warning](#sts-mimic-language-warning)
  - [Auto Vocal Extraction Trick](#auto-vocal-extraction-trick)
  - [Overdose STT Trick](#overdose-stt-trick)
  - [Extreme TTS Trick](#extreme-tts-trick)
  - [Video STS Trick](#video-sts-trick)
  - [TTM Sub-Task Tricks](#ttm-sub-task-tricks)
- [Version Information](#version-information)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Introduction & Vision

VODER is a professional‑grade voice processing tool that brings together **eight distinct audio transformation capabilities** in a single, unified interface. Unlike tools that force you to jump between multiple applications for different voice‑related tasks, VODER provides everything from standalone transcription to text‑to‑speech synthesis with voice cloning (including speaker language conversion and speech modification) to music generation with multi‑track control to sound effects to speech enhancement to voice separation to speaker identification under one roof.

**What VODER Actually Does:**

At its core, VODER orchestrates state‑of‑the‑art AI models to perform voice‑related transformations. It can transcribe speech to text with speaker identification and optional translation, generate speech from text using either designed voices or cloned references, transform one voice into another while preserving content, create music from lyrics with optional voice conversion for the vocalist and advanced sub‑tasks for track‑level control, generate sound effects from text descriptions, enhance speech quality through denoising and dereverberation, separate vocals from music using source separation, translate speech across languages while preserving voice identity, extract individual speakers from multi‑speaker audio, download and analyze content directly from YouTube and other video platforms, extract voice clips from multi‑speaker audio for use as cloning references, and even read text from images using optical character recognition. This isn't about chasing the fastest processing times or highest frame rates — it's about achieving professional‑quality results that actually sound good.

**Why VODER Exists:**

The voice synthesis market is dominated by expensive commercial platforms that charge per character or per month. ElevenLabs, OpenAI, and others offer powerful capabilities, but at costs that add up quickly for creators, developers, and businesses alike. More importantly, no existing open‑source solution offered all eight processing capabilities in a unified interface. You could find separate tools for TTS, voice conversion, music generation, voice separation, and speaker identification, but none that worked together seamlessly — and certainly none that could pull a video from YouTube, separate the vocals, identify the speakers, extract voice references, translate between languages while preserving voice, and generate a complete dialogue with background music and sound effects.

VODER was built to fill this gap. The goal from day one was to create a local, free, open‑source alternative that doesn't compromise on quality. Is it perfect? No software is. But it works, it keeps improving, and it provides genuine utility without subscription fees or usage limits.

**What Makes VODER Different:**

Most voice processing tools focus on a single use case. VODER takes a different approach — it treats voice and audio processing as a unified problem space. The same interface that generates speech from text can also convert that speech between voices, and the same voice cloning technology can apply to both speech and singing. The same transcription engine that powers speech‑to‑text also drives speaker diarization for multi‑speaker analysis. The same voice separation engine that isolates vocals for cloning also cleans up inputs for STT and STS. The same sound generation model that creates background music can also produce custom sound effects. The same translation pipeline that handles language conversion can also preserve voice identity across languages. This integration enables workflows that would otherwise require multiple tools and significant manual effort.

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
| STT + Translate | 8GB | +4GB (Whisper Turbo) +~3GB (large-v3) | 15GB | CPU only | N/A |
| STT + Diarization | 8GB | +4GB (Whisper) +2-3GB (Pyannote) | 15GB | CPU only | N/A |
| STT + Overdose | 8GB | +~8GB (VibeVoice ASR) | 16GB | Optional | 24GB (recommended) |
| TTS (VoiceDesign, no music) | 8GB | +4GB (Qwen) | 12GB | Optional | 4GB (GTX 1060) |
| TTS (VoiceDesign, with music) | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| TTS (Voice Clone, no music) | 8GB | +4GB (Qwen Base) +~3GB (SVS) | 15GB | Optional | 4GB |
| TTS (Voice Clone, with music) | 8GB | +15GB (ACE) +~3GB (SVS) | 26GB | Optional | 15GB |
| TTS + Overdose | 8GB | +~40GB (VibeVoice ASR) + 15GB (ACE XL, if music) | 48GB | Optional | 24GB VRAM or 48GB RAM |
| TTS (SLC) | 8GB | +~3GB (Whisper large-v3) +4GB (Qwen) +~3GB (SVS) | 18GB | Optional | 4GB |
| TTS (SLC Overdose) | 8GB | +~3GB (Whisper large-v3) +4GB (Qwen) +~3GB (SVS) +5GB (Seed-VC v2) | 23GB | Optional | 14GB |
| TTS (SVC) | 8GB | +~3GB (Whisper turbo) +4GB (Qwen) +~3GB (SVS) | 18GB | Optional | 4GB |
| TTS (SVC Overdose) | 8GB | +~8GB (VibeVoice ASR) +4GB (Qwen) +~3GB (SVS) +5GB (Seed-VC v2) | 22GB | Optional | 14GB |
| TTS (Modify Speech) | 8GB | +4GB (Whisper) +4GB (Qwen) +~3GB (SVS) | 19GB | Optional | 4GB |
| TTS (Extreme, no music) | 8GB | +~10GB (Fish S2-Pro) +~3GB (SVS) | 21GB | Optional | 8GB |
| TTS (Extreme + Overdose) | 8GB | +~8GB (VibeVoice ASR) +~10GB (Fish S2-Pro) +~3GB (SVS) | 29GB | Optional | 24GB |
| TTS (Extreme + music) | 8GB | +~10GB (Fish S2-Pro) +15GB (ACE) +~3GB (SVS) | 36GB | Optional | 16GB |
| STS | 8GB | +5GB (Seed-VC) +~3GB (SVS) | 16GB | Optional | 14GB |
| TTM (standard) | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| TTM (overdose) | 8GB | +~24GB (ACE-Step XL-Turbo) | 32GB | Optional | 32GB (RTX 4090) |
| TTM (VC enabled) | 8GB | +15GB (ACE) +5GB (Seed-VC) +~3GB (SVS) | 31GB | Optional | 16GB |
| TTM (complete sub-task) | 8GB | +~24GB (ACE-Step XL-Turbo) +~3GB (SVS) | 35GB | Optional | 32GB (RTX 4090) |
| TTM (complete + SFX overlay) | 8GB | +~24GB (ACE-Step XL-Turbo) +~3GB (SVS) +~3-4GB (TangoFlux, ACE offloaded first) | 38GB | Optional | 32GB (RTX 4090) |
| TTM (complete, SFX only) | 8GB | +~3GB (SVS) +~3-4GB (TangoFlux, no ACE loaded) | 15GB | Optional | 4GB |
| TTM (BGM + SFX overlay) | 8GB | +~3GB (SVS) +15GB (ACE) +~3-4GB (TangoFlux, ACE offloaded first) | 29GB | Optional | 15GB |
| TTM (BGM, SFX only) | 8GB | +~3GB (SVS) +~3-4GB (TangoFlux, no ACE loaded) | 15GB | Optional | 4GB |
| SE | 8GB | +2-3GB (UniSE) | 11GB | Optional | 4GB |
| SFX | 8GB | +3-4GB (TangoFlux) | 12GB | Optional | 4GB |
| SVS | 8GB | +~3-4GB (BS-RoFormer) | 12GB | Optional | 4GB |
| SS (standard) | 8GB | +4GB (Whisper) +2-3GB (Pyannote) +2-3GB (UniSE TSE) +~3GB (SVS) | 20GB | Optional | 4GB |
| SS (overdose) | 8GB | +~8GB (VibeVoice ASR) +2-3GB (UniSE TSE) +~3GB (SVS) | 24GB | Optional | 24GB (recommended) |

- **CPU**: 4-6 cores minimum for model loading and non-GPU operations
- **RAM**: 12GB minimum for basic modes (STT, TTS VoiceDesign, SE, SFX, SVS), 15-16GB for modes with voice cloning or diarization, 23GB for standard ACE-related modes (TTM, TTS with music), 32GB+ for overdose and complete modes
- **GPU (CUDA)**: Optional - all modes work on CPU. GPU acceleration significantly speeds up STS, TTM, and modes using Seed-VC or ACE-Step
- **VRAM**: 4GB minimum (6GB recommended, 16GB for best performance with music modes, 32GB for overdose modes). STT and diarization modes are CPU-only and require no GPU.
- **Storage**: SSD recommended for model downloads and result saving

**VRAM Guidelines:**

| VRAM | Performance Level | Suitable Modes |
|------|-------------------|----------------|
| No GPU (CPU only) | Slow | All modes (STT, STT+diarization, OCR, SE, SFX, SVS included) |
| 4GB | Usable | TTS (VoiceDesign), TTS (SLC), TTS (SVC), TTS (Modify Speech), SE, SFX, SVS |
| 6GB | Minimum | TTS (VoiceDesign), TTS (SLC), TTS (SVC), TTS (Modify Speech), SE, SFX, SVS |
| 14GB | Mid-range | STS, all TTS modes, SE, SFX |
| 15-16GB | Recommended | TTS with music, TTM (standard), TTM+VC, all modes |
| 24GB | High | All standard modes at full speed, SS (overdose), STT (overdose) |
| 32GB | Maximum | TTM (overdose), TTM (complete), all modes at full speed (RTX 4090) |
| T4 (16GB) | Server-grade | All standard modes (not typical consumer GPU) |

These aren't arbitrary numbers. They're based on actual testing of the models VODER uses.

---

## Why Hardcoded Models?

VODER uses hardcoded default models. This isn't an accident or a limitation — it's a deliberate design choice made for quality reasons.

### The Quality Imperative

The models VODER uses were selected because they represent the best available quality in their respective categories. Qwen3‑TTS for text‑to‑speech, Seed‑VC v2 for voice conversion, ACE‑Step for music generation, Whisper for speech‑to‑text, Pyannote for speaker diarization, EasyOCR for image text extraction, UniSE for speech enhancement, TangoFlux for sound effects, BS‑RoFormer Resurrection for voice separation, VibeVoice ASR for advanced transcription with speaker identification, ACE‑Step XL‑Turbo for enhanced music generation — these aren't arbitrary choices. They're the result of evaluating multiple alternatives and selecting the ones that produce the best results.

Smaller models exist. Quantized variants exist. "Fast" versions exist. We deliberately don't use them because they produce noticeably worse output. A smaller TTS model sounds less natural, has more artifacts, and fails on complex text. A quantized voice conversion model loses the subtle characteristics that make voice cloning convincing. Using degraded models would undermine the entire purpose of having VODER exist.

**The HF_TOKEN.txt File:**

You'll find a file called `HF_TOKEN.txt` in the VODER directory. This file serves two important purposes:

1. It allows VODER to access gated model repositories (such as Pyannote's speaker diarization pipeline on HuggingFace).
2. It allows advanced users to modify model configurations if they really want to.

The file contains instructions for getting your HuggingFace token. If you provide a valid token, VODER will use it for gated model repositories — **this is required for speaker diarization to function**. See the [Speaker Diarization](#speaker-diarization) section for details on setting up your token.

**We Do Not Recommend Changing Models:**

This needs to be stated clearly. The hardcoded models are there because they're the best options available. If you have technical expertise and want to experiment with different model configurations, the capability exists. But VODER is optimized for its default configuration, and deviation from these defaults may produce worse results or cause errors.

Think of it like a restaurant that only serves one dish. They chose that dish because it's the best thing they can make. You can ask them to make something else, but it won't be as good as their specialty. VODER's specialty is orchestrating these specific models together — that's what it does best.

### Custom Model Support

For those who insist on changing things, the model paths can be configured by editing the `HF_TOKEN.txt` file. Each line can specify a model override using a specific format. See the `HF_TOKEN.txt` file itself for instructions on how to format custom model paths. But again — we don't recommend this unless you know exactly what you're doing.

### Custom Versions

If someone creates a modified version of VODER with different model configurations, that's exactly what it is: a modified version. Custom configurations won't be supported in the main VODER documentation or issue tracker because the main project only guarantees quality for its default configuration.

For those interested in exploring custom model configurations, we'll maintain a separate document (CUSTOM_VERSIONS.md) where community‑contributed modifications can be documented. These are not official VODER builds, but if you want to share your experiments with different models or configurations, that file provides a place to do so.

---

## Centralized Model Management

VODER now uses a centralized model storage system under `src/models/`. This is a structural improvement that eliminates the problem of model files being scattered across different directories.

**Directory Structure:**

```
src/models/
├── tmp/                      # Temporary downloads in progress
├── checkpoints/
│   ├── whisper/              # Whisper STT model (whisper-turbo.pt, whisper-large-v3.pt)
│   ├── qwen_tts_voicedesign/ # Qwen3-TTS VoiceDesign model
│   ├── qwen_tts_base/        # Qwen3-TTS Base model
│   ├── fish_s2pro/            # Fish Audio S2-Pro model (extreme TTS)
│   ├── seed_vc_v1/           # Seed-VC v1 (44.1kHz for music)
│   ├── seed_vc_v2/           # Seed-VC v2 (22.05kHz for speech)
│   ├── acestep/              # ACE-Step music generation models (turbo, xl-turbo)
│   ├── pyannote/             # Pyannote diarization pipeline
│   ├── easyocr/              # EasyOCR models and weights
│   ├── unise/                # UniSE speech enhancement model
│   ├── tangoflux/            # TangoFlux sound effects model
│   ├── svs/                  # BS-RoFormer Resurrection for voice/music separation
│   └── vibevoice_asr/        # VibeVoice ASR for advanced transcription
```

**HuggingFace Cache Redirection:**

Some models (particularly Pyannote, EasyOCR, UniSE, TangoFlux, VibeVoice ASR, and BS-RoFormer) are downloaded through HuggingFace. VODER sets the `HF_HOME` and `TRANSFORMERS_CACHE` environment variables to point to the `src/models/` directory. This means:

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

STT (Speech‑to‑Text) is a standalone transcription mode that converts audio, video, and images into text. It uses Whisper to transcribe speech with word‑level timestamps, and can optionally identify individual speakers using Pyannote diarization. It supports translation to English using Whisper large‑v3, and can even download and transcribe content directly from YouTube URLs. For maximum transcription quality, an Overdose mode using VibeVoice ASR is available for speaker‑aware transcription. Before transcription, SVS pre‑cleanup can isolate vocals from background music or noise.

This is VODER's first mode that doesn't produce audio output — its output is a text file.

**How It Works:**

1. **Input Handling**: VODER accepts multiple input types:
   - **Audio files** (WAV, MP3, FLAC, OGG, M4A, etc.)
   - **Video files** (MP4, MKV, AVI, MOV, etc.) — audio track is extracted automatically
   - **Image files** (PNG, JPG, JPEG, BMP, TIFF) — text is extracted via EasyOCR
   - **YouTube/URLs** — audio is downloaded via yt-dlp before transcription
2. **SVS Pre‑Cleanup** (optional): If enabled, BS‑RoFormer isolates the vocal track from music and background noise before transcription. This significantly improves transcription accuracy for songs or recordings with musical accompaniment.
3. **Transcription**: Whisper Turbo loads the audio and produces a transcript with word‑level timestamps
4. **Translation** (optional): When the `translate` flag is set, Whisper large‑v3 translates the audio to English with word‑level timestamps. This supports all 99 languages that Whisper large‑v3 handles.
5. **Overdose Mode** (optional): When the `overdose` flag is set, VibeVoice ASR replaces Whisper for transcription. VibeVoice provides higher‑quality speaker‑aware transcription with built‑in speaker identification, but requires 24GB+ VRAM or 48GB+ combined system memory. Overdose cannot be used with translate (ASR does not support translation).
6. **Optional Timestamps**: The `timestamp` flag adds formatted timestamps to the output
7. **Optional Diarization**: The `dialogue` flag runs Pyannote speaker diarization and attributes each segment to a speaker
8. **Output**: Results are saved as `.txt` files in the `results/` directory

**Dual-Model Architecture:**

STT mode uses a dual‑model architecture for flexibility:

| Task | Model | Purpose |
|------|-------|---------|
| Standard transcription | Whisper large-v3-turbo | Fast, accurate transcription with timestamps |
| Translation | Whisper large-v3 | High‑quality translation from 99 languages to English |
| Overdose transcription | VibeVoice ASR | Maximum quality with built‑in speaker identification |

When translation is requested, the large‑v3 model is loaded alongside or instead of the turbo model. When overdose is requested, VibeVoice ASR entirely replaces the Whisper pipeline. This architecture ensures each task uses the model best suited to it.

**Batch Processing:**

STT mode supports processing multiple files in a single command. When you provide multiple input paths (or a directory), VODER processes each file sequentially and produces a separate output text file for each.

**Output File Naming:**

| Input Type | Output Naming |
|------------|---------------|
| Audio file (`podcast.mp3`) | `voder_stt_podcast.txt` |
| Audio with timestamps | `voder_stt_podcast_timestamp.txt` |
| Audio with translate | `voder_stt_podcast_translate.txt` |
| Audio with diarization | `voder_stt_podcast_dialogue.txt` |
| Audio with translate + dialogue | `voder_stt_podcast_translate_dialogue.txt` |
| Audio with all flags | `voder_stt_podcast_timestamp_translate_dialogue.txt` |
| YouTube URL | `voder_stt_<video_id>.txt` |
| Image file (`slide.png`) | `voder_stt_slide.txt` |

The base filename is derived from the input filename (without extension). For YouTube URLs, the video ID is used.

**CLI Usage:**

```bash
# Basic transcription
python src/voder.py stt "audio.wav"

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With speaker diarization
python src/voder.py stt "audio.wav" dialogue

# With both timestamps and diarization
python src/voder.py stt "audio.wav" timestamp dialogue

# With translation to English
python src/voder.py stt "audio.wav" translate

# With translation and diarization
python src/voder.py stt "audio.wav" translate dialogue

# With overdose mode (higher quality, requires more VRAM)
python src/voder.py stt "audio.wav" overdose

# Transcribe a YouTube video
python src/voder.py stt "https://www.youtube.com/watch?v=VIDEO_ID" timestamp dialogue

# Batch process multiple files
python src/voder.py stt "file1.mp3" "file2.wav" "file3.mp4"

# Interactive CLI
python src/voder.py cli
# Select mode 1 (STT), then follow prompts
```

**Why It's Like That:**

The dual‑model approach exists because Whisper Turbo and Whisper large‑v3 serve different strengths. Turbo is optimized for speed and general transcription accuracy. Large‑v3, while slower, provides superior translation quality across its 99 supported languages. Rather than forcing a single model for all tasks, VODER picks the right tool for the job. The Overdose option exists for users with sufficient hardware who want the absolute best transcription quality — VibeVoice ASR provides native speaker identification that goes beyond what Whisper + Pyannote can achieve, but it demands serious GPU resources.

**Best For:**

- Transcribing podcasts, interviews, and meetings
- Creating subtitles or captions for video content
- Content analysis and text mining
- Accessibility — making audio content available to deaf/hard‑of‑hearing users
- Extracting text from images (screenshots, slides, scanned documents)
- Generating dialogue scripts from existing multi‑speaker audio
- Preparing voice reference clips for TTS voice cloning dialogue mode
- Transcribing songs with vocal isolation (SVS pre‑cleanup)
- Translating foreign language content to English
- Maximum quality transcription with Overdose mode

**Technical Notes:**

STT mode is entirely CPU‑based when using Whisper models. No GPU is required for Whisper transcription. Whisper Turbo provides an excellent balance of speed and accuracy. Processing time depends on audio length — approximately 1x real‑time on a modern CPU (a 10‑minute file takes about 10 minutes to transcribe).

When the `dialogue` flag is used, Pyannote's speaker diarization pipeline runs after Whisper transcription. The two outputs are aligned using a three‑tier system (see [Speaker Diarization](#speaker-diarization) for details).

When `overdose` is enabled, VibeVoice ASR requires a GPU with 24GB+ VRAM or 48GB+ combined system memory (RAM + Swap/Pagefile). It provides speaker‑aware transcription with built‑in speaker identification, producing output comparable to Whisper + Pyannote but with higher quality segmentation.

**Memory Requirements:** STT requires approximately 12GB RAM (8GB base + ~4GB for Whisper model). With translation enabled, it requires approximately 15GB RAM (dual model loading). With diarization enabled, it requires approximately 15GB RAM. With overdose mode, it requires approximately 16GB RAM on CPU, though 24GB+ VRAM is recommended for GPU acceleration.

---

### TTS: Text-to-Speech

**What It Does:**

TTS generates speech from text using Qwen3‑TTS. When no target voice reference is provided, Qwen3‑TTS VoiceDesign interprets a natural language voice prompt to create a generated voice. When a target reference is provided via the `target` parameter, Qwen3‑TTS Base generates speech and applies voice cloning to match the reference voice. This unified mode replaces the previous separate TTS and TTS+VC modes — a single mode handles both generated and cloned voices.

**How It Works:**

VODER automatically selects the appropriate TTS model based on whether voice cloning is requested:

- **VoiceDesign mode** (no `target` parameter): The VoiceDesign model interprets natural language descriptions to generate appropriate voice characteristics. Unlike traditional TTS systems that use pre‑recorded voice samples, VoiceDesign creates voices from scratch based on your description. This makes it incredibly flexible — you can describe voices that don't exist in any database. **Trained voices** can also be used via the `voice` parameter — see [Voice Training](#voice-training) for details.

- **Voice Clone mode** (`target` parameter provided): The process happens in two stages. First, Qwen3‑TTS Base generates speech from your text using its default voice characteristics. Before that, BS‑RoFormer automatically extracts clean vocals from the target reference audio via SVS (voice separation), ensuring the best possible cloning quality even if the reference has background music or noise. Then, the voice cloning system extracts distinctive features from the cleaned reference audio and applies them to the generated speech. The result is your text spoken by a voice that matches your reference.

- **Trained Voice mode** (`voice` parameter with a trained voice name or path): When a trained voice is used, Qwen3‑TTS Base (voice cloning) is used instead of VoiceDesign. The trained `.tts` file provides the voice embedding directly, producing consistent cloned voice output without needing a reference audio file. See [Voice Training](#voice-training) for details.

**Why It's Like That:**

The unified TTS mode exists because voice generation and voice cloning are fundamentally the same operation — they just differ in how the voice characteristics are determined. By combining them into a single mode, you get a more consistent interface and the ability to mix generated and cloned voices within the same dialogue. VoiceDesign exists because not everyone wants to clone an existing voice — sometimes you need a generic voice for narration, or you want to create a character voice that doesn't correspond to any real person. Voice cloning opens possibilities that pure TTS can't match — you can clone a specific person's voice and use it consistently across all your content.

**Language Support:**

TTS supports 10 languages via the `language` parameter. The `SUPPORTED_TTS_LANGUAGES` constant defines the available options:

| Code | Language | Code | Language |
|------|----------|------|----------|
| `zh` | Chinese | `de` | German |
| `en` | English | `fr` | French |
| `ja` | Japanese | `ru` | Russian |
| `ko` | Korean | `pt` | Portuguese |
| `es` | Spanish | `it` | Italian |

When `language` is not specified, VODER uses `"Auto"` which lets the model detect the language automatically.

**Auto Vocal Extraction from Target:**

When a `target` reference audio is provided, VODER automatically runs BS‑RoFormer vocal isolation to extract clean vocals before voice cloning. This means you can use a song clip, a video snippet, or any audio with background elements as your voice reference — VODER handles the cleanup internally. For multi-reference cloning (`(path1)(path2)(path3)`), each reference is cleaned individually via SVS and then concatenated into a single composite before voice extraction. If SVS extraction fails for any reason, the original target audio is used as a fallback.

**Voice Clip Extraction Integration:**

When using TTS with voice cloning in the interactive CLI, you have the option to automatically extract voice reference clips from a multi‑speaker audio file. Instead of manually finding and providing reference audio for each character, VODER can:

1. Download audio from a YouTube URL (or accept a local file)
2. Run Whisper + Pyannote to identify speakers and their segments
3. Extract the longest segment per speaker as a voice reference clip
4. Feed those clips directly into the TTS dialogue pipeline

This eliminates the manual step of finding clean reference audio for each speaker. See [Voice Clip Extraction](#voice-clip-extraction) for full details.

**TTS Overdose Mode:**

When the `overdose` flag is added to a TTS command, two things change:

1. **Dialogue Source Analysis**: When importing audio as a dialogue source (interactive CLI), VibeVoice ASR replaces Whisper + Pyannote for transcription and speaker identification. This provides higher accuracy speaker segmentation in a single model pass, without requiring an HF_TOKEN for Pyannote.

2. **Voice Clip Extraction**: When extracting voice reference clips from a multi‑speaker audio source, VibeVoice ASR segments are used instead of the Whisper + Pyannote pipeline. Additionally, the extracted clips are automatically trimmed — the first 2 seconds and last 3 seconds of each speaker's longest segment are removed to avoid cross‑speaker overlap contamination. This ensures the voice cloning model receives the purest possible speaker audio.

3. **Background Music**: When the `music` parameter is also provided, ACE‑Step XL turbo (the overdose model) is used for background music generation instead of the standard ACE‑Step 1.5 turbo, producing higher quality instrumental music.

**Standard vs Overdose TTS:**

| Feature | Standard (Whisper + Pyannote) | Overdose (VibeVoice ASR) |
|---------|-------------------------------|--------------------------|
| Dialogue source analysis | Whisper + Pyannote (two models) | VibeVoice ASR (single model) |
| Voice clip extraction | Whisper + Pyannote | VibeVoice ASR with overlap trimming |
| Background music model | ACE-Step 1.5 turbo | ACE-Step XL turbo |
| HF_TOKEN required | Yes (for Pyannote) | No |
| Resource requirements | 12-23GB RAM | 48GB RAM or 24GB+ VRAM |

```bash
# TTS with overdose (one-liner)
python src/voder.py tts overdose script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"

# TTS with overdose + voice cloning + background music
python src/voder.py tts overdose script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav" music "soft piano"
```

**Voice Cloning (via target parameter):**

The voice cloning functionality is accessed by providing a `target` parameter with a reference audio file. In single mode, one reference file provides the voice for the entire script. In dialogue mode, each character can be assigned a different reference audio file. **Multi-reference cloning** is supported — provide multiple reference audios using the parenthesized format `(path1)(path2)(path3)`, and they will be concatenated into a single composite reference for richer voice extraction. Add the `first` keyword before the references (`target first "(path1)(path2)(path3)"`) to extract only the first reference's speaker from all other references via TSE before compiling them — useful when references contain multiple speakers and you only want the first reference's voice.

**`sts:` Prefix for Voice References:** When using `target` or a voice reference, you can prefix the path with `sts:` (e.g., `target "sts:voice_ref.wav"`) to apply an additional Seed‑VC v2 non‑mimic pass after synthesis. This improves voice fidelity to the target reference, making the output closer to true voice conversion while still preserving text‑level control over the content. The `sts:` prefix works in single TTS mode, dialogue TTS mode, SVC sub‑task, and interactive modify speech. Multi-reference format is also supported: `target "sts:(path1)(path2)(path3)"`.

**Reference Audio Requirements:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 10‑30 seconds optimal |
| Quality | Clear audio, minimal background noise (SVS auto‑cleans if needed) |
| Content | Continuous speech, not singing or silence |
| Speakers | Single speaker only |
| Format | WAV preferred, MP3 supported |
| Source | Audio files, video files, and YouTube URLs are all accepted |

**Single vs Dialogue Mode:**

In **single mode** (one reference file), the entire script uses that voice. In **dialogue mode** (multiple reference files), each character in a dialogue script is assigned a different reference audio. This is the foundation of VODER's dialogue system, and it is available in **both GUI and CLI**.

**Voice Consistency in Dialogue:**

VODER extracts voice characteristics **once per character** in dialogue mode, rather than re‑extracting for each line. This ensures consistent voice quality throughout the dialogue. If a character speaks multiple lines (e.g., 5 lines for "James"), the voice prompt is extracted once and reused for all lines of that character. This eliminates variations that occurred when re-extracting voice for each line, providing stable and professional-quality voice cloning across entire dialogues.

**Voice Stabilization:**

VoiceDesign characters in dialogue mode automatically get their voice stabilized to eliminate vocal drift in long dialogues. After 3 script lines, the outputs are concatenated, SVS-cleaned, and fed to Qwen3‑TTS Base for voice extraction. All subsequent lines use the cloned voice instead of VoiceDesign, ensuring the character's voice remains consistent even across dozens of lines. This happens automatically — no configuration is needed.

**Optional Background Music (Dialogue Only):**

When using TTS in **dialogue mode** (multiple speakers, script lines containing a colon), you can optionally add automatically generated background music. After the dialogue is synthesized, VODER generates a music track using ACE‑Step with empty lyrics `"..."` and a duration matching the exact length of the dialogue. The music is mixed at **35% volume** relative to the dialogue (configurable via `level` parameter), creating a subtle ambient bed. The final file is saved with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`). This feature is available in GUI (via a clean modal dialog), interactive CLI (prompt after voice prompts), and one‑liner CLI (optional `music` and `level` parameters). See [Optional Background Music for Dialogue](#optional-background-music-for-dialogue) for full details.

**Best For:**

- Narration and voiceover work
- Creating character voices for content
- Situations where you don't have reference audio
- Rapid prototyping of voice concepts
- Generating multiple voice variations for comparison
- **Dialogue with ambient soundtrack** (podcasts, storytelling)
- Consistent voice branding across content
- Dialogue with cloned character voices
- Matching voice characteristics between speakers
- Localization while preserving original voice characteristics

**Voice Prompt Examples (VoiceDesign mode):**

| Desired Voice | Example Prompt |
|---------------|----------------|
| Professional male | "adult male, deep voice, clear pronunciation, professional tone" |
| Warm female | "adult female, warm tone, gentle, conversational" |
| Energetic young | "young adult, energetic, fast‑paced, enthusiastic" |
| News anchor | "middle‑aged, authoritative, measured pace, broadcasting quality" |
| Storytelling | "deep narrative voice, expressive, dramatic pauses" |

**Newline Support in TTS Scripts:**

Use `\n` in script text to insert actual newlines. This works in both oneline and interactive CLI modes:

```bash
# Newline in a dialogue script line
python src/voder.py tts script "James: First line\nSecond line" voice "James: deep male"

# Newline in single mode
python src/voder.py tts script "First paragraph\nSecond paragraph" voice "professional narrator"
```

**CLI Usage:**

```bash
# VoiceDesign mode (generated voice from description)
python src/voder.py tts script "Hello world" voice "text: professional male narrator"

# Voice Clone mode (cloned voice from reference)
python src/voder.py tts script "Hello world" target "voice_reference.wav"

# Voice Clone with YouTube URL as reference
python src/voder.py tts script "Hello world" target "https://www.youtube.com/watch?v=VIDEO_ID"

# Voice Clone with specific language
python src/voder.py tts script "Bonjour le monde" target "french_speaker.wav" language "fr"

# Trained voice mode (use a trained .tts voice)
python src/voder.py tts script "Hello world" voice "my-character"

# Trained voice with specific .tts file
python src/voder.py tts script "Hello world" voice "my-character:path/to/file.tts"

# Dialogue with trained voice
python src/voder.py tts \
  script "James: Hello!" \
  script "Sarah: Hi there!" \
  voice "James: my-character" \
  voice "Sarah: another-character"

# Dialogue with mixed voices (generated + cloned + trained)
python src/voder.py tts \
  script "James: Hello!" \
  script "Sarah: Hi there!" \
  voice "James: deep male voice" \
  target "Sarah: /path/to/sarah_voice.wav"

# Dialogue with newline in script
python src/voder.py tts script "James: First line\nSecond line" voice "James: deep male"

# OCR Input (Image to Narration)
python src/voder.py tts ocr "path/to/image.png" voice "text: professional male narrator"

# Interactive CLI
python src/voder.py cli
# Select mode 2 (TTS), then follow prompts
```

**Technical Notes:**

TTS mode works on CPU without GPU acceleration. Processing time scales with text length, not with prompt complexity. The VoiceDesign model interprets prompts at generation time, so more detailed prompts give the model more information to work with but don't significantly affect processing time. When voice cloning is used, BS‑RoFormer vocal extraction adds a small overhead but significantly improves cloning quality for references with background music or noise.

**OCR Input (Image to Narration):**

You can use the `ocr` parameter to extract text from an image and synthesize it as speech. VODER uses EasyOCR to extract text from the image, then generates narration using the extracted text:

```bash
python src/voder.py tts ocr "path/to/image.png" voice "text: professional male narrator"

python src/voder.py tts ocr "script_screenshot.jpg" target "voice_ref.wav"
```

This is useful for converting screenshots of scripts, slides, or documents into spoken narration without manual text entry.

**Memory Requirements:** TTS (VoiceDesign, no music) requires approximately 12GB RAM (8GB base + 4GB for Qwen model). TTS (Voice Clone, no music) requires approximately 15GB RAM (8GB base + 4GB for Qwen + ~3GB for BS‑RoFormer SVS). With background music, add approximately 15GB for the ACE model. TTS (SLC) requires approximately 18GB RAM (8GB base + ~3GB for Whisper large-v3 + 4GB for Qwen + ~3GB for SVS). TTS (Modify Speech) requires approximately 19GB RAM (8GB base + 4GB for Whisper + 4GB for Qwen + ~3GB for SVS).

#### TTS SLC: Speaker Language Conversion

**What It Does:**

SLC (Speaker Language Conversion) translates speech from any language to English while preserving the original speaker's voice identity. It is now a TTS oneline sub‑task, invoked with `tts slc`. Translation to English is always performed using Whisper large-v3 (not turbo) — no separate `translate` keyword is needed. SLC supports video and YouTube URLs as source input, runs SVS voice isolation on the source audio before transcription, and can optionally preserve non-vocals using the `music` flag.

**How It Works:**

1. **Source Handling**: Accepts audio files, video files, and YouTube URLs as source input
2. **SVS Voice Isolation**: BS‑RoFormer isolates the vocal track from the source audio, removing background music and noise
3. **Music Extraction** (optional): When the `music` flag is used, BS‑RoFormer also extracts the instrumental track for later blending with the voice output
4. **Transcription + Translation**: Whisper large‑v3 (not turbo) transcribes and translates the cleaned source audio to English. If the audio is already in English, translation is skipped
5. **Resynthesis**: Qwen3‑TTS Base generates speech from the English text using the original audio as the voice reference
6. **Overdose Post-Processing** (optional): When `tts overdose slc` is used, Seed‑VC v2 runs a non‑mimic pass after TTS output for better voice preservation
7. **Music Blending** (optional): When the `music` flag is used, the extracted instrumental is blended with the voice output, preserving background music

```bash
# Translate French speaker to English, keeping their voice
python src/voder.py tts slc "french_speech.wav"

# Translate with music preservation (blend non-vocals back)
python src/voder.py tts slc music "french_speech.wav"

# Translate from video
python src/voder.py tts slc "presentation.mp4"

# Translate from YouTube
python src/voder.py tts slc "https://www.youtube.com/watch?v=VIDEO_ID"

# SLC with overdose for better voice preservation
python src/voder.py tts overdose slc "french_speech.wav"

# SLC with overdose + music preservation
python src/voder.py tts overdose slc music "french_speech.wav"
```

**Why It's Like That:**

SLC exists because traditional voice conversion (STS) doesn't change language — it changes voice. Traditional TTS doesn't preserve voice — it generates new speech. SLC bridges this gap by decomposing the problem: first understand what was said (transcription + translation), then say it in English with the same voice (resynthesis). This approach is more flexible than trying to do both simultaneously in a single model, and it produces higher quality results because each stage can use the best available model for its specific task. SLC always targets English because Whisper's translation capability only supports translation to English — it cannot translate between arbitrary language pairs. SLC is now a TTS sub‑task rather than a standalone mode because its pipeline is fundamentally a TTS operation with STT front‑end — it generates speech from text, which is the core definition of TTS.

**Best For:**

- Translating speech to English while preserving speaker identity
- Content localization for video and podcasts
- Creating dubbed content that sounds like the original speaker
- Processing multi‑language content into English
- Video and YouTube URL support for direct video dubbing
- Preserving background music when dubbing (via `music` flag)

**Language Support:**

| Stage | Languages |
|-------|----------|
| Input (Whisper large-v3 transcription) | 99 languages |
| Translation target | English only (Whisper can only translate to English) |
| Output | English |

**CLI Usage:**

```bash
# Translate to English with original voice
python src/voder.py tts slc "path/to/audio.wav"

# Translate with music preservation (blend non-vocals back)
python src/voder.py tts slc music "path/to/audio.wav"

# From YouTube URL
python src/voder.py tts slc "https://www.youtube.com/watch?v=VIDEO_ID"

# From video file
python src/voder.py tts slc "presentation.mp4"

# SLC with overdose for better voice preservation
python src/voder.py tts overdose slc "path/to/audio.wav"

# SLC with overdose + music preservation
python src/voder.py tts overdose slc music "path/to/audio.wav"

# Interactive CLI
python src/voder.py cli
# Select mode 2 (TTS), then choose SLC sub-task
```

**Technical Notes:**

SLC works on CPU without GPU acceleration. The pipeline is sequential: SVS voice isolation (and optional music extraction), Whisper large-v3 transcription and translation, model offloading, then Qwen3-TTS synthesis. This ensures memory requirements stay manageable — you don't need both Whisper large-v3 and Qwen3‑TTS loaded simultaneously. Video files and YouTube URLs are supported as source input. When the `music` flag is used, the instrumental track is extracted and blended with the voice output after synthesis; note that voice-music synchronization may vary as the translated speech duration may differ from the original. In overdose mode, the additional STS v2 pass requires loading Seed‑VC v2 after the TTS output, which increases peak memory requirements.

**Memory Requirements:** TTS (SLC) requires approximately 18GB RAM (8GB base + ~3GB for Whisper large-v3 + 4GB for Qwen3‑TTS + ~3GB for SVS). Models are loaded and offloaded sequentially, so peak memory depends on the larger individual model. TTS (SLC Overdose) requires approximately 23GB RAM due to the additional Seed‑VC v2 pass. With the `music` flag, SVS processes both voice and music stems, but this does not significantly increase peak memory as they are processed sequentially.

#### TTS SVC: Speaker Voice Change

**What It Does:**

SVC (Speaker Voice Change) transcribes single‑speaker audio and re‑synthesizes the same speech with a different voice. Unlike SLC which translates language, SVC preserves the original language and content — it only changes the voice.

**How It Works:**

1. **Input**: Provide an audio/video source path
2. **SVS Voice Isolation**: BS‑RoFormer isolates vocals from the source
3. **Transcription**: Whisper (or VibeVoice ASR with overdose) transcribes the speech
4. **Voice Selection**: Target voice via audio path, trained voice, or text description
5. **SVS Voice on Target**: If target is audio, it's cleaned through SVS
6. **Qwen-TTS Synthesis**: The transcribed text is synthesized using the target voice
7. **Optional STS Pass**: If `sts:` prefix is used on the target, an additional Seed‑VC v2 non‑mimic pass is applied

**CLI Usage:**

```bash
# Change speaker voice using a reference audio
python src/voder.py tts svc "speech.wav" target "voice_ref.wav"

# Change speaker voice using a text description
python src/voder.py tts svc "speech.wav" voice "deep male, authoritative"

# SVC with overdose for better transcription (VibeVoice ASR)
python src/voder.py tts overdose svc "speech.wav" target "voice.wav"

# SVC with STS pass for improved voice fidelity
python src/voder.py tts svc "speech.wav" target "sts:voice_ref.wav"

# SVC with multi-reference target (concatenated for richer voice extraction)
python src/voder.py tts svc "speech.wav" target "(ref1.wav)(ref2.wav)(ref3.wav)"

# SVC with STS pass and multi-reference
python src/voder.py tts svc "speech.wav" target "sts:(ref1.wav)(ref2.wav)"
```

**Why It's Like That:**

SVC is a convenience sub‑task that chains STT + TTS into a single command. It's not voice conversion (STS) — it's transcription followed by re‑synthesis. This means the output preserves the words but may differ in timing and prosody from the original. The `sts:` prefix adds a Seed‑VC v2 pass to improve voice fidelity to the target, making it closer to true voice conversion while still allowing text‑level control.

**Best For:**

- Changing a speaker's voice while keeping the content
- Re‑voicing recordings with a different character
- Creating voice demos from existing speech
- Prototyping voice changes before committing to STS voice conversion

**Availability:**

Oneline mode only. `tts svc "path" target "voice_ref"`. Supports `overdose` flag (switches STT engine to VibeVoice ASR), `sts:` prefix (additional Seed‑VC v2 pass), and multi‑reference targets `(path1)(path2)(path3)`.

**Key Differences from SLC:**

| Aspect | SLC | SVC |
|--------|-----|-----|
| Purpose | Translate to English + keep source voice | Change voice + keep original language |
| Language | Any → English only | Preserves original language |
| Voice | Clones source speaker's voice | Uses a different target voice |
| Transcription model | Whisper large‑v3 | Whisper turbo (or VibeVoice ASR with overdose) |
| Translation | Always (to English) | Never (preserves language) |

**Technical Notes:**

SVC uses Whisper turbo for transcription (not large‑v3 like SLC), since it doesn't need translation. With overdose, VibeVoice ASR is used instead. The Qwen‑TTS models support 10 languages (see Languages.md), so SVC output quality depends on the detected language. Single‑speaker assumption — no speaker diarization is performed.

**Memory Requirements:** TTS (SVC) requires approximately 18GB RAM. TTS (SVC Overdose) requires approximately 22GB RAM.

#### TTS Modify Speech (STT+TTS)

**What It Does:**

The Modify Speech feature transcribes audio to text using Whisper, allows you to edit the transcribed content, and then synthesizes the edited text with a chosen voice. This enables voice modification while preserving the original delivery characteristics. This feature was previously a standalone STT+TTS mode — it is now integrated into TTS interactive mode as a "modify speech? (Y/N)" prompt at the very start.

**How It Works:**

1. **Input**: Provide an audio file, video file, or YouTube URL
2. **SVS Voice Isolation**: BS‑RoFormer isolates the vocal track from the input, removing background music and noise
3. **Whisper Transcription**: Whisper converts speech to text with word‑level timestamps
4. **Text Editing**: Review and modify the transcribed text before synthesis
5. **Voice Selection**: Choose whether to use the source audio as the voice reference or provide a custom target path. Custom paths support `sts:` prefix for an additional Seed‑VC v2 non‑mimic pass, and multi‑reference format `(path1)(path2)(path3)` for richer voice extraction
6. **Preserve Non‑Vocals? (Y/N)**: If yes, SVS music extraction is run on the original source to isolate instrumentals, then the synthesized voice is blended with the instrumental track via ffmpeg — preserving background music and instrumentals in the final output
7. **Optional STS Pass**: If `sts:` prefix was used on the voice reference, an additional Seed‑VC v2 non‑mimic pass is applied after Qwen‑TTS synthesis for enhanced voice fidelity
8. **Qwen-TTS Synthesis**: The edited text is synthesized using the chosen voice via Qwen3‑TTS

The voice reference input accepts the `sts:` prefix and multi‑reference format. When preserve non‑vocals is enabled, voice‑music synchronization may vary as the synthesized speech duration may differ from the original.

**Why It's Like That:**

This feature is for when you have existing audio content that needs voice transformation. By transcribing, editing, and resynthesizing, you can change what someone says while keeping the general timing and delivery. It's not a simple voice conversion — it's a reconstructive process that allows complete content modification. The SVS voice isolation stage ensures that background music in the original audio doesn't interfere with transcription quality. Moving this into TTS interactive mode makes it more discoverable — it's a natural extension of the TTS workflow (generate speech from text), where the text happens to come from existing audio rather than manual entry.

**Best For:**

- Changing content in existing audio
- Fixing transcription errors automatically
- Localizing content into different languages
- Creating fictional dialogue from real voice samples
- Voice modification with full control over content
- Processing songs with vocal isolation

**Availability:**

Modify Speech is available in the TTS interactive CLI mode (prompted at the start) and the GUI. When you select TTS in the interactive CLI, you'll be asked "modify speech? (Y/N)" — answer Y to enter the modify speech workflow where you provide audio, edit the transcription, and choose a voice.

**Multi‑Speaker Note:**

If your base audio contains multiple speakers, Whisper will transcribe all of them. The synthesis will use a single voice for the entire text. If you need per‑speaker voice cloning, use the dialogue system with speaker diarization instead (see [Dialogue Source Analysis](#dialogue-source-analysis)).

**Technical Notes:**

Modify Speech works on CPU without GPU for the Whisper transcription stage. Voice cloning in the synthesis stage also works on CPU. This makes it accessible for users without NVIDIA graphics hardware. When `sts:` prefix is used on the voice reference, the additional Seed‑VC v2 pass adds roughly 5GB to peak memory. When preserve non‑vocals is enabled, SVS processes both voice and music stems sequentially (no additional peak memory), and the final blend uses ffmpeg `amix` with the duration of the first input (the voice track).

**Memory Requirements:** TTS (Modify Speech) requires approximately 19GB RAM (8GB base + 4GB for Whisper + 4GB for Qwen + ~3GB for BS‑RoFormer SVS). TTS (Modify Speech + STS pass) requires approximately 24GB RAM. With preserve non‑vocals, SVS processes both stems but peak memory does not significantly increase since they are processed sequentially.

---

### Voice Training

VODER can train voice clones from reference audio files and save them as `.tts` files for later reuse. This eliminates the need to keep original reference audio files around — the trained voice embedding is stored in a compact `.tts` file that can be used directly in TTS commands.

**What It Does:**

The `train voice` command trains a Qwen3‑TTS Base voice clone from one or more reference audio files. The resulting `.tts` file contains the extracted voice embedding and can be used in the `voice` parameter of TTS commands instead of a voice description.

**Command Syntax (Oneline Only):**

```bash
python src/voder.py train voice:character-name "path1" "path2" ...
```

- `character-name` is the name used to reference the trained voice later
- One or more audio file paths provide the reference audio for training
- Multiple paths are SVS-cleaned individually and concatenated into a composite before voice extraction
- Add the `first` keyword before the paths to extract only the first reference's speaker from all other references: `train voice:name first "ref1.wav" "ref2.wav"` — the first reference identifies the target speaker, and TSE extraction pulls that speaker's voice from the remaining references before compiling
- The trained voice is saved as `voder_tts_character-name_timestamp.tts` in the `voices/` directory

**Optional Test Sample:**

- Add `test` at the end of the command to generate a test sample using a hardcoded 30+ second script:
  ```bash
  python src/voder.py train voice:my-character "ref1.wav" "ref2.wav" test
  ```

- Add `test "custom script"` to use a custom test script instead:
  ```bash
  python src/voder.py train voice:my-character "ref1.wav" test "Custom test script for verification"
  ```

**Using Trained Voices in TTS:**

When using the `voice` parameter in TTS, you can provide a trained voice name or path instead of a voice description:

| Syntax | Behavior |
|--------|----------|
| `voice "character-name"` | Uses the latest `.tts` file with that name from `voices/` |
| `voice "character-name:path/to/file.tts"` | Uses a specific `.tts` file |
| `voice "character-name:another-name"` | Uses the latest `.tts` file for `another-name` from `voices/` |

When a trained voice is used, Qwen3‑TTS Base (voice cloning) is used instead of VoiceDesign. This works in both oneline and interactive CLI modes.

**Examples:**

```bash
# Train a voice from a single reference
python src/voder.py train voice:narrator "narrator_ref.wav"

# Train a voice from multiple references
python src/voder.py train voice:hero "hero_clip1.wav" "hero_clip2.wav" "hero_clip3.wav"

# Train with test sample
python src/voder.py train voice:narrator "narrator_ref.wav" test

# Train with custom test script
python src/voder.py train voice:narrator "narrator_ref.wav" test "The quick brown fox jumps over the lazy dog."

# Use trained voice in TTS (single mode)
python src/voder.py tts script "Hello world" voice "narrator"

# Use trained voice in TTS (dialogue mode)
python src/voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: hero" voice "Sarah: cheerful female"

# Use a specific .tts file
python src/voder.py tts script "Hello" voice "narrator:voices/voder_tts_narrator_20260101_120000.tts"

# Use a different trained voice name
python src/voder.py tts script "Hello" voice "my-char:hero"
```

---

### STS: Speech-to-Speech Voice Conversion

**What It Does:**

STS (Speech‑to‑Speech) transforms source audio to sound like a target voice while preserving the original content, emotion, timing, and prosody. The speaker changes, but everything they say remains exactly the same. STS now supports video input — provide an MP4 video file and receive an MP4 output with the converted voice. Vocals are automatically separated from the source audio before conversion, and the source music is mixed back afterward for a clean result.

**MSTS (Music-STS):**

STS supports musical inputs via the **MSTS** feature. When converting voice in songs or musical audio, use the `music` parameter to switch to Seed‑VC v1 (44.1kHz) instead of the standard v2 model (22.05kHz). This provides better voice conversion quality for music content because v1 is optimized for higher sample rates and musical waveforms.

- **GUI**: A dialog asks "musical inputs?" with Yes/No buttons before processing
- **Interactive CLI**: After entering base and target paths, prompted "Are the inputs musical? (Y/N):"
- **One-line CLI**: Add `music` keyword at the end: `voder.py sts path/base path/target music`
- **Output**: MSTS outputs use `voder_m_sts_timestamp.wav` naming; standard STS uses `voder_sts_timestamp.wav`

**Mimic (Style Transfer):**

STS supports a `mimic` keyword that enables full style transfer — converting not just the voice timbre but also the accent, emotional delivery, and speaking patterns of the target voice. This uses Seed‑VC v2's AR model alongside the standard CFM model. Without `mimic`, only the voice sound is transferred; with `mimic`, the entire vocal character — how the target person talks, not just how they sound — is applied to the source content.

- **One-line CLI**: Add `mimic` keyword after the target path: `voder.py sts path/base path/target mimic`
- **Mutual exclusion**: `mimic` and `music` cannot be used together — they target different models (v2 vs v1) and serve different purposes (style transfer vs music sample rate)

**nomusic (Voice-Only Output):**

By default, STS separates vocals from the source, converts them, and mixes them back with the source's instrumental/music. The `nomusic` flag skips the music recombination step and outputs only the converted voice. This is useful when you want raw converted vocals without any background — for example, to process the voice further or when the source has no meaningful music content to preserve.

- **One-line CLI**: Add `nomusic` keyword: `voder.py sts base "source.wav" target "voice.wav" nomusic`
- **Mutual exclusion**: `nomusic` and `music` cannot be used together — `music` already handles music content via VCv1

**Automatic Vocal Extraction:**

VODER automatically runs BS‑RoFormer vocal isolation on both the source and target audio. For the source, vocals are separated so the VC model processes only the voice — producing cleaner conversion — and the instrumental is extracted separately for recombination after conversion (unless `nomusic` is used). For the target, clean vocals are extracted to improve cloning quality. If SVS extraction fails, the original audio is used as a fallback.

**Multi-Reference Target (Oneline Only):**

The `target` parameter accepts multiple voice references using the parenthesized format `(path1)(path2)(path3)`. Each reference is resolved, SVS-cleaned individually, then concatenated into a single composite for richer voice cloning. This works for all STS sub-modes (standard, music, mimic). Add the `first` keyword before the references (`target first "(path1)(path2)(path3)"`) to extract only the first reference's speaker from all other references via TSE before compiling — useful when references contain multiple speakers and you only want the first reference's voice.

- CLI: `voder.py sts base "source.wav" target "(voice1.wav)(voice2.wav)(voice3.wav)"`
- With `first`: `voder.py sts base "source.wav" target first "(voice1.wav)(voice2.wav)(voice3.wav)"`
- YouTube URLs are supported within parentheses: `target "(voice1.wav)(https://youtube.com/...)"`
- Single path format still works as before: `target "voice.wav"`

**Video I/O:**

STS now supports video input with MP4 output. When you provide a video file as input, VODER extracts the audio, performs voice conversion, and re‑encodes the result as an MP4 video with the converted voice track. This enables direct voice replacement in video content without manual audio extraction and re‑encoding.

**How It Works:**

Seed‑VC v2 analyzes both the source and target audio to extract content representations and voice characteristics. It then synthesizes new audio that combines the source content with the target voice. This isn't simple audio manipulation — it's neural voice conversion that genuinely reconstructs the speech in a different voice.

**Why It's Like That:**

Voice conversion serves specific use cases that TTS can't handle. You might have archival audio that needs voice preservation but content modification. You might want to maintain the exact delivery and emotion of a performance while changing the voice. Voice conversion preserves paralinguistic features that text‑to‑speech can't reproduce.

**Best For:**

- Preserving delivery while changing voice
- Content modification in existing audio
- Voice anonymization or de‑identification
- Consistent voice application across multiple recordings
- Archival content republishing with voice updates
- Direct voice replacement in video content

**Input Considerations:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 5‑60 seconds optimal per segment |
| Content | Clear speech, minimal background music |
| Quality | Studio quality preferred, phone quality works but loses detail |
| Format | WAV, MP3, or video (MP4, MKV, AVI, MOV) |

**Technical Notes:**

STS runs on CPU without GPU. Input audio is automatically resampled to 22050 Hz for model processing, and output is resampled to 44100 Hz for playback. When video input is provided, the audio is extracted via FFmpeg, converted, and then re‑encoded into an MP4 container with the original video stream.

**Memory Requirements:** STS requires approximately 16GB RAM (8GB base + 5GB for Seed-VC + ~3GB for BS‑RoFormer SVS for auto vocal extraction).

---

### TTM: Text-to-Music

**What It Does:**

TTM (Text‑to‑Music) generates original music from lyrics and a style prompt using ACE‑Step. You provide song lyrics, describe the desired musical style, and specify duration — VODER creates original music with vocals matching your lyrics. TTM now includes voice conversion via the `vc` flag and `clone` parameter, merging the previous TTM+VC functionality into a single mode. It also supports advanced sub‑tasks for track‑level music manipulation.

**Three-Tier ACE‑Step System:**

TTM offers three tiers of ACE‑Step quality:

| Tier | Model | LM Model | Best For | Requirements |
|------|-------|----------|----------|-------------|
| Standard | acestep-v15-turbo | acestep-5Hz-lm-1.7B | General use, balanced quality/speed | 23GB RAM, 15GB VRAM |
| Overdose | acestep-v15-xl-turbo | acestep-5Hz-lm-4B | Maximum quality | 32GB+ RAM, 32GB+ VRAM |
| Complete | acestep-v15-xl-base | acestep-5Hz-lm-1.7B | Sub-tasks (complete, lego, extract) with 50 inference steps | 32GB+ RAM, 32GB+ VRAM |

**Overdose Mode:**

When enabled, Overdose uses the larger XL‑Turbo model with the 4B language model for higher quality output. This produces noticeably better musical results — richer instrumentation, better vocal quality, more coherent song structure — but requires 32GB+ VRAM or 48GB+ combined system memory. If insufficient resources are detected, VODER automatically falls back to standard mode with a warning.

**Voice Conversion (via vc flag):**

TTM now supports voice conversion directly within the mode. When the `vc` flag is enabled and a `clone` parameter is provided:

1. Music is generated with ACE‑Step (TTM stage)
2. BS‑RoFormer automatically extracts clean vocals from the clone reference
3. Seed‑VC voice conversion transforms the generated vocals to match the clone voice

This replaces the previous separate TTM+VC mode. The entire pipeline runs in sequence with automatic model offloading between stages. VC is mutually exclusive with `remix` and `repaint` sub-tasks.

**Multi-Reference Clone (Oneline Only):**

The `clone` parameter accepts multiple voice references using the parenthesized format `(path1)(path2)(path3)`. Each reference is resolved, SVS-cleaned individually, then concatenated into a single composite for richer voice cloning. This provides the VC model with a more complete voice profile from multiple samples. Add the `first` keyword before the references (`clone first "(path1)(path2)(path3)"`) to extract only the first reference's speaker from all other references via TSE before compiling — useful when references contain multiple speakers and you only want the first reference's voice.

- CLI: `voder.py ttm vc lyrics "..." styling "..." duration 30 clone "(voice1.wav)(voice2.wav)"`
- With `first`: `voder.py ttm vc lyrics "..." styling "..." duration 30 clone first "(voice1.wav)(voice2.wav)"`
- YouTube URLs are supported within parentheses: `clone "(voice1.wav)(https://youtube.com/...)"`
- Single path format still works as before: `clone "voice.wav"`

**Reference Audio for Reference-Aware Generation:**

TTM supports an optional `target` reference audio (when `vc` is not enabled) for reference‑aware music generation. You can specify `voice` or `music` extraction from the reference:
- `target voice "ref.wav"` — Extract vocals from the reference for vocal guidance
- `target music "ref.wav"` — Extract instrumental from the reference for style guidance

Additionally, `remix` and `repaint` sub-tasks now support a `reference` parameter for providing additional audio guidance during style transfer:
- `reference voice "ref.wav"` — Extract vocals from the reference for guidance
- `reference music "ref.wav"` — Extract instrumental from the reference for guidance
- `reference "ref.wav"` — Use the reference audio as‑is (no extraction)

The `reference` parameter accepts audio files, video files, and URLs (YouTube, Bilibili, TikTok). It works with both standard and overdose quality modes.

**Reference Time Spec:**

The reference path can include an optional time spec to select a specific portion of the reference audio instead of using the entire file. This applies to all TTM sub-tasks that accept references (remix, repaint, complete, lego, bgm).

| Format | Example | Description |
|--------|---------|-------------|
| `nn(path)` | `"50(ref.wav)"` | Start at nn seconds, extract up to slot max |
| `nn-nn(path)` | `"20-30(ref.wav)"` | Use specified range; slides to reach slot max if shorter |
| `nn-nn/nn-nn/nn-nn(path)` | `"20-30/40-50(ref.wav)"` | Multiple ranges from same audio, combined to reach slot max |

The time spec is optional -- the old format `reference "ref.wav"` still works and uses the entire audio. It works with voice/music prefixes: `reference voice "50(ref.wav)"`, `reference music "20-30/40-50(ref.wav)"`. It also works inside repaint multi-pass specs: `"20-80/styling(jazz)/reference-voice(30-60(vocals.wav))"`.

**Slot max by reference count:** 1 reference = 30s, 2 references = 15s each, 3 references = 10s each.

**Sliding logic:** If the specified range is shorter than the slot max, the start is slid back and/or the end is slid forward until the slot max duration is reached. If the audio is shorter than the slot max, segments loop to fill the slot. If the combined segments exceed the slot max, they are used as-is.

**Sub-Tasks:**

TTM supports advanced music manipulation sub-tasks that go beyond simple generation:

| Sub-Task | Description | CLI Syntax |
|----------|-------------|------------|
| `generate` | Standard music generation (default) | `python voder.py ttm lyrics "..." styling "..." duration 30` |
| `remix` | Style-transferred version of an existing song (supports `reference` for additional guidance, optional `lyrics` for new vocal content, multi-source up to 3 and multi-reference up to 3) | `python voder.py ttm remix "input.wav" styling "..." bias 40 result "/output/remix.wav"` |
| `repaint` | Repaint a time range of an existing track (supports `reference` for additional guidance; optional `voice`/`music` prefix on source for SVS isolation; multi-pass mode for sequential edits building on each previous result) | `python voder.py ttm repaint "source.wav" time:20-80 styling "..." result "/output/repainted.wav"` |
| `complete` | Add instrument tracks to existing audio (supports `sfx:` overlay) | `python voder.py ttm complete source "song.wav" add "drums bass" [target music "ref.wav"]` |
| `extract` | Extract vocals or music from a track | `python voder.py ttm extract "song.wav" extract "vocals"` |
| `lego` | Build a track from individual instrument stems | `python voder.py ttm lego source "song.wav" make "drums bass guitar"` |

**12 Instrument Tracks:**

The `complete` and `lego` sub-tasks support 12 distinct instrument tracks with an intelligent resolution system:

| Track | Category | Description |
|-------|----------|-------------|
| drums | Instrument | Drum kit, percussion backbone |
| bass | Instrument | Bass guitar, synth bass, upright bass |
| guitar | Instrument | Electric guitar (lead/rhythm) |
| keyboard | Instrument | Piano, organ, synthesizer keys |
| strings | Instrument | Violin, cello, string ensemble |
| brass | Instrument | Trumpet, trombone, horn section |
| woodwinds | Instrument | Flute, clarinet, saxophone |
| percussion | Instrument | Hand percussion, shakers, congas |
| synth | Instrument | Synth leads, pads, arpeggios |
| fx | Instrument | Sound effects, textures, atmospheric elements |
| vocals | Voice | Lead vocal track |
| backing_vocals | Voice | Background vocals, harmonies |

**Shorthand Expansion:**

The track resolution system supports shorthand keywords:

| Shorthand | Expands To |
|-----------|------------|
| `everything` | All 12 tracks |
| `voices` | `vocals` + `backing_vocals` |
| `instruments` | All 10 non-voice tracks |

**How It Works:**

ACE‑Step interprets your lyrics as vocal content and your style prompt as musical direction. It generates both the instrumental arrangement and the vocal performance, synchronized to your specified duration. The lyrics become the vocal melody, and the style prompt guides the instrumentation, genre, and mood.

The `bgm` and `complete` sub‑tasks also support **SFX overlay** via the `sfx:` parameter. When SFX specs are provided, TangoFlux generates each sound effect and overlays it at the specified position and volume after the main processing step. ACE‑Step is offloaded from VRAM before TangoFlux loads to free memory. See the [BGM Subtask](#ttm-mode-bgm-subtask-replace-background-music) and [TTM Sub-Task Tricks](#ttm-sub-task-tricks) sections for details.

**Why It's Like That:**

Music generation from lyrics is distinct from instrumental generation because vocals add a layer of complexity. The lyrics must be converted to actual singing, which requires understanding of melody, rhythm, and phonetics. ACE‑Step handles this by treating lyrics as both content and guidance for the vocal generation pipeline.

The three‑tier system exists because not everyone has the hardware for maximum quality. Standard mode works on modest hardware. Overdose provides the best output for users with high‑end GPUs. Complete mode enables sub‑tasks that require the XL model's advanced capabilities for track manipulation.

**Note on Background Music:**

The same ACE‑Step engine is used to generate background music for dialogue. In that context, the lyrics are set to `"..."` (a placeholder for empty vocals), and the style prompt is taken from the user's music description. This yields purely instrumental music suitable for ambient use.

**Best For:**

- Creating original background music with vocals
- Song prototyping and demo creation
- Content needing custom music with lyrics
- Experimental music creation
- Rapid music visualization from lyrics
- Music with specific vocalist voice (voice conversion)
- Adding missing instruments to existing tracks (complete)
- Creating remixes in different styles (remix)
- Repainting sections of existing songs (repaint)
- Building custom arrangements from stems (lego)

**Lyrics Format:**

ACE-Step uses structural tags in `[brackets]` to mark song sections. Text inside brackets is not sung — it tells the model the song structure. Plain text between tags is the sung lyrics.

**Structural tags:**

| Tag | Purpose |
|-----|---------|
| `[Verse]` / `[Verse 1]` / `[Verse 2]` | Verse section |
| `[Chorus]` / `[Final Chorus]` | Chorus section |
| `[Pre-Chorus]` | Build-up before chorus |
| `[Bridge]` | Contrasting section between verses |
| `[Intro]` / `[Intro: description]` | Song opening |
| `[Outro]` | Song ending |
| `[Interlude]` | Instrumental break |
| `[Instrumental]` / `[inst]` | Entirely instrumental (no vocals) |
| `[Hook]` / `[Solo]` / `[Break]` | Other structural markers |

**Special lyrics values:**

| Syntax | Meaning |
|--------|---------|
| `...` (three dots) | Empty lyrics — instrumental music only |
| `(text in parens)` | Context/style hint, not sung |
| `[text in brackets]` | Structural tag, not sung |

```
[Verse 1]
Walking down the empty street
Feeling the rhythm in my feet
The city lights are shining bright
Guiding me through the night

[Chorus]
This is our moment, this is our time
Everything's gonna be just fine
Dancing under the moonlight
Everything feels so right
```

**Multi-line Lyrics in One‑Liner:**

Use `\n` to create multi-line lyrics in a single command:

```bash
python src/voder.py ttm lyrics "Verse 1:\nWalking down the street\nFeeling the beat\n\nChorus:\nThis is our moment\nEverything feels right" styling "upbeat pop with female vocals" duration 30

python src/voder.py ttm lyrics "Bridge:\nEven when the rain falls down\nWe keep dancing through the crowd\n\nFinal Chorus:\nTogether we stand strong\nNothing can go wrong" styling "emotional ballad with piano and strings" duration 60
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

**CLI Usage:**

```bash
# Standard music generation
python src/voder.py ttm lyrics "Walking through the shadows" styling "epic cinematic" duration 30

# Overdose mode (higher quality, requires more VRAM)
python src/voder.py ttm overdose lyrics "Walking through the shadows" styling "epic cinematic" duration 30

# Overdose with voice conversion
python src/voder.py ttm overdose vc lyrics "Verse:\nAmazing lyrics here" styling "epic rock" duration 45 clone "singer.wav"

# Voice conversion (TTM+VC merged)
python src/voder.py ttm vc lyrics "Walking through shadows" styling "epic rock" duration 30 clone "singer_ref.wav"

# Voice conversion with overdose and music reference
python src/voder.py ttm overdose vc lyrics "Verse:\nAmazing lyrics here" styling "epic rock anthem" duration 20 clone "singer_voice.wav" target music "backing_ref.wav" result "/output/song.wav"

# Remix sub-task (style transfer)
python src/voder.py ttm remix "original_song.wav" styling "jazz version" bias 40 result "/output/jazz_remix.wav"

# Remix with custom lyrics (optional lyrics guide new vocal content)
python src/voder.py ttm remix "original_song.wav" lyrics "new verse words here" styling "jazz version" result "/output/jazz_remix.wav"

# Remix with reference (extract vocals from reference for guidance)
python src/voder.py ttm remix "original_song.wav" styling "jazz version" reference voice "ref.wav" result "/output/jazz_remix.wav"

# Remix with reference (extract instrumental from reference)
python src/voder.py ttm remix "original_song.wav" styling "jazz" reference music "ref.wav" result "/output/jazz_remix.wav"

# Remix with reference (use as-is)
python src/voder.py ttm remix "original_song.wav" styling "jazz" reference "ref.wav" result "/output/jazz_remix.wav"

# Overdose remix with reference
python src/voder.py ttm overdose remix "original_song.wav" styling "jazz" reference voice "ref.wav" result "/output/jazz_remix.wav"

# Overdose remix with lyrics
python src/voder.py ttm overdose remix "original_song.wav" lyrics "dreamy verse lines" styling "synthwave" result "/output/remix.wav"

# Remix vocals only (pre-extract vocals from source via SVS)
python src/voder.py ttm remix voice "original_song.wav" styling "jazz version" result "/output/voice_remix.wav"

# Remix music only (pre-extract instruments from source via SVS)
python src/voder.py ttm remix music "original_song.wav" styling "electronic" result "/output/music_remix.wav"

# Overdose remix with voice isolation
python src/voder.py ttm overdose remix voice "original_song.wav" styling "cinematic orchestral" result "/output/voice_od_remix.wav"

# Multi-source remix (vocals + instruments from different songs)
python src/voder.py ttm remix voice "vocals.wav" music "instruments.wav" styling "funk" bias 60 result "/output/multi_remix.wav"

# Multi-reference remix (2 references composed into 30s composite)
python src/voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav" result "/output/multi_ref_remix.wav"

# Multi-reference remix (3 references)
python src/voder.py ttm remix "song.wav" styling "rock" reference "ref1.wav" voice "ref2.wav" music "ref3.wav" result "/output/multi_ref3_remix.wav"

# Repaint sub-task (repaint 20s-80s section)
python src/voder.py ttm repaint "song.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"

# Repaint with voice/music isolation on source
python src/voder.py ttm repaint voice "song.wav" time:20-80 styling "more energetic" result "/output/repainted.wav"
python src/voder.py ttm repaint music "song.wav" time:20-80 styling "ambient" result "/output/repainted.wav"

# Repaint with reference
python src/voder.py ttm repaint "song.wav" time:20-80 styling "more energetic" reference voice "ref.wav" result "/output/repainted.wav"

# Overdose repaint with reference
python src/voder.py ttm overdose repaint "song.wav" time:20-80 styling "more energetic" reference music "ref.wav" result "/output/repainted.wav"

# Multi-pass repaint (each pass builds on the previous result)
python src/voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"
python src/voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"
python src/voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"

# Complete sub-task (add drums and bass to existing track)
python src/voder.py ttm complete source "vocals_only.wav" add "drums bass"

# Complete with styling prompt (influence mood and genre of generated instruments)
python src/voder.py ttm complete source "vocals_only.wav" add "drums bass" styling "dramatic cinematic"

# Complete with noblend (output generated instruments only, no blending with original)
python src/voder.py ttm complete noblend source "vocals_only.wav" add "drums bass"

# Complete with reference (add instruments matching a reference)
python src/voder.py ttm complete source "vocals_only.wav" add "everything" target music "style_ref.wav"

# Complete with SFX overlay (add instruments + overlay sound effects)
python src/voder.py ttm complete source "vocals_only.wav" add "drums bass" sfx "thunder rumble/10-5/60" sfx "door slam/5-30/80"

# Complete with SFX only (no instruments added, no music model loaded)
python src/voder.py ttm complete source "narration.wav" sfx "wind howling/15-0/40" sfx "footsteps on gravel/8-20/55"

# BGM with SFX overlay (replace music + add sound effects)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30 sfx "phone ringing/6-45/70"

# BGM with SFX only (no new music, just overlay sound effects on clean voice)
python src/voder.py ttm bgm "interview.wav" sfx "coffee shop ambience/30-0/25" sfx "doorbell/3-60/65"

# Lego sub-task (build track from stems)
python src/voder.py ttm lego source "drums_track.wav" make "bass guitar strings"

# Lego with styling prompt (influence mood and genre of generated instruments)
python src/voder.py ttm lego source "drums_track.wav" make "bass guitar strings" styling "jazz trio"

# Extract sub-task (isolate vocals or music)
python src/voder.py ttm extract "full_song.wav" extract "vocals"
python src/voder.py ttm extract "full_song.wav" extract "music"

# Interactive CLI
python src/voder.py cli
# Select mode 4 (TTM), then follow prompts
```

**Technical Notes:**

TTM works on CPU without GPU. Processing time scales primarily with duration rather than lyrics length. The style prompt complexity doesn't significantly affect processing time but does affect the musical output characteristics.

In Overdose mode, the XL‑Turbo model uses a different sampling shift (3.0 vs 1.0) for higher quality generation. The 4B language model provides better understanding of lyrics and style descriptions.

For voice conversion, BS‑RoFormer automatically extracts clean vocals from the clone reference before Seed‑VC processing. The `complete`, `lego`, and `extract` sub‑tasks use 50 inference steps and require the Complete‑mode ACE‑Step wrapper, which uses the XL‑Base model.

**TTM Parameter Reference:**

| Parameter | Description | Required/Default |
|-----------|-------------|-----------------|
| `lyrics "..."` | Song lyrics text (also optional for remix to guide new vocal content) | Required (for generate/VC) |
| `styling "..."` | Musical style/description | Required |
| `duration N` | Duration in seconds | Required |
| `vc` | Enable voice cloning flag | Optional |
| `clone "path"` | Voice clone source path | Required when `vc` is set |
| `target voice "ref.wav"` | Music reference — extract vocals | Optional (not with `vc`) |
| `target music "ref.wav"` | Music reference — extract instrumental | Optional (not with `vc`) |
| `remix "path"` | Source audio for remix style transfer | Required for remix sub-task |
| `repaint "path"` | Source audio for section repaint; optional `voice`/`music` prefix for SVS isolation | Required for repaint sub-task |
| `bias N` | Style transfer strength 0–100 | Optional (default 40, for remix/repaint) |
| `time:start-end` | Time range for repaint (single-pass mode) | Required for repaint sub-task (single-pass) |
| `"start-end/styling(...)/..."` | Multi-pass repaint spec (quoted): time range required; optional `/styling(text)`, `/lyrics(text)`, `/reference-voice(path)`, `/reference-music(path)`, `/reference(path)` (up to 3 per pass), `/bias/nn`. Multiple pass specs = multiple sequential repaint passes. | No (multi-pass mode) |
| `add "..."` | Instrument tracks to add (complete) | Required for complete sub-task (unless `sfx:` provided) |
| `make "..."` | Instrument tracks to build (lego) | Required for lego sub-task |
| `extract "..."` | Track to extract | Required for extract sub-task |
| `source "path"` | Source audio (complete/lego/extract) | Required for those sub-tasks |
| `styling "..."` | Style prompt for complete/lego sub-tasks | Optional (influences mood and genre) |
| `noblend` | Output generated instruments only without blending with original (`complete` only) | Optional (default: blend with original) |
| `usrc` | Blend with original source instead of isolated voice/music (`complete` only, requires `voice` or `music`) | Optional (default: blend with isolated source) |
| `sfx "prompt/duration-position/level"` | SFX overlay spec (bgm/complete only, multiple allowed) | Optional (see SFX Overlay Syntax below) |
| `video` | Preserve video output (`complete`/`bgm`) — downloads video from URL, merges back | Optional |
| `overdose` | Use XL-Turbo model for max quality | Optional |
| `result "path"` | Output file path | Optional |

**SFX Overlay Syntax (bgm/complete only):**

Each `sfx:` spec uses the format `prompt/duration-position/level`:

| Component | Format | Rules |
|-----------|--------|-------|
| `prompt` | Text description of the sound | Required |
| `duration` | Seconds (5-30) | Required. Auto-clamped: <5 becomes 5, >30 becomes 30 with warning. Minus signs stripped. Invalid values produce an error. |
| `position` | Seconds into source audio | Required. Non-negative. Cannot exceed source duration. Invalid values produce an error. |
| `level` | Volume 1-100% | Optional (default: 50). Minus signs stripped. <1 produces warning and becomes 1. >100 produces warning and becomes 100. Invalid values produce an error. |

Multiple `sfx:` specs can be specified by repeating the `sfx` parameter. SFX is generated by the TangoFlux model; ACE‑Step is offloaded before TangoFlux loads to free VRAM.

**Mutual Exclusions:**
- `vc` is mutually exclusive with `remix` and `repaint`
- `target` is mutually exclusive with `vc`
- `sfx:` is mutually exclusive with `noblend` (complete sub-task)

**Memory Optimisation:**

VODER explicitly offloads models from memory after each operation completes. This applies to all modes in both GUI and interactive CLI:

- **GUI Mode**: ProcessingThread calls cleanup() after finishing, releasing all loaded models
- **Interactive CLI**: Each mode offloads models before returning
- **Pattern Applied**: `del model`, `gc.collect()`, `torch.cuda.empty_cache()`

This prevents memory accumulation when performing multiple operations in a single session, making VODER more reliable for batch processing workflows.

**Memory Requirements:** TTM (standard) requires approximately 23GB RAM (8GB base + 15GB for ACE model). TTM (overdose) requires approximately 32GB+ RAM or 32GB+ VRAM. TTM (VC enabled) requires approximately 31GB RAM. TTM (complete sub-task) requires approximately 35GB RAM (32GB+ VRAM recommended). When SFX overlay is used in BGM or Complete, add approximately 3-4GB for the TangoFlux model (ACE-Step is offloaded first to free VRAM).

---

### SE: Speech Enhancement

**What It Does:**

SE (Speech Enhancement) improves audio quality by removing noise, reducing reverberation, and restoring speech clarity. It uses the UniSE model from Alibaba's Unified-Audio project to enhance degraded recordings.

**How It Works:**

UniSE is a speech enhancement model trained to separate clean speech from background noise and reverberation artifacts. The model takes degraded audio as input and produces enhanced speech output at 16kHz sample rate. It performs three key operations:

1. **Denoising**: Removes background noise such as hiss, hum, traffic, air conditioning, and other unwanted sounds
2. **Dereverberation**: Reduces room echo and reverb effects that make speech sound distant or muddy
3. **Speech Restoration**: Enhances clarity and intelligibility of degraded speech frequencies

**Why It's Like That:**

Speech enhancement is distinct from other VODER modes because it doesn't transform content — it improves quality. This is useful when you have recordings with poor audio conditions that need cleanup before further processing. Unlike voice conversion which changes the speaker, speech enhancement preserves the speaker's identity while improving clarity.

**Best For:**

- Cleaning up noisy recordings
- Improving poor-quality audio for transcription
- Restoring old or degraded speech recordings
- Pre-processing audio before voice cloning
- Enhancing remote meeting recordings
- Cleaning up field recordings or interviews

**Input Considerations:**

| Factor | Recommendation |
|--------|----------------|
| Content | Speech-only audio (not music) |
| Quality | Any quality accepted, but very degraded audio may have limits |
| Duration | Any length supported |
| Format | WAV, MP3, FLAC, OGG, MP4, MKV, AVI, MOV |

**Important Limitations:**

- **Not for musical content**: UniSE is optimized for speech enhancement, not music. Using it on music may degrade quality.
- **16kHz output**: Enhanced audio is output at 16kHz sample rate, which is optimal for speech but lower than CD quality.
- **Cannot recover missing information**: Severely clipped or corrupted audio cannot be fully restored.

**Technical Notes:**

SE mode works on both CPU and GPU. Having a GPU can significantly speed up processing for long audio files. The UniSE model is loaded on-demand and offloaded after processing to prevent memory accumulation.

**CLI Usage:**

```bash
# Basic enhancement
python src/voder.py se "noisy_audio.wav"

# Enhance audio from video
python src/voder.py se "recording.mp4"

# Save to specific location
python src/voder.py se "audio.wav" result "/path/to/enhanced.wav"

# Interactive CLI
python src/voder.py cli
# Select mode 5 (SE)
```

**Memory Requirements:** SE requires approximately 11GB RAM (8GB base + 2-3GB for UniSE model).

---

### SFX: Sound Effects Generation

**What It Does:**

SFX (Sound Effects) generates custom sound effects from text descriptions using TangoFlux. You describe the sound you want, specify duration and optional quality parameters, and VODER creates the audio.

**How It Works:**

TangoFlux is a text-to-audio diffusion model trained on a large dataset of sound effects and their descriptions. It interprets your text prompt and generates audio that matches the description through a diffusion process. The model can create a wide variety of sounds: natural (rain, thunder, animals), mechanical (engines, doors, impacts), ambient (crowds, wind, forests), and synthetic (whooshes, stingers, transitions).

**Why It's Like That:**

Sound effects are essential for audio production but traditionally require searching through libraries or recording Foley. Text-to-audio generation provides instant access to custom sounds without needing a sound library or recording setup. You can generate exactly what you need for your project.

**Best For:**

- Podcast and video sound design
- Game audio prototyping
- Film and video post-production
- Music production (transitions, impacts, atmospheres)
- Quick custom sound creation

**Parameters:**

| Parameter | Description | Range | Default | Required |
|-----------|-------------|-------|---------|----------|
| `sound` | Text description of the sound | Any text | — | Yes |
| `duration` | Duration in seconds | 1-30 | — | Yes |
| `steps` | Inference steps (quality vs speed) | 1-100 | 30 | No |
| `guide` | Guidance scale (prompt adherence) | 1.0-10.0 | 4.5 | No |
| `result` | Output file path | Any path | — | No |

**Step Count Guidelines:**

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 10-20 | Basic | Fast | Quick prototyping, previews |
| 30 | Good | Medium | Default, most use cases |
| 50-70 | High | Slow | Final production quality |
| 80-100 | Maximum | Very slow | Critical applications |

**Guidance Scale Guidelines:**

| Guide | Behavior |
|-------|----------|
| 1.0-2.0 | More creative, less adherence to prompt |
| 4.0-5.0 | Balanced (default) |
| 7.0-10.0 | Strict adherence to prompt, less variation |

**Sound Prompt Tips:**

| Sound Type | Example Prompts |
|------------|-----------------|
| Nature | "heavy rain on a tin roof with distant thunder" |
| Impacts | "deep punchy kick drum impact with reverb tail" |
| Ambient | "busy coffee shop atmosphere with clinking cups" |
| Transitions | "swoosh whoosh transition with rising pitch" |
| Mechanical | "old car engine starting and idling roughly" |
| Sci-fi | "futuristic laser blast with digital distortion" |

**Technical Notes:**

SFX mode works on both CPU and GPU. GPU acceleration significantly speeds up generation, especially at higher step counts. Output is at 44.1kHz sample rate for professional audio quality. The TangoFlux model is loaded on-demand and offloaded after processing.

In addition to standalone SFX mode, the TangoFlux model is also used for **SFX overlay** in TTM `bgm` and `complete` sub-tasks. When SFX overlay is requested in those sub-tasks, ACE-Step is offloaded from VRAM before TangoFlux loads to free memory. This model-swapping pattern ensures that even GPU-constrained systems can handle music generation followed by SFX generation without running out of VRAM.

**CLI Usage:**

```bash
# Basic sound effect
python src/voder.py sfx sound "thunder rumbling in the distance" duration 10

# With quality parameters
python src/voder.py sfx sound "rain on a tin roof" duration 15 steps 50 guide 3.5

# Save to specific location
python src/voder.py sfx sound "footsteps on gravel" duration 8 result "/output/footsteps.wav"

# Interactive CLI
python src/voder.py cli
# Select mode 6 (SFX)
```

**Memory Requirements:** SFX requires approximately 12GB RAM (8GB base + 3-4GB for TangoFlux model).

---

### SVS: Song Voice Separate

**What It Does:**

SVS (Song Voice Separate) isolates vocals from music (or music from vocals) in any audio file using BS‑RoFormer Resurrection. It produces two possible output stems — voice (vocals only) or music (instrumental only) — or both stems sequentially when the `both` parameter is used. SVS is also used internally by STS, TTS, STT, SS, and TTM for automatic vocal extraction from reference audio.

**How It Works:**

1. **Model Loading**: BS‑RoFormer Resurrection loads its source separation model from `src/models/svs/`
2. **Audio Analysis**: The input audio is analyzed to identify vocal and non‑vocal components
3. **Separation**: Using the RoFormer architecture, the model separates the audio into two stems:
   - **Voice**: Isolated vocal performance, free from instrumental accompaniment
   - **Music**: Instrumental track only, with all vocals removed
4. **Output**: The selected stem is saved as a WAV file

**Why It's Like That:**

Source separation is a fundamentally different operation from the other VODER modes. Instead of transforming content, it decomposes audio into its constituent parts. BS‑RoFormer was chosen because it represents the current state of the art in open‑source source separation — it produces clean separations that preserve audio quality far better than earlier approaches. The model is particularly effective at handling complex mixes with overlapping frequencies, which is exactly the challenge you face when trying to isolate vocals from a full band arrangement.

Making SVS a standalone mode (in addition to its internal use) gives users direct control over the separation process. Sometimes you just need an instrumental version of a song, or a clean vocal track, without any other processing.

**Internal Use by Other Modes:**

SVS is called automatically by several other VODER modes:

| Mode | How SVS Is Used |
|------|-----------------|
| **STS** | Extracts vocals and music from the source (vocals for conversion, music for recombination), and clean vocals from the target reference |
| **TTS** (voice clone) | Extracts clean vocals from target references before cloning; multi-reference targets (`(path1)(path2)`) are SVS-cleaned individually then concatenated |
| **STT** | Pre‑cleanup to isolate vocals from music before transcription |
| **TTS** (SLC) | Vocal isolation from source audio before transcription and translation to English; optional music extraction for music preservation |
| **TTS** (Modify Speech) | Vocal isolation before transcription for better accuracy |
| **SS** | Stage 1 voice isolation for speaker separation |
| **TTM** | Extracts vocals or music from source/reference audio for remix/complete/lego tasks; remix also accepts optional `lyrics` |

In all internal uses, if SVS extraction fails for any reason, VODER gracefully falls back to using the original audio. This means you never lose functionality — SVS is an enhancement, not a requirement.

**CLI Usage:**

```bash
# Extract vocals from a song
python src/voder.py svs voice "path/to/song.mp3"

# Extract instrumental (music without vocals)
python src/voder.py svs music "path/to/song.mp3"

# Extract both stems (voice first, then music)
python src/voder.py svs both "path/to/song.mp3"

# Save to specific location
python src/voder.py svs voice "path/to/song.mp3" result "output_vocals.wav"
python src/voder.py svs music "path/to/song.mp3" result "output_instrumental.wav"
python src/voder.py svs both "path/to/song.mp3" result "output/"

# Interactive CLI
python src/voder.py cli
# Select mode 7 (SVS), then follow prompts
```

**Best For:**

- Creating karaoke tracks (removing vocals)
- Isolating vocals for voice cloning references
- Creating instrumental versions of songs
- Pre‑processing audio before voice conversion
- Cleaning up reference audio for TTS voice cloning
- Audio analysis and music production workflows

**Technical Notes:**

SVS works on both CPU and GPU. GPU acceleration significantly speeds up separation for longer audio files. The BS‑RoFormer model is loaded on-demand from the `src/models/svs/` directory and offloaded after processing to prevent memory accumulation.

**Memory Requirements:** SVS requires approximately 12GB RAM (8GB base + 3-4GB for BS‑RoFormer model).

---

### SS: Speakers Separator

**What It Does:**

SS (Speakers Separator) extracts individual speakers from multi‑speaker audio. Given an audio file with multiple people talking, SS identifies each speaker, isolates their speech, and produces a separate audio file for each speaker. It uses a multi‑stage pipeline combining voice separation, speech enhancement, speaker diarization, and target speaker extraction.

**How It Works:**

SS uses a sophisticated multi‑stage pipeline:

1. **Stage 1 — SVS Voice Isolation**: BS‑RoFormer isolates the vocal track from background music, noise, and other non‑speech elements. This ensures clean input for the speaker identification stage.

2. **Stage 1b — Speech Enhancement** (optional, when `se` flag is set): UniSE further enhances the isolated vocals, removing remaining noise and reverberation for even cleaner speaker separation.

3. **Stage 2 — Speaker Identification**:
   - **Standard mode**: Whisper transcribes the audio, then Pyannote performs speaker diarization to identify who spoke when. The two outputs are aligned using VODER's three‑tier system.
   - **Overdose mode**: VibeVoice ASR handles both transcription and speaker identification in a single pass, providing higher quality segmentation with built‑in speaker labels. Requires 24GB+ VRAM or 48GB+ combined system memory.

4. **Stage 3 — Target Speaker Extraction**: For each detected speaker, UniSE's Target Speaker Extraction (TSE) capability isolates that speaker's voice from the full audio. The longest speech segment per speaker is used as an enrollment clip, and TSE extracts that speaker's voice across the entire recording.

5. **Output**: Each speaker is saved as a separate WAV file: `voder_ss_<name>_<timestamp>_speaker1.wav`, `voder_ss_<name>_<timestamp>_speaker2.wav`, etc.

**Standard vs Overdose Mode:**

| Feature | Standard (Whisper + Pyannote) | Overdose (VibeVoice ASR) |
|---------|-------------------------------|--------------------------|
| Transcription quality | Good | Higher |
| Speaker identification | Whisper + Pyannote alignment | Native built‑in |
| Requirements | 20GB RAM | 24GB+ VRAM or 48GB+ RAM |
| HF_TOKEN required | Yes (for Pyannote) | No |
| Best for | Standard use cases | Maximum quality |

**Target-Based Extraction:**

SS also supports target‑based extraction when a `target` reference audio is provided. Instead of separating all speakers, it extracts only the voice matching the target reference from the source audio. This uses UniSE TSE with the provided reference as an enrollment signal.

```bash
# Extract a specific voice from a multi-speaker recording
python src/voder.py ss "multi_speaker_audio.wav" target "voice_to_extract.wav"
```

**Why It's Like That:**

Speaker separation is one of the hardest problems in audio processing. Unlike source separation (which separates vocals from music — a relatively clear frequency boundary), speaker separation must distinguish between multiple voices that occupy the same frequency range. The multi‑stage approach exists because no single model does everything well. BS‑RoFormer handles the easy part (removing non‑speech), the diarization stage handles the hard part (identifying who's who), and UniSE TSE handles the hardest part (extracting a specific speaker from a mixture). The Overdose option exists for users with the hardware to use VibeVoice ASR, which provides better speaker segmentation as a single model.

**CLI Usage:**

```bash
# Separate all speakers from audio
python src/voder.py ss "path/to/audio.wav"

# Separate speakers with speech enhancement
python src/voder.py ss se "path/to/audio.wav"

# Separate speakers using overdose mode (higher quality)
python src/voder.py ss "path/to/audio.wav" overdose

# Extract a specific voice using a target reference
python src/voder.py ss "path/to/multi_speaker.wav" target "target_voice.wav"

# From a video file
python src/voder.py ss "interview.mp4"

# From a YouTube URL
python src/voder.py ss "https://www.youtube.com/watch?v=VIDEO_ID"

# Target extraction from YouTube
python src/voder.py ss "https://www.youtube.com/watch?v=VIDEO_ID" target "reference.wav"

# Interactive CLI
python src/voder.py cli
# Select mode 8 (SS), then follow prompts
```

**Best For:**

- Processing podcast recordings with multiple hosts
- Meeting and interview analysis
- Extracting individual speaker audio for voice cloning
- Creating clean audio samples from multi‑speaker recordings
- Academic and research applications
- Forensic audio analysis

**Technical Notes:**

SS mode works on both CPU and GPU. The standard pipeline requires HF_TOKEN for Pyannote diarization. The Overdose pipeline does not require HF_TOKEN but demands significantly more GPU resources. Each stage loads and offloads its model independently to manage memory usage. The TSE extraction stage uses the longest continuous speech segment per speaker as an enrollment signal.

**Memory Requirements:** SS (standard) requires approximately 20GB RAM (8GB base + ~3GB BS‑RoFormer + 4GB Whisper + 2-3GB Pyannote + 2-3GB UniSE TSE). SS (overdose) requires approximately 24GB RAM, though 24GB+ VRAM is recommended for VibeVoice ASR.

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
| **Dialogue source analysis** | Analyzes multi‑speaker audio to generate a dialogue script for TTS |
| **Voice clip extraction** | Identifies speakers and selects the best reference clip per speaker |
| **SS mode** (standard) | Speaker identification for target speaker extraction |
| **SS mode** (overdose) | Replaced by VibeVoice ASR's built‑in speaker identification |

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
python src/voder.py stt "screenshot.png"
# Output: results/voder_stt_screenshot.txt
```

**2. Dialogue Source Analysis:**

When using dialogue source analysis (e.g., in TTS interactive CLI), if you provide an image file as the source, VODER extracts the text via OCR and then proceeds to analyze it for dialogue content. Text formatted with character prefixes (like "James: Hello") is parsed into a dialogue script automatically.

**Technical Notes:**

EasyOCR runs entirely on CPU — no GPU is needed. It supports 80+ languages including English, Chinese, Japanese, Korean, and most European languages. Language detection is automatic; no configuration is needed.

Memory usage for EasyOCR is minimal (a few hundred MB) on top of VODER's base requirements. The OCR models are stored in `src/models/easyocr/` as part of the centralized model management system.

---

## YouTube & Video Platform Support

VODER can download audio directly from YouTube and other video platforms, then process it with any mode that accepts audio input. This eliminates the manual step of downloading files with a separate tool.

### Supported Platforms

| Platform | URL Patterns |
|----------|-------------|
| YouTube | `youtube.com/watch?v=*`, `youtu.be/*`, `youtube.com/shorts/*` |
| Bilibili | `bilibili.com/video/*`, `b23.tv/*` |
| TikTok | `tiktok.com/@user/video/*`, `vm.tiktok.com/*` |

### How It Works

When VODER detects a URL as input (starting with `http://` or `https://`), it:

1. Uses `yt-dlp` to download the best available audio stream
2. Converts the audio to MP3 format at 192kbps quality
3. Saves the temporary file for processing
4. Cleans up the temporary file after processing completes

The download happens automatically — you just paste the URL where VODER expects an audio file path.

### Cross-Mode Integration

YouTube/video support works across multiple VODER modes:

| Mode | YouTube Support |
|------|----------------|
| STT | Direct transcription from URL |
| TTS (voice clone) | Use YouTube video as voice reference via `target` parameter |
| TTS (dialogue source) | Use video as dialogue source |
| Voice clip extraction | Extract clips from YouTube video |
| STS | YouTube video as target voice reference |
| TTS (SLC) | Direct translation to English from YouTube URL, with optional `music` flag for preserving background music |
| SS | Direct speaker separation from YouTube URL |

### Error Handling & Fallbacks

- **Invalid URLs**: Clear error message, processing stops
- **Private videos**: Error message explaining the limitation
- **Region-locked content**: Error message, cannot process
- **Network errors**: Retry suggestion with connection check
- **Format fallbacks**: If MP3 conversion fails, falls back to M4A, WAV, or WebM

---

## Voice Clip Extraction

### What It Does

Voice clip extraction automatically identifies individual speakers in multi‑speaker audio and extracts a voice reference clip for each speaker. This eliminates the manual work of finding clean reference audio for voice cloning.

### How It Works

The extraction pipeline combines multiple VODER capabilities:

1. **Whisper Transcription**: Transcribes the audio with word‑level timestamps
2. **Pyannote Diarization**: Identifies speakers and their segments
3. **Speaker-to-Segment Mapping**: Each word is attributed to a speaker
4. **Longest Segment Selection**: For each speaker, finds their longest continuous speech segment
5. **FFmpeg Extraction**: Extracts the audio clip for each speaker's longest segment

The result is a set of voice reference clips, one per detected speaker, ready for use in TTS mode.

### Integration with TTS

In TTS interactive CLI mode with voice cloning, after you enter your dialogue script, VODER asks if you have a multi‑speaker audio source. If you provide one:

1. Voice clips are extracted automatically
2. Speakers are labeled numerically (1, 2, 3...)
3. Clips are matched to dialogue characters **alphabetically**
4. You can accept the auto-assignment or provide manual paths

### YouTube URL Support

Voice clip extraction works directly with YouTube URLs. If you provide a YouTube video URL as the multi-speaker source:

1. Audio is downloaded via yt-dlp
2. Extraction proceeds as normal
3. Temporary files are cleaned up automatically

---

## The Dialogue System

### What Dialogue Mode Is

Dialogue mode is VODER's system for creating multi-speaker audio content. Instead of generating a single voice speaking all the text, dialogue mode lets you create scripts where different characters speak different lines, each with their own voice.

### How It Works

1. **Script Input**: You enter lines in `Character: text` format
2. **Character Detection**: VODER automatically extracts unique character names
3. **Voice Assignment**: For each character, you provide a voice prompt (VoiceDesign) or reference audio (voice clone)
4. **Line-by-Line Generation**: Each line is synthesized separately
5. **Concatenation**: All lines are joined into a single audio file
6. **Optional Music**: Background music can be generated and mixed in

### Dialogue Source Analysis

VODER can analyze existing audio to generate dialogue scripts:

**Audio/Video Files:**
- Whisper transcribes with timestamps
- Optional Pyannote diarization identifies speakers
- Output is a structured dialogue script

**Images:**
- EasyOCR extracts text
- Text is parsed for dialogue format

**Text Files:**
- Parsed directly for character:text format

**YouTube URLs:**
- Downloaded, transcribed, and optionally diarized

### Dialogue Input in GUI

The GUI provides a row-based dialogue editor:

1. Each row has **Character** and **Dialogue** fields
2. New rows auto-add when you fill the last row
3. First row cannot be deleted; subsequent rows have delete buttons
4. Voice prompts (VoiceDesign) or audio number dropdowns (voice clone) appear for each detected character
5. SFX lines can be added using `sfx` as the character name

### Dialogue Input in CLI

#### Interactive CLI Dialogue

In interactive CLI mode:

1. Enter multiple lines, one per prompt (empty line to finish)
2. Lines without colons → single mode
3. Lines with colons → dialogue mode
4. VODER prompts for voice/audio for each character
5. Optional: Add background music with description

#### One‑Liner Dialogue

One-liner commands support dialogue via repeated parameters:

```bash
python src/voder.py tts \
  script "James: Hello" \
  script "Sarah: Hi there" \
  voice "James: deep male" \
  voice "Sarah: cheerful female" \
  music "soft piano" \
  level "0:30-60:50"
```

**Cross-use Feature (Mixing Generated and Cloned Voices):**

TTS one-line mode supports mixing generated and cloned voices in the same dialogue. Use `voice "Character: prompt"` for generated voices and `target "Character: path"` for cloned voices:

```bash
# TTS mode with mixed voices: James uses generated, Sarah uses cloned
python src/voder.py tts \
  script "James: Hello!" \
  script "Sarah: Hi there!" \
  voice "James: deep male voice" \
  target "Sarah: /path/to/sarah_voice.wav"
```

**Important:** A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

### Voice Prompt Configuration

**VoiceDesign Mode:**
- Each character gets a text field for voice description
- Prompts should describe vocal characteristics naturally
- Examples: "deep male, authoritative", "young female, energetic"

**Voice Clone Mode:**
- Load reference audio files (numbered 1, 2, 3...)
- Each character gets a dropdown to select an audio number
- Same audio can be used for multiple characters

---

## Script Directives

VODER now supports powerful **per-line directives** that can be appended to any dialogue line for fine-grained control over timing, volume, and duration.

### Time Positioning

The `/time:` directive controls when a line appears in the output timeline and allows trimming:

| Format | Meaning |
|--------|---------|
| `/time:5` | Position this line at 5 seconds from start |
| `/time:10-3` | Position at 10s, cut 3 seconds from end |
| `/time:5+2` | Position at 5s, cut 2 seconds from start |
| `/time:10-3+2` | Position at 10s, cut 3s from end, cut 2s from start |

**Use Cases:**
- Create overlapping dialogue
- Position sound effects at specific times
- Trim silence or unwanted sections from generated audio
- Create precise audio timelines without manual editing

**Example:**
```plaintext
James: Welcome to our podcast! /time:0
sfx: intro music fade /duration:5 /level:40 /time:0
Sarah: Thanks for having us! /time:2
James: Today we're discussing AI. /time:8
```

### Volume Level Control

The `/level:` directive sets the volume for a specific line:

| Format | Meaning |
|--------|---------|
| `/level:100` | Full volume (default) |
| `/level:75` | 75% volume |
| `/level:50` | 50% volume |
| `/level:25` | 25% volume (quiet background) |

**Use Cases:**
- Lower background characters or ambient dialogue
- Make sound effects subtle in the mix
- Create dynamic volume variations

**Example:**
```plaintext
Narrator: Once upon a time... /level:100
James: [whispering] Did you hear that? /level:40
sfx: distant footstep /duration:3 /level:30
Sarah: What was that? /level:90
```

### Duration for SFX

The `/duration:` directive is **required** for SFX lines and specifies the sound effect length:

| Format | Meaning |
|--------|---------|
| `/duration:3` | 3-second sound effect |
| `/duration:10` | 10-second sound effect |
| `/duration:30` | 30-second sound effect (maximum) |

**Note:** Regular dialogue lines do not use this directive — duration is determined by the speech generation model. SFX lines **must** include this directive.

---

## SFX Lines in Dialogue

You can now embed **sound effects directly in dialogue scripts** using the special `sfx:` character:

**Syntax:**
```plaintext
sfx: <sound description> /duration:<seconds> [/level:<0-100>] [/time:<position>]
```

**Requirements:**
- Character field must be `sfx` (case-insensitive)
- `/duration:nn` is **mandatory** (1-30 seconds)
- `/level:0-100` is optional (default: 100)
- `/time:nn` is optional for positioning

**Examples:**
```plaintext
James: Welcome to our show!
sfx: audience applause /duration:5 /level:60
Sarah: Thank you, thank you!
sfx: door creaking open /duration:3 /level:40
James: Looks like we have a guest!
sfx: mysterious ambient drone /duration:15 /level:25 /time:0
```

**Technical Details:**
- SFX generation uses the TangoFlux model
- SFX lines are generated during the dialogue assembly process
- Position with `/time:` directive for precise placement
- Volume controlled by `/level:` directive

**Note:** This dialogue SFX syntax (`sfx: description /duration:N /level:N /time:N`) is distinct from the TTM SFX overlay syntax used in BGM and Complete sub-tasks (`sfx "prompt/duration-position/level"`). The dialogue syntax uses slash-prefixed directives with colons; the TTM syntax uses a compact slash-delimited format. Both use the same TangoFlux model under the hood.

---

## Optional Background Music for Dialogue

### How It Works

When background music is enabled for dialogue:

1. **Dialogue Generation**: All dialogue lines are synthesized and concatenated
2. **Duration Measurement**: The total dialogue duration is measured
3. **Music Generation**: ACE-Step generates music matching the exact duration
   - Lyrics: `"..."` (empty placeholder for instrumental only)
   - Style: Your provided music description
4. **Mixing**: Music is mixed with dialogue at the specified volume level
5. **Cleanup**: Temporary files are removed, final output saved with `_m` suffix

### GUI Workflow

1. Enter dialogue in the row-based editor
2. Click Generate
3. A dialog appears: *"Enter music description (or press Skip):"*
4. Enter description (e.g., "soft piano, cinematic") or press Skip
5. Optionally enter music level specification
6. Processing continues with or without music

### Interactive CLI Workflow

1. Enter dialogue lines
2. Enter voice prompts/audio paths for each character
3. Prompt appears: `Add background music? (y/N):`
4. If yes, enter music description
5. Optionally enter level specification
6. Processing continues

### One‑Liner CLI Workflow

Add `music "description"` and optionally `level "spec"` and `reference "path"` parameters:

```bash
python src/voder.py tts \
  script "James: Hello" script "Sarah: Hi" \
  voice "James: male" voice "Sarah: female" \
  music "soft piano" \
  level "0:30-60:50"

# With reference audio for style guidance
python src/voder.py tts \
  script "James: Hello" script "Sarah: Hi" \
  voice "James: male" voice "Sarah: female" \
  music "soft piano" \
  reference "path/to/style_ref.wav"

# With video file as reference for style guidance
python src/voder.py tts \
  script "James: Hello" script "Sarah: Hi" \
  voice "James: male" voice "Sarah: female" \
  music "soft piano" \
  reference "path/to/style_ref.mp4"

# With YouTube URL as reference for style guidance
python src/voder.py tts \
  script "James: Hello" script "Sarah: Hi" \
  voice "James: male" voice "Sarah: female" \
  music "soft piano" \
  reference "https://youtube.com/watch?v=..."
```

The optional `reference` parameter provides a reference audio, video, or URL that is processed through the SVS music pipe (BS-RoFormer) to extract clean instrumental content before being passed to ACE-Step as stylistic guidance. This is useful when you want the generated background music to match the style or feel of a specific existing track. Video files have their audio extracted automatically, and YouTube/TikTok/Bilibili URLs are downloaded as audio-only before processing.

### Music Volume Level Control

The `level` parameter provides fine-grained control over background music volume throughout the dialogue:

**Format Options:**

| Format | Meaning | Example |
|--------|---------|---------|
| `"volume"` | Constant volume percentage | `"35"` = 35% throughout |
| `"start:vol-end:vol"` | Different volumes at different times | `"0:30-60:50"` = 30% at 0s, 50% at 60s |
| `"start:from-to+fade"` | Fade between volumes | `"0:30-60:50+10"` = fade from 30% to 50% over 10s starting at 0s |

**Examples:**

```bash
# Constant volume
level "35"

# Start quiet, get louder
level "0:20-120:60"

# Fade in at the beginning
level "0:0-10:35+5"

# Complex: quiet intro, louder middle, quiet outro
level "0:20-30:50-90:30"
```

**Default Behavior:**

If `level` is not specified, music is mixed at 35% volume throughout the dialogue.

### Technical Implementation

- FFmpeg volume filter with time-based expressions
- Frame-level evaluation for smooth transitions
- Automatic duration detection from dialogue file
- Memory-efficient streaming for long audio

---

## TTM Mode: BGM Subtask (Replace Background Music)

### What It Is

The TTM BGM subtask replaces background music in an existing audio or video file. It strips the current music from the source using SVS voice separation, generates new background music via ACE-Step, and mixes it at a configurable volume level. This is useful for replacing unwanted music in podcasts, interviews, videos, or any recording where you want to change the ambient soundtrack while preserving speech content. It also supports **SFX overlay** — sound effects generated by TangoFlux and placed at specific positions with controlled volume, overlaid after the BGM mixing step (or directly on clean voice if no music is provided).

### How It Works

1. **Source Resolution**: The input (audio file, video file, or URL) is resolved to a local audio file. With the `video` flag and a URL source, the video file is downloaded (not just audio) for later re-muxing.
2. **Music Stripping**: BS-RoFormer (SVS voice pipe) separates the source into clean vocals/speech and instrumental
3. **Duration Detection**: The duration of the clean audio is measured
4. **Music Generation** (if `music` is provided): ACE-Step generates new background music matching the detected duration
   - Uses ACE-Step turbo 1.5 model (standard) or ACE-Step XL 1.5 turbo model (overdose)
   - Long durations are handled by generating 250-300s chunks and concatenating
   - If a `reference` is provided (audio, video, or URL), it is processed through SVS music pipe to extract clean instrumental for style guidance
5. **Mixing** (if music was generated): New music is mixed with clean vocals at the specified volume level (0-100, default 35)
6. **SFX Overlay** (if `sfx:` specs are provided): ACE-Step is offloaded from VRAM, then TangoFlux loads and generates each sound effect. Each SFX is overlaid at its specified position and volume onto the result from step 5 (or directly onto clean vocals if no music was provided). Default SFX volume is 50%.
7. **Output**: If the source was video, the final audio is re-muxed back into the video container. With `video` flag + URL source, the video is downloaded and the result is merged back into .mp4.

### CLI Usage

```bash
# Standard quality (ACE-Step turbo 1.5)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30

# Overdose quality (ACE-Step XL 1.5 turbo)
python src/voder.py ttm overdose bgm "video.mp4" music "cinematic orchestral" level 50

# With reference for style guidance
python src/voder.py ttm bgm "recording.wav" music "jazz lounge" level 35 reference "style_ref.wav"

# From YouTube URL (audio only output)
python src/voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient chill" level 25 result "/output/new_bgm.wav"

# From YouTube URL with video output (downloads video, replaces bgm, outputs .mp4)
python src/voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"

# BGM with SFX overlay (replace music + add sound effects)
python src/voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30 sfx "phone ringing/6-45/70"

# BGM with multiple SFX overlays
python src/voder.py ttm bgm "interview.wav" music "light jazz" level 25 sfx "door opening/5-3/60" sfx "thunder clap/8-90/45"

# SFX overlay only (no new music — overlay effects directly on clean voice)
python src/voder.py ttm bgm "interview.wav" sfx "coffee shop ambience/30-0/25" sfx "doorbell/3-60/65"
```

### Output Naming

- Audio sources: `voder_ttm_bgm_{original-name}_{timestamp}.wav`
- Video sources: `voder_ttm_bgm_{original-name}_{timestamp}.mp4`

### Key Rules

- `bgm` requires `music` **or** `sfx:` (at least one must be provided)
- If only `sfx:` specs are provided (no `music`), no new background music is generated; sound effects are overlaid directly onto the clean voice after music stripping
- `bgm` cannot be combined with `vc`, `remix`, `repaint`, `complete`, `lego`, or `extract`
- Source accepts audio files, video files, and URLs (YouTube, Bilibili, TikTok)
- `video` flag: when source is a YouTube URL, downloads the video file (not just audio) and merges the result back into .mp4. For local video files, video output is automatic (no flag needed). If `video` is used with an audio source, outputs .wav with a warning.
- Reference accepts audio files, video files, and URLs — always processed through SVS music pipe for clean instrumental
- Normal (non-overdose) uses ACE-Step turbo 1.5; overdose uses ACE-Step XL 1.5 turbo
- Default volume level is 35 (range 0-100)
- `sfx:` spec format: `prompt/duration-position/level` — duration is 5-30s (auto-clamped), position is in seconds (cannot exceed source duration), level is 1-100% (default 50, auto-clamped)
- Multiple `sfx:` specs are allowed by repeating the `sfx` parameter
- When SFX overlay is active, ACE-Step is offloaded before TangoFlux loads to free VRAM

### GUI Support

In the GUI, TTM tab now includes a **BGM** sub-mode with fields for source file, music description, volume level (spinbox 0-100), and optional reference file picker.

### BGM Best Practices

1. **Match content genre** — Choose music descriptions that fit the content (jazz for interviews, orchestral for documentaries, electronic for tech reviews)
2. **Start low** — Default 35% is a safe starting point; increase gradually if speech clarity allows
3. **Use reference for style consistency** — Provide a reference track that matches the desired feel; SVS music pipe cleans it automatically
4. **Overdose for important content** — Use `overdose` flag when music quality is critical (final exports, professional productions)
5. **URL support** — You can directly reference YouTube, Bilibili, or TikTok URLs as the source, no manual download needed
6. **Video URLs with `video` flag** — Add the `video` keyword before the URL to download the video file and get a .mp4 output with replaced background music
7. **Use SFX overlay for sound design** — Add `sfx:` specs to place sound effects at specific moments without manual editing
8. **SFX-only mode for clean narration** — Skip `music` and use only `sfx:` when you just need spot effects on clean speech (e.g., adding ambient sounds to a dry recording)
9. **Position SFX carefully** — The position value must not exceed the source duration; check your source length before specifying positions

---

## TTM Mode: Instrumental Option

### Creating Instrumental Music

TTM mode now supports generating **music-only (no vocals)** output using empty lyrics:

**Using Empty Lyrics:**
```bash
# Generate instrumental background music
python src/voder.py ttm lyrics "..." styling "ambient electronic, chill" duration 60

# Generate cinematic score
python src/voder.py ttm lyrics "..." styling "orchestral strings, dramatic, cinematic" duration 90

# Generate lo-fi beat
python src/voder.py ttm lyrics "..." styling "lo-fi hip hop, chill, relaxing beat" duration 120
```

**Why It Works:**
- The ACE-Step model treats `"..."` as an empty lyrics placeholder
- Without lyrics content, the model generates instrumental music only
- Style prompt still guides the musical genre and mood

**Use Cases:**
- Background music for videos
- Ambient soundscapes
- Production music library
- Meditation/relaxation audio
- Game soundtracks

### Contextual Lyrics

Lyrics in parentheses `()` or brackets `[]` provide **context without being sung**:

```bash
# Context for style without actual lyrics
python src/voder.py ttm lyrics "(upbeat love song about summer)" styling "pop" duration 60
```

This helps the model understand the intended mood and structure while still producing instrumental or style-appropriate output.

---

## Tips & Tricks

### Getting Better Results

**For TTS Voice Prompts:**
- Be specific about age, gender, and tone
- Include speaking pace (fast, measured, slow)
- Add emotional qualities (warm, authoritative, friendly)
- Mention accent if relevant (British, Southern, etc.)

**For Voice Cloning References:**
- Use 10-30 seconds of clear speech
- Avoid background noise or music (SVS auto‑cleans if present)
- Single speaker only
- Natural conversational speech works better than reading

**For Music Generation:**
- Specify genre first, then mood
- Include instrumentation preferences
- Mention tempo or energy level
- Longer prompts give more control

### Multi-Speaker Scenarios

When working with multiple speakers:

1. **Use dialogue source analysis** — Let VODER automatically detect and label speakers
2. **Extract voice clips** — Use the auto-extraction feature for reference audio
3. **Match character names** — Use consistent naming between script and voice assignments
4. **Test voice consistency** — Generate a short test before full dialogue
5. **Consider SS mode** — Use Speakers Separator to isolate individual speakers as clean references

### Using Same Audio Source (Auto-Clone Trick)

A useful behavior when using the **same audio/video file** for both dialogue source analysis and auto-clone voice extraction:

**What Happens:**
1. Dialogue analysis generates character names as `1`, `2`, `3`... based on speaker detection
2. Auto-clone extracts the longest line per speaker, labeling them `speaker 1`, `speaker 2`, etc.
3. The system matches characters to voice references **alphabetically**

**The Trick:**
If you use the **same input file** for both dialogue source and auto-clone, the final output becomes an **exact replica of the original audio**!

**Use Cases:**
- Testing the TTS pipeline accuracy
- Verifying speaker detection quality
- Demonstrating voice cloning capabilities
- Creating backup/restoration of audio content

### Voice Cloning Best Practices

1. **Quality over quantity** — A clean 15-second clip beats a noisy 60-second clip
2. **Match the context** — Use reference audio similar to your target content
3. **Test first** — Generate a short sample before committing to long content
4. **Consistent recording** — Use the same microphone/environment when possible
5. **Let SVS handle cleanup** — Don't worry about background music in references; BS‑RoFormer will extract clean vocals automatically

### Background Music Best Practices

1. **Match the mood** — Music style should complement dialogue content
2. **Keep it subtle** — Default 35% volume is designed to not overwhelm speech
3. **Use level control** — Adjust volume for different sections (louder for intros, quieter for dialogue-heavy sections)
4. **Consider timing** — Use `/time:` directives to position SFX precisely
5. **Test mixing** — Generate without music first, then add music if needed
6. **Use reference for consistency** — Provide a reference audio via `reference "path"` when you want the generated music to stylistically match a specific track; the reference is cleaned via SVS music pipe to extract instrumental only
7. **Try TTM BGM for existing content** — For replacing music in an existing audio/video file, use `ttm bgm` instead of manually stripping and regenerating
8. **Add SFX overlay for immersion** — Use `sfx:` specs in BGM and Complete sub-tasks to add sound effects at specific moments without a separate post-production step

### Diarization Best Practices

1. **Clear audio** — Minimal background noise and music
2. **Distinct speakers** — Better accuracy with different voice types
3. **Adequate length** — 60+ seconds gives better speaker separation
4. **Limited speakers** — 2-4 speakers optimal; more than 6 reduces accuracy

### YouTube Download Tips

1. **Check availability** — Private or region-locked videos won't work
2. **Stable connection** — Network issues can corrupt downloads
3. **Patience for long videos** — Long content takes time to download
4. **Quality varies** — Source audio quality depends on original upload

### OCR Accuracy Tips

1. **High resolution** — Use the highest resolution image available
2. **Good contrast** — Dark text on light background works best
3. **Horizontal text** — Rotated or angled text may not be detected
4. **Clear fonts** — Handwritten or decorative fonts may have lower accuracy
5. **Crop if needed** — Focus on the text region for better results

### Voice Clip Extraction Best Practices

1. **Clear separation** — Audio where speakers don't overlap gives better clips
2. **Sufficient content** — Each speaker should have at least 5-10 seconds of speech
3. **Consistent quality** — Use recordings with consistent audio quality throughout
4. **YouTube sources** — Verify audio quality after download before extraction

### Sound Effects Best Practices

1. **Be descriptive** — Detailed prompts yield better results
2. **Include context** — "rain on metal roof" vs just "rain"
3. **Specify intensity** — "distant thunder" vs "loud thunder crash"
4. **Match duration to need** — Don't generate 30s for a 2s transition
5. **Test steps/guide** — Find your preferred quality/speed balance
6. **Layer with dialogue** — Use `/level:` to blend SFX with speech
7. **Use SFX overlay in TTM** — For BGM and Complete sub-tasks, use `sfx:` specs to place sound effects at precise positions without leaving the TTM workflow
8. **Respect duration limits** — SFX overlay duration is 5-30 seconds; values outside this range are auto-clamped with warnings
9. **Mind the position** — SFX overlay position must not exceed source duration; check your audio length first

### Speech Enhancement Best Practices

1. **Speech only** — Don't use on music; it's optimized for speech
2. **Moderate degradation** — Severely corrupted audio has limits
3. **Preview first** — Listen to enhanced output before using in production
4. **Chain operations** — Enhance before voice cloning for better results
5. **Match use case** — Output is 16kHz, ideal for speech applications

### SLC Tricks: Music Preservation & Voice Fidelity

SLC (now a TTS sub‑task: `tts slc`) always translates to English. Two powerful but non‑obvious tricks:

**Trick 1: Music Preservation (`music` flag):**

Add the `music` flag to preserve the non-vocal elements of the source. SLC extracts the instrumental track via SVS music pipe, translates the voice to English, then blends the instrumental back with the voice output. This is useful for dubbing songs or videos where you want to keep the background music intact. Note that voice-music synchronization may vary since the translated speech duration may differ from the original.

```bash
# Translate French song to English, keep the music
python src/voder.py tts slc music "french_song.wav"

# Overdose + music preservation for best quality
python src/voder.py tts overdose slc music "french_song.wav"
```

**Trick 2: Overdose for Better Voice Fidelity:**

SLC uses Qwen3-TTS Base to clone the original speaker's voice. For demanding use cases where voice fidelity is critical, add `overdose` to run an additional Seed-VC v2 non-mimic pass after TTS output. This applies voice conversion to further refine the output toward the original speaker's characteristics.

```bash
# Standard SLC: good voice preservation
python src/voder.py tts slc "french_speech.wav"

# Overdose SLC: maximum voice fidelity
python src/voder.py tts overdose slc "french_speech.wav"
```

### STS Mimic Language Warning

STS with the `mimic` parameter can produce lower quality results if the source speech is non‑English. The mimic style transfer relies on the AR model's understanding of speaking patterns, and this understanding is best for English. Normal STS (without `mimic`) gives very good quality regardless of what language the speech is in. If you're working with non‑English audio, use standard STS without mimic for the best results.

```bash
# Good for non-English: standard STS
python src/voder.py sts "non_english_speech.wav" "target_voice.wav"

# Potentially worse for non-English: mimic mode
python src/voder.py sts "non_english_speech.wav" "target_voice.wav" mimic
```

### Auto Vocal Extraction Trick

SVS (BS‑RoFormer vocal isolation) now runs automatically in several modes:

- **STS**: Vocals and music are extracted from the source (vocals for conversion, music recombined afterward), and clean vocals from the target reference
- **TTS (voice clone)**: Clean vocals are extracted from target references before cloning; multi-reference targets (`(path1)(path2)`) are individually cleaned and concatenated
- **TTS (Modify Speech)**: Vocals are isolated from the input before transcription

You don't need to manually isolate vocals before using them as references. Just provide the mixed audio directly — VODER handles the separation internally. This means you can use song clips, video snippets, or any audio with background elements as voice references without pre‑processing.

### Overdose STT Trick

For maximum transcription quality, use the STT `overdose` flag. VibeVoice ASR provides higher quality transcription with built‑in speaker identification, surpassing the standard Whisper + Pyannote pipeline. The trade‑off is resource requirements: you need 24GB+ VRAM or 48GB+ combined system memory.

```bash
# Standard STT: fast, good quality, low requirements
python src/voder.py stt "audio.wav" dialogue

# Overdose STT: higher quality, speaker-aware, high requirements
python src/voder.py stt "audio.wav" overdose
```

Note: Overdose cannot be combined with the `translate` flag, as VibeVoice ASR does not support translation.

### Extreme TTS Trick

The `extreme` keyword switches the TTS engine from Qwen3-TTS to **Fish Audio S2-Pro**, providing higher quality voice cloning and dramatically broader language support (80+ languages vs 10). This is especially useful when you need voice cloning for languages that Qwen3-TTS doesn't support, or when you want voice effects like `[whispering]`, `[laughing]`, `[excited]`, or `[pause]` embedded directly in your text.

**When to use extreme:**
- You need TTS in a language beyond Qwen3-TTS's 10 supported languages (Arabic, Hindi, Thai, Turkish, etc.)
- You want the highest possible voice cloning quality
- You want to use voice effects in your script text
- You're doing SLC or SVC and want better resynthesis quality

**Voice effects (`[tag]` syntax):**
Fish S2-Pro supports over 15,000 free-form voice effect tags embedded directly in your text. These tags control emotions, tones, vocal sounds, pacing, and special effects at the sub-word level. Tags are placed in `[brackets]` and affect the text from their position onward. The model also accepts all S1 Pro tags (listed below) inside `[brackets]`.

**S2-Pro well-tested tags:**

| Category | Tags |
|----------|------|
| **Emotions** | `[excited]`, `[angry]`, `[sad]` |
| **Tones / Voice Style** | `[whispering]`, `[soft voice]`, `[low voice]`, `[loud voice]`, `[shouting]` |
| **Breathing & Reactions** | `[sigh]`, `[inhale]`, `[exhale]`, `[gasp]`, `[panting]`, `[clears throat]` |
| **Vocal Sounds** | `[laughing]`, `[chuckling]`, `[giggle]`, `[sobbing]`, `[crying]`, `[groan]` |
| **Pacing** | `[pause]`, `[short pause]`, `[long pause]` |
| **Special** | `[emphasis]`, `[rustling sound]` |

**S1 Pro tags (also work in `[brackets]` for S2-Pro):**

These 64 tags were designed for Fish S1 Pro using `(parenthesis)` syntax, but they also work inside `[brackets]` with S2-Pro:

| Category | Tags |
|----------|------|
| **Emotions** | `(angry)` `(sad)` `(disdainful)` `(excited)` `(surprised)` `(satisfied)` `(unhappy)` `(anxious)` `(hysterical)` `(delighted)` `(scared)` `(worried)` `(indifferent)` `(upset)` `(impatient)` `(nervous)` `(guilty)` `(scornful)` `(frustrated)` `(depressed)` `(panicked)` `(furious)` `(empathetic)` `(embarrassed)` `(reluctant)` `(disgusted)` `(keen)` `(moved)` `(proud)` `(relaxed)` `(grateful)` `(confident)` `(interested)` `(curious)` `(confused)` `(joyful)` `(disapproving)` `(negative)` `(denying)` `(astonished)` `(serious)` `(sarcastic)` `(sneering)` `(hesitating)` `(yielding)` `(painful)` `(awkward)` `(amused)` |
| **Tone Markers** | `(in a hurry tone)` `(shouting)` `(screaming)` `(whispering)` `(soft tone)` |
| **Vocal Sounds** | `(laughing)` `(chuckling)` `(sobbing)` `(crying loudly)` `(sighing)` `(panting)` `(groaning)` |
| **Crowd Effects** | `(crowd laughing)` `(background laughter)` `(audience laughing)` |

Since the model accepts free-form descriptions, any natural language text in brackets works — you're not limited to the tags above. Examples of free-form descriptions: `[professional broadcast tone]`, `[pitch up]`, `[voice rough from crying, trying to sound normal]`, `[dead tired, end of a very long shift]`, `[calm, almost bored]`. Multi-language tags are also supported (e.g., `[低声说]` for Chinese "speak softly", `[囁き声で]` for Japanese "whisper voice").

```bash
# Extreme TTS with voice effects
python src/voder.py tts extreme script "[whispering] Hello there [pause] how are you?" target "voice.wav"

# Extreme TTS with voice design for Arabic (auto language trick)
python src/voder.py tts extreme script "مرحبا بالعالم" voice "deep male"

# Extreme TTS combined with overdose
python src/voder.py tts overdose extreme script "James: Hello" target "James: james.wav" music "soft piano"

# Extreme voice training (saves .ttse)
python src/voder.py train extreme voice:narrator "ref.wav"
```

**Voice Design language trick:** When `extreme` is used with a `voice` prompt (not `target`) and the text language is outside Qwen3-TTS VoiceDesign's 10 supported languages, VODER automatically handles this without any user intervention. It generates ~30 seconds of placeholder English text, has VoiceDesign speak it, feeds that audio to Fish S2-Pro for voice cloning, then Fish speaks the actual foreign-language text. This makes voice design available for 70+ additional languages that VoiceDesign doesn't natively support.

**Multi-speaker note:** Fish S2-Pro natively supports multi-speaker generation in a single pass using `Name: text` syntax (e.g., `SARAH: [sigh] I made coffee. DANIEL: [long pause] Yeah. Thanks.`) or via internal `<|speaker:i|>` tokens. However, VODER's dialogue mode is recommended over this feature because it provides better per-character voice control, mixing, and the ability to use different voice references for each character.

**`.ttse` vs `.tts` files:** Extreme mode uses `.ttse` trained voice files (saved via `train extreme voice:name`). Standard mode uses `.tts` files. Using a `.tts` file with extreme or a `.ttse` file without extreme produces a clear error message explaining the mismatch.

**STS voice pass with extreme:** The `sts:` prefix (which applies an additional Seed-VC v2 non-mimic pass) is not applicable with extreme mode. Fish S2-Pro's integrated cloning already produces high-fidelity output that doesn't benefit from an additional STS pass.

### Video STS Trick

STS now supports direct video input with MP4 output. Provide a video file as the base input, and VODER will extract the audio, perform voice conversion, and produce an MP4 video with the converted voice. This eliminates the manual steps of audio extraction, voice conversion, and video re‑encoding.

```bash
# Convert voice in a video directly
python src/voder.py sts "presentation.mp4" "narrator_voice.wav"
# Output: voder_sts_timestamp.mp4
```

### TTM Sub-Task Tricks

The new TTM sub‑tasks open up powerful music manipulation workflows:

**Remix (Style Transfer):**
Remix generates a style-transferred version of an existing song. The `bias` parameter (0–100, default 40) controls how much the new style is applied — 0 means pure original, 100 means pure new style. Use `voice` or `music` before a source path to pre-extract vocals or instruments from that source via SVS before remixing — this lets you remix only the vocal performance or only the instrumental, giving the model a cleaner source for more creative results. Optionally provide `lyrics` to guide the vocal content of the remix — this lets you create a remix with entirely new lyrics while keeping the musical vibe from the source. **Multi-source**: provide up to 3 source paths (each with optional `voice`/`music` prefix) and they are composed into one source with equal time allocation per source. **Multi-reference**: provide up to 3 reference entries (each with optional `voice`/`music` prefix) and they are composed into a 30s composite reference — 2 refs: 10s front of ref1 + 5s mid of each + 10s end of ref2; 3 refs: 10s front of ref1 + 10s mid of ref2 + 10s end of ref3.

```bash
python src/voder.py ttm remix "rock_song.wav" styling "acoustic jazz version" bias 50 result "/output/jazz_remix.wav"

# Remix with custom lyrics — new vocal content over the source vibe
python src/voder.py ttm remix "rock_song.wav" lyrics "sunshine in my heart" styling "acoustic jazz" result "/output/lyrics_remix.wav"

# Remix vocals only — isolate voice, then style-transfer
python src/voder.py ttm remix voice "rock_song.wav" styling "soulful R&B" result "/output/voice_remix.wav"

# Remix music only — isolate instruments, then style-transfer
python src/voder.py ttm remix music "rock_song.wav" styling "electronic synth" result "/output/music_remix.wav"

# Multi-source remix — vocals from one song, instruments from another
python src/voder.py ttm remix voice "song1.wav" music "song2.wav" styling "funk" result "/output/multi_remix.wav"

# Multi-reference remix — 2 references composed into 30s composite
python src/voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav" result "/output/remix.wav"

# Multi-reference remix — 3 references
python src/voder.py ttm remix "song.wav" styling "rock" reference "ref1.wav" voice "ref2.wav" music "ref3.wav" result "/output/remix.wav"
```

**Repaint Sections:**
Use `repaint` to fix or change a specific section of a song without regenerating the entire thing. Great for fixing a weak chorus or changing a bridge. The `time:start-end` parameter is **required** to specify the time range (single-pass mode). Optional `bias` (0–100, default 40) and `lyrics` (default `"..."`) parameters are available. Add `voice` or `music` prefix before the source path to isolate vocals or instruments via SVS before repainting. For multiple sequential edits, use **multi-pass mode** with quoted pass specs — each pass uses the previous result as the source, enabling creative layering of different styles, references, and lyrics across different time ranges.

```bash
python src/voder.py ttm repaint "song.wav" time:45-75 styling "more energetic vocals" result "/output/repainted.wav"
```

**Multi-pass Repaint:**
For complex edits, provide multiple quoted pass specs after the source path. Each spec contains a time range and optional parameters separated by `/`. The format is `"start-end[/styling(text)][/lyrics(text)][/reference-voice(path)][/reference-music(path)][/reference(path)][/bias/nn]"`. Each pass builds on the output of the previous pass.

```bash
# Two passes: restyle 20-80s as orchestral, then restyle 10-30s of that result as jazz
python src/voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"

# With per-pass references and lyrics
python src/voder.py ttm repaint "song.wav" "0-30/styling(funk)/lyrics(new words\nhere)" "15-30/styling(ambient)/reference(ref.wav)"

# Overdose multi-pass with voice reference on second pass
python src/voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/reference-voice(vocals.wav)"
```

**Add Missing Instruments:**
Use `complete` to add instruments to an existing track. If you have a vocal recording, you can add a full band behind it. Optionally use `styling` to influence the mood and genre of the generated instruments. Add `noblend` to output just the generated instruments without blending with the original source. The `complete` sub-task also supports **SFX overlay** via `sfx:` specs — sound effects are overlaid after the blend step. When only `sfx:` specs are provided (no `add`), the music model is not loaded at all, making it efficient for SFX-only overlays. Note that `sfx:` cannot be used with `noblend`.

The `voice` and `music` keywords isolate vocals or instruments from the source via SVS before processing — the model works on the isolated content and blends the result with the same isolated source by default. Add `usrc` (use source) to instead blend with the **original source before isolation**, which keeps the untouched parts intact in the final mix. `usrc` has no effect without `voice` or `music` (since there is only one source to blend with) and will be silently ignored with a warning in that case.

```bash
python src/voder.py ttm complete source "vocal_demo.wav" add "everything"

# With a styling prompt for mood control
python src/voder.py ttm complete source "vocal_demo.wav" add "drums bass" styling "dramatic cinematic"

# With noblend — generated instruments only (no mixing with original)
python src/voder.py ttm complete noblend source "vocal_demo.wav" add "drums bass"

# With SFX overlay — add instruments and overlay sound effects
python src/voder.py ttm complete source "vocal_demo.wav" add "drums bass" sfx "thunder rumble/10-5/60" sfx "door slam/5-30/80"

# SFX only — no instruments added, no music model loaded
python src/voder.py ttm complete source "narration.wav" sfx "wind howling/15-0/40" sfx "footsteps on gravel/8-20/55"

# With voice isolation — extract vocals, complete on vocals, blend with vocals
python src/voder.py ttm complete voice "song.wav" add "drums bass"

# With music isolation — extract instruments, complete on instruments, blend with instruments
python src/voder.py ttm complete music "song.wav" add "everything"

# Voice + usrc — extract vocals, complete on vocals, blend with original source (pre-isolation)
python src/voder.py ttm complete voice usrc "song.wav" add "drums bass guitar"

# Music + usrc — extract instruments, complete on instruments, blend with original source
python src/voder.py ttm complete music usrc "song.wav" add "everything"
```

**Build from Stems:**
Use `lego` to construct a custom arrangement from isolated stems. Extract individual tracks first, then rebuild with your preferred combination. Optionally use `styling` to influence the mood and genre of the generated instruments.

```bash
# First, extract what you have
python src/voder.py ttm extract "full_song.wav" extract "drums"

# Then, build around it
python src/voder.py ttm lego source "drums_only.wav" make "bass guitar strings"

# With a styling prompt for mood control
python src/voder.py ttm lego source "drums_only.wav" make "bass guitar" styling "jazz trio"
```

**Note:** The `complete`, `lego`, and `extract` sub‑tasks use the XL‑Base ACE‑Step model and require 32GB+ VRAM or 48GB+ system memory.

---

## Version Information

**Current Version:** 04/18/2026 (voder_bleed/3)

**Major Features:**
- 8 processing modes (STT, TTS, STS, TTM, SE, SFX, SVS, SS) plus TTS sub-tasks (SLC, Modify Speech) and dialogue/sub-task modes
- Unified TTS mode (VoiceDesign + voice cloning via target parameter)
- Unified TTM mode (generation + voice conversion + sub-tasks)
- SVS: Song Voice Separate with BS-RoFormer Resurrection
- SLC: Speaker Language Conversion (now a TTS sub‑task) with voice preservation
- SS: Speakers Separator with multi-stage pipeline
- STT translation support (Whisper large-v3, 99 languages)
- STT overdose mode (VibeVoice ASR)
- STT SVS pre-cleanup for song transcription
- TTS 10-language support via SUPPORTED_TTS_LANGUAGES
- Auto vocal extraction via BS-RoFormer in STS, TTS, TTS (SLC), TTS (Modify Speech); SLC also supports music preservation via `music` flag
- Video I/O support for STS (MP4 input → MP4 output)
- TTM three-tier system (standard, overdose, complete)
- TTM sub-tasks: complete, lego, extract, remix, repaint
- TTM 12 instrument tracks with shorthand expansion
- TTM SFX overlay for BGM and Complete sub-tasks
- Script directives for per-line control
- SFX character in dialogue
- Music volume level control
- TTM instrumental mode
- Auto-clone trick for exact replica
- SS target-based extraction

**Model Versions:**
- Whisper: large-v3-turbo (transcription), large-v3 (translation)
- VibeVoice ASR: microsoft/VibeVoice-ASR (overdose STT/SS)
- Qwen3-TTS: 12Hz-1.7B VoiceDesign (generated voices), 12Hz-1.7B Base (voice cloning)
- Seed-VC: v1 (44.1kHz for music) and v2 (22.05kHz for speech)
- ACE-Step: v15-turbo (standard), v15-xl-turbo (overdose/complete)
- ACE-Step LM: 5Hz-lm-1.7B (standard), 5Hz-lm-4B (overdose)
- Pyannote: speaker-diarization-community-1
- BS-RoFormer: BS-RoFormer Resurrection (SVS voice separation)
- UniSE: from alibaba/unified-audio (speech enhancement + TSE)
- TangoFlux: from declare-lab/TangoFlux
- EasyOCR: latest (image text extraction)

---

## Troubleshooting & Common Issues

### General Issues

**Issue: Out of memory errors**
- Solution: Ensure sufficient RAM for the mode you're using (see System Requirements)
- Solution: Close other memory-intensive applications
- Solution: For music modes, use shorter durations or disable overdose
- Solution: For SS mode, try standard mode instead of overdose

**Issue: Slow processing**
- Solution: All modes work on CPU; GPU speeds up certain modes
- Solution: Use shorter audio segments for STS
- Solution: For SFX, reduce `steps` parameter
- Solution: For TTM, use standard mode instead of overdose

**Issue: FFmpeg not found**
- Solution: Install FFmpeg and add to system PATH
- Solution: Verify with `ffmpeg -version`

### STT Issues

**Issue: Diarization fails with authentication error**
- Solution: Ensure HF_TOKEN.txt exists with valid token
- Solution: Accept conditions at pyannote model pages
- Solution: Verify token has read access to gated repositories

**Issue: YouTube download fails**
- Solution: Check internet connection
- Solution: Verify video is publicly available
- Solution: Update yt-dlp: `pip install --upgrade yt-dlp`

**Issue: Overdose mode fails to load**
- Solution: Ensure you have 24GB+ VRAM or 48GB+ combined system memory
- Solution: VODER automatically falls back to standard mode if resources are insufficient
- Solution: Overdose cannot be used with translate flag

**Issue: Translation produces poor results**
- Solution: Ensure audio has clear speech (use SVS pre-cleanup for songs)
- Solution: Whisper large-v3 supports 99 languages — check if your language is supported
- Solution: Shorter, cleaner audio segments produce better translations

### TTS Issues

**Issue: Voice quality inconsistent in dialogue**
- Solution: Voice is now extracted once per character automatically
- Solution: Use consistent reference audio quality
- Solution: BS-RoFormer auto‑extracts vocals from references with background music

**Issue: Background music not added**
- Solution: Music only works for dialogue mode (lines with colons)
- Solution: Ensure music description is not empty

**Issue: Language parameter not working**
- Solution: Verify the language code is one of the 10 supported languages
- Solution: Check that the text content matches the specified language

### STS Issues

**Issue: Mimic mode produces lower quality for non‑English**
- Solution: Use standard STS without mimic for non‑English source audio
- Solution: Normal STS works well regardless of language

**Issue: Video output doesn't play**
- Solution: Ensure FFmpeg is installed for video encoding
- Solution: Check that the input video has a valid audio track

### TTM Issues

**Issue: Overdose mode fails to start**
- Solution: Ensure you have 32GB+ VRAM for overdose/complete modes
- Solution: VODER automatically falls back to standard mode if resources insufficient
- Solution: Close other GPU-intensive applications

**Issue: Complete sub-task produces no output**
- Solution: Ensure valid instrument names are provided (or `sfx:` specs if using SFX-only mode)
- Solution: Use shorthand like "everything" or "vocals" for common combinations
- Solution: Check that the source audio is accessible and not corrupted

**Issue: VC cannot be used with other sub-tasks**
- Solution: VC is mutually exclusive with remix and repaint modes
- Solution: Use VC with generate mode only

**Issue: SFX overlay position exceeds source duration**
- Solution: The `position` value in `sfx:prompt/duration-position/level` must not exceed the source audio length
- Solution: Check your source file duration before specifying positions
- Solution: Invalid position values produce an error; use a valid non-negative number

**Issue: SFX overlay with noblend produces an error**
- Solution: `sfx:` cannot be used with `noblend` in the complete sub-task
- Solution: Remove `noblend` or remove `sfx:` specs

**Issue: BGM sub-task fails because no music or sfx provided**
- Solution: `bgm` requires at least `music` **or** `sfx:` — provide one or both
- Solution: Use `sfx:` only if you want to add sound effects without new background music

### SE Issues

**Issue: Enhancement degrades music quality**
- Solution: SE is designed for speech only; don't use on music

**Issue: Output sounds lower quality**
- Solution: 16kHz is normal for SE output; it's optimized for speech

### SVS Issues

**Issue: Separation quality is poor**
- Solution: Try higher quality source audio
- Solution: Very dense mixes may not separate perfectly — this is a known limitation

### SLC Issues

Note: SLC is now a TTS sub‑task (`tts slc`), not a standalone mode. SLC always translates to English using Whisper large-v3. These issues apply to the SLC sub‑task within TTS.

**Issue: Output doesn't sound like the original speaker**
- Solution: Ensure the source audio is clean and contains sufficient speech (10+ seconds)
- Solution: SVS voice isolation runs automatically; ensure the source isn't too noisy
- Solution: Try SLC overdose mode (`tts overdose slc`) which adds an STS v2 pass for better voice preservation

**Issue: Translation quality is poor**
- Solution: SLC uses Whisper large-v3 (not turbo) for maximum accuracy, but some languages have lower transcription quality
- Solution: Pre-process with speech enhancement (`se` mode) before SLC to improve transcription accuracy

**Issue: Voice-music sync is off when using `music` flag**
- Solution: This is inherent to the approach — the translated speech duration may differ from the original, causing sync drift with the instrumental track
- Solution: For best results, use sources where the instrumental and vocal timing are less critical

### SS Issues

**Issue: Only one speaker detected**
- Solution: Ensure the audio has clear speaker turns (not constant overlap)
- Solution: Try with speech enhancement (`se` flag) for cleaner input
- Solution: Overdose mode may detect more speakers than standard mode

**Issue: Pyannote token error**
- Solution: Standard SS mode requires HF_TOKEN for Pyannote
- Solution: Use overdose mode to bypass Pyannote requirement (needs more VRAM)

### SFX Issues

**Issue: Generated sound doesn't match prompt**
- Solution: Try higher `guide` value (7-10) for stricter adherence
- Solution: Make prompts more descriptive
- Solution: Increase `steps` for better quality

**Issue: SFX line in dialogue missing duration**
- Solution: `/duration:nn` is required for all SFX lines
