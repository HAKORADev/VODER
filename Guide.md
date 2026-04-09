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
  - [SE: Speech Enhancement](#se-speech-enhancement)
  - [SFX: Sound Effects Generation](#sfx-sound-effects-generation)
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
- [Version Information](#version-information)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Introduction & Vision

VODER is a professional‑grade voice processing tool that brings together **nine distinct audio transformation capabilities** in a single, unified interface. Unlike tools that force you to jump between multiple applications for different voice‑related tasks, VODER provides everything from standalone transcription to text‑to‑speech synthesis to music generation to sound effects to speech enhancement under one roof.

**What VODER Actually Does:**

At its core, VODER orchestrates state‑of‑the‑art AI models to perform voice‑related transformations. It can transcribe speech to text with speaker identification, generate speech from text using either designed voices or cloned references, transform one voice into another while preserving content, create music from lyrics with optional voice conversion for the vocalist, generate sound effects from text descriptions, enhance speech quality through denoising and dereverberation, download and analyze content directly from YouTube and other video platforms, extract voice clips from multi‑speaker audio for use as cloning references, and even read text from images using optical character recognition. This isn't about chasing the fastest processing times or highest frame rates — it's about achieving professional‑quality results that actually sound good.

**Why VODER Exists:**

The voice synthesis market is dominated by expensive commercial platforms that charge per character or per month. ElevenLabs, OpenAI, and others offer powerful capabilities, but at costs that add up quickly for creators, developers, and businesses alike. More importantly, no existing open‑source solution offered all nine processing capabilities in a unified interface. You could find separate tools for TTS, voice conversion, and music generation, but none that worked together seamlessly — and certainly none that could pull a video from YouTube, identify the speakers, extract voice references, and generate a complete dialogue with background music and sound effects.

VODER was built to fill this gap. The goal from day one was to create a local, free, open‑source alternative that doesn't compromise on quality. Is it perfect? No software is. But it works, it keeps improving, and it provides genuine utility without subscription fees or usage limits.

**What Makes VODER Different:**

Most voice processing tools focus on a single use case. VODER takes a different approach — it treats voice and audio processing as a unified problem space. The same interface that generates speech from text can also convert that speech between voices, and the same voice cloning technology can apply to both speech and singing. The same transcription engine that powers speech‑to‑text also drives speaker diarization for multi‑speaker analysis. The same sound generation model that creates background music can also produce custom sound effects. This integration enables workflows that would otherwise require multiple tools and significant manual effort.

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
| SE | 8GB | +2-3GB (UniSE) | 11GB | Optional | 4GB |
| SFX | 8GB | +3-4GB (TangoFlux) | 12GB | Optional | 4GB |

- **CPU**: 4-6 cores minimum for model loading and non-GPU operations
- **RAM**: 12GB minimum for basic modes (STT, TTS, STT+TTS, SE, SFX), 15GB for STT with diarization, 23GB for ACE-related modes (TTM, TTM+VC, or TTS/TTS+VC with music)
- **GPU (CUDA)**: Optional - all modes work on CPU. GPU acceleration significantly speeds up STS, TTM, and TTM+VC modes
- **VRAM**: 4GB minimum (6GB recommended, 16GB for best performance with music modes). STT and diarization modes are CPU-only and require no GPU.
- **Storage**: SSD recommended for model downloads and result saving

**VRAM Guidelines:**

| VRAM | Performance Level | Suitable Modes |
|------|-------------------|----------------|
| No GPU (CPU only) | Slow | All modes (STT, STT+diarization, OCR, SE, SFX included) |
| 4GB | Usable | TTS, TTS+VC (no music), STT+TTS, SE, SFX |
| 6GB | Minimum | TTS, TTS+VC (no music), STT+TTS, SE, SFX |
| 14GB | Mid-range | STS, all TTS modes, SE, SFX |
| 15-16GB | Recommended | TTS+VC with music, TTM, TTM+VC, all modes |
| 24GB | Maximum | All modes at full speed (RTX 4090) |
| T4 (16GB) | Server-grade | All modes (not typical consumer GPU) |

These aren't arbitrary numbers. They're based on actual testing of the models VODER uses.

---

## Why Hardcoded Models?

VODER uses hardcoded default models. This isn't an accident or a limitation — it's a deliberate design choice made for quality reasons.

### The Quality Imperative

The models VODER uses were selected because they represent the best available quality in their respective categories. Qwen3‑TTS for text‑to‑speech, Seed‑VC v2 for voice conversion, ACE‑Step for music generation, Whisper for speech‑to‑text, Pyannote for speaker diarization, EasyOCR for image text extraction, UniSE for speech enhancement, TangoFlux for sound effects — these aren't arbitrary choices. They're the result of evaluating multiple alternatives and selecting the ones that produce the best results.

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
├── tmp/                      # Temporary downloads in progress
├── checkpoints/
│   ├── whisper/              # Whisper STT model (whisper-turbo.pt)
│   ├── qwen_tts_voicedesign/ # Qwen3-TTS VoiceDesign model
│   ├── qwen_tts_base/        # Qwen3-TTS Base model
│   ├── seed_vc_v1/           # Seed-VC v1 (44.1kHz for music)
│   ├── seed_vc_v2/           # Seed-VC v2 (22.05kHz for speech)
│   ├── acestep/              # ACE-Step music generation models
│   ├── pyannote/             # Pyannote diarization pipeline
│   ├── easyocr/              # EasyOCR models and weights
│   ├── unise/                # UniSE speech enhancement model
│   └── tangoflux/            # TangoFlux sound effects model
```

**HuggingFace Cache Redirection:**

Some models (particularly Pyannote, EasyOCR, UniSE, and TangoFlux) are downloaded through HuggingFace. VODER sets the `HF_HOME` and `TRANSFORMERS_CACHE` environment variables to point to the `src/models/` directory. This means:

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
python src/voder.py stt "audio.wav"

# With timestamps
python src/voder.py stt "audio.wav" timestamp

# With speaker diarization
python src/voder.py stt "audio.wav" dialogue

# With both timestamps and diarization
python src/voder.py stt "audio.wav" timestamp dialogue

# Transcribe a YouTube video
python src/voder.py stt "https://www.youtube.com/watch?v=VIDEO_ID" timestamp dialogue

# Batch process multiple files
python src/voder.py stt "file1.mp3" "file2.wav" "file3.mp4"

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

When using TTS in **dialogue mode** (multiple speakers, script lines containing a colon), you can optionally add automatically generated background music. After the dialogue is synthesized, VODER generates a music track using ACE‑Step with empty lyrics `"..."` and a duration matching the exact length of the dialogue. The music is mixed at **35% volume** relative to the dialogue (configurable via `level` parameter), creating a subtle ambient bed. The final file is saved with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`). This feature is available in GUI (via a clean modal dialog), interactive CLI (prompt after voice prompts), and one‑liner CLI (optional `music` and `level` parameters). See [Optional Background Music for Dialogue](#optional-background-music-for-dialogue) for full details.

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

**OCR Input (Image to Narration):**

You can use the `ocr` parameter to extract text from an image and synthesize it as speech. VODER uses EasyOCR to extract text from the image, then generates narration using the extracted text:

```bash
python src/voder.py tts ocr "path/to/image.png" voice "professional male narrator"

python src/voder.py tts ocr "script_screenshot.jpg" voice "warm female voice"
```

This is useful for converting screenshots of scripts, slides, or documents into spoken narration without manual text entry.

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

Just like in TTS mode, when TTS+VC is used in **dialogue mode** you can optionally add automatically generated background music. The music is generated **after** all dialogue lines have been synthesized, concatenated, and voice‑cloned. It uses the same ACE‑Step process (empty lyrics, auto‑duration, configurable volume via `level` parameter) and the same output naming (`_m` suffix). The feature is accessible through the same GUI dialog, interactive CLI prompt, and one‑liner `music` and `level` parameters. This allows you to create fully produced podcast episodes, narrated stories, or interview segments with ambient background music — all in a single operation.

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

**OCR Input (Image to Narration with Voice Clone):**

You can use the `ocr` parameter to extract text from an image and synthesize it with voice cloning:

```bash
python src/voder.py tts+vc ocr "path/to/image.png" target "voice_reference.wav"

python src/voder.py tts+vc ocr "subtitle_image.jpg" target "speaker_clone.wav"
```

The extracted text is synthesized and then cloned to match the target voice reference.

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

**Technical Notes:**

TTM works on CPU without GPU. Processing time scales primarily with duration rather than lyrics length. The style prompt complexity doesn't significantly affect processing time but does affect the musical output characteristics.

**Memory Requirements:** TTM requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### TTM+VC: Text-to-Music + Voice Conversion

**What It Does:**

TTM+VC generates music from lyrics and style (same as TTM) and then applies voice conversion to change the vocalist's voice. This combines music generation with voice cloning for the singing voice.

**How It Works:**

The pipeline is straightforward: first generate the music with ACE‑Step (TTM stage), then apply Seed‑VC voice conversion to the vocal track (VC stage). The generated music's vocals are transformed to match your reference voice while preserving the melody, timing, and musical characteristics.

**Multi-line Lyrics in One‑Liner:**

Use `\n` for multi-line lyrics with voice conversion:

```bash
python src/voder.py ttm+vc lyrics "Intro:\nSoft piano notes\n\nVerse:\nWalking through the shadows\nFinding my way home\n\nChorus:\nWe are unstoppable\nNothing can bring us down" styling "epic cinematic rock with powerful vocals" duration 45 target "singer_reference.wav"
```

---

### TTM+VC: Text-to-Music + Voice Conversion

**What It Does:**

TTM+VC generates music from lyrics and style (same as TTM) and then applies voice conversion to change the vocalist's voice. This combines music generation with voice cloning for the singing voice.

**How It Works:**

The pipeline is straightforward: first generate the music with ACE‑Step (TTM stage), then apply Seed‑VC voice conversion to the vocal track (VC stage). The generated music's vocals are transformed to match your reference voice while preserving the melody, timing, and musical characteristics.

**Memory Optimisation:**

VODER explicitly offloads models from memory after each operation completes. This applies to all modes in both GUI and interactive CLI:

- **GUI Mode**: ProcessingThread calls cleanup() after finishing, releasing all loaded models
- **Interactive CLI**: Each mode offloads models before returning
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
# Select mode 7 (SE)
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
# Select mode 8 (SFX)
```

**Memory Requirements:** SFX requires approximately 12GB RAM (8GB base + 3-4GB for TangoFlux model).

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
python src/voder.py stt "screenshot.png"
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
| TTS+VC (dialogue source) | Use video as dialogue source |
| Voice clip extraction | Extract clips from YouTube video |

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

The result is a set of voice reference clips, one per detected speaker, ready for use in TTS+VC mode.

### Integration with TTS+VC

In TTS+VC interactive CLI mode, after you enter your dialogue script, VODER asks if you have a multi‑speaker audio source. If you provide one:

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
3. **Voice Assignment**: For each character, you provide a voice prompt (TTS) or reference audio (TTS+VC)
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
4. Voice prompts (TTS) or audio number dropdowns (TTS+VC) appear for each detected character
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

Both TTS and TTS+VC one-line modes support mixing generated and cloned voices in the same dialogue. Use `voice "Character: prompt"` for generated voices and `target "Character: path"` for cloned voices:

```bash
# TTS mode with mixed voices: James uses generated, Sarah uses cloned
python src/voder.py tts \
  script "James: Hello!" \
  script "Sarah: Hi there!" \
  voice "James: deep male voice" \
  target "Sarah: /path/to/sarah_voice.wav"

# TTS+VC mode with mixed voices: James uses cloned, Sarah uses generated
python src/voder.py tts+vc \
  script "James: Welcome!" \
  script "Sarah: Thanks!" \
  target "James: /path/to/james_voice.wav" \
  voice "Sarah: bright female voice"
```

**Important:** A character cannot have both `voice` and `target` assignments — each character must use either generated or cloned voice, not both.

### Voice Prompt Configuration

**TTS Mode:**
- Each character gets a text field for voice description
- Prompts should describe vocal characteristics naturally
- Examples: "deep male, authoritative", "young female, energetic"

**TTS+VC Mode:**
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

Add `music "description"` and optionally `level "spec"` parameters:

```bash
python src/voder.py tts \
  script "James: Hello" script "Sarah: Hi" \
  voice "James: male" voice "Sarah: female" \
  music "soft piano" \
  level "0:30-60:50"
```

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
- Avoid background noise or music
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

### Using Same Audio Source (Auto-Clone Trick)

A useful behavior when using the **same audio/video file** for both dialogue source analysis and auto-clone voice extraction:

**What Happens:**
1. Dialogue analysis generates character names as `1`, `2`, `3`... based on speaker detection
2. Auto-clone extracts the longest line per speaker, labeling them `speaker 1`, `speaker 2`, etc.
3. The system matches characters to voice references **alphabetically**

**The Trick:**
If you use the **same input file** for both dialogue source and auto-clone, the final output becomes an **exact replica of the original audio**!

**Use Cases:**
- Testing the TTS+VC pipeline accuracy
- Verifying speaker detection quality
- Demonstrating voice cloning capabilities
- Creating backup/restoration of audio content

### Voice Cloning Best Practices

1. **Quality over quantity** — A clean 15-second clip beats a noisy 60-second clip
2. **Match the context** — Use reference audio similar to your target content
3. **Test first** — Generate a short sample before committing to long content
4. **Consistent recording** — Use the same microphone/environment when possible

### Background Music Best Practices

1. **Match the mood** — Music style should complement dialogue content
2. **Keep it subtle** — Default 35% volume is designed to not overwhelm speech
3. **Use level control** — Adjust volume for different sections (louder for intros, quieter for dialogue-heavy sections)
4. **Consider timing** — Use `/time:` directives to position SFX precisely
5. **Test mixing** — Generate without music first, then add music if needed

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

### Speech Enhancement Best Practices

1. **Speech only** — Don't use on music; it's optimized for speech
2. **Moderate degradation** — Severely corrupted audio has limits
3. **Preview first** — Listen to enhanced output before using in production
4. **Chain operations** — Enhance before voice cloning for better results
5. **Match use case** — Output is 16kHz, ideal for speech applications

---

## Version Information

**Current Version:** 04/08/2026

**Major Features:**
- 9 processing modes (STT+TTS, TTS, TTS+VC, STS, TTM, TTM+VC, STT, SE, SFX)
- Script directives for per-line control
- SFX character in dialogue
- Music volume level control
- TTM instrumental mode
- Auto-clone trick for exact replica

**Model Versions:**
- Whisper: large-v3-turbo
- Qwen3-TTS: 12Hz-1.7B (VoiceDesign and Base)
- Seed-VC: v1 (44.1kHz) and v2 (22.05kHz)
- ACE-Step: 1.5
- Pyannote: speaker-diarization-community-1
- UniSE: from alibaba/unified-audio
- TangoFlux: from declare-lab/TangoFlux

---

## Troubleshooting & Common Issues

### General Issues

**Issue: Out of memory errors**
- Solution: Ensure sufficient RAM for the mode you're using (see System Requirements)
- Solution: Close other memory-intensive applications
- Solution: For music modes, use shorter durations

**Issue: Slow processing**
- Solution: All modes work on CPU; GPU speeds up certain modes
- Solution: Use shorter audio segments for STS
- Solution: For SFX, reduce `steps` parameter

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

### TTS/TTS+VC Issues

**Issue: Voice quality inconsistent in dialogue**
- Solution: Voice is now extracted once per character automatically
- Solution: Use consistent reference audio quality

**Issue: Background music not added**
- Solution: Music only works for dialogue mode (lines with colons)
- Solution: Ensure music description is not empty

### SE Issues

**Issue: Enhancement degrades music quality**
- Solution: SE is designed for speech only; don't use on music

**Issue: Output sounds lower quality**
- Solution: 16kHz is normal for SE output; it's optimized for speech

### SFX Issues

**Issue: Generated sound doesn't match prompt**
- Solution: Try higher `guide` value (7-10) for stricter adherence
- Solution: Make prompts more descriptive
- Solution: Increase `steps` for better quality

**Issue: SFX line in dialogue missing duration**
- Solution: `/duration:nn` is required for all SFX lines
