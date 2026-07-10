<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="256" height="256"/>
</p>

<h1 align="center">VODER</h1>

<p align="center">
  <strong>Local &bull; Free &bull; Offline</strong><br/>
  Professional-grade voice processing in a single tool.
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1hditIfW9JzusNcFhlHFoclCIIsNiRFNk?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/>
  </a>
</p>

---

**Jump to:** [Features](#features) · [What Can VODER Do?](#what-can-voder-do) · [Quick Start](#quick-start) · [Modes at a Glance](#modes-at-a-glance) · [Models Behind VODER](#models-behind-voder) · [System Requirements](#system-requirements) · [Documentation](#documentation) · [Contributing](#contributing)

---

VODER brings together **8 processing modes** under one interface — speech-to-text, text-to-speech, voice conversion, music generation, speech enhancement, sound effects, vocal separation, and speaker diarization — plus language dubbing (`tts dub`), any-to-any translation via TranslateGemma 12B, transcribe-edit-resynthesize (built into TTS interactive), three task-layer features that build on top of the modes: **`train`** (save reusable voice clones as `.tts` / `.ttse`), **side-quests** (`quest` — lightweight utility tasks like URL download and audio manipulation), and **chains** (user-defined pipelines that wire any number of voder oneline tasks together end-to-end) — and a natural-language AI agent, **VADAR**, that takes plain-English requests and runs any combination of the above for you. It runs entirely on your machine, needs no subscription, and works with or without a GPU.

---

## Features

- **Multi-Speaker Dialogue System** — Write scripts with multiple characters, each with a distinct voice. Control per-line timing, volume, and duration with script directives. Embed sound effects directly into dialogue lines and generate automatic background music that matches the spoken duration.
- **Voice Design & Cloning** — Describe a voice in plain English and VODER generates it, or provide a reference clip to clone a speaker's voice. Mix designed and cloned voices within the same dialogue.
- **Speaker Separation** — Extract individual speakers from multi-speaker recordings into separate audio files, each with a speaker-labeled transcript.
- **Voice Conversion with Video I/O** — Transform one voice into another while preserving words, emotion, and timing. Drop in an MP4 and get back a video with the converted voice.
- **Music Generation & Manipulation** — Generate full songs from lyrics and style descriptions. Remix, repaint, complete, extract stems, build individual instrument tracks, or replace background music in existing audio/video. Output up to 12 separate instrument tracks.
- **Speech-to-Text with Intelligence** — Transcribe audio, video, images, or direct URLs. Translate to any of 76 languages via TranslateGemma. Identify who spoke when with speaker diarization. Batch process multiple files.
- **Language Dubbing** — Translate speech from one language to another while preserving the original speaker's voice identity. Dub entire videos with per-segment timing alignment and background music preservation.
- **Any-to-Any Translation** — Translate between any of 76 languages using TranslateGemma 12B via the `translate (source-target)` syntax, decoupled from the ASR engine.
- **Voice Re-Synthesis** — Transcribe speech and re-read it in a different voice using `tts svc`, with an optional `sts:` prefix for high-fidelity voice conversion via Seed-VC v2.
- **Side-Quests** — Lightweight utility tasks that live outside the main engine: URL download, audio format conversion, cutting / merging / mixing / removing ranges, silence stripping, speed / pitch / soundlevel / bassboost / reverb / loudnorm effects, and more. Run `python voder.py quest` to see all available quests, grouped by category.
- **Chains** — User-defined pipelines that wire any number of voder tasks together: each chain is named, its output is captured to temp, and later chains can reference earlier chain names as input paths. Build a song, isolate its vocals, train a voice from them, then dub a video — all in one command.
- **VADAR AI Agent** — A natural-language agent that comes in two twins: **lite** (default, runs on 16GB RAM via llama.cpp + GGUF) and **heavy/overdose** (multimodal, requires 80GB+ RAM/VRAM via transformers+torch). Describe a task in plain English and VADAR thinks, decides, replies, and acts — running the right VODER commands in the right order, reading their outputs, and reporting results. Has its own tools (`read`, `list`, `search`, `memory_*`, `calculate`, catalog tools), session logging, persistent memories, and a configurable personality. The brotherhood (Eval, Summarizer, Catcher) works behind the scenes. No network access, no system shell — only VODER commands and project files.
- **Smart Input Pipeline** — Paste a YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, or X/Twitter URL directly as input. VODER verifies the link actually points to a video before downloading. Feed an image and VODER extracts text via OCR. Automatically extract voice clips from multi-speaker audio for one-click voice cloning.

---

## What Can VODER Do?

### Text-to-Speech with Voice Design & Cloning

Describe a voice in plain English — *"deep male voice, authoritative"* — and VODER generates speech that matches. Or provide a reference audio clip and VODER **clones the speaker's voice** from it. Both approaches can be **mixed in the same dialogue**: some characters designed, others cloned.

### Multi-Speaker Dialogue System

Write scripts with multiple characters, each with a distinct voice. VODER assembles the full dialogue into a single audio file with per-line control over timing, volume, and duration via **script directives** (`/time`, `/level`, `/duration`). Embed **sound effects** directly into dialogue lines using the special `sfx:` character — door creaks, applause, rain — generated on the fly from text descriptions.

### Automatic Background Music

When generating dialogue, VODER can produce a background music track that **exactly matches the spoken duration**, mixed at a configurable volume with fade transitions. An optional **reference** (audio, video, or URL) can be provided for stylistic guidance — the reference is processed through SVS to extract clean instrumental before use. No manual editing or external tools needed.

### Voice Conversion (Speech & Music)

Transform one voice into another while preserving the original words, emotion, and timing. Supports **video input/output** — drop in an MP4 and get back a video with the converted voice. For music, VODER switches to a high-fidelity 44.1kHz model. A **mimic** mode transfers not just the voice timbre but the accent and speaking style as well.

### Music Generation & Manipulation

Generate full songs from lyrics and style descriptions. Beyond basic generation, VODER supports **6 sub-tasks**: remix (style transfer with bias control), repaint (restyle a specific time range), complete (add missing instruments), lego (build individual tracks), extract (isolate specific stems), and bgm (replace background music in existing audio/video with generated music at a configurable volume). Output up to **12 individual instrument tracks** for post-production. A three-tier quality system lets you trade speed for output quality.

### Vocal & Music Separation

Isolate clean vocals from any song, or extract the instrumental. Works with audio files, videos, and direct URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter). This separation engine also runs automatically behind the scenes in TTS (to clean voice cloning references), STS (to improve conversion quality), and STT (to pre-clean audio before transcription).

### Speech-to-Text with Speaker Intelligence

Transcribe audio, video, images, or direct URLs to text. Supports **translation to English** from 99 languages, **speaker diarization** (who spoke when), and **batch processing** of multiple files. An **overdose mode** using Microsoft VibeVoice ASR delivers higher-quality transcription with built-in speaker identification.

### Sound Enhancement

Remove noise, reduce room echo, and restore clarity from degraded recordings. Upscale audio to 48kHz with AudioSR super-resolution (basic model for general audio, speech model for voice). Works on audio and video files alike.

### Speaker Separation

Extract individual speakers from multi-speaker recordings into separate audio files, each with a speaker-labeled transcript. The `blend` flag preserves non-vocals (background audio) in each speaker's output, and the `video` flag muxes separated audio with the original video for MP4 output — useful for removing unwanted speakers from a video while keeping the visuals.

### Language Conversion (TTS Sub-Task)

Translate speech from any language to English while **preserving the original speaker's voice identity** — `tts slc "audio.wav"`. Supports any-to-any translation via TranslateGemma with the `translate (source-target)` syntax. An optional `music` flag preserves the original instrumental track, and `overdose` adds a voice fidelity pass. Accepts audio files, videos, and supported platform URLs.

### Video/Audio Dubbing (TTS Sub-Task)

Dub entire videos to another language with **per-segment timing alignment** — `tts dub "video.mp4"` auto-translates to English by default. Uses VibeVoice ASR with audio events, TranslateGemma per-segment translation, Fish S2 Pro voice cloning, speed adjustment, and timeline assembly. Add `subtitle` to burn translated subtitles. Add `translate "(auto-ja)"` to target any language. Preserves background music.

### Smart Input Pipeline

Paste a URL from **YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, or X/Twitter** directly as input — VODER auto-detects the platform, verifies the link points to a real video (not a channel page, profile, photo post, or playlist), then downloads and processes it. Feed an **image** containing text and VODER extracts it via OCR for TTS processing. Automatic voice clip extraction from multi-speaker audio enables one-click voice cloning for dialogue characters.

### Side-Quests (`quest`)

Lightweight utility tasks that live outside the voder engine but are still useful in a voice-processing workflow — URL download, audio format conversion, cutting, merging, **mixing** (overlay multiple sources at specified start times), range removal, silence stripping, soundlevel / speed / pitch / bassboost / reverb / loudnorm, and more. Side-quests are grouped by category in the listing — `download` stands alone, the rest live under **Media Manipulation**. Run `python voder.py quest` (no args) to list every available side-quest with its one-line description. See [COMMAND_CATALOG.md](docs/COMMAND_CATALOG.md) for the full list.

### Chains (`chains`)

User-defined pipelines that wire any number of voder oneline tasks together. Each chain is named, runs a full voder oneline command, and its output is captured to a temp directory. Later chains can reference earlier chain names as input paths — VODER resolves them internally to the captured temp file. Side-quests work inside chains just like any other oneline task.

```
# TTM → SVS → STS pipeline
python src/voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"
```

Use ` / ` (space slash space) to separate chains. Intermediate chain outputs live in `temp_chains/`; only the **last** non-empty chain's output reaches `results/`. Empty chains are skipped (their names remain available for reuse); duplicate names cause an error and stop the pipeline. This lets you compose pipelines that VODER's built-in modes never anticipated — generate music, isolate vocals, train a voice from them, then dub a video, all in a single command.

### VADAR AI Agent (`vadar` / `overdose vadar`)

VADAR is the natural-language layer on top of everything else. Instead of remembering oneline syntax, you describe the task in plain English and VADAR figures out which VODER commands to run, in what order, and reads their outputs to verify the result. It runs locally with no network access — only VODER project files and paths you provide are reachable.

VADAR comes in two twins:

- **Lite VADAR** (default) — `python src/voder.py vadar "..."` — uses [SuperGemma 4 12B GGUF Q4_K_M](https://huggingface.co/Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit) via llama.cpp. Runs on 16GB RAM / 4 CPU cores / any T4 GPU. Text-only (no image/audio/video analysis). Model-level chain-of-thought enabled. ~7GB model.
- **Heavy VADAR** (overdose) — `python src/voder.py overdose vadar "..."` — uses [Gemma 4 12B abliterated uncensored](https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored) via transformers+torch. Multimodal (look/listen/watch). Requires 80GB+ RAM/VRAM. ~24GB model.

```
# Lite VADAR (default — runs on any machine with 16GB RAM)
python src/voder.py vadar "Generate a 30-second upbeat pop song about rain, then isolate its vocals"

# Heavy VADAR (overdose — multimodal, requires 80GB+ RAM/VRAM)
python src/voder.py overdose vadar "Look at this image and make a sound effect for it image.png"

# Interactive CLI — choose option 10, then pick lite or overdose
python src/voder.py cli
```

Both twins share the same agent architecture — tags (`<thinking>`, `<decide>`, `<reply>`, `<act>`, `<tool_call>`), the VODER brotherhood (Eval, Summarizer, Catcher), session logging, persistent memories, and configurable personality. The only difference is the model engine and whether `look`/`listen`/`watch` tools are available. Models download automatically on first run. See [Guide.md](docs/Guide.md) for the full VADAR user guide.

---

## Quick Start

```bash
git clone https://github.com/HAKORADev/VODER.git && cd VODER
pip install -r requirements.txt && pip install --upgrade protobuf==5.29.6

# Help
python src/voder.py

# GUI
python src/voder.py gui

# CLI (interactive)
python src/voder.py cli

# One-liner examples
python src/voder.py tts script "Hello world" voice "female, cheerful"
python src/voder.py stt "audio.wav" timestamp dialogue
python src/voder.py sts base "input.wav" target "voice.wav"
python src/voder.py ttm lyrics "Walking down the street" styling "upbeat pop" 30
python src/voder.py svs "song.mp3" voice
python src/voder.py ss "meeting.wav"
python src/voder.py tts slc "foreign_speech.wav"
python src/voder.py tts dub "video.mp4"
python src/voder.py tts dub translate "(auto-ja)" "video.mp4"
python src/voder.py tts svc "speech.wav" target "voice_ref.wav"
python src/voder.py se "noisy_recording.wav"
python src/voder.py sfx sound "thunder rumbling" duration 10

# Side-quests (URL download & utility tasks — see `python voder.py quest`)
python src/voder.py quest download "https://youtube.com/watch?v=..."

# Voice training (save a voice clone for reuse in TTS)
python src/voder.py train voice:narrator "ref1.wav" "ref2.wav"
python src/voder.py train extreme voice:narrator "ref1.wav"

# Chains (wire multiple voder oneline tasks together)
python src/voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"

# VADAR AI agent — lite (default, runs on 16GB RAM)
python src/voder.py vadar "Generate a 30-second upbeat pop song about rain, then isolate its vocals"

# VADAR AI agent — heavy/overdose (multimodal, requires 80GB+ RAM/VRAM)
python src/voder.py overdose vadar "Generate a 30-second upbeat pop song about rain, then isolate its vocals"
```

> **Run in Colab** — no installation needed: [Open in Google Colab](https://colab.research.google.com/drive/1hditIfW9JzusNcFhlHFoclCIIsNiRFNk?usp=sharing)

> **FFmpeg is required** for audio processing. Install via your system package manager. See [READ.md](docs/READ.md) for all setup details.

---

## Modes at a Glance

VODER has **8 main processing modes** — the engine's primary audio transformation pipelines. On top of these, three additional **tasks & features** layer utility workflows: voice training, side-quests, and chains. Sitting above all of them is **VADAR**, a natural-language AI agent that can call any of the modes or features on your behalf.

### Main Processing Modes (8)

| Mode | What It Does | Input | Output |
|------|-------------|-------|--------|
| **TTS** | Generate speech from text, design or clone voices; includes SLC (language conversion), dub (video/audio dubbing), and modify speech | Text / Image / URL / Audio | Audio |
| **STS** | Convert one voice to another | Audio / Video | Audio / Video |
| **TTM** | Generate, remix, repaint, bgm, and manipulate music | Text + Audio | Audio / Stems |
| **STT** | Transcribe audio, translate to 76 languages, identify speakers | Audio / Video / Image / URL | Text |
| **SE** | Denoise, dereverb, restore, super-resolution (48kHz) | Audio / Video | Audio / Video |
| **SFX** | Generate sound effects from text | Text | Audio |
| **SVS** | Isolate vocals from music | Audio / Video / URL | Audio |
| **SS** | Extract individual speakers | Audio / Video | Audio per speaker (+ MP4 with `video` flag) |

### Tasks & Features

| Feature | What It Does | Input | Output |
|---------|-------------|-------|--------|
| **train** | Train voice clones from reference audio, save as `.tts` / `.ttse` for reuse in TTS | Audio / Video / URL | `.tts` / `.ttse` voice file |
| **quest** | Side-quests — lightweight utility tasks outside the voder engine (`download`, `noframes`, `mix`, …) | URL / local video | Audio / Video file |
| **chains** | Compose user-defined pipelines of voder oneline tasks; later chains reference earlier chain names | A sequence of voder oneline commands | Final chain's output |
| **vadar** | Natural-language AI agent (lite) — describe a task in plain English, VADAR thinks, decides, and runs VODER commands. Text-only, runs on 16GB RAM | Natural-language request | Whatever the task produces |
| **overdose vadar** | Natural-language AI agent (heavy/overdose) — same as `vadar` but multimodal (look/listen/watch). Requires 80GB+ RAM/VRAM | Natural-language request + media | Whatever the task produces |

---

## Models Behind VODER

VODER orchestrates state-of-the-art open-source models — each selected for quality:

| Capability | Model |
|-----------|-------|
| Speech Recognition | [Whisper](https://github.com/openai/whisper) |
| Voice Synthesis & Cloning | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) |
| Voice Conversion | [Seed-VC](https://github.com/Plachtaa/seed-vc) |
| Music Generation | [ACE-Step](https://github.com/ace-step/ACE-Step-1.5) |
| Sound Effects | [TangoFlux](https://github.com/declare-lab/TangoFlux) |
| Sound Enhancement | [UniSE](https://github.com/alibaba/unified-audio), [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution) |
| Vocal / Music Separation | [BS-RoFormer](https://huggingface.co/pcunwa/BS-Roformer-Resurrection) |
| Advanced ASR & Diarization | [VibeVoice](https://github.com/microsoft/VibeVoice) |
| Any-to-Any Translation | [TranslateGemma 12B](https://huggingface.co/google/translategemma-12b-it) |
| Speaker Diarization | [pyannote](https://github.com/pyannote/pyannote-audio) |
| Image Text Extraction | [EasyOCR](https://github.com/JaidedAI/EasyOCR) |
| VADAR AI Agent (Heavy/Overdose) | [Gemma 4 12B abliterated uncensored](https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored) |
| VADAR AI Agent (Lite) | [SuperGemma 4 12B abliterated GGUF Q4_K_M](https://huggingface.co/Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit) |

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| CPU | 4-6 cores |
| RAM | 12 GB |
| GPU | Optional — all modes run on CPU |
| VRAM | 4 GB (6 GB recommended, 16 GB for music modes) |
| Storage | SSD recommended |

Some modes (SS, TTM overdose, ACE-Step complete) benefit from 24-32 GB VRAM or 48 GB+ system memory. See [Guide.md](docs/Guide.md) for the full per-mode breakdown.

> Speaker diarization requires a free [Hugging Face token](https://huggingface.co/settings/tokens) — set `HF_TOKEN` env var or `HF_TOKEN.txt`. See [READ.md](docs/READ.md) for details.

---

## Documentation

| Document | What's Inside |
|----------|--------------|
| [READ.md](docs/READ.md) | Mode descriptions, CLI examples, setup details, technical notes |
| [Guide.md](docs/Guide.md) | Architecture deep-dives, creative techniques, tips & tricks |
| [COMMAND_CATALOG.md](docs/COMMAND_CATALOG.md) | Complete one-liner reference for every mode, flag, and keyword |
| [Languages.md](docs/Languages.md) | Language support across all components (99+ languages) |
| [Bots.md](docs/Bots.md) | AI agent & automation usage guide |
| [FAQ.md](docs/FAQ.md) | Frequently asked questions |
| [voder-skill.md](docs/voder-skill.md) | AI skill definition for agent integration |
| [CHANGELOG.md](docs/CHANGELOG.md) | Development history |

---

## Contributing

VODER is open-source under [AGPL-3.0](LICENSE). Pull requests are not accepted — the codebase is maintained by a single developer. However, issues, bug reports, and feature suggestions are very welcome. If you want to build on top of VODER, fork it — that's what it's for.

---

Built for the open-source AI voice community.
