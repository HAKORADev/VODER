<p align="center">
  <img src="src/voder.png" alt="VODER Logo" width="256" height="256"/>
</p>

<h1 align="center">VODER — Voice Blender</h1>

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

VODER brings together **8 processing modes** under one interface — speech-to-text, text-to-speech, voice conversion, music generation, speech enhancement, sound effects, vocal separation, and speaker diarization — plus language dubbing (`tts dub`), any-to-any translation via TranslateGemma 12B, and transcribe-edit-resynthesize (built into TTS interactive). It runs entirely on your machine, needs no subscription, and works with or without a GPU.

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
- **Smart Input Pipeline** — Paste a YouTube, Bilibili, or TikTok URL directly as input. Feed an image and VODER extracts text via OCR. Automatically extract voice clips from multi-speaker audio for one-click voice cloning.

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

Isolate clean vocals from any song, or extract the instrumental. Works with audio files, videos, and direct YouTube URLs. This separation engine also runs automatically behind the scenes in TTS (to clean voice cloning references), STS (to improve conversion quality), and STT (to pre-clean audio before transcription).

### Speech-to-Text with Speaker Intelligence

Transcribe audio, video, images, or direct URLs to text. Supports **translation to English** from 99 languages, **speaker diarization** (who spoke when), and **batch processing** of multiple files. An **overdose mode** using Microsoft VibeVoice ASR delivers higher-quality transcription with built-in speaker identification.

### Sound Enhancement

Remove noise, reduce room echo, and restore clarity from degraded recordings. Upscale audio to 48kHz with AudioSR super-resolution. Works on audio and video files alike.

### Speaker Separation

Extract individual speakers from multi-speaker recordings into separate audio files, each with a speaker-labeled transcript.

### Language Conversion (TTS Sub-Task)

Translate speech from any language to English while **preserving the original speaker's voice identity** — `tts slc "audio.wav"`. Supports any-to-any translation via TranslateGemma with the `translate (source-target)` syntax. An optional `music` flag preserves the original instrumental track, and `overdose` adds a voice fidelity pass. Accepts audio files, videos, and YouTube URLs.

### Video/Audio Dubbing (TTS Sub-Task)

Dub entire videos to another language with **per-segment timing alignment** — `tts dub "video.mp4"` auto-translates to English by default. Uses VibeVoice ASR with audio events, TranslateGemma per-segment translation, Fish S2 Pro voice cloning, speed adjustment, and timeline assembly. Add `subtitle` to burn translated subtitles. Add `translate (auto-ja)` to target any language. Preserves background music.

### Smart Input Pipeline

Paste a **YouTube, Bilibili, or TikTok URL** directly as input — VODER downloads and processes it automatically. Feed an **image** containing text and VODER extracts it via OCR for TTS processing. Automatic voice clip extraction from multi-speaker audio enables one-click voice cloning for dialogue characters.

---

## Quick Start

```bash
git clone https://github.com/HAKORADev/VODER.git && cd VODER
pip install -r requirements.txt && pip install --upgrade protobuf==5.29.6

# GUI
python src/voder.py

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
python src/voder.py tts dub translate (auto-ja) "video.mp4"
python src/voder.py tts svc "speech.wav" target "voice_ref.wav"
python src/voder.py se "noisy_recording.wav"
python src/voder.py sfx sound "thunder rumbling" duration 10
```

> **Run in Colab** — no installation needed: [Open in Google Colab](https://colab.research.google.com/drive/1hditIfW9JzusNcFhlHFoclCIIsNiRFNk?usp=sharing)

> **FFmpeg is required** for audio processing. Install via your system package manager. See [READ.md](READ.md) for all setup details.

---

## Modes at a Glance

| Mode | What It Does | Input | Output |
|------|-------------|-------|--------|
| **TTS** | Generate speech from text, design or clone voices; includes SLC (language conversion), dub (video/audio dubbing), and modify speech | Text / Image / URL / Audio | Audio |
| **STS** | Convert one voice to another | Audio / Video | Audio / Video |
| **TTM** | Generate, remix, repaint, bgm, and manipulate music | Text + Audio | Audio / Stems |
| **STT** | Transcribe audio, translate to 76 languages, identify speakers | Audio / Video / Image / URL | Text |
| **SE** | Denoise, dereverb, restore, super-resolution (48kHz) | Audio / Video | Audio / Video |
| **SFX** | Generate sound effects from text | Text | Audio |
| **SVS** | Isolate vocals from music | Audio / Video / URL | Audio |
| **SS** | Extract individual speakers | Audio / Video | Audio per speaker |

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

---

## System Requirements

| Component | Minimum |
|-----------|---------|
| CPU | 4-6 cores |
| RAM | 12 GB |
| GPU | Optional — all modes run on CPU |
| VRAM | 4 GB (6 GB recommended, 16 GB for music modes) |
| Storage | SSD recommended |

Some modes (SS, TTM overdose, ACE-Step complete) benefit from 24-32 GB VRAM or 48 GB+ system memory. See [Guide.md](Guide.md) for the full per-mode breakdown.

> Speaker diarization requires a free [Hugging Face token](https://huggingface.co/settings/tokens) — set `HF_TOKEN` env var or `HF_TOKEN.txt`. See [READ.md](READ.md) for details.

---

## Documentation

| Document | What's Inside |
|----------|--------------|
| [READ.md](READ.md) | Mode descriptions, CLI examples, setup details, technical notes |
| [Guide.md](Guide.md) | Architecture deep-dives, creative techniques, tips & tricks |
| [COMMAND_CATALOG.md](COMMAND_CATALOG.md) | Complete one-liner reference for every mode, flag, and keyword |
| [Languages.md](Languages.md) | Language support across all components (99+ languages) |
| [Bots.md](Bots.md) | AI agent & automation usage guide |
| [CHANGELOG.md](CHANGELOG.md) | Development history |

---

## Contributing

VODER is open-source under [AGPL-3.0](LICENSE). Contributions are welcome — new modes, model integrations, UI improvements, bug fixes, or documentation.

---

Built for the open-source AI voice community.
