# CHANGELOG

- All notable changes to VODER - Voice Blender will be documented in this file.
- This project does not use version names like v1.2.3; it just timestamps changes. It will always be updated every time I notice something wrong.

## 04/03/2026
- Status: Stable, all features work, still developing

### Added
- **Background Music Chunking for Long Dialogues** — Enhanced background music generation to handle dialogues longer than 250 seconds by generating multiple music chunks and concatenating them.
  - When background music is enabled and required duration exceeds 250 seconds, the system now generates multiple consecutive music chunks (250s each) using the same music description
  - All chunks are concatenated into a single music file using FFmpeg concat demuxer before mixing with dialogue
  - This ensures uninterrupted background music throughout the entire dialogue instead of silence after the first chunk
  - Maximum chunk size set to 250 seconds for optimal performance and compatibility with ACE-Step model limits

### Fixed
- **Music Generation Minimum Duration** — Fixed VODER's minimum music duration to match ACE-Step model requirements.
  - ACE-Step model requires a minimum of 10 seconds for generation, but VODER previously allowed inputs as low as 5 seconds
  - VODER now enforces 10-second minimum for music generation to prevent generation failures
  - This applies to both GUI and CLI modes

## 04/02/2026
- Status: Stable, all features work, still developing

### Added
- **STT (Speech-to-Text) Standalone Mode** — A new dedicated `stt` mode available via one-liner CLI.
  - Transcribe audio, video, image, or YouTube URL to text using Whisper
  - Supports `timestamp` flag to include word-level timestamps in format `[mm:ss:mss-mm:ss:mss]`
  - Supports `dialogue` flag to enable speaker diarization (pyannote) with numbered speaker labels
  - Batch processing: multiple files processed sequentially in a single command
  - Supports audio files, video files (auto-extracts audio), image files (OCR via EasyOCR), and YouTube/Bilibili/TikTok URLs
  - Output saved as `.txt` files with descriptive naming (e.g., `voder_stt_timestamp_dialogue_...`, `voder_stt_ocr_...`, `voder_youtube_...`)
  - One-liner examples:
    ```
    python voder.py stt "audio.wav"
    python voder.py stt "audio.wav" timestamp
    python voder.py stt "audio.wav" dialogue
    python voder.py stt "audio.wav" timestamp dialogue
    python voder.py stt "audio1.wav" "audio2.wav"
    python voder.py stt "https://youtube.com/watch?v=..."
    ```

- **Speaker Diarization (Pyannote Integration)** — New `SpeakerDiarization` class for identifying and separating speakers in audio.
  - Uses `pyannote/speaker-diarization-community-1` model (requires HuggingFace token with accepted model conditions)
  - Word-level alignment between Whisper transcription timestamps and pyannote speaker turns
  - Three-tier speaker assignment: fully-contained words get priority, then best overlap, then nearest neighbor
  - Post-processing: assigns words without speakers to nearest speaker; merges very short utterances (<0.5s) with neighbors
  - Available in STT mode (`dialogue` flag), dialogue source analysis, and voice clip extraction
  - Local pyannote integration bundled in `src/libs/pyannote/` to avoid huggingface_hub version conflicts
  - Requires HF_TOKEN.txt with access to `pyannote/speaker-diarization-community-1`

- **Image Text Extraction (EasyOCR)** — New `EasyOCRReader` class for extracting text from images.
  - Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.webp`
  - CPU-based OCR (no GPU required)
  - Can be used as input source for STT mode and dialogue source analysis
  - Images always produce single-speaker output (no diarization possible)
  - Model cached under centralized `src/models/checkpoints/easyocr/`

- **YouTube & Video Platform Download (yt-dlp)** — New download integration for audio extraction from URLs.
  - Supports YouTube, YouTube Shorts, youtu.be, Bilibili, and TikTok URLs
  - Downloads best available audio via yt-dlp, converts to MP3 (192kbps)
  - Robust error handling for invalid URLs, unavailable videos, network errors
  - Falls back to `.m4a`, `.wav`, `.webm` if `.mp3` not created
  - Temporary files cleaned up after processing
  - Works as input for STT mode, dialogue source analysis, and voice clip extraction

- **Dialogue Source from Files & URLs** — Comprehensive multi-format dialogue source support.
  - New `validate_dialogue_source_file()` accepts: text files, audio/video files, image files, YouTube URLs
  - New `analyze_dialogue_source()` full analysis pipeline:
    - Text files: parsed with auto-formatting (auto-detects dialogue vs single mode, normalizes format)
    - Audio/Video files: Whisper transcription + optional pyannote diarization
    - Image files: EasyOCR text extraction (single speaker)
    - YouTube URLs: download + Whisper transcription + optional diarization
  - Speaker label mapping: original pyannote labels → numbered speakers (1, 2, 3...)
  - Smart speaker switching: only switches on significant gaps (0.3s) or after 3+ words to avoid rapid switching artifacts
  - Available in interactive CLI for both TTS and TTS+VC modes

- **Automatic Voice Clip Extraction from Multi-Speaker Audio** — Extract individual speaker voice clips automatically.
  - New `extract_voice_clips_from_multispeaker()` function
  - Given a multi-speaker audio source (file or YouTube URL), extracts the longest voice clip per speaker
  - Uses Whisper (word timestamps) + Pyannote (speaker diarization) for speaker identification
  - Extracts clips via FFmpeg with precise timing
  - Integrated with TTS+VC interactive CLI: after entering dialogue, user can provide a multi-speaker source and clips are auto-assigned to characters alphabetically
  - Falls back to manual entry if not enough clips extracted

- **`result` Parameter** — New universal CLI parameter to copy the latest result to a specified path.
  - Works with all one-liner modes
  - Example: `python voder.py tts script "Hello" voice "male" result "/path/to/output.txt"`
  - Creates directories as needed, preserves file metadata

### Changed
- **Centralized Model Management System** — Complete overhaul of model storage and caching.
  - All models now stored under `src/models/` with organized directory structure:
    - `models/tmp/` — temporary downloads in progress
    - `models/checkpoints/qwen_tts_voicedesign/` — Qwen3-TTS VoiceDesign
    - `models/checkpoints/qwen_tts_base/` — Qwen3-TTS Base
    - `models/checkpoints/acestep/` — ACE-Step models
    - `models/checkpoints/seed_vc_v1/` — Seed-VC v1
    - `models/checkpoints/seed_vc_v2/` — Seed-VC v2
    - `models/checkpoints/whisper/` — Whisper model
  - HuggingFace environment variables redirected to local directories (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, HUGGINGFACE_HUB_CACHE)
  - Whisper cache (XDG_CACHE_HOME) redirected for openai-whisper downloads
  - All directories auto-created at startup
  - All model classes (WhisperSTT, QwenTTSVoiceDesign, QwenTTS, SeedVCV2, SeedVCV1) use centralized paths

- **Enhanced HF_TOKEN Handling** — Improved token discovery and error messaging.
  - Multi-location fallback search: CWD, src directory, `1/` directory
  - Auto-creates template `HF_TOKEN.txt` with instructions if file doesn't exist
  - More informative warning messages with links to HuggingFace token settings and pyannote model conditions
  - Token set as `os.environ["HF_TOKEN"]` for all HuggingFace operations

- **Updated Dependencies**:
  - `torch==2.8.0`, `torchaudio==2.8.0`, `torchvision==0.23.0` (was >=2.0.0)
  - New: `torch-audiomentations>=0.12.0`, `hydra-core==1.3.2`, `opentelemetry-proto==1.0.0`, `opentelemetry-exporter-otlp-proto-grpc==1.0.0`, `sox`, `onnxruntime`, `lightning==2.4`, `yt-dlp>=2026.3.17`, `easyocr>=1.7.0`
  - `qwen-tts` removed from pip requirements — now bundled locally in `src/qwen_tts/`
  - Pyannote bundled locally in `src/libs/pyannote/` (requirements2.txt includes `pyannote.database`, `pyannote.metrics`, `pyannote.pipeline`, `asteroid-filterbanks`)

### Technical Notes
- Code size increased from ~4880 lines (voder.py stable) to ~6650 lines (voder.py bleed) — a 36% increase
- voder2.py is a slightly stripped version of voder.py (same functionality, fewer comments/docstrings)
- All new models (EasyOCR, Pyannote) are immediately offloaded after use to prevent memory accumulation
- The `load_custom_model_from_hf()` function now accepts optional `target_dir` parameter for flexible download locations

## 02/24/2026
- Status: Stable, all features work, still developing, there will be a major update!

### Added
- **MSTS (Music-STS) in STS mode** – STS now supports musical inputs via the Seed-VC v1 model (44.1kHz) for better music voice conversion quality.
  - **GUI**: When pressing Generate in STS mode, a dialog appears asking "musical inputs?" with Yes/No buttons. Yes uses v1 model at 44.1kHz; No uses standard v2 at 22.05kHz.
  - **Interactive CLI**: After entering base and target paths, user is prompted "Are the inputs musical? (Y/N):". Y uses v1 model; N uses standard v2.
  - **One-line CLI**: New `music` keyword parameter: `voder.py sts path/base path/target music`. Invalid parameters show error message.
  - **Output naming**: MSTS outputs prefixed with `voder_m_sts_timestamp.wav`; standard STS uses `voder_sts_timestamp.wav`.

### Fixed
- **TTS+VC dialogue voice cloning stability** – Voice characteristics are now extracted once per character instead of re-extracting for each line.
  - In dialogue with multiple lines per character (e.g., 5 lines for "James"), the voice prompt is extracted once and reused for all lines of that character.
  - This ensures consistent voice quality throughout the dialogue, eliminating variations that occurred when re-extracting voice for each line.
  - Applies to GUI, interactive CLI, and one-line CLI modes.

### Optimized
- **Memory offloading after processing** – Models are now explicitly unloaded from memory/VRAM after each operation completes.
  - In GUI mode: ProcessingThread now calls cleanup() after finishing, releasing all loaded models.
  - In interactive CLI mode: Each mode (TTS, TTS+VC, STS, STT+TTS, TTM, TTM+VC) now offloads models before returning.
  - This prevents memory accumulation when performing multiple operations in a single session.
  - Pattern applied: `del model`, `gc.collect()`, `torch.cuda.empty_cache()`.

## 02/12/2026
- Status: Stable, all features work, under aggressive testing, still developing

### Added
- **Full dialogue support in CLI** – Both interactive and one‑liner modes now support multi‑speaker scripts.
  - Interactive CLI: enter multiple lines with `Character: text` format; VODER automatically prompts for voice prompts (TTS) or audio file paths (TTS+VC) per character.
  - One‑liner: repeated `script` and `voice`/`target` parameters allow dialogue generation in a single command.
- **Optional background music for dialogue scripts** – Available in TTS and TTS+VC modes when the script contains at least one `Character: text` line.
  - **GUI**: Clean modal dialog appears before generation, asking for a music description. OK with non‑empty description triggers music; Skip bypasses.
  - **Interactive CLI**: After voice prompts/assignments, user is asked `Add background music? (y/N):`. Enter `y`/`yes` to provide a description; empty input skips.
  - **One‑liner CLI**: New `music "description"` parameter. If present with non‑empty value, background music is generated; `music ""` is ignored. Parameter is ignored in single mode (no colon in scripts).
  - **Automatic duration fitting**: Music length matches the exact duration of the concatenated dialogue (via `torchaudio.info`).
  - **Volume control**: Music is mixed at 35% relative volume using FFmpeg (`volume=0.35`), empirically chosen for non‑intrusive ambience.
  - **Memory management**: ACE‑Step model is explicitly released and GPU cache cleared after music generation, minimising VRAM footprint.
  - **Cleanup**: Temporary dialogue and music files are deleted; only the final mixed file remains in `results/` with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`).

### Updated
- **Row‑based dialogue editor in GUI** – Replaced free‑text script box with per‑row Character/Dialogue fields.
  - New rows auto-add when the last row is filled; first row has no delete button, subsequent rows can be deleted.
  - Voice prompt area dynamically shows each character with a text field (TTS) or audio‑number dropdown (TTS+VC).
  - Audio reference files are numbered; dropdowns update automatically when files are added/removed.

### Fixed
- **Memory optimisation for TTM+VC** – ACE‑Step model is now explicitly released and GPU cache cleared before loading Seed‑VC. Reduces peak VRAM usage and improves reliability on 8GB cards.

## 02/10/2026
- Status: Stable, all features work, under aggressive testing, still developing

### Fixed
- Seed-VC v2 unmatched tensor error which caused both STS and TTM+VC to fail. Now STS works perfectly; TTM+VC will receive further optimisations.

## 02/09/2026
- Status: unstable, untested, under development

**Initial Release - Unstable Development Build**

First public release of VODER. This is an early development version with core functionality but may contain bugs and instability issues.

### Added
- Initial GUI application with PyQt5
- Six processing modes: STT+TTS, TTS, TTS+VC, STS, TTM, TTM+VC
- Whisper integration for speech-to-text transcription
- Qwen3-TTS integration for text-to-speech synthesis
- Seed-VC v2 integration for voice conversion
- ACE-Step integration for text-to-music generation
- Interactive CLI mode
- One-line command support
