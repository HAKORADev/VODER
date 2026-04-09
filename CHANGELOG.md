# CHANGELOG

- All notable changes to VODER - Voice Blender will be documented in this file.
- This project does not use version names like v1.2.3; it just timestamps changes. It will always be updated every time I notice something wrong.

## 04/09/2026
- Status: Stable, all features work, still developing
- **Bug Hunt Activity** — Extensive bug fixes, memory optimizations, and new features

### Added

#### OCR Input Support for TTS Modes

- **OCR Parameter for TTS Mode** — New `ocr` parameter for one-liner TTS commands to extract text from images.
  - Use `ocr "path/to/image.png"` to provide an image file instead of manual text input
  - VODER uses EasyOCR to extract text from the image, then synthesizes the extracted text as speech
  - Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP
  - File validation ensures only image formats are accepted
  - Example: `python src/voder.py tts ocr "script_screenshot.png" voice "text: professional male narrator"`
  - Resources are properly cleaned up after OCR extraction (model offload, gc.collect())

- **OCR Parameter for TTS+VC Mode** — New `ocr` parameter for one-liner TTS+VC commands to extract text from images.
  - Same functionality as TTS mode but with voice cloning support
  - Extracted text is synthesized and then cloned to match the target voice reference
  - Example: `python src/voder.py tts+vc ocr "subtitle_image.jpg" target "text: speaker_clone.wav"`
  - Full resource cleanup ensures memory efficiency

- **Mimic Flag for STS Mode** — New `mimic` keyword for one-liner STS commands to enable accent and emotion conversion alongside voice timbre transfer.
  - When `mimic` is present, Seed-VC v2 uses both its AR model (accent/emotion/style) and CFM model (timbre) instead of CFM only
  - This transfers not just the voice sound but also the speaking style, tone patterns, and emotional delivery of the target voice
  - The `mimic` and `music` keywords are mutually exclusive — using both together produces an error
  - Example: `python src/voder.py sts base "source.wav" target "reference.wav" mimic`

### Fixed

- **Seed-VC v2 Inference Path and Parameters** — Fixed STS voice conversion to use the official recommended inference pipeline and parameters.
  - Switched from `convert_voice()` (legacy non-streaming path) to `convert_voice_with_streaming()` (official v2 inference path)
  - Updated CFG rates from 0.5/0.5 to 0.7/0.7 (intelligibility and similarity) to match official Seed-VC defaults
  - These rates directly control how clearly the content is preserved and how closely the output matches the reference voice
  - The streaming path also handles long audio with proper overlapping chunk processing and reference length limiting

- **STS Mode Music Flag Parsing** — Fixed argument parsing for the `music` flag in STS mode one-liner commands.
  - Music flag detection moved earlier in the parsing logic to prevent conflicts
  - Previously, the `music` keyword could be incorrectly consumed as a script parameter
  - Now correctly handled as a standalone flag alongside other keywords
  - The `music` keyword was also removed from `valid_keywords` list since it should be treated as a flag, not a keyword parameter

- **SeedVCV1 Indentation Fix** — Corrected indentation issue in SeedVCV1 voice conversion processing.
  - The tensor trimming operation `vc_target = vc_target[:, :, mel2.size(-1):]` was incorrectly indented inside a conditional block
  - Fixed to execute unconditionally after voice conversion processing
  - Ensures consistent output length across all voice conversion operations

- **Vocal Language Parameter** — Corrected vocal_language parameter initialization in voice conversion.
  - Initial change from 'en' to 'unknown' was reverted back to 'en'
  - Ensures proper language handling for English vocal content

### Optimized

- **Increased Sequence Length for Voice Conversion** — Extended max_seq_len from 4096 to 8192 in SeedVC models.
  - Updated `max_seq_len` parameter in `BaseModelArgs` class (src/modules/v2/ar.py)
  - Updated `setup_ar_caches()` call in `SeedVCV2` class
  - Allows processing of longer audio segments without sequence length truncation
  - Improves voice conversion quality for extended audio content

- **Memory Cleanup After TTM+VC Processing** — Implemented explicit memory cleanup in TTM+VC voice conversion pipeline.
  - ACE-Step model is now explicitly released after music generation with `del ace_step` and `ace_step = None`
  - Seed-VC model is now explicitly released after voice conversion with `del seed_vc` and `seed_vc = None`
  - Both stages use `gc.collect()` and `torch.cuda.empty_cache()` for proper memory reclamation
  - Reduces peak memory usage during TTM+VC operations

- **Memory Cleanup After SeedVCV1 Processing** — Implemented explicit memory cleanup after SeedVCV1 voice conversion.
  - All models in SeedVCV1 are now released after processing: whisper_model, whisper_feature_extractor, campplus_model, rmvpe, and main model
  - Uses `del` statements followed by `gc.collect()` and `torch.cuda.empty_cache()`
  - Prevents memory accumulation during batch voice conversion operations

### Changed

- **Updated VODER Logo** — Redesigned and increased logo display size in README for better visibility.
  - Logo was completely redesigned with a fresh, modern look
  - Logo size increased from 128x128 pixels to 256x256 pixels
  - Provides better visual presence on high-DPI displays

- **SE Mode Output Description** — Clarified SE mode output format in documentation.
  - Updated table to show SE mode supports both audio and video input (was unclear before)

---

## 04/08/2026
- Status: Stable, all features work, still developing
- **Major Update — Two New Modes, Script Directives, SFX Integration, and Speech Enhancement**

### Added

#### New Processing Modes

- **SE (Speech Enhancement) Mode** — A new standalone mode for audio quality improvement.
  - Uses UniSE model from [alibaba/unified-audio](https://github.com/alibaba/unified-audio) for professional speech enhancement
  - Denoising — removes background noise, hiss, and artifacts
  - Dereverberation — reduces room echo and reverb effects for cleaner speech
  - Speech restoration — enhances clarity and intelligibility of degraded recordings
  - Supports audio files (WAV, MP3, FLAC, OGG, etc.) and video files (MP4, MKV, AVI, etc.)
  - Video input: audio is automatically extracted for processing
  - Outputs at 16kHz sample rate (optimized for speech content)
  - **Not designed for musical enhancement** — use for speech-only content
  - One-liner CLI: `python voder.py se "noisy_audio.wav" result "/output/enhanced.wav"`
  - Interactive CLI: select option 7 (SE) from the menu
  - Model cached under `src/models/checkpoints/unise/`

- **SFX (Sound Effects Generation) Mode** — A new standalone mode for generating custom sound effects from text.
  - Uses TangoFlux model from [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) for text-to-audio synthesis
  - Generates any sound effect from text descriptions (nature sounds, impacts, ambient, synthesized, etc.)
  - Configurable duration: 1-30 seconds per sound effect
  - Adjustable inference steps: 1-100 (default: 30) — higher values = better quality, slower generation
  - Adjustable guidance scale: 1.0-10.0 (default: 4.5) — controls adherence to the prompt
  - 44.1kHz output quality for professional use
  - One-liner CLI format: `python voder.py sfx sound "thunder rumbling" duration 10 steps 50 guide 3.5`
  - Parameters:
    - `sound` — Text description of the sound effect (required)
    - `duration` — Duration in seconds, 1-30 (required)
    - `steps` — Inference steps for quality control (optional, default 30)
    - `guide` — Guidance scale for prompt adherence (optional, default 4.5)
    - `result` — Output file path (optional)
  - Model cached under `src/models/checkpoints/tangoflux/`

#### Dialogue System Enhancements

- **Script Directives** — Per-line control over timing, volume, and duration within dialogue scripts.
  - `/time:nn` — Position this line at `nn` seconds from the start of the output
  - `/time:nn-nn` — Position at `nn` seconds, cut `-nn` seconds from the end of the clip
  - `/time:nn+nn` — Position at `nn` seconds, cut `+nn` seconds from the start of the clip
  - `/time:nn-nn+nn` — Position at `nn` seconds, cut from both end and start
  - `/level:0-100` — Set volume level for this specific line (default: 100)
  - `/duration:1-30` — Duration for SFX lines (required when using `sfx:` character)
  - Directives are appended at the end of dialogue text, separated by spaces
  - Example: `James: Hello everyone! /time:5 /level:80`
  - Enables precise control over audio production without manual post-processing

- **SFX Character in Dialogue** — Embed sound effects directly in dialogue scripts.
  - Special character `sfx:` (case-insensitive) generates sound effects within dialogue
  - Requires `/duration:nn` directive (1-30 seconds) — mandatory for all SFX lines
  - Optional `/level:0-100` directive to control SFX volume relative to dialogue
  - SFX generation uses TangoFlux model (same as standalone SFX mode)
  - SFX clips are positioned in the timeline using `/time:` directive if specified
  - Example dialogue:
    ```
    James: Welcome to our show!
    sfx: audience applause /duration:5 /level:60
    Sarah: Thanks for having us!
    sfx: gentle ambient music /duration:15 /level:30 /time:0
    James: Let's get started with today's topic.
    ```
  - Available in both TTS and TTS+VC dialogue modes
  - Works in GUI, interactive CLI, and one-liner CLI

- **Enhanced Dialogue Assembly** — Complete rewrite of the dialogue generation pipeline.
  - New `_assemble_enhanced_dialogue()` function handles all dialogue assembly
  - Per-clip audio effects using FFmpeg (time positioning, volume control)
  - SFX generation integrated into the dialogue pipeline
  - Support for overlapping audio via time positioning
  - Automatic calculation of total duration based on positioned clips
  - Efficient temp file management with automatic cleanup

- **Cross-use Feature** — Mix generated and cloned voices in the same dialogue.
  - Both TTS and TTS+VC one-line modes now support combining `voice` and `target` parameters
  - Use `voice "Character: prompt"` for generated voices (Voice Design)
  - Use `target "Character: path"` for cloned voices (Voice Cloning)
  - Example: `python voder.py tts script "James: Hello" "Sarah: Hi" voice "James: male" target "Sarah: /path/to/sarah.wav"`
  - Enables hybrid dialogues where some characters use generated voices and others use cloned references
  - A character cannot have both `voice` and `target` — each character must use one or the other

- **Music Volume Level Control** — Fine-grained control over background music volume.
  - New `level` parameter for one-liner dialogue commands
  - Supports constant volume, time-based segments, and fade transitions
  - Format options:
    - `"volume"` — Constant volume percentage (e.g., `"35"` for 35%)
    - `"start:vol-end:vol"` — Different volumes at start and end times
    - `"start:from-to+fade"` — Fade from one volume to another over specified duration
  - Default remains 35% if `level` parameter is not specified
  - Example: `python voder.py tts script "James: Hello" voice "James: male" music "piano" level "0:30-60:50"`
  - Time-based segments allow dynamic music volume throughout the dialogue
  - FFmpeg volume filter with evaluate-per-frame for smooth transitions

#### TTM Mode Enhancement

- **Instrumental Music Generation** — TTM mode now produces music-only (no vocals) output.
  - Use empty lyrics `"..."` to generate instrumental music without any vocal content
  - The model produces music matching the style prompt without singing
  - Ideal for background music, ambient tracks, and instrumental compositions
  - Example: `python voder.py ttm lyrics "..." styling "cinematic orchestral" duration 60`
  - Works with TTM+VC mode as well — voice conversion will have no effect on instrumental output
  - Lyrics in `()` or `[]` brackets provide context without being sung (for music-with-vocals generation)

#### Auto-Clone Feature Enhancement

- **TTS+VC Dialogue + Auto-Clone Trick** — Useful behavior when using the same file for both dialogue source and auto-clone.
  - Dialogue source analysis generates character names as `1`, `2`, `3`... based on speaker detection order
  - Auto-clone voice extraction produces voice references labeled `speaker 1`, `speaker 2`, etc.
  - The system matches character names to voice references **alphabetically**
  - **Result:** Using the same input file for both dialogue source and auto-clone produces an exact replica of the original audio
  - This is useful for:
    - Testing the TTS+VC pipeline accuracy
    - Verifying speaker detection quality
    - Creating backup/restoration of audio content
    - Demonstrating the voice cloning system's capabilities

### Changed

- **Expanded Processing Modes** — VODER now has 9 processing modes (up from 7).
  - Modes: STT+TTS, TTS, TTS+VC, STS, TTM, TTM+VC, STT, SE, SFX
  - GUI dropdown updated with new mode options
  - Interactive CLI menu updated with options 7 (SE) and 8 (SFX)

- **Centralized Model Storage** — New model directories added for UniSE and TangoFlux.
  - `src/models/checkpoints/unise/` — UniSE speech enhancement model
  - `src/models/checkpoints/tangoflux/` — TangoFlux sound effects model
  - Directories auto-created at startup

- **Updated Dependencies**:
  - `transformers==4.57.3` (pinned version, was `>=4.30.0`)
  - New: `einx`, `x-transformers==2.3.1`, `safetensors`, `soxr`, `tqdm`, `packaging` — required for UniSE model
  - These enable speech enhancement model loading and inference

### Technical Notes

- Code size increased from ~6650 lines (voder.py bleed v1) to ~7100+ lines — approximately 7% increase
- New imports: `traceback` for enhanced error handling in new modes
- UniSE model loaded from `src/unise/` module with `UniSEEnhancer` class
- TangoFlux model loaded from `src/tangoflux/` module with `TangoFluxGenerator` class
- All new models (UniSE, TangoFlux) are immediately offloaded after use to prevent memory accumulation
- Background music generation now supports volume level specifications with time-based control
- FFmpeg filter expressions built dynamically for complex volume automation

---

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
