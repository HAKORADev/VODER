# CHANGELOG

- All notable changes to VODER - Voice Blender will be documented in this file.
- This project does not use version names like v1.2.3; it just timestamps changes. It will always be updated every time I notice something wrong.

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
  - New rows auto‑add when the last row is filled; first row has no delete button, subsequent rows can be deleted.
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
