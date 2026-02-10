# CHANGELOG

- All notable changes to VODER - Voice Blender will be documented in this file.
- this project does not use Version names like v1.2.3, it just timestamped to note changes, it will always be updated everytime i notice something wrong

## 03/09/2026
- Status: Stable, All features works, under aggressive testing, still developing

### Fixed
- Seed-VC v2 unmatched tensor error which causes both STS and TTM+VC to fail, now STS works perfectly, TTM+VC will get more updates

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
