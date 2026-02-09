# VODER Technical Guide

## Table of Contents

- [Introduction](#introduction)
- [Core Processing Concepts](#core-processing-concepts)
  - [FFmpeg Integration](#ffmpeg-integration)
  - [Audio Sampling and Resampling](#audio-sampling-and-resampling)
  - [Input Format Handling](#input-format-handling)
- [Processing Modes Deep Dive](#processing-modes-deep-dive)
  - [STT+TTS Mode](#stttts-mode)
  - [TTS Mode](#tts-mode)
  - [TTS+VC Mode](#ttsvc-mode)
  - [STS Mode](#sts-mode)
  - [TTM Mode](#ttm-mode)
  - [TTM+VC Mode](#ttmvc-mode)
- [Dialogue System](#dialogue-system)
  - [Dialogue Format](#dialogue-format)
  - [Voice Prompt Configuration](#voice-prompt-configuration)
  - [Character Routing](#character-routing)
  - [Processing Pipeline](#processing-pipeline)
- [Mode Compatibility and Limitations](#mode-compatibility-and-limitations)
  - [Working Modes](#working-modes)
  - [Problematic Modes](#problematic-modes)
  - [GPU Requirements](#gpu-requirements)
- [Output and File Management](#output-and-file-management)
- [Troubleshooting](#troubleshooting)

---

## Introduction

This guide provides an in-depth explanation of VODER's internal workings, technical decisions, and implementation details. Unlike model documentation (which covers the AI systems themselves), this document focuses on how VODER orchestrates these models, handles audio data, and implements the six processing modes.

Understanding these details helps users troubleshoot issues, understand why certain limitations exist, and make informed decisions about their workflows. The guide also explains design choices that may seem arbitrary but were made for specific technical reasons.

---

## Core Processing Concepts

### FFmpeg Integration

FFmpeg is a mandatory dependency for VODER. It is not optional and must be installed separately from Python packages.

**Why FFmpeg is Required:**

FFmpeg handles several critical operations that VODER cannot perform without it:

1. **Video Audio Extraction**: When users provide video files (MP4, AVI, MOV, MKV), VODER uses FFmpeg to extract only the audio track. The video content itself is never processed — only the audio is used for voice synthesis, conversion, or transformation. This is by design: VODER is a voice processing tool, not a video editor.

2. **Audio Concatenation**: The dialogue system generates multiple audio segments (one per character line) and uses FFmpeg to concatenate them into a single output file. This ensures smooth transitions between segments and proper format consistency.

3. **Format Conversion**: FFmpeg handles format conversions between different audio codecs, sample rates, and channel configurations that may arise during processing.

**Installation:**

FFmpeg must be installed separately as it is a system application, not a Python package:

- **Windows**: `winget install FFmpeg` or download from gyan.dev and add to PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `sudo apt install ffmpeg`

Without FFmpeg in the system PATH, VODER will fail when attempting video processing or dialogue concatenation operations.

### Audio Sampling and Resampling

**Sample Rate Fundamentals:**

Audio sample rate refers to how many samples of audio are captured per second, measured in Hertz (Hz). Common sample rates include:

- 22050 Hz (22.05 kHz) — Used by Seed-VC
- 44100 Hz (44.1 kHz) — CD quality, VODER's output standard
- 48000 Hz (48 kHz) — Professional video standard

**Why Resampling is Necessary:**

Seed-VC, the voice conversion model used in STS and TTM+VC modes, is specifically designed to work with audio at 22050 Hz. This is a hard requirement of the model's architecture — it cannot process audio at other sample rates.

**VODER's Resampling Pipeline:**

For modes using Seed-VC (STS and TTM+VC), VODER implements a three-stage resampling pipeline:

1. **Input Resampling (to 22050 Hz)**: When users provide audio at 44100 Hz, 48000 Hz, or any other rate, VODER uses FFmpeg to convert it to 22050 Hz before passing it to Seed-VC. This conversion is automatic and transparent to the user.

2. **Model Processing**: Seed-VC processes the 22050 Hz audio according to the selected operation (voice conversion).

3. **Output Resampling (to 44100 Hz)**: The output from Seed-VC is at 22050 Hz. VODER upsample it to 44100 Hz using FFmpeg before saving the final output. This higher sample rate produces better quality output files that are compatible with standard audio players.

**Technical Implementation:**

The resampling is handled by PyQt5's torchaudio library, which provides high-quality resampling algorithms. The code uses torchaudio.transforms.Resample to perform the conversion:

```python
# Example resampling logic
resampler = torchaudio.transforms.Resample(original_sr, 22050)
waveform_resampled = resampler(waveform_original)
```

This ensures that regardless of the input format, Seed-VC always receives audio at its required 22050 Hz sample rate, and users always receive output at the standard 44100 Hz.

### Input Format Handling

**Supported Audio Formats:**

VODER accepts various audio formats through PyQt5's torchaudio backend:

- WAV (uncompressed PCM)
- MP3 (compressed)
- FLAC (lossless compressed)
- Other formats supported by FFmpeg

**Video Format Handling:**

When users provide video files (MP4, AVI, MOV, MKV), VODER's handling is straightforward:

1. FFmpeg extracts the audio track
2. The video content is discarded
3. Only the extracted audio proceeds to processing

This design choice means users can provide video files containing the voice they want to process without needing to extract audio first. The video itself is never analyzed or transformed.

**Sample Rate Normalization:**

All input audio is normalized to a consistent internal format before processing. This ensures predictable behavior regardless of the source file's original characteristics.

---

## Processing Modes Deep Dive

### STT+TTS Mode

**What It Does:**

STT+TTS (Speech-to-Text + Text-to-Speech) transcribes audio content using Whisper, allows users to edit the transcribed text, then synthesizes the edited text with a target voice.

**Why This Mode Exists:**

This mode enables voice modification — making a person say things they never actually said. By providing the same audio file as both base (content) and target (voice), users can:

- Change words, phrases, or entire speeches
- Fix transcription errors automatically
- Localize content into different languages
- Create fictional dialogue from real voice samples

**Processing Pipeline:**

1. **Transcription**: Whisper transcribes the base audio to text with word timestamps
2. **Review**: User sees the transcribed text and can edit it
3. **Voice Extraction**: Voice characteristics are extracted from target audio
4. **Synthesis**: The edited text is synthesized using the target voice

**Why It Is CLI-Only:**

STT+TTS requires user interaction to review and edit the transcribed text. The one-liner mode cannot accommodate this interactive workflow. Users must either:

- Use the interactive CLI: `python voder.py cli` and select option 1
- Use the GUI for full visual feedback

**Multi-Voice Warning:**

If the base audio contains multiple speakers, Whisper will transcribe all of them. The synthesis will still use a single target voice for the entire text. This creates an unnatural result where multiple speakers sound like the same person. For true multi-voice dialogue, use the dialogue system.

### TTS Mode

**What It Does:**

TTS (Text-to-Speech) generates speech from text using Qwen3-TTS VoiceDesign. Users provide a text script and a voice prompt describing the desired voice characteristics.

**Voice Design Prompts:**

The voice prompt is a text description that Qwen3-TTS interprets to generate appropriate voice characteristics. Effective prompts describe:

- Gender and age (male, female, child)
- Tone and emotion (cheerful, serious, mysterious)
- Speaking style (fast, slow, dramatic)
- Accent or dialect (if applicable)

**Example Prompts:**

- "Warm adult female voice with gentle tone"
- "Energetic young male narrator"
- "Professional news anchor voice"

**Processing Simplicity:**

TTS mode is the simplest mode in VODER:

1. Receive text and voice prompt
2. Pass to Qwen3-TTS VoiceDesign
3. Generate and save audio

**No Voice Reference Required:**

Unlike TTS+VC, this mode does not require a reference audio file. The voice is generated entirely from the prompt description.

### TTS+VC Mode

**What It Does:**

TTS+VC (Text-to-Speech + Voice Cloning) generates speech from text then applies voice cloning to match a reference voice. This uses Qwen3-TTS Base (not Seed-VC).

**Processing Pipeline:**

1. **Synthesis**: Qwen3-TTS Base generates speech from the text
2. **Voice Extraction**: Extract voice characteristics from reference audio
3. **Cloning**: Apply cloned voice characteristics to the synthesized speech

**Why Qwen3-TTS Base:**

Qwen3-TTS Base includes built-in voice cloning capabilities. It can learn a voice from a reference audio sample and apply it to generated speech. This is different from Seed-VC, which performs voice conversion on existing audio.

**Single vs Dialogue Mode:**

For single-voice scripts:

- User provides one reference audio file
- The entire script uses that voice

For dialogue scripts:

- User provides multiple reference audio files (numbered)
- Each character in the script is routed to a different reference
- The dialogue system handles routing and concatenation

**CPU Compatibility:**

TTS+VC works on CPU without GPU. This makes it accessible for users without NVIDIA graphics hardware.

### STS Mode

**What It Does:**

STS (Speech-to-Speech) performs voice conversion — transforming source audio to sound like a target voice while preserving the original content, emotion, and prosody.

**Processing Pipeline:**

1. **Load Audio**: Load base (source) and target (reference) audio files
2. **Resample to 22050 Hz**: Prepare both for Seed-VC processing
3. **Voice Conversion**: Seed-VC transforms source to target voice
4. **Resample to 44100 Hz**: Prepare output for standard playback
5. **Save Result**: Export converted audio

**Seed-VC Dependency:**

STS relies entirely on Seed-VC v2 for the conversion process. All resampling and processing is designed around Seed-VC's requirements.

**Limitations:**

Seed-VC v2 may have compatibility issues with certain audio types:

- Very short audio clips (insufficient content for analysis)
- Very long audio clips (memory constraints)
- Audio with unusual characteristics (extreme dynamics, artifacts)

**GPU Requirement:**

STS requires NVIDIA GPU with minimum 8GB VRAM. Seed-VC cannot run on CPU or non-NVIDIA graphics cards.

### TTM Mode

**What It Does:**

TTM (Text-to-Music) generates music from lyrics and a style prompt using ACE-Step. Users provide song lyrics, a description of the desired musical style, and a duration.

**Lyrics Format:**

Lyrics should be formatted with section markers:

```
Verse 1:
Walking down the empty street
Feeling the rhythm in my feet

Chorus:
This is our moment, this is our time
Everything's gonna be just fine
```

**Style Prompts:**

The style prompt describes the musical characteristics:

- Genre (pop, rock, jazz, electronic)
- Mood (happy, melancholic, energetic)
- Instrumentation (piano, synthesizer, full band)
- Tempo (upbeat, slow, moderate)

**Duration Limits:**

The 5-minute (300 seconds) limit is a technical safeguard, not a validated optimum:

- Shorter durations (10-60 seconds) are more reliable
- Maximum duration may cause crashes depending on system resources
- The limit exists to prevent runaway generation that could consume all available memory

**Processing Steps:**

1. Parse lyrics and style prompt
2. Initialize ACE-Step with configuration
3. Generate music with specified parameters
4. Save output audio

**CPU Compatibility:**

TTM works on CPU without GPU. Processing will be slower but functional.

### TTM+VC Mode

**What It Does:**

TTM+VC (Text-to-Music + Voice Conversion) generates music then applies voice conversion to change the vocalist's voice.

**Processing Pipeline:**

TTM+VC is essentially TTM followed by STS:

1. **TTM Stage**: Generate music using ACE-Step (same as TTM mode)
2. **Resample to 22050 Hz**: Prepare TTM output for Seed-VC
3. **STS Stage**: Apply voice conversion using Seed-VC
4. **Resample to 44100 Hz**: Prepare final output
5. **Save Result**: Export processed music

**Composite Mode:**

TTM+VC chains two operations together. The generated music (TTM output) becomes the "base" audio for voice conversion.

**Crash Risk:**

This mode has elevated crash risk due to:

- TTM output variability (different audio characteristics)
- Seed-VC sensitivity to input quality
- Potential mismatches between TTM output sample rate and Seed-VC requirements

The error message "unmatched data" may appear when Seed-VC cannot process the TTM output successfully.

**GPU Requirement:**

TTM+VC requires NVIDIA GPU with minimum 8GB VRAM due to Seed-VC dependency.

---

## Dialogue System

### Dialogue Format

**Script Format:**

Dialogue scripts use a numbered character format:

```
1:James: "Welcome to our podcast! Today we'll discuss AI."
2:Sarah: "Thanks James! I'm excited to share my research."
3:James: "Let's start with the basics. What is AI?"
```

**Format Rules:**

- Each line starts with a number (sequence)
- Followed by colon
- Character name
- Colon
- Dialogue in quotation marks

**Sequencing:**

Dialogue lines must be numbered sequentially starting from 1:

- Valid: 1, 2, 3, 4, 5
- Invalid: 1, 4, 2, 3 (out of order)

The sequence determines processing order and is preserved in the final output.

**Character Names:**

Character names must match between the script and voice prompt configuration. Names are case-insensitive (James, james, JAMES all work) but must be spelled identically.

### Voice Prompt Configuration

**Format:**

Voice prompts map character names to audio file references:

```
James:1
Sarah:2
Narrator:3
```

**Audio File References:**

The number references the audio file shown in the GUI:

- "1" references the first loaded audio file
- "2" references the second loaded audio file
- Files are numbered sequentially as loaded in the GUI

**Example:**

If the GUI shows:

```
41.wav - James voice sample
42.wav - Sarah voice sample
43.wav - Narrator voice sample
```

The voice prompts would be:

```
James:41
Sarah:42
Narrator:43
```

Users can click on the audio file in the GUI to hear what voice it contains before routing it to a character.

### Character Routing

**How Routing Works:**

The dialogue system routes each script line to its corresponding voice reference:

1. Parse dialogue script to extract (number, character, text) tuples
2. For each line, look up the character name in voice prompts
3. Get the audio file number from the prompt mapping
4. Generate that line using the specified voice reference
5. Track which voice reference each line used

**Single Voice Simplicity:**

If only one audio file is loaded, all characters use that voice regardless of script configuration.

**Dialogue Mode Requirement:**

Dialogue mode requires at least two audio files. Single-voice dialogue uses the standard TTS+VC flow without character routing.

### Processing Pipeline

**Stage-by-Stage:**

The dialogue processing pipeline is fully automated:

1. **Parse Script**: Extract dialogue items with sequence, character, and text
2. **Parse Voice Prompts**: Build character-to-audio mapping
3. **Validate**: Ensure all characters have voice references
4. **Temporary Files**: Create temporary directory for segment audio files
5. **Iterate Lines**: For each dialogue line:
   - Load corresponding voice reference
   - Extract voice characteristics
   - Synthesize the line
   - Save to temporary file
6. **Concatenate**: Use FFmpeg to combine all segments into one file
7. **Clean Up**: Remove temporary files
8. **Export**: Save final dialogue to results folder

**No AI-to-AI Conversation:**

Despite the appearance, dialogue mode does not involve AI systems conversing with each other. Each line is synthesized independently using the specified voice reference. The "conversation" effect is achieved through:

- Sequential processing of script lines
- Voice routing that matches characters to their samples
- FFmpeg concatenation that preserves timing

This automation makes it seem like the AIs are talking to each other, but it is actually just sequential synthesis with voice routing.

---

## Mode Compatibility and Limitations

### Working Modes

The following modes have been validated and work correctly:

**STT+TTS:**
- Whisper transcription is reliable
- Qwen3-TTS synthesis produces consistent results
- Voice cloning in the synthesis stage works as expected
- User interaction for text editing enables voice modification

**TTS:**
- Qwen3-TTS VoiceDesign generates consistent voices
- Voice prompts are interpreted correctly
- No external dependencies beyond the model
- Works reliably on both GPU and CPU

**TTM (Generally):**
- ACE-Step generates music from lyrics
- Style prompts influence output characteristics
- Shorter durations (under 2 minutes) are most reliable
- May be inconsistent with very complex lyrics

**TTS+VC:**
- Qwen3-TTS Base handles voice cloning correctly
- Single-voice mode is fully validated
- Voice extraction from reference audio works consistently
- CPU operation is supported

### Problematic Modes

The following modes may have issues due to Seed-VC v2 compatibility:

**STS:**
- May fail with certain audio types
- Requires specific audio characteristics
- Can fail silently or produce poor results
- GPU memory constraints affect reliability

**TTM+VC:**
- Highest failure risk due to multi-stage pipeline
- TTM output variability affects Seed-VC input
- "Unmatched data" errors may occur
- Memory-intensive due to multiple model loads

**General Seed-VC Issues:**

Seed-VC v2 may not work correctly with:

- Very short audio clips (under 2 seconds)
- Very long audio clips (over 5 minutes)
- Audio with significant background noise
- Audio with unusual sample rates or formats
- Audio with extreme dynamics (very quiet to very loud)

### GPU Requirements

**Modes Requiring GPU:**

The following modes require NVIDIA GPU with minimum 8GB VRAM:

- STS (Speech-to-Speech Voice Conversion)
- TTM+VC (Text-to-Music + Voice Conversion)

Seed-VC cannot operate on:

- CPU-only systems
- AMD GPUs
- Intel integrated graphics
- Apple Silicon (no CUDA support)

**Modes Working Without GPU:**

The following modes work on CPU:

- TTS (Text-to-Speech)
- TTS+VC (Text-to-Speech + Voice Cloning)
- TTM (Text-to-Music)

Processing will be significantly slower without GPU acceleration.

---

## Output and File Management

**Results Directory:**

All output files are automatically saved to the `results/` folder located next to `voder.py`:

```
VODER/
  voder.py
  results/
    voder_tts_20260210_120000.wav
    voder_sts_20260210_120100.wav
    voder_tts_vc_dialogue_20260210_120200.wav
    ...
```

**Automatic Naming:**

Output files are named with:

- Mode identifier (tts, sts, ttm, etc.)
- Timestamp (YYYYMMDD_HHMMSS)
- Appropriate file extension (.wav)

**No Manual Export Required:**

Users do not need to specify output paths. The tool automatically exports results to the results folder with unique filenames based on timestamp.

---

## Troubleshooting

### Getting Help

**Preferred Contact Method:**

For issues, bugs, or feature requests, Direct Message on X (Twitter) is preferred over GitHub issues:

- **X**: [@HAKORAdev](https://x.com/HAKORAdev)

DMing allows for:

- Direct communication with the developer
- Faster response times
- Discussion of issues not suitable for public tracking
- Collaborative problem-solving

It is like having a genie that will work on your issues and tell you when to check again.

### Common Issues

**STS/TTM+VC Fails Immediately:**

- **Cause**: No NVIDIA GPU detected or insufficient VRAM
- **Solution**: Verify GPU installation and VRAM availability

**Unmatched Data Error (TTM+VC):**

- **Cause**: Seed-VC cannot process TTM output
- **Solution**: Try shorter duration, simpler style prompt, or different lyrics

**Dialogue Character Not Found:**

- **Cause**: Character name mismatch between script and voice prompts
- **Solution**: Verify names match exactly (case-insensitive but spelling must match)

**Out of Memory:**

- **Cause**: Model too large for available memory
- **Solution**: Process shorter audio, reduce TTM duration, close other applications

**FFmpeg Not Found:**

- **Cause**: FFmpeg not installed or not in system PATH
- **Solution**: Install FFmpeg separately (see Installation section)

**Quality Issues with TTM:**

- **Cause**: Complex lyrics or ambitious style prompts
- **Solution**: Simplify lyrics structure, use more conventional style descriptions

### Mode-Specific Tips

**STT+TTS:**
- Use clear audio for best transcription
- Review transcribed text carefully before synthesis
- Same audio as base and target enables voice modification

**TTS:**
- Experiment with voice prompts for different characteristics
- Shorter scripts process faster

**TTS+VC:**
- High-quality reference audio produces better clones
- 10-30 seconds of reference audio is ideal

**STS:**
- Base and target should have similar audio characteristics
- Avoid very short or very long source files

**TTM:**
- Start with shorter durations (30-60 seconds)
- Simple lyrics structures work best
- Conventional genre/style prompts are more reliable

**TTM+VC:**
- Expect higher failure rate than other modes
- Have backup TTM-only output ready
- Consider TTS+VC for voice-focused work instead

---

## Summary

VODER's architecture is designed around several key principles:

1. **Modularity**: Each mode is self-contained with clear inputs and outputs
2. **Automation**: Dialogues and multi-stage pipelines run without user intervention
3. **Compatibility**: FFmpeg handles format conversions transparently
4. **Limitations**: Seed-VC constraints define the boundaries of certain modes

Understanding these internal details helps users work within VODER's capabilities and troubleshoot issues effectively. The tool is actively developed, and reported issues help prioritize future improvements.

For questions, issues, or collaboration opportunities, reach out on X: [@HAKORAdev](https://x.com/HAKORAdev)
