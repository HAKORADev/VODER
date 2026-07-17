# VODER Command Catalog

> Complete reference of every oneline command — the 8 main processing modes (TTS, STS, TTM, STT, SE, SFX, SVS, SS) and the 3 task-layer features (Voice Training, Side-Quests, Chains) — with their flags, keywords, and syntax.
> Modes are sorted by mode order; features follow.

---

## Invocation

```
python voder.py <mode> [keyword] [value] [keyword] [value] ...
python voder.py gui              # launch GUI
python voder.py cli              # interactive CLI mode (no oneline commands)
python voder.py                  # show help message
```

---

## Mode Index

The 8 main processing modes:

| Mode | Name |
|------|------|
| `tts` | Text-to-Speech |
| `sts` | Speech-to-Speech (Voice Conversion) |
| `ttm` | Text-to-Music (generate / remix / repaint / complete / lego / extract / bgm) |
| `stt` | Speech-to-Text (Transcription) |
| `se` | Sound Enhancement |
| `sfx` | Sound Effects Generation |
| `svs` | Song Voice Separate |
| `ss` | Speakers Separator |

The 3 task-layer features:

| Feature | Name |
|---------|------|
| `train voice` | Voice Training (save reusable `.tts` / `.ttse` voice clones) |
| `quest` | Side-Quests (utility tasks outside the voder engine) |
| `chains` | Chains (user-defined pipelines of voder oneline tasks) |

### Quick Jump

| Mode / Feature | Section |
|----------------|---------|
| [Invocation](#invocation) | General syntax & modes |
| [Global Keywords](#global-keywords-available-in-all-modes) | `result` |
| [1. TTS](#1-tts--text-to-speech) | Text-to-Speech, dialogue, SLC, SVC, STS pass, directives, trained voices, newline support |
| [1a. Voice Training](#1a-voice-training--train-voice) | Train voice clones as .tts files |
| [2. STS](#2-sts--speech-to-speech-voice-conversion) | Voice Conversion |
| [3. TTM](#3-ttm--text-to-music) | Generate, VC, Remix, Repaint, Complete, Lego, Extract |
| [4. STT](#4-stt--speech-to-text-transcription) | Transcription, diarization, translate |
| [5. SE](#5-se--sound-enhancement) | Denoise, dereverb, restore, super-resolution |
| [6. SFX](#6-sfx--sound-effects-generation) | Sound effects |
| [7. SVS](#7-svs--song-voice-separate) | Vocal/instrument separation |
| [8. SS](#8-ss--speakers-separator) | Speaker extraction & separation |
| [9. quest](#9-quest--side-quests) | Side-quests (utility tasks) grouped into Media Manipulation (convert, cut, remove, merge, mix, silence, reverse, fade, soundlevel, bassboost, speed, pitch, glue, reverb, loudnorm, noframes) plus standalone `download`. |
| [10. chains](#10-chains--user-defined-pipelines) | Compose multiple voder oneline tasks into a pipeline |
| [Input Types](#input-types) | Supported file & URL formats |
| [Output](#output) | Output directory & naming |

---

## Global Keywords (available in all modes)

### `result "<path>"`

Copy the latest output file to a custom path after the command finishes.

```
python voder.py tts script "hello" voice "male voice" result "C:/output/hello.wav"
python voder.py stt "audio.wav" result "./transcript.txt"
```

---

## 1. `tts` — Text-to-Speech

Generate speech from text using voice descriptions (VoiceDesign) or voice clone targets. Also includes the SLC sub-task (Speaker Language Conversion) and SVC sub-task (Speaker Voice Change) for transcribing and re-synthesizing speech.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `script` | `"<text>"` or `"CharName: text"` | Dialogue line (plain text for single mode, `Character: text` for dialogue mode). Can appear multiple times. |
| `voice` | `"<description>"` or `"CharName: description"` or `"<trained-name>"` or `"CharName: <trained-name>"` | Voice prompt for VoiceDesign TTS, or a trained voice reference. Single mode: one prompt or trained name. Dialogue mode: `"CharName: description"` or `"CharName: trained-name"` per character. Trained voice syntax: `"character-name"` (latest .tts from voices/), `"character-name:path/to/file.tts"` (specific file), `"character-name:another-name"` (latest .tts for another-name). When a trained voice is used, Qwen3-TTS Base (voice cloning) is used instead of VoiceDesign. Can appear multiple times. |
| `target` | `"<path>"` or `"CharName: path"` | Audio path for voice cloning. Single mode: one path. Dialogue mode: `"CharName: path"` per character. **Multi-reference**: `(path1)(path2)(path3)` wraps multiple references in parentheses — each is resolved, SVS-cleaned, and concatenated into a composite for richer voice extraction. **`first` keyword**: add `first` before the references (`target first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling. **`sts:` prefix**: `target "sts:path"` triggers an additional Seed-VC v2 non-mimic voice conversion pass after cloning. Can appear multiple times. |
| `music` | `"<description>"` | Background music description (dialogue mode only). Generated via ACE-Step and mixed under speech. |
| `level` | `"<spec>"` | Music volume levels per dialogue segment, e.g. `"10:20-50 30:60-80"`. Format: `<volume%>:<start_sec>-<end_sec>`. Default: 35%. Dialogue mode only. |
| `reference` | `"<path>"` | Optional reference audio/video/URL for dialogue background music style guidance. Processed through SVS music pipe to extract clean instrumental before use. Accepts audio files, video files, and URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter). Dialogue mode only. |
| `ocr` | `"<image_path>"` | Extract text from an image via EasyOCR, then use that text as the script. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP. |
| `<number>` | `10-300` | Duration in seconds (TTM only, ignored in pure TTS). |
| `slc` | (flag) | Enable SLC (Speaker Language Conversion) sub-task. Transcribe source, clone voice, re-synthesize. See SLC Sub-Task below. |
| `svc` | `"path"` | SVC sub-task: transcribe single-speaker audio and re-synthesize with a target voice. Must be paired with `target` or `voice` for the output voice |
| `dub` | `"path"` | Enable dub sub-task. Input video or audio file path. Auto-implies extreme (Fish S2 Pro). See Dub Sub-Task below. |
| `overdose` | (flag) | Use VibeVoice ASR for dialogue source analysis and voice clip extraction instead of Whisper + pyannote. When used with `music`, also uses ACE-Step XL turbo for enhanced background music quality. With `slc`, runs an additional STS v2 pass for better voice preservation. Requires 24GB+ VRAM or 48GB+ RAM. |
| `extreme` | (flag) | Use Fish Audio S2-Pro instead of Qwen3-TTS for TTS synthesis. Provides higher quality voice cloning, 80+ language support (vs 10 for Qwen3-TTS), and voice effects via `[tag]` syntax (e.g. `[whispering]`, `[laughing]`, `[pause]`, `[excited]`, `[angry]`). Also supports 64 S1 Pro tags in `[brackets]`. Can be combined with `overdose`. Voice design always uses placeholder trick (generates English placeholder → Fish clones it → Fish speaks actual text). Trained voices use `.ttse` format (not `.tts`). |

### Single Mode

One speaker, one line. Use `voice` for VoiceDesign or `target` for voice clone.

```
# VoiceDesign: describe the voice
python voder.py tts script "hello world" voice "male voice"

# Voice clone: provide a reference audio
python voder.py tts script "hello" target "voice.wav"

# Multi-reference clone: concatenate multiple references
python voder.py tts script "hello" target "(voice1.wav)(voice2.wav)(voice3.wav)"

# Multi-reference clone with first keyword: extract only first ref's speaker from all others
python voder.py tts script "hello" target first "(voice1.wav)(voice2.wav)(voice3.wav)"

# OCR: extract text from image then speak it
python voder.py tts ocr "path/to/image.png" voice "text: female voice"
python voder.py tts ocr "path/to/image.png" target "text: voice.wav"
```

### Dialogue Mode

Multiple characters. Use `Character:` prefix in script and voice/target.

```
# Two characters with voice descriptions
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"

# Two characters with voice cloning
python voder.py tts script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav"

# Two characters with trained voices
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: hero" voice "Sarah: heroine"

# Mix trained voice and voice description
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: hero" voice "Sarah: cheerful female"

# Specific .tts file for trained voice
python voder.py tts script "James: Hello" voice "James:voices/voder_tts_hero_20260101.tts"

# Multi-reference cloning per character
python voder.py tts script "James: Hello" target "James:(clip1.wav)(clip2.wav)"

# Multi-reference cloning per character with first keyword
python voder.py tts script "James: Hello" target first "James:(clip1.wav)(clip2.wav)"

# Dialogue with background music
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano"

# Dialogue with music volume levels
python voder.py tts script "James: Hello" script "sfx: thunder /duration:3" voice "James: deep male" music "soft piano" level "10:20-50"

# Dialogue with music and reference for style guidance
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano" reference "path/to/ref.wav"

# Dialogue with music and video file reference
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano" reference "path/to/ref_video.mp4"

# Dialogue with music and YouTube URL reference
python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano" reference "https://youtube.com/watch?v=..."

# TTS with overdose (VibeVoice ASR for dialogue source, enhanced music)
python voder.py tts overdose script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"

# TTS with overdose + voice cloning + background music
python voder.py tts overdose script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav" music "soft piano"

# TTS with extreme (Fish S2-Pro for higher quality TTS)
python voder.py tts extreme script "Hello world" target "voice.wav"

# TTS with extreme + voice design (placeholder trick for all languages)
python voder.py tts extreme script "Arabic text here" voice "deep male"

# TTS with extreme + voice effects
python voder.py tts extreme script "[whispering] Hello there [pause] how are you?" target "voice.wav"

# TTS with extreme + overdose (both can be combined)
python voder.py tts overdose extreme script "James: Hello" target "James: james.wav" music "soft piano"

# TTS extreme with trained .ttse voice
python voder.py tts extreme script "Hello" voice "my-character"
```

### Overdose Notes

- When `overdose` is used with audio as dialogue source, VibeVoice ASR replaces Whisper + pyannote for transcription and diarization.
- Voice clip extraction with overdose automatically trims 2s from start and 3s from end of longest segment to avoid cross-speaker overlap.
- `music` parameter with `overdose` uses ACE-Step XL turbo instead of the standard model for enhanced background music quality.

### Extreme Notes

- When `extreme` is used, Fish Audio S2-Pro replaces Qwen3-TTS for all TTS synthesis steps.
- `extreme` and `overdose` can be used together — they affect different parts of the pipeline (overdose = STT/TTM, extreme = TTS).
- Voice effects are embedded in script text using `[tag]` syntax. Over 15,000 free-form tags are supported — any natural language description in brackets works.
  - **S2-Pro well-tested tags**: `[excited]` `[angry]` `[sad]` `[whispering]` `[soft voice]` `[low voice]` `[loud voice]` `[shouting]` `[sigh]` `[inhale]` `[exhale]` `[gasp]` `[panting]` `[clears throat]` `[laughing]` `[chuckling]` `[giggle]` `[sobbing]` `[crying]` `[groan]` `[pause]` `[short pause]` `[long pause]` `[emphasis]` `[rustling sound]`
  - **S1 Pro tags** (also work in `[brackets]` for S2-Pro): `(angry)` `(sad)` `(disdainful)` `(excited)` `(surprised)` `(satisfied)` `(unhappy)` `(anxious)` `(hysterical)` `(delighted)` `(scared)` `(worried)` `(indifferent)` `(upset)` `(impatient)` `(nervous)` `(guilty)` `(scornful)` `(frustrated)` `(depressed)` `(panicked)` `(furious)` `(empathetic)` `(embarrassed)` `(reluctant)` `(disgusted)` `(keen)` `(moved)` `(proud)` `(relaxed)` `(grateful)` `(confident)` `(interested)` `(curious)` `(confused)` `(joyful)` `(disapproving)` `(negative)` `(denying)` `(astonished)` `(serious)` `(sarcastic)` `(sneering)` `(hesitating)` `(yielding)` `(painful)` `(awkward)` `(amused)` `(in a hurry tone)` `(shouting)` `(screaming)` `(whispering)` `(soft tone)` `(laughing)` `(chuckling)` `(sobbing)` `(crying loudly)` `(sighing)` `(panting)` `(groaning)` `(crowd laughing)` `(background laughter)` `(audience laughing)`
  - **Free-form examples**: `[professional broadcast tone]`, `[pitch up]`, `[voice rough from crying, trying to sound normal]`, `[dead tired, end of a very long shift]`
  - **Multi-language tags**: Supported (e.g. `[低声说]` for Chinese, `[囁き声で]` for Japanese)
  - Tags affect text from their position onward. Placement matters: `[whispering] I didn't want to go inside.` whispers the whole line, while `I didn't want to go [whispering] inside.` whispers from "inside" onward.
- When `extreme` is used with a `voice` prompt (not `target`), VODER always generates placeholder English speech via VoiceDesign, clones it with Fish, then Fish speaks the actual text. This applies unconditionally to ensure consistent voice quality and preserve voice effects tags across all languages.
- Trained voices for extreme mode use `.ttse` files (saved via `train extreme voice:name`). Using `.tts` with extreme or `.ttse` without extreme produces a clear error message.
- Fish S2-Pro supports 80+ languages natively, far beyond Qwen3-TTS's 10.
- STS voice pass (`sts:` prefix) is optionally applicable with extreme mode — Fish's integrated cloning already produces high-fidelity output, but the additional Seed-VC v2 pass can further refine voice matching if desired.

### SLC Sub-Task

Speaker Language Conversion: transcribe speech from an audio/video source, translate to English (default) or any language (with `translate (source-target)` or `translate (target)`), clone the speaker's voice, and re-synthesize. SVS voice isolation is automatically run on the source before transcription.

| Keyword | Value | Description |
|---------|-------|-------------|
| `slc` | (flag) | Enable SLC sub-task. Translates to English by default. |
| `translate (source-target)` or `translate (target)` | `(auto-en)` / `(ja-en)` / `(ar)` etc. | Any-to-any translation via TranslateGemma 12B (76 languages). Overrides default English-only translation. `(target)` is shorthand for `(auto-target)`. |
| `music` | (flag) | Preserve non-vocals: extract music from source via SVS music and blend with voice output. |
| `"<path>"` | file | Audio/video file path or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL. |
| `result` | `"<path>"` | Copy output to custom path. |

#### Rules

- Pipeline: SVS voice isolation → Whisper large-v3 (transcribe + translate to English) → Qwen-TTS with voice cloning. With `translate (source-target)` or `translate (target)`: SVS voice isolation → Whisper large-v3 (transcribe) → TranslateGemma 12B (translate to target language) → Qwen-TTS with voice cloning.
- Default translation target is English. With `translate (source-target)` or `translate (target)`, TranslateGemma 12B handles translation to any of 76 languages.
- Uses Whisper large-v3 (not turbo) for maximum translation accuracy.
- Supports audio files, video files, and YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, and X/Twitter URLs.
- `music` flag extracts the instrumental track from the source and blends it with the voice output, preserving background music.
- `overdose slc` runs an additional STS v2 pass after TTS for better voice preservation.
- `extreme slc` uses Fish S2-Pro instead of Qwen3-TTS for the resynthesis step, producing higher quality voice cloning.
- `overdose extreme slc` combines both: VibeVoice ASR for transcription + Fish S2-Pro for resynthesis.
- Voice-music sync may vary when using the `music` flag; this is inherent to the approach.

```
# Translate speech to English, preserving voice
python voder.py tts slc "french_speech.wav"

# Translate with music preservation (blend non-vocals back)
python voder.py tts slc music "french_speech.wav"

# SLC from video
python voder.py tts slc "interview.mp4"

# SLC from YouTube
python voder.py tts slc "https://youtube.com/watch?v=..."

# SLC with overdose (additional STS v2 pass for better voice preservation)
python voder.py tts overdose slc "french_speech.wav"

# SLC with overdose + music preservation
python voder.py tts overdose slc music "french_speech.wav"

# SLC with extreme (Fish S2-Pro for resynthesis)
python voder.py tts extreme slc "french_speech.wav"

# SLC with overdose + extreme
python voder.py tts overdose extreme slc "french_speech.wav"

# SLC any-to-any: translate to Arabic with original voice (TranslateGemma 12B)
python voder.py tts slc translate "(auto-ar)" "french_speech.wav"

# SLC any-to-any with music preservation
python voder.py tts slc translate "(auto-ar)" music "french_speech.wav"

# Shorthand: (ar) is equivalent to (auto-ar)
python voder.py tts slc translate "(ar)" "french_speech.wav"

# SLC any-to-any: Japanese to English
python voder.py tts slc translate "(ja-en)" "japanese_speech.wav"
```

### SVC Sub-Task

Speaker Voice Change: transcribe single-speaker audio and re-synthesize with a different target voice. Unlike SLC (which preserves the original voice and changes language), SVC preserves the content/language but changes the speaker's voice.

**Command format:** `voder.py tts [overdose] [extreme] svc "source_path" target "voice_ref"`

**Pipeline:** SVS voice isolation → Whisper/VibeVoice transcription → Qwen-TTS/Fish synthesis with target voice → optional STS v2 pass (if `sts:` prefix on target; works with extreme mode for additional voice refinement)

```
# Basic SVC
python src/voder.py tts svc "speech.wav" target "voice_ref.wav"

# SVC with voice description
python src/voder.py tts svc "speech.wav" voice "deep male, authoritative"

# Overdose mode (VibeVoice ASR transcription)
python src/voder.py tts overdose svc "speech.wav" target "voice.wav"

# Extreme mode (Fish S2-Pro for resynthesis)
python src/voder.py tts extreme svc "speech.wav" target "voice.wav"

# Overdose + extreme combined
python src/voder.py tts overdose extreme svc "speech.wav" target "voice.wav"

# SVC with STS voice pass
python src/voder.py tts svc "speech.wav" target "sts:voice.wav"

# SVC with multi-reference target
python src/voder.py tts svc "speech.wav" target "(ref1.wav)(ref2.wav)(ref3.wav)"

# SVC with STS pass and multi-reference
python src/voder.py tts svc "speech.wav" target "sts:(ref1.wav)(ref2.wav)"
```

| Output Pattern | Mode |
|----------------|------|
| `voder_tts_svc_*.wav` | Standard SVC |
| `voder_tts_svc_sts_*.wav` | SVC with STS voice pass |

### STS Voice Pass (`sts:` Prefix)

Prefix `sts:` on any `target` reference triggers an additional Seed-VC v2 non-mimic voice conversion pass after the standard Qwen-TTS cloning synthesis. This applies an extra voice conversion layer for enhanced voice matching fidelity. The `sts:` prefix works with multi-reference format: `target "sts:(ref1)(ref2)(ref3)"`.

**Where it works:** Single TTS, Dialogue TTS, SVC sub-task, Interactive Modify Speech.

```
# Single TTS with STS pass
python src/voder.py tts script "Hello" target "sts:ref.wav"

# Dialogue TTS with STS pass
python src/voder.py tts script "Alice: Hi" voice "Alice: cheerful" target "Alice: sts:ref.wav"

# SVC with STS pass
python src/voder.py tts svc "input.wav" target "sts:ref.wav"
```

**Output naming:** `voder_tts_sts_*.wav` (single mode)

### Dub Sub-Task

Video/audio dubbing with voice cloning, optional translation, subtitle burning, and speed adjustment. Defaults to auto-detect source language and translate to English (no `translate` keyword needed for English target). Transcribes speech with VibeVoice ASR, optionally translates with TranslateGemma 12B, generates per-segment TTS with timeline-based assembly (preserving audio events for non-speech detection), re-synthesizes with Fish S2 Pro using voice cloning from each speaker's original audio, adjusts speed to match original timing (threshold 1.5x/0.5x), mixes with instrumentals, and muxes with video.

**Canonical full-form command:**
```
python voder.py tts overdose extreme se dub subtitle "(auto-en)" translate "(auto-ja)" video "path"
```
Where `overdose` and `extreme` are auto-implied by `dub` but recommended to include for clarity, `se` enables sound enhancement, `subtitle "(auto-en)"` burns subtitles with an independent translation to English, `translate "(auto-ja)"` translates the dubbed audio to Japanese, and `video "path"` specifies the input.

| Keyword | Value | Description |
|---------|-------|-------------|
| `dub` | `"path"` | Enable dub sub-task. Input video or audio file path. Auto-implies extreme (Fish S2 Pro). |
| `translate (source-target)` or `translate (target)` | `(auto-ar)` / `(ja-en)` / `(ar)` etc. | Any-to-any translation via TranslateGemma 12B (76 languages). Optional. Overrides default auto→English. `(target)` is shorthand for `(auto-target)`. |
| `subtitle` | (flag) | Transcribe dubbed audio with VibeVoice ASR and burn subtitles onto the output video (final step after dubbing). |
| `subtitle original` | (flag) | Burn subtitles derived from the original audio processing chain (TTS text with original timing). |
| `subtitle (source-target)` or `subtitle (target)` | `(auto-en)` / `(ja-en)` / `(en)` etc. | Transcribe dubbed audio and burn independently translated subtitles (separate from dub audio language). Optional. `(target)` is shorthand for `(auto-target)`. |
| `subtitle original (source-target)` or `subtitle original (target)` | `(auto-en)` / `(ja-en)` / `(en)` etc. | Burn subtitles from the original audio chain with independent translation. Optional. `(target)` is shorthand for `(auto-target)`. |
| `se` | (flag) | Enable sound enhancement before ASR. Optional. |
| `video "path"` | `"path"` | Specify input video path. |
| `video` | (flag) | When source is a URL, download the full video and output MP4 (default: audio download → WAV). Implicit when `subtitle` is used with a URL. |
| `overdose` | (flag) | Auto-implied by dub. Can be specified for clarity. |
| `result "path"` | `"path"` | Custom output path. |

#### Rules

- Pipeline: Download/extract → SVS voice+music separation → VibeVoice ASR (speaker diarization) → audio event detection (preserves non-speech segments) → TranslateGemma translation (auto→English by default; `translate (source-target)` or `translate (target)` for other targets) → Fish S2 Pro TTS per-segment with timeline-based assembly (voice cloning from source) → per-speaker speed adjustment (threshold 1.5x/0.5x) → mix with instrumentals → mux with video.
- Dub defaults to auto→English translation. No `translate` keyword is needed for English target; TranslateGemma automatically translates from auto-detected source to English.
- VibeVoice ASR is always used (overdose is implied by dub).
- Fish S2 Pro is always used for TTS synthesis (extreme is implied by dub).
- `overdose` and `extreme` are auto-implied by `dub` but can be explicitly specified for clarity.
- Per-segment TTS generation: each speech segment is synthesized individually and assembled on the original timeline, with audio events preserved for non-speech detection.
- Speed adjustment threshold: segments are speed-adjusted only when the ratio exceeds 1.5x or falls below 0.5x; otherwise original pacing is preserved.
- VibeVoice ASR and Fish S2 Pro are loaded separately (never simultaneously) to fit within 24GB VRAM.
- Video file input produces MP4 output; audio file input produces WAV output.
- URL input: audio is downloaded by default (WAV output); add the `video` keyword to download the full video (MP4 output). When `subtitle` is used with a URL, video is downloaded automatically (subtitles require video frames).
- `subtitle` (bare) transcribes the dubbed audio using VibeVoice ASR and burns subtitles onto the output video; this is the final step after dubbing.
- `subtitle original` derives subtitles from the original audio processing chain (TTS text with original timing).
- `subtitle (source-target)` or `subtitle (target)` transcribes the dubbed audio and applies an independent translation pass for subtitles.
- `subtitle original (source-target)` or `subtitle original (target)` derives subtitles from the original audio chain with an independent translation pass.
- `se` enhances vocal audio before ASR for better transcription in noisy conditions (see [Section 5](#5-se--sound-enhancement) for full SE sub-modes).
- TranslateGemma loads once for all translations (dub + subtitle) then unloads once.
- Requires 24GB+ VRAM and FFmpeg.

```
# Basic dub (voice cloning from source)
python voder.py tts dub "video.mp4"

# Dub with subtitle burning (transcribes dubbed audio for accurate subtitles)
python voder.py tts dub subtitle "video.mp4"

# Dub with subtitles from original audio processing chain
python voder.py tts dub subtitle original "video.mp4"

# Dub with translation to Arabic
python voder.py tts dub translate "(auto-ar)" "video.mp4"

# Shorthand: (ar) is equivalent to (auto-ar)
python voder.py tts dub translate "(ar)" "video.mp4"

# Dub with translation and subtitles (transcribes dubbed audio)
python voder.py tts dub translate "(auto-ar)" subtitle "video.mp4"

# Dub with independent subtitle translation + dub audio translation
python voder.py tts dub subtitle "(auto-en)" translate "(auto-ja)" "video.mp4"

# Shorthand: (en) and (ja) equivalent to (auto-en) and (auto-ja)
python voder.py tts dub subtitle "(en)" translate "(ja)" "video.mp4"

# Dub with original-chain subtitles independently translated
python voder.py tts dub subtitle original "(auto-en)" translate "(auto-ja)" "video.mp4"

# Full-form with all keywords explicit
python voder.py tts overdose extreme se dub translate "(auto-ar)" subtitle "video.mp4"

# Dub audio file (output is WAV)
python voder.py tts dub "audio.wav"

# Dub with specific source-target translation
python voder.py tts dub translate "(ja-en)" "japanese_video.mp4"

# Dub from URL — audio downloaded by default, output WAV
python voder.py tts dub "https://youtube.com/watch?v=..."

# Dub from URL with video keyword — video downloaded, output MP4 with dubbed audio muxed back
python voder.py tts dub video "https://youtube.com/watch?v=..."

# Dub from URL with subtitle keyword — video is downloaded (subtitles require frames)
python voder.py tts dub subtitle "https://youtube.com/watch?v=..."
```

### Newline Support

Use `\n` in script text for actual newlines. Works in both oneline and interactive CLI modes.

```
# Newline in dialogue script
python voder.py tts script "James: First line\nSecond line" voice "James: deep male"

# Newline in single mode
python voder.py tts script "First paragraph\nSecond paragraph" voice "professional narrator"
```

### Voice Stabilization

VoiceDesign characters in dialogue mode automatically get their voice stabilized. After 3 script lines, the outputs are concatenated, SVS-cleaned, and fed to Qwen3-TTS Base for voice extraction. All subsequent lines use the cloned voice instead of VoiceDesign, eliminating vocal drift in long dialogues. This happens automatically with no configuration needed.

### Trained Voice Usage

When using the `voice` parameter, a trained voice name or path can be used instead of a voice description. When a trained voice is used, the corresponding TTS model (Qwen3-TTS Base for `.tts`, Fish S2-Pro for `.ttse`) is used instead of VoiceDesign.

| Syntax | Behavior |
|--------|----------|
| `voice "character-name"` | Uses the latest `.tts` (or `.ttse` with `extreme`) file with that name from `voices/` |
| `voice "character-name:path/to/file.tts"` | Uses a specific `.tts` file (standard mode only) |
| `voice "character-name:path/to/file.ttse"` | Uses a specific `.ttse` file (extreme mode only) |
| `voice "character-name:another-name"` | Uses the latest `.tts` (or `.ttse`) file for `another-name` from `voices/` |

This works in both oneline and interactive CLI modes. Using a `.tts` file with `extreme` or a `.ttse` file without `extreme` produces an error.

### Script Directives (per line, at end of text)

Directives are appended at the end of a script line, separated by spaces.

| Directive | Description |
|-----------|-------------|
| `/time:nn-nn+nn` | Trim and pad the audio for that line. First bare number = pad from start (seconds). `-nn` = cut from end. `+nn` = cut from start. Parts are additive. |
| `/level:0-100` | Volume level for that line (default: 100) |
| `/duration:1-30` | SFX duration in seconds (required for `sfx:` lines) |
| `sfx: <prompt>` | Special character name: generates a sound effect via TangoFlux instead of speech |

#### Directive Examples

##### `/time:` — trim and position audio

Format: `/time:nn-nn+nn` — first bare number is the timeline position (dialogue) or pre-padding (single). `-nn` cuts from end, `+nn` cuts from start. Parts are additive.

```
# Cut 2 seconds from end
python voder.py tts script "James: Hello there /time:-2" voice "James: deep male"

# Cut 1 second from start
python voder.py tts script "Sarah: Hi /time:+1" voice "Sarah: cheerful female"

# Cut 2 from end AND 1 from start
python voder.py tts script "James: Hello /time:-2+1" voice "James: deep male"

# Multiple additive cuts: 2+1=3 from start, 1+2=3 from end
python voder.py tts script "James: Hi /time:+2+1-1-2" voice "James: deep male"

# Dialogue timeline positioning: James at 0s, Sarah starts at 5s
python voder.py tts script "James: Hello there /time:0" script "Sarah: Hi, nice to meet you /time:5" voice "James: deep male" voice "Sarah: cheerful female"

# Dialogue with positioning + trim: James at 0s cut 1s end, Sarah at 3s
python voder.py tts script "James: Long greeting /time:0-1" script "Sarah: Response /time:3" voice "James: deep male" voice "Sarah: cheerful female"
```

##### `/level:` — volume per line

Format: `/level:0-100` — sets playback volume for that specific line. Default is 100.

```
# Set this line to 50% volume
python voder.py tts script "James: (whispering) /level:50" voice "James: deep male"

# Set this line to 80% volume
python voder.py tts script "Sarah: Background comment /level:80" voice "Sarah: cheerful female"

# Mix volumes in dialogue: James loud, Sarah quiet
python voder.py tts script "James: Hello everyone! /level:100" script "Sarah: (murmurs) /level:30" voice "James: deep male" voice "Sarah: cheerful female"

# Combine /level with /time
python voder.py tts script "James: Shout /time:0 /level:100" script "Sarah: Quiet reply /time:3 /level:40" voice "James: deep male" voice "Sarah: cheerful female"
```

##### `/duration:` — SFX length

Format: `/duration:1-30` — required for `sfx:` character lines. Sets how long the generated sound effect lasts in seconds.

```
# 3-second thunder sound effect
python voder.py tts script "sfx: thunder /duration:3"

# SFX in dialogue: 5-second rain, then speech
python voder.py tts script "sfx: heavy rain /duration:5" script "James: Storm's here /time:5" voice "James: deep male"

# SFX with volume control
python voder.py tts script "sfx: explosion /duration:2 /level:60" script "James: What was that?! /time:2" voice "James: deep male"

# Multiple SFX in sequence
python voder.py tts script "sfx: door creak /duration:2" script "sfx: footsteps /duration:3 /time:2" script "James: Who's there? /time:5" voice "James: deep male"
```

---

## 1a. Voice Training — `train voice` / `train extreme voice`

> **Note:** `train` saves a voice clone as a `.tts` (standard Qwen3-TTS) or `.ttse` (extreme Fish S2-Pro) file in `voices/` for later reuse in TTS via the `voice "<name>"` parameter.

Train a voice clone from reference audio and save it for later reuse. Oneline-only command.

**Standard mode** (`train voice`) uses Qwen3-TTS Base and saves `.tts` files.
**Extreme mode** (`train extreme voice`) uses Fish Audio S2-Pro and saves `.ttse` files.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `voice:name` | `"<name>"` | Character name for the trained voice (used to reference it later via `voice` in TTS). Required. |
| `"path1" "path2" ...` | `"<path>"` | One or more audio file paths for reference audio. Multiple paths are SVS-cleaned individually and concatenated into a composite before voice extraction. Required (at least one). |
| `first` | (flag) | Extract only the first reference's speaker from all others via TSE before compiling. Only meaningful with multiple references. Optional. |
| `test` | (flag) | Generate a test sample after training using a hardcoded 30+ second script. Optional. |
| `test "script"` | `"<text>"` | Generate a test sample using a custom test script. Optional. |

### Rules

- Oneline-only command. Not available in interactive CLI or GUI.
- Standard output: `voder_tts_<name>_<timestamp>.tts` in the `voices/` directory.
- Extreme output: `voder_ttse_<name>_<timestamp>.ttse` in the `voices/` directory.
- `.tts` files can only be used without `extreme`; `.ttse` files can only be used with `extreme`. A clear error is shown on mismatch.
- The `test` keyword can appear at the end of the command (no custom script) or with a quoted custom script.
- Multiple reference paths are supported — each is SVS-cleaned before concatenation.
- The `first` keyword uses TSE (Target Speaker Extraction) to isolate only the first reference's speaker voice from all other references before compiling them into the composite.

```
# Train a voice from a single reference (standard Qwen3-TTS)
python voder.py train voice:narrator "narrator_ref.wav"

# Train from multiple references
python voder.py train voice:hero "hero_clip1.wav" "hero_clip2.wav" "hero_clip3.wav"

# Train with first keyword (extract only first ref's speaker from all others via TSE)
python voder.py train voice:hero first "hero_clip1.wav" "hero_clip2.wav" "hero_clip3.wav"

# Train with test sample (hardcoded script)
python voder.py train voice:my-character "ref1.wav" "ref2.wav" test

# Train with custom test script
python voder.py train voice:my-character "ref1.wav" test "Custom test script for verification"

# Train extreme voice (Fish S2-Pro, saves .ttse)
python voder.py train extreme voice:narrator "narrator_ref.wav"

# Train extreme from multiple references
python voder.py train extreme voice:hero "hero_clip1.wav" "hero_clip2.wav"

# Train extreme with test sample
python voder.py train extreme voice:my-character "ref1.wav" test
```

---

## 2. `sts` — Speech-to-Speech (Voice Conversion)

Convert voice from a base audio to match a target voice. Source vocals are automatically separated via SVS before conversion, and source music is mixed back afterward (unless `nomusic` is used).

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `base` | `"<path>"` | Source audio/video file path or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL. The audio whose content will be preserved. |
| `target` | `"<path>"` | Reference voice audio. The voice characteristics to apply. Auto-extracts clean vocals. **Multi-reference**: `(path1)(path2)(path3)` wraps multiple references in parentheses — each is resolved, SVS-cleaned, and concatenated into a composite for richer voice extraction. **`first` keyword**: add `first` before the references (`target first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling. |
| `music` | (flag) | Use Seed-VC v1 (44.1kHz music model) instead of v2 (22.05kHz speech model). Input must be audio (not video). Auto-extracts vocals from source and target. |
| `mimic` | (flag) | Convert style + voice (not just voice). Uses Seed-VC v2 with `convert_style=True`. Cannot be combined with `music`. Input must be audio (not video). |
| `nomusic` | (flag) | Output converted voice only without mixing back source music. Cannot be combined with `music`. |
| `original` | (flag) | Skip SVS split on the source audio. The full original source is processed directly with the SVS-cleaned target reference. Useful when the source separation step introduces artifacts that degrade the final result. Cannot be combined with `nomusic` (no music to mix back). |
| `extreme` | (flag) | Pre-process the target voice reference through Fish S2 Pro before Seed-VC conversion. Transcribes the compiled target reference with VibeVoice ASR, then synthesizes it with Fish S2 Pro to produce a cleaner, more natural voice profile. This extracts the dominant voice and removes background artifacts/noise from the reference, giving Seed-VC a cleaner input. Works with both Seed-VC v1 (music) and v2 (standard/mimic). Oneline mode only. |

### Rules

- `music` and `mimic` cannot be used together.
- `nomusic` and `music` cannot be used together.
- `original` and `nomusic` can be used together but `original` means no music is available to mix back regardless.
- Base can be audio or video in standard mode. `music` and `mimic` require audio input only (video is rejected).
- By default, source vocals and music are automatically separated via SVS before conversion; music is recombined after (unless `nomusic`). With `original`, the source is used as-is without SVS splitting.
- Target vocals are automatically cleaned via SVS before conversion.
- Output is upsampled to 44100Hz.
- `extreme` pre-processes the target reference (not the source) through Fish S2 Pro. The extreme pass transcribes and re-synthesizes the target reference, producing a cleaner voice profile for Seed-VC. If the extreme pass fails (transcription empty, encoding failure, or synthesis failure), the original target reference is used as fallback.
- Output filenames: music mode uses `voder_m_sts_*.wav`, standard/mimic/nomusic uses `voder_sts_*.wav`.

```
# Standard voice conversion (speech)
python voder.py sts base "input.wav" target "voice.wav"

# Music voice conversion (44.1kHz model)
python voder.py sts base "input.wav" target "voice.wav" music

# Style + voice mimic
python voder.py sts base "input.wav" target "voice.wav" mimic

# Voice-only output (no music recombination)
python voder.py sts base "input.wav" target "voice.wav" nomusic

# Video input (extracts audio, converts, merges back)
python voder.py sts base "input.mp4" target "voice.wav"

# Multi-reference target (oneline only, concatenates multiple voice references)
python voder.py sts base "input.wav" target "(voice1.wav)(voice2.wav)(voice3.wav)"

# Multi-reference target with first keyword (extract only first ref's speaker from all others)
python voder.py sts base "input.wav" target first "(voice1.wav)(voice2.wav)(voice3.wav)"

# Use original source without SVS splitting (avoids separation artifacts)
python voder.py sts original base "input.wav" target "voice.wav"

# Original + mimic (process full source, mimic style and voice)
python voder.py sts original mimic base "input.wav" target "voice.wav"

# Extreme pass: clean target reference with Fish S2 Pro before Seed-VC
python voder.py sts extreme base "input.wav" target "voice.wav"

# Extreme + music (Fish S2 Pro clean reference, then Seed-VC v1)
python voder.py sts extreme base "song.wav" target "voice.wav" music

# Extreme + mimic (Fish S2 Pro clean reference, then Seed-VC v2 style transfer)
python voder.py sts extreme base "input.wav" target "voice.wav" mimic

# Extreme + original (Fish S2 Pro clean reference, no SVS on source)
python voder.py sts extreme original base "input.wav" target "voice.wav"
```

---

## 3. `ttm` — Text-to-Music

The most feature-rich mode. Supports generation, remix, repaint, voice cloning, and three sub-tasks (complete, lego, extract).

### Global TTM Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `lyrics` | `"<text>"` | Song lyrics text. Write `\n` for line breaks (parsed as actual newlines). Use structural tags in `[brackets]` (e.g. `[Verse 1]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`, `[Interlude]`, `[Instrumental]`/`[inst]`, `[Pre-Chorus]`, `[Hook]`, `[Solo]`, `[Break]`) — text inside brackets is not sung. Use `...` for empty lyrics (instrumental). Use `(text in parens)` for context/style hints. |
| `styling` | `"<text>"` | Style/mood prompt for the music. Write `\n` for line breaks (parsed as actual newlines). |
| `<number>` | `10-300` | Duration in seconds (for generate and VC paths). |
| `overdose` | (flag) | Use Overdose tier (ACE-Step XL-Turbo + 4B LM + shift 3.0) instead of Standard tier (ACE-Step 1.5 Turbo). |
| `result` | `"<path>"` | Copy output to custom path (see Global Keywords). |

---

### Reference Time Spec Format

The `reference` path value can include an optional time spec to select a specific portion of the reference audio instead of using the entire file. This applies to all TTM sub-tasks that accept references (remix, repaint, complete, lego, bgm).

| Format | Example | Description |
|--------|---------|-------------|
| `nn(path)` | `"50(ref.wav)"` | Start at nn seconds, extract up to slot max |
| `nn-nn(path)` | `"20-30(ref.wav)"` | Use specified range; slides to reach slot max if shorter |
| `nn-nn/nn-nn/nn-nn(path)` | `"20-30/40-50(ref.wav)"` | Multiple ranges from same audio, combined to reach slot max |
| `stem/(path)` | `"drums/(ref.wav)"` | Extract a single stem from the reference audio via ACE-Step |
| `stem-stem/(path)` | `"bass-drums/(ref.wav)"` | Extract multiple stems and mix them together |
| `stem/nn-nn(path)` | `"drums/20-30(ref.wav)"` | Extract stem then cut to time range |

The time spec and stem spec are both optional -- the old format `reference "ref.wav"` still works and uses the entire audio.

**Stem extraction** uses the ACE-Step XL-Base model to extract specific instrument tracks from the reference audio. The 12 available stems are: `woodwinds`, `brass`, `fx`, `synth`, `strings`, `percussion`, `keyboard`, `guitar`, `bass`, `drums`, `backing_vocals`, `vocals`. Multiple stems joined by `-` are extracted individually then mixed together via ffmpeg. Stem extraction runs after SVS (voice/music) and before time-range cutting.

**Stem validation rules:**
- `voice` prefix: only vocal stems (`vocals`, `backing_vocals`) are valid. Instrument stems produce a clear error.
- `music` prefix: only instrument stems (`woodwinds`, `brass`, `fx`, `synth`, `strings`, `percussion`, `keyboard`, `guitar`, `bass`, `drums`) are valid. Vocal stems produce a clear error.
- As-is (no prefix): all 12 stems are valid. The `everything` keyword is rejected — as-is already provides the full audio.
- Unrecognized stem names are removed with a warning listing the first 5 unrecognized keywords (+ count of remaining). If any recognized stems remain, extraction proceeds with only those. If no valid stems remain, the reference is skipped.

**Slot max by reference count:**

| Refs | Slot Max Each |
|------|---------------|
| 1 | 30s |
| 2 | 15s |
| 3 | 10s |

**Sliding logic:** If the specified range is shorter than the slot max, the start is slid back and/or the end is slid forward until the slot max duration is reached. If the audio is shorter than the slot max, segments loop to fill the slot. If the combined segments exceed the slot max, they are used as-is.

**With voice/music prefix:**

```
# Start at 50 seconds, extract up to 30s
reference voice "50(ref.wav)"

# Use 20-30s range from first ref, 40-50s from second; slides each to 15s
reference music "20-30(ref1.wav)" "40-50(ref2.wav)"

# Multiple ranges from same audio, combined to reach slot max
reference voice "20-30/40-50(ref.wav)"
```

**With stem extraction:**

```
# Extract drums from reference audio
reference "drums/(ref.wav)"

# Extract vocals, then isolate the vocals stem
reference voice "vocals/(ref.wav)"

# Extract keyboard stem then cut to 20-30s range
reference "keyboard/20-30(ref.wav)"

# Extract bass and drums from music (SVS), then cut to 30-60s
reference music "bass-drums/30-60(ref.wav)"
```

**In repaint multi-pass specs:**

```
# Time spec inside a repaint pass spec
python voder.py ttm repaint "song.wav" "20-80/styling(jazz)/reference-voice(30-60(vocals.wav))"
```

---

### 3a. Standard Generate

Basic text-to-music generation. Supports optional reference audio via `target`.

| Keyword | Value | Description |
|---------|-------|-------------|
| `target` | `"<path>"` | Reference audio (as-is). Supports stem spec: `target "drums/(ref.wav)"`. All 12 stems valid. |
| `target voice` | `"<path>"` | Reference audio — extract vocals via SVS first. Supports stem spec: `target voice "vocals/(ref.wav)"`. Only vocal stems (`vocals`, `backing_vocals`) valid. |
| `target music` | `"<path>"` | Reference audio — extract instruments via SVS first. Supports stem spec: `target music "drums/(ref.wav)"`. Only instrument stems valid. |
| `voice` | (flag) | Generate song then extract vocals via SVS voice pipe. Output is clean vocals only. |

```
# Generate music from lyrics and style
python voder.py ttm lyrics "walking down the road" styling "pop rock" 30

# Multi-line lyrics using \n for line breaks
python voder.py ttm lyrics "Verse 1: walking down the road\nChorus: singing in the rain\nVerse 2: under the stars tonight" styling "pop rock" 30

# Multi-line styling
python voder.py ttm lyrics "hello world" styling "upbeat pop\nfemale vocals\npiano and drums" 30

# Generate with reference audio (as-is, full audio)
python voder.py ttm lyrics "walking down the road" styling "pop rock" 30 target "ref.wav"

# Generate with reference — extract vocals only
python voder.py ttm lyrics "walking down the road" styling "pop rock" 30 target voice "vocals.wav"

# Generate with reference — extract instruments only
python voder.py ttm lyrics "walking down the road" styling "pop rock" 30 target music "instrumental.wav"

# Generate with overdose tier (after mode name)
python voder.py ttm overdose lyrics "walking down the road" styling "pop rock" 30

# Generate song then extract vocals only
python voder.py ttm voice lyrics "walking down the road" styling "pop rock" 30

# Generate song with reference then extract vocals
python voder.py ttm voice lyrics "walking down the road" styling "pop rock" 30 target voice "ref.wav"
```

### 3b. Voice Cloning (`vc`)

Generate music then convert the vocal to match a clone voice via Seed-VC v1.

| Keyword | Value | Description |
|---------|-------|-------------|
| `vc` | (flag) | Enable voice cloning mode. |
| `clone` | `"<path>"` | Source voice audio for cloning. Auto-extracts clean vocals. **Multi-reference**: `(path1)(path2)(path3)` wraps multiple references in parentheses — each is resolved, SVS-cleaned, and concatenated into a composite for richer voice extraction. **`first` keyword**: add `first` before the references (`clone first "(path1)(path2)"`) to extract only the first reference's speaker from all others via TSE before compiling. |
| `target` | `"<path>"` | Optional reference audio for music generation (as-is). |
| `target voice` | `"<path>"` | Optional reference — extract vocals via SVS first. |
| `target music` | `"<path>"` | Optional reference — extract instruments via SVS first. |

#### Rules

- `vc` requires `lyrics`, `styling`, `duration`, and `clone`.
- `vc` cannot be combined with `remix` or `repaint`.
- `clone` without `vc` produces a warning (clone is ignored).

```
# Basic voice clone
python voder.py ttm vc lyrics "hello world" styling "pop" 30 clone "voice.wav"

# Voice clone with overdose (XL-Turbo + 4B LM)
python voder.py ttm overdose vc lyrics "hello world" styling "pop" 30 clone "voice.wav"

# Voice clone with reference (as-is)
python voder.py ttm vc lyrics "hello world" styling "pop" 30 clone "voice.wav" target "ref.wav"

# Voice clone with reference (extract vocals)
python voder.py ttm vc lyrics "hello world" styling "pop" 30 clone "voice.wav" target voice "vocals.wav"

# Voice clone with multi-reference (oneline only, concatenates multiple voice references)
python voder.py ttm vc lyrics "hello world" styling "pop" 30 clone "(voice1.wav)(voice2.wav)(voice3.wav)"

# Voice clone with multi-reference + first keyword (extract only first ref's speaker from all others)
python voder.py ttm vc lyrics "hello world" styling "pop" 30 clone first "(voice1.wav)(voice2.wav)(voice3.wav)"
```

---

### 3c. Remix

Re-generate a song in a new style. Uses ACE-Step cover method. Supports **multi-source** (up to 3) and **multi-reference** (up to 3).

| Keyword | Value | Description |
|---------|-------|-------------|
| `remix` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Source audio(s) to remix (paths directly after `remix`). Up to 3 sources with optional `voice`/`music` prefix per source. Multiple sources are composed into one (equal time per source). |
| `lyrics` | `"<text>"` | Optional lyrics to guide new vocal content in the remix. |
| `styling` | `"<text>"` | New style prompt for the remix. |
| `bias` | `"<0-100>"` | Cover strength bias. 0 = full original, 100 = full cover. Snaps to nearest 10; values ending in 5 snap down to the lower multiple of 10 (e.g., 45 → 0.4, 15 → 0.1). Default: 40 (= 0.4 strength). |
| `reference` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Optional reference audio(s). Up to 3 with optional `voice`/`music` prefix per entry. Multiple refs are composed into a 30s composite. |
| `overdose` | (flag) | Use Overdose tier. |

#### Rules

- `remix` requires `styling`.
- `lyrics` is optional. When provided, the model uses the lyrics to guide vocal generation in the remix.
- Cannot be combined with `vc`.
- Up to 3 sources; excess entries produce a warning and are trimmed.
- Up to 3 references; excess entries produce a warning and are trimmed.
- `voice` prefix before a source/reference path extracts vocals via SVS.
- `music` prefix before a source/reference path extracts instruments via SVS.
- No prefix uses the audio as-is.
- Multi-source composition: total duration = sum of all source durations; each source contributes equal time.
- Multi-reference composition (2 refs): 10s front of ref1 + 5s mid of ref1 + 5s mid of ref2 + 10s end of ref2 = 30s.
- Multi-reference composition (3 refs): 10s front of ref1 + 10s mid of ref2 + 10s end of ref3 = 30s.
- Reference can be a local file, video file, or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL.

```
# Basic remix
python voder.py ttm remix "song.wav" styling "jazz"

# Remix with custom lyrics (new vocal content)
python voder.py ttm remix "song.wav" lyrics "new verse words" styling "jazz"

# Remix with bias (stronger cover)
python voder.py ttm remix "song.wav" styling "jazz" bias 70

# Remix with reference (as-is)
python voder.py ttm remix "song.wav" styling "jazz" reference "ref.wav"

# Remix with reference (extract vocals)
python voder.py ttm remix "song.wav" styling "jazz" reference voice "vocals.wav"

# Remix with reference from URL
python voder.py ttm remix "song.wav" styling "jazz" reference "https://youtube.com/watch?v=..."

# Remix with overdose (after mode name)
python voder.py ttm overdose remix "song.wav" styling "jazz"

# Overdose remix with lyrics
python voder.py ttm overdose remix "song.wav" lyrics "dreamy verse" styling "synthwave"

# Remix vocals only (pre-extract vocals from source)
python voder.py ttm remix voice "song.wav" styling "jazz"

# Remix music only (pre-extract instruments from source)
python voder.py ttm remix music "song.wav" styling "electronic"

# Overdose remix with voice isolation
python voder.py ttm overdose remix voice "song.wav" styling "cinematic orchestral"

# Multi-source remix (vocals + instruments from different songs)
python voder.py ttm remix voice "vocals.wav" music "instruments.wav" styling "funk" bias 60

# Multi-reference remix (2 references)
python voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav"

# Multi-reference remix (3 references)
python voder.py ttm remix "song.wav" styling "rock" reference "ref1.wav" voice "ref2.wav" music "ref3.wav"
```

---

### 3d. Repaint

Re-generate a specific time range of a song in a new style. Supports two modes: **single-pass** (keyword-based, backward compatible) and **multi-pass** (quoted spec format for sequential edits that build on each previous result).

#### Single-Pass Mode (keyword-based)

| Keyword | Value | Description |
|---------|-------|-------------|
| `repaint` | `[voice/music] "<path>"` | Source audio/video file or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL to repaint. Optional `voice`/`music` prefix isolates vocals or instruments via SVS before repainting. |
| `styling` | `"<text>"` | New style prompt for the repainted section. Required. |
| `time:start-end` | `"<start>-<end>"` | Time range in seconds (e.g., `time:20-80` or `time:20.5-80.5`). Required. Supports float values. |
| `lyrics` | `"<text>"` | Optional lyrics for the repainted section. Defaults to `"..."` if omitted. |
| `bias` | `"<0-100>"` | Cover strength bias (same logic as remix). Default: 40. |
| `reference` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Optional reference audio(s). Up to 3 with optional `voice`/`music` prefix per entry. Multiple refs are composed into a 30s composite. Supports URLs and video files. |
| `overdose` | (flag) | Use Overdose tier. |

#### Multi-Pass Mode (quoted spec format)

Each pass is a quoted string containing a time range and optional parameters. Each pass uses the output of the previous pass as its source, enabling creative layering of different styles, references, and lyrics across different time ranges.

**Pass spec format:** `"start-end[/styling(text)][/lyrics(text)][/reference-voice(path)][/reference-music(path)][/reference(path)][/bias/nn]"`

| Component | Format | Description |
|-----------|--------|-------------|
| `start-end` | `"<start>-<end>"` | Time range in seconds. Required. Supports float values. Start must be less than end. |
| `/styling(text)` | `/styling(...)` | Style prompt for this pass. Optional, defaults to `"..."`. |
| `/lyrics(text)` | `/lyrics(...)` | Lyrics for this pass. Optional, defaults to `"..."`. Use `\n` for newlines. |
| `/reference-voice(path)` | `/reference-voice(...)` | Vocal reference (extracted via SVS). Optional. |
| `/reference-music(path)` | `/reference-music(...)` | Instrumental reference (extracted via SVS). Optional. |
| `/reference(path)` | `/reference(...)` | As-is reference (no SVS extraction). Optional. |
| `/bias/nn` | `/bias/nn` | Cover strength 0-100. Optional, default: 40. |

- Up to 3 references per pass; excess entries produce a warning and are trimmed.
- Multiple references in a pass are composed into a 30s composite.
- No limit on the number of passes.
- Re-editing the same time range across passes is expected behavior.
- Paths containing `/` are handled correctly (parenthesis-aware parsing).
- Each pass can have different styling, lyrics, references, and bias.

#### Rules (both modes)

- Start must be less than end. If end exceeds audio duration, it is clamped. If start exceeds duration, it produces an error.
- Cannot be combined with `vc`.
- `voice` prefix on source extracts vocals via SVS before repainting.
- `music` prefix on source extracts instruments via SVS before repainting.
- `reference voice` extracts vocals from the reference via SVS before use.
- `reference music` extracts instruments from the reference via SVS before use.
- `reference "<path>"` uses the reference audio as-is.
- Up to 3 references; excess entries produce a warning and are trimmed.
- Multiple references are composed into a 30s composite (same logic as remix).
- Reference can be a local file, video file, or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL.
- In multi-pass mode, each pass uses the output of the previous pass as its source. The model is loaded once and reused for all passes. Intermediate pass outputs are cleaned up; only the final output is retained.

```
# Repaint 20s-80s of a song
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80

# Repaint with lyrics override
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80 lyrics "new lyrics here"

# Repaint with bias
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80 bias 80

# Repaint with reference (as-is)
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80 reference "ref.wav"

# Repaint with reference (extract vocals)
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80 reference voice "vocals.wav"

# Repaint with multi-reference (2 references)
python voder.py ttm repaint "song.wav" styling "orchestral" time:20-80 reference voice "ref1.wav" music "ref2.wav"

# Repaint with overdose (after mode name)
python voder.py ttm overdose repaint "song.wav" styling "orchestral" time:20-80

# Repaint with voice isolation on source
python voder.py ttm repaint voice "song.wav" styling "funk" time:20-80

# Repaint with music isolation on source
python voder.py ttm repaint music "song.wav" styling "ambient" time:20-80

# Multi-pass: two passes with different styles and bias
python voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"

# Multi-pass: with lyrics and reference per pass
python voder.py ttm repaint "song.wav" "0-30/styling(funk)/lyrics(new words\nhere)" "15-30/styling(ambient)/reference(ref.wav)"

# Multi-pass: overdose with per-pass voice reference
python voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"

# Multi-pass: music isolation on source, with reference-music on second pass
python voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"

# Multi-pass: three passes building on each other
python voder.py ttm repaint "song.wav" "0-30/styling(ambient)" "10-25/styling(jazz)/reference(ref.wav)" "15-30/styling(rock)/bias/90/reference-voice(lead.wav)"
```

---

### 3e. Complete Sub-Task

Add missing instruments to an existing track. Uses ACE-Step XL-Base + 1.7B LM + shift 1.0 (50 inference steps).

| Keyword | Value | Description |
|---------|-------|-------------|
| `complete` | (flag) | Enable complete sub-task. |
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL (positional, after all keywords). |
| `add` | `"<instruments>"` | Instruments to add. See **Instruments Reference** below. Optional if `sfx:` specs are provided. |
| `styling` | `"<text>"` | Optional style prompt to influence the mood and genre of generated instruments (e.g., `"dramatic cinematic"`, `"upbeat pop"`). |
| `noblend` | (flag) | Output the generated instruments only, without blending with the original source audio. Output filename includes `_noblend_`. |
| `voice` | (flag) | Pre-extract vocals from source via SVS before processing. Cannot combine with `music`. |
| `music` | (flag) | Pre-extract music (remove vocals) from source via SVS before processing. Cannot combine with `voice`. |
| `usrc` | (flag) | Blend with original source (before SVS isolation) instead of the isolated voice/music. Only meaningful with `voice` or `music`. Ignored with a warning if used alone. Output filename includes `_usrc_`. |
| `video` | (flag) | Preserve video if source is a video. Merges completed audio back with video. |
| `reference` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Optional reference audio(s). Up to 3 with optional `voice`/`music` prefix per entry. Multiple refs are composed into a 30s composite. |
| `sfx:` | `prompt/duration-position/level` | Sound effect overlay spec. Multiple allowed. See **SFX Overlay Spec** below. |
| `overdose` | (flag) | Valid flag for complete mode. Note: complete mode always uses XL-Base + 1.7B LM + shift 1.0 (50 steps) regardless — the `complete_mode=True` setting overrides model selection. Including `overdose` is not wrong; it serves to identify that the task uses the big model. |

#### Rules

- Requires a source path and `add` with instruments, or `sfx:` specs (at least one must be present).
- `sfx:` cannot be used with `noblend`.
- If only `sfx:` specs are provided (no `add`), the music model is not loaded — SFX is overlaid directly on the source.
- `add` (instruments) and `sfx:` can be combined.
- `voice` and `music` are mutually exclusive. If neither, source is used as-is.
- `usrc` changes the blend target: with `voice`/`music`, the default blend is with the isolated source; `usrc` switches to the original (pre-isolation) source. Without `voice`/`music`, `usrc` is ignored with a warning.
- `noblend` skips the post-generation blend step — the output is the model's generated audio only (no mixing with the original source).
- `video` is valid with `complete` and `bgm` (not lego/extract).
- YouTube URLs: with `video` downloads video file, without downloads audio only.

```
# Add drums and bass to a track
python voder.py ttm complete "vocals_only.wav" add "drums bass"

# Add instruments with a styling prompt for mood/genre control
python voder.py ttm complete "vocals_only.wav" add "drums bass" styling "dramatic cinematic"

# Add all instruments to vocals (voice pre-extract)
python voder.py ttm complete voice "raw_song.wav" add "everything"

# Add everything from music-only source
python voder.py ttm complete music "raw_song.wav" add "everything"

# Complete with video output
python voder.py ttm complete video "song.mp4" add "drums bass guitar"

# Complete with reference
python voder.py ttm complete "vocals_only.wav" add "drums bass" reference voice "ref.wav"

# Complete with styling and reference
python voder.py ttm complete "vocals_only.wav" add "drums bass" styling "upbeat pop" reference "ref.wav"

# Complete with noblend (generated instruments only, no blending with original)
python voder.py ttm complete noblend "vocals_only.wav" add "drums bass"

# Complete from YouTube with video
python voder.py ttm complete video "https://youtube.com/watch?v=..." add "everything"

# SFX overlay only (no instrument generation, no music model loaded)
python voder.py ttm complete "podcast.wav" sfx:thunder/10-5/50

# Combine instruments and SFX overlay
python voder.py ttm complete "vocals_only.wav" add "drums bass" sfx:rain/8-22/40

# Multiple SFX overlays
python voder.py ttm complete "podcast.wav" sfx:thunder/10-5/50 sfx:rain/8-22/40

# Voice isolation + usrc: complete vocals, blend with original source (not just isolated vocals)
python voder.py ttm complete voice usrc "song.wav" add "drums bass guitar"

# Music isolation + usrc: complete instruments, blend with original source
python voder.py ttm complete music usrc "song.wav" add "everything"
```

---

### 3f. Lego Sub-Task

Generate individual instrument tracks from a source. Uses ACE-Step XL-Base + 1.7B LM + shift 1.0 (50 inference steps).

| Keyword | Value | Description |
|---------|-------|-------------|
| `lego` | (flag) | Enable lego sub-task. |
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL (positional). |
| `make` | `"<instruments>"` | Instruments to generate. See **Instruments Reference** below. |
| `styling` | `"<text>"` | Optional style prompt to influence the mood and genre of generated instruments (e.g., `"jazz trio"`, `"ambient electronic"`). |
| `voice` | (flag) | Pre-extract vocals from source via SVS. |
| `music` | (flag) | Pre-extract music from source via SVS. |
| `mix` | (flag) | Mix all generated tracks together into one file. Cannot combine with `blend`. |
| `blend` | (flag) | Mix all generated tracks, then blend the mix with the original source. Cannot combine with `mix`. |
| `reference` | `voice "<path>"` / `music "<path>"` / `"<stem>:<path>"` / `"<path>"` | Multiple references supported. Each reference can be a local audio/video file or URL. See **Lego Reference** below. |

#### Rules

- Requires a source path and `make` with instruments.
- Without `mix` or `blend`, each track is exported as a separate file.
- `mix` and `blend` are mutually exclusive.
- Source and references accept audio files, video files, and YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, and X/Twitter URLs.

```
# Generate individual drum and bass tracks
python voder.py ttm lego "vocals_only.wav" make "drums bass"

# Generate with a styling prompt for mood/genre control
python voder.py ttm lego "vocals_only.wav" make "drums bass" styling "jazz trio"

# Generate and mix into one file
python voder.py ttm lego mix "vocals_only.wav" make "drums bass guitar"

# Generate, mix, and blend with original
python voder.py ttm lego blend "vocals_only.wav" make "everything"

# Lego with voice pre-extract
python voder.py ttm lego voice "raw_song.wav" make "drums bass"

# Lego with single fallback reference
python voder.py ttm lego "source.wav" make "drums bass" reference voice "ref.wav"

# Lego with per-stem references (colon syntax)
python voder.py ttm lego "source.wav" make "drums bass" reference "drums:drums_ref.wav" "bass:bass_ref.wav"

# Lego with fallback + specific stem override
python voder.py ttm lego "source.wav" make "drums bass" reference "fallback.wav" "drums:drums_ref.wav"
```

#### Lego Reference Syntax

In lego mode, `reference` accepts multiple entries. Stem-specific references use **colon syntax** (`stem:path`):

```
reference voice "<path>"          # apply to all tracks (fallback) — extracts vocals
reference music "<path>"          # apply to all tracks (fallback) — extracts instruments
reference "<path>"                # apply to all tracks (fallback, as-is)
reference "drums:<path>"          # apply only to drums track
reference "bass:<path>"           # apply only to bass track
reference "everything:<path>"     # apply to all 12 tracks
reference "instruments:<path>"    # apply to all 10 instrument tracks
reference "voices:<path>"         # apply to vocals + backing_vocals
```

Specific stem references override the fallback. Parsing stops at the next TTM keyword (`mix`, `blend`, `result`, `make`, `add`, `overdose`, `complete`, `lego`, `video`, `extract`, `stems`, `only`, `remix`, `repaint`, `bias`, `vc`, `clone`).

---

### 3g. Extract Sub-Task

Extract/separate individual instrument stems from a source. Uses ACE-Step XL-Base + 1.7B LM + shift 1.0 (50 inference steps).

| Keyword | Value | Description |
|---------|-------|-------------|
| `extract` | (flag) | Enable extract sub-task. |
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL (positional). |
| `stems` | `"<instruments>"` | Instruments to extract. See **Instruments Reference** below. |
| `only` | (flag) | Invert selection: extract everything EXCEPT the specified stems, then mix into one file. Cannot combine with `mix`. |
| `mix` | (flag) | Mix all extracted stems into one file. Cannot combine with `only`. |

#### Rules

- Requires a source path and `stems` with instruments.
- `only` cannot be used with `everything` or all 12 stems (nothing would remain).
- Without `mix` or `only`, each stem is exported as a separate file.

```
# Extract drums and bass as separate files
python voder.py ttm extract "song.wav" stems "drums bass"

# Extract all stems and mix into one file
python voder.py ttm extract mix "song.wav" stems "everything"

# Extract everything except vocals (keep only instruments)
python voder.py ttm extract only "song.wav" stems "vocals backing_vocals"

# Extract everything except drums and bass
python voder.py ttm extract only "song.wav" stems "drums bass"
```

---

### Instruments Reference

Valid stem names for `add`, `make`, `stems`, and `reference` shortcuts:

| Stem Name | Category |
|-----------|----------|
| `drums` | Instrument |
| `bass` | Instrument |
| `guitar` | Instrument |
| `keyboard` | Instrument |
| `percussion` | Instrument |
| `strings` | Instrument |
| `synth` | Instrument |
| `brass` | Instrument |
| `woodwinds` | Instrument |
| `fx` | Instrument |
| `vocals` | Voice |
| `backing_vocals` | Voice |

**Shortcuts:**

| Shortcut | Expands To |
|----------|------------|
| `everything` | All 12 stems |
| `instruments` | All 10 instrument stems (non-vocal) |
| `voices` | `vocals` + `backing_vocals` |

```
# Examples
add "drums bass guitar"     # specific instruments
make "everything"           # all 12 tracks
stems "instruments"         # all 10 non-vocal stems
stems "voices"              # vocals + backing_vocals
```

---

### 3h. BGM Sub-Task

Replace background music in an existing audio or video file. Strips existing music via SVS voice pipe, generates new background music via ACE-Step, and mixes at a configurable volume.

| Keyword | Value | Description |
|---------|-------|-------------|
| `bgm` | `"<path>"` | Source audio/video file or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL whose background music will be replaced. |
| `music` | `"<description>"` | Description for the new background music to generate. Optional if `sfx:` specs are provided. |
| `level` | `<0-100>` | Music volume level (0 = silent, 100 = full volume). Default: 35. |
| `video` | (flag) | Preserve video output. When source is a URL, downloads the video file and merges result back into .mp4. For local video files, video output is automatic (no flag needed). |
| `reference` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Optional reference audio(s). Up to 3 with optional `voice`/`music` prefix per entry. Multiple refs are composed into a 30s composite. |
| `sfx:` | `prompt/duration-position/level` | Sound effect overlay spec. Multiple allowed. See **SFX Overlay Spec** below. |
| `overdose` | (flag) | Use Overdose tier (ACE-Step XL-Turbo + 4B LM + shift 3.0) instead of Standard tier (ACE-Step 1.5 Turbo). |

#### Rules

- `bgm` requires `music` and/or `sfx:` specs (at least one must be present).
- SFX is overlaid after BGM mixing (or directly on voice if no music description).
- `bgm` cannot be combined with `vc`, `remix`, `repaint`, `complete`, `lego`, or `extract`.
- Source is resolved through `resolve_target_to_audio()` — supports audio files, video files, and URLs.
- `video` flag: when source is a YouTube URL, downloads the video file (not just audio) and merges the result back into .mp4. For local video files, video output is automatic. If `video` is used with an audio source, outputs .wav with a warning.
- Reference supports audio files, video files, and URLs. Up to 3 references are composed into a 30s composite.
- Video inputs produce `.mp4` output with the new audio re-muxed; audio inputs produce `.wav`.
- Output naming: `voder_ttm_bgm_{original-name}_{timestamp}.wav` (audio) or `.mp4` (video).
- Normal (non-overdose) uses ACE-Step turbo 1.5 model.
- Overdose uses ACE-Step XL 1.5 turbo model.

```
# Replace background music (standard quality)
python voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30

# Replace background music with higher volume
python voder.py ttm bgm "video.mp4" music "cinematic orchestral" level 50

# Replace background music with overdose quality
python voder.py ttm overdose bgm "podcast.mp4" music "jazz lounge" level 40

# Replace background music with reference for style guidance
python voder.py ttm bgm "podcast.wav" music "upbeat electronic" level 35 reference "path/to/style_ref.wav"

# From YouTube URL with audio-only output
python voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient chill" level 25 reference "ref.wav"

# From YouTube URL with video output (downloads video, replaces bgm, outputs .mp4)
python voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"

# BGM with SFX overlay
python voder.py ttm bgm "podcast.wav" music "soft ambient piano" level 30 sfx:thunder/10-5/50

# BGM with multiple SFX overlays
python voder.py ttm bgm "podcast.wav" music "ambient chill" level 25 sfx:rain/8-22/40 sfx:thunder/10-5/50

# SFX overlay only (no BGM, overlaid directly on voice)
python voder.py ttm bgm "podcast.wav" sfx:rain/8-22/40
```

---

### SFX Overlay Spec

The `sfx:` keyword is available in `complete` and `bgm` sub-tasks. It overlays one or more generated sound effects onto the source audio.

#### Format

```
sfx:prompt/duration-position/level
```

| Part | Required | Description |
|------|----------|-------------|
| `prompt` | Yes | SFX description text (e.g., `thunder`, `rain on a tin roof`). Slash (`/`) must not appear in the prompt. |
| `duration` | Yes | SFX length in seconds (5-30). Auto-clamped: less than 5 becomes 5; greater than 30 becomes 30 with a warning. Minus signs are stripped. |
| `position` | Yes | Place the SFX at N seconds into the source audio (in seconds). Cannot be negative. Cannot exceed source duration. |
| `level` | No | Volume 1-100%. Default: 50. Minus signs are stripped. Less than 1 produces a warning and is set to 1. Greater than 100 produces a warning and is set to 100. |

#### Notes

- Multiple `sfx:` specs can be specified on a single command.
- Invalid duration/position/level format (non-numeric, missing required parts) produces an error and stops execution.

#### Valid Examples

```
sfx:thunder/10-5/50       # 10-second thunder at 5s into source, 50% volume
sfx:rain/8-22             # 8-second rain at 22s into source, default 50% volume
sfx:boom/12-30/40         # 12-second boom at 30s into source, 40% volume
```

#### Invalid Examples

```
sfx:thunder/5             # Missing position (duration-position required)
sfx:thunder/-10-5/50      # Negative duration (minus sign stripped → 10-5/50, valid)
sfx:thunder/10--5/50      # Negative position (error: position cannot be negative)
sfx:thunder/abc-5/50      # Non-numeric duration (error)
sfx:thunder/10-5/0        # Level below 1 (warning → clamped to 1)
sfx:thunder/10-5/200      # Level above 100 (warning → clamped to 100)
```

---

## 4. `stt` — Speech-to-Text (Transcription)

Transcribe audio/video to text using Whisper.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `"<path>"` | file | Audio/video/image file path or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL. Can specify multiple files (each is transcribed separately). |
| `timestamp` | (flag) | Keep Whisper word-level timestamps in the output. |
| `dialogue` | (flag) | Enable speaker diarization (requires HF_TOKEN and pyannote model access). |
| `translate` | (flag) | Translate transcription to English (uses Whisper large-v3 model). |
| `translate (source-target)` or `translate (target)` | `(auto-en)` / `(ja-en)` / `(ar)` etc. | Any-to-any translation via TranslateGemma 12B (76 languages). Use `auto` for source auto-detection. `(target)` is shorthand for `(auto-target)`. Compatible with `overdose` and `subtitle`. |
| `se` | (flag) | Apply sound enhancement before transcription (denoise/dereverb input first). |
| `overdose` | (flag) | Use VibeVoice ASR (requires 24GB+ VRAM or 48GB+ RAM). Falls back to Whisper + pyannote if unavailable. |
| `subtitle` | (flag) | Burn VibeVoice ASR subtitles onto video (auto‑implies `overdose`; video/URL only; no `translate`). Use `stt overdose subtitle` for clarity. |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- `overdose` cannot be combined with bare `translate` (without parentheses).
- `translate (source-target)` or `translate (target)` is compatible with `overdose` — TranslateGemma decouples translation from ASR.
- `subtitle` auto‑implies `overdose`, so `stt subtitle` and `stt overdose subtitle` are equivalent (explicit form recommended for clarity); cannot combine with bare `translate`; only accepts video files/URLs.
- Multiple files are processed sequentially.
- Output is saved as `.txt` in the `results/` directory.
- **Pipeline:** SVS voice isolation is always applied before transcription. With `se`, sound enhancement runs first.
- **Image input:** Also accepts image files (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.webp`) — runs EasyOCR text extraction, then transcribes the extracted text.

```
# Basic transcription
python voder.py stt "audio.wav"

# Multiple files
python voder.py stt "audio1.wav" "audio2.wav"

# With timestamps
python voder.py stt "audio.wav" timestamp

# With speaker diarization
python voder.py stt "audio.wav" dialogue

# Timestamps + diarization combined
python voder.py stt "audio.wav" timestamp dialogue

# Translate to English
python voder.py stt "audio.wav" translate

# Any-to-any translation (TranslateGemma 12B)
python voder.py stt "audio.wav" translate "(auto-ar)"

# Shorthand: (ar) is equivalent to (auto-ar)
python voder.py stt "audio.wav" translate "(ar)"

# Translate Japanese to English
python voder.py stt "audio.wav" translate "(ja-en)"

# Overdose + any-to-any translation (compatible)
python voder.py stt "audio.wav" overdose translate "(auto-fr)"

# Full combination
python voder.py stt "audio.wav" translate timestamp dialogue

# From YouTube
python voder.py stt "https://youtube.com/watch?v=..."

# Subtitle sub-task: burn subtitles onto video
python voder.py stt overdose subtitle "video.mp4"

# Subtitle with sound enhancement
python voder.py stt overdose subtitle se "noisy_video.mp4"

# Subtitle from YouTube URL
python voder.py stt overdose subtitle "https://youtube.com/watch?v=..."

# Subtitle with any-to-any translation (auto-detect source, translate to Arabic)
python voder.py stt overdose subtitle translate "(auto-ar)" "video.mp4"

# Shorthand: (ar) is equivalent to (auto-ar)
python voder.py stt overdose subtitle translate "(ar)" "video.mp4"

# Subtitle with Japanese-to-English translation
python voder.py stt overdose subtitle translate "(ja-en)" "japanese_video.mp4"
```

---

## 5. `se` — Sound Enhancement

Improve audio quality through denoising, dereverberation, restoration, and super-resolution.

### Keywords

| Keyword | Type | Description |
|---------|------|-------------|
| `voice` | sub-mode | SVS voice extraction + UniSE enhancement on vocals only |
| `sr` | sub-mode | AudioSR super-resolution on input audio (48kHz output) |
| `music` | modifier | After `sr`: apply AudioSR to non-vocals only (requires SVS) |
| `voice` | modifier | After `sr`: apply AudioSR speech model to vocals only (requires SVS) |
| `blend` | modifier | Blend enhanced/upsampled audio with complementary stem |
| `result` | keyword | Custom output path |
| `video` | flag | When source is a URL, download the full video (default: audio download). Output is MP4 with enhanced audio muxed back. |
| `"<path>"` | file | Audio/video file path or URL (multiple allowed) |

### Sub-Mode Combinations

| Command | Pipeline | Output |
|---------|----------|--------|
| `se "path"` | UniSE → enhanced audio | 16kHz WAV |
| `se voice "path"` | SVS voice → UniSE | 16kHz WAV |
| `se voice blend "path"` | SVS voice+music → UniSE on voice → blend | 48kHz WAV |
| `se sr "path"` | AudioSR (basic model) on whole input | 48kHz WAV |
| `se sr music "path"` | SVS voice+music → AudioSR (basic) on music | 48kHz WAV |
| `se sr music blend "path"` | SVS voice+music → AudioSR on music + UniSE on voice → blend | 48kHz WAV |
| `se sr voice "path"` | SVS voice → AudioSR (speech model) on vocals | 48kHz WAV |
| `se sr voice blend "path"` | SVS voice+music → AudioSR (speech) on vocals → blend with music | 48kHz WAV |
| `se sr voice music "path"` | SVS voice+music → AudioSR speech on vocals + basic on music → auto-blend | 48kHz WAV |

### Examples

```
# Default: UniSE enhancement on audio
python voder.py se "noisy_audio.wav"

# Enhance multiple files
python voder.py se "audio1.wav" "audio2.wav"

# Enhance video audio track
python voder.py se "noisy_video.mp4"

# Enhance from URL (audio downloaded by default)
python voder.py se "https://youtube.com/watch?v=..."

# Enhance from URL and output MP4 (video downloaded, audio enhanced, muxed back)
python voder.py se video "https://youtube.com/watch?v=..."

# Voice sub-mode: extract vocals then enhance
python voder.py se voice "song.wav"

# Voice + blend: enhance vocals and mix back with music at 48kHz
python voder.py se voice blend "song.wav"

# SR sub-mode: super-resolution on whole input (basic model, 48kHz output)
python voder.py se sr "speech.wav"

# SR + music: separate vocals, apply AudioSR (basic) to music stem
python voder.py se sr music "song.wav"

# SR + music + blend: AudioSR on music, UniSE on voice, blend at 48kHz
python voder.py se sr music blend "song.wav"

# SR + voice: extract vocals, apply AudioSR speech model for voice SR
python voder.py se sr voice "vocals.wav"

# SR + voice + blend: AudioSR speech on vocals, blend with music
python voder.py se sr voice blend "song.wav"

# SR + voice + music: AudioSR speech on vocals + basic on music, auto-blend
python voder.py se sr voice music "song.wav"
```

### Notes

- Default (no sub-mode): UniSE enhancement, outputs 16kHz, designed for speech
- `voice`: Extract vocals via SVS first, then enhance with UniSE
- `sr`: AudioSR super-resolution with basic model, outputs 48kHz
- `sr music`: Uses AudioSR basic model (general audio) on non-vocal stems
- `sr voice`: Uses AudioSR speech model on vocal stems for voice-optimized SR
- `sr voice music`: AudioSR speech model on vocals + basic model on music, auto-blended at 48kHz
- `blend`: Mixes processed stems at the highest available sample rate (48kHz)
- Video input: `.mp4` output with enhanced audio track (default mode only)

---

## 6. `sfx` — Sound Effects Generation

Generate sound effects from text prompts using TangoFlux.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `sound` | `"<prompt>"` | Text prompt describing the sound effect. Required. |
| `duration` | `<1-30>` | Duration in seconds. Required. Clamped to 30 if higher. |
| `steps` | `<1-100>` | Inference steps. Default: 30. |
| `guide` | `<1.0-10.0>` | Guidance scale. Rounds to nearest 0.5. Default: 4.5. |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- `sound` and `duration` are required.
- `steps` and `guide` are optional with defaults.

```
# Basic sound effect
python voder.py sfx sound "thunder cracking" duration 5

# With result path
python voder.py sfx sound "rain on a tin roof" duration 10 result "output.wav"

# Custom inference steps
python voder.py sfx sound "rain on a tin roof" duration 10 steps 50

# Full control
python voder.py sfx sound "rain on a tin roof" duration 10 steps 50 guide 3.5 result "output.wav"
```

---

## 7. `svs` — Song Voice Separate

Extract vocals or instruments from a song using BS-RoFormer.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `voice` | (flag) | Extract vocals (remove instruments). |
| `music` | (flag) | Extract instruments (remove vocals). |
| `both` | (flag) | Extract both vocals and instruments (runs two separations, outputs two files). |
| `"<path>"` | file | Audio/video file path or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL. |
| `result` | `"<path>"` | Copy output to custom path. |
| `video` | (flag) | When source is a URL, download the full video (default: audio download). Output is MP4 with separated stem muxed back, one per stem. |

### Rules

- At least one of `voice`, `music`, or `both` is required.
- `both` extracts both vocals and instruments, producing two output files.
- Video input: outputs `.mp4` with separated audio merged back.
- Audio input: outputs `.wav`.
- YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, and X/Twitter URLs: default downloads **audio** and outputs `.wav`; add the `video` flag to download the full video and output `.mp4` (one per stem).

```
# Extract vocals
python voder.py svs voice "song.mp3"

# Extract instruments
python voder.py svs music "song.mp3"

# Extract both vocals and instruments
python voder.py svs both "song.mp3"

# With result path
python voder.py svs voice "song.mp3" result "output.wav"

# Video input
python voder.py svs voice "music_video.mp4"

# From YouTube (audio downloaded by default)
python voder.py svs music "https://youtube.com/watch?v=..."

# From YouTube and output MP4 (video downloaded, separated stem muxed back)
python voder.py svs music video "https://youtube.com/watch?v=..."
```

---

## 8. `ss` — Speakers Separator

Extract all speakers from an audio source one by one (one file per detected speaker), or extract a specific speaker by number, or extract a target speaker using a reference audio. With `blend`, each separated speaker's audio is mixed with the original non-vocals (instrumental/background) track. With `video`, separated audio is muxed with the original video to produce video output.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `<N>` | number | **Optional** for blind SS (no `target`). When omitted, all detected speakers are extracted one by one (one file per speaker). When provided, only the requested speaker is extracted and the pipeline stops after that speaker. Resolution: `1` = first speaker (by diarization order), `N` = Nth speaker, `999` (or any number higher than actual count) = last speaker, `0` resolves to `1`. Must be a non-negative integer; non-numeric values produce an error. Ignored when `target` is provided. |
| `"<path>"` | file | Audio/video file path or YouTube/TikTok/Bilibili/Snapchat/Instagram/Facebook/X-Twitter URL. |
| `target` | `"<path>"` | Target voice reference audio/URL. When provided, extracts only the speaker matching this reference from the source audio. Outputs a single file. Speaker number is not needed with `target`. |
| `se` | (flag) | Apply sound enhancement before separation (denoise/dereverb the input first). |
| `overdose` | (flag) | Use VibeVoice ASR with forced-alignment refinement for transcription and diarization, providing better separation accuracy. The overdose path uses the same aligner + multi-level extraction as the TTS dub pipeline: after initial TSE extraction, word-level forced alignment identifies non-overlapping speech for refined enrollment clips, producing significantly cleaner speaker isolation. Requires 24GB+ VRAM or 48GB+ RAM. **Skipped when `target` is provided** (target uses TSE extraction, not diarization). |
| `blend` | (flag) | Blend the separated speaker's audio with the original non-vocals (instrumental/background) track. After speaker extraction (and optional SE), the output is mixed with the music/instrumental stem extracted via SVS. Output carries a `_blend` suffix. Useful for vlogs or recordings where you want to isolate a speaker while preserving background audio. |
| `video` | (flag) | Produce video output. When the input is a video file or URL, the separated speaker's audio is muxed with the original video frames to produce MP4 output. Ignored when the input is an audio-only file. Works with both target and blind modes. Also works with `blend` (muxed video contains blended audio). |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- **Pipeline (blind SS, no target, no number):** SVS voice isolation → STT + diarization → TSE extraction of every detected speaker, one by one. Outputs one file per speaker (`voder_ss_<name>_<ts>_speaker1.wav`, `_speaker2.wav`, …).
- **Pipeline (blind SS, no target, with `<N>`):** SVS voice isolation → STT + diarization → TSE extraction of the requested speaker only. Outputs exactly one file.
- **Pipeline (with target):** SVS voice isolation → TSE (Target Speaker Extraction). Looks at the target reference and extracts matching speaker from source. Outputs one file. No speaker number needed (a number is ignored if provided).
- **Pipeline (overdose, blind SS):** SVS voice isolation → VibeVoice ASR + forced-alignment multi-level extraction → TSE with aligned enrollment. The aligner provides word-level timestamps; overlap regions are filtered out; refined enrollment clips are cut from non-overlapping aligned speech for significantly better isolation. When `<N>` is provided, outputs one file for that speaker; when omitted, runs the alignment-refined pass for every speaker and outputs one file per speaker.
- **Pipeline (with blend):** SVS voice isolation (+ music extraction) → TSE extraction → blend speaker with non-vocals. Output file has `_blend` suffix.
- **Pipeline (with video):** Same as standard pipeline, then mux output audio with the original video frames. Output is MP4 instead of WAV. Ignored for audio-only inputs.
- `overdose` is only used in the blind SS pipeline (switches from pyannote to VibeVoice ASR + forced alignment for better accuracy). It is completely skipped when `target` is provided.
- `blend` works with both target and blind modes, and with `se`, `overdose`, and `video`.
- `video` works with both target and blind modes, and with `se`, `overdose`, and `blend`.
- Supports audio, video, and URLs from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter).
- `se` runs sound enhancement before anything else (cleaner input = better results).

```
# Extract ALL speakers one by one (one file per detected speaker)
python voder.py ss "conversation.wav"

# Extract ALL speakers from a video
python voder.py ss "interview.mp4"

# Extract ALL speakers from a URL
python voder.py ss "https://youtube.com/watch?v=..."

# Extract first speaker only (standard pipeline)
python voder.py ss 1 "conversation.wav"

# Extract last speaker (number resolves to last detected)
python voder.py ss 999 "conversation.wav"

# Extract specific speaker number
python voder.py ss 3 "conversation.wav"

# From video, speaker 2
python voder.py ss 2 "interview.mp4"

# From YouTube, first speaker
python voder.py ss 1 "https://youtube.com/watch?v=..."

# With sound enhancement pre-processing (single speaker)
python voder.py ss se 1 "noisy_conversation.wav"

# With sound enhancement, extract ALL speakers
python voder.py ss se "noisy_conversation.wav"

# With overdose (better accuracy, uses VibeVoice ASR + forced alignment)
python voder.py ss overdose 1 "conversation.wav"

# Overdose, extract ALL speakers (alignment-refined pass per speaker)
python voder.py ss overdose "conversation.wav"

# Overdose with specific speaker
python voder.py ss overdose 3 "conversation.wav"

# With blend (speaker + non-vocals), single speaker
python voder.py ss blend 1 "vlog.wav"

# With blend, extract ALL speakers (each speaker blended with non-vocals)
python voder.py ss blend "vlog.wav"

# Extract specific target speaker from source (outputs one file, no number needed)
python voder.py ss target "speaker_ref.wav" "conversation.wav"

# Target extraction with blend (target speaker + non-vocals)
python voder.py ss target "speaker_ref.wav" blend "conversation.wav"

# Target extraction from URL
python voder.py ss target "speaker_ref.wav" "https://youtube.com/watch?v=..."

# With video output (mux separated audio with original video), single speaker
python voder.py ss video 1 "interview.mp4"

# With video output, extract ALL speakers (one MP4 per speaker)
python voder.py ss video "interview.mp4"

# Target extraction with video output
python voder.py ss target "speaker_ref.wav" video "interview.mp4"

# Video output from URL
python voder.py ss video 1 "https://youtube.com/watch?v=..."

# Full pipeline: overdose + sound enhancement + blend + video, speaker 1
python voder.py ss overdose se blend video 1 "vlog.mp4"

# Full pipeline, extract ALL speakers
python voder.py ss overdose se blend video "vlog.mp4"
```

---

## Tasks & Features (beyond the 8 modes)

The 8 main processing modes (TTS, STS, TTM, STT, SE, SFX, SVS, SS) are covered in sections 1–8 above. The remaining sections cover three task-layer features: Voice Training (`train`) saves reusable voice clones for use in TTS; Side-Quests (`quest`) provide lightweight utility tasks (URL download, audio format conversion, cutting, merging, audio effects, and more); Chains (`chains`) compose any number of voder oneline tasks into user-defined pipelines.

---

## 9. `quest` — Side-Quests

> **Note:** `quest` performs small utility tasks (URL download, audio format conversion, cutting, merging, audio effects, etc.) that produce files for the main modes to consume.

Side-quests are lightweight utility tasks that live outside the voder engine. Each quest is a small class registered in a `SIDE_QUESTS` registry; new quests can be added over time without touching the dispatcher.

### Syntax

```
python voder.py quest <quest-name> [quest args...] [result "<path>"]
```

### Available quests

Side-quests are grouped by category in the `quest` listing (run `python voder.py quest` with no args to see the live tree). The **Media Discovery** category contains the two fetch utilities (`download` and `media-search`); the **Media Manipulation** category contains the other 17 quests, split into three sub-categories — **Sound Effects**, **Audio Editing**, and **Format & File**. Categorization is defined externally in `src/voders/quests_categories.py`, not on the quest classes themselves. The grouping is purely organizational — every side-quest is still called by its unique name (`quest <name> ...`), with no prefix.

| Quest | Sub-category | Description | Output naming |
|-------|--------------|-------------|---------------|
| `download` | Media Discovery | Download a URL as audio (default), video (`video` keyword), or image (`image` keyword). Auto-detects image/video URLs by extension. Also accepts local file paths (copies them). | `voder_quest_download_<original-name>_<timestamp>.<ext>` |
| `media-search` | Media Discovery | Search media across platforms via yt-dlp (default, video/audio) or gallery-dl (with `image` keyword, images). Multi-platform via slash-separated list. Writes a results list file to `results/downloads/others/`. | `voder_quest_media-search_<engine>_<platforms>_<query>_<timestamp>.txt` |
| `noframes` | Format & File | Extract audio from a LOCAL VIDEO file. Refuses URLs and audio-only files. | `voder_quest_noframes_<original-name>_<timestamp>.wav` |
| `convert` | Format & File | Convert a local audio file to any other audio format (40+ formats). Same-format just copies. | `voder_quest_convert_<name>_<timestamp>.<format>` |
| `compress` | Format & File | Compress an audio file at level 1 (low), 2 (default), or 3 (highest). | `voder_quest_compress_L<level>_<name>_<timestamp>.<ext>` |
| `glue` | Format & File | Glue an audio file onto a video file (or vice versa). Auto-replaces existing audio; pads silence / black frames to match longer stream. Refuses URLs and same-type pairs. | `voder_quest_glue_<audio>_onto_<video>_<timestamp>.mp4` |
| `cut` | Audio Editing | Extract a time range from a local audio/video file as a WAV. | `voder_quest_cut_<name>_<start>s-<end>s_<timestamp>.wav` |
| `remove` | Audio Editing | Inverse of `cut`: remove one or more time ranges from a local audio/video file, keeping the rest. Multi-range supported; overlapping ranges are merged. | `voder_quest_remove_<name>_<ranges>_<timestamp>.{wav,mp4}` |
| `merge` | Audio Editing | Concatenate two or more local audio files end-to-end (no upper limit). | `voder_quest_merge_<joined-names>_<timestamp>.wav` |
| `mix` | Audio Editing | Overlay multiple audio/video sources at specified start times into a single WAV. First source is the base (starts at 0s); subsequent sources can have an optional start time in seconds before them. Audio is extracted from video files. Accepts local paths and URLs. | `voder_quest_mix_<joined-names>_<timestamp>.wav` |
| `silence` | Audio Editing | Strip silent gaps from a local audio/video file → continuous-speech WAV. | `voder_quest_silence_<name>_<timestamp>.wav` |
| `reverse` | Audio Editing | Reverse a local audio OR video file (frames + audio both flipped for video). | `voder_quest_reverse_<name>_<timestamp>.{wav,mp4}` |
| `fade` | Sound Effects | Apply a cinematic 5s fade-in/out (not silence-based; rising gain). | `voder_quest_fade_<name>_<timestamp>.{wav,mp4}` |
| `soundlevel` | Sound Effects | Linear sound-level multiplier on a 0.01–10.00 scale (1.00 = original, 0.25 = 25%, 2.00 = 2× louder, 10.00 = 10× louder). Affects all frequencies equally. No EQ, no compression, no loudness normalization. | `voder_quest_soundlevel_x<value>_<name>_<timestamp>.{wav,mp4}` |
| `bassboost` | Sound Effects | Professional multi-band bass booster (low frequencies only) on a 1–100 scale (1 = subtle warmth, 100 = +24 dB sub-destroyer). Mids and highs left untouched. | `voder_quest_bassboost_v<value>_<name>_<timestamp>.{wav,mp4}` |
| `speed` | Sound Effects | Professional time-stretch (rubberband, formant-preserved) on a 0.25–10.00 scale. Audio files only. | `voder_quest_speed_x<value>_<name>_<timestamp>.wav` |
| `pitch` | Sound Effects | Professional pitch shift (rubberband, formant-shifted) on a 0.01–10.00 scale. Audio output only. Accepts local audio / video / URL. | `voder_quest_pitch_p<value>_<name>_<timestamp>.wav` |
| `reverb` | Sound Effects | Professional Schroeder-style reverb (early reflections + late-reverb tail + pre-delay + air-absorption damping + dynamic normalization + true-peak limiter) on a 1–100 scale. Audio output only. Accepts local audio / video / URL. | `voder_quest_reverb_r<value>_<name>_<timestamp>.wav` |
| `loudnorm` | Sound Effects | EBU R128 perceptual loudness normalization. Analyzes the file, then applies a linear normalization so the whole signal sits at one consistent perceived level (-16 LUFS target, -1.5 dB true-peak limit). No quality loss, no dynamic-range compression. Audio and video supported. | `voder_quest_loudnorm_<name>_<timestamp>.{wav,mp4}` |

### 9.1 `download`

| Argument | Description |
|----------|-------------|
| `video` | (optional) Switch to a full video download instead of audio. |
| `image` | (optional) Switch to an image download via gallery-dl. |
| `"<url>"` or `"<path>"` | A URL from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter) or a local file path. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- URL input: downloads via yt-dlp (audio/video) or gallery-dl (images). Audio path uses `download_url_audio` (MP3 @ 192 kbps); video path uses `download_url_video` (MP4, best quality); image path uses `download_url_image` (gallery-dl, original format). The URL is verified by the universal URL handler before downloading. Downloads that fail without cookies are automatically retried with Chrome → Brave → Edge cookies.
- **Auto-detection**: when no `video`/`image` keyword is given and the URL ends in a known image extension (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.bmp`, `.tiff`, `.svg`) it is auto-routed to the image path; URLs ending in a known video extension (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.flv`, `.m4v`, `.wmv`, `.ts`, `.mts`, `.3gp`) are auto-routed to the video path. This covers direct file links (e.g., `https://example.com/poster.jpg`) where the format is in the URL.
- For URLs without a recognizable extension (most social-media URLs — Instagram posts, Reddit posts, etc.), the default audio path is attempted; if it fails, the error message suggests the right keyword to use (`image` or `video`).
- Local file input: copies the file to `results/downloads/<type>/` with the quest naming scheme (no re-encoding).
- The `<original-name>` is derived from the platform video ID (for URLs — e.g. YouTube video ID, TikTok video ID, Bilibili BV id, Instagram reel id, Facebook video id, Twitter status id, Reddit post id) or the file's stem (for local files), sanitized to safe filename characters and capped at 40–60 characters.
- Supported platforms: YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter, Reddit. Experimental `public_net` support for other sites (attempted via yt-dlp/gallery-dl with a warning — works if the tool supports the site, but untested).
- Output locations: audio → `results/downloads/audios/`, video → `results/downloads/videos/`, image → `results/downloads/images/`.

```
# Download a YouTube URL as audio (default, MP3)
python voder.py quest download "https://youtube.com/watch?v=..."

# Download the same URL as video (MP4)
python voder.py quest download video "https://youtube.com/watch?v=..."

# Download an image (or image gallery) from Reddit/Instagram/X/etc.
python voder.py quest download image "https://reddit.com/r/.../comments/..."

# Direct image URL — auto-detected, no keyword needed
python voder.py quest download "https://example.com/poster.jpg"

# Direct video URL — auto-detected, no keyword needed
python voder.py quest download "https://example.com/clip.mp4"

# Copy a local file to results/downloads/ with the quest naming scheme
python voder.py quest download "/path/to/local.wav"
python voder.py quest download "/path/to/local.mp4"

# Save result to a specific path
python voder.py quest download "https://youtube.com/watch?v=..." result "./out.mp3"
python voder.py quest download video "https://youtube.com/watch?v=..." result "./out.mp4"
python voder.py quest download image "https://reddit.com/..." result "./out.jpg"
```

### 9.2 `media-search`

| Argument | Description |
|----------|-------------|
| `image` | (optional, first position) Switch the search engine from yt-dlp (default — video/audio platforms) to gallery-dl (image platforms). |
| `<platform(s)>` | One platform name, or multiple platform names separated by `/` (e.g., `youtube`, `youtube/reddit`, `pixiv/danbooru/gelbooru`). Platform names use letters, digits, hyphen, underscore only. Unknown names are treated as `public_net` best-effort (yt-dlp or gallery-dl will try to scrape `https://<platform>.com/search?q=<query>`). |
| `"<query>"` | The search query. Quote it if it contains spaces. |
| `<count>` | (optional) Maximum results per platform. Integer 1–100. Default 20. Out-of-range or non-integer values error out. |
| `result "<path>"` | (optional) Copy the results list file to a custom path. |

**Behavior:**

- Engine selection:
  - **yt-dlp** (default, no `image` keyword): searches video/audio platforms. For YouTube/Reddit/Bilibili it uses native yt-dlp search prefixes (`ytsearch{N}:`, `redditsearch{N}:`, `bilisearch{N}:`). For TikTok/Snapchat/Instagram/Facebook/X it builds the platform's search URL and lets yt-dlp scrape it. For unknown platform names, it builds `https://<platform>.com/search?q=<query>` and attempts a public_net scrape.
  - **gallery-dl** (`image` keyword): searches image platforms by building the platform's tag/search URL (e.g., `instagram.com/explore/tags/<tag>/`, `pixiv.net/en/tags/<tag>/artworks`, `danbooru.donmai.us/posts?tags=<tag>`, `tumblr.com/search/<query>`, `wallhaven.cc/search?q=<query>`, etc.) and running `gallery-dl -j --simulate` to dump JSON metadata without downloading the images.
- Multi-platform: when `<platform(s)>` contains slashes, the search is run **per platform**, with `<count>` applied as the per-platform cap. Duplicates are removed. Results from all platforms are combined into a single list file.
- Cookies retry: if a search returns no results or fails with a login/auth error, it is retried with Chrome → Brave → Edge cookies (same mechanism as `quest download`). This is genuinely useful for login-walled content (Instagram hashtag search, Pixiv tag search for some content, etc.).
- Output: a single text list file at `results/downloads/others/voder_quest_media-search_<engine>_<platforms>_<query>_<timestamp>.txt` containing per-platform summary + entry details (title, URL, extractor, type, duration/dimensions).
- **No results = no file**: if every platform returns zero results, no list file is created. A per-platform summary is printed to the terminal so the user can see which platforms failed and why.

**Supported yt-dlp platforms** (built-in search): `youtube`, `reddit`, `bilibili`, `tiktok`, `snapchat`, `instagram`, `facebook`, `twitter`/`x`. Any other name is attempted as `public_net`.

**Supported gallery-dl platforms** (built-in search URL): `instagram`, `pixiv`, `danbooru`, `gelbooru`, `yandere`, `konachan`, `reddit`, `twitter`/`x`, `flickr`, `pinterest`, `artstation`, `deviantart`, `tumblr`, `wallhaven`, `unsplash`, `behance`, `500px`, `imgur`, `vk`, `weibo`. Any other name is attempted as `public_net`.

```
# Search YouTube for 10 lofi tracks (yt-dlp, default engine)
python voder.py quest media-search youtube "lofi study mix" 10

# Search Instagram for cyberpunk art images (gallery-dl, image keyword)
python voder.py quest media-search image instagram "cyberpunk art" 15

# Search YouTube AND Reddit in one go (5 results each = up to 10 total)
python voder.py quest media-search youtube/reddit "asmr cooking" 5

# Search 3 image platforms at once (50 results each)
python voder.py quest media-search image pixiv/danbooru/gelbooru "blue hair" 50

# Default count is 20 if no number is given
python voder.py quest media-search youtube "lofi beats"

# Unknown platform name — tried as public_net (best-effort)
python voder.py quest media-search somesite "interesting query" 10

# Save the results list to a custom path
python voder.py quest media-search youtube "news" 20 result "./results.txt"

# Errors:
# python voder.py quest media-search youtube "q" 0       # ERROR: count must be 1-100
# python voder.py quest media-search youtube "q" 101     # ERROR: count must be 1-100
# python voder.py quest media-search youtube "q" abc     # ERROR: invalid count
# python voder.py quest media-search youtube$bad "q"     # ERROR: invalid platform name
```

**Pair with `quest download`:** once you have a URL from the results list file, fetch it with `quest download "<url>"` (audio), `quest download video "<url>"` (video), or `quest download image "<url>"` (image).

### 9.3 `noframes`

| Argument | Description |
|----------|-------------|
| `"<local_video>"` | A LOCAL video file path. URLs and audio files are refused. |
| `result "<path>"` | (optional) Copy the result WAV to a custom path. |

**Behavior:**

- Strictly a "video → audio" extractor for local video files.
- Refuses URLs (use `quest download` first to fetch a URL).
- Refuses files whose extension is not a video format (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.wmv`, `.m4v`).
- Output is always WAV (PCM 16-bit, 44.1 kHz, stereo) extracted via FFmpeg.
- The `<original-name>` is the video file's stem, sanitized to safe filename characters.

```
# Extract audio from a local MP4
python voder.py quest noframes "video.mp4"

# Save result to a specific path
python voder.py quest noframes "video.mp4" result "./out.wav"

# Refused inputs:
# python voder.py quest noframes "https://youtube.com/watch?v=..."   # ERROR: refuses URLs
# python voder.py quest noframes "audio.wav"                          # ERROR: refuses non-video files
```

### 9.4 `convert`

| Argument | Description |
|----------|-------------|
| `<format>` | Target audio format (e.g., `mp3`, `wav`, `flac`, `ogg`, `opus`, `aac`, `m4a`, `wma`, `aiff`, `ac3`, `amr`, `au`, `gsm`, `tta`, `wv`, `ape`, `mpc`, `mp2`, `mka`, `caf`, `dsf`, `dff`, `sph`, `sln`, `raw`, ...). Case-insensitive; leading dot optional. |
| `"<input>"` | A LOCAL audio file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Universal audio format converter — supports 40+ formats including the weird ones (`opus`, `ogg`, `oga`, `gsm`, `tta`, `wv`, `ape`, `mpc`, `caf`, `dsf`, `dff`, `sph`, `sln`, `raw`, `8svx`, `iklax`, `xi`, `sf`, `sf2`, `ircam`, `pvf`, `fap`, `nist`, `nistsphere`, `sox`, `vox`, `amb`, ...).
- **Same-format shortcut:** if the target format matches the input's format, the file is just copied to `results/` with the quest naming scheme (no re-encoding, no quality loss).
- Lossy formats (`mp3`, `opus`, `aac`, `ogg`, `m4a`, `wma`, `ac3`, `mp2`) are encoded at high quality bitrates (256–320 kbps for `mp3`/`mp2`, 160 kbps for `opus`, etc.).
- Lossless formats preserve full bit depth. `wav` is encoded as 24-bit / 48 kHz PCM. `flac` / `ape` / `tta` / `wv` use maximum compression level.
- All outputs are normalized to stereo / 48 kHz to ensure encoders that dislike mono / unusual rates (like `libvorbis`) work cleanly.
- The `<original-name>` is the input file's stem, sanitized to safe filename characters.

```
# Convert a WAV to MP3
python voder.py quest convert mp3 "song.wav"

# Convert to FLAC (lossless)
python voder.py quest convert flac "song.wav"

# Convert to Opus (modern, very efficient)
python voder.py quest convert opus "song.wav"

# Convert to the same format — just copies the file with the quest naming scheme
python voder.py quest convert wav "song.wav"

# Save result to a specific path
python voder.py quest convert mp3 "song.wav" result "./out.mp3"

# Format argument is case-insensitive and accepts a leading dot
python voder.py quest convert MP3 "song.wav"
python voder.py quest convert .flac "song.wav"
```

### 9.5 `compress`

| Argument | Description |
|----------|-------------|
| `[level]` | (optional) Compression level: `1` (low), `2` (default), `3` (highest). Defaults to `2` when omitted. |
| `"<input>"` | A LOCAL audio file path. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Reduces file size by re-encoding at lower bitrate (lossy formats) or lower bit-depth / sample-rate (lossless formats).
- **Lossy formats** (`mp3`, `opus`, `aac`, `ogg`, `wma`, `m4a`, `mp2`, `ac3`): L1 = 192–256 kbps, L2 = 96–128 kbps, L3 = 40–64 kbps.
- **Lossless formats** (`wav`, `flac`, `aiff`, `amb`, `au`, `caf`): L1 = 24-bit / 44.1 kHz, L2 = 16-bit / 32 kHz, L3 = 16-bit / 22.05 kHz. FLAC also raises its compression level (L1 = 8, L2 = 10, L3 = 12).
- The input's existing bit-depth and sample-rate are never upgraded — `compress` only reduces. If the input is already lower quality than the target level, the output matches the input's quality.
- AMR is forced to 8 kHz mono (AMR's only supported mode).
- The console output prints before/after size and the percent change.

```
# Default compression (level 2)
python voder.py quest compress "song.wav"

# Light compression (level 1) — barely any quality loss
python voder.py quest compress 1 "song.wav"

# Maximum compression (level 3) — smallest file, lowest quality
python voder.py quest compress 3 "song.mp3"

# Save result to a specific path
python voder.py quest compress 2 "song.wav" result "./out.wav"
```

### 9.6 `cut`

| Argument | Description |
|----------|-------------|
| `<start>-<end>` | Time range in seconds. Also accepts `mm:ss` and `hh:mm:ss`. `start` must be strictly smaller than `end`. Both must be non-negative. |
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Extracts a time range from a local audio OR video file and outputs a WAV (PCM 16-bit, 44.1 kHz, stereo).
- For video input, only the audio track is extracted (video frames are dropped — use this to grab a clip of audio from a video without keeping the frames).
- Time format examples: `20-40` (20s to 40s), `1:30-2:15` (1m30s to 2m15s), `0:00:00-0:00:05` (first 5 seconds), `1.5-3.5` (floats are allowed).
- The output filename includes the range for easy identification: `voder_quest_cut_<name>_<start>s-<end>s_<timestamp>.wav`.

```
# Extract seconds 20 through 40
python voder.py quest cut 20-40 "song.wav"

# Extract a clip using mm:ss notation
python voder.py quest cut 1:30-2:15 "song.wav"

# Extract the first 5 seconds from a video (outputs WAV, not video)
python voder.py quest cut 0-5 "video.mp4"

# Save result to a specific path
python voder.py quest cut 10-30 "song.wav" result "./clip.wav"

# Refused inputs:
# python voder.py quest cut 5-2 "song.wav"     # ERROR: start must be smaller than end
# python voder.py quest cut abc-5 "song.wav"   # ERROR: not a valid range
```

### 9.7 `remove`

| Argument | Description |
|----------|-------------|
| `"<start1>-<end1>"` | First time range to remove, in seconds. Also accepts `mm:ss` and `hh:mm:ss`. `start` must be strictly smaller than `end`. |
| `["<start2>-<end2>" ...]` | (optional) Additional ranges to remove. Any number of ranges can be passed; overlapping ranges are merged automatically so no part is ever cut twice. |
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Inverse of `cut`:** instead of keeping the requested range and dropping the rest, `remove` drops the requested ranges and keeps the rest. Use it to delete intros, outros, ad breaks, dead air, or any unwanted segments from a file.
- **Multi-range:** pass any number of `"<start>-<end>"` tokens before the input path. They are parsed, sorted, and merged with a sweep-line algorithm — overlapping or adjacent ranges collapse into a single range so no part of the file is processed twice.
- **Overlap-merge examples:**
  - `"5-10" "8-15"` → merged to `5-15` (the overlapping 8-10 section is removed once, not twice).
  - `"0-5" "3-8" "10-15"` → merged to `0-8, 10-15`.
  - `"10-20" "5-10"` → merged to `5-20` (out-of-order input is normalized).
- **File duration is read with `ffprobe`** so the final keep-segment is bounded by the actual file length (no out-of-bounds errors).
- **Keeps the rest:** after computing the merged cut-ranges, the inverse (the segments to keep) is computed and concatenated with FFmpeg's `concat` filter. Sample-accurate joins, no gaps.
- **Audio input** → WAV (PCM 24-bit, 48 kHz, stereo). **Video input** → MP4 with video re-encoded as H.264 CRF 18 (visually lossless) and audio as AAC 256 kbps. Both audio and video tracks are cut in lockstep so they stay in sync.
- The output filename lists the merged cut-ranges: `voder_quest_remove_<name>_<start1>-<end1>s[_<start2>-<end2>s...]_<timestamp>.{wav,mp4}`.

```
# Remove a single range (e.g. drop an intro from 0-12s)
python voder.py quest remove "0-12" "song.wav"

# Remove multiple non-overlapping ranges (intro + outro)
python voder.py quest remove "0-15" "180-200" "song.wav"

# Remove overlapping ranges — auto-merged to 5-15
python voder.py quest remove "5-10" "8-15" "song.wav"

# Remove from a video (both video and audio are cut in lockstep)
python voder.py quest remove "0-30" "120-150" "clip.mp4"

# Use mm:ss notation
python voder.py quest remove "1:00-1:30" "3:15-3:45" "podcast.wav"

# Save result to a specific path
python voder.py quest remove "10-20" "song.wav" result "./trimmed.wav"

# Refused inputs:
# python voder.py quest remove "10-5" "song.wav"    # ERROR: start must be smaller than end
# python voder.py quest remove "abc-5" "song.wav"  # ERROR: not a valid range
# python voder.py quest remove "0-99999" "song.wav"  # OK: range extends past EOF, gets clipped to file duration
# python voder.py quest remove "0-100000" "clip.wav"  # If all ranges cover the entire file -> ERROR: nothing would remain
```

### 9.8 `merge`

| Argument | Description |
|----------|-------------|
| `"<file1>" "<file2>" ["<file3>" ...]` | Two or more LOCAL audio files. No upper limit on the number of files. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Concatenates two or more local audio files end-to-end into a single WAV (PCM 16-bit, 44.1 kHz, stereo).
- Each input is first normalized to the same sample-rate / channel layout, then concatenated with FFmpeg's `concat` demuxer for sample-accurate joining.
- The output filename joins the (truncated) stems of the input files, so it's easy to tell at a glance which files were merged.
- Audio files of different formats, sample rates, and channel counts can all be merged in the same call — they're each normalized before concatenation.
- Video files are accepted only if they have an audio stream; only the audio is used.

```
# Merge two files
python voder.py quest merge "part1.wav" "part2.wav"

# Merge six files (no upper limit)
python voder.py quest merge "1.wav" "2.wav" "3.wav" "4.wav" "5.wav" "6.wav"

# Merge files of different formats — they're normalized first
python voder.py quest merge "intro.mp3" "body.wav" "outro.flac"

# Save result to a specific path
python voder.py quest merge "a.wav" "b.wav" result "./combined.wav"

# Refused inputs:
# python voder.py quest merge "only-one.wav"                # ERROR: needs at least two files
# python voder.py quest merge "a.wav" "/nonexistent.wav"    # ERROR: file not found
```

### 9.7b `mix`

| Argument | Description |
|----------|-------------|
| `"<base_source>"` | The first source. This is the base — it always starts at 0s. **Must NOT have a number before it** (it is the base, so it has no start-time prefix). |
| `[<seconds> "<input>"]...` | One or more additional sources to overlay on top of the base. Each can be preceded by an optional start time in seconds (a number). Sources without a number before them start at 0s. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Overlays multiple audio/video sources at specified start times into a single WAV (PCM 16-bit, 44.1 kHz, stereo).
- The **first source is the base** — it always starts at 0s. Putting a number before it is an error (the base has no start-time prefix).
- Subsequent sources can have an optional start time in seconds **before** them. Sources without a number start at 0s (overlapping the base from the start).
- A number with **no** source path after it is an error. A non-number token between sources (that isn't a recognized file path) is also an error.
- **Audio is extracted from video files** — mix accepts `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.wmv` as well as any audio format.
- Accepts **local paths AND URLs** — URLs are downloaded via yt-dlp before mixing (audio track only).
- Uses FFmpeg's `adelay` + `amix` filters: each source is delayed by its start time (in ms), then all sources are summed with `amix=duration=longest:normalize=0`. The output duration matches the longest source plus its offset.
- The output filename joins the (truncated) stems of all source files, so it's easy to tell at a glance which files were mixed.

```
# Mix two sources — base + an overlay starting at 20s
python voder.py quest mix "song.wav" 20 "vocals.wav"

# Mix three sources — base + overlay at 20s + overlay at 32s
python voder.py quest mix "song.wav" 20 "vocals.wav" 32 "beat.wav"

# Mix with a source that starts at 0s (no number before it)
python voder.py quest mix "song.wav" "ambience.wav" 10 "vocals.wav"

# Mix a URL base with a local overlay at 15s
python voder.py quest mix "https://youtube.com/watch?v=..." 15 "voiceover.wav"

# Mix audio extracted from video files
python voder.py quest mix "background.mp4" 5 "dialogue.mp4"

# Save result to a specific path
python voder.py quest mix "song.wav" 20 "vocals.wav" result "./mashup.wav"

# Refused inputs:
# python voder.py quest mix 5 "song.wav" "vocals.wav"        # ERROR: first source must not have a number before it
# python voder.py quest mix "song.wav" 20                    # ERROR: number with no source path after it
# python voder.py quest mix "song.wav" hello "vocals.wav"    # ERROR: 'hello' is not a number and not a recognized source
```

### 9.9 `silence`

| Argument | Description |
|----------|-------------|
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `[threshold]` | (optional) Silence threshold as a positive integer dB level (10–90). Default is `50` (i.e., -50 dB). Lower values are more permissive; higher values strip more aggressively. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Strips silent gaps from a local audio/video file and produces a continuous-speech WAV (PCM 16-bit, 44.1 kHz, stereo).
- Uses FFmpeg's `silenceremove` filter: removes any run longer than 0.25s that falls below the threshold, both at the start and in the middle of the file. The minimum-silence-duration of 0.25s preserves natural micro-pauses in speech.
- After silence removal, `dynaudnorm` applies a gentle dynamic-range normalization so the output levels are consistent — useful for downstream STT or voice extraction.
- Excellent as a chain step before `svs voice` to make rapid-fire continuous speech from a recording with long pauses.
- For video input, only the audio track is extracted (video frames are dropped).

```
# Strip silence with the default threshold (-50 dB)
python voder.py quest silence "podcast.wav"

# Strip more aggressively (-40 dB — strips quieter background sounds too)
python voder.py quest silence "podcast.wav" 40

# Strip very quietly (-80 dB — only true digital silence is removed)
python voder.py quest silence "podcast.wav" 80

# Save result to a specific path
python voder.py quest silence "podcast.wav" result "./tight.wav"

# Refused inputs:
# python voder.py quest silence "in.wav" 5     # ERROR: threshold must be 10-90
# python voder.py quest silence "in.wav" 95    # ERROR: threshold must be 10-90
```

### 9.10 `reverse`

| Argument | Description |
|----------|-------------|
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Reverses a local audio OR video file. Both the start-to-end ordering and the waveform are flipped — playback sounds fully backwards.
- **Audio input** → reversed WAV (PCM 16-bit, 44.1 kHz, stereo) via FFmpeg's `areverse` filter.
- **Video input** → reversed MP4 (H.264 video + AAC audio). Both video frames and audio are reversed in lockstep using FFmpeg's `reverse` + `areverse` filters, so the reversed video stays in sync with the reversed audio.
- Recognized video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.wmv`. Any other extension is treated as audio.

```
# Reverse an audio file
python voder.py quest reverse "song.wav"

# Reverse a video file — both frames and audio are flipped
python voder.py quest reverse "clip.mp4"

# Save result to a specific path
python voder.py quest reverse "song.wav" result "./backwards.wav"
```

### 9.11 `fade`

| Argument | Description |
|----------|-------------|
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `[seconds]` | (optional) Fade length in seconds per side (0.5–60). Default is `5`. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Applies a cinematic fade-in and fade-out at the start and end of the file.
- **NOT silence-based** — the edges rise to ~15% gain (not 0%) and then swell to full volume using a smooth quarter-sine curve, so the audio is always present and feels like it's *rising* into the mix rather than cutting in from silence. A final `volume=1.15` boost gives a slight lift in the body.
- For files shorter than 2 × fade duration, the fade length is automatically clamped to 25% of the file's duration per side (with a minimum of 0.5s) so the fades never overlap.
- **Audio input** → WAV (PCM 16-bit, 44.1 kHz, stereo). **Video input** → MP4 with the original video stream copied and the audio stream replaced with the faded audio (H.264 video is stream-copied, no re-encoding of video).
- The default 5s fade is the cinematic standard — long enough to feel like a film opening, short enough to not test the listener's patience.

```
# Apply the default 5-second cinematic fade in and out
python voder.py quest fade "song.wav"

# Apply a shorter 2-second fade
python voder.py quest fade "song.wav" 2

# Apply fade to a video (video is preserved, audio gets the fade)
python voder.py quest fade "clip.mp4"

# Save result to a specific path
python voder.py quest fade "song.wav" 5 result "./cinematic.wav"

# Refused inputs:
# python voder.py quest fade "song.wav" 100   # ERROR: fade duration must be 0.5-60s
```

### 9.12 `soundlevel`

| Argument | Description |
|----------|-------------|
| `<0.01-10.00>` | Linear sound-level multiplier. `1.00` = original amplitude, `0.01` = 1% of original, `0.25` = 25% of original, `1.99` = +99% louder, `2.00` = 2× louder, `10.00` = 10× louder. |
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Linear sound-level multiplier** — multiplies every sample by the same factor. `1.00` is a no-op; `0.50` halves the amplitude (quietest usable level for typical sources); `2.00` doubles it (+6 dB); `10.00` is the maximum (+20 dB).
- Affects ALL frequencies equally (bass, mids, highs all get scaled together). The frequency spectrum keeps its shape, just gets taller or shorter.
- **What it does NOT do:** no bass boost, no treble lift, no compression, no loudness normalization. It is the simplest possible gain stage.
- **Why use it:** when you just want the audio louder or quieter without changing its tonal character. For tonal shaping use `quest bassboost` (low frequencies) or `quest loudnorm` (perceptual loudness target).
- **Chaining tip:** pair with `quest bassboost` for independent control of overall level vs low-end punch. Example: `quest soundlevel 2.00 "song.wav"` then `quest bassboost 50` makes the song 2× louder AND adds +12 dB of bass — two independent dimensions.
- **Audio input** → WAV (PCM 24-bit, 48 kHz, stereo). **Video input** → MP4 with video stream copied and audio re-encoded as AAC 256 kbps.
- **Note on clipping:** pure gain above the headroom of the source will clip. If you hear distortion at values above ~3.00, follow with `quest bassboost` (which includes a true-peak limiter) or lower the value.

```
# 2× louder (+6 dB)
python voder.py quest soundlevel 2.00 "song.wav"

# 10× louder (+20 dB, max)
python voder.py quest soundlevel 10.00 "song.wav"

# Half volume (-6 dB)
python voder.py quest soundlevel 0.50 "song.wav"

# 25% volume (very quiet)
python voder.py quest soundlevel 0.25 "song.wav"

# Make a video's audio louder (video preserved, audio re-encoded as AAC 256k)
python voder.py quest soundlevel 2.50 "clip.mp4"

# Save result to a specific path
python voder.py quest soundlevel 2.00 "song.wav" result "./louder.wav"

# Refused inputs:
# python voder.py quest soundlevel 0.005 "song.wav"  # ERROR: must be 0.01-10.00
# python voder.py quest soundlevel 11 "song.wav"     # ERROR: must be 0.01-10.00
# python voder.py quest soundlevel abc "song.wav"    # ERROR: must be a number
```

### 9.13 `bassboost`

| Argument | Description |
|----------|-------------|
| `<1-100>` | Bass boost value. `1` = subtle warmth, `50` = strong club bass, `100` = +24 dB sub-destroyer. Linearly interpolated across the range. |
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Professional multi-band bass booster** — selectively boosts LOW frequencies only (20–250 Hz). Mids and highs are left untouched. The frequency spectrum's SHAPE changes (bass gets fatter relative to the rest), unlike `quest soundlevel` which scales everything equally.
- **Signal chain** (6 stages, all designed to avoid dotty/buzzy artifacts):
  1. **Sub-sonic highpass** (`highpass=f=30`) — removes inaudible sub-30 Hz rumble that would otherwise eat headroom and trigger the compressor/limiter needlessly.
  2. **Low-shelf boost** (`bass` filter) — main bass boost at 80 Hz corner, 80 Hz width. Gain scales linearly from 0 dB (value=0) to +24 dB (value=100).
  3. **Sub-bass peaking EQ** (`equalizer=f=50:w=40`) — adds an extra narrow-band boost at 50 Hz for sub-bass "punch" you can feel in your chest. Gain scales from 0 to +18 dB.
  4. **Virtual sub-bass synthesizer** (`virtualbass`) — generates sub-bass harmonics at 250 Hz cutoff so the bass is audible even on small speakers / earbuds that can't reproduce true sub-bass. Strength scales from 0.3 (subtle) to 3.0 (aggressive).
  5. **Soft-knee compressor** (`acompressor`) — glues the boosted bass into the mix and prevents transient peaks from clipping. Threshold scales from 0.5 (gentle, value=0) down to 0.15 (aggressive, value=100); ratio scales from 2:1 to 5:1. Attack 10 ms, release 200 ms, makeup +1.1×, knee 4 dB.
  6. **True-peak limiter** (`alimiter`) — final safety net at -1 dB (-0.89 linear), 5 ms attack, 50 ms release. Guarantees no clipping and no dotty noise at any value, even 100.
- **Value mapping formula:** `t = value / 100`, then `shelf_gain = 24 × t` dB, `peak_gain = 18 × t` dB, `virtual_strength = 0.3 + 2.7 × t`, `comp_threshold = max(0.05, 0.5 - 0.35 × t)`, `comp_ratio = 2.0 + 3.0 × t`.
- **Audio input** → WAV (PCM 24-bit, 48 kHz, stereo). **Video input** → MP4 with video stream copied and audio re-encoded as AAC 256 kbps.
- **Chaining tip:** combine with `quest soundlevel` (overall loudness) + `quest speed` + `quest pitch` + `quest reverb` for a slowed+reverb+bass-boosted edit. For automatic broadcast-level loudness, finish with `quest loudnorm`.

```
# Subtle warmth (+6 dB shelf)
python voder.py quest bassboost 25 "song.wav"

# Strong club bass (+12 dB shelf, +9 dB peak)
python voder.py quest bassboost 50 "song.wav"

# Maximum sub-destroyer (+24 dB shelf, +18 dB peak, full limiter engaged)
python voder.py quest bassboost 100 "song.wav"

# Bass boost a video's audio (video preserved, audio gets boosted)
python voder.py quest bassboost 70 "clip.mp4"

# Save result to a specific path
python voder.py quest bassboost 50 "song.wav" result "./bass.wav"

# Refused inputs:
# python voder.py quest bassboost 0 "song.wav"   # ERROR: must be 1-100
# python voder.py quest bassboost 101 "song.wav" # ERROR: must be 1-100
# python voder.py quest bassboost abc "song.wav" # ERROR: must be an integer
```

### 9.14 `speed`

| Argument | Description |
|----------|-------------|
| `<value>` | Time-stretch value: one of `0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, ..., 10.00` (steps of 0.25, **excluding** 1.00). `0.25` = 4× faster, `10.00` = 10× slower. |
| `"<input>"` | A LOCAL **audio** file path. URLs are refused — use `quest download` first. Video files are refused — use `quest cut` or `quest noframes` to extract audio first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Professional time-stretch** (Spotify-style slowed / sped-up versions) using FFmpeg's `rubberband` filter. Pitch and formants are preserved — the audio sounds like it was originally performed at the new tempo, not like a tape-speed change.
- The full rubberband configuration: `formant=preserved` (formants stay natural-sounding), `transients=crisp` (preserves percussive attacks), `detector=compound` (best transient detection), `phase=laminar` (preserves phase coherence), `window=standard`, `pitchq=quality` (highest-quality pitch processing), `channels=apart` (each channel processed independently for stereo width).
- **Value semantics:** `value` is the output duration multiplier. `0.25` makes the output 4× shorter (4× faster playback). `2.00` makes the output 2× longer (2× slower playback). `10.00` makes the output 10× longer (10× slower — extreme slow-mo).
- Output: WAV (PCM 24-bit, 48 kHz, stereo) so the time-stretched audio retains maximum fidelity for further processing.
- The output filename includes the value: `voder_quest_speed_x<value>_<name>_<timestamp>.wav`.

```
# 2× faster (Sped Up version)
python voder.py quest speed 0.50 "song.wav"

# 2× slower (Slowed version)
python voder.py quest speed 2.00 "song.wav"

# Extreme slow (10× — Super Slowed)
python voder.py quest speed 10.00 "song.wav"

# 1.5× faster (Sped Up but less aggressive)
python voder.py quest speed 0.75 "song.wav"

# Save result to a specific path
python voder.py quest speed 2.00 "song.wav" result "./slowed.wav"

# Refused inputs:
# python voder.py quest speed 1.00 "song.wav"    # ERROR: no-op
# python voder.py quest speed 0.30 "song.wav"    # ERROR: not a valid 0.25-step value
# python voder.py quest speed 2.00 "video.mp4"   # ERROR: audio files only
```

### 9.15 `pitch`

| Argument | Description |
|----------|-------------|
| `<0.01-10.00>` | Pitch scale factor in 0.01 increments. `1.00` is excluded (no-op). `0.50` = −1 octave (monster/demon voice), `2.00` = +1 octave (baby/chipmunk voice), `0.01` = extreme deep (≈6.64 octaves down), `10.00` = extreme high (≈3.32 octaves up). |
| `"<input>"` | A LOCAL audio file, LOCAL video file, or a URL from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter). For video inputs, only the audio stream is read (video frames are dropped). For URLs, the audio is downloaded via yt-dlp before processing and the temp file is cleaned up after. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Professional pitch shift** using FFmpeg's `rubberband` filter. Pitch is shifted without changing tempo (duration is preserved) — the opposite of `quest speed`, which changes tempo without changing pitch.
- **Formant-shifted** (`formant=shifted`, rubberband's default): formants move with the pitch. This gives the classic tape / vinyl character that makes voices sound like demons / monsters / babies, and that makes Spotify-style "slowed+reverb" songs sound right (because the original vinyl slowdown naturally shifted both pitch and formants together). If you want a "clean" modern pitch shift instead, chain `quest pitch` after `quest speed` with the inverse value — but for the demon/baby/slowed aesthetic, `formant=shifted` is the professional default.
- **Multi-pass for extreme ranges:** rubberband produces the cleanest output within ±1 octave (0.50–2.00). For values outside that range, the shift is automatically decomposed into chained one-octave passes (each ≤2.0× or ≥0.5×). For example, `pitch 0.01` becomes 6 passes of `0.5` + 1 pass of `0.64`; `pitch 10.00` becomes 3 passes of `2.0` + 1 pass of `1.25`. The total shift is the product of all passes, equal to the requested value. This keeps each rubberband invocation in its clean-operating range.
- Full rubberband configuration per pass: `formant=shifted`, `transients=crisp` (preserves percussive attacks), `detector=compound` (best transient detection), `phase=laminar` (preserves phase coherence), `window=standard`, `pitchq=quality` (highest-quality pitch processing), `channels=apart` (each channel processed independently for stereo width).
- **Output:** WAV (PCM 24-bit, 48 kHz, stereo) for maximum fidelity. The output filename includes the value: `voder_quest_pitch_p<value>_<name>_<timestamp>.wav`.

**Chain with `quest speed` for Spotify-style slowed+reverb:**

- `quest speed 2.00` → 2× slower, same pitch (tempo change, pitch preserved)
- `quest pitch 0.50` → 1 octave down, same duration (pitch change, tempo preserved)
- Combined (`speed 2.00` → `pitch 0.50`) → 2× slower AND 1 octave down = classic slowed+reverb character
- Add `quest soundlevel` (overall loudness) and `quest bassboost` (low-end punch) before for a louder, bass-boosted slowed+reverb, or `quest fade` after for a cinematic intro/outro.

```
# Monster / demon voice (1 octave down)
python voder.py quest pitch 0.50 "voice.wav"

# Baby / chipmunk voice (1 octave up)
python voder.py quest pitch 2.00 "voice.wav"

# Extreme deep (6.64 octaves down — 7 rubberband passes)
python voder.py quest pitch 0.01 "voice.wav"

# Extreme high (3.32 octaves up — 4 rubberband passes)
python voder.py quest pitch 10.00 "voice.wav"

# Pitch-shift audio extracted from a video (video frames dropped)
python voder.py quest pitch 0.75 "clip.mp4"

# Pitch-shift a YouTube URL (audio is downloaded first)
python voder.py quest pitch 1.50 "https://youtube.com/watch?v=..."

# Save result to a specific path
python voder.py quest pitch 0.50 "voice.wav" result "./demon.wav"

# Refused inputs:
# python voder.py quest pitch 1.00 "voice.wav"  # ERROR: no-op
# python voder.py quest pitch 0.005 "voice.wav" # ERROR: must be 0.01-10.00
# python voder.py quest pitch 11.00 "voice.wav" # ERROR: must be 0.01-10.00
# python voder.py quest pitch abc "voice.wav"   # ERROR: must be a number
```

### 9.16 `glue`

| Argument | Description |
|----------|-------------|
| `"<input-to-use>"` | The first source. Must be either audio or video (file extension determines which). URLs are refused — use `quest download` first. |
| `"<where-it-will-be-glued>"` | The second source. Must be the opposite type from the first (audio+video or video+audio). Same-type pairs are refused. URLs are refused. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- Glues one source onto the other, producing an MP4 that takes video frames from the video input and audio from the audio input. The order of arguments only determines which file is the "video source" vs the "audio source" — the output is always a video file.
- **Auto-replaces existing audio:** if the video input already has an audio track, it is dropped and replaced with the audio from the audio input. No `replace` keyword is needed.
- **Duration handling:** the output duration is always the **longer** of the two inputs.
  - If the audio is shorter than the video: the audio is padded with silence (`apad=pad_dur=<diff>`) so it runs until the last video frame.
  - If the video is shorter than the audio: the video is extended with black frames (`tpad=stop_mode=add:stop_duration=<diff>`) so it runs until the audio ends. (Use `quest reverse` + `quest cut` for fancier "freeze last frame" effects — `glue` uses clean black for predictability.)
- **Refused combinations:**
  - URLs of any kind (must be local files — use `quest download` first).
  - audio+audio (no video to glue onto — use `quest merge` instead).
  - video+video (no audio to glue onto — use `quest noframes` on one of them first).
- Output is MP4 (H.264 video, AAC 256 kbps audio, CRF 20, medium preset, +faststart for streaming). Output naming: `voder_quest_glue_<audio-name>_onto_<video-name>_<timestamp>.mp4`.

**Chain pattern:** A common chain is `soundlevel` → `bassboost` → `speed` → `pitch` → `glue`. For example, take a song, make it louder, bass-boost it, slow it down, pitch it down, then glue the result back onto the original music video — you get a "slowed+reverb+bass-boosted" version of the video without re-recording anything. Finish with `quest loudnorm` if you want broadcast-standard perceptual loudness.

```
# Glue a new audio track onto a video (audio replaces the original)
python voder.py quest glue "new_audio.wav" "video.mp4"

# Glue a video's frames onto an audio track (same result, swapped argument order)
python voder.py quest glue "video.mp4" "new_audio.wav"

# Audio is shorter than video — audio is padded with silence at the end
python voder.py quest glue "short_clip.wav" "long_video.mp4"

# Video is shorter than audio — video is extended with black frames at the end
python voder.py quest glue "long_audio.wav" "short_video.mp4"

# Save result to a specific path
python voder.py quest glue "audio.wav" "video.mp4" result "./final.mp4"

# Refused inputs:
# python voder.py quest glue "https://youtube.com/..." "video.mp4"  # ERROR: refuses URLs
# python voder.py quest glue "a.wav" "b.wav"                        # ERROR: refuses audio+audio
# python voder.py quest glue "a.mp4" "b.mp4"                        # ERROR: refuses video+video
# python voder.py quest glue "only_one.wav"                         # ERROR: needs two arguments
```

### 9.17 `reverb`

| Argument | Description |
|----------|-------------|
| `<1-100>` | Reverb amount on an integer 1–100 scale. `1` = barely-there small room, `25` = chamber, `50` = concert hall, `75` = large hall, `100` = cathedral-drenched. Must be an integer (no decimals). |
| `"<input>"` | A LOCAL audio file, LOCAL video file, or a URL from any supported platform (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter). For video inputs, only the audio stream is read (video frames are dropped). For URLs, the audio is downloaded via yt-dlp before processing and the temp file is cleaned up after. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **Professional algorithmic reverb** built in the classic Schroeder topology — the same architecture used by pro studio reverbs before convolution took over. Freeverb is not compiled into this build of FFmpeg, so the reverb is constructed from FFmpeg's `aecho` (multi-tap delay) plus `adelay` (pre-delay), `lowpass` (air-absorption damping), `acompressor` (peak control), `dynaudnorm` (dynamic normalization), and `alimiter` (true-peak limiter).
- **Filter chain (in order):**
  1. `highpass=f=60` — removes sub-60 Hz rumble for a clean input.
  2. `adelay=<predelay>|<predelay>:all=1` — pre-delay scales linearly from 5 ms (small room) to 80 ms (cathedral). Pre-delay separates the dry signal from the reverb tail, which is what gives big reverbs their clarity (the dry vocal cuts through first, then the tail blooms).
  3. `aecho` (early reflections) — 5 taps at 18, 27, 36, 46, 58 ms with decays 0.10–0.40 scaling with the value. These are the early reflections that give the brain its spatial cue (room size).
  4. `aecho` (late reverb tail) — 7 taps at 61, 73, 89, 103, 127, 151, 181 ms with decays 0.15–0.55 scaling with the value. These produce the diffuse "wash" that's the actual reverb tail.
  5. `lowpass=f=<cutoff>` — air-absorption damping. Cutoff scales from 6 kHz (small room — surfaces absorb HF, sounds tighter) to 13 kHz (cathedral — hard reflective surfaces preserve HF, sounds washy and bright).
  6. `acompressor` — soft-knee compressor (threshold and ratio scale with the value) to keep peaks under control when the reverb builds up.
  7. `dynaudnorm=f=200:g=15:p=0.95` — dynamic audio normalization. Works at any input level (unlike `loudnorm` with `linear=true`, which fails on quiet signals). Keeps the output at a consistent perceived loudness regardless of the reverb amount.
  8. `alimiter=limit=0.95:attack=5:release=50` — true-peak limiter as the final safety net. No clipping is possible past this stage.
- **Output:** WAV (PCM 24-bit, 48 kHz, stereo) for maximum fidelity. The output filename includes the value: `voder_quest_reverb_r<value>_<name>_<timestamp>.wav`.
- **Audio output only.** Video inputs are accepted but only the audio stream is processed (video frames are dropped). To put reverbed audio back onto a video, chain with `quest glue`.

**What reverb is NOT:**

- Reverb is its own effect, independent of pitch and time-stretch. It simulates sound bouncing around a physical space (room, hall, cathedral) — adds decaying reflections after each sound event. No pitch change, no tempo change.
- `quest speed 2.00` alone = 2× slower, same pitch (tempo change).
- `quest pitch 0.50` alone = 1 octave down, same duration (pitch change).
- `quest reverb 100` alone = cathedral-drenched, same pitch and tempo (space change).
- The "slowed+reverb" genre chains all three: `speed` (slower) + `pitch` (lower) + `reverb` (spacey).

**Chain pattern for the full demon-cathedral slowed+reverb edit:**

```
soundlevel → bassboost → speed → pitch → reverb → glue (back onto the original video)
```

For example: take a song, make it louder, bass-boost it, slow it down 2×, pitch it down 1 octave, drown it in cathedral reverb, then glue the result back onto the original music video.

```
# Barely-there small room (subtle glue on a dry recording)
python voder.py quest reverb 5 "voice.wav"

# Concert hall (classic lush reverb for vocals)
python voder.py quest reverb 50 "voice.wav"

# Cathedral-drenched (full ambient wash)
python voder.py quest reverb 100 "voice.wav"

# Reverb audio extracted from a video (video frames dropped)
python voder.py quest reverb 75 "clip.mp4"

# Reverb a YouTube URL (audio is downloaded first)
python voder.py quest reverb 80 "https://youtube.com/watch?v=..."

# Save result to a specific path
python voder.py quest reverb 50 "voice.wav" result "./wet.wav"

# Full slowed+reverb chain (loudness → bass boost → slow → pitch down → reverb → glue onto video)
python voder.py chains \
  "loud"  quest soundlevel 2.00 "song.wav" / \
  "bass"  quest bassboost 70 loud / \
  "slow"  quest speed 2.00 bass / \
  "deep"  quest pitch 0.50 slow / \
  "wet"   quest reverb 85 deep / \
  "final" quest glue wet "music_video.mp4" result "./slowed_reverb_demon.mp4"

# Refused inputs:
# python voder.py quest reverb 0 "voice.wav"     # ERROR: must be 1-100 (0 is a no-op)
# python voder.py quest reverb 101 "voice.wav"   # ERROR: must be 1-100
# python voder.py quest reverb 50.5 "voice.wav"  # ERROR: must be an integer
# python voder.py quest reverb abc "voice.wav"   # ERROR: must be an integer
# python voder.py quest reverb 50                # ERROR: needs an input
```

### 9.18 `loudnorm`

| Argument | Description |
|----------|-------------|
| `"<input>"` | A LOCAL audio or video file path. URLs are refused — use `quest download` first. |
| `result "<path>"` | (optional) Copy the result to a custom path. |

**Behavior:**

- **EBU R128 perceptual loudness normalization.** The file is analyzed in a first pass to measure its integrated loudness (LUFS), true-peak (dBTP), loudness range (LU), and noise threshold. A second pass then applies a single linear gain (via `loudnorm` with `linear=true`) that brings the whole signal to the target integrated loudness of **-16 LUFS** with a true-peak ceiling of **-1.5 dBTP**.
- **One consistent perceived level:** quiet parts and loud parts end up at the same perceptual medium. The whole file plays at a uniform loudness — ideal for podcasts, voice-overs, and dialogue recorded in different environments or with different microphones.
- **No quality loss, no dynamic-range compression.** Because the normalization is a single linear gain (not dynamic), the relative dynamics inside the file are preserved — a whisper is still quieter than a shout within the same file, but the file as a whole sits at the same perceptual level as any other `loudnorm`-processed file.
- **Difference from `quest soundlevel`:** `soundlevel` applies a user-specified fixed multiplier (e.g. `2.00` = +6 dB on every sample). `loudnorm` measures the file and computes the multiplier for you, targeting a perceptual standard (-16 LUFS). Use `soundlevel` when you know the exact gain you want; use `loudnorm` when you want every file to end up at the same perceptual level automatically.
- **Difference from `quest compress`:** `compress` reduces the dynamic range *within* a file (loud parts pulled down, quiet parts pushed up — changes the dynamics). `loudnorm` only shifts the whole file up or down as one block (preserves the dynamics).
- **Audio input** → WAV (PCM 24-bit, 48 kHz, stereo). **Video input** → MP4 with video stream copied and audio re-encoded as AAC 256 kbps.
- If the input is already at -16 LUFS (within 0.2 LU), the pass-through still runs to apply the true-peak safety limit.

```
# Normalize a podcast to broadcast standard (-16 LUFS)
python voder.py quest loudnorm "episode.wav"

# Normalize a voice-over recorded quietly so it sits at the same level as other clips
python voder.py quest loudnorm "quiet_voiceover.wav"

# Normalize a video's audio (video preserved, audio re-encoded as AAC 256k)
python voder.py quest loudnorm "clip.mp4"

# Save result to a specific path
python voder.py quest loudnorm "episode.wav" result "./episode_normalized.wav"

# Refused inputs:
# python voder.py quest loudnorm                  # ERROR: needs an input
# python voder.py quest loudnorm "missing.wav"    # ERROR: file not found
# python voder.py quest loudnorm "a" "b"          # ERROR: takes exactly one argument
```

---

## 10. `chains` — User-Defined Pipelines

> **Note:** `chains` composes the main voder oneline tasks (TTS, STS, TTM, STT, SE, SFX, SVS, SS) and the other features (`train`, `quest`) into user-defined pipelines whose intermediate outputs are kept in `temp_chains/`.

Chains let the user compose their own pipelines out of voder's existing oneline tasks. Each chain is named, runs a voder oneline command, and its output is captured to a temp directory. Later chains can reference earlier chain names as input paths — voder substitutes the chain name with the captured temp file path before running the later chain. The **last** non-empty chain's output is exported to `results/`; intermediate outputs live in `temp_chains/`.

### Syntax

```
python voder.py chains "name1" <voder command...> / "name2" <voder command that references "name1"> / "name3" <voder command that references "name1" and/or "name2"> / ... [result "<path>"]
```

- ` / ` (space, slash, space) separates chains. The slash must be its own argv element — do not attach it to neighbouring arguments.
- Each chain starts with a quoted name (or any single token; quotes are optional but recommended, especially if the name contains spaces).
- The rest of the chain's args are a normal voder oneline command (e.g., `tts script "hi" voice "male"`, `svs voice "song.wav"`, `se "vocals"`, `quest download "url"`, etc.).
- The optional trailing `result "<path>"` applies to the whole `chains` command — it copies the **final** chain's output to the given path.

### How chain names are resolved

- VODER indexes chain names as each chain runs.
- Before running a later chain, VODER walks that chain's arguments: any argument that exactly matches a previously-defined chain name is replaced with the path to that chain's output file. Non-matching arguments are left untouched (and then treated normally — as a file path, URL, or whatever the command expects).
- Chain name lookups are exact (case-sensitive) and take precedence over file/URL resolution. If a chain name happens to look like a file path or URL, it still wins.

### Output storage

- **Intermediate chains** (all but the last): their output is moved to `temp_chains/voder_chain_<safe-name>_<timestamp>.<ext>`. Any other files they created in `results/` are cleaned up so the results folder stays uncluttered.
- **Last non-empty chain**: its output stays in `results/` (or `voices/` for `train` chains). This is the user-visible result of the whole `chains` command.
- For multi-output commands (e.g., `svs both`, `ss`, TTM with stems), only the **latest** file produced by the chain is exposed as the chain's output. If you need multiple outputs, run separate chains.

### Validation rules

- **Duplicate chain names** are an error and stop the pipeline immediately. Two non-empty chains cannot share the same name.
- **Empty chains** (a name with no command following it) are **skipped**. Their names are NOT marked as used, so the same name can be reused later in the same `chains` command. Example: `"a" / "b" / "a" tts script "hi"` is valid — the first two are empty, and the third (non-empty) chain claims the name `a`.
- **Trailing empty chains** at the end are ignored, just like empty chains in the middle.
- If **all** chains are empty, the pipeline returns an error ("no valid chains to execute").

### Examples

```
# Generate a song → isolate its vocals → voice-convert them
python voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"

# Isolate vocals → enhance them → transcribe the result
python voder.py chains "vocals" svs voice "song.wav" / "enhanced" se voice "vocals" / "text" stt "enhanced" timestamp

# Train a voice from a chain's output, then use it to speak
python voder.py chains "vocal" svs voice "song.wav" / "trained" train voice:singer "vocal" / "spoken" tts script "Hello world" voice "singer"

# Download audio → transcribe it
python voder.py chains "audio" quest download "https://youtube.com/watch?v=..." / "text" stt "audio" timestamp

# Numbers and arbitrary names work too
python voder.py chains "1" tts script "hi" voice "male" / "2" se "1" / "3" stt "2" timestamp

# Empty chains are skipped (names remain reusable) — this is valid:
python voder.py chains "skip1" / "skip2" / "real" tts script "hi" voice "male"

# Duplicate names are an error and stop the pipeline:
# python voder.py chains "a" tts script "one" / "a" tts script "two"   # ERROR: Duplicate chain name: 'a'

# Use result to copy the final chain's output to a specific path
python voder.py chains "vocal" svs voice "song.wav" / "enhanced" se voice "vocal" result "./final.wav"
```

### Notes

- Chain names can be any string: numbers, letters, paths, URLs — whatever the user can keep track of. Quotes are stripped by the shell before voder sees them, so `"name1"` and `name1` are equivalent as the first argument of a chain.
- The `train` command works inside chains: its `.tts` / `.ttse` file is the chain's output and is stored in `temp_chains/` (not `voices/`) for intermediate chains. The voice file can then be referenced by name in later TTS chains via `voice "trained-name"` only if it lives in `voices/`; for chain-stored voices, reference them by their temp path via the chain name.
- Chain outputs that are audio files can be used as voice-cloning targets, SVS inputs, SE inputs, STS bases, STT inputs, TTM references, etc.
- Chain outputs that are video files (e.g., from `quest download video`) can be used anywhere a video input is accepted.

---

## 10a. Prebuilt Chains — Build, Load, Comment, Decompile, Compile, Journey

> **Note:** Prebuilt chains extend the `chains` feature with a persistent `.chain` file format. You compose a chain once with `chains build`, then load and re-run it any time with `chains load` (oneline) or via the interactive CLI's option 9. `chains comment` edits chain and per-input comments post-build. `chains decompile` extracts a `.chain` to a raw oneline `.txt` file you can edit; `chains compile` rebuilds a `.chain` from such a `.txt`. `chains journey` produces an RPG-like Markdown report narrating the chain's path, errors, and alternate dimensions.

Prebuilt chains live in `src/chains/VODER_<name>_<timestamp>.chain`. Each file is plain text in a custom key:value format. The first line is the magic header `# VODER_CHAIN v1 <timestamp> <name>`. Subsequent lines form a header block (`title:`, `description:`) followed by `---`-separated step blocks (`chain:`, `comment:`, `content:`).

### File format

```
# VODER_CHAIN v1 20260627_143022 bombo
title: Bombo Pipeline
description: Extract vocals from a song, transcribe them, then re-synthesize with a chosen voice.
---
chain: vocals
comment: Provide the source song. Accepts audio file, video file, or supported platform URL.
content: svs voice input
---
chain: lyrics
comment: This step is automated — uses the vocals extracted in chain 1.
content: stt vocals timestamp
---
chain: cover
comment: Provide a reference voice (audio file, URL, or .tts/.ttse voice profile).
content: tts script lyrics voice input
---
```

- **Line 1**: exactly 5 whitespace-separated tokens — `#`, `VODER_CHAIN`, `v1`, `<timestamp>` (YYYYMMDD_HHMMSS), `<name>` (`[A-Za-z0-9_-]+`, no spaces).
- **Header block**: `title:` and `description:` keys (both optional — empty values produce warnings but no errors).
- **Step blocks**: separated by `---`. Each step has `chain:` (required, must match `[A-Za-z0-9_-]+`, unique within file), `comment:` (optional — the step-level description shown to users), `content:` (required — single line, space-separated oneline command), and zero or more `comment.input.N:` lines (optional — per-input-slot descriptions, where `N` is the 1-indexed position of the `input` placeholder in `content:`). Use the literal token `input` as the placeholder for a manual file input. Reference prior chain names verbatim to make the step automated. Per-input comments are typically added via `chains comment` after the chain is built, but can also be hand-edited.

### Step classification

A step is classified by counting `input` placeholders and chain-name references in its `content:`:

- **manual**: has `input` placeholder(s), no chain references. The user must supply file paths.
- **automated**: only chain references, no `input`. The user just presses Enter.
- **semi-automated**: both. The user supplies files for the `input` slots; chain references auto-resolve.
- A step with **neither** (e.g., `sfx sound boom duration 5`) produces a warning, not an error.

### `chains build` — create a `.chain` file

```
python voder.py chains build "<name>" description "<title - description>" \
    chain "<step1-name>" "<comment1>" "<content1>" \
    chain "<step2-name>" "<comment2>" "<content2>" \
    ...
```

- `<name>` must match `[A-Za-z0-9_-]+`. Errors and stops if invalid or missing.
- `description` is a literal keyword followed by a single quoted string (the title/description). Can be empty `""` (warning, not error).
- Each `chain` block is 4 tokens: the literal `chain`, then quoted `<name>`, `<comment>`, `<content>`. The content is parsed internally by whitespace splitting.
- After basic structural validation, the builder runs full verification (format, naming, syntax, references) and reports every error before saving. The file is only written if all checks pass.
- Output: `src/chains/VODER_<name>_<timestamp>.chain`.

Example:

```
python voder.py chains build "bombo" description "Bombo - extract vocals, transcribe, re-synth" \
    chain "vocals" "Provide the source song (audio/video/URL)" "svs voice input" \
    chain "lyrics" "Automated - uses vocals from chain 1" "stt vocals timestamp" \
    chain "cover" "Provide a reference voice (audio/URL/.tts/.ttse)" "tts script lyrics voice input"
```

### `chains load` — run a `.chain` file (oneline)

```
python voder.py chains load "<chain-name-or-path>" [N:(v1/v2/...)]... [<another-chain> [N:(...)]...]...
```

- `<chain-name-or-path>`: a chain name (resolves to the latest matching file by timestamp) or a direct `.chain` file path.
- `N:(v1/v2/...)`: a marker supplying **manual inputs** for chain step `N`. Slash-separated values fill the `input` placeholders in content order. Number of values must match the number of `input` placeholders in that step.
- A marker value is one of:
  - A **file path** or **URL** (audio/video file, supported platform URL, voice-profile file `.tts`/`.ttse` at voice-profile-eligible slots — see below) — used verbatim.
  - The **main name of a previously-loaded prebuilt chain** — resolved at runtime to that prebuilt's final output path. The prebuilt must appear earlier in the same `chains load` invocation.
- **Automated steps (chain-name references in content) are never overridable**. If a step's `content:` references a prior chain name, that reference is auto-resolved at runtime; you cannot supply a marker value for it. This is by design — overriding automated slots would break the prebuilt chain's "ease-of-use" guarantee.
- **Forward references are rejected**. If a marker value matches the main name of a prebuilt that is loaded LATER in the same `chains load` command, the command fails immediately with `Error: step N '<name>' marker value '<value>' is a forward reference — prebuilt '<value>' is loaded later in this command (position P) but hasn't run yet. Reorder: load '<value>' before '<name>', or provide a file path/URL instead.` Prebuilts execute strictly in load order; a later prebuilt's output file does not exist yet when an earlier prebuilt runs.
- Multiple prebuilt chains can be loaded in one command: each chain name/path starts a new section. Each subsequent chain can reference prior prebuilt chain names by main name (the runner maintains a global index across prebuilts via `ChainPipeline.index`).

Examples:

```
# Run bombo, supply 2 manual inputs (step 1 and step 3)
python voder.py chains load "bombo" 1:(song.wav) 3:(ref.wav)

# Multi-prebuilt: run bombo, then run second_chain whose step 1
# uses bombo's final output (referenced by main name "bombo")
python voder.py chains load "bombo" 1:(song.wav) 3:(ref.wav) "second_chain" 1:(bombo)
```

#### Voice-profile-eligible positions

Voice profiles (`.tts` / `.ttse`) are valid only at specific positions the engine actually consumes them — not at every "audio-accepting" slot. The prebuilt-chains subsystem marks these positions in the `chains journey` report and in the interactive CLI's per-slot input prompt:

| Mode | Position | Voice-profile accepted? |
|------|----------|-------------------------|
| `tts` | `voice input` (single-mode voice slot) | Yes — engine resolves via `_resolve_voice_ref` |
| `tts` | `target input` (single-mode target slot, when user supplies `sts:<voice-ref>` value) | Yes — engine resolves via `_resolve_voice_ref` after stripping `sts:` prefix |
| `tts` | `script` / `music` / `level` / `ocr` / `duration` slots | No |
| `sts` | `base` / `target` slots | No (STS consumes audio for voice conversion, not voice profiles) |
| `stt` / `se` / `sfx` / `svs` / `ss` / `train` | any slot | No |
| `quest` | varies by quest | No |

If a user supplies a `.tts` / `.ttse` value at a non-voice-profile-eligible position, the engine will reject it at runtime with a "File not found" or "Unsupported format" error — the validator cannot catch this at build time because the value is supplied at load time, not at build time. The position-aware markers in the journey report and interactive CLI prompts help the user supply the right value at the right slot.

### `chains comment` — edit chain and per-input comments on an existing `.chain` file

```
python voder.py chains comment "<chain-name-or-path>" [N:"<new chain comment>"]... [N:(I1:<input comment>/I2:<input comment>/...)]...
```

`chains comment` rewrites an existing `.chain` file in place — it lets the chain developer add or update the step-level `comment:` and the per-input-slot `comment.input.N:` annotations after the chain has been built with `chains build`. This is the only way to attach per-input descriptions to a chain: `chains build` only takes a single per-step comment, so chains typically get built first, then documented with `chains comment`.

- `<chain-name-or-path>`: a chain name (resolves to the latest matching file by timestamp) or a direct `.chain` file path. The resolved file is rewritten in place.
- `N:"<new chain comment>"`: replaces the step-level `comment:` for step `N` (1-indexed). The double quotes are required. An empty string `N:""` clears the comment.
- `N:(I1:<input comment>/I2:<input comment>/...)`: sets per-input comments for step `N`. Each `I:<comment>` pair sets the comment for input slot `I` (1-indexed, in the order `input` placeholders appear in the step's `content:`). The `/` separates input entries. Only mentioned input indices are touched — unmentioned input slots keep their existing comment. An empty comment (`I:`) clears that input's comment.
- **Chain numbers and input numbers are 1-indexed and linear** — they correspond to the step's position in the file and the `input` placeholder's position in `content:`. They do **not** need to be in sorted order: you can write `7` then `4` then `3` for chains, and `8` / `19` / `3` / `2` for inputs. Only mentioned slots are touched; everything else is preserved verbatim.
- **Invalid numbers fail with "failed to resolve"** — if you write `9` and the chain only has 3 steps, the command fails immediately with `failed to resolve '9' in '<chain>' — chain has 3 step(s). Likely meant: 1, 2, 3.` and the file is **not** modified. Same for input indices: `4` on a step with 1 input slot fails with `failed to resolve '4' in step N '<name>' — chain has 1 input slot(s). Likely meant: 1.`
- After applying edits, the rewritten file is re-verified. If verification fails, the file is **not** saved and all errors are printed.
- The same number-resolution core is used for both chain indices and input indices, so error messages are consistent.

Examples:

```
# Linear edit: set chain comments for steps 1 and 3, set input comments for step 3
python voder.py chains comment "bombo" \
    1:"Provide the source song (audio/video/URL)" \
    3:"Provide a reference voice" \
    3:(1:The reference voice file to clone from/2:Optional style hint, e.g. "warm, slow")

# Non-linear edit: step 7 then 4 then 3 (order doesn't matter; only mentioned slots touched)
python voder.py chains comment "bombo" \
    7:"Updated comment for step 7" \
    4:(3:third input of step 4) \
    3:(8:eighth input/19:nineteenth input/2:second input)

# Only chain comment (no input comments)
python voder.py chains comment "bombo" 2:"This step is automated — uses the vocals from step 1."

# Only input comments (no chain comment)
python voder.py chains comment "bombo" 1:(1:The source song) 3:(1:Reference voice/2:Style hint)

# Clear a chain comment (empty string)
python voder.py chains comment "bombo" 2:""

# Clear an input comment (empty after the colon)
python voder.py chains comment "bombo" 3:(1:)
```

After `chains comment` runs, the new annotations appear in:
- `chains journey` Markdown report (per-input comments are listed under each manual input slot)
- Interactive CLI option 9 (per-input comments appear as `Input note:` under the `Accepted:` line during input gathering)
- The `.chain` file itself as `comment.input.N:` lines

### `chains decompile` — extract a `.chain` to a raw oneline `.txt` file

```
python voder.py chains decompile "<chain-name-or-path>" [<another> ...]
```

`chains decompile` extracts the pipeline from a `.chain` file into a plain-text `.txt` file containing the raw chains oneline command — the same command you would type at the terminal if you ran the chain inline. This lets you edit the pipeline as a single oneline command, then recompile it back into a `.chain` file with `chains compile`.

- `<chain-name-or-path>`: a chain name (resolves to the latest matching file by timestamp) or a direct `.chain` file path. Multiple chains can be decompiled in one command — each produces its own `.txt` file.
- **Output**: `results/VODER_chains_<safe-name>_decompiled_<timestamp>.txt`.
- **File format**: the `.txt` file starts with comment lines (`#`) containing the chain name, source path, decompile timestamp, title, description, and step count. Then a single line contains the raw oneline command: `"step1" <oneline command> / "step2" <oneline command> / ...`. Each step is quoted-named, followed by its oneline command. Steps are separated by ` / ` (space slash space). The literal token `input` marks a manual file input slot; prior chain names referenced verbatim are automated references.
- **Verification + error commenting**: the source `.chain` file is verified before decompiling. If verification passes, the `.txt` contains only the oneline command. If verification finds errors, the errors are **commented out** at the bottom of the `.txt` file (under a `# --- VERIFICATION ERRORS ---` header) so the file is still valid text but the user sees what's wrong. Warnings are similarly commented out under a `# --- WARNINGS ---` header. The oneline command is always written, even for a corrupted chain — so you can edit the command to fix the errors, then recompile.
- **Return value**: returns `False` if any decompiled chain had errors (so the user knows to check the commented-out sections), `True` if all chains were clean.

Example decompiled `.txt` file:

```
# VODER decompiled chain: bombo
# Source: src/chains/VODER_bombo_20260627_143022.chain
# Decompiled: July 04, 2026 at 21:32:35
# Title: Bombo Pipeline
# Description: Extract vocals from a song, transcribe them, then re-synthesize.
# Steps: 3
#
# This file contains the raw chains oneline command that produces the same
# pipeline as the source .chain file. Edit the command below, then recompile with:
#   python voder.py chains compile "VODER_bombo_20260627_143022.txt"
#
# Each chain step is quoted-named, followed by its oneline command.
# Steps are separated by ' / ' (space slash space).
# The literal token 'input' marks a manual file input slot.
# Prior chain names referenced verbatim are automated references.

"vocals" svs voice input / "lyrics" stt vocals timestamp / "cover" tts script lyrics voice input target input
```

### `chains compile` — rebuild a `.chain` from a decompiled `.txt` file

```
python voder.py chains compile "<txt-path>" [<another> ...]
```

`chains compile` is the inverse of `chains decompile`. It reads a `.txt` file produced by decompile (or hand-written in the same format), parses the oneline command, and builds a new `.chain` file. This lets you edit a pipeline as a single oneline command and then save it as a prebuilt chain.

- `<txt-path>`: a direct path to a `.txt` file. Multiple `.txt` files can be compiled in one command — each produces its own `.chain` file.
- **Output**: `src/chains/VODER_<name>_<timestamp>.chain` (same location as `chains build`).
- **Parsing**: the compiler reads the `# VODER decompiled chain: <name>` header to get the chain name, `# Title:` and `# Description:` comment lines for metadata, and the first non-comment line as the oneline command. The oneline command is split on ` / ` (space slash space) into segments, respecting quoted strings. Each segment's first quoted token is the step name; the rest is the step's content. Step names must match `[A-Za-z0-9_-]+` and be unique within the file.
- **Verification + no-build-on-error**: the compiled `.chain` text is verified via `verify_chain_text()` before saving. If verification finds any errors, the errors are printed to the terminal and the `.chain` file is **NOT saved**. This matches `chains build` behavior — a corrupted `.txt` never produces a corrupted `.chain`.
- **Comments**: `chains compile` does not preserve step-level or per-input comments from the source `.chain` (the decompiled `.txt` format doesn't carry them). The compiled `.chain` has empty step comments and no per-input comments. Use `chains comment` after compiling to re-add documentation.
- **Return value**: returns `True` if all `.txt` files compiled successfully, `False` if any had errors or couldn't be read.

Example:

```
# Decompile, edit, recompile
python voder.py chains decompile "bombo"
# edit results/VODER_chains_bombo_decompiled_*.txt in your text editor
python voder.py chains compile "results/VODER_chains_bombo_decompiled_20260704_213235.txt"
```

### `chains journey` — generate an RPG-like Markdown journey report

```
python voder.py chains journey "<chain-name-or-path>" [<another> ...]
```

- Runs full verification on each chain.
- Output: `results/voder_journey_<safe-name>_<timestamp>.md`.
- The report is structured as an RPG-like narrative with these sections:
  - **Opening narrative**: a storytelling intro that adapts to whether the chain(s) passed or failed. Single-chain: "In a world full of complexity and many of the unknowns, someone decided to build a chain called **name** to make their path easier. But did they? We shall find out." Multi-chain: the same but with "not one but N chains" and "the saga unfolds, chapter by chapter."
  - **Cast of Chains**: a summary table (name, path, steps, status).
  - **Per-chain chapter** (titled "Chapter N" for multi-chain, "Act N" for single-chain): file metadata (scroll, forged date in human-readable format, title, purpose), step/offering/echo counts, a **Waypoints** summary table, then **The Path Walked** — a step-by-step narrative where each step is a "Waypoint" with:
    - The step's intent (comment)
    - **The artisan** — a per-mode persona name with a descriptive verb: `tts` = "the Voice Weaver" (weaves spoken words from text), `sts` = "the Shape Shifter" (transforms one voice into another), `ttm` = "the Song Smith" (forging music from lyrics), `stt` = "the Scribe" (transcribes speech to text), `se` = "the Restorer" (cleanses noise), `sfx` = "the Sound Conjurer" (conjures sound effects), `svs` = "the Separator" (isolates vocals), `ss` = "the Crowd Sorter" (extracts individual speakers), `train` = "the Voice Keeper" (trains voice clones), `quest` = "the Errand Runner" (utility tasks), `chains` = "the Chain Master" (orchestrates pipelines). Unrecognized modes get "the Unknown Artisan".
    - Content (raw and resolved, with `<output of step N 'name'>` and `<manual input N>` placeholders)
    - A classification narrative: manual = "The traveler must provide N offering(s) to proceed", automated = "This step requires no offerings from the traveler; it draws entirely from what came before", semi-automated = "This step blends fate and choice", error = "This step stands at a crossroads with no clear path".
    - **Offerings awaited** at this step (per-slot format, voice-profile-eligible marker, per-input guidance from `comment.input.N`)
    - **Alternate dimension** block (when the step has errors): "But the step falters. Errors are found:" followed by the error list with fixes, then "In another dimension — where the chain took another path, a valid path — what could have happened if the error were the correct thing?" with a per-error-category what-if description (reference errors: "if the referenced step had been placed before this step, the automated reference would have resolved..."; syntax errors with invalid mode: "if the mode had been a recognized one, the artisan would have taken the stage..."; syntax errors with valid mode: "if the oneline syntax had been correct, the artisan would have executed..."; naming errors; format errors).
  - **The Saga: How the Chapters Connect** (when 2+ chains): shows the load order, each chapter's step/offering counts, which prior chapter names are available for cross-chapter reference at each position, and the linearity rule.
  - **The Ledger of the Journey**: a statistics table (chapters, waypoints, offerings, echoes, errors, whispers/warnings) plus an **Artisans summoned** table showing the per-mode persona and step count. When there are errors, an **All Errors** table (chapter, waypoint, category, message, fix) is included.
  - **Epilogue**: the final verdict — "The journey of this chain is whole. No errors were found. The path is clear — the traveler may now walk it with `chains load`." (success) or "The journey falters at N point(s). The errors above must be mended before this chain can be walked." (failure), ending with "*The journey ends here. For now.*"

### Interactive CLI — option 9 (Prebuilt Chains)

Run `python voder.py cli` and choose `9. Prebuilt Chains` for a guided UX:

- **List mode** (option 1): numbered list of all `.chain` files in `src/chains/`, sorted newest first. Pick a number, or type `back`.
- **Name/path mode** (option 2): enter a chain name (resolves to latest by timestamp) or a full file path.
- **Multi-chain**: after loading one chain, you can add more (each subsequent chain can reference prior prebuilt names). Prebuilts execute strictly in selection order — each prebuilt is gathered and executed before the next one starts, so a later prebuilt's output is available as a manual input value (by main name) for subsequent prebuilts. Forward references to not-yet-executed prebuilts are rejected at input validation time.
- **Input gathering**: for each step, shows the chain name, comment, content, classification (manual/automated/semi-automated), and a per-slot description of accepted input formats. Voice-profile-eligible slots are tagged `[voice-profile eligible]` so the user knows where `.tts`/`.ttse` files are valid. Per-input comments (set via `chains comment`) appear as `Input note:` under the `Accepted:` line. Manual inputs are gathered one by one with in-time validation (file exists, URL supported, or matches the name of a previously-loaded prebuilt chain — which is resolved to that prebuilt's final output). Automated steps show a compact `→ Automated input — press Enter to continue` line followed by a `[details]` block underneath (positioned below the progress tracker so it doesn't pollute the simple-user view) listing `recalls:` (the prior chain name and which step produced it), `file:` (the resolved output path, or `(will resolve at runtime)` if not yet available), and `command:` (the chain command with references substituted). Semi-automated steps show the same `[details]` block above the manual-input prompts.
- **Progress tracker**: shows `Prebuilt X/Y (name) — Step N/M (step-name) — <type>` plus `Input K/L for step 'name' — overall P/Q (NN%)`.
- **Execution**: after all inputs are gathered, prints "Press Enter to start execution" and runs each step. On mid-run error, prints `Something went further than expected.` with the error message (max 500 chars) and the chain/step where it failed.
- **Verification up front**: before asking for any inputs, the runner verifies the `.chain` file. If verification fails, lists all errors and aborts without prompting for inputs.

---

## 11. Extended Commands (`&&`)

> Chain multiple VODER oneline commands on a single line, where each command can reference outputs from any earlier command — including bidirectionally. This is **not** the same as `chains` (section 10): chains are linear pipelines with named steps; extended commands are independent VODER invocations that share files by known names.

### How it works

1. Write multiple VODER oneline commands separated by `"&&"` (quoted — the shell would otherwise interpret unquoted `&&` as its own operator)
2. Each command runs sequentially — if one fails, the chain stops
3. Use `result <bare-name>` (no quotes, no path) to save a command's output to `results/<bare-name>.<ext>` with a known name
4. Later commands reference earlier outputs by their full path (e.g., `results/file1.wav`)

### The `result` keyword (new behavior)

| Syntax | Behavior |
|--------|----------|
| `result file-name` | Copy the latest output to `results/file-name.<real_ext>` — extension auto-appended from what the engine actually produced |
| `result file-name.mp3` | Copy to `results/file-name.mp3` — extension stays as written (no format conversion; the user is responsible for matching) |
| `result file-name.auto` | Copy to `results/file-name.<real_ext>` — `.auto` is replaced with the engine's actual extension |
| `result "path/to/file"` | Copy to that path (old behavior — use quotes for custom paths) |

**Why no auto-extension by default?** To make sure the user really knows the real file name. When you write `result file1`, you get `results/file1.wav` (or `.mp3`, `.mp4`, etc. — whatever the engine produced). You know the exact name because VODER prints `Result saved as: results/file1.wav`. If you want a specific extension, say so explicitly — VODER won't guess.

### Examples

**Example 1: Sequential pipeline (TTS → SE)**
```bash
python voder.py tts script "hello" voice "narrator" result greeting "&&" se results/greeting.wav result clean
```
- Command 1: TTS generates speech → saved as `results/greeting.wav`
- Command 2: SE enhances `results/greeting.wav` → saved as `results/clean.wav`

**Example 2: Bidirectional (TTS → SVS → STS references both)**
```bash
python voder.py tts script "hello" result orig "&&" svs results/orig.wav voice result vocals "&&" sts base results/vocals.wav target results/orig.wav result converted
```
- Command 1: TTS → `results/orig.wav`
- Command 2: SVS extracts vocals from `orig.wav` → `results/vocals.wav`
- Command 3: STS converts `vocals.wav` using `orig.wav` as the target voice → `results/converted.wav`
- **Bidirectional:** command 3 references BOTH `vocals.wav` (from command 2) AND `orig.wav` (from command 1). You cannot do this with regular `chains` — chains are strictly linear.

**Example 3: Mix extended commands with regular chains**
```bash
python voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "vocals" svs voice "song" result song_vocals "&&" sts base results/song_vocals.wav target "ref.wav" result cover
```
- Command 1: Regular chains pipeline — TTM generates music, SVS extracts vocals → last chain output saved as `results/song_vocals.wav`
- Command 2: STS voice conversion on the extracted vocals → `results/cover.wav`

**Example 4: Independent parallel commands (no file sharing)**
```bash
python voder.py tts script "greeting" voice "narrator" result greeting "&&" sfx sound "thunder rumbling" duration 5 result thunder
```
- Two completely independent commands, each producing a named result. Useful for batch-generating assets in one line.

**Example 5: Quest download → process**
```bash
python voder.py quest download "https://youtube.com/watch?v=..." result podcast "&&" stt results/podcast.mp3 timestamp result transcript "&&" se results/podcast.mp3 result clean_audio
```
- Command 1: Download YouTube audio → `results/podcast.mp3`
- Command 2: Transcribe the podcast → `results/transcript.txt`
- Command 3: Enhance the podcast audio → `results/clean_audio.wav`

### `&&` vs `chains` — when to use which

| Feature | `chains` (section 10) | `&&` extended commands |
|---------|----------------------|------------------------|
| Separator | ` / ` (space slash space) | `"&&"` (quoted) |
| Reference style | By chain name (resolved internally) | By full file path |
| Direction | Linear (each chain references the previous) | Any direction (bidirectional, skip-ahead, parallel) |
| Intermediate outputs | Stored in `temp_chains/` (hidden) | Stored in `results/` (visible, named) |
| Use case | Linear pipeline with named steps | Multi-step workflows where later steps reference earlier outputs non-linearly |

**Rule of thumb:** if your workflow is a straight line (A → B → C → D), use `chains`. If you need to reference an earlier output out of order (A → B → C, but C also references A), use `&&`.

---

## Input Types

Most modes that accept file paths also support (see exceptions below):

| Input Type | Description |
|------------|-------------|
| Local audio | `.wav`, `.mp3`, `.flac`, `.ogg`, etc. |
| Local video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.wmv`, `.ts`, `.mts` |
| YouTube URL | `https://youtube.com/watch?v=...`, `https://youtu.be/...`, `https://youtube.com/shorts/...`, etc. |
| TikTok URL | `https://www.tiktok.com/@user/video/...`, `https://vm.tiktok.com/...`, etc. |
| Bilibili URL | `https://www.bilibili.com/video/...`, `https://b23.tv/...` |
| Snapchat URL | `https://www.snapchat.com/spotlight/...`, `https://www.snapchat.com/u/...`, etc. |
| Instagram URL | `https://www.instagram.com/reel/...`, `https://www.instagram.com/p/...`, etc. |
| Facebook URL | `https://www.facebook.com/watch?v=...`, `https://fb.watch/...`, etc. |
| X / Twitter URL | `https://twitter.com/<user>/status/...`, `https://x.com/<user>/status/...`, `https://t.co/...` |

> **Note:** All URL types go through the same universal URL handler (`src/url_handler.py`). The handler runs a two-step detection: first a shape check (host + path patterns per platform, instant and offline) that rejects channel pages, profiles, playlists, and other non-video URLs; then a `yt-dlp` video verification step (online, `download=False`) that confirms the link actually resolves to a downloadable video stream before downloading. Short-link domains (`youtu.be`, `b23.tv`, `vm.tiktok.com`, `fb.watch`, `t.co`, etc.) are recognized as video URLs by default.

Video files are automatically handled: audio is extracted for processing, then merged back with the original video track for output (where applicable).

---

## Output

- All outputs are saved to the `results/` directory in the current working directory.
- Output filenames follow the pattern: `voder_<mode>[_<detail>]_<timestamp>.<ext>`
- Use `result "<path>"` to copy the latest output to a custom location.
