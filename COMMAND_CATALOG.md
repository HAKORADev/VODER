# VODER Command Catalog

> Complete reference of every oneline mode, its flags, keywords, and syntax.
> Sorted by mode order.

---

## Invocation

```
python voder.py <mode> [keyword] [value] [keyword] [value] ...
python voder.py cli              # interactive CLI mode (no oneline commands)
python voder.py                  # launch GUI (no commands)
```

---

## Mode Index

| Mode | Name |
|------|------|
| `tts` | Text-to-Speech |
| `sts` | Speech-to-Speech (Voice Conversion) |
| `ttm` | Text-to-Music (generate / remix / repaint / complete / lego / extract / bgm) |
| `stt` | Speech-to-Text (Transcription) |
| `se` | Speech Enhancement |
| `sfx` | Sound Effects Generation |
| `svs` | Song Voice Separate |
| `slc` | Speaker Language Conversion |
| `ss` | Speakers Separator |

### Quick Jump

| Mode | Section |
|------|---------|
| [Invocation](#invocation) | General syntax & modes |
| [Global Keywords](#global-keywords-available-in-all-modes) | `result` |
| [1. TTS](#1-tts--text-to-speech) | Text-to-Speech, dialogue, directives |
| [2. STS](#2-sts--speech-to-speech-voice-conversion) | Voice Conversion |
| [3. TTM](#3-ttm--text-to-music) | Generate, VC, Remix, Repaint, Complete, Lego, Extract |
| [4. STT](#4-stt--speech-to-text-transcription) | Transcription, diarization, translate |
| [5. SE](#5-se--speech-enhancement) | Denoise, dereverb |
| [6. SFX](#6-sfx--sound-effects-generation) | Sound effects |
| [7. SVS](#7-svs--song-voice-separate) | Vocal/instrument separation |
| [8. SLC](#8-slc--speaker-language-conversion) | Language conversion |
| [9. SS](#9-ss--speakers-separator) | Speaker extraction & separation |
| [Input Types](#input-types) | Supported file & URL formats |
| [Output](#output) | Output directory & naming |
| [Rejected Modes](#rejected-modes) | `stt+tts` etc. |

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

Generate speech from text using voice descriptions (VoiceDesign) or voice clone targets.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `script` | `"<text>"` or `"CharName: text"` | Dialogue line (plain text for single mode, `Character: text` for dialogue mode). Can appear multiple times. |
| `voice` | `"<description>"` or `"CharName: description"` | Voice prompt for VoiceDesign TTS. Single mode: one prompt. Dialogue mode: `"CharName: description"` per character. Can appear multiple times. |
| `target` | `"<path>"` or `"CharName: path"` | Audio path for voice cloning. Single mode: one path. Dialogue mode: `"CharName: path"` per character. Can appear multiple times. |
| `music` | `"<description>"` | Background music description (dialogue mode only). Generated via ACE-Step and mixed under speech. |
| `level` | `"<spec>"` | Music volume levels per dialogue segment, e.g. `"10:20-50 30:60-80"`. Format: `<volume%>:<start_sec>-<end_sec>`. Default: 35%. Dialogue mode only. |
| `reference` | `"<path>"` | Optional reference audio/video/URL for dialogue background music style guidance. Processed through SVS music pipe to extract clean instrumental before use. Accepts audio files, video files, and YouTube/TikTok/Bilibili URLs. Dialogue mode only. |
| `ocr` | `"<image_path>"` | Extract text from an image via EasyOCR, then use that text as the script. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP. |
| `<number>` | `10-300` | Duration in seconds (TTM only, ignored in pure TTS). |
| `overdose` | (flag) | Use VibeVoice ASR for dialogue source analysis and voice clip extraction instead of Whisper + pyannote. When used with `music`, also uses ACE-Step XL turbo for enhanced background music quality. Requires 24GB+ VRAM or 48GB+ RAM. |

### Single Mode

One speaker, one line. Use `voice` for VoiceDesign or `target` for voice clone.

```
# VoiceDesign: describe the voice
python voder.py tts script "hello world" voice "male voice"

# Voice clone: provide a reference audio
python voder.py tts script "hello" target "voice.wav"

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
```

### Overdose Notes

- When `overdose` is used with audio as dialogue source, VibeVoice ASR replaces Whisper + pyannote for transcription and diarization.
- Voice clip extraction with overdose automatically trims 2s from start and 3s from end of longest segment to avoid cross-speaker overlap.
- `music` parameter with `overdose` uses ACE-Step XL turbo instead of the standard model for enhanced background music quality.

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

## 2. `sts` — Speech-to-Speech (Voice Conversion)

Convert voice from a base audio to match a target voice.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `base` | `"<path>"` | Source audio/video file path or YouTube/TikTok/Bilibili URL. The audio whose content will be preserved. |
| `target` | `"<path>"` | Reference voice audio. The voice characteristics to apply. Auto-extracts clean vocals. |
| `music` | (flag) | Use Seed-VC v1 (44.1kHz music model) instead of v2 (22.05kHz speech model). Input must be audio (not video). Auto-extracts vocals from target. |
| `mimic` | (flag) | Convert style + voice (not just voice). Uses Seed-VC v2 with `convert_style=True`. Cannot be combined with `music`. Input must be audio (not video). |

### Rules

- `music` and `mimic` cannot be used together.
- Base can be audio or video in standard mode. `music` and `mimic` require audio input only (video is rejected).
- Target vocals are automatically cleaned via SVS before conversion.
- Output is upsampled to 44100Hz.
- Output filenames: music mode uses `voder_m_sts_*.wav`, standard/mimic uses `voder_sts_*.wav`.

```
# Standard voice conversion (speech)
python voder.py sts base "input.wav" target "voice.wav"

# Music voice conversion (44.1kHz model)
python voder.py sts base "input.wav" target "voice.wav" music

# Style + voice mimic
python voder.py sts base "input.wav" target "voice.wav" mimic

# Video input (extracts audio, converts, merges back)
python voder.py sts base "input.mp4" target "voice.wav"
```

---

## 3. `ttm` — Text-to-Music

The most feature-rich mode. Supports generation, remix, repaint, voice cloning, and three sub-tasks (complete, lego, extract).

### Global TTM Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `lyrics` | `"<text>"` | Song lyrics text. Write `\n` for line breaks (parsed as actual newlines). |
| `styling` | `"<text>"` | Style/mood prompt for the music. Write `\n` for line breaks (parsed as actual newlines). |
| `<number>` | `10-300` | Duration in seconds (for generate and VC paths). |
| `overdose` | (flag) | Use Overdose tier (ACE-Step XL-Turbo + 4B LM + shift 3.0) instead of Standard tier (ACE-Step 1.5 Turbo). |
| `result` | `"<path>"` | Copy output to custom path (see Global Keywords). |

---

### 3a. Standard Generate

Basic text-to-music generation. Supports optional reference audio via `target`.

| Keyword | Value | Description |
|---------|-------|-------------|
| `target` | `"<path>"` | Reference audio (as-is). |
| `target voice` | `"<path>"` | Reference audio — extract vocals via SVS first. |
| `target music` | `"<path>"` | Reference audio — extract instruments via SVS first. |

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
```

### 3b. Voice Cloning (`vc`)

Generate music then convert the vocal to match a clone voice via Seed-VC v1.

| Keyword | Value | Description |
|---------|-------|-------------|
| `vc` | (flag) | Enable voice cloning mode. |
| `clone` | `"<path>"` | Source voice audio for cloning. Auto-extracts clean vocals. |
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
- Reference can be a local file, video file, or YouTube/TikTok/Bilibili URL.

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

Re-generate a specific time range of a song in a new style.

| Keyword | Value | Description |
|---------|-------|-------------|
| `repaint` | `"<path>"` | Source audio/video file or YouTube/TikTok/Bilibili URL to repaint. |
| `styling` | `"<text>"` | New style prompt for the repainted section. |
| `time:start-end` | `"<start>-<end>"` | Time range in seconds (e.g., `time:20-80` or `time:20.5-80.5`). Required. Supports float values. |
| `lyrics` | `"<text>"` | Optional lyrics for the repainted section. Defaults to `"..."` if omitted. |
| `bias` | `"<0-100>"` | Cover strength bias (same logic as remix). Default: 40. |
| `reference` | `[voice/music] "<path>" [voice/music "<path>" ...]` | Optional reference audio(s). Up to 3 with optional `voice`/`music` prefix per entry. Multiple refs are composed into a 30s composite. Supports URLs and video files. |
| `overdose` | (flag) | Use Overdose tier. |

#### Rules

- `repaint` requires `styling` and `time:start-end`.
- Start must be less than end. If end exceeds audio duration, it is clamped. If start exceeds duration, it produces an error.
- Cannot be combined with `vc`.
- `reference voice` extracts vocals from the reference via SVS before use.
- `reference music` extracts instruments from the reference via SVS before use.
- `reference "<path>"` uses the reference audio as-is.
- Up to 3 references; excess entries produce a warning and are trimmed.
- Multiple references are composed into a 30s composite (same logic as remix).
- Reference can be a local file, video file, or YouTube/TikTok/Bilibili URL.

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
```

---

### 3e. Complete Sub-Task

Add missing instruments to an existing track. Uses ACE-Step XL-Base + 1.7B LM + shift 1.0 (50 inference steps).

| Keyword | Value | Description |
|---------|-------|-------------|
| `complete` | (flag) | Enable complete sub-task. |
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili URL (positional, after all keywords). |
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
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili URL (positional). |
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
- Source and references accept audio files, video files, and YouTube/TikTok/Bilibili URLs.

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
| `"<path>"` | source | Source audio/video file or YouTube/TikTok/Bilibili URL (positional). |
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
| `bgm` | `"<path>"` | Source audio/video file or YouTube/TikTok/Bilibili URL whose background music will be replaced. |
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
| `"<path>"` | file | Audio/video/image file path or YouTube/TikTok/Bilibili URL. Can specify multiple files (each is transcribed separately). |
| `timestamp` | (flag) | Keep Whisper word-level timestamps in the output. |
| `dialogue` | (flag) | Enable speaker diarization (requires HF_TOKEN and pyannote model access). |
| `translate` | (flag) | Translate transcription to English (uses Whisper large-v3 model). |
| `se` | (flag) | Apply speech enhancement before transcription (denoise/dereverb input first). |
| `overdose` | (flag) | Use VibeVoice ASR (requires 24GB+ VRAM or 48GB+ RAM). Falls back to Whisper + pyannote if unavailable. |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- `overdose` cannot be combined with `translate`.
- Multiple files are processed sequentially.
- Output is saved as `.txt` in the `results/` directory.
- **Pipeline:** SVS voice isolation is always applied before transcription. With `se`, speech enhancement runs first.
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

# Full combination
python voder.py stt "audio.wav" translate timestamp dialogue

# From YouTube
python voder.py stt "https://youtube.com/watch?v=..."
```

---

## 5. `se` — Speech Enhancement

Denoise, dereverb, and restore speech audio using UniSE.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `"<path>"` | file | Audio/video file path or URL. Can specify multiple files (each is enhanced separately). |
| `result` | `"<path>"` | Copy output to custom path. |

### Notes

- Outputs 16kHz audio (speech-focused, not for musical enhancement).
- Video input: outputs `.mp4` with enhanced audio track.
- Audio input: outputs `.wav`.
- Supports local audio/video files and URLs (YouTube, TikTok, Bilibili).

```
# Enhance a single audio file
python voder.py se "noisy_audio.wav"

# Enhance multiple files
python voder.py se "audio1.wav" "audio2.wav"

# Enhance video audio track
python voder.py se "noisy_video.mp4"

# Enhance from URL
python voder.py se "https://youtube.com/watch?v=..."
```

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
| `"<path>"` | file | Audio/video file path or YouTube/TikTok/Bilibili URL. |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- At least one of `voice`, `music`, or `both` is required.
- `both` extracts both vocals and instruments, producing two output files.
- Video input: outputs `.mp4` with separated audio merged back.
- Audio input: outputs `.wav`.
- YouTube/TikTok/Bilibili URLs: downloads video, separates, outputs `.mp4`.

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

# From YouTube
python voder.py svs music "https://youtube.com/watch?v=..."
```

---

## 8. `slc` — Speaker Language Conversion

Transcribe speech, translate (optional), then re-speak in the detected/target language while preserving the speaker's voice.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `translate` | (flag) | Force translation to English regardless of detected language. |
| `target` | `"<path>"` | Target voice reference audio. If omitted, the input audio itself is used as the voice reference. |
| `"<path>"` | file | Audio file path or YouTube/TikTok/Bilibili URL. |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- Input must be audio (video not supported).
- Pipeline: Whisper STT -> (optional translate) -> Qwen-TTS with voice cloning.
- If audio is already English and `translate` is specified, translation is skipped.
- If the detected language is not supported by Qwen-TTS, auto-translates to English.

```
# Convert language (auto-detect, keep voice)
python voder.py slc "french_speech.wav"

# Translate to English
python voder.py slc translate "french_speech.wav"

# With custom voice target
python voder.py slc "french_speech.wav" target "voice_ref.wav"

# Translate with custom voice
python voder.py slc translate "french_speech.wav" target "voice_ref.wav"

# From YouTube
python voder.py slc "https://youtube.com/watch?v=..."
```

---

## 9. `ss` — Speakers Separator

Extract all individual speakers from an audio source one by one, or extract a specific target speaker.

### Keywords

| Keyword | Value | Description |
|---------|-------|-------------|
| `"<path>"` | file | Audio/video file path or YouTube/TikTok/Bilibili URL. |
| `target` | `"<path>"` | Target voice reference audio/URL. When provided, extracts only the speaker matching this reference from the source audio. Outputs a single file containing the targeted speaker's content. The model looks at the target voice and tries to find/extract that speaker from the source. |
| `se` | (flag) | Apply speech enhancement before separation (denoise/dereverb the input first). |
| `overdose` | (flag) | Use VibeVoice ASR instead of Whisper + pyannote for transcription and diarization, providing better separation accuracy. Requires 24GB+ VRAM or 48GB+ RAM. **Skipped when `target` is provided** (target uses TSE extraction, not diarization). |
| `result` | `"<path>"` | Copy output to custom path. |

### Rules

- **Pipeline (no target):** SVS voice isolation -> STT + diarization -> per-speaker TSE extraction. Each detected speaker is output as a separate file.
- **Pipeline (with target):** SVS voice isolation -> TSE (Target Speaker Extraction). Looks at the target reference and extracts matching speaker from source. Outputs one file.
- `overdose` is only used in the no-target pipeline (switches from Whisper+pyannote to VibeVoice ASR for better accuracy). It is completely skipped when `target` is provided.
- Supports audio, video, YouTube, TikTok, and Bilibili URLs.
- `se` runs speech enhancement before anything else (cleaner input = better results).

```
# Extract all speakers (standard pipeline)
python voder.py ss "conversation.wav"

# Extract from video
python voder.py ss "interview.mp4"

# From YouTube
python voder.py ss "https://youtube.com/watch?v=..."

# With speech enhancement pre-processing
python voder.py ss se "noisy_conversation.wav"

# With overdose (better accuracy, uses VibeVoice ASR)
python voder.py ss overdose "conversation.wav"

# Extract specific target speaker from source (outputs one file)
python voder.py ss target "speaker_ref.wav" "conversation.wav"

# Target extraction from URL
python voder.py ss target "speaker_ref.wav" "https://youtube.com/watch?v=..."

# Overdose + speech enhancement combined
python voder.py ss overdose se "noisy_conversation.wav"
```

---

## Rejected Modes

| Input | Result |
|-------|--------|
| `stt+tts` / `stt_tts` / `stttts` | Rejected — requires interactive text editing. Use `tts` mode with your text, or `python voder.py cli` for interactive CLI. |

---

## Input Types

Most modes that accept file paths also support (see exceptions below):

| Input Type | Description |
|------------|-------------|
| Local audio | `.wav`, `.mp3`, `.flac`, `.ogg`, etc. |
| Local video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.wmv`, `.ts`, `.mts` |
| YouTube URL | `https://youtube.com/watch?v=...` — auto-downloads audio/video |
| TikTok URL | Auto-downloads audio/video |
| Bilibili URL | Auto-downloads audio/video |

> **Note:** Internally, all three URL types are handled by the same URL detection function. YouTube Shorts and Bilibili links are also supported.

**Exceptions:** `slc` does not support video input (audio only).

Video files are automatically handled: audio is extracted for processing, then merged back with the original video track for output (where applicable).

---

## Output

- All outputs are saved to the `results/` directory in the current working directory.
- Output filenames follow the pattern: `voder_<mode>[_<detail>]_<timestamp>.<ext>`
- Use `result "<path>"` to copy the latest output to a custom location.
