# VODER TranslateGemma + Dub Implementation Plan

## Overview
Add TranslateGemma 12B (any-to-any translation) to VODER, upgrade existing translate features from Whisper any-to-English to true any-to-any, and implement a new TTS `dub` subtask for video/audio dubbing.

## Architecture

### New Model: TranslateGemma
- **Model**: `google/translategemma-12b-it` (Gemma 3-based, 76 languages, 2K context)
- **Loader**: `AutoModelForImageTextToText` + `AutoProcessor` from transformers
- **Dir**: `MODELS_CHECKPOINTS_DIR/translate_gemma`
- **Dtype**: `torch.bfloat16` on CUDA, `torch.float32` on CPU
- **Pattern**: Same lazy-load + cleanup as other models (ensure_model/cleanup)

### Language Spec Syntax
- `translate` → default any-to-English via Whisper (backward compat)
- `translate (auto-en)` → auto-detect source, English output via TranslateGemma
- `translate (ar-en)` → Arabic to English via TranslateGemma
- `translate (en-ar)` → English to Arabic via TranslateGemma
- `translate (auto-ar)` → auto-detect source, Arabic output via TranslateGemma
- Parser regex: `\((\w+)-(\w+)\)` after `translate` keyword
- Stored as: `params['translate_langs'] = {'source': 'auto', 'target': 'en'}`

### Backward Compatibility
- `stt translate` without `()` → still uses Whisper any-to-English (unchanged)
- `stt translate (auto-en)` → uses TranslateGemma any-to-English
- `tts slc` without `translate` → still works (any-to-English default)
- `tts slc translate (auto-ar)` → NEW: any-to-Arabic
- Remove mutual exclusivity of `translate (source-target) + overdose` and `translate (source-target) + subtitle` (TranslateGemma is separate from ASR)
- Bare `translate` remains mutually exclusive with `overdose` (uses Whisper's built-in)

### Dub Default Behavior
- `tts dub "video.mp4"` → auto-translates to English by default (no translate keyword needed)
- `tts dub translate (auto-ja) "video.mp4"` → overrides target to Japanese
- Dub always translates (default auto→English); `translate` keyword only overrides the target language

---

## Implementation Steps

### Step 1: TranslateGemma Class
File: `src/voder.py`
- Add `TRANSLATE_GEMMA_DIR` constant at top
- Add `class TranslateGemma` with:
  - `__init__()`: set model_dir, model=None, processor=None
  - `ensure_model()`: download from HF if not cached, load model + processor
  - `translate(text, source_lang, target_lang)`: single text translation
  - `translate_segments(segments, source_lang, target_lang)`: batch translate
  - `cleanup()`: del model/processor, gc.collect(), torch.cuda.empty_cache()
- No in-code comments

### Step 2: Language Spec Parser
File: `src/voder.py` in `parse_oneline_args()`
- In STT mode: detect `translate` keyword, then peek next arg for `(...)` pattern
- In TTS mode (SLC): same pattern after `slc translate`
- In TTS mode (dub): same pattern after `dub translate`
- Store as `params['translate_langs'] = {'source': str, 'target': str}`
- If bare `translate` without `()`: uses Whisper any-to-English (no lang_spec)
- Validate lang codes against TranslateGemma supported list (76 languages)

### Step 3: Shared Translation Helper
File: `src/voder.py`
- `_translate_with_gemma(text, source_lang, target_lang)`: load, translate, cleanup
- `_translate_segments_with_gemma(segments, source_lang, target_lang)`: batch translate

### Step 4: Upgrade STT Translate (Any-to-Any)
File: `src/voder.py` in `oneline_stt()`
- Remove mutual exclusivity: `overdose + translate (source-target)` now allowed
- Remove mutual exclusivity: `subtitle + translate (source-target)` now allowed
- Flow with TranslateGemma:
  - If overdose: VibeVoice ASR → TranslateGemma translate
  - If basic: Whisper transcribe → TranslateGemma translate
  - Default: any-to-English, with `(source-target)`: any-to-any
- If `translate_langs['target']` != 'en': output text in target language

### Step 5: Upgrade STT Subtitle with Translate
File: `src/voder.py` in `oneline_stt_subtitle()`
- Accept `translate` flag with `(source-target)` syntax
- After VibeVoice ASR → TranslateGemma translate → burn translated subtitles
- `stt subtitle translate (auto-ar) "video.mp4"` → Arabic subtitles

### Step 6: Upgrade SLC with Any-to-Any
File: `src/voder.py` in `oneline_tts()` SLC branch
- Parse `translate (source-target)` after `slc`
- After Whisper transcription → TranslateGemma translate to target language
- TTS generates speech in target language
- `tts slc translate (auto-ar) "song.wav"` → Arabic speech with original voice

### Step 7: TTS Dub Subtask
File: `src/voder.py`
- New keyword `dub` in TTS mode parsing
- `dub` implies `extreme` (Fish S2 Pro) for best quality
- Optional `subtitle` keyword → also burn subtitles
- Optional `translate (source-target)` → override target language (default: auto→English)
- Optional `video "path"` keyword → explicitly mark video input (auto-detected if subtitle used)

**Dub Pipeline (per-segment with timeline assembly):**
1. Input: audio/video file or URL
2. If URL → download video
3. If video → extract audio + keep video path
4. SVS: separate vocals from instrumentals (voice + music)
5. VibeVoice ASR `transcribe_with_events()`: transcribe vocals with audio events
   - Speech segments → get `start`, `end`, `speaker`, `text`
   - Audio events → get `is_event`, `event_type` (silence, music, noise)
   - Audio events are NOT translated, NOT dubbed
6. TranslateGemma: translate per-segment with timing-aware prompt
   - Provide: segment text, original duration, word count
   - Prompt: "Keep translation concise to match timing. Less is better than more."
7. Fish S2 Pro TTS: generate speech per segment (not per speaker)
   - Short segments avoid Fish drift/glitches
   - Voice cloned from entire source vocal track
8. Per-segment speed adjustment:
   - Get TTS output duration
   - Calculate speed factor: tts_duration / seg_duration
   - If ratio > 1.5 or < 0.5: apply ffmpeg atempo
   - Each segment independently adjusted to match its original timing
9. Timeline assembly:
   - Create silent base matching original audio duration
   - Overlay each TTS segment at its original `start` position
   - Uses `_overlay_segment_on_base()` (ffmpeg adelay + amix)
   - Non-speech segments stay silent (original music mixed separately)
10. Mix new vocals with original instrumentals (ffmpeg amix)
11. If video:
    a. If subtitle: burn subtitles on video
    b. Else: mux new audio with video (replace audio track)
12. Output: dubbed video (.mp4) or audio (.wav)

**VibeVoice ASR `transcribe_with_events()` Method:**
- New method on VibeVoiceASR class
- Returns segments with `is_event` (bool) and `event_type` (str) fields
- Preserves audio event tags: `[Silence]`, `[Lyric]`, `[Music]`, `[Noise]`, etc.
- Event-only segments (no text after tag): `is_event=True`, `text=""`
- Tagged speech segments (`[Lyric] text`): `is_event=False`, `event_type='lyric'`, `text='text'`
- Existing `transcribe()` method unchanged for backward compatibility

**Dub Command Examples:**
```
tts dub "video.mp4"                          → auto→English
tts dub subtitle "video.mp4"                 → auto→English + subtitles
tts dub translate (auto-ja) "video.mp4"      → auto→Japanese
tts dub translate (auto-ar) subtitle "video.mp4"  → auto→Arabic + subtitles
tts dub "audio.wav"                          → auto→English (audio only)
tts dub translate (en-ja) "audio.wav"        → English→Japanese (audio only)
```

### Step 8: Dispatch Updates
File: `src/voder.py` in `execute_oneline_command()`
- TTS mode: check `params.get('dub')` → route to `oneline_tts_dub()`

### Step 9: Parser Updates for Dub
File: `src/voder.py` in `parse_oneline_args()`
- TTS mode: add `dub` keyword parsing
- `dub` keyword → `params['dub'] = True`
- After `dub`, peek for `translate`, `subtitle`, `video`, path
- Default `dub_translate_langs` to `{'source': 'auto', 'target': 'en'}` when no translate specified

### Step 10: Usage/Help Updates
File: `src/voder.py` in `show_oneline_usage()`
- Add translate `(source-target)` syntax examples
- Add dub examples
- Update translate description

### Step 11: Doc Updates
Files: Guide.md, Bots.md, COMMAND_CATALOG.md, voder-skill.md, CHANGELOG.md, README.md, READ.md, Languages.md
- Add TranslateGemma to model stack
- Document any-to-any translate syntax
- Document dub subtask with per-segment pipeline
- Add examples and tips
- Update changelog
- Add TranslateGemma section to Languages.md
- Add dub section to READ.md
- Update README.md with dub and translate features

---

## Supported Language Codes (TranslateGemma 76 languages)
af, am, ar, az, be, bg, bn, bs, ca, cs, cy, da, de, el, en, es, et, eu, fa, fi, fr, ga, gl, gu, ha, he, hi, hr, hu, id, is, it, ja, jv, ka, kk, km, kn, ko, lo, lt, lv, mk, ml, mn, mr, ms, mt, my, ne, nl, no, pa, pl, ps, pt, ro, ru, si, sk, sl, so, sq, sr, sv, sw, ta, te, tg, th, tk, tl, tr, uk, ur, uz, vi, yo, zh

## Memory Requirements
- TranslateGemma 12B: ~24GB VRAM (bfloat16) or ~48GB RAM (float32 CPU)
- Dub pipeline total: TranslateGemma 12B + Fish S2 Pro + VibeVoice ASR (not loaded simultaneously)
- Stage-by-stage loading/unloading to fit in 24GB VRAM

## Key Design Decisions
1. TranslateGemma loaded/unloaded per-stage (never co-exists with other heavy models)
2. Speed adjustment uses ffmpeg atempo (no ML, fast, reliable)
3. Per-segment TTS generation (not per-speaker) — avoids Fish drift, better timing alignment
4. Timeline-based assembly (overlay at original positions) instead of simple concatenation
5. Audio events preserved for non-speech detection — silence/music/noise segments left undubbed
6. Dub defaults to auto→English (user doesn't need to specify translate for English target)
7. Bare `translate` remains mutually exclusive with `overdose` (uses Whisper, not TranslateGemma)
8. `translate (source-target)` is compatible with `overdose` (uses TranslateGemma, decoupled from ASR)
9. No code comments per user preference
10. Modular helpers to avoid code duplication between STT/SLC/Dub

---

## Progress Tracking

- [x] Step 1: TranslateGemma class
- [x] Step 2: Language spec parser
- [x] Step 3: Shared translation helpers
- [x] Step 4: Upgrade STT translate (any-to-any)
- [x] Step 5: Upgrade STT subtitle with translate
- [x] Step 6: Upgrade SLC with any-to-any
- [x] Step 7: TTS dub subtask (per-segment + timeline assembly + audio events)
- [x] Step 8: Dispatch updates
- [x] Step 9: Parser updates for dub (default auto→English)
- [x] Step 10: Usage/help updates
- [x] Step 11: Doc updates (README.md, READ.md, Languages.md, Guide.md, Bots.md, COMMAND_CATALOG.md, voder-skill.md, CHANGELOG.md)
- [ ] Push

## Recent Improvements (this session)

### Dub Pipeline v2 Improvements:
1. **`transcribe_with_events()`** — New VibeVoiceASR method that preserves audio events (`[Silence]`, `[Lyric]`, `[Music]`, `[Noise]`, etc.) alongside speech segments. Audio events tagged with `is_event=True` and `event_type` fields. Used by dub to identify non-speech segments that should not be translated.

2. **Per-segment TTS generation** — Instead of generating one big TTS output per speaker, the dub pipeline now generates TTS per speech segment. This avoids Fish S2 Pro drift/glitches on long text, and enables per-segment timing alignment.

3. **Per-segment speed adjustment** — Each TTS segment is independently speed-adjusted to match its original segment duration. Threshold: speed_ratio > 1.5 or < 0.5 (was 1.3/0.7 per-speaker).

4. **Timeline-based assembly** — Instead of simple concatenation, the dub pipeline now builds a silent base matching the original audio duration and overlays each TTS segment at its original `start` position using `_overlay_segment_on_base()` (ffmpeg adelay + amix). This produces near-perfect timing alignment.

5. **Dub defaults to auto→English** — When no `translate` keyword is specified with `dub`, the pipeline defaults to translating from auto-detected source to English. The `translate (source-target)` syntax only overrides the target language.

6. **Helper functions** — New `_overlay_segment_on_base()`, `_extract_audio_segment()`, and `_build_dub_timeline()` helpers for the dub pipeline.

### Doc Updates:
- Languages.md: Added full TranslateGemma 12B section with 76 languages, syntax, usage table, technical notes. Updated VibeVoice ASR section for dub pipeline and `transcribe_with_events()`. Updated SVS modes. Added dub/translate workflows to Cross-Language section.
- READ.md: Added section 1.6 Dub (Video/Audio Dubbing) with full pipeline description and CLI examples. Updated SLC section with any-to-any translation. Added dub CLI examples to one-line commands. Updated STT examples with translate (source-target). Added TranslateGemma to AI Model Integration.
- README.md: Updated features, added dub and any-to-any translation, updated modes table, added TranslateGemma to models table, updated quick start with dub examples.
- Guide.md: Rewrote dub section for per-segment pipeline. Updated all "55 languages" to "76 languages". Added `transcribe_with_events()` to VibeVoice section.
- Bots.md: Updated dub section for auto→English default. Updated all "55 languages" to "76 languages". Added translate (source-target) examples.
- COMMAND_CATALOG.md: Updated dub section. Updated all "55 languages" to "76 languages". Added subtitle translate examples.
- voder-skill.md: Updated dub section for per-segment pipeline. Updated all "55 languages" to "76 languages".
- CHANGELOG.md: Updated dub entries for per-segment pipeline, audio events, timeline assembly, auto→English default.
