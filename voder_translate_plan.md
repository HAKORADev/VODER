# VODER TranslateGemma + Dub Implementation Plan

## Overview
Add TranslateGemma 12B (any-to-any translation) to VODER, upgrade existing translate features from Whisper any-to-English to true any-to-any, and implement a new TTS `dub` subtask for video/audio dubbing.

## Architecture

### New Model: TranslateGemma
- **Model**: `google/translategemma-12b-it` (Gemma 3-based, 55 languages, 2K context)
- **Loader**: `AutoModelForImageTextToText` + `AutoProcessor` from transformers
- **Dir**: `MODELS_CHECKPOINTS_DIR/translate_gemma`
- **Dtype**: `torch.bfloat16` on CUDA, `torch.float32` on CPU
- **Pattern**: Same lazy-load + cleanup as other models (ensure_model/cleanup)

### Language Spec Syntax
- `translate` → default any-to-English (backward compat, now uses TranslateGemma)
- `translate (auto-en)` → auto-detect source, English output
- `translate (ar-en)` → Arabic to English
- `translate (en-ar)` → English to Arabic
- `translate (auto-ar)` → auto-detect source, Arabic output
- Parser regex: `\((\w+)-(\w+)\)` after `translate` keyword
- Stored as: `params['translate_langs'] = {'source': 'auto', 'target': 'en'}`

### Backward Compatibility
- `stt translate` without `()` → still works, now uses TranslateGemma any-to-English instead of Whisper
- `tts slc` without `translate` → still works (any-to-English default)
- `tts slc translate (auto-ar)` → NEW: any-to-Arabic
- Remove mutual exclusivity of `translate + overdose` and `translate + subtitle` (TranslateGemma is separate from ASR)

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
- If bare `translate` without `()`: `translate_langs = {'source': 'auto', 'target': 'en'}`
- Validate lang codes against TranslateGemma supported list (55 languages)

### Step 3: Shared Translation Helper
File: `src/voder.py`
- `_translate_with_gemma(text, source_lang, target_lang)`: load, translate, cleanup
- `_translate_segments_with_gemma(segments, source_lang, target_lang)`: batch translate

### Step 4: Upgrade STT Translate (Any-to-Any)
File: `src/voder.py` in `oneline_stt()`
- Remove mutual exclusivity: `overdose + translate` now allowed
- Remove mutual exclusivity: `subtitle + translate` now allowed
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
- Optional `translate (source-target)` → translate to target language
- Optional `video "path"` keyword → explicitly mark video input (auto-detected if subtitle used)

**Dub Pipeline:**
1. Input: audio/video file or URL
2. If URL → download video
3. If video → extract audio + keep video path
4. SVS: separate vocals from instrumentals
5. VibeVoice ASR: transcribe vocals with word timestamps + speaker labels
6. Speaker detection: if dialogue detected → pyannote diarization
7. For each speaker:
   a. Extract speaker audio clip (SS-style)
   b. Clone voice for Fish S2 Pro
8. TranslateGemma: translate text with timing-aware prompt
   - Provide: original text, timestamps, total duration, word count
   - Prompt engineering: "Translate to [target]. Keep concise to match timing."
9. Fish S2 Pro TTS: generate speech per speaker segment
10. Speed adjustment per segment:
    - Get TTS output duration
    - Calculate speed factor: original_segment_duration / tts_duration
    - Apply ffmpeg atempo (0.5x-2.0x range, chain if needed)
    - Clamp to reasonable range (0.7x-1.5x)
11. Concatenate all adjusted segments
12. Mix new vocals with original instrumentals (ffmpeg amix)
13. If video:
    a. If subtitle: burn subtitles on video
    b. Else: mux new audio with video (replace audio track)
14. Output: dubbed video (.mp4) or audio (.wav)

**Speed Adjustment Helper:**
- `_adjust_audio_speed(input_path, target_duration, output_path)`:
  - Get input duration via ffprobe
  - Calculate atempo factor
  - Chain atempo filters if outside 0.5-2.0 range
  - Apply via ffmpeg

**Dub Command Examples:**
```
tts dub "video.mp4"
tts dub subtitle "video.mp4"
tts dub translate (auto-ar) "video.mp4"
tts dub translate (auto-ar) subtitle "video.mp4"
tts dub "audio.wav"
tts dub translate (en-ja) "audio.wav"
```

### Step 8: Dispatch Updates
File: `src/voder.py` in `execute_oneline_command()`
- TTS mode: check `params.get('dub')` → route to `oneline_tts_dub()`

### Step 9: Parser Updates for Dub
File: `src/voder.py` in `parse_oneline_args()`
- TTS mode: add `dub` keyword parsing
- `dub` keyword → `params['dub'] = True`
- After `dub`, peek for `translate`, `subtitle`, `video`, path

### Step 10: Usage/Help Updates
File: `src/voder.py` in `show_oneline_usage()`
- Add translate `(source-target)` syntax examples
- Add dub examples
- Update translate description

### Step 11: Doc Updates
Files: Guide.md, Bots.md, COMMAND_CATALOG.md, voder-skill.md, CHANGELOG.md
- Add TranslateGemma to model stack
- Document any-to-any translate syntax
- Document dub subtask
- Add examples and tips
- Update changelog

---

## Supported Language Codes (TranslateGemma 55 languages)
af, am, ar, az, be, bg, bn, bs, ca, cs, cy, da, de, el, en, es, et, eu, fa, fi, fr, ga, gl, gu, ha, he, hi, hr, hu, id, is, it, ja, jv, ka, kk, km, kn, ko, lo, lt, lv, mk, ml, mn, mr, ms, mt, my, ne, nl, no, pa, pl, ps, pt, ro, ru, si, sk, sl, so, sq, sr, sv, sw, ta, te, tg, th, tk, tl, tr, uk, ur, uz, vi, yo, zh

## Memory Requirements
- TranslateGemma 12B: ~24GB VRAM (bfloat16) or ~48GB RAM (float32 CPU)
- Dub pipeline total: TranslateGemma 12B + Fish S2 Pro + VibeVoice ASR (not loaded simultaneously)
- Stage-by-stage loading/unloading to fit in 24GB VRAM

## Key Design Decisions
1. TranslateGemma loaded/unloaded per-stage (never co-exists with other heavy models)
2. Speed adjustment uses ffmpeg atempo (no ML, fast, reliable)
3. Multi-speaker dub uses pyannote for diarization, then per-speaker cloning
4. Overlapping speakers: best-effort, noted as limitation
5. Multilingual input: not supported (limitation, same as current STT)
6. No code comments per user preference
7. Modular helpers to avoid code duplication between STT/SLC/Dub

---

## Progress Tracking

- [ ] Step 1: TranslateGemma class
- [ ] Step 2: Language spec parser
- [ ] Step 3: Shared translation helpers
- [ ] Step 4: Upgrade STT translate (any-to-any)
- [ ] Step 5: Upgrade STT subtitle with translate
- [ ] Step 6: Upgrade SLC with any-to-any
- [ ] Step 7: TTS dub subtask
- [ ] Step 8: Dispatch updates
- [ ] Step 9: Parser updates for dub
- [ ] Step 10: Usage/help updates
- [ ] Step 11: Doc updates
- [ ] Push
