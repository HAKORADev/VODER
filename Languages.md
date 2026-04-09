# VODER — Language Support Reference

VODER is not an English‑only tool. The AI models it orchestrates collectively support over 99 languages across transcription, speech synthesis, music generation, and image text extraction. This document provides a complete breakdown of which languages each component supports, how language detection works, and what the current limitations are.

---

## Overview

| Component | Modes | Languages | Auto‑Detect | Notes |
|-----------|-------|-----------|-------------|-------|
| **Whisper** (`large-v3-turbo`) | STT, STT+TTS, Dialogue Source | 99 | Yes | Detects spoken language from audio |
| **Qwen3‑TTS VoiceDesign** | TTS | 10 + 2 dialects | Yes | Detects language from input text |
| **Qwen3‑TTS Base** | TTS+VC, STT+TTS | 10 + 2 dialects | Yes | Detects language from input text |
| **ACE‑Step 1.5** | TTM, TTM+VC, Background Music | 50 | Yes | Detects language from lyrics/caption |
| **EasyOCR** | STT, TTS, TTS+VC (image input) | 85 | No | Hardcoded to English in VODER |
| **TangoFlux** | SFX | 1 (English) | No | Text encoder trained on English only |
| **Seed‑VC v2** | STS | Any | N/A | Language‑agnostic (audio waveforms) |
| **Seed‑VC v1** | MSTS | Any | N/A | Language‑agnostic (audio waveforms) |
| **UniSE** | SE | Any | N/A | Language‑agnostic (audio waveforms) |
| **Pyannote** | Diarization | Any | N/A | Language‑agnostic (voice embeddings) |

---

## Whisper — Speech‑to‑Text

**Model:** `openai/whisper` → `large-v3-turbo`
**Modes:** STT, STT+TTS, Dialogue Source Analysis
**Language handling:** Auto‑detects spoken language from the first 30 seconds of audio. No user configuration required. Language can be manually overridden via the `language` parameter in Whisper's API, but VODER uses auto‑detection by default.

**Supported languages (99 total):**

```
en  English          zh  Chinese           de  German            es  Spanish
ru  Russian          ko  Korean            fr  French            ja  Japanese
pt  Portuguese       it  Italian           nl  Dutch             pl  Polish
tr  Turkish          ar  Arabic            sv  Swedish           ca  Catalan
id  Indonesian       hi  Hindi             fi  Finnish           vi  Vietnamese
he  Hebrew           uk  Ukrainian         el  Greek             ms  Malay
cs  Czech            ro  Romanian          da  Danish            hu  Hungarian
ta  Tamil            no  Norwegian         th  Thai              ur  Urdu
hr  Croatian         bg  Bulgarian         lt  Lithuanian        la  Latin
mi  Maori            ml  Malayalam         cy  Welsh             sk  Slovak
te  Telugu           fa  Persian           lv  Latvian           bn  Bengali
sr  Serbian          az  Azerbaijani       sl  Slovenian         kn  Kannada
et  Estonian         mk  Macedonian        br  Breton            eu  Basque
is  Icelandic        ne  Nepali            mn  Mongolian         bs  Bosnian
kk  Kazakh           sq  Albanian          sw  Swahili           gl  Galician
mr  Marathi          pa  Punjabi           si  Sinhala           km  Khmer
sn  Shona            yo  Yoruba            so  Somali            af  Afrikaans
oc  Occitan          ka  Georgian          be  Belarusian        tg  Tajik
sd  Sindhi           gu  Gujarati          am  Amharic           yi  Yiddish
lo  Lao              uz  Uzbek             fo  Faroese           ht  Haitian Creole
ps  Pashto           tk  Turkmen           nn  Nynorsk           mt  Maltese
sa  Sanskrit         lb  Luxembourgish    my  Myanmar          bo  Tibetan
tl  Tagalog          mg  Malagasy          as  Assamese         tt  Tatar
haw Hawaiian          ln  Lingala           ha  Hausa             ba  Bashkir
jw  Javanese          su  Sundanese         yue Cantonese
```

**Technical notes:**
- `large-v3-turbo` does **not** support the translation task (speech to English text). Transcription outputs in the original spoken language.
- When the `dialogue` flag is enabled, Whisper transcription is combined with Pyannote diarization and aligned using a three‑tier overlap matching system. The language of the transcription matches the spoken language.
- Processing is CPU‑only. Approximately 1x real‑time on a modern CPU.

---

## Qwen3‑TTS VoiceDesign — Text‑to‑Speech

**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
**Modes:** TTS (single and dialogue)
**Language handling:** Auto‑detects language from the input text. Set to `"Auto"` by default in VODER. The model reads the text content and determines the appropriate language without any user configuration.

**Supported languages (10 total):**

| Language | Full Name |
|----------|-----------|
| Chinese | Mandarin Chinese |
| English | English |
| Japanese | Japanese |
| Korean | Korean |
| German | German |
| French | French |
| Russian | Russian |
| Portuguese | Portuguese |
| Spanish | Spanish |
| Italian | Italian |

**Chinese dialects (2):**

| Dialect | Associated Speaker | Description |
|---------|-------------------|-------------|
| Beijing Mandarin | Dylan | Youthful Beijing male voice |
| Sichuan Mandarin | Eric | Lively Chengdu male voice |

**Technical notes:**
- Language validation is case‑insensitive. Both `"English"` and `"english"` are accepted.
- `"Auto"` is the default and works reliably for all 10 languages. If the target language is known, setting it explicitly can improve consistency in ambiguous cases (e.g., mixed‑language text).
- The model uses full English language names as identifiers (e.g., `"Chinese"`, `"English"`), not ISO codes.

---

## Qwen3‑TTS Base — Text‑to‑Speech with Voice Cloning

**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
**Modes:** TTS+VC, STT+TTS
**Language handling:** Auto‑detects language from the input text. Set to `"Auto"` by default in VODER. This is the same language detection system used by VoiceDesign — the Base model variant supports it identically.

**Supported languages:** Same 10 languages + 2 dialects as VoiceDesign (see above).

**Technical notes:**
- Voice cloning extracts an x‑vector speaker embedding from the reference audio, which is language‑independent. A Chinese reference audio can be used to clone a voice that speaks English, Japanese, or any other supported language.
- In dialogue mode, the voice embedding is extracted **once per character** at the start and reused for all their lines, ensuring consistent voice quality regardless of language changes between lines.
- The `generate_voice_clone()` method supports batch language lists, but VODER passes `"Auto"` for all lines.

---

## ACE‑Step 1.5 — Music Generation

**Model:** `ACE-Step/Ace-Step1.5` → `acestep-v15-turbo`
**Modes:** TTM, TTM+VC, Background Music (dialogue)
**Language handling:** Language is auto‑detected from the lyrics or caption text by the language model (Qwen3‑based). The `vocal_language` parameter defaults to `"unknown"`, which triggers automatic detection. Language can be set explicitly if needed.

**Supported languages for lyrics (50 total):**

```
ar  Arabic        az  Azerbaijani  bg  Bulgarian     bn  Bengali
ca  Catalan       cs  Czech         da  Danish        de  German
el  Greek         en  English       es  Spanish       fa  Persian
fi  Finnish       fr  French        he  Hebrew        hi  Hindi
hr  Croatian      ht  Haitian Creole hu Hungarian    id  Indonesian
is  Icelandic     it  Italian       ja  Japanese      ko  Korean
la  Latin         lt  Lithuanian    ms  Malay        ne  Nepali
nl  Dutch         no  Norwegian     pa  Punjabi       pl  Polish
pt  Portuguese    ro  Romanian      ru  Russian       sa  Sanskrit
sk  Slovak        sr  Serbian       sv  Swedish      sw  Swahili
ta  Tamil         te  Telugu        th  Thai         tl  Tagalog
tr  Turkish       uk  Ukrainian     ur  Urdu         vi  Vietnamese
yue Cantonese     zh  Chinese
```

**Technical notes:**
- Uses constrained decoding (FSM‑based) to ensure only valid language codes appear in the generated metadata.
- No language‑specific model variants exist — a single model handles all 50 languages.
- For background music in dialogue mode, lyrics are set to `"..."` (placeholder for empty vocals), so language detection is irrelevant — the model generates instrumental music only.
- The LM models are based on Qwen3, which is inherently multilingual.

---

## EasyOCR — Image Text Extraction

**Model:** `JaidedAI/EasyOCR`
**Modes:** STT, TTS, TTS+VC (when source is an image file)
**Language handling:** Hardcoded to `['en']` (English only). EasyOCR does **not** support auto‑detection — languages must be explicitly specified in the language list when initializing the reader.

**Current VODER configuration:**

```python
# src/voder.py → EasyOCRReader.ensure_model()
self.reader = easyocr.Reader(
    ['en'],                              # English only
    model_storage_directory=self.easyocr_dir,
    download_enabled=True,
    gpu=False                             # CPU only
)
```

**Available languages (85 total) — not all active in VODER:**

EasyOCR supports 85 languages across multiple scripts, but VODER currently only loads the English model. English is compatible with every other language for combined recognition, meaning you can add any language alongside English without conflicts.

**Latin script (40):**

```
af  Afrikaans       az  Azerbaijani    bs  Bosnian        cs  Czech
cy  Welsh           da  Danish         de  German         en  English
es  Spanish         et  Estonian       fr  French         ga  Irish
hr  Croatian        hu  Hungarian      id  Indonesian     is  Icelandic
it  Italian         ku  Kurdish        la  Latin          lt  Lithuanian
lv  Latvian         mi  Maori          ms  Malay         mt  Maltese
nl  Dutch           no  Norwegian      oc  Occitan        pi  Pali
pl  Polish          pt  Portuguese     rs_latin Serbian  sk  Slovak
sl  Slovenian       sq  Albanian       sv  Swedish       sw  Swahili
tl  Tagalog         tr  Turkish        uz  Uzbek         vi  Vietnamese
```

**Arabic script (4):**

```
ar  Arabic          fa  Persian        ug  Uyghur         ur  Urdu
```

**Bengali script (3):**

```
bn  Bengali         as  Assamese       mni  Manipuri
```

**Cyrillic script (17):**

```
ru  Russian         rs_cyrillic Serbian be  Belarusian     bg  Bulgarian
uk  Ukrainian       mn  Mongolian      abq Abaza         ady Adyghe
kbd Kabardian       ava  Avar          dar Dargwa        inh Ingush
che Chechen         lbe  Lak           lez Lezgian       tab Tabassaran
tjk Tajik
```

**Devanagari script (13):**

```
hi  Hindi           mr  Marathi        ne  Nepali         bh  Bihari
mai Maithili       ang Angika        bho Bhojpuri      mah Magahi
sck Sindhi         new Newari        gom Konkani       sa  Sanskrit
bgc Haryanvi
```

**Other scripts (8):**

```
th  Thai           ch_sim Chinese Simplified  ch_tra Chinese Traditional
ja  Japanese        ko  Korean         ta  Tamil          te  Telugu
kn  Kannada
```

**How to change the language list:**

To add support for additional languages, modify the language list in the `EasyOCRReader.ensure_model()` method in `src/voder.py`. For example, to support Japanese and Chinese alongside English:

```python
self.reader = easyocr.Reader(
    ['en', 'ch_sim', 'ja'],          # English + Simplified Chinese + Japanese
    model_storage_directory=self.easyocr_dir,
    download_enabled=True,
    gpu=False
)
```

The first time a language is added, EasyOCR downloads the required model files (approximately 50–100MB per language). Subsequent runs use the cached models.

**Important limitations:**
- Not all language combinations are compatible. Languages that share common characters (e.g., Latin‑script languages) work together. Languages with different scripts (e.g., Japanese and Arabic) may not work together reliably.
- EasyOCR runs on CPU only in VODER (`gpu=False`). GPU acceleration is not enabled.
- Adding more languages increases memory usage and may slow down initial loading.

---

## TangoFlux — Sound Effects Generation

**Model:** `declare-lab/TangoFlux` + `google/flan-t5-large` (text encoder)
**Modes:** SFX
**Language handling:** English only. The text encoder (Flan‑T5‑Large) was trained primarily on English data. Non‑English prompts will tokenize without errors but produce degraded or unrelated audio output. There is no translation pipeline in the code.

**Technical notes:**
- All training data (AudioCaps, WavCaps) is English‑only.
- Write prompts in English for best results. Descriptive, concise prompts with environmental context work best (e.g., `"heavy rain on a tin roof in a dark forest"`).

---

## Language‑Agnostic Components

These components operate on audio waveforms or speaker identity directly. They do not process text or use language codes in any way, which means they work with any spoken language without configuration.

### Seed‑VC v2 — Voice Conversion (Speech)

**Modes:** STS

Operates on raw audio waveforms at 22.05kHz. The content encoder (ASTRAL Quantization) extracts linguistic content regardless of language, and the style encoder (CAMPPlus) extracts speaker characteristics. No text, language codes, or language‑specific models are involved.

### Seed‑VC v1 — Voice Conversion (Music)

**Modes:** MSTS (Music‑STS)

Same language‑agnostic approach as v2, but runs at 44.1kHz for music content. Uses Whisper‑small as content encoder (trained on approximately 100 languages) and includes RMVPE pitch extraction for singing voice.

### UniSE — Speech Enhancement

**Modes:** SE

Processes raw audio to remove noise, reduce reverberation, and restore speech clarity. Uses WavLM for semantic feature extraction (trained on multilingual speech data). No language configuration or text processing is involved. Outputs at 16kHz.

### Pyannote — Speaker Diarization

**Modes:** STT (with `dialogue` flag), Dialogue Source Analysis, Voice Clip Extraction

Identifies and labels individual speakers based on voice embeddings, not linguistic content. Evaluated on multilingual datasets including English (AMI), Mandarin (AISHELL‑4), French (REPERE), Romanian (RAMC), and multi‑language corpora (CALLHOME, DIHARD, VoxConverse). Requires HF_TOKEN for model access.

---

## Cross‑Language Workflows

VODER's architecture enables workflows that cross language boundaries between components:

**Transcribe in one language, synthesize in another:**
```
Audio (Japanese) → Whisper STT → Japanese text → Qwen3‑TTS TTS → English speech
```

**Clone a voice from one language, generate in another:**
```
Reference audio (Korean) + text (German) → Qwen3‑TTS Base TTS+VC → German speech with Korean voice
```

**Transcribe multilingual audio, generate multilingual dialogue:**
```
Audio (mixed English/Japanese) → Whisper + Pyannote → Dialogue text → Qwen3‑TTS Auto → Multilingual speech
```

**Generate music with non‑English lyrics:**
```
Lyrics (Spanish) + style prompt (English) → ACE‑Step TTM → Spanish vocal music
```

These workflows work because each component handles language independently. Whisper auto‑detects the input language, Qwen3‑TTS auto‑detects the output language, and voice cloning operates on speaker identity rather than language. The components don't need to agree on a language — each one handles its own detection.
