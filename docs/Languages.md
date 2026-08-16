# VODER — Language Support Reference

VODER is not an English‑only tool. The AI models it orchestrates collectively support over 99 languages across transcription, speech synthesis, music generation, and image text extraction. This document provides a complete breakdown of which languages each component supports, how language detection works, and what the current limitations are.

---

## Overview

| Component | Modes | Languages | Auto‑Detect | Notes |
|-----------|-------|-----------|-------------|-------|
| **Whisper** (`large-v3-turbo` / `large-v3`) | STT, TTS (modify speech), Dialogue Source | 99 | Yes | Detects spoken language from audio; dual‑model architecture |
| **Qwen3‑TTS VoiceDesign** | TTS | 10 + 2 dialects | Yes | Detects language from input text |
| **Qwen3‑TTS Base** | TTS+VC, TTS (modify speech) | 10 + 2 dialects | Yes | Detects language from input text |
| **Fish Audio S2‑Pro** | TTS (extreme), SLC (extreme), SVC (extreme), Modify Speech (extreme) | 80+ | Yes | Detects language from input text; voice effects via [tag] syntax |
| **ACE‑Step 1.5** | TTM, TTM+VC, Background Music | 50 | Yes | Detects language from lyrics/caption |
| **MiniMax Music 3** | TTM (extreme), TTS music (extreme) | 80+ | Yes | Detects language from lyrics; Qwen3-8B language model |
| **EasyOCR** | STT, TTS, TTS+VC (image input) | 85 | No | Hardcoded to English in VODER |
| **TangoFlux** | SFX | 1 (English) | No | Text encoder trained on English only |
| **Seed‑VC v2** | STS | Any | N/A | Language‑agnostic (audio waveforms) |
| **Seed‑VC v1** | MSTS | Any | N/A | Language‑agnostic (audio waveforms) |
| **UniSE** | SE | Any | N/A | Language‑agnostic (audio waveforms) |
| **AudioSR** | SE (sr sub-modes) | Any | N/A | Language‑agnostic (audio waveforms, lowpass conditioning) |
| **Pyannote** | Diarization | Any | N/A | Language‑agnostic (voice embeddings) |
| **VibeVoice ASR** | STT (overdose), SS, TTS (dub) | 53 | Yes | Native speaker diarization; 24GB+ VRAM or 48GB+ RAM required; audio events preserved for dub pipeline |
| **TranslateGemma 12B** | STT (translate), TTS (SLC translate, dub), STT (subtitle translate) | 76 | Yes | Any-to-any translation; decoupled from ASR; 24GB+ VRAM recommended; auto-detects source language with `auto` |

---

## Whisper — Speech‑to‑Text

**Model:** `openai/whisper` → dual‑model architecture
- `large-v3-turbo` — used for standard transcription (fast, efficient)
- `large-v3` — used for translation tasks (supports the `translate` task which turbo does not)

**Modes:** STT, TTS (modify speech), Dialogue Source Analysis
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

**Translation capability:**
Audio in any of the 99 supported languages can be translated to English text using the `large-v3` model. The translation task is only available on `large-v3` — `large-v3-turbo` does not support it. When translation is requested, the full `large-v3` model is loaded instead of turbo, and Whisper outputs English text regardless of the input language.

**Pre‑cleanup (SVS vocal isolation):**
Before transcription, VODER can run BS‑RoFormer SVS vocal isolation on the input audio to separate voice from background music or noise. This pre‑cleanup step ensures Whisper receives clean vocal content, which significantly improves transcription accuracy on mixed audio (songs, podcasts with music beds, field recordings). The separated vocal stem is passed directly to Whisper; the instrumental stem is discarded.

**Technical notes:**
- When the `dialogue` flag is enabled, Whisper transcription is combined with Pyannote diarization and aligned using a three‑tier overlap matching system. The language of the transcription matches the spoken language.
- Processing is CPU‑only. Approximately 1x real‑time on a modern CPU.

---

## VibeVoice ASR — Advanced Speech Recognition

**Model:** `microsoft/VibeVoice-ASR` with `Qwen/Qwen2.5-7B` language model backbone
**Modes:** STT (with `overdose` flag), SS (Speakers Separator)
**Language handling:** Auto‑detects spoken language from audio. Does **not** support translation — the `overdose` and `translate` flags are mutually exclusive.

**Supported languages (53 total, ordered from highest to lowest accuracy):**

```
English           Chinese            Spanish            Portuguese
German            Japanese           Korean             French
Russian           Indonesian         Swedish            Italian
Hebrew            Dutch              Polish              Norwegian
Turkish           Thai               Arabic              Hungarian
Catalan           Czech              Danish              Persian
Afrikaans         Hindi              Finnish             Estonian
Afar              Greek              Romanian            Vietnamese
Bulgarian         Icelandic          Slovenian           Slovak
Lithuanian        Swahili            Ukrainian           Kalaallisut
Latvian           Croatian           Nepali              Serbian
Filipino          Yiddish            Malay               Urdu
Mongolian         Armenian           Javanese
```

**Technical notes:**
- VibeVoice ASR provides **native speaker diarization** — it identifies and labels individual speakers as part of the transcription process without needing a separate diarization model like Pyannote.
- Requires 24GB+ VRAM or 48GB+ system memory. If insufficient resources are detected, SS mode falls back to Whisper + Pyannote speaker diarization.
- Does **not** support translation to English. For translation, use standard STT mode with Whisper (`translate` flag) or TranslateGemma (`translate (source-target)` syntax).
- The `overdose` flag in STT mode switches the transcription pipeline from Whisper to VibeVoice ASR. When `overdose` is active, the `dialogue` flag is redundant and ignored since VibeVoice handles speaker identification natively.
- In SS mode, VibeVoice ASR is the primary model used for speaker identification and segmentation. Each identified speaker's audio is extracted into a separate file, and a speaker-labeled transcript is produced.
- In the TTS dub pipeline, VibeVoice ASR uses the `transcribe_with_events()` method which preserves audio event tags (`[Silence]`, `[Lyric]`, `[Music]`, `[Noise]`, etc.) alongside speech segments. Audio events are used to identify non-speech portions of the audio that should not be translated or dubbed — these segments are left as silence or original audio in the dubbed output.

---

## TranslateGemma 12B — Any-to-Any Translation

**Model:** `google/translategemma-12b-it` (12B parameters)
**Modes:** STT (with `translate (source-target)` syntax), TTS (SLC with translate, dub), STT (subtitle with translate)
**Language handling:** Accepts explicit source and target language codes. Supports `auto` as source language for auto-detection (VibeVoice ASR or Whisper detects the source language, then TranslateGemma translates). TranslateGemma is decoupled from the ASR engine — it receives text from either VibeVoice ASR or Whisper and translates it independently.

**Supported languages (76 total):**

```
af  Afrikaans      am  Amharic        ar  Arabic         az  Azerbaijani
be  Belarusian     bg  Bulgarian      bn  Bengali        bs  Bosnian
ca  Catalan        cs  Czech          cy  Welsh          da  Danish
de  German         el  Greek          en  English        es  Spanish
et  Estonian       eu  Basque         fa  Persian        fi  Finnish
fr  French         ga  Irish          gl  Galician       gu  Gujarati
ha  Hausa          he  Hebrew         hi  Hindi          hr  Croatian
hu  Hungarian      id  Indonesian     is  Icelandic      it  Italian
ja  Japanese       jv  Javanese       ka  Georgian       kk  Kazakh
km  Khmer          kn  Kannada        ko  Korean         lo  Lao
lt  Lithuanian     lv  Latvian        mk  Macedonian     ml  Malayalam
mn  Mongolian      mr  Marathi        ms  Malay          mt  Maltese
my  Myanmar        ne  Nepali         nl  Dutch          no  Norwegian
pa  Punjabi        pl  Polish         ps  Pashto         pt  Portuguese
ro  Romanian       ru  Russian        si  Sinhala        sk  Slovak
sl  Slovenian      so  Somali         sq  Albanian       sr  Serbian
sv  Swedish        sw  Swahili        ta  Tamil          tg  Tajik
th  Thai           tk  Turkmen        tl  Tagalog        tr  Turkish
uk  Ukrainian      ur  Urdu           uz  Uzbek          vi  Vietnamese
yo  Yoruba         zh  Chinese
```

**Syntax:**
- `translate` (bare) — Uses Whisper's built-in any-to-English translation (backward compatible, limited to English output)
- `translate "(auto-en)"` — Auto-detect source language, translate to English via TranslateGemma
- `translate "(en)"` — Shorthand for `translate "(auto-en)"`
- `translate "(auto-ja)"` — Auto-detect source language, translate to Japanese via TranslateGemma
- `translate "(ja)"` — Shorthand for `translate "(auto-ja)"`
- `translate "(ja-en)"` — Translate Japanese to English via TranslateGemma
- `translate "(ar-fr)"` — Translate Arabic to French via TranslateGemma

**Where TranslateGemma is used:**

| Context | Syntax | Behavior |
|---------|--------|----------|
| STT standard | `translate` (bare) | Whisper large-v3 any-to-English (no TranslateGemma) |
| STT standard | `translate (source-target)` | Whisper transcribe → TranslateGemma translate |
| STT overdose | `translate (source-target)` | VibeVoice ASR transcribe → TranslateGemma translate |
| STT subtitle | `translate (source-target)` | VibeVoice ASR → TranslateGemma → burned subtitles |
| TTS SLC | `translate (source-target)` | Whisper transcribe → TranslateGemma translate → TTS resynthesize |
| TTS dub | default or `translate (source-target)` | VibeVoice ASR → TranslateGemma translate per segment → Fish S2 Pro TTS per segment |

**Technical notes:**
- TranslateGemma is loaded and unloaded per-use to prevent memory conflicts with VibeVoice ASR and Fish S2 Pro. It never co-exists with either model in GPU memory.
- The model uses the `AutoModelForImageTextToText` architecture from transformers with a chat template for translation prompts.
- On GPU with 24GB+ VRAM, runs in bfloat16. On systems with less VRAM, falls back to CPU float32 (slow but functional).
- Stored at `src/models/checkpoints/translategemma/`. Auto-downloads on first use.
- The bare `translate` flag (without parentheses) remains incompatible with `overdose` in STT — it uses Whisper's built-in translation which conflicts with VibeVoice ASR. The `translate (source-target)` syntax is compatible with overdose because TranslateGemma is decoupled from the ASR engine.
- In the TTS dub pipeline, TranslateGemma receives per-segment timing context to encourage concise translations that match the original speech duration. The prompt includes original duration, word count, and instructions to keep the translation concise.

---

## Qwen3‑TTS VoiceDesign — Text‑to‑Speech

**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
**Modes:** TTS (single and dialogue)
**Language handling:** Auto‑detects language from the input text. Set to `"Auto"` by default in VODER. The model reads the text content and determines the appropriate language without any user configuration. The language parameter is now exposed via the `SUPPORTED_TTS_LANGUAGES` constant, which maps ISO 639‑1 codes to full English names.

**Supported languages (10 total):**

| ISO Code | Language |
|----------|----------|
| zh | Chinese |
| en | English |
| ja | Japanese |
| ko | Korean |
| de | German |
| fr | French |
| ru | Russian |
| pt | Portuguese |
| es | Spanish |
| it | Italian |

**Chinese dialects (2):**

| Dialect | Associated Speaker | Description |
|---------|-------------------|-------------|
| Beijing Mandarin | Dylan | Youthful Beijing male voice |
| Sichuan Mandarin | Eric | Lively Chengdu male voice |

**Technical notes:**
- Language validation is case‑insensitive. Both `"English"` and `"english"` are accepted.
- `"Auto"` is the default and works reliably for all 10 languages. If the target language is known, setting it explicitly can improve consistency in ambiguous cases (e.g., mixed‑language text).
- The model uses full English language names as identifiers (e.g., `"Chinese"`, `"English"`), not ISO codes. The `SUPPORTED_TTS_LANGUAGES` dictionary in `src/voder.py` provides the mapping between the two.

---

## Qwen3‑TTS Base — Text‑to‑Speech with Voice Cloning

**Model:** `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
**Modes:** TTS+VC, TTS (modify speech)
**Language handling:** Auto‑detects language from the input text. Set to `"Auto"` by default in VODER. This is the same language detection system used by VoiceDesign — the Base model variant supports it identically. The language parameter is now exposed via the `SUPPORTED_TTS_LANGUAGES` constant, which maps ISO 639‑1 codes to full English names.

**Supported languages:** Same 10 languages + 2 dialects as VoiceDesign (see above).

**Technical notes:**
- Voice cloning extracts an x‑vector speaker embedding from the reference audio, which is language‑independent. A Chinese reference audio can be used to clone a voice that speaks English, Japanese, or any other supported language.
- In dialogue mode, the voice embedding is extracted **once per character** at the start and reused for all their lines, ensuring consistent voice quality regardless of language changes between lines.
- The `generate_voice_clone()` method supports batch language lists, but VODER passes `"Auto"` for all lines.

---

## Fish Audio S2‑Pro — Text‑to‑Speech (Extreme Mode)

**Model:** `fishaudio/s2-pro` — Dual-Autoregressive architecture (4B Slow AR + 400M Fast AR)
**Modes:** TTS (with `extreme` flag), SLC (with `extreme` flag), SVC (with `extreme` flag), Modify Speech (with `extreme` prompt)
**Language handling:** Auto‑detects language from the input text. No phoneme or language‑specific preprocessing is required — the model handles all languages natively through its dual-AR architecture.

**Supported languages (80+ total):**

**Tier 1 (highest quality):**
- Japanese (ja), English (en), Chinese (zh)

**Tier 2:**
- Korean (ko), Spanish (es), Portuguese (pt), Arabic (ar), Russian (ru), French (fr), German (de)

**Global coverage (partial list):**
sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo, and more.

**Voice effects (`[tag]` syntax):**
S2-Pro supports sub‑word level fine‑grained control of prosody, emotion, and vocal characteristics using `[tag]` syntax embedded in the text. Over 15,000 unique tags are supported — the model accepts free‑form natural language descriptions, not just a fixed set. Tags affect text from their position onward. The model also accepts all S1 Pro tags inside `[brackets]`.

**S2-Pro well-tested tags:**

| Category | Tags |
|----------|------|
| **Emotions** | `[excited]`, `[angry]`, `[sad]` |
| **Tones / Voice Style** | `[whispering]`, `[soft voice]`, `[low voice]`, `[loud voice]`, `[shouting]` |
| **Breathing & Reactions** | `[sigh]`, `[inhale]`, `[exhale]`, `[gasp]`, `[panting]`, `[clears throat]` |
| **Vocal Sounds** | `[laughing]`, `[chuckling]`, `[giggle]`, `[sobbing]`, `[crying]`, `[groan]` |
| **Pacing** | `[pause]`, `[short pause]`, `[long pause]` |
| **Special** | `[emphasis]`, `[rustling sound]` |

**S1 Pro tags (also work in `[brackets]` for S2-Pro):**
These 64 tags were designed for Fish S1 Pro using `(parenthesis)` syntax, but they also work inside `[brackets]` with S2-Pro:

| Category | Tags |
|----------|------|
| **Emotions** | `(angry)` `(sad)` `(disdainful)` `(excited)` `(surprised)` `(satisfied)` `(unhappy)` `(anxious)` `(hysterical)` `(delighted)` `(scared)` `(worried)` `(indifferent)` `(upset)` `(impatient)` `(nervous)` `(guilty)` `(scornful)` `(frustrated)` `(depressed)` `(panicked)` `(furious)` `(empathetic)` `(embarrassed)` `(reluctant)` `(disgusted)` `(keen)` `(moved)` `(proud)` `(relaxed)` `(grateful)` `(confident)` `(interested)` `(curious)` `(confused)` `(joyful)` `(disapproving)` `(negative)` `(denying)` `(astonished)` `(serious)` `(sarcastic)` `(sneering)` `(hesitating)` `(yielding)` `(painful)` `(awkward)` `(amused)` |
| **Tone Markers** | `(in a hurry tone)` `(shouting)` `(screaming)` `(whispering)` `(soft tone)` |
| **Vocal Sounds** | `(laughing)` `(chuckling)` `(sobbing)` `(crying loudly)` `(sighing)` `(panting)` `(groaning)` |
| **Crowd Effects** | `(crowd laughing)` `(background laughter)` `(audience laughing)` |

Free-form examples: `[professional broadcast tone]`, `[pitch up]`, `[voice rough from crying, trying to sound normal]`, `[dead tired, end of a very long shift]`. Multi-language tags are also supported (e.g., `[低声说]` for Chinese "speak softly", `[囁き声で]` for Japanese "whisper voice").

**Voice Design with extreme mode:**
When `extreme` is used with a `voice` prompt (not `target`), VODER always generates ~30 seconds of placeholder English speech via VoiceDesign, feeds it to Fish S2‑Pro for voice cloning, then Fish speaks the actual text. This applies unconditionally — even for languages VoiceDesign already supports — because it ensures consistent voice quality across all languages, preserves voice effects tags (like `[whispering]`, `[angry]`) that VoiceDesign would otherwise misinterpret, and eliminates the need for language detection. This enables voice design for 70+ additional languages that VoiceDesign doesn't natively support, while also improving results for the 10 supported ones.

**Technical notes:**
- The model uses an RVQ‑based codec with 10 codebooks at ~21 Hz frame rate
- Voice cloning from reference audio (10–30 seconds) captures timbre, speaking style, and emotional tendencies without fine‑tuning
- Supports native multi‑speaker in one pass using `Name: text` syntax or via `<|speaker:i|>` tokens, but VODER's dialogue mode is recommended for better per‑character control
- Activated with the `extreme` keyword after `overdose` in command syntax
- Trained voices are saved as `.ttse` files (not `.tts`); using the wrong format produces a clear error message

---

## ACE‑Step 1.5 — Music Generation

**Model:** `ACE-Step/Ace-Step1.5`
**Modes:** TTM, TTM+VC, Background Music (dialogue)
**Language handling:** Language is auto‑detected from the lyrics or caption text by the language model (Qwen3‑based). The `vocal_language` parameter defaults to `"unknown"`, which triggers automatic detection. Language can be set explicitly if needed.

**Three‑tier quality system:**

| Tier | Music Model | Language Model | Inference Steps | Use Case |
|------|-------------|----------------|-----------------|----------|
| **Standard** | `acestep-v15-turbo` | `acestep-5Hz-lm-1.7B` | 8 | Default generation; fast results |
| **Overdose** | `acestep-v15-xl-turbo` | `acestep-5Hz-lm-4B` | 8 (shift=3.0) | High‑quality output; increased detail |
| **Complete** | `acestep-v15-xl-base` | `acestep-5Hz-lm-1.7B` | 50 | Sub‑tasks requiring maximum fidelity |

**Sub‑tasks (Complete tier only):** `complete`, `lego`, `extract`, `remix`, `repaint`

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

## MiniMax Music 3 — Music Generation (Extreme TTM)

**Model:** `MiniMaxAI/MiniMax-Music3`
**Modes:** TTM (extreme), TTS background music (extreme)
**Language handling:** The Global LLM is initialized from Qwen3-8B, which is inherently multilingual. The model was trained on over 10 million hours of audio data covering more than 80 languages. Language is detected from the lyrics text — no manual language specification is needed.

**Supported languages:** 80+ languages, including but not limited to:
- **Tier 1:** English, Chinese (Mandarin/Cantonese), Japanese
- **Tier 2:** Korean, Spanish, Portuguese, Arabic, Russian, French, German
- **Global coverage:** Italian, Turkish, Ukrainian, Urdu, Vietnamese, Thai, Hindi, Indonesian, Malay, Dutch, Polish, Swedish, Catalan, Czech, Romanian, Danish, Finnish, Hungarian, Greek, Hebrew, Bulgarian, Norwegian, and many more

**Lyrics section tags (language-agnostic):**
The model accepts lyrics with structural section tags. Tags are lowercased automatically and must be on their own line:
- `[intro]`, `[verse]`, `[pre-chorus]`, `[chorus]`, `[post-chorus]`, `[bridge]`, `[instrumental]`, `[solo]`, `[outro]`
- Custom tags like `[bass-drop]` or `[breakdown]` are also accepted

**Music description language:**
The music description (styling) should be written in English for best results, as the model's training data is predominantly English-captioned. However, the model can understand descriptions in other languages with slightly reduced precision.

**Notes:**
- For TTS background music (extreme), VODER uses instrumental-only lyrics (`[intro]`, `[instrumental]`, `[outro]`) so the generated music has no vocals that would clash with the spoken dialogue.
- The model does not support reference audio, voice cloning, or source audio modification — only text-conditioned generation.
- Output is 44.1 kHz, 16-bit stereo WAV.

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

### UniSE — Sound Enhancement

**Modes:** SE (default, voice sub-modes)

Processes raw audio to remove noise, reduce reverberation, and restore speech clarity. Uses WavLM for semantic feature extraction (trained on multilingual speech data). No language configuration or text processing is involved. Outputs at 16kHz.

### AudioSR — Audio Super-Resolution

**Modes:** SE (sr, sr music sub-modes)

Upscales low-sample-rate audio to 48kHz using a latent diffusion model. Operates purely on audio signal characteristics — no language configuration or text processing involved. Two model variants: `basic` (general audio/music) and `speech` (speech-optimized). Uses lowpass filtering as conditioning — language-agnostic by design.

### Pyannote — Speaker Diarization

**Modes:** STT (with `dialogue` flag), Dialogue Source Analysis, Voice Clip Extraction

Identifies and labels individual speakers based on voice embeddings, not linguistic content. Evaluated on multilingual datasets including English (AMI), Mandarin (AISHELL‑4), French (REPERE), Romanian (RAMC), and multi‑language corpora (CALLHOME, DIHARD, VoxConverse). Requires HF_TOKEN for model access.

### BS‑RoFormer Resurrection — Voice/Music Separation (SVS)

**Modes:** SVS (standalone), STS (automatic vocal extraction), STT (pre‑cleanup), TTS (voice cloning cleanup, dub vocal isolation)

Operates on raw audio waveforms to separate vocal content from instrumental content. The model does not process text or use language codes — it works purely on audio signal characteristics. Separation quality is consistent across all languages because it relies on spectral and temporal features rather than linguistic content. Two stems are supported: `voice` (vocal isolation) and `music` (instrumental extraction).

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

**Translate speech to English with original voice (TTS SLC):**
```
Audio (any language) → Whisper translate → English text → Qwen3‑TTS TTS with original voice reference → English speech with original voice
```

**Change speaker voice across languages (TTS SLC):**
```
Audio (Spanish) + target reference (English speaker) → TTS SLC → English speech with English speaker voice
```

**Voice design with extreme mode:**
```
Text (any language) + voice prompt ("deep male") + extreme flag → VoiceDesign generates English placeholder → Fish S2-Pro clones it → Fish speaks actual text in any language
```

**Extreme voice cloning for 80+ languages:**
```
Reference audio (Hindi) + text (Hindi) + extreme flag → Fish S2-Pro → Hindi speech with cloned voice
```

**Any-to-any translation with TranslateGemma (STT):**
```
Audio (Arabic) → VibeVoice ASR → Arabic text → TranslateGemma → Japanese text
```

**Dub video to another language (TTS dub):**
```
Video (Japanese) → SVS voice isolation → VibeVoice ASR → Japanese segments → TranslateGemma → English segments → Fish S2 Pro (voice cloning) → Per-segment speed adjustment → Timeline assembly → Mix with music track → English dubbed video
```

**Translate and subtitle (STT subtitle):**
```
Video (French) → SVS voice isolation → VibeVoice ASR → French segments → TranslateGemma → English segments → ASS subtitle burn → English subtitled video
```

These workflows work because each component handles language independently. Whisper auto‑detects the input language, TranslateGemma handles any-to-any translation, Qwen3‑TTS auto‑detects the output language, and voice cloning operates on speaker identity rather than language. The components don't need to agree on a language — each one handles its own detection. TranslateGemma decouples translation from ASR, enabling any-to-any translation regardless of which transcription engine (Whisper or VibeVoice) is used.

---

## SUPPORTED_TTS_LANGUAGES Constant

The `SUPPORTED_TTS_LANGUAGES` dictionary is defined in `src/voder.py` and provides the mapping between ISO 639‑1 codes and full English language names used by the TTS pipeline:

```python
SUPPORTED_TTS_LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian"
}
```

This constant is used by both Qwen3‑TTS VoiceDesign and Qwen3‑TTS Base to validate and map language parameters. When a user specifies an ISO code, it is resolved to the full English name expected by the model. When `"Auto"` is specified, the model performs its own language detection from the input text and the mapping is bypassed.
