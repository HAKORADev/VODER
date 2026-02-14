# VODER Technical Guide

## Table of Contents

- [Introduction & Vision](#introduction--vision)
- [The Philosophy: Quality Over Speed](#the-philosophy-quality-over-speed)
- [Why Hardcoded Models?](#why-hardcoded-models)
  - [The Quality Imperative](#the-quality-imperative)
  - [Custom Model Support](#custom-model-support)
  - [Custom Versions](#custom-versions)
- [Processing Modes Deep Dive](#processing-modes-deep-dive)
  - [TTS: Text-to-Speech](#tts-text-to-speech)
  - [TTS+VC: Text-to-Speech + Voice Cloning](#ttsvc-text-to-speech--voice-cloning)
  - [STS: Speech-to-Speech Voice Conversion](#sts-speech-to-speech-voice-conversion)
  - [TTM: Text-to-Music](#ttm-text-to-music)
  - [TTM+VC: Text-to-Music + Voice Conversion](#ttmvc-text-to-music--voice-conversion)
  - [STT+TTS: Speech-to-Text + Synthesis](#stttts-speech-to-text--synthesis)
- [The Dialogue System](#the-dialogue-system)
  - [What Dialogue Mode Is](#what-dialogue-mode-is)
  - [How It Works](#how-it-works)
  - [Why Not Multi-Speaker Input?](#why-not-multi-speaker-input)
  - [Dialogue Input in GUI](#dialogue-input-in-gui)
  - [Dialogue Input in CLI](#dialogue-input-in-cli)
    - [Interactive CLI Dialogue](#interactive-cli-dialogue)
    - [One‑Liner Dialogue](#one-liner-dialogue)
  - [Voice Prompt Configuration](#voice-prompt-configuration)
  - [Optional Background Music for Dialogue](#optional-background-music-for-dialogue)
    - [How It Works](#how-it-works-1)
    - [GUI Workflow](#gui-workflow)
    - [Interactive CLI Workflow](#interactive-cli-workflow)
    - [One‑Liner CLI Workflow](#one-liner-cli-workflow)
    - [Technical Implementation](#technical-implementation)
- [Tips & Tricks](#tips--tricks)
  - [Getting Better Results](#getting-better-results)
  - [Multi-Speaker Scenarios](#multi-speaker-scenarios)
  - [Using Same Audio Source](#using-same-audio-source)
  - [Voice Cloning Best Practices](#voice-cloning-best-practices)
  - [Background Music Best Practices](#background-music-best-practices)
- [Version Information](#version-information)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)

---

## Introduction & Vision

VODER is a professional‑grade voice processing tool that brings together six distinct audio transformation capabilities in a single, unified interface. Unlike tools that force you to jump between multiple applications for different voice‑related tasks, VODER provides everything from text‑to‑speech synthesis to music generation under one roof.

**What VODER Actually Does:**

At its core, VODER orchestrates state‑of‑the‑art AI models to perform voice‑related transformations. It can convert speech to text and back to speech with a different voice, generate speech from text using either designed voices or cloned references, transform one voice into another while preserving content, and create music from lyrics with optional voice conversion for the vocalist. This isn't about chasing the fastest processing times or highest frame rates — it's about achieving professional‑quality results that actually sound good.

**Why VODER Exists:**

The voice synthesis market is dominated by expensive commercial platforms that charge per character or per month. ElevenLabs, OpenAI, and others offer powerful capabilities, but at costs that add up quickly for creators, developers, and businesses alike. More importantly, no existing open‑source solution offered all six processing capabilities in a unified interface. You could find separate tools for TTS, voice conversion, and music generation, but none that worked together seamlessly.

VODER was built to fill this gap. The goal from day one was to create a local, free, open‑source alternative that doesn't compromise on quality. Is it perfect? No software is. But it works, it keeps improving, and it provides genuine utility without subscription fees or usage limits.

**What Makes VODER Different:**

Most voice processing tools focus on a single use case. VODER takes a different approach — it treats voice and audio processing as a unified problem space. The same interface that generates speech from text can also convert that speech between voices, and the same voice cloning technology can apply to both speech and singing. This integration enables workflows that would otherwise require multiple tools and significant manual effort.

---

## The Philosophy: Quality Over Speed

### We Don't Chase FPS

This is worth emphasizing because it's fundamental to VODER's design philosophy. There are no "recommended requirements" in the traditional sense. This isn't a video game where higher frame rates give you a better experience. The only metric that matters is avoiding one thing: Out Of Memory (OOM) errors.

When we say "minimum requirements" with 8GB VRAM, that's not a performance target — it's a reliability floor. If you have exactly 8GB, VODER will work. If you have 12GB, it won't process things twice as fast. It just means you have more headroom for longer audio files or more complex operations. The quality remains the same because we're not offering quality presets that sacrifice output fidelity for speed.

**Why We Don't Offer Fast Modes:**

Every other tool on the market offers "fast" or "efficient" variants of their models. Smaller models, quantized weights, reduced quality settings. We explicitly chose not to include these options. Here's why: a degraded model produces output that is genuinely worse, not just faster to generate. If you're using voice synthesis for content creation, professional work, or anything where quality matters, you'd be better off not using the tool at all than using a degraded version.

Think of it like photography. You can have a cheap smartphone camera that takes pictures instantly, or you can use a professional camera that requires proper technique and takes slightly longer. The smartphone photo is "faster" but the professional camera photo is objectively better quality. VODER is the professional camera of voice processing tools.

**The OOM Reality:**

Some operations require significant memory. Voice conversion models, especially, need to load multiple neural network components and maintain activations throughout the processing pipeline. If you try to process a 10‑minute audio file and run out of VRAM, the solution isn't to use a smaller model — it's to process shorter segments. VODER doesn't offer shortcuts that compromise quality because shortcuts in AI almost always mean worse output.

**System Requirements Explained:**

When we list minimum requirements, we're being honest about what actually works. All VODER modes run on CPU — no GPU is required. However, having a GPU with sufficient VRAM can significantly improve processing speed for certain modes.

| Mode | Base Memory | Additional | Total RAM | GPU (CUDA) | VRAM |
|------|--------------|------------|-----------|------------|------|
| TTS, TTS+VC (no music) | 8GB | +4GB (Qwen) | 12GB | Optional | 4GB (GTX 1060 Ti) |
| TTS, TTS+VC (with music) | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| STT+TTS | 8GB | +4GB (Qwen) | 12GB | Optional | 4GB (GTX 1060 Ti) |
| STS | 8GB | +5GB (Seed-VC) | 13GB | Optional | 14GB |
| TTM | 8GB | +15GB (ACE) | 23GB | Optional | 15GB (RTX 3080/16GB GPU) |
| TTM+VC | 8GB | +15GB (ACE) | 23GB | Optional | 16GB |

- **CPU**: 4-6 cores minimum for model loading and non-GPU operations
- **RAM**: 12GB minimum for basic modes, 23GB for ACE-related modes (TTM, TTM+VC, or TTS/TTS+VC with music)
- **GPU (CUDA)**: Optional - all modes work on CPU. GPU acceleration significantly speeds up STS, TTM, and TTM+VC modes
- **VRAM**: 4GB minimum (6GB recommended, 16GB for best performance with music modes)
- **Storage**: SSD recommended for model downloads and result saving

**VRAM Guidelines:**

| VRAM | Performance Level | Suitable Modes |
|------|-------------------|----------------|
| No GPU (CPU only) | Slow | All modes |
| 4GB | Usable | TTS, TTS+VC (no music), STT+TTS |
| 6GB | Minimum | TTS, TTS+VC (no music), STT+TTS |
| 14GB | Mid-range | STS, all TTS modes |
| 15-16GB | Recommended | TTS+VC with music, TTM, TTM+VC |
| 24GB | Maximum | All modes at full speed (RTX 4090) |
| T4 (16GB) | Server-grade | All modes (not typical consumer GPU) |

These aren't arbitrary numbers. They're based on actual testing of the models VODER uses.

---

## Why Hardcoded Models?

VODER uses hardcoded default models. This isn't an accident or a limitation — it's a deliberate design choice made for quality reasons.

### The Quality Imperative

The models VODER uses were selected because they represent the best available quality in their respective categories. Qwen3‑TTS for text‑to‑speech, Seed‑VC v2 for voice conversion, ACE‑Step for music generation — these aren't arbitrary choices. They're the result of evaluating multiple alternatives and selecting the ones that produce the best results.

Smaller models exist. Quantized variants exist. "Fast" versions exist. We deliberately don't use them because they produce noticeably worse output. A smaller TTS model sounds less natural, has more artifacts, and fails on complex text. A quantized voice conversion model loses the subtle characteristics that make voice cloning convincing. Using degraded models would undermine the entire purpose of having VODER exist.

**The HF_TOKEN.txt File:**

You'll find a file called `HF_TOKEN.txt` in the VODER directory. This exists for one reason: to allow advanced users to modify model configurations if they really want to. The file contains instructions for getting your HuggingFace token, and if you provide a valid token, VODER will use it for gated model repositories.

**We Do Not Recommend Changing Models:**

This needs to be stated clearly. The hardcoded models are there because they're the best options available. If you have technical expertise and want to experiment with different model configurations, the capability exists. But VODER is optimized for its default configuration, and deviation from these defaults may produce worse results or cause errors.

Think of it like a restaurant that only serves one dish. They chose that dish because it's the best thing they can make. You can ask them to make something else, but it won't be as good as their specialty. VODER's specialty is orchestrating these specific models together — that's what it does best.

### Custom Versions

If someone creates a modified version of VODER with different model configurations, that's exactly what it is: a modified version. Custom configurations won't be supported in the main VODER documentation or issue tracker because the main project only guarantees quality for its default configuration.

For those interested in exploring custom model configurations, we'll maintain a separate document (CUSTOM_VERSIONS.md) where community‑contributed modifications can be documented. These are not official VODER builds, but if you want to share your experiments with different models or configurations, that file provides a place to do so.

---

## Processing Modes Deep Dive

### TTS: Text-to-Speech

**What It Does:**

TTS generates speech from text using Qwen3‑TTS VoiceDesign. You provide a text script and a voice prompt describing the desired voice characteristics, and VODER produces audio of that voice saying that text.

**How It Works:**

The VoiceDesign model interprets natural language descriptions to generate appropriate voice characteristics. Unlike traditional TTS systems that use pre‑recorded voice samples, VoiceDesign creates voices from scratch based on your description. This makes it incredibly flexible — you can describe voices that don't exist in any database.

**Why It's Like That:**

VoiceDesign exists because not everyone wants to clone an existing voice. Sometimes you need a generic voice for narration, or you want to create a character voice that doesn't correspond to any real person. The descriptive approach provides infinite flexibility without requiring reference audio files.

**Optional Background Music (Dialogue Only):**

When using TTS in **dialogue mode** (multiple speakers, script lines containing a colon), you can optionally add automatically generated background music. After the dialogue is synthesized, VODER generates a music track using ACE‑Step with empty lyrics `"..."` and a duration matching the exact length of the dialogue. The music is mixed at **35% volume** relative to the dialogue, creating a subtle ambient bed. The final file is saved with an `_m` suffix (e.g., `voder_tts_dialogue_..._m.wav`). This feature is available in GUI (via a clean modal dialog), interactive CLI (prompt after voice prompts), and one‑liner CLI (optional `music` parameter). See [Optional Background Music for Dialogue](#optional-background-music-for-dialogue) for full details.

**Best For:**

- Narration and voiceover work
- Creating character voices for content
- Situations where you don't have reference audio
- Rapid prototyping of voice concepts
- Generating multiple voice variations for comparison
- **Dialogue with ambient soundtrack** (podcasts, storytelling)

**Voice Prompt Examples:**

| Desired Voice | Example Prompt |
|---------------|----------------|
| Professional male | "adult male, deep voice, clear pronunciation, professional tone" |
| Warm female | "adult female, warm tone, gentle, conversational" |
| Energetic young | "young adult, energetic, fast‑paced, enthusiastic" |
| News anchor | "middle‑aged, authoritative, measured pace, broadcasting quality" |
| Storytelling | "deep narrative voice, expressive, dramatic pauses" |

**Technical Notes:**

TTS mode works on CPU without GPU acceleration. Processing time scales with text length, not with prompt complexity. The VoiceDesign model interprets prompts at generation time, so more detailed prompts give the model more information to work with but don't significantly affect processing time.

**Memory Requirements:** TTS requires approximately 12GB RAM (8GB base + 4GB for Qwen model).

---

### TTS+VC: Text-to-Speech + Voice Cloning

**What It Does:**

TTS+VC generates speech from text and then applies voice cloning to match a reference voice. The text is synthesized using Qwen3‑TTS Base, and the output is transformed to sound like the voice in your reference audio.

**How It Works:**

The process happens in two stages. First, Qwen3‑TTS Base generates speech from your text using its default voice characteristics. Then, the voice cloning system extracts distinctive features from your reference audio and applies them to the generated speech. The result is your text spoken by a voice that matches your reference.

**Why It's Like That:**

Voice cloning opens possibilities that pure TTS can't match. You can clone a specific person's voice and use it consistently across all your content. You can match voices between different speakers in a dialogue. You can create synthetic content that sounds like real people (with appropriate consent and ethical considerations).

**Optional Background Music (Dialogue Only):**

Just like in TTS mode, when TTS+VC is used in **dialogue mode** you can optionally add automatically generated background music. The music is generated **after** all dialogue lines have been synthesized, concatenated, and voice‑cloned. It uses the same ACE‑Step process (empty lyrics, auto‑duration, 35% volume) and the same output naming (`_m` suffix). The feature is accessible through the same GUI dialog, interactive CLI prompt, and one‑liner `music` parameter. This allows you to create fully produced podcast episodes, narrated stories, or interview segments with ambient background music — all in a single operation.

**Best For:**

- Consistent voice branding across content
- Dialogue with cloned character voices
- Matching voice characteristics between speakers
- Creating content in a voice you don't have but can record
- Localization while preserving original voice characteristics
- **Produced dialogue with background ambience**

**Reference Audio Requirements:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 10‑30 seconds optimal |
| Quality | Clear audio, minimal background noise |
| Content | Continuous speech, not singing or silence |
| Speakers | Single speaker only |
| Format | WAV preferred, MP3 supported |

**Single vs Dialogue Mode:**

In **single mode** (one reference file), the entire script uses that voice. In **dialogue mode** (multiple reference files), each character in a dialogue script is assigned a different reference audio. This is the foundation of VODER's dialogue system, and it is available in **both GUI and CLI**.

**Technical Notes:**

TTS+VC works on CPU without GPU. The voice cloning happens during synthesis, not as a post‑processing step, which ensures the cloned voice characteristics are integrated throughout the generated speech rather than applied superficially.

**Memory Requirements:** TTS+VC requires approximately 12GB RAM (8GB base + 4GB for Qwen model). If using background music, it requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### STS: Speech-to-Speech Voice Conversion

**What It Does:**

STS (Speech‑to‑Speech) transforms source audio to sound like a target voice while preserving the original content, emotion, timing, and prosody. The speaker changes, but everything they say remains exactly the same.

**How It Works:**

Seed‑VC v2 analyzes both the source and target audio to extract content representations and voice characteristics. It then synthesizes new audio that combines the source content with the target voice. This isn't simple audio manipulation — it's neural voice conversion that genuinely reconstructs the speech in a different voice.

**Why It's Like That:**

Voice conversion serves specific use cases that TTS and TTS+VC can't handle. You might have archival audio that needs voice preservation but content modification. You might want to maintain the exact delivery and emotion of a performance while changing the voice. Voice conversion preserves paralinguistic features that text‑to‑speech can't reproduce.

**Best For:**

- Preserving delivery while changing voice
- Content modification in existing audio
- Voice anonymization or de‑identification
- Consistent voice application across multiple recordings
- Archival content republishing with voice updates

**Input Considerations:**

| Factor | Recommendation |
|--------|----------------|
| Duration | 5‑60 seconds optimal per segment |
| Content | Clear speech, minimal background music |
| Quality | Studio quality preferred, phone quality works but loses detail |
| Format | WAV or high‑bitrate MP3 |

**Technical Notes:**

STS runs on CPU without GPU. Input audio is automatically resampled to 22050 Hz for model processing, and output is resampled to 44100 Hz for playback.

**Memory Requirements:** STS requires approximately 13GB RAM (8GB base + 5GB for Seed-VC model).

---

### TTM: Text-to-Music

**What It Does:**

TTM (Text‑to‑Music) generates original music from lyrics and a style prompt using ACE‑Step. You provide song lyrics, describe the desired musical style, and specify duration — VODER creates original music with vocals matching your lyrics.

**How It Works:**

ACE‑Step interprets your lyrics as vocal content and your style prompt as musical direction. It generates both the instrumental arrangement and the vocal performance, synchronized to your specified duration. The lyrics become the vocal melody, and the style prompt guides the instrumentation, genre, and mood.

**Why It's Like That:**

Music generation from lyrics is distinct from instrumental generation because vocals add a layer of complexity. The lyrics must be converted to actual singing, which requires understanding of melody, rhythm, and phonetics. ACE‑Step handles this by treating lyrics as both content and guidance for the vocal generation pipeline.

**Note on Background Music:**

The same ACE‑Step engine is used to generate background music for dialogue. In that context, the lyrics are set to `"..."` (a placeholder for empty vocals), and the style prompt is taken from the user's music description. This yields purely instrumental music suitable for ambient use.

**Best For:**

- Creating original background music with vocals
- Song prototyping and demo creation
- Content needing custom music with lyrics
- Experimental music creation
- Rapid music visualization from lyrics

**Lyrics Format:**

```
Verse 1:
Walking down the empty street
Feeling the rhythm in my feet
The city lights are shining bright
Guiding me through the night

Chorus:
This is our moment, this is our time
Everything's gonna be just fine
Dancing under the moonlight
Everything feels so right
```

**Style Prompt Examples:**

| Genre/Mood | Example Prompt |
|------------|----------------|
| Upbeat pop | "upbeat pop, catchy melody, modern production, female vocals" |
| Rock ballad | "electric guitar, driving drums, powerful vocals, emotional" |
| Electronic dance | "synthesizer, dance beat, energetic, electronic production" |
| Acoustic folk | "acoustic guitar, gentle arrangement, folk style, warm vocals" |

**Duration Considerations:**

| Duration | Use Case |
|----------|----------|
| 10‑30 seconds | Short clips, transitions, soundbites |
| 30‑60 seconds | Full verses or choruses |
| 60‑120 seconds | Complete short songs |
| 120‑300 seconds | Full compositions with multiple sections |

Shorter durations are more reliable and consistent. Very long durations may produce variable results depending on the complexity of lyrics and style combination.

**Technical Notes:**

TTM works on CPU without GPU. Processing time scales primarily with duration rather than lyrics length. The style prompt complexity doesn't significantly affect processing time but does affect the musical output characteristics.

**Memory Requirements:** TTM requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### TTM+VC: Text-to-Music + Voice Conversion

**What It Does:**

TTM+VC generates music from lyrics and style (same as TTM) and then applies voice conversion to change the vocalist's voice. This combines music generation with voice cloning for the singing voice.

**How It Works:**

The pipeline is straightforward: first generate the music with ACE‑Step (TTM stage), then apply Seed‑VC voice conversion to the vocal track (VC stage). The generated music's vocals are transformed to match your reference voice while preserving the melody, timing, and musical characteristics.

**Memory Optimisation:**

VODER now explicitly clears the ACE‑Step model from CPU memory and runs garbage collection before loading Seed‑VC. This reduces peak RAM usage and makes the pipeline more reliable.

**Why It's Like That:**

Sometimes the generated vocals from ACE‑Step don't match the specific voice you need. TTM+VC allows you to generate music efficiently with default vocals, then swap in a cloned voice. This is particularly useful for consistent voice branding in music content or when you need a specific singer's voice in your generated music.

**Best For:**

- Music with specific vocalist voice
- Consistent voice across multiple generated tracks
- Voice‑preserving music modifications
- Professional music production workflows
- Content requiring both music generation and voice cloning

**Technical Notes:**

TTM+VC runs on CPU. This is a composite mode that chains TTM and STS operations, so it inherits the memory requirements of both stages. Longer durations increase the chance of issues.

**Memory Requirements:** TTM+VC requires approximately 23GB RAM (8GB base + 15GB for ACE model).

---

### STT+TTS: Speech-to-Text + Synthesis

**What It Does:**

STT+TTS transcribes audio to text using Whisper, allows you to edit the transcribed content, and then synthesizes the edited text with a target voice. This enables voice modification while preserving the original delivery characteristics.

**How It Works:**

The transcription stage converts speech to text with word‑level timestamps. You can review and modify the transcribed text before synthesis. The synthesis stage then reads your (possibly edited) text and produces audio in the target voice. This preserves the timing and delivery structure from the original audio if you don't modify the text significantly.

**Why It's Like That:**

This mode is for when you have existing audio content that needs voice transformation. By transcribing, editing, and resynthesizing, you can change what someone says while keeping the general timing and delivery. It's not a simple voice conversion — it's a reconstructive process that allows complete content modification.

**Best For:**

- Changing content in existing audio
- Fixing transcription errors automatically
- Localizing content into different languages
- Creating fictional dialogue from real voice samples
- Voice modification with full control over content

**Interactive Nature:**

STT+TTS requires user interaction for text editing, which is why it's only available in interactive CLI mode and GUI mode. The one‑liner mode cannot accommodate this workflow. You must either use `python src/voder.py cli` and select the STT+TTS option, or use the GUI for full visual feedback.

**Multi‑Speaker Warning:**

If your base audio contains multiple speakers, Whisper will transcribe all of them. The synthesis will use a single target voice for the entire text. This creates an unnatural result where multiple speakers sound like the same person. For true multi‑voice dialogue, use the dialogue system instead.

**Technical Notes:**

STT+TTS works on CPU without GPU for the Whisper transcription stage. Voice cloning in the synthesis stage also works on CPU. This makes it accessible for users without NVIDIA graphics hardware.

**Memory Requirements:** STT+TTS requires approximately 12GB RAM (8GB base + 4GB for Qwen model).

---

## The Dialogue System

### What Dialogue Mode Is

VODER's dialogue system enables multi‑speaker script generation. You write a script with multiple characters, assign voice references to each character, and VODER generates a cohesive audio track where each line is spoken by the appropriate voice.

**What It Is NOT:**

Despite how it might seem, dialogue mode is not AI systems conversing with each other. There are no neural networks having conversations. Each line is synthesized independently, one after another, using the specified voice reference. The "conversation" effect is achieved through:

- Sequential processing of script lines in order
- Voice routing that matches characters to their assigned samples
- FFmpeg concatenation that preserves timing and flow
- Independent synthesis of each line with consistent voice characteristics

It's automation, not artificial conversation intelligence.

**Dialogue is Now Available in CLI:**

Earlier versions of VODER restricted dialogue creation to the GUI. **This is no longer the case.** As of the latest update, dialogue mode is fully supported in **both GUI and CLI**, including one‑liner commands and interactive CLI input. All references to "dialogue is GUI‑only" in older documentation are outdated.

### How It Works

The dialogue processing pipeline follows these stages:

1. **Parse Script**: Extract dialogue items with sequence number, character name, and text
2. **Parse Voice Prompts**: Build character‑to‑audio‑reference mapping
3. **Validate**: Ensure every character has a voice reference
4. **Temporary Files**: Create temporary directory for segment audio files
5. **Iterate Lines**: For each dialogue line:
   - Load corresponding voice reference audio
   - Extract voice characteristics from reference
   - Synthesize the line text using that voice
   - Save segment to temporary file
6. **Concatenate**: Use FFmpeg to combine all segments into one file
7. **Optional Background Music**: If requested, generate and mix music
8. **Clean Up**: Remove temporary files
9. **Export**: Save final dialogue to results folder

### Why Not Multi-Speaker Input?

You might wonder: why not just load audio with multiple speakers and let the AI figure it out? Here's why that approach fails:

**Speaker Separation is Hard:**

Even state‑of‑the‑art speaker diarization systems make mistakes. When you have multiple speakers in one audio file, separating who said what accurately is a challenging problem. Errors in speaker identification lead to wrong voice assignments, which ruins the final output.

**Voice Consistency Issues:**

When speakers aren't cleanly separated, the voice cloning produces inconsistent results. One line might sound like the target voice, the next line might drift. This creates a jarring listening experience where the same character sounds different from one sentence to the next.

**No Character Control:**

Multi‑speaker input gives you no control over which voice says which line. If you have three voice references and one audio file with two speakers, how do you assign voices? The system can't know your creative intent.

**Dialogue Mode Solves These Problems:**

With dialogue mode, you have explicit control:

- You write exactly what each character says
- You assign specific voice references to specific characters
- Each line gets processed independently with the correct voice
- The result is consistent, controllable, professional

---

### Dialogue Input in GUI

VODER's GUI now uses a **row‑based dialogue editor** instead of a free‑text script box. This design makes character assignment explicit and reduces parsing errors.

**Script Entry:**

- Each line is a separate row containing **Character** and **Dialogue** fields.
- New rows are automatically added when you fill the last row.
- The first row has no delete button; subsequent rows can be deleted individually.
- The GUI automatically tracks which characters appear in the script and displays a corresponding voice prompt area for each character.

**Voice Prompt Assignment:**

- In **TTS mode** (Voice Design): each character gets a `QLineEdit` where you type a natural‑language voice description.
- In **TTS+VC mode** (Voice Cloning): each character gets a `QComboBox` dropdown listing the numbers of the audio files you have loaded. You simply select the number corresponding to the reference audio you want for that character.

**Audio Reference Management (TTS+VC):**

- Use the **"Add Audio"** button to load reference files.
- Each loaded file is assigned a unique number (1, 2, 3…).
- You can play or delete any file from the list at any time.
- When you add or delete files, the dropdowns in the voice prompt area are automatically updated with the current set of numbers.

**Example GUI Workflow (TTS+VC):**

1. Switch to TTS+VC mode.
2. In the script area, add rows:  
   `James: Welcome to the podcast.`  
   `Sarah: Thanks for having me.`  
   `James: Let's talk about AI.`
3. Load three audio files: `james_voice.wav`, `sarah_voice.wav`, `james_voice.wav` (again for the second James line).
4. The voice prompt area automatically shows rows for **james** and **sarah** with dropdowns containing `1, 2, 3`.
5. Assign:  
   `james` → `1`  
   `sarah` → `2`  
   `james` → `3` (or you can reuse `1` if you prefer the same reference)
6. (Optional) A dialog will appear asking if you want background music — see [Optional Background Music for Dialogue](#optional-background-music-for-dialogue).
7. Click **Generate**. VODER synthesizes each line with the appropriate cloned voice, concatenates them, and (if requested) mixes with background music.

**Why This Design:**

- Eliminates format errors (no more `1:James: "text"` syntax mistakes).
- Makes character‑to‑audio assignment visual and immediate.
- Prevents mismatches between script characters and available references.
- Enables quick auditioning of different voice assignments.

---

### Dialogue Input in CLI

VODER now provides two ways to create dialogue in the command line: **interactive** and **one‑liner**.

#### Interactive CLI Dialogue

Run `python src/voder.py cli`, select mode 2 (TTS) or 3 (TTS+VC). You will be prompted to enter script lines. Enter one line per row, using the format:

```
Character: text
```

Type your lines, press Enter after each, and leave an empty line to finish. VODER automatically detects dialogue mode (because lines contain `:`). It then asks for voice prompts (TTS) or audio file paths (TTS+VC) **for each character that appeared**, in sorted order.

**Example (TTS):**

```
$ python src/voder.py cli
...
Select Mode: 2
Enter script lines. Use format 'Character: text' for dialogue, or plain text for single speech.
Empty line finishes script entry.
> James: Welcome to the show.
> Sarah: Glad to be here.
> 
Voice prompts for 2 character(s):
james: deep male voice, authoritative
sarah: bright female voice, cheerful
```

**After** collecting all voice prompts/assignments, you will be asked:

```
Add background music? (y/N):
```

If you answer `y` or `yes`, you can enter a music description. Leaving the description blank or pressing Enter without input skips the music. VODER then generates the full dialogue (with or without background music).

**Example (TTS+VC):**

```
$ python src/voder.py cli
...
Select Mode: 3
Enter script lines...
> Narrator: Once upon a time...
> Alice: I wonder what this does.
> Bob: Let's find out.
> 
Audio file paths for 3 character(s):
narrator: /voices/narrator.wav
alice: /voices/alice.wav
bob: /voices/bob.wav
Add background music? (y/N): y
Music description: soft piano, cinematic strings
```

**Why Interactive CLI Dialogue Exists:**

- Users who prefer terminal workflows can now create full multi‑speaker content without launching the GUI.
- The interactive prompts ensure that every character receives a valid voice reference before processing begins.
- It bridges the gap between full automation (one‑liner) and visual interfaces.
- The optional music prompt fits naturally into this interactive flow.

#### One‑Liner Dialogue

One‑liner commands now support dialogue through **multiple values per parameter**. This is the recommended method for automated scripts and AI agents.

**Syntax for TTS dialogue:**

```bash
python src/voder.py tts script "Character1: line1" "Character2: line2" voice "Character1: voice description" "Character2: voice description"
```

**Syntax for TTS+VC dialogue:**

```bash
python src/voder.py tts+vc script "Character1: line1" "Character2: line2" target "Character1: /path/to/reference1.wav" "Character2: /path/to/reference2.wav"
```

**Optional Background Music in One‑Liner:**

To add background music, simply include a `music` parameter with your description:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep male" "Sarah: cheerful female" music "soft piano, cinematic"
```

If the `music` parameter is supplied but the script is **not** in dialogue mode (i.e., no colon in any `script` parameter), it is ignored with a warning. If the `music` parameter is present but its value is an empty string (`music ""`), it is treated as if no music was requested.

**Important Rules:**

- The order of `script` values must match the dialogue line order.
- The order of `voice`/`target` values must match the character order (first appearance in script).
- For single‑speaker scripts, you may omit the colon in both script and voice/target; the system will treat it as single mode and ignore any `music` parameter.
- You can also use explicit keyword repetition if preferred (backward compatible).

**Examples:**

```bash
python src/voder.py tts script "James: Hello, Sarah." "Sarah: Hi James, how are you?" "James: I'm great, thanks for asking!" voice "James: deep male, warm" "Sarah: young female, cheerful" music "ambient electronic, chill"

python src/voder.py tts+vc script "Host: Welcome to the podcast." "Guest: Thanks for having me." "Host: So, tell us about your work." target "Host: /voices/host.wav" "Guest: /voices/guest.wav" "Host: /voices/host.wav" music "soft piano, strings"
```

**Validation:**

If any character in the script does not have a matching voice/target entry, VODER will reject the command with a clear error message listing the missing characters.

---

### Voice Prompt Configuration

The mapping between characters and their voice references (audio file numbers or file paths) is handled differently in GUI and CLI, but the underlying concept is the same.

#### In GUI (TTS+VC)

- Audio files are loaded into a list and automatically numbered.
- The voice prompt area provides a dropdown per character showing all available numbers.
- You select the number that corresponds to the desired reference file.

#### In GUI (TTS)

- The voice prompt area provides a text field per character.
- You type a natural‑language description (e.g., "warm female narrator").

#### In CLI (TTS)

- For dialogue, you supply `voice "Character: description"` entries.
- For single mode, you supply `voice "description"` (no colon).

#### In CLI (TTS+VC)

- For dialogue, you supply `target "Character: /path/to/audio.wav"` entries.
- For single mode, you supply `target "/path/to/audio.wav"` (no colon).

**No More Numbered Prompts in GUI:**

Older versions of VODER required you to write prompts like `James:1` in a text box. This is **no longer used in the GUI**. The dropdown system eliminates syntax errors and makes voice assignment explicit.

---

### Optional Background Music for Dialogue

VODER includes a unique feature that automatically generates and mixes ambient background music into dialogue scripts. This is **only available for dialogue mode** (i.e., when the script contains at least one line with a colon) and works for both **TTS** and **TTS+VC** modes.

#### How It Works

1. **Dialogue Synthesis** – VODER first generates all dialogue segments, concatenates them into a single audio file using FFmpeg, and saves it temporarily.
2. **Duration Measurement** – The exact duration of the dialogue audio is calculated (using `torchaudio.info`).
3. **Music Generation** – The ACE‑Step model is loaded (if not already) and used to generate a music track with:
   - **Lyrics**: `"..."` (a placeholder that yields pure instrumental music)
   - **Style prompt**: the description provided by the user (e.g., `"soft piano, cinematic strings"`)
   - **Duration**: exactly the length of the dialogue audio (rounded to nearest whole second)
4. **Volume Adjustment** – The music track is reduced to **35% of its original volume** using FFmpeg's `volume=0.35` filter. This level has been empirically chosen to provide a noticeable but non‑intrusive ambient bed.
5. **Mixing** – The attenuated music is mixed with the dialogue using FFmpeg's `amix` filter, which sums the two streams and preserves the longer duration (the music is generated to match exactly, so both durations are equal).
6. **Memory Management** – After dialogue synthesis, the Qwen‑TTS model is released from memory. After music generation, the ACE‑Step handler is explicitly deleted and, if CUDA is available, `torch.cuda.empty_cache()` is called. This reduces peak VRAM usage and makes the feature viable on 8GB cards.
7. **File Cleanup** – Both the temporary dialogue file and the temporary music file are deleted. Only the final mixed file remains in the `results/` directory.
8. **Output Naming** – The output file is named with an `_m` suffix, e.g., `voder_tts_dialogue_20250212_143022_m.wav`. This makes it immediately clear that the file contains background music.

#### GUI Workflow

When you click **Generate** in TTS or TTS+VC mode **and** the script contains at least one line with a colon (i.e., dialogue mode), VODER displays a clean modal dialog before any processing begins:

<p align="center">
  <i>Background Music Dialog</i><br>
  <code>Enter music description (or press Skip):</code><br>
  <code>[ OK ] [ Skip ]</code>
</p>

- **OK**: If you enter a non‑empty description and click OK, VODER will proceed with music generation as described above. If the description is empty, a warning is shown and you are returned to the dialog.
- **Skip**: Clicking Skip bypasses music generation entirely.

The dialog is styled consistently with the rest of VODER's GUI and respects the same color scheme and font choices.

#### Interactive CLI Workflow

In interactive CLI mode (TTS or TTS+VC), after you have entered all script lines and provided all voice prompts/audio paths, VODER asks:

```
Add background music? (y/N):
```

- If you type `y` or `yes`, it then prompts:
  ```
  Music description:
  ```
  Enter your description (e.g., `soft piano, cinematic`). If you press Enter without typing anything, the description is considered empty and VODER **skips** music generation (no warning; it's treated as a normal skip).
- If you type anything else (or just press Enter), music is skipped.

This flow is natural, non‑intrusive, and requires only one extra decision point.

#### One‑Liner CLI Workflow

For one‑liner commands, the `music` parameter is used:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep" "Sarah: bright" music "soft piano"
```

- If the `music` parameter is **present and its value is non‑empty**, background music is generated.
- If the `music` parameter is **present but its value is an empty string** (`music ""`), it is ignored (no music).
- If the `music` parameter is **absent**, no music is generated.
- If the script is **not** in dialogue mode (i.e., all `script` parameters are plain text without colon), the `music` parameter is ignored and a warning is printed.

This design allows automated scripts to optionally include music without breaking existing workflows.

#### Technical Implementation

The feature is implemented in `ProcessingThread` with two new modes:

- `tts_voice_design_dialogue` – handles TTS dialogue + optional music
- `tts_vc_dialogue` – handles TTS+VC dialogue + optional music

Both modes follow the same pattern:

```python
# 1. Generate dialogue audio
dialogue_temp = synthesize_and_concat(...)

# 2. If music_description is not None:
if music_description:
    # 3. Get dialogue duration
    duration = get_audio_duration(dialogue_temp)
    # 4. Generate music with ACE‑Step
    music_temp = ace.generate(lyrics="...", style_prompt=music_description, duration=duration)
    # 5. Mix at 35% volume using FFmpeg
    mixed_temp = ffmpeg_mix(dialogue_temp, music_temp, volume=0.35)
    # 6. Replace output with mixed file
    os.replace(mixed_temp, output_path)
    # 7. Clean up temporary files
    os.unlink(dialogue_temp)
    os.unlink(music_temp)
else:
    os.replace(dialogue_temp, output_path)
```

**Why 35%?** This value was determined through listening tests: at 35% relative volume, the music is clearly audible but does not compete with the spoken word for attention. Higher volumes (>40%) begin to mask speech; lower volumes (<30%) become too subtle. The value is hardcoded for consistency – there is no user‑adjustable volume control, because that would introduce another variable and complicate the user experience. If you need different mixing levels, you can always post‑process the output with an external audio editor.

**Why `"..."` as lyrics?** ACE‑Step requires a non‑empty lyrics string. Using three dots `...` is a conventional placeholder that reliably produces instrumental music with no discernible vocals. It has been tested across many style prompts and consistently yields the desired ambient track.

**Why auto‑fit duration?** Manually specifying a duration would create two problems: (1) the user would need to know the exact dialogue length in advance, and (2) the music would either be cut off or fade out before the dialogue ends. By auto‑fitting, VODER guarantees that the music plays for the entire dialogue and stops exactly when the speech ends. This creates a polished, professional feel.

**Memory optimisation:** The dialogue generation stage loads either Qwen3‑TTS VoiceDesign or Qwen3‑TTS Base. After the dialogue file is written, these models are allowed to be garbage‑collected. When music is requested, ACE‑Step is loaded, used, and then explicitly deleted with `del self.ace_tt` followed by `torch.cuda.empty_cache()`. This frees GPU memory before the next operation (which is none, since mixing is done via FFmpeg on CPU). This careful management makes the feature usable even on 8GB GPUs.

**File naming:** The `_m` suffix is added to the base filename. This is a simple, visible indicator that the file contains background music. It also prevents accidental overwriting of the non‑music version if you generate both variants.

**Cleanup:** All temporary files (individual dialogue segments, concatenated dialogue, generated music) are deleted. Only the final output file remains in `results/`. This keeps your working directory tidy and avoids accumulating gigabytes of intermediate audio.

---

## Tips & Tricks

### Getting Better Results

**For TTS (Voice Design):**

- Be specific in voice prompts — "warm adult female" is better than just "female"
- Include pacing hints if you want specific rhythm — "slow, deliberate" or "fast, energetic"
- Mention the use case if relevant — "podcast host" or "news broadcast voice"
- Experiment with variations — small prompt changes can significantly affect output

**For TTS+VC (Voice Cloning):**

- Use 10‑30 seconds of clean reference audio
- Avoid background music or noise in reference
- Ensure consistent volume throughout reference
- Single continuous speech is better than multiple short clips
- The reference voice quality directly affects clone quality

**For STS (Voice Conversion):**

- Base and target should have similar audio characteristics
- If base is phone‑quality, target should also be phone‑quality
- Very short clips (under 2 seconds) may not convert well
- Very long clips (over 5 minutes) may cause memory issues
- Clear speech converts better than expressive/emphatic speech

**For TTM (Music Generation):**

- Structure lyrics with verse/chorus markers for better organization
- Keep lyrics simple for more coherent results
- Style prompts work best when specific — "80s synthpop" is better than "good"
- Shorter durations (30‑60 seconds) are more reliable
- Complex lyrics with unusual structures may produce inconsistent results

**For TTM+VC:**

- Because this mode chains two models, it uses more VRAM than either alone
- The memory optimisation now frees ACE‑Step before loading Seed‑VC, reducing OOM risk
- If you encounter memory issues, try reducing duration or processing shorter segments

### Multi-Speaker Scenarios

**Always Use Dialogue Mode:**

If you need multiple voices, use dialogue mode. This is not optional advice — it's how VODER is designed to work. Trying to work around dialogue mode by:

- Loading multi‑speaker audio and hoping for the best
- Manually stitching together separate TTS+VC outputs
- Using voice conversion on multi‑speaker audio

...all of these produce worse results than simply using the dialogue system that's built specifically for this purpose.

**Dialogue Planning:**

1. Write your script with character names
2. Gather reference audio for each character (10‑30 seconds each)
3. In GUI: load references, assign via dropdowns
4. In CLI: provide references via repeated `target` parameters or interactively
5. (Optional) Decide if you want background music
6. Generate dialogue in one operation
7. Review and iterate if needed

**Character Consistency:**

Once you've assigned a voice reference to a character, keep using the same reference for that character throughout the project. Changing references mid‑dialogue creates inconsistent results.

### Using Same Audio Source

**The Modification Trick:**

For STT+TTS mode, if you use the same audio file as both base (content) and target (voice), you get voice modification. The transcribed text becomes editable, and the synthesis uses the same voice characteristics from the original audio. This allows you to:

- Change words or phrases while keeping the original voice
- Fix awkward phrasing while maintaining voice consistency
- Localize content while preserving original voice characteristics
- Create fictional quotes from real voice samples

**When This Works Best:**

- Reference audio is clean and of good quality
- You want minimal change to the overall delivery
- You're making small edits, not rewriting entire passages
- The original voice has clear, consistent characteristics

### Voice Cloning Best Practices

**Reference Audio Quality Hierarchy:**

| Quality | Characteristics | Result |
|--------|-----------------|--------|
| Excellent | Studio recording, no noise, consistent volume | Best clone quality |
| Good | Clean recording, minimal background, consistent | Good clone quality |
| Acceptable | Some background, slight inconsistencies | Acceptable quality |
| Poor | Heavy noise, compression artifacts, inconsistent | Poor clone quality |

**What to Avoid in Reference Audio:**

- Background music or sounds
- Multiple speakers (even briefly)
- Extreme volume variations
- Phone‑quality or highly compressed audio
- Emotional extremes that distort voice characteristics
- Audio that has been heavily processed or filtered

**The 10‑30 Second Sweet Spot:**

Reference audio between 10 and 30 seconds produces the best results. Shorter references may not capture enough voice characteristics. Longer references don't significantly improve quality and take longer to process.

### Background Music Best Practices

**When to Use It:**

Background music enhances dialogue when used tastefully. It's particularly effective for:

- Podcast intros and outros
- Narrative storytelling (audiobooks, guided meditations)
- Cinematic dialogue scenes
- Interview segments with ambient backing
- Educational content that benefits from a relaxed mood

**When to Skip It:**

Not every dialogue needs music. Consider skipping if:

- The content is informational/dry (music can be distracting)
- You plan to add music later in post‑production
- The dialogue itself is the primary focus (e.g., news reading)
- You're testing or iterating rapidly (music generation adds time)

**Choosing a Music Description:**

The style prompt for background music should match the mood of the content. Some guidelines:

| Mood | Description Example |
|------|---------------------|
| Relaxed, thoughtful | "soft piano, gentle strings, ambient" |
| Energetic, upbeat | "upbeat electronic, modern production" |
| Mysterious, suspenseful | "dark ambient, low drone, cinematic" |
| Inspirational, uplifting | "orchestral, emotional, building crescendo" |
| Corporate, professional | "corporate background, subtle, professional" |

**Avoid** overly complex or specific descriptions like "solo violin in D minor, arpeggios, with reverb" – ACE‑Step works better with broader genre/mood cues.

**Duration Handling:**

The music is always exactly as long as the dialogue. This means:

- If your dialogue is 42.7 seconds, the music will be 42 seconds (rounded to nearest whole second).
- No fade‑out is applied; the music stops abruptly when the dialogue ends. This is intentional – if you need a fade, you can add it later.
- Very short dialogue (<10 seconds) may still generate music, but ACE‑Step performs best with durations ≥10 seconds.

**Volume Level:**

The fixed 35% volume has been carefully chosen. If you find it too loud or too soft, you can adjust it with an external audio editor. We do not provide a user‑adjustable volume slider because it would add complexity and likely be misused (e.g., set to 100% and then complain that the music overpowers the speech).

**Performance Impact:**

Generating background music adds approximately 10‑20 seconds of processing time for a 30‑second dialogue (on a modern CPU). On GPU, it is faster. This is usually negligible compared to the time saved by not having to manually find, edit, and mix a music track.

**File Management:**

All temporary files are deleted. Only the final `.wav` file (with the `_m` suffix) remains in `results/`. If you need both the dialogue‑only and music‑mixed versions, generate twice (once without music, once with). The naming convention prevents accidental overwrites.

---

## Version Information

**Timestamp‑Based Versioning:**

VODER uses timestamp‑based versioning rather than semantic versioning (v1.2.3, etc.). Each build is identified by its creation timestamp in YYYYMMDD_HHMMSS format. This approach reflects VODER's development philosophy — continuous improvement rather than numbered releases.

**Why Not Semantic Versioning:**

Traditional semantic versioning implies discrete releases with specific feature sets and bug fixes between versions. VODER development doesn't follow that pattern. Changes are made when they're ready, tested, and merged. A user downloading VODER today gets the absolute latest version with all improvements since the last commit.

**Version Tracking:**

- Each commit to the main branch gets a timestamp
- The CHANGELOG.md documents significant changes with dates
- No numbered releases means no "latest stable version" confusion
- Everyone always uses the current development version

**No PyPI Package:**

Unlike IMDER, VODER is not distributed via PyPI. Running from source is the only way to use VODER. This ensures:

- Always access to latest features
- No version compatibility issues
- Direct access to development version
- Transparency in what's running

---

## Troubleshooting & Common Issues

### STS/TTM+VC Fails Immediately

**Cause:** No NVIDIA GPU detected or insufficient VRAM

**Solution:**
```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Check VRAM
nvidia-smi
```

If no GPU is detected, Seed‑VC modes cannot work. These modes require NVIDIA GPU with minimum 8GB VRAM.

### Out of Memory Errors

**Cause:** Model too large for available memory

**Solution:**
- Process shorter audio segments
- Reduce TTM duration (shorter music = less memory)
- Close other GPU‑intensive applications
- Ensure no other processes are using GPU memory

### FFmpeg Not Found

**Cause:** FFmpeg not installed or not in system PATH

**Solution:**
```bash
# Verify FFmpeg installation
ffmpeg -version

# Install if needed
# Windows: winget install FFmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### HuggingFace Model Download Fails

**Cause:** Network issues or gated repository access

**Solution:**
1. Check internet connection
2. Add HuggingFace token to HF_TOKEN.txt for gated models
3. Clear cache and retry:
   ```bash
   rm -rf ./models ./checkpoints
   python src/voder.py
   ```

### Voice Cloning Produces Poor Results

**Cause:** Poor quality reference audio

**Solution:** Use high‑quality reference audio:
- 10‑30 seconds duration
- Clear speech, minimal background noise
- Single speaker, no music
- Consistent volume levels
- No post‑processing or effects

### Dialogue Character Not Found (GUI)

**Cause:** Character name mismatch between script and voice prompt assignments

**Solution:** The GUI automatically tracks characters from the script rows and displays prompts for each. If a character is missing from the voice prompt area, it means no row in the script uses that character. Ensure the character name is spelled consistently across all rows.

### Dialogue Character Not Found (CLI)

**Cause:** Missing `voice` or `target` entry for a character in one‑liner mode

**Solution:** Check that every character that appears in `script` parameters also appears exactly once in the corresponding `voice`/`target` parameters, with the same spelling (case‑insensitive). Example:

```bash
python src/voder.py tts script "James: Hello" "Sarah: Hi" voice "James: deep voice" "Sarah: cheerful voice"
```

### GUI: Audio Dropdowns Not Appearing

**Cause:** TTS+VC mode requires at least one audio file loaded

**Solution:** Click **"Add Audio"** and load at least one reference file. The dropdowns will populate automatically.

### Background Music Not Added (GUI)

**Cause:** You pressed Skip or left the description empty

**Solution:** In the dialog, enter a non‑empty description and click OK. If you accidentally skipped, you must regenerate with the correct option.

### Background Music Not Added (One‑Liner)

**Cause:** The `music` parameter was omitted, its value was empty, or the script was not in dialogue mode

**Solution:** Ensure:
- At least one `script` parameter contains a colon (`Character: text`)
- You include `music "description"` with a non‑empty string
- You are in TTS or TTS+VC mode

### Background Music Generation Fails

**Cause:** ACE‑Step model not loaded, insufficient resources, or invalid music description

**Solution:**
- Check that you have sufficient RAM (16GB recommended)
- Try a simpler music description (e.g., "piano")
- Verify that FFmpeg is installed and in PATH
- If using GPU, ensure you have at least 8GB VRAM (or use CPU – slower but works)

### Music Volume Seems Off

**Cause:** Subjective perception; 35% is a fixed default

**Solution:** If you consistently find the volume too high or too low, you can post‑process the output file with an audio editor. For automated workflows, you could add an FFmpeg command after generation to adjust the volume further.

### Quality Issues with TTM

**Cause:** Complex lyrics or ambitious style prompts

**Solution:**
- Simplify lyrics structure
- Use more conventional style descriptions
- Try shorter durations first
- Start with well‑known genres ("pop", "rock") before experimenting

### Mode-Specific Reference

| Mode | Common Issue | Solution |
|------|--------------|----------|
| STT+TTS | Multi‑speaker confusion | Use dialogue mode instead |
| TTS | Unnatural voice | More detailed prompts |
| TTS+VC | Clone quality issues | Better reference audio |
| STS | Conversion fails | Shorter input, check VRAM |
| TTM | Inconsistent music | Shorter duration, simpler lyrics |
| TTM+VC | Out of memory | Memory optimisation already helps; try shorter duration |
| Dialogue (any) | Missing character assignment | Ensure every character has a voice prompt/audio path |
| Dialogue (music) | Music not generated | Use non‑empty description, ensure dialogue mode |

---

## Final Notes

VODER is a tool built for creators, developers, and audio professionals who need professional‑grade voice processing without subscription fees or usage limits. It prioritizes quality over speed, simplicity over complexity, and utility over marketing.

All six processing modes work reliably. The "problematic modes" designation from earlier versions is outdated — Seed‑VC v2 has proven stable across the use cases VODER supports. If you encounter issues, they're more likely to be related to resource constraints or input quality than mode‑specific bugs.

**Dialogue is now everywhere.** The GUI provides a visual, error‑free script editor with dropdown voice assignments. The CLI offers both interactive and one‑liner dialogue creation. **And now, dialogue can be optionally enhanced with automatically generated, duration‑fitted background music.** This feature completes the dialogue production pipeline, allowing you to create finished, polished audio content in a single operation.

**Background music is the final piece.** With it, VODER transforms from a mere voice processor into a complete audio production workstation. Podcasters can generate entire episodes with music beds. Storytellers can add cinematic ambience. Educators can create engaging narrated content. All of this is possible because we integrated the music generation model we already had (ACE‑Step) into the dialogue pipeline in a thoughtful, user‑friendly way.

**Choose the interface that fits your workflow.** If you love visual interaction, use the GUI. If you live in the terminal, use the interactive CLI. If you're an AI agent or need to automate thousands of generations, use the one‑liner. Every interface has full access to dialogue, voice cloning, and now background music.

**Remember:** Quality over speed. Use dialogue mode for multi‑speaker content. Reference audio quality matters. Music descriptions should match the mood. And when in doubt, start with simpler configurations before experimenting with advanced workflows.

For questions, issues, or collaboration opportunities, visit the GitHub repository or reach out through (X)[https://x.com/HAKORAdev].

---

*VODER — They say what you want them to say.*
