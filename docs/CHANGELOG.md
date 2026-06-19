# CHANGELOG

- All notable changes to VODER will be documented in this file.
- This project does not use version names like v1.2.3; it just timestamps changes. It will always be updated every time I notice something wrong.
- If you are really interested on what happens in this project, tracing the commit history would be better because I forget to document every change (or if you are mad enough, just read voder.py).

## 06/20/2026
- Status: Stable, all features work, still developing
- **13 New Side-Quests: audio utility belt + slowed+reverb toolkit for the chains era**

### Added

- **`quest bassboost <1-100> <audio|video>`** — Professional multi-band bass booster. Scale 1–100 where `1` = subtle warmth (+0.24 dB shelf), `50` = strong club bass (+12 dB shelf, +9 dB peak), `100` = sub-destroyer (+24 dB shelf, +18 dB peak). Selectively boosts low frequencies only (20–250 Hz); mids and highs are left untouched. The 6-stage signal chain (all designed to avoid dotty / buzzy artifacts at any value): 1) `highpass=f=30` removes inaudible sub-30 Hz rumble that would eat headroom; 2) `bass` low-shelf filter at 80 Hz corner with 80 Hz width, gain scales linearly 0→+24 dB; 3) `equalizer=f=50:w=40:t=q` adds a narrow peaking boost at 50 Hz for sub-bass punch, gain scales 0→+18 dB; 4) `virtualbass` synthesizer at 250 Hz cutoff generates sub-bass harmonics audible on small speakers (strength 0.3→3.0); 5) `acompressor` soft-knee compressor (threshold 0.5→0.15, ratio 2:1→5:1, attack 10 ms, release 200 ms, makeup +1.1×, knee 4 dB) glues the bass into the mix and prevents transient peaks from clipping; 6) `alimiter` true-peak limiter at -1 dB (5 ms attack, 50 ms release) as the final safety net. Audio input → 24-bit/48k WAV. Video input → MP4 with video copied, audio re-encoded as AAC 256k.

- **`quest volume` refactored to pure linear gain** — Previously `quest volume` was a chimera: named "volume" but described as a "bass booster" and scaled like gain (1–1000, 100=×2). It ran a 6-stage chain (bass → virtualbass → treble → volume → acompressor → loudnorm). That conflated two independent effects in one knob. The quest now does ONE thing: pure linear volume gain (multiply every sample by `1 + value/100`). Scale 1–1000 unchanged (100 = 2×, 1000 = 11×). Affects all frequencies equally. No bass boost, no treble lift, no compression, no loudness normalization. Use it when you just want the audio louder / quieter without changing its tonal character. For tonal shaping, use the new `quest bassboost` (low-frequency enhancement, 1–100 scale). The two quests compose: `volume 200 "song.wav"` (3× louder) then `bassboost 50` (+12 dB of bass) = independent control of overall loudness vs low-end punch.

- **`quest reverb <1-100> <audio|video|URL>`** — Professional algorithmic reverb on a 1–100 integer scale. `1` = barely-there small room, `25` = chamber, `50` = concert hall, `75` = large hall, `100` = cathedral-drenched. Built in the classic Schroeder topology (the same architecture used by pro studio reverbs before convolution took over). Freeverb is not compiled into this FFmpeg build, so the reverb is constructed from FFmpeg's `aecho` (multi-tap delay) for early reflections AND late-reverb tail, plus `adelay` (pre-delay 5–80 ms scaling with value), `lowpass` (air-absorption damping, cutoff 6–13 kHz scaling with value), `acompressor` (peak control), `dynaudnorm` (dynamic normalization that works at any input level — `loudnorm` with `linear=true` fails on quiet signals), and `alimiter` (true-peak limiter as final safety net). Audio output only — video inputs are accepted but only the audio stream is processed. Accepts local audio files, local video files, and YouTube / Bilibili / TikTok URLs (audio downloaded via yt-dlp, temp file cleaned up after). Output: WAV (PCM 24-bit, 48 kHz, stereo). This is the third independent dimension of the slowed+reverb toolkit — reverb simulates sound bouncing in a physical space, independent of `quest speed` (tempo change) and `quest pitch` (frequency change). The full slowed+reverb edit chains all of them: `volume → bassboost → speed → pitch → reverb → glue`.

- **`quest pitch <0.01-10.00> <audio|video|URL>`** — Professional pitch shift using FFmpeg's `rubberband` filter. Range 0.01–10.00 in 0.01 increments (1.00 is excluded as a no-op). Pitch is shifted without changing tempo (the mirror of `quest speed`, which changes tempo without changing pitch). `0.50` = −1 octave (monster / demon voice), `2.00` = +1 octave (baby / chipmunk voice), `0.01` = extreme deep (≈6.64 octaves down), `10.00` = extreme high (≈3.32 octaves up). Uses `formant=shifted` (rubberband's default) — formants move with the pitch, giving the classic tape / vinyl character that makes the demon / baby / slowed+reverb aesthetic actually sound right. For values outside the ±1-octave clean range (0.50–2.00), the shift is automatically decomposed into chained one-octave passes (e.g., `pitch 0.01` becomes 6 passes of 0.5 + 1 pass of 0.64), keeping each rubberband invocation in its clean-operating range. Per-pass config: `formant=shifted`, `transients=crisp`, `detector=compound`, `phase=laminar`, `pitchq=quality`, `channels=apart`. Accepts local audio files, local video files (only audio stream read, video frames dropped), and YouTube / Bilibili / TikTok URLs (audio downloaded via yt-dlp, temp file cleaned up after). Output: WAV (PCM 24-bit, 48 kHz, stereo). Audio output only.

- **`quest glue "<input-to-use>" "<where-it-will-be-glued>"`** — Mux / replace utility: glues an audio file onto a video file (or vice versa). The video source's frames are combined with the audio source's audio, producing an MP4. Order of arguments only determines which file is the "video source" vs the "audio source" — output is always a video file. **Auto-replaces existing audio:** if the video input already has an audio track, it is dropped and replaced with the audio from the audio input (no `replace` keyword needed). **Duration handling:** output duration is always the *longer* of the two inputs — if audio is shorter than video, audio is padded with silence (`apad=pad_dur=<diff>`) until the last video frame; if video is shorter than audio, video is extended with black frames (`tpad=stop_mode=add:stop_duration=<diff>`) until the audio ends. **Refused combinations:** URLs of any kind (must be local files — use `quest download` first), audio+audio (use `quest merge` instead), video+video (use `quest noframes` on one of them first). Output: MP4 (H.264 video, AAC 256 kbps audio, CRF 20, +faststart). Output naming: `voder_quest_glue_<audio-name>_onto_<video-name>_<timestamp>.mp4`.

- **New chain pattern — slowed+reverb toolkit:** `quest volume` → `quest bassboost` → `quest speed` → `quest pitch` → `quest reverb` → `quest glue` produces a louder, bass-boosted, slowed-down, pitch-down, cathedral-drenched version of a song glued back onto its original music video. All six quests are chain-friendly (each produces a single output file that the next chain can reference by name). Three independent effect axes combine into the classic slowed+reverb aesthetic: `speed` changes tempo, `pitch` changes frequency, `reverb` adds spatial ambience. `volume` and `bassboost` are amplitude / tonal controls layered on top.

- **8 new side-quests** joining the existing `download` and `noframes`. All are auto-discovered from `src/voders/quests/` and appear immediately in `python voder.py quest`. All accept the optional `result "<path>"` keyword. All work inside `chains` pipelines as named steps.

  - **`quest convert <format> <input>`** — Universal audio format converter supporting 40+ formats including the weird ones (`opus`, `ogg`, `oga`, `gsm`, `tta`, `wv`, `ape`, `mpc`, `caf`, `dsf`, `dff`, `sph`, `sln`, `raw`, `8svx`, `iklax`, `xi`, `sf`, `sf2`, `ircam`, `pvf`, `fap`, `nist`, `nistsphere`, `sox`, `vox`, `amb`, ...). Same-format conversion just copies the file (no re-encoding, no quality loss). Format argument is case-insensitive and accepts a leading dot. Lossy formats are encoded at high bitrates (256–320 kbps for MP3/MP2, 160 kbps for Opus, etc.); lossless formats preserve full bit depth. All outputs normalized to stereo / 48 kHz.

  - **`quest compress [1|2|3] <input>`** — Compresses an audio file at three levels. Level 1 = low (mild size reduction, retains quality). Level 2 = default (balanced). Level 3 = highest (smallest file, lowest quality). Lossy formats get lower bitrates (256k → 128k → 64k for MP3). Lossless formats get lower bit-depth / sample-rate (24-bit/44.1k → 16-bit/32k → 16-bit/22.05k for WAV). FLAC also raises its compression level (8 → 10 → 12). The input's existing bit-depth and sample-rate are never upgraded — `compress` only reduces. The console output prints before/after size and percent change.

  - **`quest cut <start>-<end> <input>`** — Extracts a time range from a local audio or video file and outputs a WAV (PCM 16-bit, 44.1 kHz, stereo). Time format accepts plain seconds (`20-40`), `mm:ss` (`1:30-2:15`), `hh:mm:ss` (`0:00:00-0:00:05`), and floats (`1.5-3.5`). `start` must be strictly smaller than `end`, both must be non-negative. For video input, only the audio track is extracted (video frames are dropped).

  - **`quest merge <file1> <file2> [<file3> ...]`** — Concatenates two or more local audio files end-to-end into a single WAV. No upper limit on the number of files. Each input is normalized to the same sample-rate / channel layout before being concatenated with FFmpeg's `concat` demuxer for sample-accurate joining. Files of different formats, sample rates, and channel counts can all be merged in the same call.

  - **`quest silence <input> [threshold]`** — Strips silent gaps from a local audio/video file and produces a continuous-speech WAV. Uses FFmpeg's `silenceremove` filter: removes any run longer than 0.25s below the threshold (default -50 dB; user-supplied threshold is a positive integer 10–90 meaning -10 dB to -90 dB). After silence removal, `dynaudnorm` applies gentle dynamic-range normalization. Excellent as a chain step before `svs voice` to make rapid-fire continuous speech from a recording with long pauses.

  - **`quest reverse <input>`** — Reverses a local audio OR video file. Audio input → reversed WAV via `areverse`. Video input → reversed MP4 with both video frames (`reverse` filter) and audio (`areverse` filter) flipped in lockstep, so the reversed video stays in sync with the reversed audio. Recognized video extensions: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.webm`, `.m4v`, `.3gp`, `.wmv`.

  - **`quest fade <input> [seconds]`** — Applies a cinematic fade-in and fade-out (default 5s per side, customizable 0.5–60s). NOT silence-based — the edges rise to ~15% gain (not 0%) using a smooth quarter-sine curve, so the audio is always present and feels like it's *rising* into the mix. A final `volume=1.15` boost gives a slight lift in the body. For files shorter than 2 × fade duration, the fade length is auto-clamped to 25% of file duration per side (min 0.5s). Video input → MP4 with video stream copied, audio replaced with the faded audio.

  - **`quest volume <1-1000> <input>`** — Pure linear volume gain. Scale 1–1000 where every 100 means +100% amplitude (100 = 2×, 200 = 3×, 1000 = 11×). Multiplies every sample by the same factor — affects ALL frequencies equally. The frequency spectrum keeps its shape, just gets taller. No bass boost, no treble lift, no compression, no loudness normalization — the simplest possible gain stage. Use it when you just want the audio louder / quieter without changing its tonal character. For low-frequency enhancement use `quest bassboost` instead. Audio input → 24-bit/48k WAV. Video input → MP4 with video copied, audio re-encoded as AAC 256k.

  - **`quest speed <value> <input>`** — Professional time-stretch (Spotify-style slowed / sped-up versions). Values are 0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, ... up to 10.00 in 0.25 steps (excluding 1.00 which is a no-op). 0.25 = 4× faster (output is 1/4 the input duration), 10.00 = 10× slower (output is 10× the input duration). Uses FFmpeg's `rubberband` filter with `formant=preserved`, `transients=crisp`, `detector=compound`, `phase=laminar`, `pitchq=quality`, `channels=apart` — pitch and formants stay natural-sounding, like the audio was originally performed at the new tempo. Audio files only (refuses video — use `quest cut` or `quest noframes` first). Output: 24-bit/48k WAV for maximum fidelity.

### Documentation

- **docs/COMMAND_CATALOG.md** — Added section 9.11 `bassboost` (full argument table, 6-stage signal-chain description, value-mapping formula, examples, refused inputs). Rewrote section 9.10 `volume` to describe pure linear gain behavior, what it does NOT do (no bass boost, no compression, no loudness normalization), chaining tip with `bassboost`, and a clipping warning at high values. Renumbered subsequent sections 9.11→9.12 (`speed`), 9.12→9.13 (`pitch`), 9.13→9.14 (`glue`), 9.14→9.15 (`reverb`). Updated the "Available quests" table in section 9 to list all 15 quests with their descriptions and output naming patterns. Updated the slowed+reverb chain pattern in the `glue` and `reverb` sections to insert `bassboost` between `volume` and `speed`, with a 6-chain example: `volume 200 → bassboost 70 → speed 2.00 → pitch 0.50 → reverb 85 → glue onto video`.

---

## 06/19/2026
- Status: Stable, all features work, still developing
- **Chains & Side-Quests: User-Defined Pipelines and Utility Tasks (task-layer features, not modes)**

### Added

- **`quest` with no arguments lists available side-quests** — Running `python voder.py quest` with no further arguments now prints the list of registered side-quests (each quest's name and its one-line description) followed by a usage reminder, instead of erroring out with "quest mode requires a quest name". This makes the side-quest system self-documenting — new quests dropped into `src/voders/quests/` are immediately discoverable from the CLI with no extra wiring. The listing is generated dynamically from the live `SIDE_QUESTS` registry, so it always reflects whatever quests are actually loaded. Existing quest invocations (`quest download "..."`, `quest noframes "..."`, etc.) are unchanged.

- **CLI Default to Help** — Running `python voder.py` with no arguments now prints the help message instead of launching the GUI. Use `python voder.py gui` to launch the GUI as before.

- **Side-Quests (`quest` feature)** — New `quest` oneline feature for lightweight utility tasks that live outside the voder engine but are still useful in a voice-processing workflow. This is a **task-layer feature**, not a processing mode — it does not transform audio itself. Each quest is a small class registered in a `SIDE_QUESTS` registry, so future quests can be added without touching the dispatcher — just subclass `SideQuest`, implement `parse()` and `execute()`, and call `_register_side_quest()`.
  - `quest download "<url>"` — download a YouTube/Bilibili/TikTok URL as audio (default) and drop it into `results/`. Output naming: `voder_quest_download_<original-name>_<timestamp>.<ext>`.
  - `quest download video "<url>"` — download the same URL as a full video file (MP4).
  - `quest download "<local-file>"` — copy a local audio/video file to `results/` with the quest naming scheme (no re-encoding).
  - `quest noframes "<local_video>"` — extract the audio track from a LOCAL video file. Strictly refuses URLs and audio-only files. Output is always WAV (PCM 16-bit 44.1 kHz stereo) extracted via FFmpeg. Output naming: `voder_quest_noframes_<original-name>_<timestamp>.wav`.
  - All quests support the optional `result "<path>"` keyword to copy the output to a custom path.
  - Examples: `quest download "https://youtube.com/watch?v=..."`, `quest download video "https://youtube.com/watch?v=..."`, `quest noframes "video.mp4"`.

- **Chains (`chains` feature)** — New `chains` oneline feature that lets the user compose their own pipelines out of voder's existing oneline tasks. This is a **task-layer feature**, not a processing mode — it does not transform audio itself, it composes the main modes (and `train` / `quest`) into user-defined pipelines. Each chain is named, runs a voder oneline command, and its output is captured to a temp directory. Later chains can reference earlier chain names as input paths — voder substitutes the chain name with the captured temp file path before running the later chain. The **last** non-empty chain's output is exported to `results/`; all intermediate outputs live in `temp_chains/` so they don't pollute the results folder.
  - Syntax: `voder.py chains "name1" <voder command...> / "name2" <voder command that references "name1"> / ...`
  - ` / ` (space, slash, space) is the chain separator. The slash must be its own argv element.
  - Each chain starts with a name. Names can be anything — numbers, letters, paths, URLs — whatever the user can keep track of. Shell strips quotes from argv, so `"name1"` and `name1` are equivalent as the first argument of a chain.
  - VODER indexes chain names as each chain runs. Before running a later chain, VODER walks that chain's arguments and replaces any argument that exactly matches a previously-defined chain name with the path to that chain's output file. Non-matching arguments are left untouched. Chain name lookups take precedence over file/URL resolution — if a chain name happens to look like a file path or URL, it still wins.
  - Intermediate chain outputs are moved to `temp_chains/voder_chain_<safe-name>_<timestamp>.<ext>`. The last non-empty chain's output stays in `results/` (or `voices/` for `train` chains).
  - For multi-output commands (e.g., `svs both`, `ss`, TTM with stems), only the latest file produced by the chain is exposed as the chain's output. If you need multiple outputs, run separate chains.
  - Validation rules:
    - **Duplicate chain names** (two non-empty chains with the same name) are an error and stop the pipeline immediately.
    - **Empty chains** (a name with no command following it) are **skipped**. Their names are NOT marked as used, so the same name can be reused later in the same `chains` command. This is by design — it lets the user "lay out" a pipeline skeleton with empty chains first, then fill in real commands.
    - **Trailing empty chains** are ignored, just like middle empty chains.
    - If **all** chains are empty, the pipeline returns an error ("no valid chains to execute").
  - The optional trailing `result "<path>"` keyword copies the **final** chain's output to a custom path.
  - The `train` command works inside chains: its `.tts` / `.ttse` file is the chain's output and is stored in `temp_chains/` for intermediate chains. Side-quests (like `quest download`) also work inside chains.
  - Examples: `chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"`, `chains "audio" quest download "https://youtube.com/watch?v=..." / "text" stt "audio" timestamp`, `chains "skip1" / "skip2" / "real" tts script "hi" voice "male"` (empty chains skipped, names reusable).

- **`ChainPipeline` class** — New orchestrator class for the `chains` feature. Implements `split_segments()` (splits argv on `/`), `parse_chain_segment()` (extracts `(name, command_args)` from each segment), `validate()` (enforces duplicate-name detection and skips empty chains while keeping their names reusable), `substitute_refs()` (replaces chain-name references with indexed temp file paths), and `execute()` (snapshots `results/` and `voices/` before each chain, runs the chain via `parse_and_execute_oneline`, then captures new files; intermediate chain outputs are moved to `temp_chains/`, the last chain's output stays in place).

- **`SideQuest` base class + `DownloadQuest` / `NoFramesQuest` subclasses** — New class hierarchy for side-quests. Each quest subclasses `SideQuest` and implements `parse(args)` (validates arguments and returns `(parsed_dict, error_or_None)`) and `execute(parsed, results_dir, timestamp, result_path=None)` (does the work and returns `True`/`False`). Quests are registered in the global `SIDE_QUESTS` dict via `_register_side_quest(QuestClass)`.

- **`oneline_quest()` and `oneline_chains()` dispatchers** — New dispatcher functions wired into `execute_oneline_command()` for the `quest` and `chains` features.

- **`quest` and `chains` added to valid oneline command dispatch** — `validate_oneline_mode()` and `show_oneline_usage()` updated to include the new feature commands with examples. These dispatch as oneline commands but are explicitly **task-layer features**, not main processing modes (the 8 main modes remain TTS, STS, TTM, STT, SE, SFX, SVS, SS).

### Changed

- **URL Audio/Video Switch for Oneline Modes** — When a YouTube/Bilibili/TikTok URL is provided as input, the audio vs. video download decision is now user-controlled via the `video` keyword across all relevant oneline modes, instead of being silently decided per-mode.
  - **SE** (`se <url>`, `se voice <url>`, `se sr <url>`, …): URL input now downloads **audio** by default. Add the `video` keyword to download the full video and produce MP4 output (audio extracted from the downloaded video, enhanced, and muxed back into the original frames).
  - **SVS standalone** (`svs <url>`, `svs extract <url>`, `svs only <stem> <url>`): URL input now downloads **audio** by default. Add the `video` keyword to download the full video and mux separated stems back into MP4 (one video per stem, same as local video input).
  - **TTS dub** (`tts dub <url>`): URL input now downloads **audio** by default and produces a WAV output. Add the `video` keyword to download the full video and produce MP4 output with the dubbed audio muxed back in. (Dub already accepts local audio files for audio-only output, so this extends that behavior to URL sources.)
  - Modes where video is **logically required** remain video-only and ignore the keyword: `stt subtitle` (burns subtitles onto frames) and `tts dub subtitle` (subtitles always imply a video frame track).
  - Modes that already honor the `video` keyword are unchanged: `ttm complete`, `ttm bgm`, and `ss`.

### Documentation

- **README.md** — Added Side-Quests and Chains (and Voice Training) to the Features list, the "What Can VODER Do?" section, and the Quick Start examples. Split the "Modes at a Glance" table into a "Main Processing Modes (8)" sub-table (TTS, STS, TTM, STT, SE, SFX, SVS, SS) and a separate "Tasks & Features (not modes)" sub-table for `train`, `quest`, and `chains`, so the docs no longer present these utility/pipeline features as if they were processing modes.
- **docs/READ.md** — Replaced the numbered sections "9. Side-Quests (`quest`)" and "10. Chains (`chains`)" with a new umbrella section "Tasks & Features (beyond the 8 modes)" that contains "Voice Training (`train`)", "Side-Quests (`quest`)", and "Chains (`chains`)" as subsections. Updated the intro and Table of Contents so the doc is consistent about VODER having 8 main processing modes plus 3 task-layer features.
- **docs/COMMAND_CATALOG.md** — Split the Mode Index table into "8 main processing modes" and "3 task-layer features (not modes)". Updated the Quick Jump table to mark `1a. Voice Training`, `9. quest`, and `10. chains` as features, not modes. Added a "Tasks & Features (beyond the 8 modes)" section break before section 9, and added clarifying blockquote notes to each feature section header.
- **docs/Guide.md** — Moved Side-Quests and Chains out of the "Processing Modes Deep Dive" section into a new top-level "Task-Layer Features (beyond the 8 modes)" section. Added clarifying notes to the Voice Training, Side-Quests, and Chains section headers explaining that they are features, not processing modes.
- **docs/Bots.md** — Split the Mode Options table into "8 main processing modes" and "3 task-layer features (not modes)" sub-tables. Added clarifying blockquote notes to the Voice Training, Side-Quests, and Chains section headers. Updated the Quick Start comment to refer to `chains` as a feature rather than a mode. Added new Workflow 17 (URL → Audio File via Side-Quest) and Workflow 18 (Multi-Step Pipeline via Chains).
- **docs/voder-skill.md** — Split the Catalog Navigation table into "8 main processing modes" and "3 task-layer features (not modes)" sub-tables. Added clarifying notes to sections 2.1a (Voice Training), 2.9 (Side-Quests), and 2.10 (Chains) marking them as features. Updated the Overview to introduce the three task-layer features, and updated the closing line from "all 10 modes" to "all 8 main processing modes plus the 3 task-layer features".

### Refactor

- **Modular Project Layout** — VODER's monolithic `src/voder.py` + `src/gui.py` pair has been split into a proper `src/voders/` package so future work can land in smaller, focused files instead of one 16k-line blob. No user-facing syntax or behavior changes — every oneline command, the GUI, and the interactive CLI work exactly as before. The 8 main processing modes (TTS, STS, TTM, STT, SE, SFX, SVS, SS) and the 3 task-layer features (`train`, `quest`, `chains`) all keep their existing command surfaces.
  - New package: `src/voders/` with `__init__.py`, `gui.py`, `cli.py`, `sidequests.py`, and a `quests/` subpackage with `__init__.py` plus one file per quest.
  - `src/voders/gui.py` — moved verbatim from `src/gui.py`. The internal `_src_dir` now resolves to the parent `src/` directory (one level up from `voders/`) so the GUI's `subprocess.Popen([python, voder_path, ...])` calls still point at `src/voder.py`. The GUI is launched the same way: `python voder.py gui`.
  - `src/voders/cli.py` — extracted `print_banner()` and `interactive_cli_mode()` out of `voder.py`. The interactive CLI lazy-imports the per-mode `cli_*_mode()` entry points from `voder` at call time (avoids a circular import between `voder.py` and `voders.cli`). Launched the same way: `python voder.py cli`.
  - `src/voders/sidequests.py` — new home for the entire side-quest + chains subsystem: the `SideQuest` base class, the `SIDE_QUESTS` registry, the `_register_side_quest()` helper, the `oneline_quest()` dispatcher, the `ChainPipeline` orchestrator class, and the `oneline_chains()` dispatcher. `ChainPipeline.execute()` lazy-imports `parse_and_execute_oneline` from `voder` at call time (same circular-import avoidance).
  - `src/voders/quests/` — one file per quest. The file's stem (without `.py`) **is** the quest name used by `voder.py quest <name>`. Each quest file defines a single `Quest` subclass of `SideQuest` with its `name` attribute set to match the filename, and self-registers via `_register_side_quest(Quest)` at import time.
    - `src/voders/quests/download.py` → `quest download` (was `DownloadQuest` in voder.py)
    - `src/voders/quests/noframes.py` → `quest noframes` (was `NoFramesQuest` in voder.py)
  - **Quest auto-discovery** — When `voders.sidequests` is first imported, it scans its own `quests/` directory, imports every `*.py` file (skipping `_`-prefixed dunders), finds the `Quest` class in each, and registers it under its `name` attribute. Adding a new quest is now a single file: drop `src/voders/quests/<new-name>.py`, define `class Quest(SideQuest)` with `name = '<new-name>'` and `parse()`/`execute()` methods, call `_register_side_quest(Quest)` at the bottom of the file — no edits needed anywhere else.
  - **`src/voder.py` shrink** — Removed `print_banner()`, `interactive_cli_mode()`, `class SideQuest`, `class DownloadQuest`, `class NoFramesQuest`, `SIDE_QUESTS`, `_register_side_quest()`, `_register_side_quest(DownloadQuest)` / `_register_side_quest(NoFramesQuest)`, `oneline_quest()`, `class ChainPipeline`, and `oneline_chains()` from `voder.py`. The dispatch table in `execute_oneline_command()` still calls `oneline_quest(params)` and `oneline_chains(params)`, but those names are now imported at the top of `voder.py` from `voders.sidequests`. The `__main__` block now lazy-imports `launch` from `voders.gui` (for the `gui` subcommand) and `interactive_cli_mode` from `voders.cli` (for the `cli` subcommand). `parse_and_execute_oneline()` stays in `voder.py` (it's the entry point chains call back into).
  - **Backward-compatible re-exports** — `voder.py` re-exports `SideQuest`, `ChainPipeline`, `oneline_quest`, `oneline_chains`, `SIDE_QUESTS`, and `_register_side_quest` from `voders.sidequests` at module load time, so any external code (or test script) that does `voder.ChainPipeline`, `voder.SIDE_QUESTS`, `voder.oneline_quest`, etc. continues to work without changes. The concrete quest classes (`DownloadQuest`, `NoFramesQuest`) are no longer re-exported — they live at `voders.quests.download.Quest` and `voders.quests.noframes.Quest` and should be imported from there if anything needs them directly.
  - **`src/gui.py` deleted** — Moved to `src/voders/gui.py` (see above). No other files in the project imported `gui` directly; `voder.py`'s `__main__` block was the only caller, and it now uses the new path.

---

## 05/29/2026
- Status: Stable, all features work, still developing
- **Major Integrations and Modernifications**

### Added

#### Fish Audio S2-Pro Integration

- **`extreme` Keyword** — New keyword for TTS mode and its sub-tasks (TTS, SLC, SVC, Modify Speech) that switches the TTS engine from Qwen3-TTS to Fish Audio S2-Pro for higher quality voice cloning and broader language support. Can be used alongside `overdose` (they serve different purposes: overdose = STT/TTM model selection, extreme = TTS model upgrade). Placed after `overdose` in syntax and prompts. Also available for STS mode to pre-process the target voice reference through Fish S2 Pro before Seed-VC conversion, producing a cleaner voice profile that extracts the dominant voice and removes background artifacts/noise.

- **Fish Audio S2-Pro** — New TTS model integrated into VODER. A dual-autoregressive (4B + 400M) model with RVQ-based codec supporting 80+ languages, voice effects via `[tag]` syntax, and superior voice cloning quality.
  - Model: `fishaudio/s2-pro` from HuggingFace, stored at `src/models/checkpoints/fish_s2pro/`
  - Source code: `src/fish_speech/` (stripped from fish-speech repo, inference-only)
  - Auto-downloads on first use
  - S2-Pro well-tested tags: emotions (`[excited]`, `[angry]`, `[sad]`), tones (`[whispering]`, `[soft voice]`, `[low voice]`, `[loud voice]`, `[shouting]`), breathing (`[sigh]`, `[inhale]`, `[exhale]`, `[gasp]`, `[panting]`, `[clears throat]`), vocal sounds (`[laughing]`, `[chuckling]`, `[giggle]`, `[sobbing]`, `[crying]`, `[groan]`), pacing (`[pause]`, `[short pause]`, `[long pause]`), special (`[emphasis]`, `[rustling sound]`), and 15,000+ free-form tags including multi-language
  - 64 S1 Pro tags also work in `[brackets]` (designed for S1 `(parenthesis)` syntax, compatible with S2-Pro): emotions, tone markers, vocal sounds, crowd effects
  - Supports native multi-speaker in one pass using `Name: text` syntax or via `<|speaker:i|>` tokens (noted in Guide but dialogue mode recommended instead)

- **Train Extreme** — `voder.py train extreme voice:name "ref.wav"` trains a voice using Fish S2-Pro and saves it as a `.ttse` file (instead of `.tts`). `.tts` files only work without extreme, `.ttse` files only work with extreme — a clear error message is shown if mismatched.

- **Voice Design with Extreme Mode** — When `extreme` is used with a `voice` prompt (not `target`), VODER always generates ~30s placeholder English text, has VoiceDesign speak it, feeds that audio to Fish for cloning, then Fish speaks the actual text. This applies unconditionally — even for languages VoiceDesign already supports — to ensure consistent voice quality, preserve voice effects tags across all languages, and eliminate the need for language detection. This enables voice design for languages like Arabic, Hindi, Thai, Turkish, and 70+ others that VoiceDesign doesn't natively support, while also improving results for the 10 supported ones.

- **Extreme in SLC** — `tts extreme slc "path.wav"` and `tts overdose extreme slc "path.wav"` use Fish S2-Pro for the resynthesis step. Supports `.ttse` premade voice files.

- **Extreme in SVC** — `tts extreme svc "path.wav" target "ref.wav"` uses Fish S2-Pro for the re-synthesis step. Supports `.ttse` premade voice files.

- **Extreme in Modify Speech** — TTS interactive modify speech now includes an "Enable extreme? (Y/N)" prompt after overdose. When enabled, Fish S2-Pro replaces Qwen3-TTS for voice extraction and synthesis.

- **Extreme in STS** — `sts extreme base "source.wav" target "voice.wav"` pre-processes the target voice reference through Fish S2 Pro before Seed-VC conversion. The compiled target reference is transcribed with VibeVoice ASR, then re-synthesized with Fish S2 Pro to produce a cleaner, more natural voice profile that extracts the dominant voice and removes background artifacts/noise. This gives Seed-VC a cleaner reference input, improving voice conversion quality especially when the reference contains mixed audio or background noise. Works with both Seed-VC v1 (`music` flag) and v2 (standard/mimic). Oneline mode only. If the extreme pass fails, the original target reference is used as fallback.

- **TTM Voice** — `ttm voice` keyword generates a song via ACE-Step then automatically extracts clean vocals via the SVS voice pipe. Output is the isolated vocal track. Supports `target` reference audio and `overdose` quality. Syntax: `voder.py ttm voice lyrics "..." styling "..." 30`

- **TTM Reference Stem Extraction** — Reference and target paths in TTM now support optional stem extraction via ACE-Step XL-Base. Syntax: `stem/(path)` extracts a single stem, `stem-stem/(path)` extracts and mixes multiple stems, `stem/nn-nn(path)` combines stem extraction with time-range cutting. The 12 available stems are: `woodwinds`, `brass`, `fx`, `synth`, `strings`, `percussion`, `keyboard`, `guitar`, `bass`, `drums`, `backing_vocals`, `vocals`. Stem extraction runs after SVS (voice/music) and before time-range cutting. Examples: `reference "drums/(ref.wav)"`, `reference music "bass-drums/30-60(ref.wav)"`, `target voice "keyboard/(ref.wav)"`. Works across all TTM sub-tasks that accept references (remix, repaint, complete, lego, bgm) and the `target` keyword.

#### TranslateGemma 12B Integration

- **TranslateGemma 12B** — New translation model integrated into VODER (`google/translategemma-12b-it`, 12B parameters). Supports true any-to-any translation across 55 languages, replacing Whisper's any-to-English-only translation limitation. Requires 24GB+ VRAM. Stored at `src/models/checkpoints/translategemma/`.

- **`translate (source-target)` Syntax** — New syntax for STT any-to-any translation using TranslateGemma. The bare `translate` flag (without parentheses) still uses Whisper for any-to-English translation (backward compatible). The `translate (source-target)` syntax uses TranslateGemma and supports any target language. `auto` can be used for source language auto-detection. Examples: `translate (auto-ar)` auto-detects source and translates to Arabic; `translate (ja-en)` translates Japanese to English; `translate (auto-en)` auto-detects source and translates to English.

- **STT Subtitle + Translate** — `stt subtitle translate (auto-ar) "video.mp4"` now supports translated subtitles. When `translate (source-target)` is used with `subtitle`, VibeVoice ASR transcribes the speech, TranslateGemma translates the transcript, and the translated subtitles are burned onto the video.

- **STT Overdose + Translate** — `stt overdose translate "(auto-fr)" "audio.wav"` is now supported. Previously, `overdose` and `translate` were mutually exclusive because Whisper could not translate with VibeVoice ASR. TranslateGemma decouples translation from ASR, allowing overdose-quality transcription with any-to-any translation. The bare `translate` (without parentheses) remains incompatible with `overdose` (it uses Whisper's built-in translation which conflicts with VibeVoice ASR).

- **SLC Any-to-Any Translation** — `tts slc translate (auto-ar) "audio.wav"` translates speech to any target language instead of only English. The `translate (source-target)` syntax replaces Whisper's English-only limitation with TranslateGemma's 55-language support. The original SLC behavior (translate to English using Whisper) is preserved when no `translate` syntax is used.

- **TTS Dub Sub-Task** — New `tts dub` sub-task for video/audio dubbing with voice cloning, translation, and speed adjustment. Pipeline: SVS voice isolation → VibeVoice ASR with audio events → speaker detection → TranslateGemma per-segment translation (with timing context) → Fish S2 Pro TTS per segment (voice cloning from source, short segments avoid drift) → per-segment speed adjustment → timeline-based assembly (overlay each segment at its original position) → mix with instrumentals → mux with video. Auto-translates to English by default (no `translate` keyword needed). Supports optional `subtitle` keyword to burn translated subtitles onto the output video. Commands: `tts dub "video.mp4"`, `tts dub subtitle "video.mp4"`, `tts dub translate "(auto-ar)" "video.mp4"`, `tts dub translate "(auto-ar)" subtitle "video.mp4"`, `tts dub "audio.wav"`.

- **Dub Pipeline Architecture** — The dub pipeline chains SVS voice+music separation, VibeVoice ASR with audio events for transcription and non-speech detection, TranslateGemma for any-to-any translation with per-segment timing context, Fish S2 Pro for per-segment TTS with voice cloning from source audio, per-segment audio speed adjustment to match original segment timing, timeline-based assembly using `_overlay_segment_on_base()` for proper temporal positioning, instrumental track mixing for music preservation, and FFmpeg video muxing for video output. VibeVoice ASR, TranslateGemma, and Fish S2 Pro are loaded separately (never simultaneously) to fit within 24GB VRAM. Dub defaults to auto-to-English translation when no `translate` keyword is specified.

- **VibeVoice ASR `transcribe_with_events()`** — New method that preserves audio event tags (`[Silence]`, `[Lyric]`, `[Music]`, `[Noise]`, `[Applause]`, `[Laughter]`, `[Cough]`, `[Breath]`) alongside speech segments. Audio events are tagged with `is_event=True` and `event_type` fields. Used by the dub pipeline to identify non-speech portions that should not be translated or dubbed — these segments are left as silence in the dubbed output. The existing `transcribe()` method continues to filter out audio events for backward compatibility.

- **Audio Timeline Assembly Helpers** — New helper functions for dub pipeline: `_overlay_segment_on_base()` overlays an audio segment at a specific time position using ffmpeg `adelay` + `amix`, and `_extract_audio_segment()` extracts a time range from an audio file.

#### STT Subtitle Sub-Task

- **`subtitle` Keyword** — New STT sub-task keyword that transcribes a video's speech using VibeVoice ASR and burns the subtitles directly onto the video as ASS-format overlays. Implies `overdose` (VibeVoice ASR is always used). Only accepts video files and URLs — audio, text, and image files are rejected.
- **Dynamic Subtitle Positioning** — Subtitles are dynamically scaled and positioned at the bottom of the frame relative to the video resolution. Font size, margins, outline width, and shadow offset are all calculated proportionally to the video height, ensuring consistent appearance from 480p to 4K.
- **Overlap Handling** — When overlapping speech is detected (two speakers talking simultaneously), the primary speaker's text appears on the first line (white), and the overlapping speaker's text appears on a second line beneath it in cyan, making it visually clear that a different speaker is talking.
- **Forced Alignment for Per-Sentence Subtitles** — Subtitles now use Meta's MMS-FA forced alignment model to produce per-word timestamps, grouped into 3-5 word subtitle segments for accurate timing. Previously, subtitles displayed entire ASR segments (often 10+ seconds) at once; now each subtitle line shows a short phrase with precise start/end times. This applies to all subtitle paths: `stt subtitle`, `tts dub subtitle`, and `tts dub subtitle original`. For translated subtitles, the original text is aligned first to get accurate timings, then each chunk is translated while preserving the original timing — producing semi-accurate translated subtitles that sync with the original audio. If forced alignment fails, the system falls back to original ASR segment timings.
- **Full Pipeline** — Download video (if URL) → Extract audio via FFmpeg → SVS voice isolation (BS-RoFormer) → Optional sound enhancement (`se`) → VibeVoice ASR transcription → MMS-FA forced alignment (per-word timestamps grouped into 3-5 word segments) → Burn ASS subtitles onto video via FFmpeg → Output MP4.
- Syntax: `voder.py stt subtitle "video.mp4"`, `voder.py stt subtitle se "noisy_video.mp4"`, `voder.py stt subtitle "https://youtube.com/watch?v=..."`

#### SE Sound Enhancement Modernization

- **SE renamed to Sound Enhancement** — "Speech Enhancement" renamed to "Sound Enhancement" across all code and documentation, reflecting the expanded scope beyond just speech.

- **SE Sub-Modes** — SE mode now supports sub-mode keywords for targeted enhancement:
  - `se "path"` — Default UniSE enhancement (denoise, dereverb, restore speech, 16kHz output)
  - `se voice "path"` — SVS voice extraction → UniSE enhancement on vocals only
  - `se voice blend "path"` — SVS voice+music → UniSE on voice → blend enhanced vocals with music at 48kHz
  - `se sr "path"` — AudioSR super-resolution on whole audio (basic model, 48kHz output)
  - `se sr music "path"` — SVS voice+music → AudioSR (basic model) on music → upsampled music only at 48kHz
  - `se sr music blend "path"` — SVS voice+music → AudioSR on music + UniSE on voice → blend at 48kHz
  - `se sr voice "path"` — SVS voice extraction → AudioSR (speech model) on vocals → upsampled vocals at 48kHz
  - `se sr voice blend "path"` — SVS voice+music → AudioSR speech on vocals → blend with music at 48kHz
  - `se sr voice music "path"` — SVS voice+music → AudioSR speech on vocals + AudioSR basic on music → auto-blend at 48kHz

- **AudioSR Integration** — New audio super-resolution model integrated into VODER. Uses `haoheliu/versatile_audio_super_resolution` (AudioSR) for upscaling low-sample-rate audio to 48kHz. Two model variants: `basic` (general audio/music) and `speech` (speech-optimized). Source code: `src/audiosr/` (stripped from versatile_audio_super_resolution repo, inference-only). Model checkpoint: `src/models/checkpoints/audiosr/`. Auto-downloads on first use. Handles long audio via chunked overlap-add processing.

- **`AudioSREnhancer` Class** — New model wrapper class following VODER's standard load/use/cleanup pattern. Supports both `basic` and `speech` model variants. Auto-selects chunked processing for audio >10.24s.

- **`_mix_audio_at_target_sr()` Helper** — New helper function for blending two audio files at a target sample rate. Resamples both inputs to the target rate, converts to mono, pads to equal length, sums with peak normalization. Used by `blend` keyword in SE sub-modes to preserve upsampled quality (never downsamples the upsampled track).

- **SE Parser Update** — SE oneline parser now accepts sub-mode keywords (`voice`, `sr`, `music`, `blend`) with validation rules: `music` only valid after `sr` or `sr voice`, `voice` after `sr` creates `sr_voice` sub-mode, `blend` valid with `voice`, `sr music`, or `sr voice` sub-modes only. Removed invalid `se sr blend` sub-mode (plain `se sr` uses basic model on full input).

#### SS Mode Enhancements

- **SS `blend` Keyword** — New oneline keyword for SS mode that blends each separated speaker's audio with the original non-vocals (instrumental/background) track extracted via SVS. After speaker extraction (and optional SE), each output is mixed with the music/instrumental stem at 48kHz via `_mix_audio_at_target_sr()`. Outputs carry a `_blend` suffix. Works in both target and auto-separation modes, and with `se` and `overdose`. Useful for vlogs or recordings where you want to isolate a speaker while preserving background audio. Examples: `ss blend "vlog.wav"`, `ss target "ref.wav" blend "conversation.wav"`, `ss overdose se blend "noisy_conversation.wav"`.

- **SS `video` Keyword** — New oneline keyword for SS mode that produces video output. When the input is a video file or URL, each separated speaker's audio is muxed with the original video frames via ffmpeg to produce MP4 output (one video per speaker). Ignored when the input is an audio-only file (prints info message and continues). Works in both target and auto-separation modes, and with `se`, `overdose`, and `blend`. Useful for removing unwanted speakers from a video while keeping the visuals — e.g., extracting only your own speech from a vlog recording. Examples: `ss video "interview.mp4"`, `ss target "ref.wav" video "interview.mp4"`, `ss overdose se blend video "vlog.mp4"`.

### Changed

- Interactive TTS mode now prompts for `extreme` after `overdose` (both in the main flow and in modify speech)
- Oneline TTS parser accepts `extreme` keyword after `overdose`
- `_assemble_enhanced_dialogue()` accepts `use_extreme` and `fish_voice_data` parameters for dialogue-level Fish synthesis
- Voice mismatch check (`.tts` vs `.ttse`) applies across all TTS sub-tasks
- Voice design with extreme mode now always uses the placeholder trick unconditionally (generates English placeholder → Fish clones it → Fish speaks actual text), even for languages VoiceDesign already supports. This fixes voice effects tags being misinterpreted by VoiceDesign and ensures consistent quality across all languages
- Extreme mode dialogue with voice-design characters now correctly applies the placeholder trick per character (was previously broken — VoiceDesign was used directly, causing crashes with unsupported languages and voice effects tags)
- Removed dead `_detect_text_language()` function (was defined but never called)
- `ttm voice` keyword now works in standard TTM mode (was previously restricted to complete/lego tasks only)

---

## 05/21/2026
- Status: Stable, all features work, still developing
- **Major Update & Bug Hunt Activity**

### Added

#### TTS Voice Training

- **`train voice:character-name` Command** — New oneline command to train Qwen-TTS Base voice clones and save them as `.tts` files for reuse.
  - Creates `voices/` directory in project root if not present
  - Saves trained voice prompts as `voder_tts_character-name_timestamp.tts` files
  - Supports multiple reference audios (video/audio/URL) with SVS voice extraction
  - Optional `test` keyword generates a test sample using a hardcoded 30+ second script after training
  - Optional `test "custom script"` uses a user-provided test script
  - Oneline only: `python voder.py train voice:james "ref1.wav" "ref2.wav" test`

- **Trained Voice Usage in TTS** — The `voice` parameter now accepts trained voice references in addition to voice descriptions.
  - `voice "character-name"` uses the latest `.tts` file with that name from `voices/`
  - `voice "character-name:path/to/file.tts"` uses a specific `.tts` file
  - `voice "character-name:another-name"` uses the latest `.tts` for `another-name`
  - When a trained voice is used, Qwen-TTS Base (voice cloning) is used instead of VoiceDesign
  - Works in both oneline and interactive CLI modes

#### TTS Script Newlines

- **`\n` Newline Support** — TTS scripts now support `\n` for actual newlines in both oneline and interactive CLI modes.
  - Oneline: `tts script "James: First line\nSecond line" voice "James: deep male"`
  - Interactive: Enter `\n` in text to create line breaks within a character's speech

#### Voice Stabilization

- **Automatic VoiceDesign Stabilization** — VoiceDesign characters in dialogue mode are automatically stabilized to prevent vocal drift across long scripts.
  - After a VoiceDesign character produces 3 script lines, the outputs are concatenated, SVS-cleaned, and fed to Qwen-TTS Base for voice extraction
  - All subsequent lines for that character use the cloned voice instead of VoiceDesign
  - This eliminates the gradual voice changes that occur when VoiceDesign regenerates voice characteristics for each line
  - Operates transparently — no user configuration needed

#### STS Universal SVS Pre/Post-Processing

- **SVS Voice Isolation on Source** — All STS modes (VCv1 music, VCv2 speech, VCv2 mimic) now isolate source vocals via SVS before feeding them to the VC model, instead of passing the raw source. This gives the VC model clean voice input and produces significantly cleaner conversions.
  - Source → SVS voice → VC model (only vocals processed)
  - Source → SVS music → saved for recombination after conversion
  - VC output + source music → final output (instrumental untouched by VC)

- **`nomusic` Flag** — Added `nomusic` flag for STS to output converted voice only without mixing back the source music.
  - Useful for extracting raw converted vocals for further processing
  - Mutually exclusive with `music` flag
  - CLI example: `python voder.py sts base "song.wav" target "voice.wav" nomusic`

- **VCv2 SVS Pipeline** — VCv2 (standard STS, mimic) now uses the same SVS voice/music isolation as VCv1 (music STS), making the pre/post-processing universal across all STS modes.

#### TTS Multi-Reference Voice Cloning

- **Multi-Reference Cloning** — TTS voice cloning now supports multiple reference audios per character, concatenated into a single composite for richer voice extraction.
  - Oneline format: `target "(path1.wav)(path2.wav)(path3.wav)"` (single mode) or `target "James:(clip1.wav)(clip2.wav)"` (dialogue mode)
  - Each reference is resolved (video/URL supported), cleaned via SVS voice extraction, then concatenated via ffmpeg
  - The composite reference is fed to Qwen-TTS for a single voice extraction pass
  - Interactive CLI: keeps asking for additional references per character until user hits Enter (first reference required)
  - Backward compatible: single path format still works as before

- **STS Multi-Reference Target** — STS oneline mode now supports multiple voice references via `target "(path1)(path2)(path3)"`, same parenthesized format as TTS. Each reference is resolved, SVS-cleaned, then concatenated into a composite for richer voice conversion. Works for all STS sub-modes (standard, music, mimic).

- **TTM VC Multi-Reference Clone** — TTM VC oneline mode now supports multiple voice references via `clone "(path1)(path2)(path3)"`, same parenthesized format as TTS target. Each reference is resolved, SVS-cleaned, then concatenated into a composite for richer voice cloning.

- **`first` Keyword for Multi-Reference** — Oneline mode supports the `first` keyword before multi-reference paths to extract only the first reference's speaker from all other references via TSE before compiling. Works with: `train voice:name first "ref1" "ref2"`, `target first "(path1)(path2)"` (TTS/STS), `clone first "(path1)(path2)"` (TTM VC). Warns and ignores if only one reference is provided.

#### TTM VC SVS Pipeline

- **SVS Isolation for TTM VC** — TTM VC (both standard and overdose) now isolates TTM output vocals via SVS before feeding them to the VC model, and mixes converted vocals back with TTM instrumental after conversion.
  - TTM output → SVS voice → VC model (only vocals converted)
  - TTM output → SVS music → saved for recombination
  - VC output + TTM music → final output

#### TTM Remix Multi-Source with SVS Isolation

- **Multi-Source Support (Up to 3)** — TTM remix now accepts up to three audio/video sources combined into a single composite for style transfer.
  - Paths are provided directly after `remix` keyword (no separate `source` keyword needed)
  - Each source supports voice/music prefix: `voice "path"` extracts vocals, `music "path"` extracts instrumental, bare path uses audio directly
  - Multiple sources are composed into a 30-second composite before style transfer
  - CLI examples:
    - `python voder.py ttm remix "song1.wav" "song2.wav" "song3.wav" styling "jazz fusion"`
    - `python voder.py ttm remix voice "vocals.wav" music "inst.wav" styling "funk"`

#### Multi-Reference Support (Up to 3)

- **Enhanced Reference System** — TTM sub-tasks now accept up to three reference audio files for style guidance.
  - CLI keyword: `reference` — accepts paths with optional voice/music prefix
  - Format: `reference voice "ref_vocals.wav" music "ref_inst.wav"` or plain `reference "ref.wav"`
  - Voice-prefixed references extract vocals via SVS; music-prefixed extract instrumental
  - Multiple references are composed into a 30-second composite for style guidance
  - Works with: remix, repaint, complete, lego sub-tasks

#### TTM Reference Time Spec

- **Optional Time Spec for References** — TTM reference paths now support an optional time specification prefix to select exact audio segments for reference, instead of using the entire audio.
  - Format: `"nn(path)"` — start at nn seconds, extract up to slot-max seconds (30s/15s/10s depending on ref count)
  - Format: `"nn-nn(path)"` — use specified range, auto-slides to reach slot-max if shorter
  - Format: `"nn-nn/nn-nn/nn-nn(path)"` — multiple ranges from same audio, combined to reach slot-max
  - Works with voice/music prefix: `reference voice "50(ref.wav)"`, `reference music "20-30/40-50(ref.wav)"`
  - Works in repaint multi-pass specs: `"20-80/styling(jazz)/reference-voice(30-60(vocals.wav))"`
  - Slot max: 1 reference = 30s, 2 references = 15s each, 3 references = 10s each
  - Sliding logic: if range is shorter than slot-max, start slides back and/or end slides forward until slot-max reached; if audio is shorter than slot-max, segments loop
  - Fully optional — old format still works, auto-compose behavior unchanged when no time spec is provided

#### TTM Remix Optional Lyrics

- **Lyrics Parameter** — TTM remix now accepts an optional `lyrics` parameter to provide new vocal text.
  - Format: `lyrics "verse words here\nmore on next line"`
  - Applies new lyrics to the remixed section while preserving musical style
  - CLI example: `python voder.py ttm overdose remix "song.wav" lyrics "dreamy verse" styling "synthwave"`

#### SFX Overlay for BGM and Complete Tasks

- **Sound Effect Overlays** — TTM BGM and Complete sub-tasks now support embedding sound effects.
  - CLI keyword: `sfx:` — format: `"sfx:prompt/duration-position/level"`
  - SFX specs must be enclosed in quotes for proper CLI parsing
  - Auto-cuts duration if specified SFX length exceeds remaining source time
  - Position values cannot exceed source duration
  - Examples:
    - `python voder.py ttm bgm "audio.wav" music "piano" "sfx:thunder/10-5/50"`
    - `python voder.py ttm complete "source.wav" add "drums" "sfx:rain/8-22"`

#### TTM Repaint Multi-Pass Mode

- **Sequential Multi-Pass Editing** — TTM repaint now supports multiple passes where each edit builds on the previous output.
  - Each pass specifies: time range, styling, lyrics, references, and bias independently
  - Each pass uses the previous pass's result as its input source
  - Supports up to 3 references per pass with voice/music prefix parsing
  - Format: `start-end/styling(text)/lyrics(text)/reference-voice(path)/reference-music(path)/reference(path)/bias/0-100`
  - CLI example: `python voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"`
  - Example with references: `python voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(dnb)/bias/80/reference-voice(vocals.wav)"`
  - Intermediate files cleaned up automatically after all passes complete

- **Voice/Music Prefix for Repaint Source** — Repaint accepts optional prefix to control SVS processing of the source.
  - `repaint voice "song.wav"` — extracts vocals before repainting
  - `repaint music "song.wav"` — extracts instrumental before repainting
  - `repaint "song.wav"` — uses source directly (existing behavior)

#### SS Mode Overlap-Aware Pipeline

- **Overlap-Aware SS for Overdose** — SS mode with overdose uses VibeVoice ASR with special overlap preservation.
  - Added `transcribe_with_overlaps()` method that instructs VibeVoice to preserve overlapping speaker timestamps
  - Enrollment selection uses overlap-aware scoring: picks segments with least overlap first, then longest duration
  - If best segment has overlap, finds the largest non-overlapped gap within it for clean enrollment
  - Iterative refinement loop re-runs VibeVoice on extracted audio until single speaker confirmed

- **Overlap-Aware SS for Non-Overdose** — Non-overdose SS path uses pyannote with exclusive diarization, skips Whisper.
  - Added `diarize_full()` method to SpeakerDiarization class for exclusive diarization access
  - Uses pyannote's exclusive speaker diarization for clean enrollment without ASR
  - Iterative refinement loop re-runs pyannote on extracted speaker until single speaker confirmed
  - If multiple speakers detected after extraction, cuts to longest exclusive segment as enrollment

- **SS Pipe for TTS Dialogue Voice Extraction** — Interactive TTS dialogue mode now uses `ss_extract_speakers()` pipe (same pattern as `svs_extract_vocals()`) to automatically extract per-speaker audio clips from imported dialogue audio. Speaker numbers from STT transcription map directly to SS outputs, eliminating manual reference entry and improving voice cloning accuracy for both TTS and TTS-overdose modes.

#### TTM Complete Task Enhancements

- **Blend Source Control (usrc)** — Complete task accepts `usrc` keyword to control which source elements blend into generated tracks.
  - CLI format: `ttm complete "source.wav" usrc voice music styling "orchestral"`

- **noblend Flag** — Complete task accepts `noblend` flag to generate tracks without blending with source.
  - Useful for completely new instrument/vocal tracks independent of source
  - CLI format: `ttm complete "source.wav" only "drums" noblend`

- **5Hz LM CoT Pipeline** — Complete task uses 5Hz language model Chain-of-Thought for enhanced generation quality.

- **Styling Prompt** — Complete and lego sub-tasks now accept `styling` parameter for style guidance.

#### TTM BGM Video Support

- **video Flag and Reference for BGM** — BGM subtask supports video input and video URLs as source.
  - Video files are processed: audio extracted, music stripped, new music generated, then remuxed
  - Reference audio for BGM now accepts video files (audio extracted before use)
  - Output format matches input format (video in, video out)

#### TTS SLC Sub-Task (Speaker Language Conversion)

- **`tts slc` Oneline Command** — SLC (Speaker Language Conversion) is now a TTS oneline sub-task instead of a standalone mode. Always translates to English using Whisper large-v3 (not turbo) — no separate `translate` keyword needed.
  - `tts slc "path.wav"` — transcribe, translate to English, resynthesize with original voice
  - `tts slc music "path.wav"` — same as above, but also extracts instrumental via SVS music and blends it with the voice output for music preservation
  - `tts overdose slc "path.wav"` — after TTS output, runs STS v2 non-mimic pass with source vocals for enhanced voice preservation
  - `tts overdose slc music "path.wav"` — overdose + music preservation combined
  - Supports video files, YouTube URLs, and audio files (modernized from standalone SLC which was audio-only)
  - SVS voice isolation on source before transcription for cleaner results
  - Uses Whisper large-v3 exclusively (not turbo) for both transcription and translation — single model load, no redundant second instance
  - `music` flag: SVS music extraction from source + ffmpeg blend with TTS voice output; note voice-music sync may vary
  - `translate` keyword removed — translation to English is now hardcoded (Whisper can only translate to English, not between arbitrary language pairs)

#### TTS SVC Sub-Task (Speaker Voice Conversion)

- **`tts svc` Oneline Command** — SVC (Speaker Voice Conversion) is a new TTS oneline sub-task that transcribes single-speaker audio and re-synthesizes it with a different voice.
  - `tts svc "input_path" target "voice_ref"` — Transcribe single-speaker audio and re-synthesize with a different voice. Pipeline: SVS voice isolation → Whisper transcription (or VibeVoice ASR with `overdose`) → Qwen-TTS synthesis with target voice
  - Supports `overdose` flag: `tts overdose svc "path" target "ref"` uses VibeVoice ASR for transcription
  - Target can be an audio path, trained voice name, or text description (VoiceDesign)
  - Multi-reference targets supported: `target "(ref1.wav)(ref2.wav)(ref3.wav)"` concatenates multiple references for richer voice extraction
  - Output naming: `voder_tts_svc_*.wav`, `voder_tts_svc_sts_*.wav`

#### STS Voice Pass (sts: Prefix)

- **`target "sts:voice_ref.wav"`** — Prefix `sts:` on any target reference triggers an additional Seed-VC v2 non-mimic voice conversion pass after the standard Qwen-TTS cloning.
  - Works in single TTS mode: `tts script "text" target "sts:voice.wav"`
  - Works in dialogue mode: `target "Character: sts:voice.wav"` — each line for that character gets the STS pass applied individually before mixing
  - Works in SVC sub-task: `tts svc "input.wav" target "sts:ref.wav"`
  - Works in interactive modify speech: prefix the custom voice reference with `sts:` to apply the pass
  - Multi-reference format supported: `target "sts:(ref1)(ref2)(ref3)"`
  - The STS pass takes the TTS output as the source (speech content to preserve) and the `sts:` reference as the target voice, producing higher voice fidelity
  - Output naming: `voder_tts_sts_*.wav` (single), `voder_tts_svc_sts_*.wav` (SVC), per-line in dialogue (temporary)

#### TTS Interactive Speech Modification (STT+TTS Integration)

- **"Modify Speech?" Prompt** — STT+TTS functionality is now integrated into TTS interactive mode as the first prompt, instead of being a standalone mode.
  - When entering TTS interactive mode, user is asked "Want to modify speech? (Y/N)"
  - If yes: provide audio/video/URL → SVS voice isolation → Whisper transcription → edit text → choose voice (source audio or custom path with optional `sts:` prefix and multi-reference support)
  - After voice selection: "Preserve non-vocals? (Y/N)" — if yes, extracts instrumental via SVS music and blends with voice output
  - If `sts:` prefix used on voice reference: additional Seed-VC v2 non-mimic pass after Qwen-TTS synthesis for enhanced voice fidelity
  - Modernized: supports video files and YouTube URLs (old STT+TTS mode only supported audio)
  - SVS voice isolation before transcription for cleaner results

### Fixed

- **TSE Enrollment Cap (5s) + Peak Normalize** — Fixed enrollment tensor dimensions.
  - TSE enrollment capped at 5 seconds maximum
  - Peak normalization applied to match training configuration
  - Prevents crashes when pad length exceeds source tensor dimensions

- **Circular Pad Crash Fix** — Fixed crash in `tse_extract` when enrollment tensor is 1D.
  - Falls back to repeat enrollment to 5s instead of circular padding
  - Fixes crashes when pad > source tensor dimensions

- **STT Overdose Mode Fixes** — Multiple fixes for overdose path:
  - Skip pyannote and Whisper when in overdose mode (uses VibeVoice only)
  - Fixed compress ratio for overdosed audio processing
  - Fixed dtype issues in the processing pipeline

- **Overdose Per-Segment Timestamps** — Fixed timestamp generation for single-speaker overdose.
  - Strips Lyric/Silence tags from VibeVoice output before processing
  - Produces clean timestamped segments for single speaker confirmation

- **SE Pipeline Reorder** — Speech enhancement now applied post-extraction instead of pre-extraction.
  - Previously: SE applied to combined source before TSE
  - Now: SE applied to each separated speaker file individually after TSE extraction
  - Pipeline files kept in temp until SE completes, then exported to results

- **Case-insensitive character names** — `.tts` voice file naming and lookup now normalize to lowercase, so `JAMes`, `jamES`, and `james` all resolve to the same voice file.

### Changed

- **Interactive CLI Menu Renumbered** — Menu reduced from 10 options to 8 (removed STT+TTS and SLC as standalone modes).

### Removed

- **SLC Mode Removed** — Standalone `slc` mode removed from oneline and interactive CLI. Now available as `tts slc` sub-task with modernized features (video/URL support, SVS isolation, overdose pass).

- **STT+TTS Mode Removed** — Standalone STT+TTS mode removed from interactive CLI menu. Now integrated into TTS interactive mode as "modify speech?" prompt with modernized features (video/URL support, SVS isolation).

### Technical Notes

- Code size increased with multi-pass repaint parsing, sequential edit execution, and SFX overlay support
- New function: `_parse_sfx_specs()` for parsing SFX spec strings with auto-cut overflow handling
- New function: `_parse_repaint_pass_spec()` for multi-pass repaint specification parsing
- New function: `_resolve_audio_entry()` for voice/music prefixed audio processing
- New function: `_compose_refs()` for multi-reference composition into 30s composite
- New function: `_compose_sources()` for multi-source composition into single audio
- New function: `_parse_multi_refs()` for parsing TTS multi-reference target format (`(path1)(path2)(path3)`)
- New function: `_concat_audio_files()` for concatenating multiple audio files via ffmpeg
- New function: `_resolve_multi_refs()` for resolving, SVS-cleaning, and concatenating multiple voice references
- New function: `_save_voice_prompt()` for saving trained voice prompts as `.tts` files
- New function: `_load_voice_prompt()` for loading trained voice prompts from `.tts` files
- New function: `_parse_ref_time_spec()` for parsing TTM reference time spec format (`nn(path)`, `nn-nn(path)`, `nn-nn/nn-nn(path)`)
- New function: `_extract_ref_segments()` for extracting and sliding time segments from reference audio
- New function: `_find_voice_file()` for finding latest trained voice file by character name
- New function: `_resolve_voice_ref()` for resolving trained voice references (name, name:path, name:othername)
- New function: `_is_trained_voice_ref()` for checking if a voice value is a trained voice reference
- New function: `_ensure_voices_dir()` for creating `voices/` directory
- New function: `oneline_train()` for `train voice:name` command execution
- `TRAIN_TEST_SCRIPT` constant added for default test script in train mode
- `_assemble_enhanced_dialogue()` updated with automatic VoiceDesign voice stabilization after 3 lines
- `train` added to valid oneline modes
- `\n` replacement added to dialogue script parsing in both oneline and interactive CLI modes
- Added `random` import for composite reference/source composition randomization
- `remix_entries` parameter renamed to `source_entries` for consistency
- VibeVoice ASR `transcribe_with_overlaps()` method added for overlap-aware transcription
- SpeakerDiarization `diarize_full()` method added for exclusive diarization access
- New function: `ss_extract_speakers()` for SS pipe extraction of per-speaker audio clips (reusable pipe, same pattern as `svs_extract_vocals()`/`svs_extract_music()`)
- `oneline_sts()` target parameter now supports multi-reference `(path1)(path2)(path3)` format via `_parse_multi_refs()`/`_resolve_multi_refs()`
- `oneline_ttm()` clone parameter now supports multi-reference `(path1)(path2)(path3)` format via `_parse_multi_refs()`/`_resolve_multi_refs()`
- Removed dead function `extract_voice_clips_from_multispeaker()` — replaced by `ss_extract_speakers()` SS pipe (longest-segment cut logic superseded by TSE extraction)
- Standalone SLC mode refactored into `tts slc` oneline sub-task — SLC oneline handler and interactive SLC menu option removed; logic relocated to TTS oneline parser as `slc` sub-task keyword
- Standalone STT+TTS interactive mode refactored into TTS interactive "modify speech?" prompt — STT+TTS menu option removed; speech modification flow now gated behind first prompt in TTS interactive mode
- Interactive CLI menu renumbered from 10 options to 8 (STT+TTS and SLC removed as top-level modes)
- `slc` removed from valid oneline modes list; `stt+tts` removed from interactive mode map
- TTS oneline parser now recognizes `slc` as a sub-task keyword (similar to existing TTM sub-task pattern)
- TTS interactive mode entry point now includes speech modification gate before normal TTS flow
- SVC uses the standard Whisper model (turbo) for transcription by default, VibeVoice ASR when overdose is enabled; the `overdose` flag in SVC only controls which STT engine is used (it does not trigger a Seed-VC v2 pass)
- The `sts:` prefix on target references triggers Seed-VC v2 non-mimic conversion — this is the explicit opt-in mechanism for enhanced voice fidelity, replacing the previous automatic overdose VC pass in SVC and modify speech
- In dialogue mode, STS passes are applied per-line before the final mix, ensuring each character's voice is individually converted
- SVC target parameter now supports multi-reference format `(path1)(path2)(path3)` via `_parse_multi_refs()`/`_resolve_multi_refs()` for richer voice extraction
- Modify speech interactive mode now supports `sts:` prefix and multi-reference format on the voice reference input, replacing the old automatic overdose VC pass
- Fixed STS pass parameter order in `SeedVCV2.convert()` calls — source (speech content) is now correctly passed as the first argument and reference (voice to mimic) as the second in all three locations: SVC STS pass, TTS oneline STS pass, and dialogue STS pass
- Modify Speech interactive output naming now uses `voder_tts_ms_` prefix (was `voder_tts_`), matching the sub-task naming convention used by SLC and SVC

---

## 04/28/2026
- Status: Stable, all features work, still developing
- **Enhancement — Dialogue Background Music Reference Support & TTM BGM Subtask**

### Added

#### Dialogue Background Music Reference Support

- **Reference Audio for Dialogue Background Music** — Dialogue background music generation now accepts an optional `reference` parameter that provides stylistic guidance to ACE-Step during music generation.
  - When `reference "path"` is provided alongside the `music` parameter, the reference audio is first processed through the SVS music pipe (BS-RoFormer) to extract clean instrumental music
  - This ensures that only the musical content from the reference is used for style guidance, not any vocals or noise
  - Works in both one-liner CLI and GUI modes
  - CLI syntax: `python src/voder.py tts script "..." voice "..." music "description" reference "path/to/ref.wav"`
  - The reference is passed as a secondary style input to ACE-Step, improving stylistic consistency when the user wants the generated music to match a specific existing track

#### TTM BGM Subtask

- **TTM BGM — Replace Background Music** — A new TTM subtask that takes an existing audio/video source, strips the current music, generates new background music, and mixes it at a configurable volume level.
  - CLI keyword: `bgm` — routes to the new background music replacement pipeline
  - Command format: `ttm [overdose] bgm "source_path" music "description" level 0-100 [reference "path"] [result "path"]`
  - Source can be a local audio file, video file, or a direct URL (YouTube, Bilibili, TikTok)
  - The pipeline first uses SVS voice pipe (BS-RoFormer) to strip existing music from the source, isolating clean speech/vocals
  - Duration is automatically detected from the stripped audio
  - New background music is generated via ACE-Step using the provided music description, split into 250-300s chunks if the duration exceeds the model limit, and concatenated
  - Optional `reference` audio is processed through SVS music pipe to extract clean instrumental before being passed to ACE-Step for style guidance
  - The new music is mixed with the clean vocals at the specified `level` (0-100, default 35)
  - If the source was a video, the final audio is re-muxed back into the video container
  - Output naming: `voder_ttm_bgm_{original-name}_{timestamp}.wav` for audio, `.mp4` for video
  - Normal (non-overdose) uses ACE-Step turbo 1.5 model for standard quality
  - Overdose uses ACE-Step XL 1.5 turbo model for enhanced quality
  - GUI: Added "BGM" as a new sub-mode in the TTM tab with fields for source file, music description, volume level, and optional reference file

### Changed

- **TTS Music Keyword** — `music` keyword added to the one-liner parser's `valid_keywords` list, enabling it alongside other TTS dialogue parameters
- **TTM Sub-Tasks Expanded** — TTM now supports six sub-tasks: `complete`, `lego`, `extract`, `remix`, `repaint`, and the new `bgm`

---

## 04/18/2026
- Status: Stable, all features work, still developing
- **Major Update — Three New Modes, Mode Mergers, Speaker Diarization, Vocal Extraction, and TTM Sub-Tasks**

### Added

#### Mode Mergers (Simplification)

- **TTS+VC merged into TTS** — TTS mode now supports voice cloning directly via the `target` parameter. Previously a separate mode, TTS+VC is now integrated into TTS. When a `target` audio file is provided alongside `script` and `voice`, the TTS pipeline applies voice cloning. The old `tts+vc` command is no longer accepted — use `tts` with `target` instead.

- **TTM+VC merged into TTM** — TTM mode now supports voice conversion directly via the `vc` flag and `clone` parameter. Previously a separate mode, TTM+VC is now integrated into TTM. When `vc` is enabled and a `clone` audio is provided, the pipeline chains ACE-Step generation with Seed-VC voice conversion. The old `ttm+vc` command is no longer accepted — use `ttm vc` with `clone` instead.

- **VODER now has 10 processing modes** (down from 9 listed, but TTS+VC and TTM+VC are absorbed into TTS and TTM respectively, while 3 new modes are added: SVS, SLC, SS)

#### New Processing Modes

- **SVS (Song Voice Separate)** — A new standalone mode for vocal/music separation using BS-RoFormer.
  - Uses BS-RoFormer Resurrection model from `pcunwa/BS-Roformer-Resurrection` on HuggingFace
  - Separates vocals from music (voice isolation) and music from vocals (instrumental extraction)
  - Two separation stems: `voice` (extract clean vocals) and `music` (extract instrumental)
  - Supports audio and video input (video audio auto-extracted)
  - Supports YouTube URL input — downloads and separates automatically
  - Model checkpoint directory: `src/models/checkpoints/svs/`
  - Used internally by STS mode for automatic vocal extraction from target reference audio
  - Used internally by TTS mode for vocal extraction from voice cloning targets
  - Used internally by STT mode for pre-cleanup vocal isolation (SVS Stage 1) before transcription

- **SLC (Speaker Language Conversion)** — A new mode that translates speech from one language to another while preserving the speaker's voice identity.
  - Translates audio content from any of Whisper's 99 supported languages to English (or other TTS-supported languages)
  - Preserves the original speaker's vocal characteristics, tone, and delivery style
  - Uses Whisper for transcription/translation and Qwen3-TTS for resynthesis
  - When no target parameter is provided, uses the original input as voice reference — effectively translating speech to English with the same original voice
  - When a target reference is provided, can change speaker voice while translating language
  - For preserving original language (if it's one of the 10 TTS-supported languages), SLC with a different target reference can change the speaker's voice — sometimes matching or surpassing STS mode quality
  - Supports audio files and YouTube URLs as input

- **SS (Speakers Separator)** — A new mode for extracting individual speaker audio from multi-speaker recordings.
  - Uses VibeVoice ASR (microsoft/VibeVoice-ASR) for speaker identification and segmentation
  - Automatically identifies individual speakers and produces separate audio files for each
  - Produces a mapped transcript with speaker labels and timestamps
  - Supports audio and video input
  - Requires 24GB+ VRAM or 48GB+ combined system memory (RAM+Swap/Pagefile)
  - Falls back gracefully to Whisper + pyannote if VibeVoice ASR cannot load

#### New AI Models

- **VibeVoice ASR** — Microsoft's state-of-the-art automatic speech recognition model for speaker diarization and transcription.
  - Model: `microsoft/VibeVoice-ASR` with `Qwen/Qwen2.5-7B` language model backbone
  - Uses Qwen2.5-7B as language model for transcription quality
  - Supports SDPA attention implementation for efficient inference
  - Processes audio at 24kHz sample rate
  - Provides native speaker diarization with speaker IDs and timestamps
  - Offers `transcribe()` method for timestamped speaker-labeled output
  - Offers `transcribe_plain_text()` for clean text without timestamps/speakers
  - Requires 24GB+ VRAM for GPU mode or 48GB+ system memory for CPU mode
  - Repository: https://github.com/microsoft/VibeVoice
  - Model directory: `src/models/checkpoints/vibevoice_asr/`
  - Source code directory: `src/asr/` (bundled locally)

- **BS-RoFormer Resurrection** — Advanced source separation model for vocal/music isolation.
  - Model: `pcunwa/BS-Roformer-Resurrection` on HuggingFace (no GitHub repo available)
  - Two separation stems: `voice` (vocal isolation) and `music` (instrumental extraction)
  - Used by SVS mode for standalone vocal/music separation
  - Used internally by STS for automatic vocal extraction from target reference audio (improves voice conversion quality by removing background music/instruments)
  - Used internally by TTS dialogue mode for vocal extraction from voice cloning targets
  - Used internally by STT mode for pre-cleanup vocal isolation before transcription
  - Source code directory: `src/bs_roformer/` (bundled locally)
  - Model directory: `src/models/checkpoints/svs/`

- **ACE-Step XL-Turbo (Overdose)** — Higher-quality music generation model for TTM mode.
  - Model config: `acestep-v15-xl-turbo` with `acestep-5Hz-lm-4B` language model
  - Provides `shift=3.0` for enhanced generation quality over standard turbo
  - Available via `overdose` flag in TTM mode
  - Requires 32GB+ VRAM or 48GB+ system memory
  - Falls back to standard ACE-Step turbo if resources are insufficient

- **ACE-Step XL-Base (Complete)** — High-quality music generation model for TTM advanced tasks.
  - Model config: `acestep-v15-xl-base` with `acestep-5Hz-lm-1.7B` language model
  - Used for TTM sub-tasks: complete, extract, lego (requires 50 inference steps)
  - Requires 32GB+ VRAM or 48GB+ system memory
  - Cannot proceed if resources are insufficient (hard requirement, no fallback)

#### TTM Mode Sub-Tasks

- **TTM Sub-Tasks** — TTM mode now supports five advanced music processing sub-tasks beyond basic generation:
  - `complete` — Completes an existing audio track by generating specified missing instrument/vocal tracks. Uses ACE-Step XL-Base (50 inference steps). Accepts source audio, track classes, optional styling, duration, and reference audio.
  - `lego` — Generates a specific instrument or vocal track based on the context of existing audio. Uses ACE-Step XL-Base. Supports styling and reference audio parameters.
  - `extract` — Extracts a specific instrument or vocal track from mixed audio. Uses ACE-Step XL-Base. Supports specifying track name and duration.
  - `remix` — Applies style transfer to existing audio (music cover). CLI keyword is `remix` (internal method: `cover`). Uses overdose ACE-Step model when `overdose` flag is set, otherwise standard ACE-Step turbo (8 inference steps). Accepts `bias` parameter (0-100, default 40).
  - `repaint` — Repaints a specific time range of existing audio. Uses overdose ACE-Step model when `overdose` flag is set, otherwise standard ACE-Step turbo. Accepts `time:start-end` time range, optional lyrics and `bias` (0-100, default 40).
  - **12 ACE-Step Instrument Tracks**: woodwinds, brass, fx, synth, strings, percussion, keyboard, guitar, bass, drums, backing_vocals, vocals
  - **Track Groups**: `instruments` (10 instrument tracks), `voices` (vocals + backing_vocals), `everything` (all 12 tracks)
  - **Track Resolution System**: `resolve_acestep_tracks()` function handles group expansion, deduplication, and validation
  - **Reference Audio Parsing**: `parse_ref_raw()` function handles `track_name:path` format for per-track reference audio

#### Enhanced STT Mode

- **Translation in STT** — STT mode now supports translation of audio to English using Whisper large-v3.
  - New `translate` option in CLI: "Translate to English? (Y/N)"
  - Uses Whisper large-v3 (not turbo) for the translate task — turbo model does not support translation
  - Downloads and caches `whisper-large-v3.pt` separately from the turbo model
  - Translation works with or without diarization enabled
  - When both translation and diarization are enabled, produces English translated text with speaker labels aligned using overlap matching
  - Output follows the same format as regular transcription but in English

- **Overdose Mode in STT** — STT mode now offers an optional "overdose" quality tier using VibeVoice ASR.
  - Available when translation is NOT enabled (overdose and translate are mutually exclusive)
  - New `overdose` option in CLI: "Enable overdose? (Y/N)"
  - Uses VibeVoice ASR instead of Whisper for transcription
  - Provides native speaker diarization with speaker IDs (no separate pyannote needed)
  - Falls back to Whisper if VibeVoice ASR fails to load (insufficient resources)
  - Requires 24GB+ VRAM or 48GB+ system memory

- **Pre-Cleanup SE in STT (SVS Stage 1)** — STT mode now automatically runs vocal isolation before transcription.
  - Uses BS-RoFormer to extract clean vocals from the input audio
  - Removes background music, instruments, and noise before Whisper processes the audio
  - Applied automatically to all STT operations (transcription, translation, overdose)
  - If SVS isolation fails, proceeds with original audio with a warning
  - Temporary files are cleaned up after processing

#### Enhanced STS Mode

- **Video I/O for STS** — STS mode now supports video input and video output.
  - Video input: Load a video file (MP4, MKV, etc.) as base input — audio is auto-extracted
  - Video output: When base input is a video, output is rendered as MP4 with the converted audio replacing the original audio track
  - Uses FFmpeg for audio-video merging
  - Output naming: video inputs produce `.mp4` output; audio inputs produce `.wav` output
  - Error handling: Falls back to audio-only output if video merge fails

- **Automatic Vocal Extraction for STS** — STS mode now automatically extracts clean vocals from target reference audio.
  - When a target reference is provided, BS-RoFormer is used to extract vocals before voice conversion
  - Removes background music, instruments, and noise from the target
  - If vocal extraction fails, uses original target with a warning
  - Temporary files are cleaned up after processing
  - This improves voice conversion quality significantly for references with background content

#### Enhanced TTS Mode

- **Language Parameter** — TTS mode now exposes a `language` parameter that maps to `SUPPORTED_TTS_LANGUAGES`.
  - Supports 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
  - New `SUPPORTED_TTS_LANGUAGES` constant dictionary with ISO code to full name mapping
  - Default remains "Auto" for auto-detection

- **Auto Vocal Extraction for Voice Cloning** — TTS mode now automatically extracts clean vocals from voice cloning targets.
  - When using `target` parameter for voice cloning, BS-RoFormer extracts vocals from the reference
  - Produces cleaner voice embeddings for more accurate cloning
  - Applied in both single and dialogue modes

- **YouTube URL Support for Voice Prompts** — TTS voice prompts now accept YouTube URLs.
  - Enter a YouTube URL instead of a voice description to clone a voice from that URL
  - Audio is downloaded, vocals are extracted via SVS, then used for voice cloning
  - Cleanup is handled automatically

#### Enhanced Dialogue System

- **YouTube URL Support for Character Voice Assignment** — Dialogue character voice assignments now accept YouTube URLs.
  - In dialogue mode, provide a YouTube URL as a character's voice reference
  - VODER downloads the audio, extracts vocals via SVS, and uses it for voice cloning
  - Multiple characters can each have different YouTube URL references

### Changed

- **10 Processing Modes** — VODER now has 10 distinct processing modes.
  - Modes: STT+TTS, TTS, STS, TTM, STT, SE, SFX, SVS, SLC, SS
  - TTS+VC is no longer a separate mode — use TTS with `target` parameter for voice cloning
  - TTM+VC is no longer a separate mode — use TTM with `vc` flag for voice conversion
  - The old `tts+vc` and `ttm+vc` commands are no longer accepted and will produce an error

- **ACE-Step Three-Tier System** — ACE-Step now operates in three quality tiers:
  - **Standard**: `acestep-v15-turbo` with `acestep-5Hz-lm-1.7B`, shift=1.0, 8 inference steps
  - **Overdose**: `acestep-v15-xl-turbo` with `acestep-5Hz-lm-4B`, shift=3.0, 8 inference steps
  - **Complete**: `acestep-v15-xl-base` with `acestep-5Hz-lm-1.7B`, shift=1.0, 50 inference steps (for sub-tasks)

- **Enhanced WhisperSTT** — Whisper model now uses dual-model architecture.
  - STT transcription: `large-v3-turbo` (fast, efficient)
  - Translation: `large-v3` (supports translate task, which turbo does not)
  - Custom checkpoint save/load system with `_save_checkpoint()` and `_load_model()` methods
  - Both models cached under `src/models/checkpoints/whisper/`

- **Enhanced SeedVCV1** — Seed-VC v1 now supports automatic vocal extraction.
  - Before voice conversion, BS-RoFormer extracts clean vocals from target reference
  - This removes background music and instruments that could interfere with voice conversion quality
  - Falls back gracefully if SVS extraction fails

- **Enhanced QwenTTS** — Qwen3-TTS now exposes language parameter.
  - `synthesize()` method accepts `language` parameter (default: "Auto")
  - Maps to `SUPPORTED_TTS_LANGUAGES` for validation
  - Allows explicit language control instead of relying solely on auto-detection

- **Centralized Model Storage Updates**:
  - `src/models/checkpoints/svs/` — BS-RoFormer SVS models
  - `src/models/checkpoints/vibevoice_asr/` — VibeVoice ASR model

- **System Resource Detection** — New `get_system_resources()` function for dynamic hardware assessment.
  - Detects single GPU VRAM, total GPU VRAM, system RAM, and swap/pagefile
  - Uses `psutil` where available, falls back to `/proc/meminfo` on Linux
  - Used by VibeVoice ASR, ACE-Step Overdose, and ACE-Step Complete for resource checks
  - Returns `(single_gpu_gb, total_sys_gb)` tuple

- **YouTube URL Expansion** — New `resolve_target_to_audio()` and `download_youtube_video()` functions.
  - `resolve_target_to_audio()`: Resolves any path/URL to a downloadable audio file (supports YouTube URLs, video files, audio files)
  - `download_youtube_video()`: Downloads full video from YouTube URL (used by SVS and TTM modes)
  - Automatic cleanup of temporary downloaded files

- **Updated Dependencies**:
  - New: `rotary_embedding_torch==0.3.5` — Required for BS-RoFormer model
  - New: `beartype==0.14.1` — Required for BS-RoFormer model
  - New: `ml_collections` — Required for BS-RoFormer model
  - Changed: `huggingface-hub>=0.16.0` → `huggingface-hub==0.34.0` (pinned version)

### Technical Notes

- Code size increased from ~7,263 lines (stable voder.py) to ~10,916 lines (bleed voder.py) — approximately 50% increase (+3,653 lines)
- New source directories: `src/asr/` (VibeVoice ASR, 8 files), `src/bs_roformer/` (BS-RoFormer, 10+ files)
- TTS+VC and TTM+VC modes have been fully absorbed into TTS and TTM; the old `tts+vc` and `ttm+vc` commands are no longer accepted
- The 10 modes are: STT+TTS, TTS, STS, TTM, STT, SE, SFX, SVS, SLC, SS (TTS+VC absorbed into TTS, TTM+VC absorbed into TTM)
- All new models (VibeVoice ASR, BS-RoFormer) support explicit cleanup with `cleanup()` methods

---

## 04/09/2026
- Status: Stable, all features work, still developing
- **Bug Hunt Activity** — Extensive bug fixes, memory optimizations, and new features

### Added

#### OCR Input Support for TTS Modes

- **OCR Parameter for TTS Mode** — New `ocr` parameter for one-liner TTS commands to extract text from images.
  - Use `ocr "path/to/image.png"` to provide an image file instead of manual text input
  - VODER uses EasyOCR to extract text from the image, then synthesizes the extracted text as speech
  - Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP
  - File validation ensures only image formats are accepted
  - Example: `python src/voder.py tts ocr "script_screenshot.png" voice "text: professional male narrator"`
  - Resources are properly cleaned up after OCR extraction (model offload, gc.collect())

- **OCR Parameter for TTS+VC Mode** — New `ocr` parameter for one-liner TTS+VC commands to extract text from images.
  - Same functionality as TTS mode but with voice cloning support
  - Extracted text is synthesized and then cloned to match the target voice reference
  - Example: `python src/voder.py tts ocr "subtitle_image.jpg" target "text: speaker_clone.wav"`
  - Full resource cleanup ensures memory efficiency

- **Mimic Flag for STS Mode** — New `mimic` keyword for one-liner STS commands to enable accent and emotion conversion alongside voice timbre transfer.
  - When `mimic` is present, Seed-VC v2 uses both its AR model (accent/emotion/style) and CFM model (timbre) instead of CFM only
  - This transfers not just the voice sound but also the speaking style, tone patterns, and emotional delivery of the target voice
  - The `mimic` and `music` keywords are mutually exclusive — using both together produces an error
  - Example: `python src/voder.py sts base "source.wav" target "reference.wav" mimic`

### Fixed

- **Seed-VC v2 Inference Path and Parameters** — Fixed STS voice conversion to use the official recommended inference pipeline and parameters.
  - Switched from `convert_voice()` (legacy non-streaming path) to `convert_voice_with_streaming()` (official v2 inference path)
  - Updated CFG rates from 0.5/0.5 to 0.7/0.7 (intelligibility and similarity) to match official Seed-VC defaults
  - These rates directly control how clearly the content is preserved and how closely the output matches the reference voice
  - The streaming path also handles long audio with proper overlapping chunk processing and reference length limiting

- **STS Mode Music Flag Parsing** — Fixed argument parsing for the `music` flag in STS mode one-liner commands.
  - Music flag detection moved earlier in the parsing logic to prevent conflicts
  - Previously, the `music` keyword could be incorrectly consumed as a script parameter
  - Now correctly handled as a standalone flag alongside other keywords
  - The `music` keyword was also removed from `valid_keywords` list since it should be treated as a flag, not a keyword parameter

- **SeedVCV1 Indentation Fix** — Corrected indentation issue in SeedVCV1 voice conversion processing.
  - The tensor trimming operation `vc_target = vc_target[:, :, mel2.size(-1):]` was incorrectly indented inside a conditional block
  - Fixed to execute unconditionally after voice conversion processing
  - Ensures consistent output length across all voice conversion operations

- **Vocal Language Parameter** — Corrected vocal_language parameter initialization in voice conversion.
  - Initial change from 'en' to 'unknown' was reverted back to 'en'
  - Ensures proper language handling for English vocal content

### Optimized

- **Increased Sequence Length for Voice Conversion** — Extended max_seq_len from 4096 to 8192 in SeedVC models.
  - Updated `max_seq_len` parameter in `BaseModelArgs` class (src/modules/v2/ar.py)
  - Updated `setup_ar_caches()` call in `SeedVCV2` class
  - Allows processing of longer audio segments without sequence length truncation
  - Improves voice conversion quality for extended audio content

- **Memory Cleanup After TTM+VC Processing** — Implemented explicit memory cleanup in TTM+VC voice conversion pipeline.
  - ACE-Step model is now explicitly released after music generation with `del ace_step` and `ace_step = None`
  - Seed-VC model is now explicitly released after voice conversion with `del seed_vc` and `seed_vc = None`
  - Both stages use `gc.collect()` and `torch.cuda.empty_cache()` for proper memory reclamation
  - Reduces peak memory usage during TTM+VC operations

- **Memory Cleanup After SeedVCV1 Processing** — Implemented explicit memory cleanup after SeedVCV1 voice conversion.
  - All models in SeedVCV1 are now released after processing: whisper_model, whisper_feature_extractor, campplus_model, rmvpe, and main model
  - Uses `del` statements followed by `gc.collect()` and `torch.cuda.empty_cache()`
  - Prevents memory accumulation during batch voice conversion operations

### Changed

- **Updated VODER Logo** — Redesigned and increased logo display size in README for better visibility.
  - Logo was completely redesigned with a fresh, modern look
  - Logo size increased from 128x128 pixels to 256x256 pixels
  - Provides better visual presence on high-DPI displays

- **SE Mode Output Description** — Clarified SE mode output format in documentation.
  - Updated table to show SE mode supports both audio and video input (was unclear before)

---

## 04/08/2026
- Status: Stable, all features work, still developing
- **Major Update — Two New Modes, Script Directives, SFX Integration, and Speech Enhancement**

### Added

#### New Processing Modes

- **SE (Speech Enhancement) Mode** — A new standalone mode for audio quality improvement.
  - Uses UniSE model from [alibaba/unified-audio](https://github.com/alibaba/unified-audio) for professional speech enhancement
  - Denoising — removes background noise, hiss, and artifacts
  - Dereverberation — reduces room echo and reverb effects for cleaner speech
  - Speech restoration — enhances clarity and intelligibility of degraded recordings
  - Supports audio files (WAV, MP3, FLAC, OGG, etc.) and video files (MP4, MKV, AVI, etc.)
  - Video input: audio is automatically extracted for processing
  - Outputs at 16kHz sample rate (optimized for speech content)
  - **Not designed for musical enhancement** — use for speech-only content
  - One-liner CLI: `python voder.py se "noisy_audio.wav" result "/output/enhanced.wav"`
  - Interactive CLI: select option 7 (SE) from the menu
  - Model cached under `src/models/checkpoints/unise/`

- **SFX (Sound Effects Generation) Mode** — A new standalone mode for generating custom sound effects from text.
  - Uses TangoFlux model from [declare-lab/TangoFlux](https://github.com/declare-lab/TangoFlux) for text-to-audio synthesis
  - Generates any sound effect from text descriptions (nature sounds, impacts, ambient, synthesized, etc.)
  - Configurable duration: 1-30 seconds per sound effect
  - Adjustable inference steps: 1-100 (default: 30) — higher values = better quality, slower generation
  - Adjustable guidance scale: 1.0-10.0 (default: 4.5) — controls adherence to the prompt
  - 44.1kHz output quality for professional use
  - One-liner CLI format: `python voder.py sfx sound "thunder rumbling" duration 10 steps 50 guide 3.5`
  - Parameters:
    - `sound` — Text description of the sound effect (required)
    - `duration` — Duration in seconds, 1-30 (required)
    - `steps` — Inference steps for quality control (optional, default 30)
    - `guide` — Guidance scale for prompt adherence (optional, default 4.5)
    - `result` — Output file path (optional)
  - Model cached under `src/models/checkpoints/tangoflux/`

#### Dialogue System Enhancements

- **Script Directives** — Per-line control over timing, volume, and duration within dialogue scripts.
  - `/time:nn` — Position this line at `nn` seconds from the start of the output
  - `/time:nn-nn` — Position at `nn` seconds, cut `-nn` seconds from the end of the clip
  - `/time:nn+nn` — Position at `nn` seconds, cut `+nn` seconds from the start of the clip
  - `/time:nn-nn+nn` — Position at `nn` seconds, cut from both end and start
  - `/level:0-100` — Set volume level for this specific line (default: 100)
  - `/duration:1-30` — Duration for SFX lines (required when using `sfx:` character)
  - Directives are appended at the end of dialogue text, separated by spaces
  - Example: `James: Hello everyone! /time:5 /level:80`
  - Enables precise control over audio production without manual post-processing

- **SFX Character in Dialogue** — Embed sound effects directly in dialogue scripts.
  - Special character `sfx:` (case-insensitive) generates sound effects within dialogue
  - Requires `/duration:nn` directive (1-30 seconds) — mandatory for all SFX lines
  - Optional `/level:0-100` directive to control SFX volume relative to dialogue
  - SFX generation uses TangoFlux model (same as standalone SFX mode)
  - SFX clips are positioned in the timeline using `/time:` directive if specified
  - Example dialogue:
    ```
    James: Welcome to our show!
    sfx: audience applause /duration:5 /level:60
    Sarah: Thanks for having us!
    sfx: gentle ambient music /duration:15 /level:30 /time:0
    James: Let's get started with today's topic.
    ```
  - Available in both TTS and TTS+VC dialogue modes
  - Works in GUI, interactive CLI, and one-liner CLI

- **Enhanced Dialogue Assembly** — Complete rewrite of the dialogue generation pipeline.
  - New `_assemble_enhanced_dialogue()` function handles all dialogue assembly
  - Per-clip audio effects using FFmpeg (time positioning, volume control)
  - SFX generation integrated into the dialogue pipeline
  - Support for overlapping audio via time positioning
  - Automatic calculation of total duration based on positioned clips
  - Efficient temp file management with automatic cleanup

- **Cross-use Feature** — Mix generated and cloned voices in the same dialogue.
  - Both TTS and TTS+VC one-line modes now support combining `voice` and `target` parameters
  - Use `voice "Character: prompt"` for generated voices (Voice Design)
  - Use `target "Character: path"` for cloned voices (Voice Cloning)
  - Example: `python voder.py tts script "James: Hello" "Sarah: Hi" voice "James: male" target "Sarah: /path/to/sarah.wav"`
  - Enables hybrid dialogues where some characters use generated voices and others use cloned references
  - A character cannot have both `voice` and `target` — each character must use one or the other

- **Music Volume Level Control** — Fine-grained control over background music volume.
  - New `level` parameter for one-liner dialogue commands
  - Supports constant volume, time-based segments, and fade transitions
  - Format options:
    - `"volume"` — Constant volume percentage (e.g., `"35"` for 35%)
    - `"start:vol-end:vol"` — Different volumes at start and end times
    - `"start:from-to+fade"` — Fade from one volume to another over specified duration
  - Default remains 35% if `level` parameter is not specified
  - Example: `python voder.py tts script "James: Hello" voice "James: male" music "piano" level "0:30-60:50"`
  - Time-based segments allow dynamic music volume throughout the dialogue
  - FFmpeg volume filter with evaluate-per-frame for smooth transitions

#### TTM Mode Enhancement

- **Instrumental Music Generation** — TTM mode now produces music-only (no vocals) output.
  - Use empty lyrics `"..."` to generate instrumental music without any vocal content
  - The model produces music matching the style prompt without singing
  - Ideal for background music, ambient tracks, and instrumental compositions
  - Example: `python voder.py ttm lyrics "..." styling "cinematic orchestral" duration 60`
  - Works with TTM+VC mode as well — voice conversion will have no effect on instrumental output
  - Lyrics in `()` or `[]` brackets provide context without being sung (for music-with-vocals generation)

#### Auto-Clone Feature Enhancement

- **TTS+VC Dialogue + Auto-Clone Trick** — Useful behavior when using the same file for both dialogue source and auto-clone.
  - Dialogue source analysis generates character names as `1`, `2`, `3`... based on speaker detection order
  - Auto-clone voice extraction produces voice references labeled `speaker 1`, `speaker 2`, etc.
  - The system matches character names to voice references **alphabetically**
  - **Result:** Using the same input file for both dialogue source and auto-clone produces an exact replica of the original audio
  - This is useful for:
    - Testing the TTS+VC pipeline accuracy
    - Verifying speaker detection quality
    - Creating backup/restoration of audio content
    - Demonstrating the voice cloning system's capabilities

### Changed

- **Expanded Processing Modes** — VODER now has 9 processing modes (up from 7).
  - Modes: STT+TTS, TTS, TTS+VC, STS, TTM, TTM+VC, STT, SE, SFX
  - GUI dropdown updated with new mode options
  - Interactive CLI menu updated with options 7 (SE) and 8 (SFX)

- **Centralized Model Storage** — New model directories added for UniSE and TangoFlux.
  - `src/models/checkpoints/unise/` — UniSE speech enhancement model
  - `src/models/checkpoints/tangoflux/` — TangoFlux sound effects model
  - Directories auto-created at startup

- **Updated Dependencies**:
  - `transformers==4.57.3` (pinned version, was `>=4.30.0`)
  - New: `einx`, `x-transformers==2.3.1`, `safetensors`, `soxr`, `tqdm`, `packaging` — required for UniSE model
  - These enable speech enhancement model loading and inference

### Technical Notes

- Code size increased from ~6650 lines (voder.py bleed v1) to ~7100+ lines — approximately 7% increase
- New imports: `traceback` for enhanced error handling in new modes
- UniSE model loaded from `src/unise/` module with `UniSEEnhancer` class
- TangoFlux model loaded from `src/tangoflux/` module with `TangoFluxGenerator` class
- All new models (UniSE, TangoFlux) are immediately offloaded after use to prevent memory accumulation
- Background music generation now supports volume level specifications with time-based control
- FFmpeg filter expressions built dynamically for complex volume automation

---

## 04/03/2026
- Status: Stable, all features work, still developing

### Added
- **Background Music Chunking for Long Dialogues** — Enhanced background music generation to handle dialogues longer than 250 seconds by generating multiple music chunks and concatenating them.
  - When background music is enabled and required duration exceeds 250 seconds, the system now generates multiple consecutive music chunks (250s each) using the same music description
  - All chunks are concatenated into a single music file using FFmpeg concat demuxer before mixing with dialogue
  - This ensures uninterrupted background music throughout the entire dialogue instead of silence after the first chunk
  - Maximum chunk size set to 250 seconds for optimal performance and compatibility with ACE-Step model limits

### Fixed
- **Music Generation Minimum Duration** — Fixed VODER's minimum music duration to match ACE-Step model requirements.
  - ACE-Step model requires a minimum of 10 seconds for generation, but VODER previously allowed inputs as low as 5 seconds
  - VODER now enforces 10-second minimum for music generation to prevent generation failures
  - This applies to both GUI and CLI modes

## 04/02/2026
- Status: Stable, all features work, still developing

### Added
- **STT (Speech-to-Text) Standalone Mode** — A new dedicated `stt` mode available via one-liner CLI.
  - Transcribe audio, video, image, or YouTube URL to text using Whisper
  - Supports `timestamp` flag to include word-level timestamps in format `[mm:ss:mss-mm:ss:mss]`
  - Supports `dialogue` flag to enable speaker diarization (pyannote) with numbered speaker labels
  - Batch processing: multiple files processed sequentially in a single command
  - Supports audio files, video files (auto-extracts audio), image files (OCR via EasyOCR), and YouTube/Bilibili/TikTok URLs
  - Output saved as `.txt` files with descriptive naming (e.g., `voder_stt_timestamp_dialogue_...`, `voder_stt_ocr_...`, `voder_youtube_...`)
  - One-liner examples:
    ```
    python voder.py stt "audio.wav"
    python voder.py stt "audio.wav" timestamp
    python voder.py stt "audio.wav" dialogue
    python voder.py stt "audio.wav" timestamp dialogue
    python voder.py stt "audio1.wav" "audio2.wav"
    python voder.py stt "https://youtube.com/watch?v=..."
    ```

- **Speaker Diarization (Pyannote Integration)** — New `SpeakerDiarization` class for identifying and separating speakers in audio.
  - Uses `pyannote/speaker-diarization-community-1` model (requires HuggingFace token with accepted model conditions)
  - Word-level alignment between Whisper transcription timestamps and pyannote speaker turns
  - Three-tier speaker assignment: fully-contained words get priority, then best overlap, then nearest neighbor
  - Post-processing: assigns words without speakers to nearest speaker; merges very short utterances (<0.5s) with neighbors
  - Available in STT mode (`dialogue` flag), dialogue source analysis, and voice clip extraction
  - Local pyannote integration bundled in `src/libs/pyannote/` to avoid huggingface_hub version conflicts
  - Requires HF_TOKEN.txt with access to `pyannote/speaker-diarization-community-1`

- **Image Text Extraction (EasyOCR)** — New `EasyOCRReader` class for extracting text from images.
  - Supports `.png`, `.jpg`, `.jpeg`, `.bmp`, `.gif`, `.tiff`, `.webp`
  - CPU-based OCR (no GPU required)
  - Can be used as input source for STT mode and dialogue source analysis
  - Images always produce single-speaker output (no diarization possible)
  - Model cached under centralized `src/models/checkpoints/easyocr/`

- **YouTube & Video Platform Download (yt-dlp)** — New download integration for audio extraction from URLs.
  - Supports YouTube, YouTube Shorts, youtu.be, Bilibili, and TikTok URLs
  - Downloads best available audio via yt-dlp, converts to MP3 (192kbps)
  - Robust error handling for invalid URLs, unavailable videos, network errors
  - Falls back to `.m4a`, `.wav`, `.webm` if `.mp3` not created
  - Temporary files cleaned up after processing
  - Works as input for STT mode, dialogue source analysis, and voice clip extraction

- **Dialogue Source from Files & URLs** — Comprehensive multi-format dialogue source support.
  - New `validate_dialogue_source_file()` accepts: text files, audio/video files, image files, YouTube URLs
  - New `analyze_dialogue_source()` full analysis pipeline:
    - Text files: parsed with auto-formatting (auto-detects dialogue vs single mode, normalizes format)
    - Audio/Video files: Whisper transcription + optional pyannote diarization
    - Image files: EasyOCR text extraction (single speaker)
    - YouTube URLs: download + Whisper transcription + optional diarization
  - Speaker label mapping: original pyannote labels → numbered speakers (1, 2, 3...)
  - Smart speaker switching: only switches on significant gaps (0.3s) or after 3+ words to avoid rapid switching artifacts
  - Available in interactive CLI for both TTS and TTS+VC modes

- **Automatic Voice Clip Extraction from Multi-Speaker Audio** — *(Replaced by SS pipe — see 05/21/2026 `ss_extract_speakers()`)* ~~Extract individual speaker voice clips automatically.~~
  - ~~New `extract_voice_clips_from_multispeaker()` function~~ — Removed (dead code)
  - ~~Given a multi-speaker audio source (file or YouTube URL), extracts the longest voice clip per speaker~~
  - ~~Uses Whisper (word timestamps) + Pyannote (speaker diarization) for speaker identification~~
  - ~~Extracts clips via FFmpeg with precise timing~~
  - ~~Integrated with TTS+VC interactive CLI: after entering dialogue, user can provide a multi-speaker source and clips are auto-assigned to characters alphabetically~~
  - ~~Falls back to manual entry if not enough clips extracted~~

- **`result` Parameter** — New universal CLI parameter to copy the latest result to a specified path.
  - Works with all one-liner modes
  - Example: `python voder.py tts script "Hello" voice "male" result "/path/to/output.txt"`
  - Creates directories as needed, preserves file metadata

### Changed
- **Centralized Model Management System** — Complete overhaul of model storage and caching.
  - All models now stored under `src/models/` with organized directory structure:
    - `models/tmp/` — temporary downloads in progress
    - `models/checkpoints/qwen_tts_voicedesign/` — Qwen3-TTS VoiceDesign
    - `models/checkpoints/qwen_tts_base/` — Qwen3-TTS Base
    - `models/checkpoints/acestep/` — ACE-Step models
    - `models/checkpoints/seed_vc_v1/` — Seed-VC v1
    - `models/checkpoints/seed_vc_v2/` — Seed-VC v2
    - `models/checkpoints/whisper/` — Whisper model
  - HuggingFace environment variables redirected to local directories (HF_HOME, HF_HUB_CACHE, TRANSFORMERS_CACHE, HUGGINGFACE_HUB_CACHE)
  - Whisper cache (XDG_CACHE_HOME) redirected for openai-whisper downloads
  - All directories auto-created at startup
  - All model classes (WhisperSTT, QwenTTSVoiceDesign, QwenTTS, SeedVCV2, SeedVCV1) use centralized paths

- **Enhanced HF_TOKEN Handling** — Improved token discovery and error messaging.
  - Multi-location fallback search: CWD, src directory, `1/` directory
  - Auto-creates template `HF_TOKEN.txt` with instructions if file doesn't exist
  - More informative warning messages with links to HuggingFace token settings and pyannote model conditions
  - Token set as `os.environ["HF_TOKEN"]` for all HuggingFace operations

- **Updated Dependencies**:
  - `torch==2.8.0`, `torchaudio==2.8.0`, `torchvision==0.23.0` (was >=2.0.0)
  - New: `torch-audiomentations>=0.12.0`, `hydra-core==1.3.2`, `opentelemetry-proto==1.0.0`, `opentelemetry-exporter-otlp-proto-grpc==1.0.0`, `sox`, `onnxruntime`, `lightning==2.4`, `yt-dlp>=2026.3.17`, `easyocr>=1.7.0`
  - `qwen-tts` removed from pip requirements — now bundled locally in `src/qwen_tts/`
  - Pyannote bundled locally in `src/libs/pyannote/` (requirements2.txt includes `pyannote.database`, `pyannote.metrics`, `pyannote.pipeline`, `asteroid-filterbanks`)

### Technical Notes
- Code size increased from ~4880 lines (voder.py stable) to ~6650 lines (voder.py bleed) — a 36% increase
- voder2.py is a slightly stripped version of voder.py (same functionality, fewer comments/docstrings)
- All new models (EasyOCR, Pyannote) are immediately offloaded after use to prevent memory accumulation
- The `load_custom_model_from_hf()` function now accepts optional `target_dir` parameter for flexible download locations

## 02/24/2026
- Status: Stable, all features work, still developing, there will be a major update!

### Added
- **MSTS (Music-STS) in STS mode** – STS now supports musical inputs via the Seed-VC v1 model (44.1kHz) for better music voice conversion quality.
  - **GUI**: When pressing Generate in STS mode, a dialog appears asking "musical inputs?" with Yes/No buttons. Yes uses v1 model at 44.1kHz; No uses standard v2 at 22.05kHz.
  - **Interactive CLI**: After entering base and target paths, user is prompted "Are the inputs musical? (Y/N):". Y uses v1 model; N uses standard v2.
  - **One-line CLI**: New `music` keyword parameter: `voder.py sts path/base path/target music`. Invalid parameters show error message.
  - **Output naming**: MSTS outputs prefixed with `voder_m_sts_timestamp.wav`; standard STS uses `voder_sts_timestamp.wav`.

### Fixed
- **TTS+VC dialogue voice cloning stability** – Voice characteristics are now extracted once per character instead of re-extracting for each line.
  - In dialogue with multiple lines per character (e.g., 5 lines for "James"), the voice prompt is extracted once and reused for all lines of that character.
  - This ensures consistent voice quality throughout the dialogue, eliminating variations that occurred when re-extracting voice for each line.
  - Applies to GUI, interactive CLI, and one-line CLI modes.

### Optimized
- **Memory offloading after processing** – Models are now explicitly unloaded from memory/VRAM after each operation completes.
  - In GUI mode: ProcessingThread now calls cleanup() after finishing, releasing all loaded models.
  - In interactive CLI mode: Each mode (TTS, TTS+VC, STS, STT+TTS, TTM, TTM+VC) now offloads models before returning.
  - This prevents memory accumulation when performing multiple operations in a single session.
  - Pattern applied: `del model`, `gc.collect()`, `torch.cuda.empty_cache()`.

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
  - New rows auto-add when the last row is filled; first row has no delete button, subsequent rows can be deleted.
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
