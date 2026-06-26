# VODER Prebuilt Chains — Implementation Plan

Working plan for the prebuilt chains feature. Tracked in repo so context isn't lost across sessions. Update as decisions change.

---

## 1. Confirmed Design Decisions

| # | Decision | Choice |
|---|---|---|
| 1 | `.chain` file internal format | **Custom KV** (key:value with `---` separators between steps) |
| 2 | `chains build` content syntax | **Single quoted string** per chain content (parsed internally) |
| 3 | Oneline `chains load` override | **Yes** — user can put a chain number in parens to override automated refs |
| 4 | Voice profile validity | **Valid wherever audio is valid** (validator accepts `.tts`/`.ttse` for any audio position) |
| 5 | Multi-prebuilt-chain linking | **Explicit name reference** (chain B's file references chain A by name; runner maintains global index across prebuilts) |
| 6 | Progress tracker granularity | **Chain + input level** (no sub-step tracking within a chain — would require plumbing callback through ~15 oneline handlers) |
| 7 | Analyze journey on error | **Continue hypothetically** (show error inline, then continue narrative with placeholder) |
| 8 | Default file location | **`src/chains/`** next to voder.py |

---

## 2. `.chain` File Format Spec

Custom KV format. Plain text, UTF-8, LF newlines.

```
# VODER_CHAIN v1 <timestamp> <name>
title: <title or empty>
description: <description or empty>
---
chain: <chain-name>
comment: <comment or empty>
content: <chain-content-as-single-line>
---
chain: <chain-name>
comment: <comment or empty>
content: <chain-content-as-single-line>
---
```

**Rules:**
- Line 1 MUST start with `# VODER_CHAIN v1 ` followed by ISO-8601-ish timestamp (`YYYYMMDD_HHMMSS`) and the chain name (no spaces).
- `title:` and `description:` are optional. If absent, treated as empty string. Validator warns but does not error.
- Each `---` separates a chain step block. First block is the header (title/description), subsequent blocks are chain steps.
- `chain:` is required in every step block. Name must match `[A-Za-z0-9_-]+` (no spaces). Duplicate names within a file are an error.
- `comment:` is optional in step blocks. Empty `comment:` line is fine.
- `content:` is required. Single line. This is the oneline command for that chain, space-separated, with `input` as the placeholder for manual inputs and prior chain names as references for automated inputs.
- Comments in the file (lines starting with `#` outside line 1) are NOT allowed — the spec said "no in-code comments" applies to source files; for .chain files we keep the format strict. The only `#` line is line 1.

**Example file `src/chains/VODER_bombo_20260627_143022.chain`:**
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
comment: Provide a reference voice (audio file, URL, or .tts/.ttse voice profile). The transcribed lyrics from chain 2 will be spoken in this voice.
content: tts script lyrics voice input
---
```

**Classification logic per chain step:**
- Count `input` placeholders in content → `manual_input_count`
- Count references to prior chain names in content → `automated_input_count`
- If `manual_input_count == 0` and `automated_input_count > 0` → **automated**
- If `manual_input_count > 0` and `automated_input_count == 0` → **manual**
- If both > 0 → **semi-automated**
- If both == 0 → **error** (chain produces nothing useful, no inputs at all)

---

## 3. `chains build` Command Spec

**Usage:**
```
python voder.py chains build "<name>" description "<title - short description>" \
    chain "<chain1-name>" "<comment1>" "<content1>" \
    chain "<chain2-name>" "<comment2>" "<content2>" \
    ...
```

**Argument parsing rules:**
- `argv[2]` = `<name>` (required). Must match `[A-Za-z0-9_-]+`. Error+stop if missing or has spaces.
- `argv[3]` = literal `description` keyword. If missing → error.
- `argv[4]` = `<title - short description>` (required after `description`). Can be empty string `""` (validator warns but proceeds).
- Then repeating triplets starting with literal `chain` keyword:
  - `chain` (literal)
  - `<chain-name>` (quoted, must match `[A-Za-z0-9_-]+`, no duplicates within file)
  - `<comment>` (quoted, can be empty `""`)
  - `<content>` (quoted, single string — split on whitespace internally to get argv tokens)
- At least one `chain` block required.

**Build phases:**
1. **Basic validation**: name format, description keyword present, each chain block has 4 elements (chain/name/comment/content), chain names unique, content non-empty.
2. **Deep verification**: parse each chain's content as oneline argv, run through `parse_oneline_args` + `validate_oneline_mode` to check syntax validity. Report per-chain errors with line/chain reference. Do NOT execute anything.
3. **Save**: generate timestamp, write to `src/chains/VODER_<name>_<timestamp>.chain`. Create `src/chains/` if missing.
4. **Confirm**: print saved path + summary (N chains, M manual inputs total, K automated refs).

**Test mode**: user can immediately run `python voder.py chains load "<name>"` to test.

---

## 4. `chains load` Command Spec (Oneline)

**Usage:**
```
python voder.py chains load "<chain-name-or-path>" [1:"(input1/input2/...)" 2:"(...)" ...] [another-chain-name-or-path 1:"(...)" ...]
```

**Argument parsing rules:**
- `argv[2]` = first prebuilt chain name (latest by timestamp) or full path to `.chain` file.
- Then optional markers `N:"(a/b/c)"` where N is the chain position (1-indexed) within the file. Slash-separated values inside parens supply the MANUAL inputs for that chain step.
- For AUTOMATED inputs (chain references), user can put a chain number instead of a file path — the runner resolves it to that chain's output path.
- Multiple prebuilt chains can be loaded in sequence: after the first chain's markers, another chain name/path starts a new prebuilt chain section. Each subsequent prebuilt chain can reference the prior prebuilt's name to receive its final output.

**Example:**
```
python voder.py chains load "bombo" 1:"(song.wav)" 3:"(ref.wav)"
```
- Loads latest `VODER_bombo_*.chain`
- Chain 1 (vocals, manual): supply `song.wav` as the input
- Chain 2 (lyrics, automated): uses chain 1's output (default)
- Chain 3 (cover, semi-automated): supply `ref.wav` as the manual input; `lyrics` reference auto-resolves to chain 2's output

**Multi-prebuilt example:**
```
python voder.py chains load "bombo" 1:"(song.wav)" 3:"(ref.wav)" "second_chain" 1:"(bombo)"
```
- Runs `bombo` first → final output X
- Runs `second_chain` next; its first chain's `input` placeholder is overridden by `bombo` (resolves to X)

**Validation:**
- File must exist and end in `.chain`. Else: error.
- Marker numbers must be 1..N where N = chains in file. Else: error.
- Number of inputs in parens must match `manual_input_count` for that chain. Else: error (with the expected count in the message).
- If user supplies a chain number as an "input" for an automated override, that chain number must be < current chain number (can only reference prior chains). Else: error.

---

## 5. `chains analyze` Command Spec (Oneline)

**Usage:**
```
python voder.py chains analyze "<chain-path-or-name>" ["<another>" ...]
```

**Output:** `results/voder_analyze_chain_<safe-name>_<timestamp>.md`

**Report structure:**
1. **Header**: title, generated timestamp, chain name, file path, total chains, total inputs.
2. **Per-chain summary table**: | # | name | type (manual/automated/semi) | inputs (manual N / automated K) | comment excerpt |
3. **Journey section**: narrative walk-through, one subsection per chain step:
   - Step header: `### Step N: <name> (type)`
   - Comment (if any)
   - Content (with references resolved to "<would-use-output-of-step-X>")
   - Input list (if manual): each input's expected file format
   - If verification fails for this step: `> ERROR: <message>` block, then `> Assuming this step succeeded, the journey continues...`
   - If verification passes: `> OK: syntax valid, expected inputs match`
4. **Errors summary** at bottom: if any errors found, list them; else `All checks passed. Chain is ready to use.`

**Verification checks performed:**
- File format (line 1 magic, header keys, step block structure)
- Chain name uniqueness
- Chain name format `[A-Za-z0-9_-]+`
- Content parses as valid oneline (via `parse_oneline_args` + `validate_oneline_mode`)
- `input` placeholders count ≥ 0
- Chain references point to prior chains (no forward refs, no self-refs)
- Each chain has at least one input (manual or automated) — else warning
- Optional: title/description/comment presence (warning, not error)

---

## 6. Interactive CLI Chains Mode (UX Spec)

New menu item: `9. Prebuilt Chains` in `interactive_cli_mode` dispatch table.

**File:** `voders/interactiveCLI/chains.py` (new file, ~400-500 lines)

**Mode entry flow:**

1. Print mode banner: `--- Prebuilt Chains Mode ---`
2. Print two options:
   - `1. List available chains`
   - `2. Enter chain name (or path)`
3. Read user choice. Invalid → warning + retry.

**Branch 1 — List:**
- Enumerate `src/chains/*.chain` (and `chains/*.chain` if it exists — wait, decision 8 says only `src/chains/`).
- Sort by internal timestamp (line 1) descending.
- Print numbered list: `1. bombo (2026-06-27 14:30) — Bombo Pipeline`
- Read user input: number → load that chain; `back` → return to mode entry; invalid → warning + retry.

**Branch 2 — Name/Path:**
- Read user input. If contains `/` or `\\`, treat as path. Else treat as name → find latest `VODER_<name>_*.chain`.
- Non-existent or non-`.chain` → warning + retry.

**Multi-chain selection (both branches):**
- After loading first chain, ask: `Load another prebuilt chain? (number to add, name to add, or Enter to proceed)`
- If number/name: validate, add to list, repeat.
- If empty input: proceed to input gathering.
- Selected chains run in order; each subsequent chain can reference prior prebuilt chain names.

**Input gathering phase (per loaded prebuilt chain, per chain step within):**

For each chain step in order:
1. Print step header: `Chain <prebuilt_idx>.<step_idx> — "<chain_name>" (<type>)`
2. Print comment (if non-empty): `Comment: <comment>`
3. Determine step type:
   - **Manual**: for each `input` placeholder, ask user for a file path. Show: `Input <n>/<total> for chain "<name>" — valid inputs: <formats>`. Validate. On invalid → warning + retry. On valid → store path.
   - **Automated**: print `This chain is automated — uses outputs from prior chains. Press Enter to continue.` Show which prior chains it references: `References: chain "<name1>" (step <n1>), chain "<name2>" (step <n2>)`. Wait for Enter.
   - **Semi-automated**: do both — show automated references AND ask for manual inputs.
4. Update progress tracker (see below).

**Progress tracker format (printed before each step):**
```
============================================================
Prebuilt Chain 1 of 2 (bombo) — Step 2 of 3 (lyrics) — Automated
  Resolved inputs: vocals = /path/to/temp/voder_chain_vocals_xxx.wav
============================================================
```

For manual/semi-automated, also show: `Input 1 of 2 supplied: song.wav`

**After all inputs gathered for all selected prebuilt chains:**
- Print summary: `All inputs ready. Press Enter to run chain(s).`
- Wait for Enter.

**Execution phase:**
- Print: `Running prebuilt chain 1 of 2: bombo (3 steps)...`
- For each chain step:
  - Substitute placeholders: `input` → user-supplied paths; prior chain names → resolved temp paths.
  - Print: `[Chain 1.2/3] name="lyrics"  >>>  stt /path/to/vocals timestamp`
  - Call `parse_and_execute_oneline(substituted_argv)`.
  - On success: snapshot `results/` + `voices/` + `temp_chains/` to find new files; move intermediate outputs to `temp_chains/` (reuse existing `ChainPipeline` logic).
  - On failure: print `Something went further than expected.` + error message (max 500 chars) + `Error occurred at: Prebuilt Chain <P>, Step <S> (<chain-name>), Input <I>`. Abort.
- After last step of last prebuilt chain: print `Done! Final output: <path>`. Print `--- What's Next? ---` + `1. Blend Again / 2. Exit` (same as other modes).

---

## 7. Verification Logic Spec

**Three verification contexts, all using the same core function:**

`verify_chain_file(chain_path) -> (ok: bool, errors: list[VerificationError], warnings: list[str])`

Returns structured errors so the caller (build, load, analyze, interactive) can format them appropriately.

**VerificationError structure:**
```python
{
    'step_index': int | None,   # None for file-level errors
    'step_name': str | None,
    'category': str,            # 'format' | 'syntax' | 'reference' | 'input' | 'naming'
    'message': str,
    'fix': str,                 # suggested fix
}
```

**Verification checks (in order):**

1. **File-level format checks:**
   - Line 1 starts with `# VODER_CHAIN v1 `
   - Line 1 has 4 whitespace-separated tokens: `#`, `VODER_CHAIN`, `v1`, `<timestamp>`, `<name>` (5 tokens counting the magic)
   - Timestamp matches `YYYYMMDD_HHMMSS` format
   - Name matches `[A-Za-z0-9_-]+`
   - Header block has `title:` and/or `description:` (warnings if absent)
   - At least one step block follows the header

2. **Per-step format checks:**
   - Block has `chain:` key (required)
   - Block has `content:` key (required)
   - `comment:` optional
   - No unknown keys

3. **Naming checks:**
   - Each chain name matches `[A-Za-z0-9_-]+`
   - Chain names unique within file

4. **Content syntax checks (per step):**
   - Split content on whitespace → argv list
   - First token must be a valid oneline mode (`tts`/`sts`/`ttm`/`stt`/`se`/`sfx`/`svs`/`ss`/`train`/`quest`)
   - Run `parse_oneline_args(argv)` — must not return `error`
   - `validate_oneline_mode(mode)` must return non-None

5. **Reference checks (per step):**
   - For each token in content that isn't `input` and isn't a recognized oneline keyword (from the parser's known keywords list) and isn't a file path / URL — check if it's a prior chain name. If it looks like a reference but doesn't match a prior chain name → error.
   - Forward references (chain N references chain M where M > N) → error.
   - Self-references → error.
   - `input` placeholder count ≥ 0 (always true, but checked).

6. **Input sanity checks (per step):**
   - If `input` count == 0 AND no references → warning: "chain has no inputs"
   - If `input` count > 5 → warning: "chain has many inputs, consider splitting"

7. **Optional checks (warnings only):**
   - Title empty
   - Description empty
   - Comment empty for a step

**Where verification runs:**
- **`chains build`**: runs full verification BEFORE saving the file. Errors → don't save, print all errors, exit non-zero. Warnings → print but save anyway.
- **`chains load`**: runs full verification before running. Errors → don't run, print all errors, exit non-zero.
- **`chains analyze`**: runs full verification, includes results in the .md report. Always writes the report (even if errors).
- **Interactive CLI chains mode**: runs full verification after loading the file but BEFORE asking user for inputs. Errors → "This chain file has verification errors:" + list + "Press 1 to try another chain, 2 to exit". Warnings → print but proceed.

---

## 8. Architecture — Where Code Lives

**Engine-level (in `voder.py` or new module):**

| Component | Location | Purpose |
|---|---|---|
| `.chain` file parser | new module `voders/prebuilt_chains.py` | Parse `.chain` files into structured data |
| `verify_chain_file()` | `voders/prebuilt_chains.py` | Verification logic (shared by all 4 contexts) |
| `chains build` handler | `voders/prebuilt_chains.py` → called from `oneline_chains` | Build a `.chain` file from CLI args |
| `chains load` handler | `voders/prebuilt_chains.py` → called from `oneline_chains` | Load and run a `.chain` file with optional input overrides |
| `chains analyze` handler | `voders/prebuilt_chains.py` → called from `oneline_chains` | Generate `.md` report |
| `ChainPipeline` extension | modify `voders/sidequests.py` | Add `execute_prebuilt()` method that takes a pre-built list of (name, command_args) tuples with pre-resolved inputs (instead of calling `substitute_refs` live) |
| Mode → input format table | new constant in `voder.py` near line 4073 | `MODE_INPUT_FORMATS = {'tts': [...], 'sts': [...], ...}` used by UX to advertise valid inputs |
| Subcommand dispatch in `parse_oneline_args` | modify `voder.py` line 5054 | Peel off `build` / `load` / `analyze` keywords before treating rest as raw pipeline |

**UX-level (in `voders/interactiveCLI/chains.py`):**

| Component | Purpose |
|---|---|
| `cli_chains_mode()` | Interactive CLI entry point — menu, list, load, multi-select |
| Input gathering loop | Per-step input prompts with validation, progress tracker |
| Execution orchestrator | Calls `parse_and_execute_oneline` for each step, captures outputs, handles errors |

**Constants:**

| Name | Location | Value |
|---|---|---|
| `PREBUILT_CHAINS_DIR` | `voder.py` | `os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chains')` |
| `CHAIN_FILE_MAGIC` | `voders/prebuilt_chains.py` | `"# VODER_CHAIN v1"` |
| `CHAIN_FILE_EXT` | `voders/prebuilt_chains.py` | `".chain"` |
| `MODE_INPUT_FORMATS` | `voder.py` near line 4073 | dict mapping mode → list of acceptable input descriptions |

**No in-code comments anywhere.** Per project convention.

---

## 9. Implementation Phases (ordered, atomic, committable)

Each phase = one commit. Each phase is independently testable.

### Phase 1: `.chain` file format + parser + verifier (foundation)
- Create `voders/prebuilt_chains.py` with:
  - `parse_chain_file(path) -> dict` (returns `{name, timestamp, title, description, chains: [{name, comment, content}]}`)
  - `verify_chain_file(path) -> (ok, errors, warnings)` (implements all checks in §7)
  - `classify_chain_step(step, prior_names) -> 'manual' | 'automated' | 'semi-automated' | 'error'`
  - `find_chain_by_name(name) -> path | None` (latest by timestamp)
  - `list_chains() -> list[(path, name, timestamp, title)]`
- No CLI changes yet. Pure library code.
- **Test**: write a sample `.chain` file by hand, run `parse_chain_file` + `verify_chain_file` on it, confirm output.

### Phase 2: `chains build` command
- Modify `parse_oneline_args` in `voder.py` (line 5054) to peel off `build` / `load` / `analyze` keywords.
- Modify `oneline_chains` in `voders/sidequests.py` to dispatch to `prebuilt_chains.handle_build(params)` when `build` keyword detected.
- Implement `handle_build(params)` in `voders/prebuilt_chains.py`:
  - Basic validation (name, description keyword, chain block structure)
  - Call `verify_chain_file` on the would-be file content (in-memory) → if errors, print all and return False
  - Generate timestamp, write file, print confirmation
- **Test**: run `python voder.py chains build "bombo" description "Bombo Pipeline" chain "vocals" "Provide song" "svs voice input" chain "lyrics" "" "stt vocals timestamp" chain "cover" "Provide ref voice" "tts script lyrics voice input"` and confirm file created.

### Phase 3: `chains analyze` command
- Implement `handle_analyze(params)` in `voders/prebuilt_chains.py`:
  - Parse + verify the chain file
  - Generate the journey narrative (with hypothetical continuation on errors)
  - Write to `results/voder_analyze_chain_<name>_<timestamp>.md`
- Modify `oneline_chains` to dispatch to `handle_analyze` when `analyze` keyword detected.
- **Test**: run `python voder.py chains analyze "bombo"` and confirm `.md` file generated.

### Phase 4: `chains load` command (oneline prebuilt execution)
- Implement `handle_load(params)` in `voders/prebuilt_chains.py`:
  - Parse the load syntax: `load <name-or-path> [N:"(a/b/c)"]... [another-name N:"(...)"]...`
  - For each prebuilt chain:
    - Verify the file
    - Resolve manual inputs (from markers) and automated overrides (chain numbers in parens)
    - Execute chain steps in order using existing `ChainPipeline` machinery
    - Capture outputs, register in global index under prebuilt chain name
- Modify `oneline_chains` to dispatch to `handle_load` when `load` keyword detected.
- **Test**: run `python voder.py chains load "bombo" 1:"(song.wav)" 3:"(ref.wav)"` and confirm execution.

### Phase 5: `MODE_INPUT_FORMATS` table + UX helpers
- Add `MODE_INPUT_FORMATS` constant in `voder.py` near line 4073:
  ```python
  MODE_INPUT_FORMATS = {
      'tts': 'audio file / video file / supported platform URL / .tts or .ttse voice profile',
      'sts': 'audio file / video file / supported platform URL / .tts or .ttse voice profile',
      'ttm': 'audio file / video file / supported platform URL / text file',
      'stt': 'audio file / video file / supported platform URL',
      'se':  'audio file / video file / supported platform URL',
      'sfx': '(no file input — uses sound prompt text)',
      'svs': 'audio file / video file / supported platform URL',
      'ss':  'audio file / video file / supported platform URL',
      'train': 'audio file / video file / supported platform URL',
      'quest': 'varies by quest',
  }
  ```
- Add helper `get_input_formats_for_chain_step(content) -> str` that parses content's mode and returns the format string.

### Phase 6: Interactive CLI chains mode
- Create `voders/interactiveCLI/chains.py` with `cli_chains_mode()`.
- Add `9. Prebuilt Chains` to dispatch table in `voders/interactiveCLI/__init__.py`.
- Implement:
  - Menu (list / name-or-path / multi-select)
  - Per-step input gathering with validation and progress tracker
  - Execution orchestrator (reuses `parse_and_execute_oneline`)
  - Error handling (max 500 chars, location info)
  - "Blend Again / Exit" loop at end
- **Test**: run `python voder.py cli`, choose 9, walk through bombo chain end-to-end.

### Phase 7: Docs + CHANGELOG
- Update `docs/COMMAND_CATALOG.md` with full `chains build` / `chains load` / `chains analyze` reference + examples.
- Update `docs/Guide.md` with prebuilt chains architecture section.
- Update `docs/CHANGELOG.md` with new feature entry.
- Update `README.md` with brief mention.
- Update `docs/voder-skill.md` if it references chains.

### Phase 8: Verification + commit + push
- Run structural smoke test (extend `/home/z/my-project/scripts/smoke_test.py` to cover the new files).
- Run semantic resolution test on `voders/prebuilt_chains.py` and `voders/interactiveCLI/chains.py`.
- Confirm no in-code comments.
- Commit all changes with descriptive message.
- Push to origin/main.

---

## 10. Open Items / Things to Double-Check

- [ ] Does `parse_oneline_args` accept a list of argv tokens, or does it need the full `sys.argv[1:]` shape? (Check signature before reusing in `handle_build`'s content verification.)
- [ ] When the prebuilt chain runner calls `parse_and_execute_oneline` for each step, the existing `ChainPipeline.execute` snapshots `results/` to find new files. Reuse this mechanism — don't reinvent.
- [ ] For automated chains where the user just presses Enter, do we still need to call `parse_and_execute_oneline`? YES — the chain still runs, it just doesn't need user input. The "press Enter" is just UX acknowledgement.
- [ ] For semi-automated chains: order of operations matters. Show automated refs first (so user understands context), THEN ask for manual inputs.
- [ ] Voice profile validation: when user supplies a `.tts` or `.ttse` file as input, accept it wherever audio is accepted. The downstream oneline handler will handle resolution.
- [ ] Multi-prebuilt-chain in interactive: when user adds a second prebuilt chain, the input gathering for the second chain can reference the first prebuilt's name as an automated input. The runner's global index must include prebuilt names mapping to their final output paths.
- [ ] Error message 500-char limit: applies to interactive CLI only. Analyze .md report shows full error messages.
- [ ] The user said "we can do a starter verification and show a user a tracker for each chain and the per-chain validation, same logic that will be used in the oneline chains verification will be used here" — so interactive CLI does verification UP FRONT (before input gathering), showing per-chain validation results in a tracker. Then if all pass, proceeds to input gathering. If any fail, aborts with the "things went further than expected" message (though in this case it's "things went wrong before we even started").
- [ ] Actually re-reading: "if error, it will show the 'things went further than expected'....I mean that message I told you earlier" — so this message is for mid-run errors (things we couldn't predict). For pre-run verification errors, we show the errors directly and abort cleanly. Mid-run errors get the "things went further than expected" framing.
- [ ] The `quest` mode's input formats vary — for `MODE_INPUT_FORMATS`, just say "varies by quest". If a chain step uses `quest`, the UX can show that and let the user supply any file; the quest's own parser will validate.
- [ ] `sfx` mode has NO file input (only text prompt). If a chain step is `sfx sound "explosion"`, there are 0 manual inputs. The UX should detect this and skip the input-gathering phase entirely for that step (just show "Press Enter to run" or similar).
- [ ] When user runs multiple prebuilt chains and one fails mid-run, do we abort ALL remaining prebuilt chains? YES — the spec says "if something wrong happened... it will show a message for the user that something went further than expected and show the error message maximum 500 characters and will show what chain and exactly what Input step where the error happened". Doesn't mention continuing, so abort.

---

## 11. Status Tracker

- [x] Phase 1: `.chain` file format + parser + verifier
- [x] Phase 2: `chains build` command
- [x] Phase 3: `chains analyze` command
- [x] Phase 4: `chains load` command
- [x] Phase 5: `MODE_INPUT_FORMATS` table
- [x] Phase 6: Interactive CLI chains mode
- [x] Phase 7: Docs + CHANGELOG
- [x] Phase 8: Verification + commit + push (commit deb4c6f pushed to origin/main)
