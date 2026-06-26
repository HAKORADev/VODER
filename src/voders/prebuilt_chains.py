import os
import re
import sys
import glob
import time

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

CHAIN_FILE_MAGIC = "# VODER_CHAIN v1"
CHAIN_FILE_EXT = ".chain"
PREBUILT_CHAINS_DIR = os.path.join(_SRC_DIR, "chains")

_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

_VALID_CONTENT_MODES = {'tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train', 'quest'}


def _err(step_index, step_name, category, message, fix=""):
    return {
        "step_index": step_index,
        "step_name": step_name,
        "category": category,
        "message": message,
        "fix": fix,
    }


def build_chain_text(name, timestamp, title, description, steps):
    lines = [f"{CHAIN_FILE_MAGIC} {timestamp} {name}"]
    if title:
        lines.append(f"title: {title}")
    else:
        lines.append("title:")
    if description:
        lines.append(f"description: {description}")
    else:
        lines.append("description:")
    for step in steps:
        lines.append("---")
        lines.append(f"chain: {step['name']}")
        if step.get("comment"):
            lines.append(f"comment: {step['comment']}")
        else:
            lines.append("comment:")
        lines.append(f"content: {step['content']}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_chain_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        return None, [_err(None, None, "format", f"Could not read file: {e}",
                            "Check the file path and permissions.")]
    return _parse_chain_text(raw)


def _parse_chain_text(raw):
    errors = []
    lines = raw.splitlines()
    if not lines:
        errors.append(_err(None, None, "format", "File is empty",
                           "Add the magic header line and at least one chain step."))
        return None, errors

    magic_line = lines[0].strip()
    magic_parts = magic_line.split()
    if len(magic_parts) != 5 or " ".join(magic_parts[:3]) != CHAIN_FILE_MAGIC:
        errors.append(_err(None, None, "format",
                           f"First line must be exactly '{CHAIN_FILE_MAGIC} <timestamp> <name>' (5 whitespace-separated tokens)",
                           "Fix line 1 — name cannot contain spaces."))
        return None, errors

    timestamp = magic_parts[3]
    name = magic_parts[4]
    if not _TIMESTAMP_RE.match(timestamp):
        errors.append(_err(None, None, "format",
                           f"Timestamp '{timestamp}' does not match YYYYMMDD_HHMMSS",
                           "Generate with time.strftime('%Y%m%d_%H%M%S')."))
    if not _NAME_RE.match(name):
        errors.append(_err(None, None, "naming",
                           f"Chain name '{name}' contains invalid characters",
                           "Use only letters, digits, underscores, hyphens. No spaces."))

    blocks = []
    current = {}
    in_header = True
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---":
            if current:
                blocks.append(current)
                current = {}
            in_header = False
            continue
        if ":" not in stripped:
            errors.append(_err(None, None, "format",
                               f"Line does not match 'key: value' format: {stripped}",
                               "Use 'key: value' on each non-separator line."))
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if in_header:
            if key in ("title", "description"):
                current[key] = value
            else:
                errors.append(_err(None, None, "format",
                                   f"Unknown header key '{key}'",
                                   "Header only allows 'title' and 'description'."))
        else:
            if key in ("chain", "comment", "content"):
                current[key] = value
            else:
                errors.append(_err(None, None, "format",
                                   f"Unknown step key '{key}'",
                                   "Step blocks only allow 'chain', 'comment', 'content'."))
    if current:
        blocks.append(current)

    if not blocks:
        errors.append(_err(None, None, "format",
                           "No header block found",
                           "Add 'title:' and 'description:' lines after the magic line."))
        return None, errors

    header = blocks[0]
    title = header.get("title", "")
    description = header.get("description", "")
    step_blocks = blocks[1:]

    chains = []
    for idx, blk in enumerate(step_blocks, start=1):
        if "chain" not in blk:
            errors.append(_err(idx, None, "format",
                               f"Step {idx} missing 'chain:' key",
                               "Add 'chain: <name>' to this step."))
            continue
        cname = blk["chain"]
        if not _NAME_RE.match(cname):
            errors.append(_err(idx, cname, "naming",
                               f"Chain name '{cname}' has invalid characters",
                               "Use only letters, digits, underscores, hyphens."))
        if "content" not in blk:
            errors.append(_err(idx, cname, "format",
                               f"Step {idx} '{cname}' missing 'content:' key",
                               "Add 'content: <oneline command>' to this step."))
            continue
        content = blk["content"]
        if not content.strip():
            errors.append(_err(idx, cname, "format",
                               f"Step {idx} '{cname}' has empty content",
                               "Add the oneline command for this step."))
            continue
        comment = blk.get("comment", "")
        chains.append({
            "name": cname,
            "comment": comment,
            "content": content,
            "content_tokens": content.split(),
        })

    if not chains and not any(e["category"] == "format" for e in errors):
        errors.append(_err(None, None, "format",
                           "No chain steps found",
                           "Add at least one step block after the header."))

    if not chains:
        return None, errors

    parsed = {
        "name": name,
        "timestamp": timestamp,
        "title": title,
        "description": description,
        "chains": chains,
    }
    return parsed, errors


def verify_chain_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        return False, [_err(None, None, "format", f"Could not read file: {e}",
                            "Check the file path and permissions.")], []
    return verify_chain_text(raw)


def verify_chain_text(raw):
    parsed, errors = _parse_chain_text(raw)
    warnings = []
    if parsed is None:
        return False, errors, warnings

    chain_names = [c["name"] for c in parsed["chains"]]
    seen = set()
    for idx, c in enumerate(parsed["chains"], start=1):
        if c["name"] in seen:
            errors.append(_err(idx, c["name"], "naming",
                               f"Duplicate chain name '{c['name']}'",
                               "Rename this step to be unique within the file."))
        seen.add(c["name"])

    for idx, c in enumerate(parsed["chains"], start=1):
        errors.extend(_verify_content_syntax(idx, c))

    for idx, c in enumerate(parsed["chains"], start=1):
        errors.extend(_verify_references(idx, c, chain_names))

    for idx, c in enumerate(parsed["chains"], start=1):
        manual_count = sum(1 for t in c["content_tokens"] if t == "input")
        auto_count = sum(1 for t in c["content_tokens"] if t in chain_names and t != c["name"])
        if manual_count == 0 and auto_count == 0:
            warnings.append(f"Step {idx} '{c['name']}' has no 'input' placeholders and no chain references — it will run without any external input (OK for modes like sfx, but unusual for tts/sts/etc.).")
        if manual_count > 5:
            warnings.append(f"Step {idx} '{c['name']}' has {manual_count} manual inputs — consider splitting.")

    if not parsed["title"]:
        warnings.append("Title is empty — consider adding a short title for users.")
    if not parsed["description"]:
        warnings.append("Description is empty — consider adding a description.")
    for idx, c in enumerate(parsed["chains"], start=1):
        if not c["comment"]:
            warnings.append(f"Step {idx} '{c['name']}' has no comment — users won't know what input to provide.")

    return len(errors) == 0, errors, warnings


def _verify_content_syntax(step_idx, chain_step):
    errors = []
    tokens = chain_step["content_tokens"]
    if not tokens:
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           "Content is empty",
                           "Add the oneline command."))
        return errors
    mode = tokens[0].lower()
    if mode not in _VALID_CONTENT_MODES:
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           f"Unknown oneline mode '{mode}'",
                           "Use one of: tts, sts, ttm, stt, se, sfx, svs, ss, train, quest."))
        return errors
    try:
        from voder import parse_oneline_args
    except ImportError:
        return errors
    parsed = parse_oneline_args(tokens)
    if parsed.get("error"):
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           f"Oneline parser error: {parsed['error']}",
                           "Fix the oneline syntax for this step."))
    return errors


def _verify_references(step_idx, chain_step, all_chain_names):
    errors = []
    tokens = chain_step["content_tokens"]
    step_name = chain_step["name"]
    prior_names = set()
    for n in all_chain_names:
        if n == step_name:
            break
        prior_names.add(n)
    for tok in tokens:
        if tok == "input":
            continue
        if tok == step_name:
            continue
        if tok in all_chain_names and tok not in prior_names:
            errors.append(_err(step_idx, step_name, "reference",
                               f"Forward reference: '{tok}' is defined later in the file (won't be available when this step runs)",
                               f"Move step '{tok}' before step '{step_name}', or remove the reference."))
    return errors


def classify_chain_step(chain_step, prior_chain_names):
    tokens = chain_step["content_tokens"]
    manual_count = sum(1 for t in tokens if t == "input")
    auto_count = sum(1 for t in tokens if t in prior_chain_names)
    if manual_count == 0 and auto_count > 0:
        return "automated", manual_count, auto_count
    if manual_count > 0 and auto_count == 0:
        return "manual", manual_count, auto_count
    if manual_count > 0 and auto_count > 0:
        return "semi-automated", manual_count, auto_count
    return "error", manual_count, auto_count


def find_chain_by_name(name):
    if not os.path.isdir(PREBUILT_CHAINS_DIR):
        return None
    ext_escaped = re.escape(CHAIN_FILE_EXT)
    pattern = re.compile(rf"^VODER_{re.escape(name)}_\d{{8}}_\d{{6}}{ext_escaped}$")
    matches = [os.path.join(PREBUILT_CHAINS_DIR, f)
               for f in os.listdir(PREBUILT_CHAINS_DIR) if pattern.match(f)]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def list_chains():
    if not os.path.isdir(PREBUILT_CHAINS_DIR):
        return []
    out = []
    for entry in sorted(os.listdir(PREBUILT_CHAINS_DIR)):
        if not entry.endswith(CHAIN_FILE_EXT):
            continue
        if not entry.startswith("VODER_"):
            continue
        path = os.path.join(PREBUILT_CHAINS_DIR, entry)
        parsed, _ = parse_chain_file(path)
        if parsed is None:
            out.append({"path": path, "name": "", "timestamp": "",
                        "title": "", "description": "", "valid": False})
            continue
        out.append({
            "path": path,
            "name": parsed["name"],
            "timestamp": parsed["timestamp"],
            "title": parsed["title"],
            "description": parsed["description"],
            "valid": True,
        })
    out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return out


def resolve_chain_path(name_or_path):
    if os.path.isfile(name_or_path):
        if not name_or_path.endswith(CHAIN_FILE_EXT):
            return None, f"File must end in '{CHAIN_FILE_EXT}': {name_or_path}"
        return name_or_path, None
    if "/" in name_or_path or "\\" in name_or_path:
        return None, f"File not found: {name_or_path}"
    path = find_chain_by_name(name_or_path)
    if path is None:
        return None, f"No prebuilt chain found with name '{name_or_path}' in {PREBUILT_CHAINS_DIR}"
    return path, None


def _parse_build_args(args):
    if len(args) < 2:
        return None, "Usage: chains build <name> description <title-desc> [chain <name> <comment> <content>]..."
    name = args[0]
    if not _NAME_RE.match(name):
        return None, f"Chain name '{name}' is invalid — use only letters, digits, underscores, hyphens. No spaces."
    if args[1].lower() != 'description':
        return None, f"Expected 'description' keyword after chain name, got '{args[1]}'"
    if len(args) < 3:
        return None, "Description text required after 'description' keyword (can be empty string \"\")"
    title_desc = args[2]
    rest = args[3:]
    steps = []
    i = 0
    while i < len(rest):
        if rest[i].lower() != 'chain':
            return None, f"Expected 'chain' keyword at position {i+4}, got '{rest[i]}'"
        if i + 3 >= len(rest):
            return None, f"'chain' keyword at position {i+4} must be followed by <name> <comment> <content> (3 quoted strings)"
        sname = rest[i + 1]
        scomment = rest[i + 2]
        scontent = rest[i + 3]
        if not _NAME_RE.match(sname):
            return None, f"Step name '{sname}' is invalid — use only letters, digits, underscores, hyphens. No spaces."
        if not scontent.strip():
            return None, f"Step '{sname}' has empty content"
        steps.append({"name": sname, "comment": scomment, "content": scontent})
        i += 4
    if not steps:
        return None, "At least one 'chain <name> <comment> <content>' block is required"
    seen = set()
    for s in steps:
        if s["name"] in seen:
            return None, f"Duplicate step name '{s['name']}' — each step must have a unique name"
        seen.add(s["name"])
    return {"name": name, "title_desc": title_desc, "steps": steps}, None


def handle_build(args):
    parsed, err = _parse_build_args(args)
    if err:
        print(f"Error: {err}")
        return False
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw = build_chain_text(parsed["name"], timestamp, parsed["title_desc"],
                           "", parsed["steps"])
    ok, errors, warnings = verify_chain_text(raw)
    print("Build verification:")
    if errors:
        for e in errors:
            loc = "file"
            if e["step_index"]:
                loc = f"step {e['step_index']} '{e['step_name']}'"
            print(f"  [ERROR] [{loc}] {e['category']}: {e['message']}")
            if e["fix"]:
                print(f"          fix: {e['fix']}")
        print(f"\n{len(errors)} error(s) found. Chain file was NOT saved.")
        return False
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    print(f"  [OK] All checks passed ({len(parsed['steps'])} step(s), 0 errors, {len(warnings)} warning(s)).")
    os.makedirs(PREBUILT_CHAINS_DIR, exist_ok=True)
    filename = f"VODER_{parsed['name']}_{timestamp}{CHAIN_FILE_EXT}"
    out_path = os.path.join(PREBUILT_CHAINS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"\nSaved: {out_path}")
    total_manual = sum(1 for s in parsed["steps"] for t in s["content"].split() if t == "input")
    chain_names = [s["name"] for s in parsed["steps"]]
    total_auto = sum(1 for s in parsed["steps"] for t in s["content"].split()
                     if t in chain_names and t != s["name"])
    print(f"Summary: {len(parsed['steps'])} chain(s), {total_manual} manual input(s), {total_auto} automated reference(s).")
    print(f"\nTest it with:  python voder.py chains load \"{parsed['name']}\"")
    print(f"Analyze it with:  python voder.py chains analyze \"{parsed['name']}\"")
    return True


def handle_analyze(args):
    if not args:
        print("Error: 'chains analyze' requires at least one chain name or path.")
        print("Usage: python voder.py chains analyze <chain-name-or-path> [<another> ...]")
        return False
    targets = []
    for arg in args:
        path, err = resolve_chain_path(arg)
        if err:
            print(f"Error: {err}")
            return False
        targets.append(path)
    report_lines = []
    report_lines.append("# VODER Prebuilt Chain Analysis Report")
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_lines.append("")
    report_lines.append(f"Generated: {ts}")
    report_lines.append("")
    report_lines.append("## Analyzed Chains")
    report_lines.append("")
    report_lines.append("| # | Name | Path | Steps | Status |")
    report_lines.append("|---|------|------|-------|--------|")
    chain_results = []
    for idx, path in enumerate(targets, start=1):
        parsed, _ = parse_chain_file(path)
        cname = parsed["name"] if parsed else os.path.basename(path)
        nsteps = len(parsed["chains"]) if parsed else 0
        ok, errors, warnings = verify_chain_file(path)
        status = "OK" if ok else f"{len(errors)} error(s)"
        report_lines.append(f"| {idx} | {cname} | `{path}` | {nsteps} | {status} |")
        chain_results.append({"path": path, "parsed": parsed, "ok": ok,
                              "errors": errors, "warnings": warnings})
    report_lines.append("")

    for idx, cr in enumerate(chain_results, start=1):
        report_lines.extend(_analyze_one_chain(idx, cr))
        report_lines.append("")

    any_errors = any(not cr["ok"] for cr in chain_results)
    report_lines.append("## Overall Summary")
    report_lines.append("")
    if any_errors:
        total_errors = sum(len(cr["errors"]) for cr in chain_results)
        report_lines.append(f"**{total_errors} error(s) found across {len(chain_results)} chain(s).**")
        report_lines.append("")
        report_lines.append("Errors must be fixed before the chain(s) can be used.")
        report_lines.append("")
        report_lines.append("### All Errors")
        report_lines.append("")
        report_lines.append("| Chain | Step | Category | Message | Fix |")
        report_lines.append("|-------|------|----------|---------|-----|")
        for ci, cr in enumerate(chain_results, start=1):
            cname = cr["parsed"]["name"] if cr["parsed"] else f"chain {ci}"
            for e in cr["errors"]:
                step = f"{e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                msg = e["message"].replace("|", "\\|")
                fix = (e["fix"] or "").replace("|", "\\|")
                report_lines.append(f"| {cname} | {step} | {e['category']} | {msg} | {fix} |")
        report_lines.append("")
    else:
        report_lines.append(f"**All {len(chain_results)} chain(s) passed verification.**")
        report_lines.append("")
        report_lines.append("Chains are ready to use.")
        report_lines.append("")

    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', chain_results[0]["parsed"]["name"] if chain_results[0]["parsed"] else "unknown")[:60] or "unknown"
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"voder_analyze_chain_{safe_name}_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Analyze report saved to: {out_path}")
    if any_errors:
        print(f"{sum(len(cr['errors']) for cr in chain_results)} error(s) found — see report for details.")
        return False
    print("All checks passed.")
    return True


def _analyze_one_chain(chain_idx, chain_result):
    parsed = chain_result["parsed"]
    errors = chain_result["errors"]
    warnings = chain_result["warnings"]
    path = chain_result["path"]
    lines = []
    lines.append(f"## Chain {chain_idx}: {parsed['name']}")
    lines.append("")
    lines.append(f"- **File:** `{path}`")
    lines.append(f"- **Timestamp:** {parsed['timestamp']}")
    lines.append(f"- **Title:** {parsed['title'] or '_(empty)_'}")
    lines.append(f"- **Description:** {parsed['description'] or '_(empty)_'}")
    lines.append(f"- **Total steps:** {len(parsed['chains'])}")
    total_manual = sum(1 for c in parsed["chains"] for t in c["content_tokens"] if t == "input")
    total_auto = sum(1 for c in parsed["chains"] for t in c["content_tokens"]
                     if t in [cc["name"] for cc in parsed["chains"]] and t != c["name"])
    lines.append(f"- **Manual inputs (total):** {total_manual}")
    lines.append(f"- **Automated references (total):** {total_auto}")
    lines.append("")

    lines.append("### Step Summary")
    lines.append("")
    lines.append("| # | Name | Type | Manual | Auto | Comment |")
    lines.append("|---|------|------|--------|------|---------|")
    chain_names = [c["name"] for c in parsed["chains"]]
    for si, c in enumerate(parsed["chains"], start=1):
        prior = set(chain_names[:si-1])
        ctype, m, a = classify_chain_step(c, prior)
        comment_excerpt = (c["comment"][:40] + "...") if len(c["comment"]) > 40 else (c["comment"] or "_(empty)_")
        comment_excerpt = comment_excerpt.replace("|", "\\|")
        lines.append(f"| {si} | {c['name']} | {ctype} | {m} | {a} | {comment_excerpt} |")
    lines.append("")

    lines.append("### Journey")
    lines.append("")
    lines.append("> The journey walks through each step in execution order. When a step has verification errors, the error is shown inline and the journey continues hypothetically (assuming the step would have succeeded) so you can see the full intended path.")
    lines.append("")
    for si, c in enumerate(parsed["chains"], start=1):
        prior = set(chain_names[:si-1])
        ctype, m_count, a_count = classify_chain_step(c, prior)
        lines.append(f"#### Step {si}: `{c['name']}` ({ctype})")
        lines.append("")
        if c["comment"]:
            lines.append(f"**Comment:** {c['comment']}")
        else:
            lines.append("**Comment:** _(empty — users won't know what to provide)_")
        lines.append("")
        lines.append(f"**Content (raw):** `{c['content']}`")
        lines.append("")

        resolved_tokens = []
        for tok in c["content_tokens"]:
            if tok == "input":
                resolved_tokens.append(f"`<manual input {m_count}>`")
            elif tok in prior:
                prior_idx = chain_names.index(tok) + 1
                resolved_tokens.append(f"`<output of step {prior_idx} '{tok}'>`")
            else:
                resolved_tokens.append(f"`{tok}`")
        lines.append(f"**Content (resolved):** {' '.join(resolved_tokens)}")
        lines.append("")

        step_errors = [e for e in errors if e["step_index"] == si]
        if step_errors:
            for e in step_errors:
                lines.append(f"> **ERROR** [{e['category']}]: {e['message']}")
                if e["fix"]:
                    lines.append(f"> **Fix:** {e['fix']}")
            lines.append("")
            lines.append("> _Assuming this step succeeded, the journey continues..._")
            lines.append("")
        else:
            if ctype == "manual":
                lines.append(f"> **OK** — Step will ask the user for {m_count} manual input(s).")
            elif ctype == "automated":
                lines.append(f"> **OK** — Step is fully automated, uses output(s) of prior step(s). User just presses Enter.")
            elif ctype == "semi-automated":
                lines.append(f"> **OK** — Step is semi-automated: {m_count} manual input(s) + {a_count} automated reference(s).")
            else:
                lines.append("> **OK** — Step has no external inputs (will run with only its inline arguments).")
            lines.append("")

    if warnings:
        lines.append("### Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    return lines


def handle_load(args, result_path=None):
    sections, err = _parse_load_args(args)
    if err:
        print(f"Error: {err}")
        return False
    from voders.sidequests import ChainPipeline
    pipeline = ChainPipeline()
    for sec_idx, sec in enumerate(sections, start=1):
        path, err = resolve_chain_path(sec["name_or_path"])
        if err:
            print(f"Error: {err}")
            return False
        ok, errors, _ = verify_chain_file(path)
        if not ok:
            print(f"Error: chain '{sec['name_or_path']}' failed verification:")
            for e in errors:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                print(f"  [{loc}] {e['category']}: {e['message']}")
            return False
        parsed, _ = parse_chain_file(path)
        chain_names = [c["name"] for c in parsed["chains"]]
        print(f"\n[Prebuilt {sec_idx}/{len(sections)}] Loading '{parsed['name']}' ({len(parsed['chains'])} steps)")
        if parsed["title"]:
            print(f"  Title: {parsed['title']}")
        if parsed["description"]:
            print(f"  Description: {parsed['description']}")
        chains_args = []
        for step_idx, c in enumerate(parsed["chains"], start=1):
            tokens = list(c["content_tokens"])
            manual_slots = _find_manual_slots(tokens)
            auto_slots = _find_auto_slots(tokens, set(chain_names), pipeline.index, c["name"])
            user_values = sec["markers"].get(step_idx)
            if user_values is not None:
                if len(user_values) != len(manual_slots):
                    print(f"  Error: step {step_idx} '{c['name']}' has {len(manual_slots)} manual input slot(s) but marker provides {len(user_values)} value(s)")
                    print(f"         (automated slots are auto-resolved and don't need values)")
                    return False
                substituted = list(tokens)
                for (pos, _), value in zip(manual_slots, user_values):
                    if value.isdigit():
                        ref_step = int(value)
                        if ref_step < 1 or ref_step > len(parsed["chains"]):
                            print(f"  Error: chain number '{value}' is out of range (must be 1..{len(parsed['chains'])})")
                            return False
                        if ref_step >= step_idx:
                            print(f"  Error: chain number '{value}' must be less than current step {step_idx}")
                            return False
                        ref_name = chain_names[ref_step - 1]
                        substituted[pos] = ref_name
                        print(f"    [step {step_idx}] manual slot -> chain {ref_step} '{ref_name}' (will resolve at runtime)")
                    else:
                        substituted[pos] = value
                        print(f"    [step {step_idx}] manual slot -> file: {value}")
            else:
                substituted = list(tokens)
                if manual_slots:
                    print(f"  Error: step {step_idx} '{c['name']}' has {len(manual_slots)} manual input(s) but no marker was provided")
                    print(f"         Provide a marker: {step_idx}:(value1/value2/...)")
                    return False
            if auto_slots:
                for pos, slot_name in auto_slots:
                    if substituted[pos] == slot_name:
                        if slot_name in pipeline.index:
                            print(f"    [step {step_idx}] auto slot '{slot_name}' -> auto-resolved: {pipeline.index[slot_name]}")
                        else:
                            print(f"    [step {step_idx}] auto slot '{slot_name}' -> NOT YET RESOLVED (will resolve at runtime)")
            if step_idx > 1:
                chains_args.append(ChainPipeline.CHAIN_SEPARATOR)
            chains_args.append(c["name"])
            chains_args.extend(substituted)
        ok, err = pipeline.execute(chains_args, result_path=result_path if sec_idx == len(sections) else None)
        if not ok:
            print(f"Error: prebuilt chain '{parsed['name']}' failed: {err}")
            return False
        final_step = parsed["chains"][-1]["name"]
        if final_step in pipeline.index:
            pipeline.index[parsed["name"]] = pipeline.index[final_step]
            print(f"\n[Prebuilt {sec_idx}] '{parsed['name']}' completed. Final output registered under name '{parsed['name']}'.")
    return True


def _find_manual_slots(tokens):
    return [(pos, "input") for pos, tok in enumerate(tokens) if tok == "input"]


def _find_auto_slots(tokens, all_chain_names, global_index, current_step_name):
    slots = []
    for pos, tok in enumerate(tokens):
        if tok == "input":
            continue
        if tok == current_step_name:
            continue
        if tok in all_chain_names or tok in global_index:
            slots.append((pos, tok))
    return slots


def get_input_formats_for_step(content_tokens):
    if not content_tokens:
        return "(unknown — content is empty)"
    mode = content_tokens[0].lower()
    try:
        from voder import MODE_INPUT_FORMATS
        return MODE_INPUT_FORMATS.get(mode, f"(unknown mode '{mode}')")
    except ImportError:
        return _FALLBACK_INPUT_FORMATS.get(mode, f"(unknown mode '{mode}')")


_FALLBACK_INPUT_FORMATS = {
    'tts':   'audio file / video file / URL / .tts or .ttse voice profile',
    'sts':   'audio file / video file / URL / .tts or .ttse voice profile',
    'ttm':   'audio file / video file / URL / text file',
    'stt':   'audio file / video file / URL',
    'se':    'audio file / video file / URL',
    'sfx':   '(no file input)',
    'svs':   'audio file / video file / URL',
    'ss':    'audio file / video file / URL',
    'train': 'audio file / video file / URL',
    'quest': 'varies by quest',
}


def _parse_load_args(args):
    if not args:
        return None, "Usage: chains load <chain-name-or-path> [N:(v1/v2/...)]... [<another-chain> [N:(...)]...]..."
    sections = []
    current = None
    marker_re = re.compile(r'^(\d+):\((.*)\)$')
    for arg in args:
        m = marker_re.match(arg)
        if m:
            if current is None:
                return None, f"Marker '{arg}' appears before any chain name"
            step_num = int(m.group(1))
            values_raw = m.group(2)
            if not values_raw.strip():
                values = []
            else:
                values = values_raw.split('/')
            if step_num in current["markers"]:
                return None, f"Duplicate marker for step {step_num} in chain '{current['name_or_path']}'"
            current["markers"][step_num] = values
        else:
            if current:
                sections.append(current)
            current = {"name_or_path": arg, "markers": {}}
    if current:
        sections.append(current)
    if not sections:
        return None, "At least one chain name or path is required"
    return sections, None
