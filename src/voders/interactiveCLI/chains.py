import os
import sys
import time
import shutil

from voder import (
    parse_and_execute_oneline,
    validate_file_exists,
    is_youtube_url,
    VIDEO_EXTENSIONS,
    VOICE_PROFILE_EXTENSIONS,
)

from voders.prebuilt_chains import (
    parse_chain_file,
    verify_chain_file,
    classify_chain_step,
    find_chain_by_name,
    list_chains,
    resolve_chain_path,
    get_input_formats_for_step,
    PREBUILT_CHAINS_DIR,
    CHAIN_FILE_EXT,
)
from voders.sidequests import ChainPipeline


def _print_separator(title=None):
    print("=" * 60)
    if title:
        print(title)
        print("=" * 60)


def _select_chain_by_list():
    chains = list_chains()
    if not chains:
        print(f"\nNo prebuilt chains found in: {PREBUILT_CHAINS_DIR}")
        print(f"Build one with:  python voder.py chains build <name> description \"...\" chain ...")
        return None
    print("\nAvailable prebuilt chains:")
    print("-" * 60)
    for idx, c in enumerate(chains, start=1):
        if not c["valid"]:
            print(f"  {idx}. [INVALID FILE] {os.path.basename(c['path'])}")
            continue
        ts_display = ""
        if c["timestamp"]:
            ts_display = f"  ({c['timestamp'][:4]}-{c['timestamp'][4:6]}-{c['timestamp'][6:8]} {c['timestamp'][9:11]}:{c['timestamp'][11:13]})"
        title_display = c["title"] or "(no title)"
        print(f"  {idx}. {c['name']}{ts_display}")
        print(f"      {title_display}")
    print("-" * 60)
    while True:
        choice = input("\nEnter number to load, or 'back' to return: ").strip()
        if choice.lower() == 'back':
            return None
        if not choice.isdigit():
            print("Invalid input. Enter a number or 'back'.")
            continue
        n = int(choice)
        if n < 1 or n > len(chains):
            print(f"Number out of range. Enter 1-{len(chains)} or 'back'.")
            continue
        return chains[n - 1]["path"]


def _select_chain_by_name():
    while True:
        choice = input("\nEnter chain name (latest by timestamp) or full path (.chain file):\n> ").strip()
        if not choice:
            print("Empty input. Please enter a name or path.")
            continue
        if choice.lower() == 'back':
            return None
        path, err = resolve_chain_path(choice)
        if err:
            print(f"Warning: {err}")
            print("Try again, or type 'back' to return.")
            continue
        return path


def _select_chain():
    print("\n--- Prebuilt Chains Mode ---")
    print("Load and run a saved chain file, or chain multiple prebuilts together.")
    print()
    print("1. List available chains (choose by number)")
    print("2. Enter chain name or path directly")
    print("3. Back to main menu")
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == '1':
            path = _select_chain_by_list()
            if path is None:
                continue
            return path
        if choice == '2':
            path = _select_chain_by_name()
            if path is None:
                continue
            return path
        if choice == '3':
            return None
        print("Invalid choice. Please enter 1, 2, or 3.")


def _select_multiple_chains():
    selected = []
    while True:
        if not selected:
            print("\n--- Select first prebuilt chain ---")
        else:
            print(f"\n--- {len(selected)} chain(s) selected so far ---")
            for i, p in enumerate(selected, start=1):
                parsed, _ = parse_chain_file(p)
                if parsed:
                    print(f"  {i}. {parsed['name']} — {parsed['title'] or '(no title)'}")
                else:
                    print(f"  {i}. [INVALID] {p}")
            print("\nAdd another prebuilt chain? (subsequent chains can reference")
            print("prior prebuilt names to receive their final output)")
        path = _select_chain()
        if path is None:
            if selected:
                while True:
                    proceed = input("\nProceed with selected chain(s)? (y/n): ").strip().lower()
                    if proceed in ('y', 'yes'):
                        return selected
                    if proceed in ('n', 'no'):
                        return None if not selected else None
                    print("Please enter 'y' or 'n'.")
                continue
            else:
                return None
        selected.append(path)
        while True:
            more = input("\nAdd another chain? (y/n): ").strip().lower()
            if more in ('y', 'yes'):
                break
            if more in ('n', 'no'):
                return selected
            print("Please enter 'y' or 'n'.")


def _validate_input_file(value, content_tokens):
    if value.isdigit():
        return True, None
    if os.path.isfile(value):
        return True, None
    if is_youtube_url(value):
        return True, None
    return False, "File not found and not a supported URL. Please enter a valid file path or URL."


def _gather_inputs_for_chain(parsed, pipeline, prebuilt_idx, total_prebuilts):
    chain_names = [c["name"] for c in parsed["chains"]]
    total_steps = len(parsed["chains"])
    total_manual = sum(1 for c in parsed["chains"]
                       for t in c["content_tokens"] if t == "input")
    gathered = {}
    manual_gathered_count = 0
    for step_idx, c in enumerate(parsed["chains"], start=1):
        step_name = c["name"]
        tokens = c["content_tokens"]
        prior_names = set(chain_names[:step_idx-1]) | set(pipeline.index.keys())
        ctype, m_count, a_count = classify_chain_step(c, prior_names)
        _print_separator(f"Prebuilt {prebuilt_idx}/{total_prebuilts} ({parsed['name']}) — "
                         f"Step {step_idx}/{total_steps} ({step_name}) — {ctype}")
        if c["comment"]:
            print(f"Comment: {c['comment']}")
        else:
            print("Comment: (none provided — see chain content for context)")
        print(f"Content: {c['content']}")
        if ctype == "automated":
            ref_descs = []
            for tok in tokens:
                if tok in prior_names:
                    ref_descs.append(f"'{tok}'")
            if ref_descs:
                print(f"\nThis step is fully automated. It uses output(s) from: {', '.join(ref_descs)}.")
            else:
                print("\nThis step has no external inputs (runs with inline arguments only).")
            print("Press Enter to continue to the next step.")
            input()
            gathered[step_idx] = []
            continue
        if ctype == "semi-automated":
            ref_descs = []
            for tok in tokens:
                if tok in prior_names:
                    ref_descs.append(f"'{tok}'")
            print(f"\nThis step is semi-automated. It auto-uses output(s) from: {', '.join(ref_descs)}.")
            print("You also need to provide manual input(s) below.")
        else:
            print(f"\nThis step requires {m_count} manual input(s).")
        formats = get_input_formats_for_step(tokens)
        print(f"Valid inputs: {formats}")
        manual_slots = [(pos, tok) for pos, tok in enumerate(tokens) if tok == "input"]
        step_inputs = []
        for slot_idx, (pos, _) in enumerate(manual_slots, start=1):
            manual_gathered_count += 1
            overall_pct = int(100 * manual_gathered_count / max(1, total_manual))
            print(f"\n  [Input {slot_idx}/{len(manual_slots)} for step '{step_name}' "
                  f"— overall {manual_gathered_count}/{total_manual} ({overall_pct}%)]")
            while True:
                value = input("  > ").strip()
                if not value:
                    print("  Empty input. Please enter a value (or a chain number to use that chain's output).")
                    continue
                ok, err = _validate_input_file(value, tokens)
                if not ok:
                    print(f"  Warning: {err}")
                    continue
                if value.isdigit():
                    ref_step = int(value)
                    if ref_step < 1 or ref_step > total_steps:
                        print(f"  Warning: chain number {ref_step} out of range (1-{total_steps}). Try again.")
                        continue
                    if ref_step >= step_idx:
                        print(f"  Warning: chain number must be less than current step {step_idx}. Try again.")
                        continue
                    ref_name = chain_names[ref_step - 1]
                    print(f"  OK — will use output of chain {ref_step} '{ref_name}'.")
                else:
                    print(f"  OK — using: {value}")
                step_inputs.append(value)
                break
        gathered[step_idx] = step_inputs
    return gathered


def _execute_prebuilt(parsed, gathered, pipeline, prebuilt_idx, total_prebuilts):
    chain_names = [c["name"] for c in parsed["chains"]]
    total_steps = len(parsed["chains"])
    print()
    _print_separator(f"Ready to run: Prebuilt {prebuilt_idx}/{total_prebuilts} '{parsed['name']}' ({total_steps} steps)")
    print("Press Enter to start execution.")
    input()
    chains_args = []
    for step_idx, c in enumerate(parsed["chains"], start=1):
        tokens = list(c["content_tokens"])
        manual_slots = [(pos, tok) for pos, tok in enumerate(tokens) if tok == "input"]
        substituted = list(tokens)
        step_inputs = gathered.get(step_idx, [])
        for (pos, _), value in zip(manual_slots, step_inputs):
            if value.isdigit():
                ref_step = int(value)
                ref_name = chain_names[ref_step - 1]
                substituted[pos] = ref_name
            else:
                substituted[pos] = value
        if step_idx > 1:
            chains_args.append(ChainPipeline.CHAIN_SEPARATOR)
        chains_args.append(c["name"])
        chains_args.extend(substituted)
    is_last_prebuilt = (prebuilt_idx == total_prebuilts)
    try:
        ok, err = pipeline.execute(chains_args, result_path=None)
    except Exception as e:
        err_msg = str(e)[:500]
        print()
        print("=" * 60)
        print("Something went further than expected.")
        print(f"Error (at prebuilt {prebuilt_idx} '{parsed['name']}'): {err_msg}")
        print("=" * 60)
        return False
    if not ok:
        err_msg = (err or "unknown error")[:500]
        print()
        print("=" * 60)
        print("Something went further than expected.")
        print(f"Error (at prebuilt {prebuilt_idx} '{parsed['name']}'): {err_msg}")
        print("=" * 60)
        return False
    final_step = parsed["chains"][-1]["name"]
    if final_step in pipeline.index:
        pipeline.index[parsed["name"]] = pipeline.index[final_step]
    return True


def cli_chains_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- Prebuilt Chains Mode ---")
    print("Load a saved chain file and run it with interactive input gathering.")
    print("Multi-chain supported: each subsequent chain can reference prior chains by name.")

    selected_paths = _select_multiple_chains()
    if not selected_paths:
        print("No chains selected. Returning to main menu.")
        return False

    print()
    _print_separator(f"Selected {len(selected_paths)} prebuilt chain(s)")
    for i, p in enumerate(selected_paths, start=1):
        parsed, _ = parse_chain_file(p)
        if parsed:
            print(f"  {i}. {parsed['name']} — {parsed['title'] or '(no title)'}")
            print(f"     Path: {p}")
            print(f"     Steps: {len(parsed['chains'])}")
        else:
            print(f"  {i}. [INVALID] {p}")
    print()

    pipeline = ChainPipeline()
    all_gathered = []
    for sec_idx, path in enumerate(selected_paths, start=1):
        parsed, _ = parse_chain_file(path)
        if parsed is None:
            print(f"Error: could not parse chain file: {path}")
            return False
        ok, errors, warnings = verify_chain_file(path)
        if not ok:
            print(f"\nVerification failed for chain '{parsed['name']}':")
            for e in errors:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                print(f"  [{loc}] {e['category']}: {e['message']}")
            print("\nThis chain file has verification errors and cannot be run.")
            return False
        if warnings:
            print(f"\nWarnings for chain '{parsed['name']}':")
            for w in warnings:
                print(f"  - {w}")
        print(f"\n--- Loading Prebuilt {sec_idx}/{len(selected_paths)}: '{parsed['name']}' ---")
        if parsed["title"]:
            print(f"Title: {parsed['title']}")
        if parsed["description"]:
            print(f"Description: {parsed['description']}")
        gathered = _gather_inputs_for_chain(parsed, pipeline, sec_idx, len(selected_paths))
        all_gathered.append((parsed, gathered))

    for sec_idx, (parsed, gathered) in enumerate(all_gathered, start=1):
        ok = _execute_prebuilt(parsed, gathered, pipeline, sec_idx, len(all_gathered))
        if not ok:
            return False

    print()
    _print_separator("All prebuilt chains completed!")
    final_name = all_gathered[-1][0]["name"]
    if final_name in pipeline.index:
        print(f"Final output: {pipeline.index[final_name]}")
    else:
        print("Final output: (see results/ directory for the most recent file)")
    print(f"\nResults directory: {results_dir}")
    return True
