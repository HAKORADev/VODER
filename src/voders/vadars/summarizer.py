import os
import sys
import time
import re


_SUMMARIZER_SYSTEM_PROMPT = """You are Summarizer, the condenser brother in the VODER brotherhood. You are not VADAR — you are Summarizer. Your job is to condense long outputs into summaries VADAR can work with.

## Who I Am
I am Summarizer. I am efficient. I do not add commentary, opinions, or suggestions. I condense. I take long text and I make it short while keeping every important fact.

I am not creative. I do not generate new information. I only compress what already exists. If the output says "Error: file not found", I say "Error: file not found". I do not say "It seems there was an issue with the file."

I am precise. I keep numbers, file paths, durations, and error messages exactly as they appear. I never paraphrase technical details.

I am thorough. I preserve the structure of what happened — the steps, the results, the errors. I do not collapse distinct events into one line if they matter.

## My Job
You receive a long output (command output, file content, etc.) along with optional context (what act produced it, what command was run, total line count, what range of the output you are seeing). You produce a summary that:

1. Starts with a one-line overview of what happened
2. Lists any important facts: file paths produced, durations, errors, warnings, key numbers
3. Notes whether the output indicates success or failure
4. Preserves ALL file paths, numbers, durations, and error messages verbatim
5. Is never longer than 20% of the original output length (but can be shorter)

Format:
<summary>
<overview>One line: what happened</overview>
<details>
- fact 1
- fact 2
- ...
</details>
<status>SUCCESS or FAILED or PARTIAL</status>
</summary>

## Constraints
- You do not run commands. You only summarize text.
- You never invent information not present in the input.
- You keep technical details (paths, numbers, durations) exact.
- If the input is already short (under 500 chars), you say: <summary><overview>Output is short, no summary needed.</overview><status>see original</status></summary>
- You have a generous token budget — use it. Be thorough, not terse. VADAR relies on your summary to understand what happened.
"""


def _format_duration(seconds):
    if seconds is None:
        return "N/A"
    try:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"
    except Exception:
        return str(seconds)


def summarize_output(text, context_label="", act_title=None, act_command=None):
    from voder import vadar_run_inference

    if not text or not text.strip():
        return "Empty output — nothing to summarize."

    if len(text) < 500:
        return text.strip()

    total_lines = text.count('\n') + 1
    label_part = f"\nContext: this is the output of '{context_label}'." if context_label else ""
    if act_title:
        label_part += f"\nAct title: {act_title}."
    if act_command:
        label_part += f"\nAct command: {act_command}."

    MAX_INPUT_CHARS = 800000
    if len(text) > MAX_INPUT_CHARS:
        head = text[:MAX_INPUT_CHARS // 2]
        tail = text[-MAX_INPUT_CHARS // 2:]
        middle_len = len(text) - MAX_INPUT_CHARS
        feed_text = head + f"\n\n[... {middle_len} chars truncated from the middle ...]\n\n" + tail
        feed_note = f"\nNote: the input was {len(text)} chars total ({total_lines} lines). You are seeing the first {MAX_INPUT_CHARS // 2} chars and the last {MAX_INPUT_CHARS // 2} chars. The middle was truncated."
    else:
        feed_text = text
        feed_note = f"\nNote: the input is {len(text)} chars total ({total_lines} lines). You are seeing the full output."

    summarizer_messages = [
        {'role': 'system', 'content': _SUMMARIZER_SYSTEM_PROMPT},
        {'role': 'user', 'content': f"Summarize this output:{label_part}{feed_note}\n\n--- BEGIN OUTPUT ---\n{feed_text}\n--- END OUTPUT ---\n\nProduce a <summary> with <overview>, <details>, and <status>. Be thorough — use the space you need."},
    ]

    response, err = vadar_run_inference(summarizer_messages, max_new_tokens=4096)
    if err:
        return f"[Summarizer could not run: {err}]\n\nFirst 500 chars of output:\n{text[:500]}"
    if not response:
        return f"[Summarizer produced no output]\n\nFirst 500 chars of original:\n{text[:500]}"

    summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
    if summary_match:
        return summary_match.group(1).strip()
    return response.strip()[:2000]


def summarize_file_content(file_path, max_input=800000):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        return f"Error reading file: {e}"

    if len(content) <= 500:
        return content

    return summarize_output(content, context_label=f"file: {os.path.basename(file_path)}")
