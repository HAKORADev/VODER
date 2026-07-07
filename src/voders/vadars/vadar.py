import os
import sys
import re
import time
import threading
import traceback

from voders.vadars.eval import evaluate_plan, evaluate_act_result
from voders.vadars.summarizer import summarize_output

from voders.vadars import VADAR_SESSIONS_DIR
from voders.vadars.system_prompt import generate_system_prompt
from voders.vadars.context import ContextManager, create_session, log_input, log_output, log_act
from voders.vadars.tools import TOOL_REGISTRY
from voders.vadars.tools.validator import catch_and_fix


TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>', re.DOTALL)
ACT_RE = re.compile(r'<act>\s*(\S+)\s+(.*?)\s*</act>', re.DOTALL)
REPLY_RE = re.compile(r'<reply>(.*?)</reply>', re.DOTALL)
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
DECIDE_RE = re.compile(r'<decide>(.*?)</decide>', re.DOTALL)
EVAL_RE = re.compile(r'<eval>(.*?)</eval>', re.DOTALL)

EOS_REPLY = '<EOS_REPLY>'
EOS_ACT = '<EOS_ACT>'
EOS_DONE = '<EOS_DONE>'

_RESULTS_DIR = os.path.join(os.getcwd(), 'results')


def _parse_model_output(text):
    result = {
        'thoughts': [],
        'decisions': [],
        'replies': [],
        'acts': [],
        'tool_calls': [],
        'evals': [],
        'eos_reply': False,
        'eos_act': False,
        'eos_done': False,
        'raw': text,
    }
    for m in THINK_RE.finditer(text):
        result['thoughts'].append(m.group(1).strip())
    for m in DECIDE_RE.finditer(text):
        result['decisions'].append(m.group(1).strip())
    for m in REPLY_RE.finditer(text):
        result['replies'].append(m.group(1).strip())
    for m in ACT_RE.finditer(text):
        result['acts'].append({'title': m.group(1).strip(), 'command': m.group(2).strip()})
    for m in TOOL_CALL_RE.finditer(text):
        result['tool_calls'].append({'tool': m.group(1).strip(), 'args': m.group(2).strip()})
    for m in EVAL_RE.finditer(text):
        result['evals'].append(m.group(1).strip())
    if EOS_REPLY in text:
        result['eos_reply'] = True
    if EOS_ACT in text:
        result['eos_act'] = True
    if EOS_DONE in text:
        result['eos_done'] = True
    return result


def _snapshot_results():
    snap = {}
    if not os.path.isdir(_RESULTS_DIR):
        return snap
    for f in os.listdir(_RESULTS_DIR):
        p = os.path.join(_RESULTS_DIR, f)
        if os.path.isfile(p):
            try:
                snap[f] = os.path.getmtime(p)
            except Exception:
                pass
    return snap


def _new_result_files(before_snap):
    after_snap = _snapshot_results()
    new_files = []
    for f, mtime in after_snap.items():
        if f not in before_snap:
            new_files.append(os.path.join(_RESULTS_DIR, f))
        elif mtime > before_snap[f]:
            new_files.append(os.path.join(_RESULTS_DIR, f))
    return new_files


def _execute_tool_call(tool_name, tool_args, session_dir=None, act_outputs=None, model=None, processor=None):
    tool = TOOL_REGISTRY.get(tool_name)
    if tool is None:
        return False, f"Unknown tool: {tool_name}", 0
    t0 = time.time()
    try:
        if tool_name in ('look', 'listen', 'watch'):
            result = tool(tool_args, model=model, processor=processor)
        elif tool_name == 'read':
            result = tool(tool_args, session_dir=session_dir, act_outputs=act_outputs)
        else:
            result = tool(tool_args)
        elapsed = time.time() - t0
        return True, result, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        return False, f"Tool '{tool_name}' error: {e}", elapsed


def _execute_act(title, command, session_dir, act_outputs, user_request="",
                 summarize_threshold=1500, used_titles=None):
    if used_titles is not None and title in used_titles:
        return False, f"Act title '{title}' already exists in this session. Use a unique title."
    if used_titles is not None:
        used_titles.add(title)

    from voder import parse_and_execute_oneline
    cmd_tokens = command.split()
    if not cmd_tokens:
        return False, "Empty command"

    results_before = _snapshot_results()

    try:
        old_stdout = sys.stdout
        captured = []
        class _Capture:
            def write(self, text):
                captured.append(text)
                old_stdout.write(text)
            def flush(self):
                old_stdout.flush()
        sys.stdout = _Capture()
        success = parse_and_execute_oneline(cmd_tokens)
        sys.stdout = old_stdout
        output = ''.join(captured)
        act_outputs[title] = output
        log_act(session_dir, title, command, output, success)

        new_files = _new_result_files(results_before)
        if new_files:
            output += f"\n[RESULT FILES]: {', '.join(new_files)}"
        elif success:
            output += "\n[WARNING]: Command reported success but no new result files were found in results/."

        if user_request:
            verdict, reason = evaluate_act_result(user_request, title, command, output, success)
            print(f"[EVAL]: {verdict.upper()} — {reason}")
            log_act(session_dir, f"{title}_eval", f"eval verdict: {verdict}", reason, verdict == 'correct')

        if len(output) > summarize_threshold:
            summary = summarize_output(output, context_label=title)
            print(f"[SUMMARIZER]: condensed {len(output)} chars -> {len(summary)} chars")
            return success, summary

        return success, output
    except Exception as e:
        sys.stdout = old_stdout
        act_outputs[title] = str(e)
        log_act(session_dir, title, command, str(e), False)
        return False, str(e)


def _run_inference_streamed(messages, max_new_tokens=1024):
    from voder import vadar_run_inference_streamed
    return vadar_run_inference_streamed(messages, max_new_tokens=max_new_tokens)


def _run_inference(messages, max_new_tokens=1024):
    from voder import vadar_run_inference
    return vadar_run_inference(messages, max_new_tokens=max_new_tokens)


def _detect_inputs(text):
    url_re = re.compile(r'https?://\S+')
    path_re = re.compile(r'[\w/\\\-\.]+\.(?:wav|mp3|flac|ogg|aac|m4a|mp4|avi|mov|mkv|flv|webm|png|jpg|jpeg|gif|bmp|tiff)', re.IGNORECASE)
    inputs = []
    for m in url_re.finditer(text):
        inputs.append(m.group(0))
    for m in path_re.finditer(text):
        p = m.group(0)
        if os.path.exists(p) and p not in inputs:
            inputs.append(p)
    return inputs


def _auto_hear_inputs(inputs, session_dir, act_outputs, model, processor):
    for inp in inputs:
        ext = os.path.splitext(inp)[1].lower()
        if ext in {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a'}:
            print(f"[AUTO-HEAR]: listening to {inp}")
            result = _execute_tool_call('listen', inp, session_dir, act_outputs, model, processor)
            print(f"[AUTO-HEAR RESULT]: {str(result[1])[:300]}")
        elif ext in {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm'}:
            print(f"[AUTO-HEAR]: watching {inp}")
            result = _execute_tool_call('watch', inp, session_dir, act_outputs, model, processor)
            print(f"[AUTO-HEAR RESULT]: {str(result[1])[:300]}")
        elif ext in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}:
            print(f"[AUTO-HEAR]: looking at {inp}")
            result = _execute_tool_call('look', inp, session_dir, act_outputs, model, processor)
            print(f"[AUTO-HEAR RESULT]: {str(result[1])[:300]}")


def _process_tool_calls(tool_calls, ctx, session_dir, act_outputs, model, processor):
    for tc in tool_calls:
        tool_name = tc['tool']
        tool_args = tc['args']
        print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")

        ok, err, fixed_args, _ = catch_and_fix(tool_name, tool_args)
        if not ok:
            print(f"[CATCHER]: {err}")
            print(f"[TOOL_RESULT]: SKIPPED — retry in next iteration")
            ctx.add('tool', f"Tool '{tool_name}' was invalid: {err}. Fix the call and retry.")
            continue

        success, result, elapsed = _execute_tool_call(
            tool_name, fixed_args, session_dir, act_outputs, model, processor
        )
        display = result[:800] if len(result) > 800 else result
        print(f"[TOOL_RESULT]: {display}")
        print(f"[TOOL_STATS]: {tool_name} | {'OK' if success else 'FAILED'} | {elapsed:.2f}s")
        ctx.add('tool', f"Tool '{tool_name}' result:\n{result}")


def _run_agent_loop(ctx, user_input, session_dir, act_outputs, model, processor,
                    used_titles, interactive=False, approval_event=None):
    max_iterations = 30 if interactive else 20
    last_vadar_reply_time = time.time()

    detected = _detect_inputs(user_input)
    if detected:
        _auto_hear_inputs(detected, session_dir, act_outputs, model, processor)
        ctx.add('system', f"I have automatically listened to/watched/looked at the inputs you mentioned: {', '.join(detected)}. Use what you learned to plan your act.")

    for iteration in range(max_iterations):
        messages = ctx.get_for_inference()
        response, err = _run_inference(messages)
        if err:
            print(f"\n[VADAR inference error]: {err}")
            return False
        if not response or not response.strip():
            break

        ctx.add('assistant', response)
        parsed = _parse_model_output(response)

        if parsed['thoughts'] and parsed['decisions'] and not parsed['acts']:
            thoughts_text = ' '.join(parsed['thoughts'])
            decisions_text = ' '.join(parsed['decisions'])
            verdict, reason = evaluate_plan(user_input, thoughts_text, decisions_text)
            print(f"[EVAL]: {verdict.upper()} — {reason}")
            if verdict == 'wrong':
                ctx.add('system', f"Eval says your plan is WRONG: {reason}. Fix it and try again.")
                continue

        if parsed['replies'] and not parsed['acts'] and not parsed['tool_calls']:
            for reply in parsed['replies']:
                print(f"\n[VADAR]: {reply}")
                log_output(session_dir, reply)
                last_vadar_reply_time = time.time()

            if interactive and parsed['decisions'] and approval_event is not None:
                print("\n[VADAR]: This is my plan. Type 'go' to approve, or tell me what to change.")
                approval_event.clear()
                while not approval_event.is_set():
                    approval_event.wait(timeout=0.5)
                approval_str = getattr(approval_event, '_user_response', 'go')
                if approval_str.lower().strip() in ('go', 'ok', 'yes', 'proceed', 'do it', 'continue', ''):
                    ctx.add('user', f"go")
                else:
                    ctx.add('user', approval_str)
                    continue

        if parsed['tool_calls']:
            _process_tool_calls(
                parsed['tool_calls'], ctx, session_dir, act_outputs, model, processor
            )

        for act in parsed['acts']:
            title = act['title']
            command = act['command']
            print(f"\n[ACT]: {title} -> {command}")
            print(f"{'─'*40}")
            success, output = _execute_act(
                title, command, session_dir, act_outputs,
                user_request=user_input, used_titles=used_titles
            )
            print(f"{'─'*40}")
            status = 'SUCCESS' if success else 'FAILED'
            print(f"[ACT RESULT]: {title} -> {status}")
            ctx.add('tool', f"Act '{title}' result ({status}):\n{output[-2000:]}")

        if parsed['eos_done']:
            print("\n[VADAR]: Task complete.")
            return True

        if not parsed['replies'] and not parsed['acts'] and not parsed['tool_calls']:
            if parsed['raw'].strip():
                print(f"\n[VADAR]: {parsed['raw'].strip()[:500]}")
                log_output(session_dir, parsed['raw'].strip()[:500])
                last_vadar_reply_time = time.time()
            break

        if parsed['eos_reply'] and not parsed['acts'] and not parsed['tool_calls']:
            break

    return True


def run_vadar_oneline(user_input, result_path=None):
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        return False

    tasks = re.split(r'\s*&&\s*', user_input)
    if len(tasks) > 1:
        print(f"[VADAR]: {len(tasks)} tasks detected (split by &&).")
        all_ok = True
        for i, task in enumerate(tasks, 1):
            task = task.strip()
            if not task:
                continue
            print(f"\n{'='*40}")
            print(f"Task {i}/{len(tasks)}: {task[:100]}")
            print(f"{'='*40}")
            ok = _run_oneline_single(task, result_path if i == len(tasks) else None,
                                     model, processor)
            if not ok:
                all_ok = False
        return all_ok

    return _run_oneline_single(user_input, result_path, model, processor)


def _run_oneline_single(user_input, result_path, model, processor):
    session_dir, session_name = create_session('oneline')
    log_input(session_dir, user_input)

    system_prompt = generate_system_prompt(session_type='oneline', user_input=user_input)
    ctx = ContextManager(session_dir)
    ctx.add('system', system_prompt)
    ctx.add('user', user_input)

    act_outputs = {}
    used_titles = set()

    print(f"\n{'='*60}")
    print(f"VADAR session: {session_name}")
    print(f"{'='*60}\n")

    _run_agent_loop(ctx, user_input, session_dir, act_outputs, model, processor, used_titles,
                    interactive=False)

    _finalize_session(session_dir, ctx)

    print(f"\n{'='*60}")
    print(f"VADAR session ended: {session_name}")
    print(f"Session log: {session_dir}")
    print(f"{'='*60}")
    return True


def run_vadar_interactive():
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        return False

    session_dir, session_name = create_session('interactive')
    ctx = ContextManager(session_dir)
    system_prompt = generate_system_prompt(session_type='interactive')
    ctx.add('system', system_prompt)

    act_outputs = {}
    used_titles = set()
    approval_event = threading.Event()

    print(f"\n{'='*60}")
    print(f"VADAR Interactive Mode")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print("Type 'exit' or 'quit' to end. Type 'clear' to reset context.\n")

    print("[VADAR]: Hey! I'm VADAR. What can I do for you?\n")

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[VADAR]: Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ('exit', 'quit'):
            print("\n[VADAR]: Goodbye!")
            break
        if user_input.lower() == 'clear':
            ctx = ContextManager(session_dir)
            system_prompt = generate_system_prompt(session_type='interactive')
            ctx.add('system', system_prompt)
            act_outputs = {}
            used_titles = set()
            print("\n[VADAR]: Context cleared.\n")
            continue

        if approval_event.is_set() is False and not approval_event._user_response if hasattr(approval_event, '_user_response') else False:
            approval_event._user_response = user_input
            approval_event.set()
            continue

        log_input(session_dir, user_input)
        ctx.add('user', user_input)

        def approval_waiter():
            try:
                resp = input("[You]: ").strip()
                approval_event._user_response = resp
                approval_event.set()
            except (EOFError, KeyboardInterrupt):
                approval_event._user_response = 'go'
                approval_event.set()

        _run_agent_loop(ctx, user_input, session_dir, act_outputs, model, processor,
                        used_titles, interactive=True, approval_event=approval_event)

        print()

    _finalize_session(session_dir, ctx)
    print(f"\nSession log: {session_dir}")
    return True


def _finalize_session(session_dir, ctx):
    try:
        from voders.vadars import VADAR_GLOBAL_CONTEXT_FILE
        log_path = os.path.join(session_dir, 'log.txt')
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                log_content = f.read()
            if len(log_content) > 500:
                summary = summarize_output(log_content, context_label=f"session {os.path.basename(session_dir)}")
                existing = ""
                if os.path.exists(VADAR_GLOBAL_CONTEXT_FILE):
                    with open(VADAR_GLOBAL_CONTEXT_FILE, 'r', encoding='utf-8') as f:
                        existing = f.read()
                parts = existing.split('\n---\n') if existing.strip() else []
                parts.append(f"Session {os.path.basename(session_dir)}:\n{summary}")
                parts = parts[-5:]
                with open(VADAR_GLOBAL_CONTEXT_FILE, 'w', encoding='utf-8') as f:
                    f.write('\n---\n'.join(parts))
    except Exception:
        pass
