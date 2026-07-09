import os
import sys
import re
import time
import threading
import traceback

from voders.vadars.eval import evaluate_plan, evaluate_act_result
from voders.vadars.summarizer import summarize_output

from voders.vadars import VADAR_SESSIONS_DIR, VADAR_GLOBAL_CONTEXT_FILE
from voders.vadars.system_prompt import generate_system_prompt
from voders.vadars.context import ContextManager, create_session, log_input, log_output, log_act
from voders.vadars.tools import TOOL_REGISTRY
from voders.vadars.catcher import catch_and_fix
from voders.vadars.tools.validator import validate_tool_basic


TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>', re.DOTALL)
ACT_RE = re.compile(r'<act>\s*(\S+)\s+(.*?)\s*</act>', re.DOTALL)
REPLY_RE = re.compile(r'<reply>(.*?)</reply>', re.DOTALL)
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
DECIDE_RE = re.compile(r'<decide>(.*?)</decide>', re.DOTALL)
EVAL_RE = re.compile(r'<eval>(.*?)</eval>', re.DOTALL)
ORDERED_ACTION_RE = re.compile(r'<(act|tool_call)>(.*?)</\1>', re.DOTALL)

EOS_REPLY = '<EOS_REPLY>'
EOS_ACT = '<EOS_ACT>'
EOS_DONE = '<EOS_DONE>'

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_inference_lock = threading.Lock()


def _read_global_context_file():
    try:
        if os.path.exists(VADAR_GLOBAL_CONTEXT_FILE):
            with open(VADAR_GLOBAL_CONTEXT_FILE, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        pass
    return ''


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


def _parse_ordered_actions(text):
    actions = []
    for m in ORDERED_ACTION_RE.finditer(text):
        tag = m.group(1)
        content = m.group(2).strip()
        if tag == 'act':
            parts = content.split(None, 1)
            if len(parts) >= 2:
                actions.append({'type': 'act', 'title': parts[0].strip(), 'command': parts[1].strip()})
            elif len(parts) == 1 and parts[0]:
                actions.append({'type': 'act', 'title': parts[0].strip(), 'command': ''})
        elif tag == 'tool_call':
            parts = content.split(None, 1)
            if len(parts) >= 2:
                actions.append({'type': 'tool_call', 'tool': parts[0].strip(), 'args': parts[1].strip()})
            elif len(parts) == 1 and parts[0]:
                actions.append({'type': 'tool_call', 'tool': parts[0].strip(), 'args': ''})
    return actions


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
                 summarize_threshold=1500, used_titles=None, interactive=False, ctx=None):
    if used_titles is not None and title in used_titles:
        return False, f"Act title '{title}' already exists in this session. Use a unique title."
    if used_titles is not None:
        used_titles.add(title)

    from voder import parse_and_execute_oneline
    cmd_tokens = command.split()
    if not cmd_tokens:
        return False, "Empty command"

    results_before = _snapshot_results()
    act_t0 = time.time()

    if interactive:
        print(f"\n[ACT] {title}: PENDING — {command}")
        old_stdout = sys.stdout
        capture_buf = []
        class _LiveCountCapture:
            def __init__(self):
                self.line_count = 0
                self._last_print = 0
            def write(self, text):
                capture_buf.append(text)
                old_stdout.write(text)
                newlines = text.count('\n')
                self.line_count += newlines
                now = time.time()
                if now - self._last_print > 0.3 or '\n' not in text:
                    sys.stdout = old_stdout
                    print(f"\r[ACT] {title}: {self.line_count} lines", end='', flush=True)
                    sys.stdout = self
                    self._last_print = now
            def flush(self):
                old_stdout.flush()
        live_capture = _LiveCountCapture()
        sys.stdout = live_capture
    else:
        print(f"\n[ACT]: {title} -> {command}")
        print(f"{'─'*40}")
        old_stdout = sys.stdout
        capture_buf = []
        class _Capture:
            def write(self, text):
                capture_buf.append(text)
                old_stdout.write(text)
            def flush(self):
                old_stdout.flush()
        sys.stdout = _Capture()

    try:
        success = parse_and_execute_oneline(cmd_tokens)
        sys.stdout = old_stdout
        output = ''.join(capture_buf)
        act_elapsed = time.time() - act_t0

        log_act(session_dir, title, command, output, success)

        new_files = _new_result_files(results_before)
        if new_files:
            output += f"\n[RESULT FILES]: {', '.join(new_files)}"
        elif success:
            output += "\n[WARNING]: Command reported success but no new result files were found in results/."

        act_outputs[title] = output

        if interactive:
            status_str = '✓ SUCCESS' if success else '✗ FAILED'
            print(f"\r[ACT] {title}: {status_str} ({act_elapsed:.1f}s, {output.count(chr(10))} lines)    ")
        else:
            print(f"{'─'*40}")
            status = 'SUCCESS' if success else 'FAILED'
            print(f"[ACT RESULT]: {title} -> {status} ({act_elapsed:.1f}s)")

        if user_request:
            recent_msgs = ctx.get_messages()[-10:] if ctx else []
            global_ctx = _read_global_context_file()
            print(f"[EVAL]: evaluating act result...")
            verdict, reason = evaluate_act_result(user_request, title, command, output, success,
                                                  recent_messages=recent_msgs, global_context=global_ctx)
            print(f"[EVAL]: {verdict.upper()} — {reason}")
            log_act(session_dir, f"{title}_eval", f"eval verdict: {verdict}", reason, verdict == 'correct')

        if len(output) > summarize_threshold:
            input_chars = len(output)
            input_tokens_approx = input_chars // 4
            print(f"[SUMMARIZER]: running on {input_chars} chars (~{input_tokens_approx}K tokens)...")
            sum_t0 = time.time()
            summary = summarize_output(output, context_label=title, act_title=title, act_command=command)
            sum_elapsed = time.time() - sum_t0
            output_tokens_approx = len(summary) // 4
            print(f"[SUMMARIZER]: done ({sum_elapsed:.2f}s, ~{input_tokens_approx}K → ~{output_tokens_approx}K tokens)")
            return success, summary

        return success, output
    except Exception as e:
        sys.stdout = old_stdout
        act_elapsed = time.time() - act_t0
        act_outputs[title] = str(e)
        log_act(session_dir, title, command, str(e), False)
        if interactive:
            print(f"[ACT] {title}: ✗ FAILED ({act_elapsed:.1f}s) — {e}")
        else:
            print(f"{'─'*40}")
            print(f"[ACT RESULT]: {title} -> FAILED ({act_elapsed:.1f}s)")
        return False, str(e)


def _run_inference_streamed(messages, max_new_tokens=1024):
    from voder import vadar_run_inference_streamed
    with _inference_lock:
        return vadar_run_inference_streamed(messages, max_new_tokens=max_new_tokens)


def _run_inference(messages, max_new_tokens=1024):
    from voder import vadar_run_inference
    with _inference_lock:
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
    from voders.vadars.tools.impl import (
        _VIDEO_EXTENSIONS, _IMAGE_EXTENSIONS, _AUDIO_EXTENSIONS,
        _resolve_media_target,
    )
    for inp in inputs:
        is_url = inp.startswith('http://') or inp.startswith('https://')
        ext = os.path.splitext(inp)[1].lower()
        kind = None
        if ext in _AUDIO_EXTENSIONS:
            kind = 'audio'
        elif ext in _VIDEO_EXTENSIONS:
            kind = 'video'
        elif ext in _IMAGE_EXTENSIONS or is_url:
            kind = 'image'
        if kind is None:
            continue
        local, err = _resolve_media_target(inp, kind)
        if err:
            print(f"[AUTO-HEAR]: {err}")
            continue
        tool_name = {'audio': 'listen', 'video': 'watch', 'image': 'look'}[kind]
        print(f"[AUTO-HEAR]: {tool_name} on {local}")
        result = _execute_tool_call(tool_name, local, session_dir, act_outputs, model, processor)
        print(f"[AUTO-HEAR RESULT]: {str(result[1])[:300]}")


def _process_tool_calls(tool_calls, ctx, session_dir, act_outputs, model, processor):
    total_calls = 0
    total_passed = 0
    total_failed = 0
    total_time = 0.0
    try:
        from voder import vadar_load_config
        max_retries = vadar_load_config().get('catcher_max_retries', 3)
    except Exception:
        max_retries = 3
    for tc in tool_calls:
        tool_name = tc['tool']
        tool_args = tc['args']
        total_calls += 1
        print(f"\n[TOOL_CALL {total_calls}]: {tool_name} {tool_args}")

        ok = False
        err = None
        fixed_args = tool_args

        for attempt in range(max_retries):
            t0 = time.time()
            basic_ok, basic_err = validate_tool_basic(tool_name, fixed_args if attempt > 0 else tool_args)
            basic_t = time.time() - t0
            if not basic_ok:
                print(f"[VALIDATOR]: ✗ FAIL ({basic_t:.2f}s) — {basic_err}")
                ok = False
                err = basic_err
            else:
                if attempt == 0:
                    print(f"[VALIDATOR]: ✓ PASS ({basic_t:.2f}s) — sending to Catcher for deep check...")
                t0 = time.time()
                ok, err, fixed_args, _ = catch_and_fix(tool_name, fixed_args if attempt > 0 else tool_args)
                catcher_t = time.time() - t0
                verdict_str = '✓ OK' if ok else '✗ CANNOT_FIX'
                print(f"[CATCHER]: {verdict_str} ({catcher_t:.2f}s){'' if ok else f' — {err}'}")

            if ok:
                break
            if attempt < max_retries - 1:
                retry_src = 'VALIDATOR' if not basic_ok else 'CATCHER'
                print(f"[{retry_src}]: asking VADAR to retry ({attempt+1}/{max_retries})")
                ctx.add('system', f"My tool call '{tool_name} {tool_args}' was invalid: {err}. Fix it and retry.")
                response, inf_err = _run_inference_streamed(ctx.get_for_inference(), max_new_tokens=512)
                if inf_err or not response or not response.strip():
                    break
                ctx.add('assistant', response)
                parsed_retry = _parse_model_output(response)
                if parsed_retry['tool_calls']:
                    for rtc in parsed_retry['tool_calls']:
                        if rtc['tool'] == tool_name:
                            fixed_args = rtc['args']
                            break
                    else:
                        break
                else:
                    break
            else:
                print(f"[VALIDATOR/CATCHER]: max retries reached — skipping")
                ctx.add('tool', f"Tool '{tool_name}' was invalid after {max_retries} retries: {err}")

        if not ok:
            total_failed += 1
            print(f"[TOOL_STATS]: {tool_name} | FAILED | calls={total_calls} passed={total_passed} failed={total_failed} total_time={total_time:.2f}s")
            continue

        success, result, elapsed = _execute_tool_call(
            tool_name, fixed_args, session_dir, act_outputs, model, processor
        )
        total_time += elapsed
        if success:
            total_passed += 1
        else:
            total_failed += 1
        display = result[:800] if len(result) > 800 else result
        print(f"[TOOL_RESULT]: {display}")
        print(f"[TOOL_STATS]: {tool_name} | {'OK' if success else 'FAILED'} | {elapsed:.2f}s | calls={total_calls} passed={total_passed} failed={total_failed} total_time={total_time:.2f}s")
        is_mem = tool_name == 'memory_read'
        added = ctx.add('tool', f"Tool '{tool_name}' result:\n{result}", is_memory=is_mem)
        if is_mem and not added:
            info = ctx.memory_capacity_info()
            mem_list = '\n'.join(f"  [{m['index']}] ({m['tokens']} tokens) {m['preview']}" for m in info['memories'])
            if not mem_list:
                mem_list = "  (no memories currently in context)"
            mem_err = (
                f"Memory context is FULL ({info['used']}/{info['max']} tokens used). "
                f"I could not add the memory_read result to my context. "
                f"I must talk to the user and ask which memory to delete. "
                f"Memories currently in context:\n{mem_list}\n"
                f"Use memory_delete <vadar|user> <id> to free space, then re-read."
            )
            print(f"[MEMORY]: FULL — {info['used']}/{info['max']} tokens")
            ctx.add('system', mem_err)


def _run_agent_loop(ctx, user_input, session_dir, act_outputs, model, processor,
                    used_titles, interactive=False, approval_event=None,
                    waiting_for_approval=None, act_log=None):
    max_iterations = 30 if interactive else 20
    last_vadar_reply_time = time.time()
    acts_have_run = False

    detected = _detect_inputs(user_input)
    if detected:
        _auto_hear_inputs(detected, session_dir, act_outputs, model, processor)
        ctx.add('system', f"I have automatically listened to/watched/looked at the inputs you mentioned: {', '.join(detected)}. Use what you learned to plan your act.")

    for iteration in range(max_iterations):
        messages = ctx.get_for_inference()
        ts_msg = {'role': 'system', 'content': f"Current time: {time.strftime('%Y/%m/%d:%I%p:%M:%S')}"}
        messages.append(ts_msg)
        response, err = _run_inference_streamed(messages)
        if err or not response or not response.strip():
            response, err = _run_inference(messages)
        if err:
            print(f"\n[VADAR inference error]: {err}")
            return False
        if not response or not response.strip():
            print("\n[VADAR]: (model produced no output)")
            break

        ctx.add('assistant', response)
        parsed = _parse_model_output(response)

        if parsed['thoughts']:
            for thought in parsed['thoughts']:
                print(f"\n[VADAR THINK]: {thought}")

        if parsed['decisions']:
            for decision in parsed['decisions']:
                print(f"\n[VADAR DECIDE]: {decision}")

        if parsed['thoughts'] and parsed['decisions']:
            thoughts_text = ' '.join(parsed['thoughts'])
            decisions_text = ' '.join(parsed['decisions'])
            recent_msgs = ctx.get_messages()[-10:]
            global_ctx = _read_global_context_file()
            print(f"\n[EVAL]: evaluating plan...")
            verdict, reason = evaluate_plan(user_input, thoughts_text, decisions_text,
                                            acts=parsed['acts'], recent_messages=recent_msgs,
                                            global_context=global_ctx)
            print(f"[EVAL]: {verdict.upper()} — {reason}")
            if verdict == 'wrong':
                ctx.add('system', f"Eval says your plan is WRONG: {reason}. Fix it and try again.")
                continue

        if parsed['replies']:
            for reply in parsed['replies']:
                print(f"\n[VADAR]: {reply}")
                log_output(session_dir, reply)
                last_vadar_reply_time = time.time()

            if interactive and approval_event is not None and waiting_for_approval is not None:
                has_plan_signals = bool(parsed['decisions']) or (not parsed['acts'] and not parsed['eos_act'] and not parsed['eos_done'])
                if has_plan_signals and not parsed['eos_act'] and not acts_have_run:
                    print("\n[VADAR]: This is my plan. Type 'go' to approve, or tell me what to change.")
                    waiting_for_approval[0] = True
                    approval_event.clear()
                    while not approval_event.is_set():
                        approval_event.wait(timeout=0.5)
                    waiting_for_approval[0] = False
                    approval_str = getattr(approval_event, '_user_response', 'go')
                    if approval_str.lower().strip() in ('go', 'ok', 'yes', 'proceed', 'do it', 'continue', ''):
                        ctx.add('user', f"go")
                    else:
                        ctx.add('user', approval_str)
                        continue

        if parsed['tool_calls'] or parsed['acts']:
            ordered = _parse_ordered_actions(response)
            has_eos_act = parsed['eos_act']
            acts_without_eos = False
            for item in ordered:
                if item['type'] == 'tool_call':
                    _process_tool_calls(
                        [item], ctx, session_dir, act_outputs, model, processor
                    )
                elif item['type'] == 'act':
                    if not has_eos_act:
                        acts_without_eos = True
                        continue
                    from voder import parse_oneline_args, validate_oneline_mode
                    title = item['title']
                    command = item['command']
                    cmd_tokens = command.split()
                    if cmd_tokens:
                        mode = cmd_tokens[0].lower()
                        if validate_oneline_mode(mode) is None:
                            print(f"\n[ACT VALIDATOR]: '{title}' — invalid mode '{mode}'. Blocked.")
                            ctx.add('system', f"Your act '{title}' uses invalid mode '{mode}'. Valid modes: tts, sts, ttm, stt, se, sfx, svs, ss, train, quest, chains. Fix and re-emit.")
                            continue
                        test_parse = parse_oneline_args(cmd_tokens)
                        if test_parse.get('error'):
                            print(f"\n[ACT VALIDATOR]: '{title}' — syntax error: {test_parse['error']}. Blocked.")
                            ctx.add('system', f"Your act '{title}' has a syntax error: {test_parse['error']}. Fix and re-emit.")
                            continue
                    print(f"\n[ACT]: {title} -> {command}")
                    print(f"{'─'*40}")
                    success, output = _execute_act(
                        title, command, session_dir, act_outputs,
                        user_request=user_input, used_titles=used_titles,
                        interactive=interactive, ctx=ctx
                    )
                    print(f"{'─'*40}")
                    status = 'SUCCESS' if success else 'FAILED'
                    print(f"[ACT RESULT]: {title} -> {status}")
                    ctx.add('tool', f"Act '{title}' result ({status}):\n{output[-2000:]}")
                    if act_log is not None:
                        act_log.append({'title': title, 'command': command, 'success': success, 'output': output})
                    acts_have_run = True
            if acts_without_eos:
                ctx.add('system', "You emitted acts but did not emit <EOS_ACT>. Acts will not execute until you emit <EOS_ACT>. Emit it now if you want the acts to run.")

        if parsed['eos_done']:
            print("\n[VADAR]: Task complete.")
            return True

        if not parsed['replies'] and not parsed['acts'] and not parsed['tool_calls']:
            if parsed['thoughts']:
                continue
            if parsed['raw'].strip():
                print(f"\n[VADAR]: {parsed['raw'].strip()[:500]}")
                log_output(session_dir, parsed['raw'].strip()[:500])
                last_vadar_reply_time = time.time()
            break

        if parsed['eos_reply']:
            break

    return True


def run_vadar_oneline(user_input, result_path=None):
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        return False

    tasks = [t.strip() for t in re.split(r'\s*&&\s*', user_input) if t.strip()]
    if not tasks:
        return False

    session_dir, session_name = create_session('oneline')
    log_input(session_dir, user_input)

    system_prompt = generate_system_prompt(session_type='oneline', user_input=user_input, exclude_session=session_name)
    ctx = ContextManager(session_dir)
    try:
        if processor is not None and hasattr(processor, 'tokenizer'):
            ctx.set_tokenizer(processor.tokenizer)
    except Exception:
        pass
    ctx.add('system', system_prompt)

    act_outputs = {}
    used_titles = set()
    act_log = []

    print(f"\n{'='*60}")
    print(f"VADAR session: {session_name}")
    if len(tasks) > 1:
        print(f"Multi-task: {len(tasks)} tasks (split by &&) — sharing one session.")
    print(f"{'='*60}\n")

    all_ok = True
    for i, task in enumerate(tasks, 1):
        if len(tasks) > 1:
            print(f"\n{'─'*40}")
            print(f"Task {i}/{len(tasks)}: {task[:100]}")
            print(f"{'─'*40}")
        log_input(session_dir, f"[Task {i}] {task}")
        ctx.add('user', task)
        ok = _run_agent_loop(ctx, task, session_dir, act_outputs, model, processor, used_titles,
                             interactive=False, act_log=act_log)
        if not ok:
            all_ok = False

    _finalize_session(session_dir, ctx)

    if act_log:
        print(f"\n{'='*60}")
        print("VADAR Session Report")
        print(f"{'='*60}")
        print(f"Acts run: {len(act_log)}")
        succeeded = sum(1 for a in act_log if a['success'])
        failed = len(act_log) - succeeded
        print(f"Succeeded: {succeeded} | Failed: {failed}")
        print(f"{'─'*60}")
        for i, a in enumerate(act_log, 1):
            status = '✓ SUCCESS' if a['success'] else '✗ FAILED'
            print(f"  {i}. [{status}] {a['title']}")
            print(f"     Command: {a['command'][:100]}")
            result_files = [line for line in a['output'].split('\n') if '[RESULT FILES]' in line or '[WARNING]' in line]
            for rf in result_files:
                print(f"     {rf.strip()}")
        print(f"{'='*60}")

    if result_path:
        results_dir = _RESULTS_DIR
        if os.path.isdir(results_dir):
            files = sorted(
                [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))],
                key=os.path.getmtime,
                reverse=True,
            )
            if files:
                import shutil
                try:
                    shutil.copy2(files[0], result_path)
                    print(f"Result copied to: {result_path}")
                except Exception as e:
                    print(f"Note: could not copy to result path: {e}")

    print(f"\nVADAR session ended: {session_name}")
    print(f"Session log: {session_dir}")
    print(f"{'='*60}")
    return all_ok


def run_vadar_interactive():
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        return False

    session_dir, session_name = create_session('interactive')
    last_user_msg_time = [None]
    last_vadar_reply_time = [time.time()]
    ctx = ContextManager(session_dir)
    try:
        if processor is not None and hasattr(processor, 'tokenizer'):
            ctx.set_tokenizer(processor.tokenizer)
    except Exception:
        pass
    system_prompt = generate_system_prompt(session_type='interactive',
                                           last_user_msg_time=last_user_msg_time[0],
                                           last_vadar_reply_time=last_vadar_reply_time[0],
                                           exclude_session=session_name)
    ctx.add('system', system_prompt)

    act_outputs = {}
    used_titles = set()
    approval_event = threading.Event()
    waiting_for_approval = [False]

    from voders.vadars.system_prompt import _read_ping_time
    ping_interval = _read_ping_time()
    ping_stop = threading.Event()
    last_user_activity = [time.time()]
    ping_count = [0]
    vadar_busy = [False]
    busy_lock = threading.Lock()

    ping_ctx = {'ctx': ctx, 'session_dir': session_dir, 'model': model, 'processor': processor,
                'act_outputs': act_outputs, 'used_titles': used_titles, 'user_input': ''}

    def ping_thread():
        if ping_interval == 0:
            return
        while not ping_stop.is_set():
            ping_stop.wait(timeout=1)
            if ping_stop.is_set():
                break
            with busy_lock:
                if vadar_busy[0] or waiting_for_approval[0]:
                    last_user_activity[0] = time.time()
                    continue
                if last_vadar_reply_time[0] is None:
                    continue
                elapsed = time.time() - max(last_user_activity[0], last_vadar_reply_time[0])
                if elapsed < ping_interval:
                    continue
                vadar_busy[0] = True
            ping_count[0] += 1
            ts = time.strftime("%Y/%m/%d:%I%p:%M:%S")
            ping_msg = f"PING #{ping_count[0]} — {ts} — {int(elapsed)}s of silence."
            print(f"\n[{ping_msg}]")
            vadar_busy[0] = True
            try:
                ping_ctx['ctx'].add('system', ping_msg + " You may reply or stay silent. If you reply, keep it brief.")
                response, inf_err = _run_inference_streamed(ping_ctx['ctx'].get_for_inference(), max_new_tokens=256)
                if inf_err:
                    response, inf_err = _run_inference(ping_ctx['ctx'].get_for_inference(), max_new_tokens=256)
                if not inf_err and response and response.strip():
                    ping_ctx['ctx'].add('assistant', response)
                    parsed = _parse_model_output(response)
                    for reply in parsed['replies']:
                        print(f"\n[VADAR]: {reply}")
                        log_output(ping_ctx['session_dir'], reply)
                        last_vadar_reply_time[0] = time.time()
                    if parsed['eos_done']:
                        print("\n[VADAR]: I'm done. Ending session.")
                        ping_stop.set()
                        break
            finally:
                vadar_busy[0] = False
                last_user_activity[0] = time.time()

    if ping_interval > 0:
        ping_thread_obj = threading.Thread(target=ping_thread, daemon=True)
        ping_thread_obj.start()
    else:
        ping_thread_obj = None

    print(f"\n{'='*60}")
    print(f"VADAR Interactive Mode")
    print(f"Session: {session_name}")
    if ping_interval > 0:
        print(f"Ping: every {ping_interval}s of silence (only when idle)")
    else:
        print(f"Ping: disabled")
    print(f"{'='*60}")
    print("Type 'exit' or 'quit' to end.\n")

    print("[VADAR]: Hey! I'm VADAR. What can I do for you?\n")
    last_vadar_reply_time[0] = time.time()

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[VADAR]: Goodbye!")
            break

        last_user_activity[0] = time.time()
        ping_count[0] = 0

        if not user_input:
            print("[VADAR]: Cannot send an empty message. Please type something.")
            continue
        if user_input.lower() in ('exit', 'quit'):
            print("\n[VADAR]: Goodbye!")
            break

        if waiting_for_approval[0]:
            approval_event._user_response = user_input
            approval_event.set()
            continue

        last_user_msg_time[0] = time.time()
        log_input(session_dir, user_input)
        ctx.add('user', user_input)
        ping_ctx['user_input'] = user_input

        with busy_lock:
            vadar_busy[0] = True
        try:
            _run_agent_loop(ctx, user_input, session_dir, act_outputs, model, processor,
                            used_titles, interactive=True, approval_event=approval_event,
                            waiting_for_approval=waiting_for_approval)
        finally:
            with busy_lock:
                vadar_busy[0] = False

        last_vadar_reply_time[0] = time.time()
        last_user_activity[0] = time.time()
        print()

    ping_stop.set()
    if ping_thread_obj is not None:
        ping_thread_obj.join(timeout=5)
    _finalize_session(session_dir, ctx)
    print(f"\nSession log: {session_dir}")
    return True


def _finalize_session(session_dir, ctx):
    try:
        from voder import vadar_load_config
        config = vadar_load_config()
        gc_cap_pct = config.get('global_context_cap_percent', 15) / 100.0
        messages = ctx.get_messages()
        conv_text = '\n'.join(f"[{m['role'].upper()}] {m['content']}" for m in messages)
        if len(conv_text) > 500:
            input_chars = len(conv_text)
            input_tokens_approx = input_chars // 4
            print(f"\n[SUMMARIZER]: running on session ({input_chars} chars, ~{input_tokens_approx}K tokens)...")
            sum_t0 = time.time()
            summary = summarize_output(conv_text, context_label=f"session {os.path.basename(session_dir)}")
            sum_elapsed = time.time() - sum_t0
            output_tokens_approx = len(summary) // 4
            print(f"[SUMMARIZER]: done ({sum_elapsed:.2f}s, ~{input_tokens_approx}K → ~{output_tokens_approx}K tokens)")
            session_block = f"=== SESSION: {os.path.basename(session_dir)} ===\n{summary}\n=== END SESSION ==="
            existing = ""
            if os.path.exists(VADAR_GLOBAL_CONTEXT_FILE):
                with open(VADAR_GLOBAL_CONTEXT_FILE, 'r', encoding='utf-8') as f:
                    existing = f.read()
            blocks = []
            if existing.strip():
                import re as _re
                block_re = _re.compile(r'=== SESSION: .*? ===\n.*?\n=== END SESSION ===', _re.DOTALL)
                blocks = block_re.findall(existing)
            blocks.append(session_block)
            max_global_tokens = int(ctx.max_tokens * gc_cap_pct)
            def _est_tokens(text):
                if ctx._tokenizer is not None:
                    try:
                        return len(ctx._tokenizer.encode(text))
                    except Exception:
                        pass
                return len(text) // 4 + 1
            while blocks and _est_tokens('\n---\n'.join(blocks)) > max_global_tokens:
                blocks.pop(0)
            combined = '\n---\n'.join(blocks)
            with open(VADAR_GLOBAL_CONTEXT_FILE, 'w', encoding='utf-8') as f:
                f.write(combined)
    except Exception as e:
        print(f"[FINALIZE SESSION]: error updating global context — {e}")
