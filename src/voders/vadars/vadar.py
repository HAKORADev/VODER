import os
import sys
import re
import time
import traceback

from voders.vadars.eval import evaluate_plan, evaluate_act_result
from voders.vadars.summarizer import summarize_output

from voders.vadars import VADAR_SESSIONS_DIR
from voders.vadars.system_prompt import generate_system_prompt
from voders.vadars.context import ContextManager, create_session, log_input, log_output, log_act
from voders.vadars.tools import TOOL_REGISTRY
from voders.vadars.tools.impl import (
    tool_read, tool_look, tool_listen, tool_watch,
    tool_list, tool_search, tool_calculate,
    tool_memory_read, tool_memory_write, tool_memory_edit, tool_memory_delete,
    tool_read_role, tool_make_role, tool_edit_role, tool_delete_role,
    tool_read_role_extras, tool_make_role_extras, tool_edit_role_extras, tool_delete_role_extras,
    tool_search_media,
)
from voders.vadars.tools.validator import catch_and_fix


TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>', re.DOTALL)
ACT_RE = re.compile(r'<act>\s*(\S+)\s+(.*?)\s*</act>', re.DOTALL)
REPLY_RE = re.compile(r'<reply>(.*?)</reply>', re.DOTALL)
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
DECIDE_RE = re.compile(r'<decide>(.*?)</decide>', re.DOTALL)
EVAL_RE = re.compile(r'<eval>(.*?)</eval>', re.DOTALL)

EOS_REPLY = ['<EOS_REPLY>']
EOS_ACT = '<EOS_ACT>'
EOS_DONE = '<EOS_DONE>'


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


def _execute_act(title, command, session_dir, act_outputs, user_request="", summarize_threshold=1500):
    from voder import parse_and_execute_oneline
    cmd_tokens = command.split()
    if not cmd_tokens:
        return False, "Empty command"
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


def _run_inference(messages, max_new_tokens=1024):
    from voder import vadar_run_inference
    return vadar_run_inference(messages, max_new_tokens=max_new_tokens)


def _process_tool_calls(tool_calls, session_dir, act_outputs, model, processor):
    total_calls = 0
    total_failed = 0
    for tc in tool_calls:
        tool_name = tc['tool']
        tool_args = tc['args']
        total_calls += 1
        print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")
        ok, err, fixed_args, catch_failures = catch_and_fix(tool_name, tool_args)
        if not ok:
            total_failed += catch_failures
            print(f"[CATCHER]: fixed/invalid — {err}")
            print(f"[TOOL_RESULT]: SKIPPED (invalid call, catcher flagged it)")
            continue
        success, result, elapsed = _execute_tool_call(tool_name, fixed_args, session_dir, act_outputs, model, processor)
        if not success:
            total_failed += 1
        display = result[:800] if len(result) > 800 else result
        print(f"[TOOL_RESULT]: {display}")
        print(f"[TOOL_STATS]: {tool_name} | {'OK' if success else 'FAILED'} | {elapsed:.2f}s | calls={total_calls} failed={total_failed}")
        from voders.vadars.context import ContextManager
        if session_dir:
            pass
        return result, total_calls, total_failed
    return "", total_calls, total_failed


def run_vadar_oneline(user_input, result_path=None):
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        return False

    session_dir, session_name = create_session('oneline')
    log_input(session_dir, user_input)

    system_prompt = generate_system_prompt(session_type='oneline', user_input=user_input)
    ctx = ContextManager(session_dir)
    ctx.add('system', system_prompt)
    ctx.add('user', user_input)

    act_outputs = {}
    max_iterations = 20
    last_vadar_reply_time = time.time()

    print(f"\n{'='*60}")
    print(f"VADAR session: {session_name}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        messages = ctx.get_for_inference()
        response, err = _run_inference(messages)
        if err:
            print(f"VADAR inference error: {err}")
            return False
        if not response or not response.strip():
            print("VADAR produced no output. Ending session.")
            break

        ctx.add('assistant', response)
        parsed = _parse_model_output(response)

        for reply in parsed['replies']:
            print(f"\n[VADAR]: {reply}")
            log_output(session_dir, reply)
            last_vadar_reply_time = time.time()

        if parsed['tool_calls']:
            for tc in parsed['tool_calls']:
                tool_name = tc['tool']
                tool_args = tc['args']
                print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")
                ok, err, fixed_args, catch_failures = catch_and_fix(tool_name, tool_args)
                if not ok:
                    print(f"[CATCHER]: {err}")
                    print(f"[TOOL_RESULT]: SKIPPED (invalid call)")
                    ctx.add('tool', f"Tool '{tool_name}' was invalid: {err}")
                    continue
                success, result, elapsed = _execute_tool_call(tool_name, fixed_args, session_dir, act_outputs, model, processor)
                display = result[:800] if len(result) > 800 else result
                print(f"[TOOL_RESULT]: {display}")
                print(f"[TOOL_STATS]: {tool_name} | {'OK' if success else 'FAILED'} | {elapsed:.2f}s")
                ctx.add('tool', f"Tool '{tool_name}' result:\n{result}")

        for act in parsed['acts']:
            title = act['title']
            command = act['command']
            print(f"\n[ACT]: {title} -> {command}")
            print(f"{'─'*40}")
            success, output = _execute_act(title, command, session_dir, act_outputs, user_request=user_input)
            print(f"{'─'*40}")
            status = 'SUCCESS' if success else 'FAILED'
            print(f"[ACT RESULT]: {title} -> {status}")
            ctx.add('tool', f"Act '{title}' result ({status}):\n{output[-2000:]}")

        if parsed['eos_done']:
            print("\n[VADAR]: Task complete.")
            break

        if not parsed['replies'] and not parsed['acts'] and not parsed['tool_calls']:
            if parsed['raw'].strip():
                print(f"\n[VADAR]: {parsed['raw'].strip()[:500]}")
                log_output(session_dir, parsed['raw'].strip()[:500])
            break

        if parsed['eos_reply'] and not parsed['acts'] and not parsed['tool_calls']:
            break

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
    last_user_msg_time = None
    last_vadar_reply_time = None

    print(f"\n{'='*60}")
    print(f"VADAR Interactive Mode")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to start a fresh context.\n")

    print("[VADAR]: Hey! I'm VADAR, your VODER agent. What can I do for you?\n")
    last_vadar_reply_time = time.time()

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
            last_user_msg_time = None
            last_vadar_reply_time = time.time()
            print("\n[VADAR]: Context cleared. Fresh start!\n")
            continue

        last_user_msg_time = time.time()
        log_input(session_dir, user_input)
        ctx.add('user', user_input)

        max_iterations = 30
        for iteration in range(max_iterations):
            messages = ctx.get_for_inference()
            response, err = _run_inference(messages)
            if err:
                print(f"\n[VADAR inference error]: {err}")
                break
            if not response or not response.strip():
                break

            ctx.add('assistant', response)
            parsed = _parse_model_output(response)

            if parsed['thoughts'] and parsed['decisions'] and not parsed['acts'] and not parsed['tool_calls']:
                thoughts_text = ' '.join(parsed['thoughts'])
                decisions_text = ' '.join(parsed['decisions'])
                verdict, reason = evaluate_plan(user_input, thoughts_text, decisions_text)
                print(f"[EVAL]: {verdict.upper()} — {reason}")
                if verdict == 'wrong':
                    ctx.add('system', f"Eval says your plan is WRONG: {reason}. Fix it and try again.")
                    continue

            for reply in parsed['replies']:
                print(f"\n[VADAR]: {reply}")
                log_output(session_dir, reply)
                last_vadar_reply_time = time.time()

            if parsed['tool_calls']:
                for tc in parsed['tool_calls']:
                    tool_name = tc['tool']
                    tool_args = tc['args']
                    print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")
                    ok, err, fixed_args, catch_failures = catch_and_fix(tool_name, tool_args)
                    if not ok:
                        print(f"[CATCHER]: {err}")
                        print(f"[TOOL_RESULT]: SKIPPED (invalid call)")
                        ctx.add('tool', f"Tool '{tool_name}' was invalid: {err}")
                        continue
                    success, result, elapsed = _execute_tool_call(tool_name, fixed_args, session_dir, act_outputs, model, processor)
                    display = result[:800] if len(result) > 800 else result
                    print(f"[TOOL_RESULT]: {display}")
                    print(f"[TOOL_STATS]: {tool_name} | {'OK' if success else 'FAILED'} | {elapsed:.2f}s")
                    ctx.add('tool', f"Tool '{tool_name}' result:\n{result}")

            for act in parsed['acts']:
                title = act['title']
                command = act['command']
                print(f"\n[ACT]: {title} -> {command}")
                print(f"{'─'*40}")
                success, output = _execute_act(title, command, session_dir, act_outputs, user_request=user_input)
                print(f"{'─'*40}")
                status = 'SUCCESS' if success else 'FAILED'
                print(f"[ACT RESULT]: {title} -> {status}")
                ctx.add('tool', f"Act '{title}' result ({status}):\n{output[-2000:]}")

            if parsed['eos_done']:
                break

            if not parsed['replies'] and not parsed['acts'] and not parsed['tool_calls']:
                if parsed['raw'].strip():
                    print(f"\n[VADAR]: {parsed['raw'].strip()[:500]}")
                    log_output(session_dir, parsed['raw'].strip()[:500])
                break

            if parsed['eos_reply'] and not parsed['acts'] and not parsed['tool_calls']:
                break

        print()

    print(f"\nSession log: {session_dir}")
    return True
