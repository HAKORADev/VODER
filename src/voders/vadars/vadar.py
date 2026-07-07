import os
import sys
import re
import time
import traceback

from voders.vadars import (
    VADAR_SESSIONS_DIR, VADAR_ABOUT_DIR, VADAR_PING_TIME_FILE,
    VADAR_GLOBAL_CONTEXT_FILE,
)
from voders.vadars.system_prompt import generate_system_prompt
from voders.vadars.context import ContextManager, create_session, log_input, log_output, log_act
from voders.vadars.tools import TOOL_REGISTRY
from voders.vadars.tools.impl import (
    tool_read, tool_look, tool_listen, tool_watch,
    tool_list, tool_search, tool_calculate,
    tool_memory_read, tool_memory_write, tool_memory_edit, tool_memory_delete,
)


TOOL_CALL_RE = re.compile(r'<tool_call>\s*(\w+)\s*(.*?)\s*</tool_call>', re.DOTALL)
ACT_RE = re.compile(r'<act>\s*(\S+)\s+(.*?)\s*</act>', re.DOTALL)
REPLY_RE = re.compile(r'<reply>(.*?)</reply>', re.DOTALL)
THINK_RE = re.compile(r'<think>(.*?)</think>', re.DOTALL)
DECIDE_RE = re.compile(r'<decide>(.*?)</decide>', re.DOTALL)
EVAL_RE = re.compile(r'<eval>(.*?)</eval>', re.DOTALL)

EOS_REPLY = '<EOS_REPLY>'
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
        return f"Unknown tool: {tool_name}"
    try:
        if tool_name in ('look', 'listen', 'watch'):
            return tool(tool_args, model=model, processor=processor)
        elif tool_name == 'read':
            return tool(tool_args, session_dir=session_dir, act_outputs=act_outputs)
        else:
            return tool(tool_args)
    except Exception as e:
        return f"Tool '{tool_name}' error: {e}"


def _execute_act(title, command, session_dir, act_outputs):
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
        return success, output
    except Exception as e:
        sys.stdout = old_stdout
        act_outputs[title] = str(e)
        log_act(session_dir, title, command, str(e), False)
        return False, str(e)


def _run_inference(messages, max_new_tokens=1024):
    from voder import vadar_run_inference
    return vadar_run_inference(messages, max_new_tokens=max_new_tokens)


def run_vadar_oneline(user_input, result_path=None):
    from voder import vadar_load_model
    model, processor, err = vadar_load_model()
    if err:
        print(f"VADAR is not available — {err}")
        if "not found" in err.lower() or "not downloaded" in err.lower():
            print(f"\nTo download the model, run:")
            print(f"  python voder.py vadar-download")
        return False

    session_dir, session_name = create_session('oneline')
    log_input(session_dir, user_input)

    system_prompt = generate_system_prompt(session_type='oneline', user_input=user_input)
    ctx = ContextManager(session_dir)
    ctx.add('system', system_prompt)
    ctx.add('user', user_input)

    act_outputs = {}
    max_iterations = 20

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

        for tc in parsed['tool_calls']:
            tool_name = tc['tool']
            tool_args = tc['args']
            print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")
            result = _execute_tool_call(tool_name, tool_args, session_dir, act_outputs, model, processor)
            print(f"[TOOL_RESULT]: {result[:500]}")
            ctx.add('tool', f"Tool '{tool_name}' result:\n{result}")

        for act in parsed['acts']:
            title = act['title']
            command = act['command']
            print(f"\n[ACT]: {title} -> {command}")
            print(f"{'─'*40}")
            success, output = _execute_act(title, command, session_dir, act_outputs)
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
        if "not found" in err.lower() or "not downloaded" in err.lower():
            print(f"\nTo download the model, run:")
            print(f"  python voder.py vadar-download")
        return False

    session_dir, session_name = create_session('interactive')
    ctx = ContextManager(session_dir)
    system_prompt = generate_system_prompt(session_type='interactive')
    ctx.add('system', system_prompt)

    act_outputs = {}

    print(f"\n{'='*60}")
    print(f"VADAR Interactive Mode")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to start a fresh context.\n")

    print("[VADAR]: Hey! I'm VADAR, your VODER agent. What can I do for you?\n")

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
            print("\n[VADAR]: Context cleared. Fresh start!\n")
            continue

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

            for reply in parsed['replies']:
                print(f"\n[VADAR]: {reply}")
                log_output(session_dir, reply)

            for tc in parsed['tool_calls']:
                tool_name = tc['tool']
                tool_args = tc['args']
                print(f"\n[TOOL_CALL]: {tool_name} {tool_args}")
                result = _execute_tool_call(tool_name, tool_args, session_dir, act_outputs, model, processor)
                display = result[:800] if len(result) > 800 else result
                print(f"[TOOL_RESULT]: {display}")
                ctx.add('tool', f"Tool '{tool_name}' result:\n{result}")

            for act in parsed['acts']:
                title = act['title']
                command = act['command']
                print(f"\n[ACT]: {title} -> {command}")
                print(f"{'─'*40}")
                success, output = _execute_act(title, command, session_dir, act_outputs)
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
