import os
import sys
import re
import time
import traceback

from voders.vadars import VADAR_MODEL_DIR, VADAR_SESSIONS_DIR
from voders.vadars.system_prompt import generate_system_prompt
from voders.vadars.context import ContextManager, create_session, log_input, log_output, log_act
from voders.vadars.tools import TOOL_REGISTRY
from voders.vadars.tools.impl import (
    tool_read, tool_look, tool_listen, tool_watch,
    tool_list, tool_search, tool_calculate,
    tool_memory_read, tool_memory_write, tool_memory_edit, tool_memory_delete,
)

_model = None
_processor = None
_model_loading_attempted = False


def _try_import_torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _try_import_transformers():
    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
        return AutoModelForMultimodalLM, AutoProcessor
    except ImportError:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            return AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return None, None


def load_model(force_reload=False):
    global _model, _processor, _model_loading_attempted
    if _model is not None and not force_reload:
        return _model, _processor
    if _model_loading_attempted and not force_reload:
        return None, None
    _model_loading_attempted = True

    torch = _try_import_torch()
    if torch is None:
        print("VADAR: torch is not installed. Install with: pip install torch")
        return None, None

    AutoModel, AutoProc = _try_import_transformers()
    if AutoModel is None:
        print("VADAR: transformers is not installed. Install with: pip install transformers")
        return None, None

    model_path = VADAR_MODEL_DIR
    if not os.path.isdir(model_path):
        print(f"VADAR: model directory not found at {model_path}")
        print("VADAR: download the model from https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored")
        print(f"VADAR: place the files in {model_path}/")
        return None, None

    has_weights = any(
        f.endswith('.safetensors') or f.endswith('.bin')
        for f in os.listdir(model_path)
    )
    if not has_weights:
        print(f"VADAR: no model weights found in {model_path}")
        print("VADAR: download the model .safetensors file from HuggingFace and place it there.")
        return None, None

    print(f"VADAR: loading model from {model_path}...")
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        _model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        _processor = AutoProc.from_pretrained(model_path, trust_remote_code=True)
        _model.eval()
        print("VADAR: model loaded successfully.")
        return _model, _processor
    except Exception as e:
        print(f"VADAR: failed to load model: {e}")
        traceback.print_exc()
        _model = None
        _processor = None
        return None, None


def _run_inference(messages, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=64):
    global _model, _processor
    if _model is None or _processor is None:
        return None, "Model not loaded. Call load_model() first or install the required dependencies."

    torch = _try_import_torch()
    if torch is None:
        return None, "torch not available"

    try:
        if hasattr(_processor, 'apply_chat_template'):
            text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = '\n'.join(f"{m['role']}: {m['content']}" for m in messages) + '\nassistant: '

        inputs = _processor(
            text=text,
            return_tensors='pt',
        ).to(_model.device if hasattr(_model, 'device') else 'cpu')

        with torch.no_grad():
            output = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )

        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        new_tokens = output[0][input_len:]
        response = _processor.decode(new_tokens, skip_special_tokens=True)
        return response, None
    except Exception as e:
        return None, str(e)


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


def run_vadar_oneline(user_input, result_path=None):
    model, processor = load_model()
    if model is None:
        print("VADAR is not available — model not loaded.")
        print("To enable VADAR:")
        print(f"  1. Download the model from https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored")
        print(f"  2. Place the files in {VADAR_MODEL_DIR}/")
        print(f"  3. Install dependencies: pip install torch transformers psutil")
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
    model, processor = load_model()
    if model is None:
        print("VADAR is not available — model not loaded.")
        print("To enable VADAR:")
        print(f"  1. Download the model from https://huggingface.co/OpenYourMind/gemma-4-12B-it-abliterated-uncensored")
        print(f"  2. Place the files in {VADAR_MODEL_DIR}/")
        print(f"  3. Install dependencies: pip install torch transformers psutil")
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
