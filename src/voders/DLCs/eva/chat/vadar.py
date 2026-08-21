import os
import sys
import time
import re
import threading

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

VADAR_MODEL_DIR = os.path.join(_src_dir, "models", "checkpoints", "vadar_eva")
VADAR_GGUF_REPO = "Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit"
VADAR_GGUF_FILENAME = "SuperGemma-4-12b-abliterated-Q4_K_M.gguf"
VADAR_OLLAMA_MODEL_NAME = "vadar-eva"

VADAR_HEAVY_MODEL_DIR = os.path.join(_src_dir, "models", "checkpoints", "vadar_heavy")
VADAR_HEAVY_GGUF_REPO = "OBLITERATUS/Qwen3.8-27B-OBLITERATED"
VADAR_HEAVY_GGUF_FILENAME = "Qwen3.8-27B-OBLITERATED-Q5_K_M.gguf"
VADAR_HEAVY_OLLAMA_MODEL_NAME = "vadar-heavy"

VADAR_MEMORIES_DIR = os.path.join(_src_dir, "voders", "DLCs", "eva", "chat", "memories")
VADAR_ABOUT_DIR = os.path.join(_src_dir, "voders", "DLCs", "eva", "chat", "about")
VADAR_PING_TIME_FILE = os.path.join(_src_dir, "voders", "DLCs", "eva", "chat", "ping-time.txt")
VADAR_SESSIONS_DIR = os.path.join(_src_dir, "voders", "DLCs", "eva", "chat", "sessions")

_MODEL_CONFIGS = {
    'vadar': {
        'ollama_name': VADAR_OLLAMA_MODEL_NAME,
        'model_dir': VADAR_MODEL_DIR,
        'gguf_repo': VADAR_GGUF_REPO,
        'gguf_filename': VADAR_GGUF_FILENAME,
        'model_size_gb': 8.0,
        'temperature': 0.8,
        'top_p': 0.95,
        'top_k': 64,
        'repeat_penalty': 1.1,
        'display_name': 'Gemma 4 12B (abliterated, GGUF Q4_K_M)',
    },
    'vadar-heavy': {
        'ollama_name': VADAR_HEAVY_OLLAMA_MODEL_NAME,
        'model_dir': VADAR_HEAVY_MODEL_DIR,
        'gguf_repo': VADAR_HEAVY_GGUF_REPO,
        'gguf_filename': VADAR_HEAVY_GGUF_FILENAME,
        'model_size_gb': 19.0,
        'temperature': 0.7,
        'top_p': 0.95,
        'top_k': 64,
        'repeat_penalty': 1.15,
        'display_name': 'Qwen3.8-27B OBLITERATED (GGUF Q5_K_M)',
    },
}

_OVERHEAD_GB = 4.0
_GPU_OVERHEAD_GB = 0.128
_KV_PER_TOKEN_MB = 0.4
_MIN_CONTEXT = 2048
_MAX_CONTEXT = 262144

_THINK_START = "<|channel>thought\n"
_THINK_END = "<channel|>"

_loaded_models = {}
_context_lengths = {}


def _get_config(model_name='vadar'):
    return _MODEL_CONFIGS.get(model_name, _MODEL_CONFIGS['vadar'])


def _calculate_dynamic_context(model_name='vadar'):
    config = _get_config(model_name)
    model_size_gb = config['model_size_gb']

    try:
        import psutil
        ram_available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        ram_available_gb = 12

    vram_gb = 0
    has_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
    except Exception:
        pass

    if has_gpu:
        overhead = _GPU_OVERHEAD_GB
        vram_for_kv = max(0, vram_gb - model_size_gb - overhead)
        ram_spill_for_kv = max(0, ram_available_gb * 0.5)
        total_kv = vram_for_kv + ram_spill_for_kv
    else:
        total_kv = max(0, ram_available_gb - model_size_gb - _OVERHEAD_GB)

    context_tokens = int((total_kv * 1024) / _KV_PER_TOKEN_MB)
    context_tokens = max(_MIN_CONTEXT, min(context_tokens, _MAX_CONTEXT))
    return context_tokens


def _ensure_ollama_running():
    try:
        import ollama
        ollama.list()
        return True, None
    except Exception:
        pass
    try:
        import subprocess
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1)
            try:
                import ollama
                ollama.list()
                return True, None
            except Exception:
                continue
        return False, "Ollama did not respond within 15 seconds"
    except Exception as e:
        return False, str(e)


def _ollama_model_exists(ollama_name):
    try:
        import ollama
        models = ollama.list()
        if hasattr(models, 'models'):
            model_list = models.models
        elif isinstance(models, dict):
            model_list = models.get('models', [])
        else:
            model_list = models
        for m in model_list:
            name = getattr(m, 'model', getattr(m, 'name', ''))
            if ollama_name in str(name):
                return True
        return False
    except Exception:
        return False


def vadar_check_model_downloaded(model_name='vadar'):
    config = _get_config(model_name)
    return _ollama_model_exists(config['ollama_name'])


def vadar_load_model(model_name='vadar', force_reload=False):
    global _loaded_models, _context_lengths

    if model_name in _loaded_models and not force_reload:
        return True, None

    config = _get_config(model_name)

    ok, err = _ensure_ollama_running()
    if err:
        return False, err

    if not _ollama_model_exists(config['ollama_name']):
        import subprocess
        gguf_path = os.path.join(config['model_dir'], config['gguf_filename'])
        if not os.path.exists(gguf_path):
            print(f"Downloading {model_name} model ({config['gguf_filename']})...")
            os.makedirs(config['model_dir'], exist_ok=True)
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=config['gguf_repo'],
                    filename=config['gguf_filename'],
                    local_dir=config['model_dir'],
                )
            except Exception as e:
                return False, f"Failed to download model: {e}"

        modelfile_path = os.path.join(config['model_dir'], "Modelfile")
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM ./{config['gguf_filename']}\n")
            f.write(f"PARAMETER num_ctx {_MAX_CONTEXT}\n")
            f.write(f"PARAMETER temperature {config['temperature']}\n")
            f.write(f"PARAMETER top_p {config['top_p']}\n")
            f.write(f"PARAMETER top_k {config['top_k']}\n")
            f.write(f"PARAMETER repeat_penalty {config['repeat_penalty']}\n")

        try:
            subprocess.run(
                ["ollama", "create", config['ollama_name'], "-f", modelfile_path],
                cwd=config['model_dir'],
                check=True,
                capture_output=True,
                text=True,
                timeout=600,
            )
        except Exception as e:
            return False, f"Failed to create Ollama model: {e}"

    _context_lengths[model_name] = _calculate_dynamic_context(model_name)
    _loaded_models[model_name] = True
    return True, None


def _read_about_file(name):
    path = os.path.join(VADAR_ABOUT_DIR, name)
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _read_ping_time():
    if not os.path.exists(VADAR_PING_TIME_FILE):
        return 15
    try:
        with open(VADAR_PING_TIME_FILE, 'r') as f:
            content = f.read().strip()
            val = int(content)
            if val == 0:
                return 0
            if val < 5:
                return 15
            return val
    except Exception:
        return 15


def _format_time_diff(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes {int(seconds % 60)} seconds"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hours {int((seconds % 3600) // 60)} minutes"
    days = int(seconds // 86400)
    remaining = seconds % 86400
    hours = int(remaining // 3600)
    minutes = int((remaining % 3600) // 60)
    if days < 30:
        return f"{days} days {hours} hours {minutes} minutes"
    months = int(days // 30)
    remaining_days = days % 30
    if months < 12:
        return f"{months} months {remaining_days} days {hours} hours"
    years = int(months // 12)
    remaining_months = months % 12
    return f"{years} years {remaining_months} months {remaining_days} days"


def _get_last_seen(exclude_session=None):
    if not os.path.isdir(VADAR_SESSIONS_DIR):
        return None
    sessions = []
    for entry in os.listdir(VADAR_SESSIONS_DIR):
        full = os.path.join(VADAR_SESSIONS_DIR, entry)
        if not os.path.isdir(full):
            continue
        if exclude_session and entry == exclude_session:
            continue
        log_path = os.path.join(full, 'log.txt')
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            mtime = os.path.getmtime(log_path)
            sessions.append((mtime, entry))
    if not sessions:
        return None
    sessions.sort(reverse=True)
    last_mtime, last_session = sessions[0]
    now = time.time()
    diff = now - last_mtime
    return _format_time_diff(diff), time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(last_mtime))


def _list_memory_files(category):
    mem_dir = os.path.join(VADAR_MEMORIES_DIR, category)
    if not os.path.isdir(mem_dir):
        return []
    files = sorted([f for f in os.listdir(mem_dir) if f.endswith('.md')])
    return files


def _read_memory_file(category, filename):
    path = os.path.join(VADAR_MEMORIES_DIR, category, filename)
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _build_memory_section():
    vadar_mems = _list_memory_files('vadar')
    user_mems = _list_memory_files('user')
    parts = []
    if vadar_mems:
        parts.append("My memories (vadar/):")
        for f in vadar_mems:
            content = _read_memory_file('vadar', f)
            if content:
                parts.append(f"  [{f}] {content[:200]}")
    if user_mems:
        parts.append("User memories (user/):")
        for f in user_mems:
            content = _read_memory_file('user', f)
            if content:
                parts.append(f"  [{f}] {content[:200]}")
    if not parts:
        return ''
    parts.append("")
    parts.append("To update memories: tell the user what to write in these files. The files live at src/voders/DLCs/eva/chat/memories/vadar/ and memories/user/. You cannot write to them directly — instruct the user.")
    return '\n'.join(parts)


def vadar_get_system_prompt(exclude_session=None, is_ping=False):
    parts = []

    now = time.time()
    timestamp_str = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(now))
    parts.append(f"Current time: {timestamp_str}")

    last_seen = _get_last_seen(exclude_session=exclude_session)
    if last_seen:
        ago, when = last_seen
        parts.append(f"Last seen: {ago} ago ({when})")
    else:
        parts.append("Last seen: this is our first conversation")

    personality = _read_about_file('personality.md')
    if personality:
        parts.append(personality)

    custom = _read_about_file('custom-vadar.md')
    if custom:
        parts.append(custom)

    how_to = _read_about_file('how-to-respond.md')
    if how_to:
        parts.append(how_to)

    user_about = _read_about_file('user.md')
    if user_about:
        parts.append(f"About the user:\n{user_about}")

    roleplay = _read_about_file('roleplay.md')
    if roleplay:
        parts.append("Roleplay (active):")
        parts.append(roleplay)
        roleplay_extras = _read_about_file('roleplay-extras.md')
        if roleplay_extras:
            parts.append("Roleplay extras:")
            parts.append(roleplay_extras)

    mem_section = _build_memory_section()
    if mem_section:
        parts.append(mem_section)

    ping_time = _read_ping_time()
    if is_ping:
        parts.append("")
        parts.append(f"[SYSTEM PING] The user has been silent. You may reply briefly or respond with exactly '(silence)' to stay quiet. This is not a user message — it is an automated check-in. Do not mention the ping mechanism to the user.")
    elif ping_time == 0:
        parts.append("Ping: disabled.")
    else:
        parts.append(f"Ping: every {ping_time}s of silence, you receive an automated system check-in. You may reply or stay silent.")

    return '\n\n'.join(parts)


def _strip_thinking(text):
    result = text
    while True:
        start = result.find(_THINK_START)
        if start == -1:
            break
        end = result.find(_THINK_END, start)
        if end == -1:
            result = result[:start]
            break
        result = result[:start] + result[end + len(_THINK_END):]
    return result.strip()


class _ThinkingStripper:
    def __init__(self):
        self.buffer = ''
        self.in_thinking = False
        self.output = []

    def feed(self, chunk):
        self.buffer += chunk
        while self.buffer:
            if self.in_thinking:
                end_idx = self.buffer.find(_THINK_END)
                if end_idx != -1:
                    self.buffer = self.buffer[end_idx + len(_THINK_END):]
                    self.in_thinking = False
                    continue
                partial = self._partial_match(self.buffer, _THINK_END)
                if partial > 0:
                    self.buffer = self.buffer[-partial:]
                else:
                    self.buffer = ''
                return None

            start_idx = self.buffer.find(_THINK_START)
            if start_idx != -1:
                before = self.buffer[:start_idx]
                self.buffer = self.buffer[start_idx + len(_THINK_START):]
                self.in_thinking = True
                if before:
                    return before
                continue

            partial = self._partial_match(self.buffer, _THINK_START)
            if partial > 0:
                safe = self.buffer[:-partial]
                self.buffer = self.buffer[-partial:]
                if safe:
                    return safe
                return None

            output = self.buffer
            self.buffer = ''
            return output

    def flush(self):
        if self.in_thinking:
            return ''
        if self.buffer:
            output = self.buffer
            self.buffer = ''
            return output
        return ''

    @staticmethod
    def _partial_match(buf, marker):
        for i in range(1, min(len(marker), len(buf)) + 1):
            if marker.startswith(buf[-i:]):
                return i
        return 0


def vadar_chat_stream(user_message, conversation_history=None, model_name='vadar'):
    config = _get_config(model_name)
    ok, err = vadar_load_model(model_name)
    if err:
        yield f"VADAR error: {err}"
        return

    import ollama

    messages = []
    sys_prompt = vadar_get_system_prompt()
    messages.append({'role': 'system', 'content': sys_prompt})

    if conversation_history:
        for msg in conversation_history:
            messages.append(msg)

    messages.append({'role': 'user', 'content': user_message})

    stripper = _ThinkingStripper()
    try:
        for chunk in ollama.chat(
            model=config['ollama_name'],
            messages=messages,
            stream=True,
            options={
                'num_predict': 2048,
                'temperature': config['temperature'],
                'top_p': config['top_p'],
                'top_k': config['top_k'],
                'repeat_penalty': config['repeat_penalty'],
            },
        ):
            content = chunk.get('message', {}).get('content', '')
            if not content:
                continue
            output = stripper.feed(content)
            if output:
                yield output

        remaining = stripper.flush()
        if remaining:
            yield remaining
    except Exception as e:
        yield f"VADAR stream error: {e}"


def _list_sessions():
    if not os.path.isdir(VADAR_SESSIONS_DIR):
        return []
    sessions = []
    for entry in os.listdir(VADAR_SESSIONS_DIR):
        full = os.path.join(VADAR_SESSIONS_DIR, entry)
        if not os.path.isdir(full):
            continue
        log_path = os.path.join(full, 'log.txt')
        if os.path.exists(log_path) and os.path.getsize(log_path) > 0:
            mtime = os.path.getmtime(log_path)
            sessions.append((mtime, entry))
    sessions.sort(reverse=True)
    return sessions


def _load_session_log(session_name):
    log_path = os.path.join(VADAR_SESSIONS_DIR, session_name, 'log.txt')
    if not os.path.exists(log_path):
        return []
    conversation = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            current_role = None
            current_content = []
            for line in f:
                if line.startswith('[USER] '):
                    if current_role and current_content:
                        conversation.append({
                            'role': 'user' if current_role == 'USER' else 'assistant',
                            'content': '\n'.join(current_content).strip()
                        })
                    current_role = 'USER'
                    current_content = [line[7:].rstrip()]
                elif line.startswith('[VADAR] '):
                    if current_role and current_content:
                        conversation.append({
                            'role': 'user' if current_role == 'USER' else 'assistant',
                            'content': '\n'.join(current_content).strip()
                        })
                    current_role = 'VADAR'
                    current_content = [line[8:].rstrip()]
                elif line.startswith('[SYSTEM]'):
                    continue
                else:
                    if current_role:
                        current_content.append(line.rstrip())
            if current_role and current_content:
                conversation.append({
                    'role': 'user' if current_role == 'USER' else 'assistant',
                    'content': '\n'.join(current_content).strip()
                })
    except Exception:
        pass
    return conversation


def _est_tokens(text):
    return max(1, len(text) // 4)


def _conversation_token_count(conversation):
    total = 0
    for msg in conversation:
        total += _est_tokens(msg.get('content', ''))
        total += 4
    return total


def _trim_conversation(conversation, context_length):
    total = _conversation_token_count(conversation)
    if total <= context_length:
        return conversation
    drop = max(2, int(total * 0.05))
    drop_tokens = 0
    i = 0
    while i < len(conversation) and drop_tokens < drop:
        drop_tokens += _est_tokens(conversation[i].get('content', '')) + 4
        i += 1
    return conversation[i:]


def _do_ping(session_dir, log_path, conversation, ping_count, model_name='vadar'):
    config = _get_config(model_name)
    ping_interval = _read_ping_time()
    if ping_interval == 0:
        return False, conversation

    ts = time.strftime("%Y/%m/%d %H:%M:%S")
    ping_msg = f"[SYSTEM PING #{ping_count}] {ts} — The user has been silent. You may reply briefly or respond with '(silence)' to stay quiet."

    sys_prompt = vadar_get_system_prompt(exclude_session=os.path.basename(session_dir), is_ping=True)
    messages = [{'role': 'system', 'content': sys_prompt}]
    messages.extend(conversation)
    messages.append({'role': 'system', 'content': ping_msg})

    try:
        import ollama
        response = ollama.chat(
            model=config['ollama_name'],
            messages=messages,
            stream=False,
            options={
                'num_predict': 512,
                'temperature': config['temperature'],
                'top_p': config['top_p'],
                'top_k': config['top_k'],
                'repeat_penalty': config['repeat_penalty'],
            },
        )
        raw = response.get('message', {}).get('content', '')
        reply = _strip_thinking(raw).strip()

        if not reply or reply.lower() in ('(silence)', 'silence', 'none', ''):
            return False, conversation

        print(f"\n[VADAR]: {reply}\n")
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[VADAR] {reply}\n")
        except Exception:
            pass

        conversation.append({'role': 'assistant', 'content': reply})
        ctx_len = _context_lengths.get(model_name, 4096)
        conversation = _trim_conversation(conversation, ctx_len)
        return True, conversation
    except Exception:
        return False, conversation


def vadar_interactive(resume_session=None, model_name='vadar'):
    config = _get_config(model_name)
    ok, err = vadar_load_model(model_name)
    if err:
        print(f"VADAR is not available — {err}")
        return False

    ctx_len = _context_lengths.get(model_name, 4096)

    if resume_session:
        session_name = resume_session
        session_dir = os.path.join(VADAR_SESSIONS_DIR, session_name)
        if not os.path.isdir(session_dir):
            print(f"Session not found: {session_name}")
            return False
        conversation = _load_session_log(session_name)
        print(f"\nResuming session: {session_name} ({len(conversation)} messages loaded)")
    else:
        session_name = time.strftime("%Y%m%d_%H%M%S") + "_chat"
        session_dir = os.path.join(VADAR_SESSIONS_DIR, session_name)
        os.makedirs(session_dir, exist_ok=True)
        conversation = []

    log_path = os.path.join(session_dir, 'log.txt')

    def _log(role, content):
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{role.upper()}] {content}\n")
        except Exception:
            pass

    ping_interval = _read_ping_time()
    ping_stop = threading.Event()
    last_user_activity = [time.time()]
    ping_count = [0]
    vadar_busy = [False]
    busy_lock = threading.Lock()

    def ping_thread():
        if ping_interval == 0:
            return
        while not ping_stop.is_set():
            ping_stop.wait(timeout=1)
            if ping_stop.is_set():
                break
            with busy_lock:
                if vadar_busy[0]:
                    last_user_activity[0] = time.time()
                    continue
            elapsed = time.time() - last_user_activity[0]
            if elapsed < ping_interval:
                continue
            with busy_lock:
                vadar_busy[0] = True
            ping_count[0] += 1
            nonlocal_conversation = conversation
            _, nonlocal_conversation = _do_ping(session_dir, log_path, nonlocal_conversation, ping_count[0], model_name)
            with busy_lock:
                vadar_busy[0] = False
            last_user_activity[0] = time.time()

    if ping_interval > 0:
        ping_thread_obj = threading.Thread(target=ping_thread, daemon=True)
        ping_thread_obj.start()
    else:
        ping_thread_obj = None

    print(f"\n{'='*60}")
    print(f"VADAR — Project Eva Chat")
    print(f"{'='*60}")
    print(f"Model: {config['display_name']}")
    print(f"Context: {ctx_len} tokens")
    if ping_interval > 0:
        print(f"Ping: every {ping_interval}s of silence")
    else:
        print(f"Ping: disabled")
    print(f"Session: {session_name}")
    print(f"{'='*60}")
    print("Commands: 'exit' to quit, 'resume' to resume a previous session")
    print()

    if not resume_session:
        greeting = "Hey! I'm VADAR. What's on your mind?"
        print(f"[VADAR]: {greeting}\n")
        _log('assistant', greeting)

    while True:
        try:
            user_input = input("[You]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[VADAR]: Goodbye!")
            _log('user', '(session ended)')
            break

        last_user_activity[0] = time.time()
        ping_count[0] = 0

        with busy_lock:
            vadar_busy[0] = True

        if not user_input:
            with busy_lock:
                vadar_busy[0] = False
            continue

        if user_input.lower() in ('exit', 'quit'):
            print("\n[VADAR]: Goodbye!")
            _log('user', '(session ended)')
            break

        if user_input.lower() == 'resume':
            sessions = _list_sessions()
            if not sessions:
                print("[VADAR]: No previous sessions found.")
            else:
                print("\nAvailable sessions (most recent first):")
                for i, (mtime, name) in enumerate(sessions[:10], 1):
                    when = time.strftime("%Y/%m/%d %H:%M", time.localtime(mtime))
                    print(f"  {i}. {name} ({when})")
                print()
                try:
                    choice = input("Enter session name (or 'cancel'): ").strip()
                    if choice and choice.lower() != 'cancel':
                        ping_stop.set()
                        if ping_thread_obj:
                            ping_thread_obj.join(timeout=2)
                        return vadar_interactive(resume_session=choice, model_name=model_name)
                except (EOFError, KeyboardInterrupt):
                    print()
            with busy_lock:
                vadar_busy[0] = False
            continue

        _log('user', user_input)

        print()
        print("[VADAR]: ", end='', flush=True)
        full_response = []
        for chunk in vadar_chat_stream(user_input, conversation, model_name=model_name):
            print(chunk, end='', flush=True)
            full_response.append(chunk)
        print('\n')

        response_text = ''.join(full_response).strip()
        if response_text:
            _log('assistant', response_text)
            conversation.append({'role': 'user', 'content': user_input})
            conversation.append({'role': 'assistant', 'content': response_text})
            conversation = _trim_conversation(conversation, ctx_len)

        with busy_lock:
            vadar_busy[0] = False

    ping_stop.set()
    if ping_thread_obj is not None:
        ping_thread_obj.join(timeout=5)

    return True
