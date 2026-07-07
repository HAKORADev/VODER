import os
import time
import copy

from voders.vadars import VADAR_SESSIONS_DIR


class ContextManager:
    def __init__(self, session_dir, max_context_tokens=None, slide_ratio=None,
                 memory_cap_ratio=None, global_context_cap_ratio=None):
        try:
            from voder import vadar_load_config
            config = vadar_load_config()
            ctx_len = config.get('context_length', 262144)
            slide_thresh = config.get('sliding_window_threshold_percent', 95) / 100.0
            mem_cap = config.get('memory_cap_percent', 20) / 100.0
            gc_cap = config.get('global_context_cap_percent', 15) / 100.0
        except Exception:
            ctx_len = 262144
            slide_thresh = 0.95
            mem_cap = 0.20
            gc_cap = 0.15

        self.session_dir = session_dir
        self.max_tokens = max_context_tokens if max_context_tokens is not None else ctx_len
        self.slide_ratio = slide_ratio if slide_ratio is not None else slide_thresh
        self.memory_cap_ratio = memory_cap_ratio if memory_cap_ratio is not None else mem_cap
        self.global_context_cap_ratio = global_context_cap_ratio if global_context_cap_ratio is not None else gc_cap
        self.messages = []
        self.dropped_count = 0
        self.memory_tokens = 0
        self.context_file = os.path.join(session_dir, 'context.txt')
        self.log_file = os.path.join(session_dir, 'log.txt')
        os.makedirs(session_dir, exist_ok=True)

    def add(self, role, content, tool_call=None, is_memory=False):
        msg = {'role': role, 'content': content, 'is_memory': is_memory}
        if tool_call:
            msg['tool_call'] = tool_call

        if is_memory:
            mem_tokens = self._estimate_tokens(content)
            max_mem_tokens = int(self.max_tokens * self.memory_cap_ratio)
            if self.memory_tokens + mem_tokens > max_mem_tokens:
                self._evict_oldest_memory(mem_tokens)
            self.memory_tokens += self._estimate_tokens(content)

        self.messages.append(msg)
        self._save_log(msg)
        self._slide_if_needed()

    def _evict_oldest_memory(self, needed_tokens):
        max_mem_tokens = int(self.max_tokens * self.memory_cap_ratio)
        while self.memory_tokens + needed_tokens > max_mem_tokens:
            evicted = False
            for i, m in enumerate(self.messages):
                if m.get('is_memory'):
                    self.memory_tokens -= self._estimate_tokens(m['content'])
                    self.messages.pop(i)
                    evicted = True
                    break
            if not evicted:
                break

    def get_messages(self):
        return list(self.messages)

    def get_for_inference(self):
        return [{'role': m['role'], 'content': m['content']} for m in self.messages]

    def _estimate_tokens(self, text):
        return len(text) // 4 + 1

    def _total_tokens(self):
        return sum(self._estimate_tokens(m['content']) for m in self.messages)

    def _slide_if_needed(self):
        total = self._total_tokens()
        if total <= self.max_tokens:
            return
        drop_ratio = 1.0 - self.slide_ratio
        drop_count = max(1, int(len(self.messages) * drop_ratio))
        if drop_count >= len(self.messages):
            drop_count = len(self.messages) - 1
        dropped = []
        kept = list(self.messages)
        for _ in range(drop_count):
            for i in range(len(kept)):
                if kept[i]['role'] != 'system':
                    if kept[i].get('is_memory'):
                        self.memory_tokens -= self._estimate_tokens(kept[i]['content'])
                    dropped.append(kept.pop(i))
                    break
            else:
                break
        self.messages = kept
        self.dropped_count += len(dropped)
        self._save_context()

    def _save_log(self, msg):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                ts = time.strftime('%Y/%m/%d %H:%M:%S')
                f.write(f"[{ts}] {msg['role'].upper()}: {msg['content'][:500]}")
                if msg.get('tool_call'):
                    f.write(f" [TOOL_CALL: {msg['tool_call']}]")
                f.write('\n')
        except Exception:
            pass

    def _save_context(self):
        try:
            with open(self.context_file, 'w', encoding='utf-8') as f:
                for m in self.messages:
                    f.write(f"=== {m['role'].upper()} ===\n")
                    f.write(m['content'])
                    f.write('\n\n')
        except Exception:
            pass


def create_session(session_type='interactive'):
    ts = time.strftime('%Y%m%d_%H%M%S')
    session_name = f"{ts}_{session_type}"
    session_dir = os.path.join(VADAR_SESSIONS_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    for fname in ['inputs.txt', 'outputs.txt', 'acts.txt', 'log.txt', 'context.txt']:
        fpath = os.path.join(session_dir, fname)
        if not os.path.exists(fpath):
            with open(fpath, 'w', encoding='utf-8') as f:
                pass
    return session_dir, session_name


def log_input(session_dir, text):
    fpath = os.path.join(session_dir, 'inputs.txt')
    try:
        with open(fpath, 'a', encoding='utf-8') as f:
            ts = time.strftime('%Y/%m/%d %H:%M:%S')
            f.write(f"[{ts}] {text}\n")
    except Exception:
        pass


def log_output(session_dir, text):
    fpath = os.path.join(session_dir, 'outputs.txt')
    try:
        with open(fpath, 'a', encoding='utf-8') as f:
            ts = time.strftime('%Y/%m/%d %H:%M:%S')
            f.write(f"[{ts}] {text}\n")
    except Exception:
        pass


def log_act(session_dir, title, command, result, success):
    fpath = os.path.join(session_dir, 'acts.txt')
    try:
        with open(fpath, 'a', encoding='utf-8') as f:
            ts = time.strftime('%Y/%m/%d %H:%M:%S')
            status = 'SUCCESS' if success else 'FAILED'
            f.write(f"[{ts}] ACT '{title}': {command}\n")
            f.write(f"  RESULT: {status}\n")
            if result:
                for line in str(result).split('\n')[-20:]:
                    f.write(f"    {line}\n")
            f.write('\n')
    except Exception:
        pass
