import re
import sys
import time


_TAG_PATTERN = re.compile(r'<(/?)(thinking|decide|reply|act|tool_call|eval|EOS_REPLY|EOS_ACT|EOS_DONE)([^>]*)>')

_TAG_LABELS = {
    'thinking': 'THINKING',
    'decide': 'DECIDE',
    'reply': 'REPLY',
    'act': 'ACT',
    'tool_call': 'TOOL',
    'eval': 'EVAL',
}

_THINK_START = "<|channel>thought\n"
_THINK_END = "<channel|>"

_KNOWN_TAG_NAMES = ['thinking', 'decide', 'reply', 'act', 'tool_call', 'eval',
                    'EOS_REPLY', 'EOS_ACT', 'EOS_DONE',
                    '/thinking', '/decide', '/reply', '/act', '/tool_call', '/eval']


def _split_word(word):
    if len(word) <= 5:
        return [word]
    chunks = []
    i = 0
    while i < len(word):
        remaining = len(word) - i
        if remaining <= 7:
            chunks.append(word[i:])
            break
        chunks.append(word[i:i+5])
        i += 5
    return chunks


class StreamParser:
    def __init__(self, agent_label='VADAR', interactive=False):
        self.agent_label = agent_label
        self.interactive = interactive
        self.buffer = ''
        self.collected = []
        self.in_model_think = False
        self.model_think_chars = 0
        self.model_think_start = 0
        self.model_think_last_print = 0

    def feed(self, chunk):
        self.buffer += chunk
        self.collected.append(chunk)
        self._process()

    def _process(self):
        while self.buffer:
            if self.in_model_think:
                end_idx = self.buffer.find(_THINK_END)
                if end_idx != -1:
                    self.model_think_chars += end_idx
                    self.buffer = self.buffer[end_idx + len(_THINK_END):]
                    self.in_model_think = False
                    elapsed = time.time() - self.model_think_start
                    label = f'{self.agent_label} MODEL THINKING'
                    sys.stdout.write(f'\r[{label}]: {self.model_think_chars} chars ({elapsed:.1f}s)  \n')
                    sys.stdout.flush()
                    self.model_think_chars = 0
                    continue
                else:
                    partial = self._partial_match(self.buffer, _THINK_END)
                    if partial > 0:
                        safe = self.buffer[:-partial]
                        self.model_think_chars += len(safe)
                        self.buffer = self.buffer[-partial:]
                    else:
                        self.model_think_chars += len(self.buffer)
                        self.buffer = ''
                    now = time.time()
                    if self.interactive and now - self.model_think_last_print > 0.5:
                        label = f'{self.agent_label} MODEL THINKING'
                        sys.stdout.write(f'\r[{label}]: {self.model_think_chars} chars...  ')
                        sys.stdout.flush()
                        self.model_think_last_print = now
                    return

            if _THINK_START in self.buffer:
                idx = self.buffer.find(_THINK_START)
                before = self.buffer[:idx]
                self.buffer = self.buffer[idx + len(_THINK_START):]
                self.in_model_think = True
                self.model_think_chars = 0
                self.model_think_start = time.time()
                self.model_think_last_print = self.model_think_start
                label = f'{self.agent_label} MODEL THINKING'
                if self.interactive:
                    sys.stdout.write(f'\n[{label}]: 0 chars...  ')
                    sys.stdout.flush()
                if before:
                    self._display_raw(before)
                continue

            partial_think = self._partial_match(self.buffer, _THINK_START)
            if partial_think > 0:
                safe = self.buffer[:-partial_think]
                self.buffer = self.buffer[-partial_think:]
                if safe:
                    self._display_raw(safe)
                return

            m = _TAG_PATTERN.search(self.buffer)
            if m is None:
                lt = self.buffer.rfind('<')
                if lt >= 0:
                    possible = self.buffer[lt:]
                    if self._could_be_tag(possible):
                        safe = self.buffer[:lt]
                        if safe:
                            self._display_raw(safe)
                        self.buffer = possible
                        return
                self._display_raw(self.buffer)
                self.buffer = ''
                return

            before = self.buffer[:m.start()]
            if before:
                self._display_raw(before)

            tag_name = m.group(2)
            is_close = m.group(1) == '/'
            tag_end = m.end()

            if tag_name in ('EOS_REPLY', 'EOS_ACT', 'EOS_DONE'):
                sys.stdout.write(f'\n[{tag_name}]\n')
                sys.stdout.flush()
                self.buffer = self.buffer[tag_end:]
                continue

            if is_close:
                self.buffer = self.buffer[tag_end:]
                continue

            close_str = f'</{tag_name}>'
            close_idx = self.buffer.find(close_str, tag_end)

            if close_idx != -1:
                content = self.buffer[tag_end:close_idx]
                self._display_tag_complete(tag_name, content)
                self.buffer = self.buffer[close_idx + len(close_str):]
                continue
            else:
                partial_close = self._partial_match(self.buffer[tag_end:], close_str)
                if partial_close > 0:
                    safe_content = self.buffer[tag_end:-partial_close]
                    remaining = self.buffer[-partial_close:]
                    self._display_tag_start(tag_name)
                    if safe_content:
                        self._display_tag_content(tag_name, safe_content)
                    self.buffer = remaining
                    self._pending_tag = tag_name
                    self._pending_content = safe_content
                    self._pending_start = time.time()
                    return
                else:
                    content = self.buffer[tag_end:]
                    self._display_tag_start(tag_name)
                    if content:
                        self._display_tag_content(tag_name, content)
                    self.buffer = ''
                    self._pending_tag = tag_name
                    self._pending_content = content
                    self._pending_start = time.time()
                    return

    def _display_raw(self, text):
        if not text.strip():
            return
        if not self.interactive:
            sys.stdout.write(text)
            sys.stdout.flush()
        else:
            label = self.agent_label
            sys.stdout.write(f'\n[{label}]: {len(text)} chars...  ')
            sys.stdout.flush()

    def _display_tag_start(self, tag_name):
        label = _TAG_LABELS.get(tag_name, tag_name.upper())
        if not self.interactive or tag_name == 'reply':
            sys.stdout.write(f'\n[{self.agent_label} {label}]: ')
            sys.stdout.flush()
        else:
            sys.stdout.write(f'\n[{self.agent_label} {label}]: 0 chars...  ')
            sys.stdout.flush()

    def _display_tag_content(self, tag_name, content):
        if not self.interactive or tag_name == 'reply':
            words = content.split(' ')
            for i, word in enumerate(words):
                if i > 0:
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                for chunk in _split_word(word):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
        else:
            now = time.time()
            if now - getattr(self, '_last_progress', 0) > 0.5:
                label = _TAG_LABELS.get(tag_name, tag_name.upper())
                char_count = len(getattr(self, '_pending_content', ''))
                sys.stdout.write(f'\r[{self.agent_label} {label}]: {char_count} chars...  ')
                sys.stdout.flush()
                self._last_progress = now

    def _display_tag_complete(self, tag_name, content):
        label = _TAG_LABELS.get(tag_name, tag_name.upper())
        start_t = time.time()
        if not self.interactive or tag_name == 'reply':
            sys.stdout.write(f'\n[{self.agent_label} {label}]: ')
            sys.stdout.flush()
            words = content.split(' ')
            for i, word in enumerate(words):
                if i > 0:
                    sys.stdout.write(' ')
                    sys.stdout.flush()
                for chunk in _split_word(word):
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
            sys.stdout.write(f'\n')
        else:
            sys.stdout.write(f'\r[{self.agent_label} {label}]: {len(content)} chars  \n')
        sys.stdout.flush()

    def _could_be_tag(self, text):
        if not text.startswith('<'):
            return False
        for tag in _KNOWN_TAG_NAMES:
            if tag.startswith(text[1:]) or text[1:].startswith(tag):
                return True
        return False

    def _partial_match(self, buf, marker):
        for i in range(1, min(len(marker), len(buf)) + 1):
            if marker.startswith(buf[-i:]):
                return i
        return 0

    def flush(self):
        if self.in_model_think:
            elapsed = time.time() - self.model_think_start
            label = f'{self.agent_label} MODEL THINKING'
            sys.stdout.write(f'\r[{label}]: {self.model_think_chars} chars ({elapsed:.1f}s)  \n')
            sys.stdout.flush()
            self.in_model_think = False

        if hasattr(self, '_pending_tag') and self._pending_tag:
            label = _TAG_LABELS.get(self._pending_tag, self._pending_tag.upper())
            if not self.interactive or self._pending_tag == 'reply':
                pass
            else:
                sys.stdout.write(f'\r[{self.agent_label} {label}]: {len(self._pending_content)} chars  \n')
                sys.stdout.flush()
            self._pending_tag = ''
            self._pending_content = ''

        if self.buffer.strip():
            self._display_raw(self.buffer)

        self.buffer = ''

    def get_full_text(self):
        return ''.join(self.collected)
