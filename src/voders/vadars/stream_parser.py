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
        self.state = 'OUTSIDE'
        self.current_tag = ''
        self.tag_content = ''
        self.tag_start_time = 0
        self.tag_char_count = 0
        self.last_progress = 0
        self.collected = []
        self.in_model_think = False
        self.model_think_chars = 0
        self.model_think_start = 0
        self.model_think_last_print = 0

    def feed(self, chunk):
        self.buffer += chunk
        self.collected.append(chunk)

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
                else:
                    safe = self._safe_split_marker(self.buffer, _THINK_END)
                    if safe < len(self.buffer):
                        self.model_think_chars += safe
                        self.buffer = self.buffer[safe:]
                    else:
                        self.model_think_chars += len(self.buffer)
                        self.buffer = ''
                    now = time.time()
                    if self.interactive and now - self.model_think_last_print > 0.5:
                        label = f'{self.agent_label} MODEL THINKING'
                        sys.stdout.write(f'\r[{label}]: {self.model_think_chars} chars...  ')
                        sys.stdout.flush()
                        self.model_think_last_print = now
                    break
                continue

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
                    self._process_outside(before)
                continue

            partial = self._check_partial(self.buffer, _THINK_START)
            if partial > 0:
                safe = self.buffer[:-partial]
                self.buffer = self.buffer[-partial:]
                if safe:
                    self._process_outside(safe)
                break

            if self.state == 'OUTSIDE':
                self._process_outside(self.buffer)
                self.buffer = ''
                break
            else:
                close_tag = f'</{self.current_tag}>'
                end_idx = self.buffer.find(close_tag)
                if end_idx != -1:
                    content = self.buffer[:end_idx]
                    self.tag_content += content
                    self.tag_char_count += len(content)
                    self.buffer = self.buffer[end_idx + len(close_tag):]
                    self._finish_tag()
                else:
                    safe = self._safe_split_marker(self.buffer, close_tag)
                    if safe > 0:
                        content = self.buffer[:safe]
                        self.tag_content += content
                        self.tag_char_count += len(content)
                        if not self.interactive:
                            self._stream_words(content)
                        else:
                            now = time.time()
                            if now - self.last_progress > 0.5:
                                label = _TAG_LABELS.get(self.current_tag, self.current_tag.upper())
                                sys.stdout.write(f'\r[{self.agent_label} {label}]: {self.tag_char_count} chars...  ')
                                sys.stdout.flush()
                                self.last_progress = now
                        self.buffer = self.buffer[safe:]
                    break

    def _process_outside(self, text):
        pos = 0
        while pos < len(text):
            m = _TAG_PATTERN.search(text, pos)
            if m is None:
                remaining = text[pos:]
                if remaining.strip():
                    if not self.interactive:
                        sys.stdout.write(remaining)
                        sys.stdout.flush()
                    else:
                        label = f'{self.agent_label}'
                        sys.stdout.write(f'\n[{label}]: {len(remaining)} chars...  ')
                        sys.stdout.flush()
                break
            before = text[pos:m.start()]
            if before.strip():
                if not self.interactive:
                    sys.stdout.write(before)
                    sys.stdout.flush()
                else:
                    label = f'{self.agent_label}'
                    sys.stdout.write(f'\n[{label}]: {len(before)} chars...  ')
                    sys.stdout.flush()
            tag_name = m.group(2)
            is_close = m.group(1) == '/'
            if not is_close and tag_name not in ('EOS_REPLY', 'EOS_ACT', 'EOS_DONE'):
                self.state = 'INSIDE'
                self.current_tag = tag_name
                self.tag_content = ''
                self.tag_char_count = 0
                self.tag_start_time = time.time()
                self.last_progress = self.tag_start_time
                label = _TAG_LABELS.get(tag_name, tag_name.upper())
                if not self.interactive or tag_name == 'reply':
                    sys.stdout.write(f'\n[{self.agent_label} {label}]: ')
                    sys.stdout.flush()
                else:
                    sys.stdout.write(f'\n[{self.agent_label} {label}]: 0 chars...  ')
                    sys.stdout.flush()
            elif tag_name in ('EOS_REPLY', 'EOS_ACT', 'EOS_DONE'):
                sys.stdout.write(f'\n[{tag_name}]\n')
                sys.stdout.flush()
            pos = m.end()

    def _stream_words(self, content):
        words = content.split(' ')
        for i, word in enumerate(words):
            if i > 0:
                sys.stdout.write(' ')
                sys.stdout.flush()
            for chunk in _split_word(word):
                sys.stdout.write(chunk)
                sys.stdout.flush()

    def _finish_tag(self):
        elapsed = time.time() - self.tag_start_time
        label = _TAG_LABELS.get(self.current_tag, self.current_tag.upper())
        if not self.interactive:
            if self.current_tag == 'reply':
                sys.stdout.write(f'\n[{self.agent_label} {label}]: {self.tag_char_count} chars ({elapsed:.1f}s)\n')
            else:
                sys.stdout.write(f'\n[{self.agent_label} {label}]: {self.tag_char_count} chars ({elapsed:.1f}s)\n')
        else:
            if self.current_tag == 'reply':
                sys.stdout.write(f'\n')
            else:
                sys.stdout.write(f'\r[{self.agent_label} {label}]: {self.tag_char_count} chars ({elapsed:.1f}s)  \n')
        sys.stdout.flush()
        self.state = 'OUTSIDE'
        self.current_tag = ''
        self.tag_content = ''
        self.tag_char_count = 0

    def _safe_split_marker(self, buf, marker):
        for i in range(1, min(len(marker), len(buf)) + 1):
            if marker.startswith(buf[-i:]):
                return len(buf) - i
        return len(buf)

    def _check_partial(self, buf, marker):
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
        if self.state != 'OUTSIDE' and self.tag_content:
            self._finish_tag()
        self.buffer = ''

    def get_full_text(self):
        return ''.join(self.collected)
