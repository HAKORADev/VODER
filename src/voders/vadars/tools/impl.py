import os
import re
import sys
import subprocess
import tempfile
import shutil
import json
import math as _math

from voders.vadars.tools import register_tool
from voders.vadars import VADAR_MEMORIES_DIR, VADAR_SUPPORTED_LIBS_FILE

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}
_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.xml', '.csv', '.tsv', '.html', '.css', '.log', '.chain'}


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _ffprobe_duration(path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', path],
            capture_output=True, text=True, timeout=10,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def _parse_time_range(spec):
    if not spec or '-' not in spec:
        return None, None
    parts = spec.split('-')
    if len(parts) != 2:
        return None, None
    def parse_ts(ts):
        ts = ts.strip()
        if ':' not in ts:
            return float(ts)
        parts = ts.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    try:
        return parse_ts(parts[0]), parse_ts(parts[1])
    except Exception:
        return None, None


def _format_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


@register_tool('list')
def tool_list(args):
    parts = args.strip().split(None, 1)
    list_type = parts[0].lower() if parts else 'all'
    path = parts[1].strip() if len(parts) > 1 else _PROJECT_ROOT
    if not _is_within_project(path):
        return f"Error: path '{path}' is outside the VODER project directory. I can only list files inside the project."
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    files = []
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
        for fname in fnames:
            files.append(os.path.join(root, fname))
    if list_type == 'all' or list_type == '':
        counts = {'videos': 0, 'images': 0, 'audios': 0, 'texts': 0, 'others': 0}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in _VIDEO_EXTENSIONS: counts['videos'] += 1
            elif ext in _IMAGE_EXTENSIONS: counts['images'] += 1
            elif ext in _AUDIO_EXTENSIONS: counts['audios'] += 1
            elif ext in _TEXT_EXTENSIONS: counts['texts'] += 1
            else: counts['others'] += 1
        return f"{counts['videos']} videos, {counts['images']} images, {counts['audios']} audios, {counts['texts']} text files, {counts['others']} others (total: {len(files)})"
    ext_map = {
        'videos': _VIDEO_EXTENSIONS, 'images': _IMAGE_EXTENSIONS,
        'audios': _AUDIO_EXTENSIONS, 'texts': _TEXT_EXTENSIONS,
    }
    if list_type in ext_map:
        filtered = [f for f in files if os.path.splitext(f)[1].lower() in ext_map[list_type]]
        if not filtered:
            return f"No {list_type} found in {path}"
        return '\n'.join(sorted(filtered)[:200])
    if list_type == 'others':
        filtered = [f for f in files if os.path.splitext(f)[1].lower() not in
                    (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _TEXT_EXTENSIONS)]
        if not filtered:
            return f"No other files found in {path}"
        return '\n'.join(sorted(filtered)[:200])
    if list_type.startswith('.'):
        ext = list_type.lower()
        filtered = [f for f in files if os.path.splitext(f)[1].lower() == ext]
        if not filtered:
            return f"No *{ext} files found in {path}"
        return '\n'.join(sorted(filtered)[:200])
    return f"Unknown list type '{list_type}'. Use: videos, images, audios, texts, others, all, or .extension"


@register_tool('search')
def tool_search(args):
    m = re.match(r'^(.+?)\s+path\s+(\S+)(?:\s+formats\s+(.+))?$', args.strip())
    if not m:
        m2 = re.match(r'^(.+?)\s+(\S+)(?:\s+(.+))?$', args.strip())
        if not m2:
            return "Usage: search <query> path <path> [formats <format1,format2,...>]"
        query = m2.group(1).strip('"\'')
        path = m2.group(2).strip('"\'')
        formats = m2.group(3)
    else:
        query = m.group(1).strip('"\'')
        path = m.group(2).strip('"\'')
        formats = m.group(3)
    if not _is_within_project(path):
        return f"Error: path '{path}' is outside the VODER project directory."
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    results = []
    ext_filter = None
    if formats:
        ext_filter = set()
        for fmt in formats.split(','):
            fmt = fmt.strip().lower()
            if not fmt.startswith('.'):
                fmt = '.' + fmt
            ext_filter.add(fmt)
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.git')]
        for fname in fnames:
            if query.lower() in fname.lower():
                if ext_filter and os.path.splitext(fname)[1].lower() not in ext_filter:
                    continue
                results.append(os.path.join(root, fname))
    if not results:
        return f"No files matching '{query}' found in {path}"
    results.sort()
    return '\n'.join(results[:200])


@register_tool('read')
def tool_read(args, session_dir=None, act_outputs=None):
    parts = args.strip().split(None, 1)
    if not parts:
        return "Usage: read <path|act_title> [start-end]"
    target = parts[0].strip('"\'')
    range_spec = parts[1].strip() if len(parts) > 1 else None
    content = None
    if act_outputs and target in act_outputs:
        content = act_outputs[target]
    elif os.path.isfile(target) and _is_within_project(target):
        try:
            with open(target, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    elif os.path.isfile(target) and not _is_within_project(target):
        return f"Error: '{target}' is outside the VODER project directory. I can only read files inside the project."
    else:
        if act_outputs:
            available = ', '.join(act_outputs.keys())
            return f"Target '{target}' not found. Available act titles: {available}"
        return f"Target '{target}' not found."
    if content is None:
        return f"Could not read '{target}'."
    lines = content.split('\n')
    total = len(lines)
    if range_spec and '-' in range_spec:
        start_str, end_str = range_spec.split('-', 1)
        try:
            start = int(start_str.strip())
            end = int(end_str.strip())
        except ValueError:
            return f"Invalid line range '{range_spec}'. Use start-end (e.g., 20-30)."
        if start < 1: start = 1
        if end > total: end = total
        if start > end:
            return f"Start line ({start}) must be smaller than end line ({end})."
        selected = lines[start-1:end]
        result = f"Lines {start}-{end} of {total}:\n"
        for i, line in enumerate(selected, start=start):
            result += f"{i:6d}: {line}\n"
        return result
    preview = '\n'.join(lines[:100])
    return f"Total lines: {total}\n--- First 100 lines ---\n{preview}"


@register_tool('memory_read')
def tool_memory_read(args):
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        return "Usage: memory_read <vadar|user> <id>"
    category = parts[0].lower()
    mem_id = parts[1].strip()
    if category not in ('vadar', 'user'):
        return "Category must be 'vadar' or 'user'."
    path = os.path.join(VADAR_MEMORIES_DIR, category, f"{mem_id}.txt")
    if not os.path.exists(path):
        return f"Memory {category}/{mem_id} not found."
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading memory: {e}"


@register_tool('memory_write')
def tool_memory_write(args):
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        return "Usage: memory_write <vadar|user> <content>"
    category = parts[0].lower()
    content = parts[1].strip()
    if category not in ('vadar', 'user'):
        return "Category must be 'vadar' or 'user'."
    mem_dir = os.path.join(VADAR_MEMORIES_DIR, category)
    os.makedirs(mem_dir, exist_ok=True)
    existing = [f for f in os.listdir(mem_dir) if f.endswith('.txt')]
    next_id = len(existing) + 1
    path = os.path.join(mem_dir, f"{next_id}.txt")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Memory written: {category}/{next_id}"
    except Exception as e:
        return f"Error writing memory: {e}"


@register_tool('memory_edit')
def tool_memory_edit(args):
    parts = args.strip().split(None, 2)
    if len(parts) < 3:
        return "Usage: memory_edit <vadar|user> <id> <content>"
    category = parts[0].lower()
    mem_id = parts[1].strip()
    content = parts[2].strip()
    if category not in ('vadar', 'user'):
        return "Category must be 'vadar' or 'user'."
    path = os.path.join(VADAR_MEMORIES_DIR, category, f"{mem_id}.txt")
    if not os.path.exists(path):
        return f"Memory {category}/{mem_id} not found. Use memory_write to create new memories."
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Memory updated: {category}/{mem_id}"
    except Exception as e:
        return f"Error editing memory: {e}"


@register_tool('memory_delete')
def tool_memory_delete(args):
    parts = args.strip().split(None, 1)
    if len(parts) < 2:
        return "Usage: memory_delete <vadar|user> <id>"
    category = parts[0].lower()
    mem_id = parts[1].strip()
    if category not in ('vadar', 'user'):
        return "Category must be 'vadar' or 'user'."
    path = os.path.join(VADAR_MEMORIES_DIR, category, f"{mem_id}.txt")
    if not os.path.exists(path):
        return f"Memory {category}/{mem_id} not found."
    try:
        os.remove(path)
        return f"Memory deleted: {category}/{mem_id}"
    except Exception as e:
        return f"Error deleting memory: {e}"


@register_tool('calculate')
def tool_calculate(args):
    code = args.strip()
    if not code:
        return "Usage: calculate <python code using supported libraries>"
    try:
        with open(VADAR_SUPPORTED_LIBS_FILE, 'r') as f:
            supported = [line.strip() for line in f if line.strip()]
    except Exception:
        supported = ['math']
    safe_globals = {'__builtins__': {}}
    for lib in supported:
        try:
            safe_globals[lib] = __import__(lib)
        except ImportError:
            pass
    safe_globals['__builtins__'] = {
        'print': print, 'range': range, 'len': len, 'int': int, 'float': float,
        'str': str, 'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple,
        'set': set, 'abs': abs, 'min': min, 'max': max, 'sum': sum, 'round': round,
        'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'map': map,
        'filter': filter, 'True': True, 'False': False, 'None': None,
    }
    old_stdout = sys.stdout
    captured = []
    class _Capture:
        def write(self, text):
            captured.append(text)
        def flush(self):
            pass
    sys.stdout = _Capture()
    try:
        exec(code, safe_globals)
        output = ''.join(captured)
        if not output:
            for key in safe_globals:
                if key != '__builtins__' and not key.startswith('_'):
                    val = safe_globals[key]
                    if not callable(val) and not hasattr(val, '__module__'):
                        output = str(val)
                        break
        return output.strip() if output.strip() else "(code executed, no output)"
    except Exception as e:
        return f"Error: {e}"
    finally:
        sys.stdout = old_stdout


@register_tool('look')
def tool_look(args, model=None, processor=None):
    path = args.strip().strip('"\'')
    if not path:
        return "Usage: look <image_path|url>"
    if not os.path.exists(path):
        return f"Image not found: {path}"
    if not _is_within_project(path):
        return f"Error: '{path}' is outside the VODER project directory."
    ext = os.path.splitext(path)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
        return f"'{path}' does not appear to be an image file."
    if model is None or processor is None:
        return f"Image found at {path} (model not loaded — cannot analyze visually). File size: {os.path.getsize(path)} bytes."
    return f"Image at {path} loaded. Model analysis would be performed here."


@register_tool('listen')
def tool_listen(args, model=None, processor=None):
    parts = args.strip().split(None, 1)
    path = parts[0].strip('"\'') if parts else ''
    range_spec = parts[1].strip() if len(parts) > 1 else None
    if not path:
        return "Usage: listen <audio_path|url> [HH:MM:SS-HH:MM:SS]"
    if not os.path.exists(path):
        return f"Audio not found: {path}"
    if not _is_within_project(path):
        return f"Error: '{path}' is outside the VODER project directory."
    dur = _ffprobe_duration(path)
    if dur is None:
        return f"Could not determine duration of {path}"
    if range_spec:
        start, end = _parse_time_range(range_spec)
        if start is None or end is None:
            return f"Invalid time range '{range_spec}'. Use HH:MM:SS-HH:MM:SS format."
        if start >= end:
            return f"Start time ({_format_timestamp(start)}) must be before end time ({_format_timestamp(end)})."
        if end > dur:
            end = dur
        return f"Audio segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)} from {path}. Model analysis would be performed here."
    return f"Audio: {path}\nDuration: {_format_timestamp(dur)}\nModel analysis would be performed here."


@register_tool('watch')
def tool_watch(args, model=None, processor=None):
    parts = args.strip().split(None, 1)
    path = parts[0].strip('"\'') if parts else ''
    range_spec = parts[1].strip() if len(parts) > 1 else None
    if not path:
        return "Usage: watch <video_path|url> [HH:MM:SS-HH:MM:SS]"
    if not os.path.exists(path):
        return f"Video not found: {path}"
    if not _is_within_project(path):
        return f"Error: '{path}' is outside the VODER project directory."
    ext = os.path.splitext(path)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        return f"'{path}' does not appear to be a video file."
    dur = _ffprobe_duration(path)
    if dur is None:
        return f"Could not determine duration of {path}"
    if range_spec:
        start, end = _parse_time_range(range_spec)
        if start is None or end is None:
            return f"Invalid time range '{range_spec}'. Use HH:MM:SS-HH:MM:SS format."
        if start >= end:
            return f"Start time ({_format_timestamp(start)}) must be before end time ({_format_timestamp(end)})."
        if end > dur:
            end = dur
        return f"Video segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)} from {path}. Model analysis would be performed here."
    return f"Video: {path}\nDuration: {_format_timestamp(dur)}\nModel analysis would be performed here."
