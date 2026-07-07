import os
import re
import sys
import subprocess
import tempfile
import shutil
import json
import math as _math
import urllib.request

from voders.vadars.tools import register_tool
from voders.vadars import (
    VADAR_MEMORIES_DIR, VADAR_SUPPORTED_LIBS_FILE,
    VADAR_ROLEPLAY_FILE, VADAR_ROLEPLAY_EXTRAS_FILE,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}
_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.xml', '.csv', '.tsv', '.html', '.css', '.log', '.chain'}

_EXT_CATEGORY_MAP = {
    'videos': _VIDEO_EXTENSIONS,
    'images': _IMAGE_EXTENSIONS,
    'audios': _AUDIO_EXTENSIONS,
    'texts': _TEXT_EXTENSIONS,
}


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _is_url(s):
    return isinstance(s, str) and (s.startswith('http://') or s.startswith('https://'))


def _download_url_to_local(url, kind):
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    try:
        from voder import download_url_audio, download_url_video
    except Exception as e:
        return None, f"VODER download functions unavailable: {e}"
    if kind == 'audio':
        ok, err, path = download_url_audio(url, temp_dir=_RESULTS_DIR)
    elif kind == 'video':
        ok, err, path = download_url_video(url, temp_dir=_RESULTS_DIR)
    else:
        try:
            ext = os.path.splitext(url.split('?')[0])[1].lower() or '.jpg'
            if ext not in _IMAGE_EXTENSIONS:
                ext = '.jpg'
            local = os.path.join(_RESULTS_DIR, f"vadar_img_{int(__import__('time').time())}{ext}")
            urllib.request.urlretrieve(url, local)
            return local, None
        except Exception as e:
            return None, f"Image download failed: {e}"
    if not ok or not path:
        return None, err or "Download failed."
    return path, None


def _resolve_media_target(target, kind):
    target = (target or '').strip().strip('"\'')
    if not target:
        return None, "Missing path or URL."
    if _is_url(target):
        local, err = _download_url_to_local(target, kind)
        if err:
            return None, f"Download failed for {target}: {err}"
        return local, None
    if not os.path.exists(target):
        return None, f"File not found: {target}"
    if not _is_within_project(target):
        return None, f"Path '{target}' is outside the VODER project directory."
    return target, None


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


def _parse_format_list(formats_str):
    if not formats_str:
        return None
    combined = set()
    include_others = False
    for raw in formats_str.split(','):
        fmt = raw.strip().lower()
        if not fmt:
            continue
        if fmt in _EXT_CATEGORY_MAP:
            combined |= _EXT_CATEGORY_MAP[fmt]
        elif fmt == 'others':
            include_others = True
        elif fmt == 'all':
            combined |= _VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _TEXT_EXTENSIONS
        else:
            if not fmt.startswith('.'):
                fmt = '.' + fmt
            combined.add(fmt)
    return combined, include_others


@register_tool('list')
def tool_list(args):
    tokens = args.strip().split()
    types_tokens = []
    path = _PROJECT_ROOT
    for tok in tokens:
        if os.path.sep in tok or os.path.exists(tok) or tok in ('.', '..'):
            path = tok.strip('"\'')
        else:
            types_tokens.append(tok.lower())
    if not _is_within_project(path):
        return f"Error: path '{path}' is outside the VODER project directory. I can only list files inside the project."
    if not os.path.isdir(path):
        return f"Error: '{path}' is not a directory."
    files = []
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'node_modules', '.git')]
        for fname in fnames:
            files.append(os.path.join(root, fname))
    if not types_tokens or (len(types_tokens) == 1 and types_tokens[0] in ('all', '')):
        counts = {'videos': 0, 'images': 0, 'audios': 0, 'texts': 0, 'others': 0}
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in _VIDEO_EXTENSIONS: counts['videos'] += 1
            elif ext in _IMAGE_EXTENSIONS: counts['images'] += 1
            elif ext in _AUDIO_EXTENSIONS: counts['audios'] += 1
            elif ext in _TEXT_EXTENSIONS: counts['texts'] += 1
            else: counts['others'] += 1
        return f"{counts['videos']} videos, {counts['images']} images, {counts['audios']} audios, {counts['texts']} text files, {counts['others']} others (total: {len(files)})"
    combined = set()
    include_others = False
    for lt in types_tokens:
        if lt in _EXT_CATEGORY_MAP:
            combined |= _EXT_CATEGORY_MAP[lt]
        elif lt == 'others':
            include_others = True
        elif lt == 'all':
            combined |= _VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _TEXT_EXTENSIONS
        elif lt.startswith('.'):
            combined.add(lt.lower())
    filtered = []
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if include_others and ext not in (_VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _TEXT_EXTENSIONS):
            filtered.append(f)
        elif ext in combined:
            filtered.append(f)
    if not filtered:
        return f"No files matching types '{' '.join(types_tokens)}' found in {path}"
    return '\n'.join(sorted(filtered)[:200])


@register_tool('search')
def tool_search(args):
    m = re.match(r'^(.+?)\s+path\s+(\S+)(?:\s+formats\s+(.+))?$', args.strip(), re.IGNORECASE)
    if not m:
        m2 = re.match(r'^(.+?)\s+(\S+)(?:\s+(.+))?$', args.strip())
        if not m2:
            return "Usage: search <query> path <path> [formats <videos,images,.ext,...>]"
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
    ext_filter = None
    include_others = False
    if formats:
        ext_filter, include_others = _parse_format_list(formats)
    results = []
    all_known = _VIDEO_EXTENSIONS | _IMAGE_EXTENSIONS | _AUDIO_EXTENSIONS | _TEXT_EXTENSIONS
    for root, dirs, fnames in os.walk(path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', '.git')]
        for fname in fnames:
            if query.lower() not in fname.lower():
                continue
            if ext_filter is not None:
                ext = os.path.splitext(fname)[1].lower()
                if ext in ext_filter:
                    pass
                elif include_others and ext not in all_known:
                    pass
                else:
                    continue
            results.append(os.path.join(root, fname))
    if not results:
        return f"No files matching '{query}' found in {path}"
    results.sort()
    return '\n'.join(results[:200])


def _parse_line_ranges(ranges_str):
    ranges = []
    for token in ranges_str.split():
        if '-' not in token:
            return None, f"Invalid range '{token}'. Use start-end (e.g., 20-30)."
        parts = token.split('-')
        if len(parts) != 2:
            return None, f"Invalid range '{token}'. Use start-end (e.g., 20-30)."
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
        except ValueError:
            return None, f"Invalid range '{token}'. Start and end must be integers."
        if start >= end:
            return None, f"Start line ({start}) must be smaller than end line ({end}) in range '{token}'."
        if start < 1:
            return None, f"Start line ({start}) must be at least 1 in range '{token}'."
        ranges.append((start, end))
    if not ranges:
        return None, "No valid ranges provided."
    return ranges, None


@register_tool('read')
def tool_read(args, session_dir=None, act_outputs=None):
    parts = args.strip().split(None, 1)
    if not parts:
        return "Usage: read <path|act_title> [start-end start-end ...]"
    target = parts[0].strip('"\'')
    ranges_str = parts[1].strip() if len(parts) > 1 else None
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

    if ranges_str:
        ranges, err = _parse_line_ranges(ranges_str)
        if err:
            return err
        out_parts = [f"Total lines: {total}", f"Target: {target}", ""]
        for (start, end) in ranges:
            s = max(1, start)
            e = min(total, end)
            if s > e:
                out_parts.append(f"Range {start}-{end}: out of bounds (file has {total} lines).")
                continue
            out_parts.append(f"--- Lines {s}-{e} of {total} ---")
            for i, line in enumerate(lines[s-1:e], start=s):
                out_parts.append(f"{i:6d}: {line}")
            out_parts.append("")
        return '\n'.join(out_parts).rstrip()

    try:
        from voder import vadar_load_config
        config = vadar_load_config()
        preview_count = config.get('read_preview_lines', 100)
    except Exception:
        preview_count = 100
    if preview_count < 1:
        preview_count = 100

    summary_block = ""
    if len(content) > 1500:
        try:
            from voders.vadars.summarizer import summarize_output
            summary = summarize_output(content, context_label=target)
            summary_block = f"--- Summary ---\n{summary}\n\n"
        except Exception:
            pass

    if total > preview_count:
        preview_lines = lines[-preview_count:]
        start_idx = total - preview_count + 1
    else:
        preview_lines = lines[:]
        start_idx = 1
    header = f"Total lines: {total}\nTarget: {target}\n"
    if total > preview_count:
        header += f"Showing latest {preview_count} lines (numbered). Use 'read {target} start-end' to read specific ranges.\n"
    body = []
    for i, line in enumerate(preview_lines, start=start_idx):
        body.append(f"{i:6d}: {line}")
    return summary_block + header + '\n'.join(body)


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


def _generate_tts_narration(text, output_path):
    try:
        from voder import parse_and_execute_oneline
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(output_path))
        tokens = ['tts', 'script', text, 'voice', 'calm narrator, clear pronunciation, male']
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        try:
            parse_and_execute_oneline(tokens)
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout
            os.chdir(old_cwd)
        results_dir = os.path.join(os.path.dirname(output_path), 'results')
        if os.path.isdir(results_dir):
            files = sorted(
                [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith('.wav')],
                key=os.path.getmtime, reverse=True,
            )
            if files:
                shutil.copy2(files[0], output_path)
                shutil.rmtree(results_dir, ignore_errors=True)
                return True
        return False
    except Exception:
        return False


def _concat_audio(first_path, second_path, output_path):
    cmd = ['ffmpeg', '-y', '-i', first_path, '-i', second_path,
           '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
           '-map', '[out]', '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1',
           output_path]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0 and os.path.exists(output_path)


def _format_narration_text(start, end):
    def to_words(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        parts = []
        if h > 0:
            parts.append(f"hour {h}")
        if m > 0:
            parts.append(f"minute {m}")
        if s > 0 or not parts:
            parts.append(f"second {s}")
        return " ".join(parts)
    return f"From {to_words(start)} to {to_words(end)}."


def _cut_media_segment(input_path, start, end, output_path, is_video=False):
    cmd = ['ffmpeg', '-y', '-i', input_path, '-ss', str(start), '-to', str(end),
           '-c:a', 'pcm_s16le', '-ar', '16000', '-ac', '1']
    if is_video:
        cmd = ['ffmpeg', '-y', '-i', input_path, '-ss', str(start), '-to', str(end),
               '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
               '-c:a', 'aac', '-ar', '16000', '-ac', '1']
    cmd.append(output_path)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0 and os.path.exists(output_path)


def _run_multimodal_inference(processor, model, text_prompt, images=None, audios=None, videos=None):
    import torch
    try:
        content = [{'type': 'text', 'text': text_prompt}]
        if images:
            for img_path in images:
                from PIL import Image
                content.append({'type': 'image'})
        if audios:
            for _ in audios:
                content.append({'type': 'audio'})
        if videos:
            for _ in videos:
                content.append({'type': 'video'})

        messages = [{'role': 'user', 'content': content}]
        if hasattr(processor, 'apply_chat_template'):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = text_prompt

        kwargs = {'text': text, 'return_tensors': 'pt'}
        if images:
            from PIL import Image
            kwargs['images'] = [Image.open(p).convert('RGB') for p in images]
        if audios:
            import librosa
            kwargs['audios'] = [librosa.load(p, sr=16000)[0] for p in audios]
        if videos:
            kwargs['videos'] = videos

        inputs = processor(**kwargs).to(model.device if hasattr(model, 'device') else 'cpu')
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        new_tokens = output[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        return f"[multimodal analysis error: {e}]"


@register_tool('look')
def tool_look(args, model=None, processor=None):
    raw = args.strip().strip('"\'')
    if not raw:
        return "Usage: look <image_path_or_url>"
    local, err = _resolve_media_target(raw, 'image')
    if err:
        return err
    ext = os.path.splitext(local)[1].lower()
    if ext not in _IMAGE_EXTENSIONS:
        return f"'{local}' does not appear to be an image file."
    if model is None or processor is None:
        return f"Image at {local} ({os.path.getsize(local)} bytes). Model not loaded — cannot analyze."
    result = _run_multimodal_inference(processor, model, "Describe this image in detail. What do you see?", images=[local])
    return f"Image: {local}\nAnalysis: {result}"


@register_tool('listen')
def tool_listen(args, model=None, processor=None):
    parts = args.strip().split(None, 1)
    target = parts[0].strip('"\'') if parts else ''
    range_spec = parts[1].strip() if len(parts) > 1 else None
    if not target:
        return "Usage: listen <audio_path_or_url> [HH:MM:SS-HH:MM:SS]"
    local, err = _resolve_media_target(target, 'audio')
    if err:
        return err
    dur = _ffprobe_duration(local)
    if dur is None:
        return f"Could not determine duration of {local}"

    if not range_spec:
        try:
            from voder import vadar_load_config
            config = vadar_load_config()
            auto_threshold = config.get('listen_auto_threshold', 30)
        except Exception:
            auto_threshold = 30
        if dur > auto_threshold:
            return f"Audio: {local}\nDuration: {_format_timestamp(dur)}\nAudio is longer than {auto_threshold}s. Use listen <target> HH:MM:SS-HH:MM:SS to listen to a segment."
        if model is None or processor is None:
            return f"Audio: {local}\nDuration: {_format_timestamp(dur)}\nModel not loaded — cannot analyze."
        result = _run_multimodal_inference(processor, model, "Listen to this audio and describe what you hear.", audios=[local])
        return f"Audio: {local}\nDuration: {_format_timestamp(dur)}\nAnalysis: {result}"

    start, end = _parse_time_range(range_spec)
    if start is None or end is None:
        return f"Invalid time range '{range_spec}'. Use HH:MM:SS-HH:MM:SS format."
    if start >= end:
        return f"Start time ({_format_timestamp(start)}) must be before end time ({_format_timestamp(end)})."
    if end > dur:
        end = dur

    try:
        from voder import vadar_load_config
        config = vadar_load_config()
        max_seg = config.get('listen_max_segment', 60)
    except Exception:
        max_seg = 60

    seg_dur = end - start
    if seg_dur > max_seg:
        return f"Segment {_format_timestamp(start)}-{_format_timestamp(end)} is {seg_dur:.0f}s. Max is {max_seg}s. Use a smaller range."

    tmp_dir = tempfile.mkdtemp(prefix='vadar_listen_')
    seg_path = os.path.join(tmp_dir, f'segment.wav')
    try:
        if not _cut_media_segment(local, start, end, seg_path, is_video=False):
            return f"Error: failed to cut audio segment {start}-{end} from {local}"
        if model is None or processor is None:
            return f"Audio segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)}. Model not loaded."

        narration_text = _format_narration_text(start, end)
        narration_path = os.path.join(tmp_dir, 'narration.wav')
        combined_path = os.path.join(tmp_dir, 'combined.wav')
        has_narration = False
        if _generate_tts_narration(narration_text, narration_path):
            if _concat_audio(narration_path, seg_path, combined_path):
                has_narration = True

        feed_path = combined_path if has_narration else seg_path
        prompt = f"This is an audio segment from {_format_timestamp(start)} to {_format_timestamp(end)} (duration: {seg_dur:.0f}s). The audio may start with a voice narration stating the time range. Describe what you hear."
        result = _run_multimodal_inference(processor, model, prompt, audios=[feed_path])
        return f"Audio segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)}.\nAnalysis: {result}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@register_tool('watch')
def tool_watch(args, model=None, processor=None):
    parts = args.strip().split(None, 1)
    target = parts[0].strip('"\'') if parts else ''
    range_spec = parts[1].strip() if len(parts) > 1 else None
    if not target:
        return "Usage: watch <video_path_or_url> [HH:MM:SS-HH:MM:SS]"
    local, err = _resolve_media_target(target, 'video')
    if err:
        return err
    ext = os.path.splitext(local)[1].lower()
    if ext not in _VIDEO_EXTENSIONS:
        return f"'{local}' does not appear to be a video file."
    dur = _ffprobe_duration(local)
    if dur is None:
        return f"Could not determine duration of {local}"

    if not range_spec:
        try:
            from voder import vadar_load_config
            config = vadar_load_config()
            auto_threshold = config.get('listen_auto_threshold', 30)
        except Exception:
            auto_threshold = 30
        if dur > auto_threshold:
            return f"Video: {local}\nDuration: {_format_timestamp(dur)}\nVideo is longer than {auto_threshold}s. Use watch <target> HH:MM:SS-HH:MM:SS to watch a segment."
        if model is None or processor is None:
            return f"Video: {local}\nDuration: {_format_timestamp(dur)}\nModel not loaded — cannot analyze."
        result = _run_multimodal_inference(processor, model, "Watch this video and describe what you see and hear.", videos=[local])
        return f"Video: {local}\nDuration: {_format_timestamp(dur)}\nAnalysis: {result}"

    start, end = _parse_time_range(range_spec)
    if start is None or end is None:
        return f"Invalid time range '{range_spec}'. Use HH:MM:SS-HH:MM:SS format."
    if start >= end:
        return f"Start time ({_format_timestamp(start)}) must be before end time ({_format_timestamp(end)})."
    if end > dur:
        end = dur

    seg_dur = end - start
    try:
        from voder import vadar_load_config
        config = vadar_load_config()
        max_seg = config.get('listen_max_segment', 60)
    except Exception:
        max_seg = 60
    if seg_dur > max_seg:
        return f"Segment {_format_timestamp(start)}-{_format_timestamp(end)} is {seg_dur:.0f}s. Max is {max_seg}s. Use a smaller range."

    tmp_dir = tempfile.mkdtemp(prefix='vadar_watch_')
    seg_path = os.path.join(tmp_dir, f'segment.mp4')
    try:
        if not _cut_media_segment(local, start, end, seg_path, is_video=True):
            return f"Error: failed to cut video segment {start}-{end} from {local}"
        if model is None or processor is None:
            return f"Video segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)}. Model not loaded."
        prompt = f"This is a video segment from {_format_timestamp(start)} to {_format_timestamp(end)}. Describe what you see and hear."
        result = _run_multimodal_inference(processor, model, prompt, videos=[seg_path])
        return f"Video segment {_format_timestamp(start)}-{_format_timestamp(end)} of {_format_timestamp(dur)}.\nAnalysis: {result}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@register_tool('read_role')
def tool_read_role(args):
    try:
        with open(VADAR_ROLEPLAY_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return "Roleplay file is empty. No role is set. Use make_role to create one."
        return content
    except FileNotFoundError:
        return "Roleplay file does not exist yet. Use make_role to create one."
    except Exception as e:
        return f"Error reading roleplay: {e}"


@register_tool('make_role')
def tool_make_role(args):
    content = args.strip()
    if not content:
        return "Usage: make_role <roleplay description in 'I' perspective>"
    try:
        with open(VADAR_ROLEPLAY_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        return f"Roleplay set. Use read_role_extras to build details, or start acting in character."
    except Exception as e:
        return f"Error creating roleplay: {e}"


@register_tool('edit_role')
def tool_edit_role(args):
    content = args.strip()
    if not content:
        return "Usage: edit_role <new roleplay description>"
    try:
        with open(VADAR_ROLEPLAY_FILE, 'r', encoding='utf-8') as f:
            old = f.read().strip()
        if not old:
            return "Roleplay file is empty. Use make_role first."
        with open(VADAR_ROLEPLAY_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        return "Roleplay updated. Extras cleared — rebuild them with read_role_extras + edit_role_extras."
    except Exception as e:
        return f"Error editing roleplay: {e}"


@register_tool('delete_role')
def tool_delete_role(args):
    try:
        with open(VADAR_ROLEPLAY_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        return "Roleplay deleted. No role is active."
    except Exception as e:
        return f"Error deleting roleplay: {e}"


@register_tool('read_role_extras')
def tool_read_role_extras(args):
    try:
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        if not content:
            return "Roleplay extras are empty. Use edit_role_extras to add details that expand the roleplay."
        return content
    except Exception as e:
        return f"Error reading roleplay extras: {e}"


@register_tool('make_role_extras')
def tool_make_role_extras(args):
    content = args.strip()
    if not content:
        return "Usage: make_role_extras <extras details in 'I' perspective>"
    try:
        with open(VADAR_ROLEPLAY_FILE, 'r', encoding='utf-8') as f:
            role = f.read().strip()
        if not role:
            return "No roleplay is set. Use make_role first."
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        return "Roleplay extras created."
    except Exception as e:
        return f"Error creating roleplay extras: {e}"


@register_tool('edit_role_extras')
def tool_edit_role_extras(args):
    content = args.strip()
    if not content:
        return "Usage: edit_role_extras <new extras details>"
    try:
        with open(VADAR_ROLEPLAY_FILE, 'r', encoding='utf-8') as f:
            role = f.read().strip()
        if not role:
            return "No roleplay is set. Use make_role first."
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        return "Roleplay extras updated."
    except Exception as e:
        return f"Error editing roleplay extras: {e}"


@register_tool('delete_role_extras')
def tool_delete_role_extras(args):
    try:
        with open(VADAR_ROLEPLAY_EXTRAS_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        return "Roleplay extras deleted."
    except Exception as e:
        return f"Error deleting roleplay extras: {e}"


_PLATFORM_SEARCH_URLS = {
    'youtube': lambda q, n: f"ytsearch{n}:{q}",
    'bilibili': lambda q, n: f"bilisearch{n}:{q}",
    'tiktok': lambda q, n: f"https://www.tiktok.com/search?q={q}",
    'snapchat': lambda q, n: f"https://www.snapchat.com/spotlight/trending",
    'instagram': lambda q, n: f"https://www.instagram.com/explore/tags/{q.strip('#')}/",
    'facebook': lambda q, n: f"https://www.facebook.com/watch/search/?q={q}",
    'twitter': lambda q, n: f"https://x.com/search?q={q}&f=live",
    'x': lambda q, n: f"https://x.com/search?q={q}&f=live",
}


@register_tool('search_media')
def tool_search_media(args):
    parts = args.strip().split(None, 2)
    if len(parts) < 3:
        return "Usage: search_media <platform> <search query> <number>\nPlatforms: youtube, bilibili, tiktok, snapchat, instagram, facebook, twitter/x"
    platform = parts[0].lower().strip()
    query = parts[1].strip().strip('"\'')
    try:
        count = int(parts[2].strip())
    except ValueError:
        return f"Invalid number '{parts[2]}'. Must be an integer."
    if count < 1:
        return "Number must be at least 1."
    if count > 50:
        count = 50
    builder = _PLATFORM_SEARCH_URLS.get(platform)
    if builder is None:
        return f"Unsupported platform '{platform}'. Supported: {', '.join(_PLATFORM_SEARCH_URLS.keys())}"
    search_url = builder(query, count)
    try:
        cmd = [
            'yt-dlp', search_url,
            '--flat-playlist',
            '--playlist-end', str(count),
            '--print', 'Title: %(title)s | URL: %(url)s | Platform: %(extractor)s',
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            err = r.stderr.strip()[-500:] if r.stderr else 'unknown error'
            return f"Search failed: {err}"
        all_lines = r.stdout.strip().split('\n') if r.stdout.strip() else []
        video_indicators = ('/watch', '/video/', '/status/', '/spotlight/', '/explore/', '/reel/', 'youtu.be/', 'tiktok.com/@')
        filtered = [line for line in all_lines if any(ind in line.lower() for ind in video_indicators)]
        results = '\n'.join(filtered) if filtered else '\n'.join(all_lines)
        if not results.strip():
            return f"No video results found for '{query}' on {platform}."
        return f"Search results for '{query}' on {platform} ({len(filtered) if filtered else len(all_lines)} videos, {count} max):\n\n{results}"
    except FileNotFoundError:
        return "yt-dlp is not installed. Install with: pip install yt-dlp"
    except subprocess.TimeoutExpired:
        return "Search timed out (60s). Try fewer results or a simpler query."
    except Exception as e:
        return f"Search error: {e}"
