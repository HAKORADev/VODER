import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


def _to_seconds(token):
    if ':' in token:
        parts = token.split(':')
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                return None
        if len(parts) == 3:
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
        return None
    try:
        return float(token)
    except ValueError:
        return None


class Quest(SideQuest):
    name = 'cut'
    category = 'Media Manipulation'
    description = 'Extract a time range from a local audio/video file as a WAV. Syntax: quest cut <start>-<end> <path>  (seconds, e.g. 20-40 or 1:30-2:15).'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest cut takes exactly two arguments: <start>-<end> <path>  (e.g., cut 20-40 \"clip.mp4\")"
        range_token = args[0]
        path = args[1]
        if '-' not in range_token:
            return None, f"range must be in the form <start>-<end> (got '{range_token}')"
        m = re.match(r'^([\d:]+(?:\.\d+)?)\-([\d:]+(?:\.\d+)?)$', range_token)
        if not m:
            return None, f"range must be <start>-<end> using numbers or mm:ss or hh:mm:ss (got '{range_token}')"
        start_s = _to_seconds(m.group(1))
        end_s = _to_seconds(m.group(2))
        if start_s is None or end_s is None:
            return None, f"could not parse start/end times in '{range_token}'"
        if start_s < 0 or end_s < 0:
            return None, "start and end must be non-negative"
        if start_s >= end_s:
            return None, f"start ({start_s}s) must be strictly smaller than end ({end_s}s)"
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        return {'start': start_s, 'end': end_s, 'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        start = parsed['start']
        end = parsed['end']
        path = parsed['path']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        dur = end - start
        out_name = f"voder_quest_cut_{safe_name}_{int(start)}s-{int(end)}s_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        cmd = [
            'ffmpeg', '-y',
            '-ss', f'{start:.3f}',
            '-i', path,
            '-t', f'{dur:.3f}',
            '-vn',
            '-c:a', 'pcm_s16le',
            '-ar', '44100',
            '-ac', '2',
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to cut range {start}-{end}s")
            if r.stderr:
                print(r.stderr[-600:])
            return False
        print(f"Quest cut ({start}s -> {end}s, {dur:.2f}s long) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
