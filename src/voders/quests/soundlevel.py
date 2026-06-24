import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


class Quest(SideQuest):
    name = 'soundlevel'
    description = 'Linear sound-level multiplier. 1.00 = original, 0.01 = 1% of original, 0.25 = 25% of original, 1.99 = +99% louder, 10.00 = 10x louder. Affects all frequencies equally (no EQ, no compression, no loudness normalization). Syntax: quest soundlevel <0.01-10.00> <local-audio-or-video-path>.'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest soundlevel takes exactly two arguments: <0.01-10.00> <local-audio-or-video-path>"
        try:
            value = float(args[0])
        except ValueError:
            return None, f"first argument must be a number 0.01-10.00 (got '{args[0]}')"
        if not (0.01 <= value <= 10.00):
            return None, f"soundlevel multiplier must be between 0.01 and 10.00 (got {value})"
        path = args[1]
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        return {'value': value, 'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        value = parsed['value']
        path = parsed['path']
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'

        af = f"volume={value:.3f}"
        value_tag = f"{value:.2f}".replace('.', 'p')

        if is_video:
            out_name = f"voder_quest_soundlevel_x{value_tag}_{safe_name}_{timestamp}.mp4"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-af', af,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '256k',
                '-movflags', '+faststart',
                out_path,
            ]
        else:
            out_name = f"voder_quest_soundlevel_x{value_tag}_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vn', '-af', af,
                '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply soundlevel multiplier")
            if r.stderr:
                print(r.stderr[-800:])
            return False
        print(f"Quest soundlevel (x{value:.2f}) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
