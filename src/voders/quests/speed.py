import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VALID_SPEEDS = []
_v = 0.25
while _v <= 10.005:
    VALID_SPEEDS.append(round(_v, 2))
    _v += 0.25


class Quest(SideQuest):
    name = 'speed'
    description = 'Professional time-stretch (Spotify-style slowed/sped-up) for local audio files. Values 0.25-10.00 in 0.25 steps (excluding 1.0). 0.25 = 4x faster, 10.00 = 10x slower. Pitch and formants are preserved via rubberband. Syntax: quest speed <0.25|0.50|0.75|1.25|...|10.00> <audio-path>.'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest speed takes exactly two arguments: <speed-value> <local-audio-path>"
        try:
            value = float(args[0])
        except ValueError:
            return None, f"speed value must be a number (got '{args[0]}')"
        rounded = round(value, 2)
        if rounded not in VALID_SPEEDS:
            return None, (
                f"speed value must be one of 0.25, 0.50, 0.75, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50, ..., 10.00 "
                f"(got {value}). 1.00 is excluded (no-op)."
            )
        if abs(rounded - 1.00) < 0.001:
            return None, "speed value 1.00 is a no-op; please pick a different value"
        path = args[1]
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        ext = os.path.splitext(path)[1].lower()
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}
        if ext in video_exts:
            return None, "quest speed works on audio files only; use quest cut / quest noframes on video first"
        return {'value': rounded, 'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        value = parsed['value']
        path = parsed['path']
        tempo = 1.0 / value
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        out_name = f"voder_quest_speed_x{value:.2f}_{safe_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        if tempo >= 1.0:
            label = f"{tempo:.2f}x faster"
        else:
            label = f"{1.0/tempo:.2f}x slower"

        af = (
            f"rubberband=tempo={tempo:.4f}:pitch=1.0:"
            f"formant=preserved:transients=crisp:detector=compound:"
            f"phase=laminar:window=standard:smoothing=off:pitchq=quality:channels=apart"
        )
        cmd = [
            'ffmpeg', '-y', '-i', path,
            '-vn', '-af', af,
            '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply professional time-stretch")
            if r.stderr:
                print(r.stderr[-800:])
            return False
        print(f"Quest speed (value={value}, {label}, formant-preserved) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
