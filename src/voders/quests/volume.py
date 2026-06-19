import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


class Quest(SideQuest):
    name = 'volume'
    description = 'Professional bass booster + volume amplifier. Scale 1-1000: every 100 means +100% gain (100 = 2x, 1000 = 11x). Includes low-shelf bass boost, virtual sub-bass, soft-knee compression, and broadcast loudness normalization to avoid clipping and dotty noise. Syntax: quest volume <1-1000> <local-audio-or-video-path>.'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest volume takes exactly two arguments: <1-1000> <local-audio-or-video-path>"
        try:
            value = int(args[0])
        except ValueError:
            return None, f"first argument must be an integer 1-1000 (got '{args[0]}')"
        if not (1 <= value <= 1000):
            return None, f"volume value must be between 1 and 1000 (got {value})"
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

        linear_gain = 1.0 + (value / 100.0)
        bass_gain_db = min(20.0, value * 0.02)
        bass_cutoff = 80 + (value / 1000.0) * 40
        bass_width = 60 + (value / 1000.0) * 80
        treble_gain_db = min(6.0, value * 0.006)
        virtual_strength = 0.5 + (value / 1000.0) * 2.5
        virtual_cutoff = 200 + (value / 1000.0) * 100

        comp_threshold = max(0.05, 0.5 - (value / 1000.0) * 0.35)
        comp_ratio = 2.0 + (value / 1000.0) * 3.0

        af = (
            f"bass=g={bass_gain_db:.2f}:f={bass_cutoff:.1f}:w={bass_width:.1f},"
            f"virtualbass=cutoff={virtual_cutoff:.1f}:strength={virtual_strength:.2f},"
            f"treble=g={treble_gain_db:.2f}:f=4000:w=3000,"
            f"volume={linear_gain:.3f},"
            f"acompressor=threshold={comp_threshold:.3f}:ratio={comp_ratio:.2f}:attack=10:release=200:makeup=1.2:knee=4,"
            f"loudnorm=I=-14:TP=-1.0:LRA=11:linear=true"
        )

        if is_video:
            out_name = f"voder_quest_volume_v{value}_{safe_name}_{timestamp}.mp4"
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
            out_name = f"voder_quest_volume_v{value}_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vn', '-af', af,
                '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply volume/bass boost")
            if r.stderr:
                print(r.stderr[-800:])
            return False
        print(
            f"Quest volume (value={value}, linear x{linear_gain:.2f}, "
            f"bass +{bass_gain_db:.1f}dB, treble +{treble_gain_db:.1f}dB) complete: {out_path}"
        )
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
