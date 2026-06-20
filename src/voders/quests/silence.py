import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


class Quest(SideQuest):
    name = 'silence'
    category = 'Media Manipulation'
    description = 'Strip silent gaps from a local audio/video file and produce a continuous-speech WAV. Useful as a chain step before svs/stt to get tight, gap-free speech.'

    def parse(self, args):
        if len(args) < 1 or len(args) > 2:
            return None, "quest silence takes one or two arguments: <input-path> [threshold-dB]  (e.g., silence \"in.wav\"  or  silence \"in.wav\" 40)"
        path = args[0]
        threshold_db = -50
        if len(args) == 2:
            try:
                t = int(args[1])
            except ValueError:
                return None, f"threshold must be an integer dB level (e.g., 40 for -40dB), got '{args[1]}'"
            if not (10 <= t <= 90):
                return None, f"threshold must be between 10 and 90 (i.e., -10dB to -90dB), got {t}"
            threshold_db = -t
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        return {'path': path, 'threshold_db': threshold_db}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        path = parsed['path']
        threshold_db = parsed['threshold_db']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        out_name = f"voder_quest_silence_{safe_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        af = (
            f"silenceremove="
            f"start_periods=1:start_duration=0.1:start_threshold={threshold_db}dB:"
            f"stop_periods=-1:stop_duration=0.25:stop_threshold={threshold_db}dB,"
            f"aresample=44100,"
            f"dynaudnorm=f=200:g=15:p=0.9"
        )
        cmd = [
            'ffmpeg', '-y', '-i', path,
            '-vn', '-af', af,
            '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to strip silence")
            if r.stderr:
                print(r.stderr[-600:])
            return False
        print(f"Quest silence (threshold {threshold_db}dB) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
