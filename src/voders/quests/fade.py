import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


def _ffprobe_duration(path):
    try:
        r = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nw=1:nk=1', path],
            capture_output=True, text=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


class Quest(SideQuest):
    name = 'fade'
    category = 'Media Manipulation'
    description = 'Apply a 5-second cinematic fade-in and fade-out to a local audio or video file. Edges rise to ~15% gain then swell to full volume using a smooth quarter-sine curve — never silent, always rising.'

    def parse(self, args):
        if len(args) < 1 or len(args) > 2:
            return None, "quest fade takes one or two arguments: <input-path> [fade-seconds]  (default fade length is 5s)"
        path = args[0]
        fade_dur = 5.0
        if len(args) == 2:
            try:
                fade_dur = float(args[1])
            except ValueError:
                return None, f"fade duration must be a number of seconds (got '{args[1]}')"
            if not (0.5 <= fade_dur <= 60):
                return None, f"fade duration must be between 0.5 and 60 seconds (got {fade_dur})"
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        return {'path': path, 'fade_dur': fade_dur}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        path = parsed['path']
        fade_dur = parsed['fade_dur']
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'

        duration = _ffprobe_duration(path) or 0.0
        if duration <= 2 * fade_dur:
            fade_dur = max(0.5, duration / 4.0) if duration > 2 else fade_dur
            print(f"Note: file is short; clamping fade length to {fade_dur:.2f}s per side")

        if is_video:
            out_name = f"voder_quest_fade_{safe_name}_{timestamp}.mp4"
            out_path = os.path.join(results_dir, out_name)
            out_start = max(0.0, duration - fade_dur)
            af = (
                f"afade=t=in:st=0:d={fade_dur:.3f}:curve=qsin:unity=0.15:silence=0,"
                f"afade=t=out:st={out_start:.3f}:d={fade_dur:.3f}:curve=qsin:unity=0.15:silence=0,"
                f"volume=1.15"
            )
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-af', af,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '192k',
                '-movflags', '+faststart',
                out_path,
            ]
        else:
            out_name = f"voder_quest_fade_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            out_start = max(0.0, duration - fade_dur)
            af = (
                f"afade=t=in:st=0:d={fade_dur:.3f}:curve=qsin:unity=0.15:silence=0,"
                f"afade=t=out:st={out_start:.3f}:d={fade_dur:.3f}:curve=qsin:unity=0.15:silence=0,"
                f"volume=1.15"
            )
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vn', '-af', af,
                '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply fade")
            if r.stderr:
                print(r.stderr[-600:])
            return False
        print(f"Quest fade ({fade_dur:.1f}s in/out, cinematic) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
