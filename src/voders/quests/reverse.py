import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


class Quest(SideQuest):
    name = 'reverse'
    description = 'Reverse a local audio OR video file. Audio-only inputs produce a reversed WAV; video inputs produce a reversed MP4 (frames and audio both flipped).'

    def parse(self, args):
        if len(args) != 1:
            return None, "quest reverse takes exactly one argument: <local-audio-or-video-path>"
        path = args[0]
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        ext = os.path.splitext(path)[1].lower()
        if ext not in VIDEO_EXTENSIONS:
            audio_exts = {'.wav', '.mp3', '.flac', '.ogg', '.oga', '.opus', '.aac', '.m4a', '.wma', '.aiff', '.aif', '.ac3', '.amr', '.au', '.gsm', '.tta', '.wv', '.ape', '.mpc', '.mp2', '.mka', '.caf', '.amb', '.sph'}
            if ext not in audio_exts:
                return None, f"unrecognized file extension '{ext}' — quest reverse supports common audio and video files"
        return {'path': path, 'is_video': ext in VIDEO_EXTENSIONS}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        path = parsed['path']
        is_video = parsed['is_video']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'

        if is_video:
            out_name = f"voder_quest_reverse_{safe_name}_{timestamp}.mp4"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-filter_complex', '[0:v]reverse[v];[0:a]areverse[a]',
                '-map', '[v]', '-map', '[a]',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                '-c:a', 'aac', '-b:a', '192k',
                '-movflags', '+faststart',
                out_path,
            ]
        else:
            out_name = f"voder_quest_reverse_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-af', 'areverse',
                '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to reverse {'video' if is_video else 'audio'}")
            if r.stderr:
                print(r.stderr[-600:])
            return False
        print(f"Quest reverse ({'video' if is_video else 'audio'}) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
