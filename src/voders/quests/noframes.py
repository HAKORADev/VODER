import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


class Quest(SideQuest):
    name = 'noframes'
    category = 'Media Manipulation'
    description = 'Extract audio from a LOCAL VIDEO file. Refuses URLs and audio-only files.'

    def parse(self, args):
        from voder import is_youtube_url
        if len(args) != 1:
            return None, "quest noframes takes exactly one argument: a local video file path"
        path = args[0]
        if is_youtube_url(path):
            return None, "quest noframes refuses URLs — provide a local video file"
        if not os.path.exists(path):
            return None, f"file not found: {path}"
        ext = os.path.splitext(path)[1].lower()
        video_exts = ('.mp4', '.mkv', '.mov', '.avi', '.webm', '.flv', '.wmv', '.m4v')
        if ext not in video_exts:
            return None, f"quest noframes refuses non-video files (got '{ext}'). Provide a local video file."
        return {'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        path = parsed['path']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        out_name = f"voder_quest_noframes_{safe_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)
        cmd = ['ffmpeg', '-i', path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', out_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path):
            print(f"Error: ffmpeg failed to extract audio")
            if result.stderr:
                print(result.stderr[-500:])
            return False
        print(f"Quest noframes complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
