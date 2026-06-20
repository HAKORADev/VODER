import os
import re
import shutil
import subprocess
import tempfile

from voders.sidequests import SideQuest, _register_side_quest


class Quest(SideQuest):
    name = 'merge'
    category = 'Media Manipulation'
    description = 'Concatenate two or more local audio files end-to-end into a single WAV. Syntax: quest merge <file1> <file2> [<file3> ...]  (no upper limit).'

    def parse(self, args):
        if len(args) < 2:
            return None, "quest merge needs at least two audio files (e.g., merge a.wav b.wav c.wav)"
        for a in args:
            if not os.path.exists(a):
                return None, f"input file not found: {a}"
            ext = os.path.splitext(a)[1].lower().lstrip('.')
            if not ext:
                return None, f"input file '{a}' has no extension; cannot determine format"
        return {'files': list(args)}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        files = parsed['files']
        os.makedirs(results_dir, exist_ok=True)
        joined_name = '_'.join(
            re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(f))[0])[:15]
            for f in files
        )[:80] or 'merged'
        out_name = f"voder_quest_merge_{joined_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        tmp_dir = tempfile.mkdtemp(prefix='voder_merge_')
        try:
            normalized = []
            for idx, f in enumerate(files):
                norm = os.path.join(tmp_dir, f"part_{idx:04d}.wav")
                cmd = [
                    'ffmpeg', '-y', '-i', f,
                    '-vn', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                    norm,
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0 or not os.path.exists(norm):
                    print(f"Error: failed to normalize '{f}' before merging")
                    if r.stderr:
                        print(r.stderr[-500:])
                    return False
                normalized.append(norm)

            list_path = os.path.join(tmp_dir, 'concat_list.txt')
            with open(list_path, 'w', encoding='utf-8') as fh:
                for n in normalized:
                    esc = n.replace("'", r"'\''")
                    fh.write(f"file '{esc}'\n")

            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', list_path,
                '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                out_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(out_path) or r.returncode != 0:
                print("Error: ffmpeg failed to merge files")
                if r.stderr:
                    print(r.stderr[-600:])
                return False
            print(f"Quest merge ({len(files)} files) complete: {out_path}")
            if result_path:
                try:
                    shutil.copy2(out_path, result_path)
                    print(f"Result copied to: {result_path}")
                except Exception as e:
                    print(f"Note: could not copy to result path: {e}")
            return True
        finally:
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass


_register_side_quest(Quest)
