import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


def _probe_loudnorm(path):
    cmd = [
        'ffmpeg', '-hide_banner', '-i', path,
        '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json',
        '-f', 'null', '-',
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return None, r.stderr[-800:] if r.stderr else "ffmpeg loudnorm probe failed"
    stderr = r.stderr or ''
    m = re.search(r'\{[^{}]*"input_i"[^{}]*\}', stderr, re.DOTALL)
    if not m:
        return None, "could not find loudnorm JSON in ffmpeg output"
    try:
        import json
        data = json.loads(m.group(0))
        return data, None
    except Exception as e:
        return None, f"failed to parse loudnorm JSON: {e}"


class Quest(SideQuest):
    name = 'loudnorm'
    category = 'Media Manipulation'
    description = 'EBU R128 perceptual loudness normalization. Analyzes the file, computes the integrated loudness, then applies a linear normalization so the whole signal sits at one consistent perceived level (-16 LUFS target, -1.5 dB true-peak limit). Quiet parts and loud parts end up at the same perceptual medium — no quality loss, no dynamic-range compression. Audio and video supported. Syntax: quest loudnorm <local-audio-or-video-path>.'

    def parse(self, args):
        if len(args) != 1:
            return None, "quest loudnorm takes exactly one argument: <local-audio-or-video-path>"
        path = args[0]
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        return {'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        path = parsed['path']
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'

        print(f"Loudnorm: probing perceptual loudness of {path}...")
        stats, err = _probe_loudnorm(path)
        if stats is None:
            print(f"Error: loudness analysis failed: {err}")
            return False

        input_i = stats.get('input_i', 'NA')
        input_tp = stats.get('input_tp', 'NA')
        input_lra = stats.get('input_lra', 'NA')
        input_thresh = stats.get('input_thresh', 'NA')
        target_offset = stats.get('target_offset', '0.0')
        try:
            input_i_f = float(input_i)
        except (ValueError, TypeError):
            input_i_f = None
        print(f"  Input integrated loudness: {input_i} LUFS")
        print(f"  Input true peak:           {input_tp} dBTP")
        print(f"  Input loudness range:      {input_lra} LU")
        print(f"  Target offset:             {target_offset} LU")

        if input_i_f is not None and abs(input_i_f - (-16.0)) < 0.2:
            print(f"  Already at target (-16 LUFS) — applying pass-through normalization for true-peak safety.")

        norm_filter = (
            f"loudnorm="
            f"I=-16:TP=-1.5:LRA=11:"
            f"measured_I={input_i}:"
            f"measured_TP={input_tp}:"
            f"measured_LRA={input_lra}:"
            f"measured_thresh={input_thresh}:"
            f"offset={target_offset}:"
            f"linear=true:print_format=summary"
        )

        if is_video:
            out_name = f"voder_quest_loudnorm_{safe_name}_{timestamp}.mp4"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-af', norm_filter,
                '-c:v', 'copy',
                '-c:a', 'aac', '-b:a', '256k',
                '-movflags', '+faststart',
                out_path,
            ]
        else:
            out_name = f"voder_quest_loudnorm_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vn', '-af', norm_filter,
                '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply loudnorm")
            if r.stderr:
                print(r.stderr[-1200:])
            return False
        print(f"Quest loudnorm (target -16 LUFS, -1.5 dBTP) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
