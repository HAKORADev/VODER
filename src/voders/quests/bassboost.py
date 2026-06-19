import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


class Quest(SideQuest):
    name = 'bassboost'
    description = 'Professional multi-band bass booster (low frequencies only). Scale 1-100: 1 = subtle warmth, 50 = strong club bass, 100 = +24 dB sub-destroyer. Signal chain: sub-sonic highpass -> lowshelf boost @ 80 Hz -> peaking boost @ 50 Hz for sub-bass punch -> virtualbass synthesizer for sub harmonics on small speakers -> soft-knee compressor to glue and prevent dotty noise -> safety true-peak limiter at -1 dB. Mids and highs are left untouched. Audio input -> 24-bit/48k WAV. Video input -> MP4 with video copied, audio re-encoded as AAC 256k. Syntax: quest bassboost <1-100> <local-audio-or-video-path>.'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest bassboost takes exactly two arguments: <1-100> <local-audio-or-video-path>"
        try:
            value = int(args[0])
        except ValueError:
            return None, f"first argument must be an integer 1-100 (got '{args[0]}')"
        if not (1 <= value <= 100):
            return None, f"bassboost value must be between 1 and 100 (got {value})"
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

        t = value / 100.0
        shelf_gain_db = 24.0 * t
        peak_gain_db = 18.0 * t
        virtual_strength = 0.3 + 2.7 * t
        comp_threshold = max(0.05, 0.5 - 0.35 * t)
        comp_ratio = 2.0 + 3.0 * t

        af = (
            f"highpass=f=30,"
            f"bass=g={shelf_gain_db:.2f}:f=80:w=80,"
            f"equalizer=f=50:g={peak_gain_db:.2f}:w=40:t=q,"
            f"virtualbass=cutoff=250:strength={virtual_strength:.2f},"
            f"acompressor=threshold={comp_threshold:.3f}:ratio={comp_ratio:.2f}:attack=10:release=200:makeup=1.1:knee=4,"
            f"alimiter=limit=0.89:attack=5:release=50"
        )

        if is_video:
            out_name = f"voder_quest_bassboost_v{value}_{safe_name}_{timestamp}.mp4"
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
            out_name = f"voder_quest_bassboost_v{value}_{safe_name}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            cmd = [
                'ffmpeg', '-y', '-i', path,
                '-vn', '-af', af,
                '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
                out_path,
            ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply bass boost")
            if r.stderr:
                print(r.stderr[-800:])
            return False
        print(
            f"Quest bassboost (value={value}, shelf +{shelf_gain_db:.1f}dB @ 80Hz, "
            f"peak +{peak_gain_db:.1f}dB @ 50Hz) complete: {out_path}"
        )
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
