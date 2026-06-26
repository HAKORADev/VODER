import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


def _decompose_pitch(target):
    passes = []
    current = float(target)
    while current > 2.0 + 1e-6:
        passes.append(2.0)
        current /= 2.0
    while current < 0.5 - 1e-6:
        passes.append(0.5)
        current *= 2.0
    if abs(current - 1.0) > 1e-6:
        passes.append(round(current, 4))
    return passes


class Quest(SideQuest):
    name = 'pitch'
    description = 'Professional pitch shift (rubberband, formant-shifted for that tape/vinyl character). Range 0.01-10.00 in 0.01 steps (1.00 is a no-op). 0.50 = -1 octave (monster/demon), 2.00 = +1 octave (baby/chipmunk), 0.01 = extreme deep, 10.00 = extreme high. Extreme ranges outside 0.50-2.00 are split into multiple one-octave passes for clean output. Audio output only. Accepts local audio, local video, and any supported URL (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter). Chain with quest speed for Spotify-style slowed+reverb. Syntax: quest pitch <0.01-10.00> "<audio|video|URL>".'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest pitch takes exactly two arguments: <0.01-10.00> <audio|video|URL>"
        try:
            value = float(args[0])
        except ValueError:
            return None, f"pitch value must be a number 0.01-10.00 (got '{args[0]}')"
        rounded = round(value, 2)
        if not (0.01 <= rounded <= 10.00):
            return None, f"pitch value must be between 0.01 and 10.00 (got {value})"
        if abs(rounded - 1.00) < 0.001:
            return None, "pitch value 1.00 is a no-op; please pick a different value"
        location = args[1]
        from voder import is_supported_url
        is_url = is_supported_url(location)
        if not is_url and not os.path.exists(location):
            return None, f"input not found and not a recognized URL: {location} (only local files and supported URLs are accepted — YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter)"
        return {'value': rounded, 'location': location, 'is_url': is_url}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        value = parsed['value']
        location = parsed['location']
        is_url = parsed['is_url']
        os.makedirs(results_dir, exist_ok=True)

        cleanup_temp = False
        if is_url:
            from voder import download_url_audio, derive_output_name, is_video_url
            is_vid, verify_err, _pid = is_video_url(location, verify=True)
            if not is_vid:
                print(f"Error: {verify_err or 'This link is not a video'}")
                return False
            ok, err, audio_path = download_url_audio(location, skip_verify=True)
            if not ok:
                print(f"Error: failed to download URL: {err}")
                return False
            local_input = audio_path
            original_name = derive_output_name(location)
            cleanup_temp = True
        else:
            local_input = location
            original_name = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(location))[0])[:60] or 'input'

        safe_name = original_name
        out_name = f"voder_quest_pitch_p{value:.2f}_{safe_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        passes = _decompose_pitch(value)
        if not passes:
            passes = [1.0]
        per_pass = (
            "rubberband=pitch={p:.4f}:tempo=1.0:"
            "formant=shifted:transients=crisp:detector=compound:"
            "phase=laminar:window=standard:smoothing=off:pitchq=quality:channels=apart"
        )
        af = ','.join(per_pass.format(p=p) for p in passes)

        cmd = [
            'ffmpeg', '-y', '-i', local_input,
            '-vn', '-af', af,
            '-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2',
            out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)

        if cleanup_temp and os.path.exists(local_input):
            try:
                os.remove(local_input)
            except Exception:
                pass

        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to apply professional pitch shift")
            if r.stderr:
                print(r.stderr[-800:])
            return False

        if value < 1.0:
            label = f"{1.0 / value:.2f}x lower"
        else:
            label = f"{value:.2f}x higher"
        pass_note = f"{len(passes)} pass{'es' if len(passes) > 1 else ''}"
        print(f"Quest pitch (value={value}, {label}, formant-shifted, {pass_note}) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
