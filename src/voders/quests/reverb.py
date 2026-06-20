import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


class Quest(SideQuest):
    name = 'reverb'
    category = 'Media Manipulation'
    description = 'Professional Schroeder-style reverb (early reflections + late-reverb tail + pre-delay + air-absorption damping + compressor + dynamic normalization + true-peak limiter) on a 1-100 scale. 1 = barely-there small room, 25 = chamber, 50 = concert hall, 75 = large hall, 100 = cathedral-drenched. Audio output only. Accepts local audio, local video, and YouTube/Bilibili/TikTok URLs. Chain with quest speed + quest pitch for the full demon-cathedral slowed+reverb edit. Syntax: quest reverb <1-100> "<audio|video|URL>".'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest reverb takes exactly two arguments: <1-100> <audio|video|URL>"
        try:
            value = int(args[0])
        except ValueError:
            return None, f"first argument must be an integer 1-100 (got '{args[0]}')"
        if not (1 <= value <= 100):
            return None, f"reverb value must be between 1 and 100 (got {value})"
        location = args[1]
        from voder import is_youtube_url
        is_url = is_youtube_url(location)
        if not is_url and not os.path.exists(location):
            return None, f"input not found and not a recognized URL: {location} (only local files and YouTube/Bilibili/TikTok URLs are supported)"
        return {'value': value, 'location': location, 'is_url': is_url}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        value = parsed['value']
        location = parsed['location']
        is_url = parsed['is_url']
        os.makedirs(results_dir, exist_ok=True)

        cleanup_temp = False
        if is_url:
            from voder import download_youtube_audio
            ok, err, audio_path = download_youtube_audio(location)
            if not ok:
                print(f"Error: failed to download URL: {err}")
                return False
            local_input = audio_path
            original_name = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(location))[0])[:40] or 'url'
            cleanup_temp = True
        else:
            local_input = location
            original_name = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(location))[0])[:60] or 'input'

        safe_name = original_name
        out_name = f"voder_quest_reverb_r{value}_{safe_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        t = value / 100.0

        predelay_ms = int(5 + t * 75)

        early_in_gain = 0.85
        early_out_gain = 1.0
        early_base_decay = 0.10 + t * 0.30
        early_delays = "18|27|36|46|58"
        early_decays = "|".join(
            f"{early_base_decay * (1.0 - 0.05 * i):.3f}" for i in range(5)
        )

        late_in_gain = 0.55
        late_out_gain = 1.0
        late_base_decay = 0.15 + t * 0.40
        late_delays = "61|73|89|103|127|151|181"
        late_decays = "|".join(
            f"{late_base_decay * (1.0 - 0.04 * i):.3f}" for i in range(7)
        )

        lp_cutoff = int(6000 + t * 7000)
        comp_threshold = max(0.20, 0.55 - t * 0.20)
        comp_ratio = 1.8 + t * 1.6
        comp_makeup = 1.0 + t * 0.20

        af_parts = [
            "highpass=f=60",
            f"adelay={predelay_ms}|{predelay_ms}:all=1",
            f"aecho=in_gain={early_in_gain:.3f}:out_gain={early_out_gain:.3f}:delays={early_delays}:decays={early_decays}",
            f"aecho=in_gain={late_in_gain:.3f}:out_gain={late_out_gain:.3f}:delays={late_delays}:decays={late_decays}",
            f"lowpass=f={lp_cutoff}",
            f"acompressor=threshold={comp_threshold:.3f}:ratio={comp_ratio:.2f}:attack=12:release=200:makeup={comp_makeup:.2f}:knee=6",
            "dynaudnorm=f=200:g=15:p=0.95",
            "alimiter=limit=0.95:attack=5:release=50",
        ]
        af = ','.join(af_parts)

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
            print(f"Error: ffmpeg failed to apply professional reverb")
            if r.stderr:
                print(r.stderr[-800:])
            return False

        if value < 25:
            label = "small room"
        elif value < 50:
            label = "chamber"
        elif value < 75:
            label = "concert hall"
        else:
            label = "cathedral"
        print(
            f"Quest reverb (value={value}, {label}, predelay={predelay_ms}ms, "
            f"lp_cutoff={lp_cutoff}Hz) complete: {out_path}"
        )
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
