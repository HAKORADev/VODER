import os
import re
import shutil
import subprocess
import tempfile

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


def _extract_audio(input_path, out_path):
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vn', '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
        out_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0 and os.path.exists(out_path)


class Quest(SideQuest):
    name = 'mix'
    description = 'Overlay multiple audio/video sources at specified start times into a single WAV. First source is the base (starts at 0s). Subsequent sources can have an optional start time in seconds before them. Audio is extracted from video files. Syntax: quest mix "<base>" [<seconds> "<input>"]...'

    def parse(self, args):
        if len(args) < 1:
            return None, 'quest mix needs at least one source (the base): mix "source.wav" [20 "other.wav"]...'
        sources = []
        i = 0
        n = len(args)
        if args[0].lstrip('-').isdigit():
            return None, 'first source must not have a number before it — it is the base (starts at 0s)'
        sources.append({'path': args[0], 'start': 0.0})
        i = 1
        pending_start = None
        while i < n:
            arg = args[i]
            if arg.replace('.', '', 1).lstrip('-').isdigit():
                try:
                    pending_start = float(arg)
                except ValueError:
                    return None, f"invalid number '{arg}' — start time must be a number"
                i += 1
                if i >= n:
                    return None, f"number '{arg}' is not followed by a source path"
            else:
                start = pending_start if pending_start is not None else 0.0
                sources.append({'path': arg, 'start': start})
                pending_start = None
                i += 1
        if pending_start is not None:
            return None, 'trailing number with no source path after it'
        for s in sources:
            p = s['path']
            if not os.path.exists(p) and not _looks_like_url(p):
                return None, f"input file not found: {p}"
        return {'sources': sources}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        sources = parsed['sources']
        os.makedirs(results_dir, exist_ok=True)
        joined_name = '_'.join(
            re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(s['path']))[0])[:10]
            for s in sources
        )[:80] or 'mixed'
        out_name = f"voder_quest_mix_{joined_name}_{timestamp}.wav"
        out_path = os.path.join(results_dir, out_name)

        tmp_dir = tempfile.mkdtemp(prefix='voder_mix_')
        try:
            normalized = []
            for idx, s in enumerate(sources):
                raw = s['path']
                norm = os.path.join(tmp_dir, f"src_{idx:04d}.wav")
                if _looks_like_url(raw):
                    from voder import download_url_audio
                    ok_dl, err_msg, dl_path = download_url_audio(raw)
                    if not ok_dl:
                        print(f"Error: failed to download '{raw}': {err_msg}")
                        return False
                    if not _extract_audio(dl_path, norm):
                        print(f"Error: failed to extract audio from downloaded '{raw}'")
                        try:
                            os.unlink(dl_path)
                        except Exception:
                            pass
                        return False
                    try:
                        os.unlink(dl_path)
                    except Exception:
                        pass
                elif os.path.splitext(raw)[1].lower() in VIDEO_EXTENSIONS:
                    if not _extract_audio(raw, norm):
                        print(f"Error: failed to extract audio from video '{raw}'")
                        return False
                else:
                    if not _extract_audio(raw, norm):
                        print(f"Error: failed to read audio '{raw}'")
                        return False
                normalized.append(norm)

            inputs = []
            filter_parts = []
            for idx, norm in enumerate(normalized):
                inputs.extend(['-i', norm])
            amix_inputs = []
            for idx, s in enumerate(sources):
                delay_ms = int(s['start'] * 1000)
                if delay_ms > 0:
                    filter_parts.append(f'[{idx}:a]adelay={delay_ms}|{delay_ms}[a{idx}]')
                    amix_inputs.append(f'[a{idx}]')
                else:
                    amix_inputs.append(f'[{idx}:a]')
            amix_chain = ''.join(amix_inputs)
            n_inputs = len(normalized)
            filter_parts.append(f'{amix_chain}amix=inputs={n_inputs}:duration=longest:normalize=0[out]')
            filter_str = ';'.join(filter_parts)

            cmd = [
                'ffmpeg', '-y',
            ] + inputs + [
                '-filter_complex', filter_str,
                '-map', '[out]',
                '-c:a', 'pcm_s16le', '-ar', '44100', '-ac', '2',
                out_path,
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(out_path) or r.returncode != 0:
                print("Error: ffmpeg failed to mix sources")
                if r.stderr:
                    print(r.stderr[-800:])
                return False

            print(f"Quest mix ({len(sources)} sources) complete: {out_path}")
            for s in sources:
                dur = _ffprobe_duration(out_path)
                if dur:
                    print(f"  '{os.path.basename(s['path'])}' started at {s['start']:.1f}s")
            print(f"  Output duration: {_ffprobe_duration(out_path):.1f}s")

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


def _looks_like_url(s):
    return s.startswith('http://') or s.startswith('https://')


_register_side_quest(Quest)
