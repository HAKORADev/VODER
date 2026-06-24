import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}


def _to_seconds(token):
    if ':' in token:
        parts = token.split(':')
        if len(parts) == 2:
            try:
                return float(parts[0]) * 60 + float(parts[1])
            except ValueError:
                return None
        if len(parts) == 3:
            try:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            except ValueError:
                return None
        return None
    try:
        return float(token)
    except ValueError:
        return None


def _merge_overlapping(ranges):
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = []
    for start, end in sorted_ranges:
        if not merged:
            merged.append([start, end])
            continue
        last_start, last_end = merged[-1]
        if start <= last_end:
            if end > last_end:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


class Quest(SideQuest):
    name = 'remove'
    description = 'Inverse of cut: remove one or more time ranges from a local audio/video file, keeping the rest. Multi-range supported. Overlapping ranges are merged (no double-cutting). Syntax: quest remove "<start1>-<end1>" ["<start2>-<end2>" ...] <local-path>.'

    def parse(self, args):
        if len(args) < 2:
            return None, "quest remove takes at least one range and a path: \"<start>-<end>\" [...] <local-path>"
        path = args[-1]
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        range_tokens = args[:-1]
        ranges = []
        for tok in range_tokens:
            m = re.match(r'^([\d:]+(?:\.\d+)?)\-([\d:]+(?:\.\d+)?)$', tok)
            if not m:
                return None, f"range must be <start>-<end> using numbers or mm:ss or hh:mm:ss (got '{tok}')"
            start_s = _to_seconds(m.group(1))
            end_s = _to_seconds(m.group(2))
            if start_s is None or end_s is None:
                return None, f"could not parse start/end times in '{tok}'"
            if start_s < 0 or end_s < 0:
                return None, "start and end must be non-negative"
            if start_s >= end_s:
                return None, f"start ({start_s}s) must be strictly smaller than end ({end_s}s) in '{tok}'"
            ranges.append((start_s, end_s))
        return {'ranges': ranges, 'path': path}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        ranges = parsed['ranges']
        path = parsed['path']
        ext = os.path.splitext(path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'

        merged = _merge_overlapping(ranges)
        if len(merged) < len(ranges):
            print(f"Note: merged {len(ranges)} range(s) into {len(merged)} non-overlapping range(s).")

        ffprobe_cmd = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', path,
        ]
        pr = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
        if pr.returncode != 0:
            print(f"Error: ffprobe failed to read duration")
            if pr.stderr:
                print(pr.stderr[-600:])
            return False
        try:
            total_duration = float(pr.stdout.strip())
        except ValueError:
            print(f"Error: could not parse duration from ffprobe output: {pr.stdout!r}")
            return False

        keep_segments = []
        cursor = 0.0
        for rs, re_ in merged:
            if rs > cursor:
                keep_segments.append((cursor, rs))
            cursor = max(cursor, re_)
        if cursor < total_duration:
            keep_segments.append((cursor, total_duration))

        if not keep_segments:
            print(f"Error: all ranges cover the entire file — nothing would remain")
            return False

        ranges_tag = '_'.join(f"{int(rs)}-{int(re_)}s" for rs, re_ in merged)
        if is_video:
            out_name = f"voder_quest_remove_{safe_name}_{ranges_tag}_{timestamp}.mp4"
            out_path = os.path.join(results_dir, out_name)
            audio_codec = ['-c:a', 'aac', '-b:a', '256k']
            video_codec = ['-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
        else:
            out_name = f"voder_quest_remove_{safe_name}_{ranges_tag}_{timestamp}.wav"
            out_path = os.path.join(results_dir, out_name)
            audio_codec = ['-c:a', 'pcm_s24le', '-ar', '48000', '-ac', '2']
            video_codec = []

        filter_parts = []
        for idx, (s, e) in enumerate(keep_segments):
            filter_parts.append(f"[0:a]atrim={s:.3f}:{e:.3f},asetpts=PTS-STARTPTS[a{idx}]")
        concat_inputs = ''.join(f"[a{i}]" for i in range(len(keep_segments)))
        filter_parts.append(f"{concat_inputs}concat=n={len(keep_segments)}:v=0:a=1[outa]")
        audio_filter = ';'.join(filter_parts)

        cmd = ['ffmpeg', '-y', '-i', path]
        if is_video:
            keep_video_parts = []
            for idx, (s, e) in enumerate(keep_segments):
                keep_video_parts.append(f"[0:v]trim={s:.3f}:{e:.3f},setpts=PTS-STARTPTS[v{idx}]")
            concat_v_inputs = ''.join(f"[v{i}]" for i in range(len(keep_segments)))
            keep_video_parts.append(f"{concat_v_inputs}concat=n={len(keep_segments)}:v=1:a=0[outv]")
            video_filter = ';'.join(keep_video_parts)
            cmd += ['-filter_complex', f"{video_filter};{audio_filter}",
                    '-map', '[outv]', '-map', '[outa]']
            cmd += video_codec + audio_codec
        else:
            cmd += ['-filter_complex', audio_filter, '-map', '[outa]']
            cmd += audio_codec
        cmd += [out_path]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to remove ranges {ranges_tag}")
            if r.stderr:
                print(r.stderr[-1200:])
            return False
        print(f"Quest remove (cut {len(merged)} range(s), kept {len(keep_segments)} segment(s)) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
