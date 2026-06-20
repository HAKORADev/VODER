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
    name = 'glue'
    category = 'Media Manipulation'
    description = 'Glue an audio file onto a video file (or vice versa). The "where-it-will-be-glued" gets the other source attached and any existing audio is auto-replaced. If audio is shorter than video, audio is padded with silence until the last video frame. If video is shorter than audio, video is extended with black frames until the audio ends. Refuses URLs and refuses audio-audio / video-video pairs. Syntax: quest glue "<input-to-use>" "<where-it-will-be-glued>".'

    def parse(self, args):
        if len(args) != 2:
            return None, "quest glue takes exactly two arguments: <input-to-use> <where-it-will-be-glued>"
        a, b = args[0], args[1]
        if '://' in a or '://' in b:
            return None, "quest glue refuses URLs — download both files first with quest download"
        if not os.path.exists(a):
            return None, f"first file not found: {a}"
        if not os.path.exists(b):
            return None, f"second file not found: {b}"
        a_ext = os.path.splitext(a)[1].lower()
        b_ext = os.path.splitext(b)[1].lower()
        a_is_video = a_ext in VIDEO_EXTENSIONS
        b_is_video = b_ext in VIDEO_EXTENSIONS
        if a_is_video and b_is_video:
            return None, "quest glue refuses video+video pairs — provide one audio and one video"
        if (not a_is_video) and (not b_is_video):
            return None, "quest glue refuses audio+audio pairs — provide one audio and one video"
        if a_is_video:
            video_path, audio_path = a, b
        else:
            video_path, audio_path = b, a
        return {
            'video_path': video_path,
            'audio_path': audio_path,
        }, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        video_path = parsed['video_path']
        audio_path = parsed['audio_path']
        os.makedirs(results_dir, exist_ok=True)

        v_dur = _ffprobe_duration(video_path)
        a_dur = _ffprobe_duration(audio_path)
        video_safe = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(video_path))[0])[:40] or 'video'
        audio_safe = re.sub(r'[^A-Za-z0-9_\-]', '_', os.path.splitext(os.path.basename(audio_path))[0])[:40] or 'audio'
        out_name = f"voder_quest_glue_{audio_safe}_onto_{video_safe}_{timestamp}.mp4"
        out_path = os.path.join(results_dir, out_name)

        if v_dur is None or a_dur is None:
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-i', audio_path,
                '-map', '0:v',
                '-map', '1:a',
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                '-c:a', 'aac', '-b:a', '256k',
                '-movflags', '+faststart',
                out_path,
            ]
            max_dur = max(v_dur or 0.0, a_dur or 0.0)
        else:
            max_dur = max(v_dur, a_dur)
            filter_parts = []
            if a_dur < v_dur:
                extra = v_dur - a_dur
                filter_parts.append(f"[1:a]apad=pad_dur={extra:.3f}[a]")
                audio_label = '[a]'
            else:
                audio_label = '1:a'
            if v_dur < a_dur:
                extra = a_dur - v_dur
                filter_parts.append(f"[0:v]tpad=stop_mode=add:stop_duration={extra:.3f}[v]")
                video_label = '[v]'
            else:
                video_label = '0:v'

            if filter_parts:
                filter_complex = ';'.join(filter_parts)
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-filter_complex', filter_complex,
                    '-map', video_label,
                    '-map', audio_label,
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '256k',
                    '-t', f'{max_dur:.3f}',
                    '-movflags', '+faststart',
                    out_path,
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-i', audio_path,
                    '-map', '0:v',
                    '-map', '1:a',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '256k',
                    '-t', f'{max_dur:.3f}',
                    '-movflags', '+faststart',
                    out_path,
                ]

        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to glue audio onto video")
            if r.stderr:
                print(r.stderr[-800:])
            return False
        print(f"Quest glue (audio '{audio_safe}' onto video '{video_safe}', total {max_dur:.2f}s) complete: {out_path}")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
