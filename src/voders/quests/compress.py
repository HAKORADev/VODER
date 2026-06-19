import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


LOSSY_LEVELS = {
    1: {'mp3': '256k', 'opus': '128k', 'aac': '192k', 'ogg': '256k', 'oga': '256k', 'wma': '160k', 'm4a': '192k', 'mp2': '256k', 'ac3': '256k'},
    2: {'mp3': '128k', 'opus': '72k',  'aac': '96k',  'ogg': '128k', 'oga': '128k', 'wma': '96k',  'm4a': '96k',  'mp2': '128k', 'ac3': '128k'},
    3: {'mp3': '64k',  'opus': '40k',  'aac': '56k',  'ogg': '64k',  'oga': '64k',  'wma': '56k',  'm4a': '56k',  'mp2': '64k',  'ac3': '64k'},
}

LOSSLESS_TARGETS = {
    1: {'sample_rate': 44100, 'bits': 24, 'flac_compression': 8},
    2: {'sample_rate': 32000, 'bits': 16, 'flac_compression': 10},
    3: {'sample_rate': 22050, 'bits': 16, 'flac_compression': 12},
}

LOSSY_FORMATS = {'mp3', 'opus', 'aac', 'ogg', 'oga', 'wma', 'm4a', 'mp4', 'mp2', 'ac3', 'amr'}
LOSSLESS_FORMATS = {'wav', 'flac', 'aiff', 'aif', 'aifc', 'amb', 'au', 'snd', 'caf'}


class Quest(SideQuest):
    name = 'compress'
    description = 'Compress an audio file at level 1 (low), 2 (default), or 3 (highest). Lower bitrates for lossy formats, lower bit-depth/sample-rate for lossless.'

    def parse(self, args):
        if len(args) == 1:
            level = 2
            path = args[0]
        elif len(args) == 2:
            try:
                level = int(args[0])
            except ValueError:
                return None, f"first argument must be a compression level (1, 2, or 3), got '{args[0]}'"
            if level not in (1, 2, 3):
                return None, f"compression level must be 1, 2, or 3 (got {level})"
            path = args[1]
        else:
            return None, "quest compress takes at most two arguments: [level] <input-audio-path>"
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        if not ext:
            return None, "input file has no extension; cannot determine format"
        if ext not in LOSSY_FORMATS and ext not in LOSSLESS_FORMATS:
            return None, f"compress does not support '.{ext}'; use quest convert to change to a known format first"
        return {'level': level, 'path': path, 'ext': ext}, None

    def _probe_audio_props(self, path):
        try:
            r = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                 '-show_entries', 'stream=sample_rate,bits_per_sample,channels',
                 '-of', 'default=nw=1', path],
                capture_output=True, text=True,
            )
            props = {}
            for line in r.stdout.splitlines():
                if '=' in line:
                    k, v = line.split('=', 1)
                    props[k.strip()] = v.strip()
            return props
        except Exception:
            return {}

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        level = parsed['level']
        path = parsed['path']
        ext = parsed['ext']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        out_name = f"voder_quest_compress_L{level}_{safe_name}_{timestamp}.{ext}"
        out_path = os.path.join(results_dir, out_name)

        cmd = ['ffmpeg', '-y', '-i', path]
        if ext in LOSSY_FORMATS:
            bitrate_map = LOSSY_LEVELS[level]
            codec_map = {
                'mp3': 'libmp3lame', 'opus': 'libopus', 'aac': 'aac',
                'ogg': 'libvorbis', 'oga': 'libvorbis', 'wma': 'wmav2',
                'm4a': 'aac', 'mp4': 'aac', 'mp2': 'mp2', 'ac3': 'ac3',
                'amr': 'libopencore_amrnb',
            }
            codec = codec_map.get(ext, 'aac')
            cmd += ['-vn', '-c:a', codec, '-b:a', bitrate_map[ext], '-ac', '2']
            if ext == 'opus':
                cmd += ['-vbr', 'on', '-application', 'voip']
            elif ext == 'amr':
                cmd += ['-ar', '8000', '-ac', '1']
        else:
            target = LOSSLESS_TARGETS[level]
            props = self._probe_audio_props(path)
            src_rate = int(props.get('sample_rate', '0') or '0') or 44100
            src_bits = int(props.get('bits_per_sample', '0') or '0') or 16

            target_rate = min(src_rate, target['sample_rate'])
            target_bits = min(src_bits if src_bits in (16, 24, 32) else 16, target['bits'])
            if target_bits not in (16, 24):
                target_bits = 16

            if ext == 'flac':
                cmd += ['-vn', '-c:a', 'flac', '-compression_level', str(target['flac_compression']),
                        '-ar', str(target_rate), '-ac', '2']
            else:
                muxer_map = {'wav': 'wav', 'aiff': 'aiff', 'aif': 'aiff', 'aifc': 'aiff',
                             'amb': 'amb', 'au': 'au', 'snd': 'au', 'caf': 'caf'}
                muxer = muxer_map.get(ext, ext)
                le_codec = f'pcm_s{target_bits}le'
                be_codec = f'pcm_s{target_bits}be'
                actual_codec = be_codec if ext in ('aiff', 'aif', 'aifc') else le_codec
                cmd += ['-vn', '-c:a', actual_codec, '-ar', str(target_rate), '-ac', '2', '-f', muxer]

        cmd.append(out_path)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out_path) or r.returncode != 0:
            print(f"Error: ffmpeg failed to compress")
            if r.stderr:
                print(r.stderr[-600:])
            return False
        src_size = os.path.getsize(path)
        dst_size = os.path.getsize(out_path)
        ratio = (1.0 - dst_size / max(src_size, 1)) * 100.0
        print(f"Quest compress (level {level}) complete: {out_path}  ({src_size} -> {dst_size} bytes, {ratio:+.1f}% size)")
        if result_path:
            try:
                shutil.copy2(out_path, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)

