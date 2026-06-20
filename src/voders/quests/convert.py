import os
import re
import shutil
import subprocess

from voders.sidequests import SideQuest, _register_side_quest


AUDIO_FORMATS = {
    'wav', 'mp3', 'flac', 'ogg', 'oga', 'opus', 'aac', 'm4a', 'mp4',
    'wma', 'asf', 'aiff', 'aif', 'aifc', 'ac3', 'amr', 'amb', 'au', 'snd',
    'gsm', 'vox', 'tta', 'wv', 'ape', 'mpc', 'mp2', 'mp1', 'mka', 'tta',
    'caf', 'dsf', 'dff', 'ircam', 'pvf', 'sln', 'iklax', 'xi', '8svx',
    'sf', 'sf2', 'sph', 'fap', 'nist', 'nistsphere', 'sox', 'raw',
}

FORMAT_CODECS = {
    'wav':  ('pcm_s24le',   'wav'),
    'mp3':  ('libmp3lame',  'mp3'),
    'flac': ('flac',        'flac'),
    'ogg':  ('libvorbis',   'ogg'),
    'oga':  ('libvorbis',   'oga'),
    'opus': ('libopus',     'opus'),
    'aac':  ('aac',         'adts'),
    'm4a':  ('aac',         'ipod'),
    'mp4':  ('aac',         'mp4'),
    'wma':  ('wmav2',       'asf'),
    'asf':  ('wmav2',       'asf'),
    'aiff': ('pcm_s16be',   'aiff'),
    'aif':  ('pcm_s16be',   'aiff'),
    'aifc': ('pcm_s16be',   'aiff'),
    'ac3':  ('ac3',         'ac3'),
    'amr':  ('libopencore_amrnb', 'amr'),
    'amb':  ('pcm_s16le',   'amb'),
    'au':   ('pcm_s16le',   'au'),
    'snd':  ('pcm_s16le',   'au'),
    'gsm':  ('gsm',         'gsm'),
    'tta':  ('tta',         'tta'),
    'wv':   ('wavpack',     'wv'),
    'ape':  ('ape',         'ape'),
    'mpc':  ('musepack',    'mpc'),
    'mp2':  ('mp2',         'mp2'),
    'mp1':  ('mp1',         'mp1'),
    'mka':  ('libvorbis',   'matroska'),
    'caf':  ('pcm_s16le',   'caf'),
    'dsf':  ('dsd_lsbf',    'dsf'),
    'dff':  ('dsd_lsbf',    'dff'),
    'sph':  ('pcm_s16le',   'sph'),
    'nist': ('pcm_s16le',   'nist'),
    'sln':  ('pcm_s16le',   'sln'),
    'raw':  ('pcm_s16le',   's16le'),
}

BITRATE_FORMATS = {'mp3', 'ogg', 'oga', 'opus', 'aac', 'm4a', 'mp4', 'wma', 'asf', 'ac3', 'mp2', 'mp1'}


class Quest(SideQuest):
    name = 'convert'
    category = 'Media Manipulation'
    description = 'Convert a local audio file to any other audio format (wav, mp3, flac, ogg, opus, aac, m4a, wma, aiff, ac3, amr, au, gsm, tta, wv, ape, mpc, mp2, mka, caf, dsf, dff, sph, sln, raw, ...). Same-format just copies.'

    def parse(self, args):
        if len(args) < 2:
            return None, "quest convert requires: <format> <input-audio-path>  (e.g., convert mp3 \"in.wav\")"
        if len(args) > 2:
            return None, "quest convert takes exactly two arguments: <format> <input-audio-path>"
        target = args[0].lower().lstrip('.')
        path = args[1]
        if target not in AUDIO_FORMATS:
            return None, f"unsupported target format '{target}'. Supported: {', '.join(sorted(AUDIO_FORMATS))}"
        if not os.path.exists(path):
            return None, f"input file not found: {path}"
        src_ext = os.path.splitext(path)[1].lower().lstrip('.')
        if src_ext not in AUDIO_FORMATS and src_ext != '':
            return None, f"input file does not look like a known audio format (got '.{src_ext}')"
        return {'target': target, 'path': path, 'src_ext': src_ext}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        target = parsed['target']
        path = parsed['path']
        src_ext = parsed['src_ext']
        os.makedirs(results_dir, exist_ok=True)
        original_name = os.path.splitext(os.path.basename(path))[0]
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', original_name)[:60] or 'input'
        out_name = f"voder_quest_convert_{safe_name}_{timestamp}.{target}"
        out_path = os.path.join(results_dir, out_name)

        if src_ext == target:
            shutil.copy2(path, out_path)
            print(f"Quest convert (same format, copied): {out_path}")
            final = out_path
        else:
            codec, muxer = FORMAT_CODECS.get(target, ('pcm_s16le', target))
            cmd = ['ffmpeg', '-y', '-i', path, '-vn', '-ac', '2', '-ar', '48000']
            if target in BITRATE_FORMATS:
                q = '320k' if target in ('mp3', 'mp2', 'mp1') else '256k'
                cmd += ['-c:a', codec, '-b:a', q, '-f', muxer]
            elif target == 'wav':
                cmd += ['-c:a', 'pcm_s24le', '-f', 'wav']
            elif target in ('flac', 'ape', 'tta', 'wv'):
                cmd += ['-c:a', codec, '-compression_level', '12', '-f', muxer]
            elif target == 'opus':
                cmd += ['-c:a', 'libopus', '-b:a', '160k', '-vbr', 'on', '-f', 'opus']
            elif target in ('ogg', 'oga'):
                cmd += ['-c:a', 'libvorbis', '-q:a', '8', '-f', muxer]
            else:
                cmd += ['-c:a', codec, '-f', muxer]
            cmd.append(out_path)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if not os.path.exists(out_path) or r.returncode != 0:
                print(f"Error: ffmpeg failed to convert to {target}")
                if r.stderr:
                    print(r.stderr[-600:])
                return False
            print(f"Quest convert ({src_ext} -> {target}) complete: {out_path}")
            final = out_path

        if result_path:
            try:
                shutil.copy2(final, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True


_register_side_quest(Quest)
