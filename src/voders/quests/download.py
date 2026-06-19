import os
import re
import shutil

from voders.sidequests import SideQuest, _register_side_quest


class Quest(SideQuest):
    name = 'download'
    description = 'Download a URL as audio (default) or video (with the video keyword). Output goes to results/.'

    def parse(self, args):
        from voder import is_youtube_url
        want_video = False
        url = None
        i = 0
        while i < len(args):
            a = args[i]
            al = a.lower()
            if al == 'video':
                want_video = True
                i += 1
            elif url is None and is_youtube_url(a):
                url = a
                i += 1
            elif url is None and os.path.exists(a):
                url = a
                i += 1
            else:
                return None, f"Unexpected argument for quest download: {a}"
        if url is None:
            return None, "quest download requires a URL or local file path"
        return {'url': url, 'want_video': want_video}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        from voder import is_youtube_url, download_youtube_audio, download_youtube_video
        url = parsed['url']
        want_video = parsed['want_video']
        os.makedirs(results_dir, exist_ok=True)
        original_name = self._derive_name(url)
        if want_video:
            if is_youtube_url(url):
                downloaded, info_or_err = download_youtube_video(url)
                if downloaded is None:
                    print(f"Error: {info_or_err}")
                    return False
                ext = os.path.splitext(downloaded)[1] or '.mp4'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(results_dir, out_name)
                shutil.move(downloaded, out_path)
                final = out_path
            else:
                if not os.path.exists(url):
                    print(f"Error: file not found: {url}")
                    return False
                ext = os.path.splitext(url)[1] or '.mp4'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(results_dir, out_name)
                shutil.copy2(url, out_path)
                final = out_path
            print(f"Quest download (video) complete: {final}")
        else:
            if is_youtube_url(url):
                ok, err, audio_path = download_youtube_audio(url)
                if not ok:
                    print(f"Error: {err}")
                    return False
                ext = os.path.splitext(audio_path)[1] or '.mp3'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(results_dir, out_name)
                shutil.move(audio_path, out_path)
                final = out_path
            else:
                if not os.path.exists(url):
                    print(f"Error: file not found: {url}")
                    return False
                ext = os.path.splitext(url)[1] or '.wav'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(results_dir, out_name)
                shutil.copy2(url, out_path)
                final = out_path
            print(f"Quest download (audio) complete: {final}")
        if result_path:
            try:
                shutil.copy2(final, result_path)
                print(f"Result copied to: {result_path}")
            except Exception as e:
                print(f"Note: could not copy to result path: {e}")
        return True

    @staticmethod
    def _derive_name(url):
        from voder import is_youtube_url
        if is_youtube_url(url):
            m = re.search(r'(?:v=|youtu\.be/|/video/|/embed/)([\w\-]{6,})', url)
            if m:
                return re.sub(r'[^A-Za-z0-9_\-]', '_', m.group(1))[:40]
            return re.sub(r'[^A-Za-z0-9_\-]', '_', url)[:40]
        base = os.path.basename(url)
        stem = os.path.splitext(base)[0]
        return re.sub(r'[^A-Za-z0-9_\-]', '_', stem)[:60] or 'input'


_register_side_quest(Quest)
