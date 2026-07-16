import os
import shutil

from voders.sidequests import SideQuest, _register_side_quest


_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.tiff', '.svg'}
_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v', '.wmv', '.ts', '.mts', '.3gp'}


def _url_extension(url):
    if not url:
        return ''
    cleaned = url.split('?')[0].split('#')[0]
    return os.path.splitext(cleaned)[1].lower()


class Quest(SideQuest):
    name = 'download'
    description = 'Download a URL as audio (default), video (with video keyword), or image (with image keyword). Supports YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter, Reddit. Experimental public_net support for other sites. Auto-detects image/video URLs by extension. Cookies retry (Chrome/Brave/Edge) on failure.'

    def parse(self, args):
        from voder import is_supported_url
        want_video = False
        want_image = False
        url = None
        i = 0
        while i < len(args):
            a = args[i]
            al = a.lower()
            if al == 'video':
                want_video = True
                i += 1
            elif al == 'image':
                want_image = True
                i += 1
            elif url is None and is_supported_url(a):
                url = a
                i += 1
            elif url is None and os.path.exists(a):
                url = a
                i += 1
            else:
                return None, f"Unexpected argument for quest download: {a}"
        if url is None:
            return None, "quest download requires a URL or local file path"
        if want_video and want_image:
            return None, "quest download: specify only one of 'video' or 'image'"
        return {'url': url, 'want_video': want_video, 'want_image': want_image}, None

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        from voder import (
            is_supported_url, is_public_net_url, download_url_audio,
            download_url_video, download_url_image, derive_output_name,
            is_video_url, platform_name, detect_platform,
        )
        url = parsed['url']
        want_video = parsed['want_video']
        want_image = parsed['want_image']
        if not want_video and not want_image and (url.startswith('http://') or url.startswith('https://')):
            ext = _url_extension(url)
            if ext in _IMAGE_EXTS:
                print(f"Auto-detected image URL (extension '{ext}') — routing to image download.")
                want_image = True
            elif ext in _VIDEO_EXTS:
                print(f"Auto-detected video URL (extension '{ext}') — routing to video download.")
                want_video = True
        downloads_dir = os.path.join(results_dir, 'downloads')
        os.makedirs(downloads_dir, exist_ok=True)
        original_name = self._derive_name(url)

        if want_image:
            images_dir = os.path.join(downloads_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)
            if is_supported_url(url) and not os.path.exists(url):
                downloaded, err = download_url_image(url, temp_dir=images_dir)
                if downloaded is None:
                    print(f"Error: {err}")
                    print(f"Hint: this URL did not resolve as an image. Try 'quest download \"{url}\"' for audio or 'quest download video \"{url}\"' for video.")
                    return False
                ext = os.path.splitext(downloaded)[1] or '.jpg'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(images_dir, out_name)
                if os.path.abspath(downloaded) != os.path.abspath(out_path):
                    shutil.move(downloaded, out_path)
                final = out_path
                print(f"Quest download (image) complete: {final}")
            else:
                if not os.path.exists(url):
                    print(f"Error: file not found: {url}")
                    return False
                ext = os.path.splitext(url)[1] or '.jpg'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(images_dir, out_name)
                shutil.copy2(url, out_path)
                final = out_path
                print(f"Quest download (image) complete: {final}")
        elif want_video:
            videos_dir = os.path.join(downloads_dir, 'videos')
            os.makedirs(videos_dir, exist_ok=True)
            if is_supported_url(url) and not os.path.exists(url):
                is_vid, verify_err, _pid = is_video_url(url, verify=True)
                if not is_vid:
                    print(f"Error: {verify_err or 'This link is not a video'}")
                    print(f"Hint: try 'quest download image \"{url}\"' if this is an image, or 'quest download \"{url}\"' for audio.")
                    return False
                downloaded, info_or_err = download_url_video(url)
                if downloaded is None:
                    print(f"Error: {info_or_err}")
                    return False
                ext = os.path.splitext(downloaded)[1] or '.mp4'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(videos_dir, out_name)
                shutil.move(downloaded, out_path)
                final = out_path
            else:
                if not os.path.exists(url):
                    print(f"Error: file not found: {url}")
                    return False
                ext = os.path.splitext(url)[1] or '.mp4'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(videos_dir, out_name)
                shutil.copy2(url, out_path)
                final = out_path
            print(f"Quest download (video) complete: {final}")
        else:
            audios_dir = os.path.join(downloads_dir, 'audios')
            os.makedirs(audios_dir, exist_ok=True)
            if is_supported_url(url) and not os.path.exists(url):
                is_vid, verify_err, _pid = is_video_url(url, verify=True)
                if not is_vid:
                    print(f"Error: {verify_err or 'This link is not a video'}")
                    print(f"Hint: this URL was not recognized as a video. Try 'quest download image \"{url}\"' if it is an image, or 'quest download video \"{url}\"' to force video.")
                    return False
                ok, err, audio_path = download_url_audio(url, skip_verify=True)
                if not ok:
                    print(f"Error: {err}")
                    print(f"Hint: this URL did not resolve as audio/video. Try 'quest download image \"{url}\"' if it is an image.")
                    return False
                ext = os.path.splitext(audio_path)[1] or '.mp3'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(audios_dir, out_name)
                shutil.move(audio_path, out_path)
                final = out_path
            else:
                if not os.path.exists(url):
                    print(f"Error: file not found: {url}")
                    return False
                ext = os.path.splitext(url)[1] or '.wav'
                out_name = f"voder_quest_download_{original_name}_{timestamp}{ext}"
                out_path = os.path.join(audios_dir, out_name)
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
        from voder import is_supported_url, derive_output_name
        if is_supported_url(url):
            return derive_output_name(url)
        base = os.path.basename(url)
        stem = os.path.splitext(base)[0]
        import re
        return re.sub(r'[^A-Za-z0-9_\-]', '_', stem)[:60] or 'input'


_register_side_quest(Quest)
