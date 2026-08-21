import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


FLUX2_MAX_REFS = 5
VACE_MAX_REFS = 3
HYWORLD_MAX_REFS = 3
TRELLIS_MAX_REFS = 0


def _resolve_url(url, media_type='image'):
    try:
        from voder import download_url_image, download_url_video
        if media_type == 'video':
            local_path, err = download_url_video(url)
        elif media_type == 'audio':
            video_path, err = download_url_video(url)
            if not video_path:
                print(f"Error downloading video for audio extraction: {err}")
                return None
            audio_path = video_path.rsplit('.', 1)[0] + '.wav'
            try:
                import subprocess
                subprocess.run(
                    ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', audio_path],
                    capture_output=True, check=True, timeout=120
                )
                print(f"Audio extracted: {audio_path}")
                return audio_path
            except Exception as e:
                print(f"Error extracting audio: {e}")
                return None
        else:
            local_path, err = download_url_image(url)
        if local_path:
            print(f"Downloaded to: {local_path}")
            return local_path
        print(f"Error downloading {media_type}: {err or 'download failed'}")
        return None
    except ImportError:
        print("Error: VODER universal URL downloader not available (voder.py must be importable).")
        return None


def resolve_input_path(input_path, media_type='image'):
    if not input_path:
        return None

    if isinstance(input_path, tuple):
        tag, url = input_path
        if tag in ('url_image', 'url_video', 'url_audio'):
            media_type = tag.replace('url_', '')
            print(f"Downloading {media_type} from URL (forced): {url}")
            return _resolve_url(url, media_type=media_type)
        else:
            print(f"Downloading {media_type} from URL: {url}")
            return _resolve_url(url, media_type=media_type)

    if not (input_path.startswith('http://') or input_path.startswith('https://')):
        if os.path.exists(input_path):
            return input_path
        print(f"Error: input not found: {input_path}")
        return None

    print(f"Downloading {media_type} from URL: {input_path}")
    return _resolve_url(input_path, media_type=media_type)


def resolve_references(references, default_media_type='image'):
    if not references:
        return []
    resolved = []
    for ref in references:
        if isinstance(ref, tuple):
            tag, url = ref
            if tag in ('url_image', 'url_video', 'url_audio'):
                media_type = tag.replace('url_', '')
                resolved_ref = _resolve_url(url, media_type=media_type)
            else:
                resolved_ref = _resolve_url(url, media_type=default_media_type)
        elif ref and (ref.startswith('http://') or ref.startswith('https://')):
            resolved_ref = _resolve_url(ref, media_type=default_media_type)
        elif ref and os.path.exists(ref):
            resolved_ref = ref
        else:
            print(f"Warning: reference not found: {ref}")
            resolved_ref = None
        if resolved_ref:
            resolved.append(resolved_ref)
    return resolved


def check_reference_limit(refs, max_refs, model_name):
    if refs is None:
        return []
    if len(refs) > max_refs:
        print(f"Warning: {model_name} supports max {max_refs} references, got {len(refs)}. Using first {max_refs}.")
        refs = refs[:max_refs]
    return refs
