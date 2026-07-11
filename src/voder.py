import sys
import os

_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

_libs_dir = os.path.join(_src_dir, "libs")
if _libs_dir not in sys.path:
    sys.path.insert(0, _libs_dir)

MODELS_DIR = os.path.join(_src_dir, "models")

MODELS_TMP_DIR = os.path.join(MODELS_DIR, "tmp")
MODELS_CHECKPOINTS_DIR = os.path.join(MODELS_DIR, "checkpoints")

QWEN_TTS_VOICEDESIGN_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "qwen_tts_voicedesign")
QWEN_TTS_BASE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "qwen_tts_base")
FISH_S2PRO_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "fish_s2pro")
ACESTEP_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "acestep")
SEED_VC_V1_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v1")
SEED_VC_V2_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "seed_vc_v2")
WHISPER_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "whisper")
UNISE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "unise")
TANGOFLUX_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "tangoflux")
SVS_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "svs")
VIBEVOICE_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "vibevoice_asr")
TRANSLATE_GEMMA_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "translate_gemma")
AUDIOSR_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "audiosr")
ALIGNER_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "aligner")
HEAVY_VADAR_MODEL_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "heavy_vadar")
HEAVY_VADAR_MODEL_REPO = "OpenYourMind/gemma-4-12B-it-abliterated-uncensored"
LITE_VADAR_MODEL_DIR = os.path.join(MODELS_CHECKPOINTS_DIR, "lite_vadar")
LITE_VADAR_MODEL_REPO = "Jiunsong/SuperGemma-4-12b-abliterated-gguf-4bit"
LITE_VADAR_TEMPLATE_REPO = "Jiunsong/SuperGemma-4-12b-abliterated"
LITE_VADAR_GGUF_FILENAME = "SuperGemma-4-12b-abliterated-Q4_K_M.gguf"
LITE_VADAR_TEMPLATE_FILENAME = "chat_template.jinja"
VADAR_MODEL_DIR = HEAVY_VADAR_MODEL_DIR
VADAR_MODEL_REPO = HEAVY_VADAR_MODEL_REPO
VADAR_CONFIG_PATH = os.path.join(_src_dir, "voders", "vadars", "config.json")

_VADAR_CONFIG_DEFAULTS = {
    'model_repo': ('OpenYourMind/gemma-4-12B-it-abliterated-uncensored', str, None, None),
    'context_length': (262144, int, 4096, 262144),
    'sliding_window_threshold_percent': (95, int, 50, 99),
    'sliding_window_delete_percent': (5, int, 1, 50),
    'memory_cap_percent': (20, int, 1, 50),
    'global_context_cap_percent': (15, int, 1, 50),
    'temperature': (0.8, float, 0.1, 2.0),
    'top_p': (0.95, float, 0.1, 1.0),
    'top_k': (64, int, 1, 1000),
    'max_new_tokens': (1024, int, 64, 32768),
    'stream': (True, bool, None, None),
    'ping_time': (15, int, 0, 3600),
    'summarize_threshold': (1500, int, 100, 50000),
    'read_preview_lines': (100, int, 10, 500),
    'listen_max_segment': (60, int, 5, 300),
    'listen_auto_threshold': (30, int, 5, 120),
    'catcher_max_retries': (3, int, 1, 10),
    'lite_gpu_layers': (-1, int, -1, 999),
    'lite_n_threads': (-1, int, -1, 256),
    'lite_repeat_penalty': (1.1, float, 1.0, 2.0),
    'lite_verbose': (False, bool, None, None),
    'lite_context_length': (0, int, 0, 262144),
}

_VADAR_CONFIG = None


def vadar_load_config(force=False):
    global _VADAR_CONFIG
    if _VADAR_CONFIG is not None and not force:
        return _VADAR_CONFIG
    config = {}
    for key, (default, typ, min_val, max_val) in _VADAR_CONFIG_DEFAULTS.items():
        config[key] = default
    if os.path.exists(VADAR_CONFIG_PATH):
        try:
            import json
            with open(VADAR_CONFIG_PATH, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            for key, (default, typ, min_val, max_val) in _VADAR_CONFIG_DEFAULTS.items():
                if key in user_config:
                    val = user_config[key]
                    try:
                        if typ == bool:
                            val = bool(val)
                        elif typ == int:
                            val = int(val)
                        elif typ == float:
                            val = float(val)
                        elif typ == str:
                            val = str(val)
                    except (ValueError, TypeError):
                        continue
                    if min_val is not None and val < min_val:
                        val = min_val
                    if max_val is not None and val > max_val:
                        val = max_val
                    if key == 'ping_time' and isinstance(val, int):
                        if val == 0:
                            pass
                        elif val < 5:
                            val = 15
                    config[key] = val
        except Exception:
            pass
    _VADAR_CONFIG = config
    return config

os.environ["HF_HOME"] = MODELS_DIR
os.environ["HF_HUB_CACHE"] = MODELS_TMP_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_TMP_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MODELS_TMP_DIR

os.environ["XDG_CACHE_HOME"] = MODELS_TMP_DIR

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MODELS_TMP_DIR, exist_ok=True)
os.makedirs(MODELS_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(QWEN_TTS_VOICEDESIGN_DIR, exist_ok=True)
os.makedirs(QWEN_TTS_BASE_DIR, exist_ok=True)
os.makedirs(FISH_S2PRO_DIR, exist_ok=True)
os.makedirs(ACESTEP_DIR, exist_ok=True)
os.makedirs(SEED_VC_V1_DIR, exist_ok=True)
os.makedirs(SEED_VC_V2_DIR, exist_ok=True)
os.makedirs(WHISPER_DIR, exist_ok=True)
os.makedirs(UNISE_DIR, exist_ok=True)
os.makedirs(TANGOFLUX_DIR, exist_ok=True)
os.makedirs(SVS_DIR, exist_ok=True)
os.makedirs(VIBEVOICE_DIR, exist_ok=True)
os.makedirs(TRANSLATE_GEMMA_DIR, exist_ok=True)
os.makedirs(AUDIOSR_DIR, exist_ok=True)
os.makedirs(ALIGNER_DIR, exist_ok=True)

import time
import math
import tempfile
import shutil
import gc
import traceback
import threading
import numpy as np
import torch
import torchaudio
import yaml
import soundfile as sf
from omegaconf import DictConfig
from hydra.utils import instantiate
from huggingface_hub import hf_hub_download
import subprocess
import json
import re
import random
from urllib.parse import urlparse

HF_TOKEN_FILE = "HF_TOKEN.txt"

from voders.sidequests import (
    SideQuest,
    oneline_quest,
    SIDE_QUESTS,
    _register_side_quest,
)

PLATFORMS = {
    "youtube": {
        "name": "YouTube",
        "domains": [
            "youtube.com", "www.youtube.com", "m.youtube.com",
            "music.youtube.com",
        ],
        "short_domains": ["youtu.be"],
        "video_patterns": [
            r"^/watch\b", r"^/shorts/", r"^/embed/", r"^/v/",
            r"^/live/", r"^/[a-zA-Z0-9_-]{11}$",
        ],
        "non_video_patterns": [
            r"^/@", r"^/channel/", r"^/user/", r"^/c/", r"^/playlist",
            r"^/feed", r"^/results", r"^/account", r"^/gaming",
            r"^/studio", r"^/podcasts", r"^/news", r"^/music$",
        ],
    },
    "tiktok": {
        "name": "TikTok",
        "domains": [
            "tiktok.com", "www.tiktok.com", "m.tiktok.com",
        ],
        "short_domains": ["vm.tiktok.com", "vt.tiktok.com"],
        "video_patterns": [
            r"^/[^/]+/video/", r"^/[^/]+/photo/", r"^/t/",
        ],
        "non_video_patterns": [
            r"^/discover/", r"^/tag/", r"^/search", r"^/foryou",
            r"^/following", r"^/@[^/]+$", r"^/$",
        ],
    },
    "bilibili": {
        "name": "Bilibili",
        "domains": [
            "bilibili.com", "www.bilibili.com", "m.bilibili.com",
        ],
        "short_domains": ["b23.tv", "www.b23.tv"],
        "video_patterns": [
            r"^/video/", r"^/bangumi/play/", r"^/[bB][vV][0-9a-zA-Z]+",
            r"^/ep", r"^/cheese/play/",
        ],
        "non_video_patterns": [
            r"^/space", r"^/read/", r"^/account",
            r"^/doc/", r"^/audio/", r"^/$",
        ],
    },
    "snapchat": {
        "name": "Snapchat",
        "domains": [
            "snapchat.com", "www.snapchat.com", "story.snapchat.com",
            "m.snapchat.com",
        ],
        "short_domains": [],
        "video_patterns": [
            r"^/spotlight/", r"^/spotlight\b", r"^/add/", r"^/u/",
            r"^/t/", r"^/p/", r"^/story/",
        ],
        "non_video_patterns": [
            r"^/discover", r"^/accounts", r"^/$",
        ],
    },
    "instagram": {
        "name": "Instagram",
        "domains": [
            "instagram.com", "www.instagram.com", "m.instagram.com",
        ],
        "short_domains": ["instagr.am"],
        "video_patterns": [
            r"^/reel/", r"^/reels/", r"^/tv/", r"^/p/",
        ],
        "non_video_patterns": [
            r"^/explore", r"^/accounts", r"^/direct", r"^/[^/]+$", r"^/$",
        ],
    },
    "facebook": {
        "name": "Facebook",
        "domains": [
            "facebook.com", "www.facebook.com", "m.facebook.com",
            "web.facebook.com", "fb.com",
        ],
        "short_domains": ["fb.watch"],
        "video_patterns": [
            r"^/watch\b", r"^/watch/", r"^/reel/", r"^/video\.php",
            r"^/[^/]+/videos/", r"^/[^/]+/reel/",
            r"^/story\.php", r"^/share/v/", r"^/share/p/",
        ],
        "non_video_patterns": [
            r"^/profile", r"^/groups", r"^/pages", r"^/photos",
            r"^/marketplace", r"^/gaming", r"^/$",
        ],
    },
    "twitter": {
        "name": "X (Twitter)",
        "domains": [
            "twitter.com", "www.twitter.com", "x.com", "www.x.com",
            "mobile.twitter.com",
        ],
        "short_domains": ["t.co"],
        "video_patterns": [
            r"^/[^/]+/status/", r"^/[^/]+/statuses/",
            r"^/i/broadcasts/", r"^/[^/]+/live/",
        ],
        "non_video_patterns": [
            r"^/home", r"^/search", r"^/explore", r"^/notifications",
            r"^/messages", r"^/settings", r"^/compose", r"^/i/",
            r"^/[^/]+$", r"^/$",
        ],
    },
    "reddit": {
        "name": "Reddit",
        "domains": [
            "reddit.com", "www.reddit.com", "old.reddit.com",
            "m.reddit.com", "new.reddit.com",
        ],
        "short_domains": ["redd.it"],
        "video_patterns": [
            r"^/r/[^/]+/comments/", r"^/comments/", r"^/user/[^/]+/comments/",
        ],
        "non_video_patterns": [
            r"^/r/[^/]+/?$", r"^/r/[^/]+/about", r"^/user/[^/]+/?$",
            r"^/search", r"^/$",
        ],
    },
}


_URL_DOMAIN_INDEX = {}
_URL_SHORT_DOMAIN_INDEX = {}
for _platform_id, _platform_info in PLATFORMS.items():
    for _domain in _platform_info["domains"]:
        _URL_DOMAIN_INDEX[_domain.lower()] = _platform_id
    for _short in _platform_info.get("short_domains", []):
        _URL_DOMAIN_INDEX[_short.lower()] = _platform_id
        _URL_SHORT_DOMAIN_INDEX[_short.lower()] = _platform_id


def _normalize_url(url):
    if url is None:
        return url
    url = url.strip()
    if not url:
        return url
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", url):
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
    return url


def _host_of(url):
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def detect_platform(url):
    if not url:
        return None
    normalized = _normalize_url(url)
    host = _host_of(normalized)
    if not host:
        return None
    return _URL_DOMAIN_INDEX.get(host)


def platform_name(platform_id):
    if not platform_id or platform_id not in PLATFORMS:
        return "URL"
    return PLATFORMS[platform_id]["name"]


def is_supported_url(url):
    if detect_platform(url) is not None:
        return True
    normalized = _normalize_url(url)
    if normalized and (normalized.startswith('http://') or normalized.startswith('https://')):
        return True
    return False


def is_public_net_url(url):
    return detect_platform(url) is None and is_supported_url(url)


def _matches_any(path, patterns):
    return any(re.match(p, path) for p in patterns)


def classify_url(url):
    platform_id = detect_platform(url)
    if not platform_id:
        normalized = _normalize_url(url)
        if normalized and (normalized.startswith('http://') or normalized.startswith('https://')):
            return "public_net", None
        return "unsupported", None
    normalized = _normalize_url(url)
    try:
        parsed = urlparse(normalized)
    except Exception:
        return "unsupported", None
    host = _host_of(normalized)
    path = parsed.path or "/"
    info = PLATFORMS[platform_id]
    if host in _URL_SHORT_DOMAIN_INDEX:
        if path and path != "/":
            return "video", platform_id
        return "ambiguous", platform_id
    if _matches_any(path, info["non_video_patterns"]):
        return "non_video", platform_id
    if _matches_any(path, info["video_patterns"]):
        return "video", platform_id
    return "ambiguous", platform_id


def verify_is_video(url):
    try:
        import yt_dlp
    except ImportError:
        return False, "yt-dlp not installed. Run: pip install yt-dlp"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "noplaylist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return False, "Could not extract video information"
            itype = info.get("_type")
            if itype in ("playlist", "multi_video"):
                return False, "URL points to a playlist, not a single video"
            if itype == "url":
                return False, "URL does not resolve to a playable video"
            formats = info.get("formats")
            direct_url = info.get("url")
            if not formats and not direct_url:
                return False, "URL does not contain a downloadable video stream"
            duration = info.get("duration")
            if duration is not None and duration <= 0 and not formats and not direct_url:
                return False, "URL does not contain a downloadable video stream"
            return True, info
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Unsupported URL" in msg:
            return False, "URL is not supported by yt-dlp"
        if "Video unavailable" in msg or "Private video" in msg:
            return False, "Video is unavailable"
        if "is not a valid URL" in msg:
            return False, "Invalid URL"
        if "HTTP Error" in msg:
            return False, f"Network error: {msg}"
        if "Connection" in msg or "Timed out" in msg:
            return False, "Connection error: check your internet connection"
        return False, f"Could not verify video: {msg}"
    except Exception as e:
        return False, f"Verification error: {str(e)}"


def is_video_url(url, verify=True):
    category, platform_id = classify_url(url)
    if category == "unsupported":
        return False, "URL is not from a supported platform", None
    if category == "non_video":
        pname = platform_name(platform_id)
        return False, f"This {pname} link does not point to a video", None
    if not verify:
        return True, None, platform_id
    ok, result = verify_is_video(_normalize_url(url))
    if not ok:
        return False, result, None
    return True, None, platform_id


def derive_video_id(url):
    platform_id = detect_platform(url)
    if not platform_id:
        return None
    normalized = _normalize_url(url)
    try:
        parsed = urlparse(normalized)
    except Exception:
        return None
    host = _host_of(normalized)
    path = parsed.path or ""
    query = parsed.query or ""

    if platform_id == "youtube":
        m = re.search(r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/|/v/)([A-Za-z0-9_-]{6,})", normalized)
        if m:
            return m.group(1)
        return None

    if platform_id == "tiktok":
        m = re.search(r"/video/(\d+)", path)
        if m:
            return m.group(1)
        m = re.search(r"/t/([A-Za-z0-9]+)", path)
        if m:
            return m.group(1)
        return None

    if platform_id == "bilibili":
        m = re.search(r"/video/([bB][vV][0-9a-zA-Z]+)", path)
        if m:
            return m.group(1)
        m = re.search(r"/(av\d+)", path, re.IGNORECASE)
        if m:
            return m.group(1)
        if "b23.tv" in host:
            tail = path.strip("/")
            if tail:
                return tail.split("/")[0]
        return None

    if platform_id == "snapchat":
        m = re.search(r"/spotlight/([A-Za-z0-9_-]+)", path)
        if m:
            return m.group(1)
        m = re.search(r"/u/([A-Za-z0-9_-]+)", path)
        if m:
            return m.group(1)
        m = re.search(r"/t/([A-Za-z0-9_-]+)", path)
        if m:
            return m.group(1)
        return None

    if platform_id == "instagram":
        m = re.search(r"/(reel|reels|p|tv)/([A-Za-z0-9_-]+)", path)
        if m:
            return m.group(2)
        return None

    if platform_id == "facebook":
        m = re.search(r"/videos/(?:[^/]+/)?(\d+)", path)
        if m:
            return m.group(1)
        m = re.search(r"(?:^|[?&])v=(\d+)", normalized)
        if m:
            return m.group(1)
        m = re.search(r"/reel/(\d+)", path)
        if m:
            return m.group(1)
        if "fb.watch" in host:
            tail = path.strip("/")
            if tail:
                return tail.split("/")[0]
        return None

    if platform_id == "twitter":
        m = re.search(r"/status(?:es)?/(\d+)", path)
        if m:
            return m.group(1)
        return None

    return None


def _safe_url_name(value, max_len=40):
    if not value:
        return "input"
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", str(value))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return (cleaned or "input")[:max_len]


def derive_output_name(url):
    platform_id = detect_platform(url)
    if not platform_id:
        base = os.path.basename(url or "")
        stem = os.path.splitext(base)[0]
        return _safe_url_name(stem, max_len=60)
    vid = derive_video_id(url)
    if vid:
        return _safe_url_name(vid, max_len=40)
    return _safe_url_name(platform_id, max_len=20)


def download_url_audio(url, temp_dir=None, skip_verify=False):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    platform_id = detect_platform(url)
    pname = platform_name(platform_id) if platform_id else "public_net"

    category, _ = classify_url(url)
    if category == "non_video":
        return False, f"This {pname} link does not point to a video", None
    if category == "public_net":
        print(f"WARNING: This platform is not officially supported. Results may vary — they are untested and we do not know what you may face.")
    if category == "unsupported":
        return False, f"URL is not supported", None

    try:
        import yt_dlp
    except ImportError:
        return False, "yt-dlp not installed. Run: pip install yt-dlp", None

    output_path = os.path.join(temp_dir, f"voder_url_{int(time.time())}")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "noplaylist": True,
    }

    normalized = _normalize_url(url)
    print(f"Downloading audio from {pname}: {normalized}")

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(normalized, download=False)
            if info is None:
                return False, "Failed to fetch video information", None

            itype = info.get("_type")
            if itype in ("playlist", "multi_video"):
                return False, f"This {pname} link points to a playlist, not a single video", None
            if itype == "url":
                return False, f"This {pname} link does not resolve to a playable video", None

            formats = info.get("formats")
            direct_url = info.get("url")
            if not formats and not direct_url:
                return False, f"This {pname} link does not contain a downloadable video stream", None

            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            print(f"Video: {title} ({duration}s)")

            if not info:
                return False, "Network error: Could not access video", None

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            if "is not a valid URL" in error_msg:
                return False, f"Invalid {pname} URL", None
            elif "Video unavailable" in error_msg or "Private video" in error_msg:
                return False, "Video is unavailable", None
            elif "Unsupported URL" in error_msg:
                return False, f"URL is not supported by yt-dlp", None
            elif "HTTP Error" in error_msg:
                return False, f"Network error: {error_msg}", None
            elif "Connection" in error_msg or "Timed out" in error_msg:
                return False, "Connection error: Check your internet connection", None
            else:
                return False, f"Download error: {error_msg}", None
        except Exception as e:
            return False, f"Error checking video: {str(e)}", None

        try:
            print("Extracting audio...")
            ydl.download([normalized])
        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            need_cookies = any(k in error_msg for k in ('Sign in', 'login', 'Login', 'restricted', 'age', 'private', 'cookies', 'HTTP Error', '403', '401'))
            if need_cookies:
                print("Download failed without cookies. Retrying with browser cookies...")
                cookies_ok, cookies_err = _ydl_download_with_cookies_retry(
                    ydl_opts, normalized,
                    lambda y: y.download([normalized])
                )
                if cookies_ok is not None:
                    pass
                else:
                    if "HTTP Error" in error_msg:
                        return False, f"Network error during download: {error_msg}", None
                    elif "Connection" in error_msg:
                        return False, "Connection lost during download", None
                    else:
                        return False, f"Download failed (with and without cookies): {cookies_err or error_msg}", None
            else:
                if "HTTP Error" in error_msg:
                    return False, f"Network error during download: {error_msg}", None
                elif "Connection" in error_msg:
                    return False, "Connection lost during download", None
                else:
                    return False, f"Download failed: {error_msg}", None
        except Exception as e:
            return False, f"Download error: {str(e)}", None

    mp3_path = output_path + ".mp3"
    if os.path.exists(mp3_path):
        print(f"Audio downloaded successfully: {mp3_path}")
        return True, None, mp3_path

    for ext in [".m4a", ".wav", ".webm"]:
        alt_path = output_path + ext
        if os.path.exists(alt_path):
            if ext != ".mp3":
                try:
                    waveform, sr = torchaudio.load(alt_path)
                    torchaudio.save(mp3_path, waveform, sr)
                    os.unlink(alt_path)
                    print(f"Audio downloaded and converted: {mp3_path}")
                    return True, None, mp3_path
                except Exception:
                    return True, None, alt_path
            return True, None, alt_path

    return False, "Downloaded file not found", None


def download_url_video(url, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()

    platform_id = detect_platform(url)
    pname = platform_name(platform_id) if platform_id else "public_net"

    category, _ = classify_url(url)
    if category == "non_video":
        return None, f"This {pname} link does not point to a video"
    if category == "public_net":
        print(f"WARNING: This platform is not officially supported. Results may vary — they are untested and we do not know what you may face.")
    if category == "unsupported":
        return None, f"URL is not supported"

    try:
        import yt_dlp
    except ImportError:
        return None, "yt-dlp not installed. Run: pip install yt-dlp"

    output_path = os.path.join(temp_dir, f"voder_svs_{int(time.time())}.mp4")
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": output_path,
        "merge_output_format": "mp4",
        "quiet": False,
        "no_warnings": False,
        "noplaylist": True,
    }

    normalized = _normalize_url(url)
    print(f"Downloading video from {pname}: {normalized}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(normalized, download=False)
            if info is None:
                return None, "Failed to fetch video information"

            itype = info.get("_type")
            if itype in ("playlist", "multi_video"):
                return None, f"This {pname} link points to a playlist, not a single video"
            if itype == "url":
                return None, f"This {pname} link does not resolve to a playable video"

            formats = info.get("formats")
            direct_url = info.get("url")
            if not formats and not direct_url:
                return None, f"This {pname} link does not contain a downloadable video stream"

            title = info.get("title", "Unknown")
            duration = info.get("duration", 0)
            print(f"Video: {title} ({duration}s)")
            try:
                ydl.download([normalized])
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)
                need_cookies = any(k in error_msg for k in ('Sign in', 'login', 'Login', 'restricted', 'age', 'private', 'cookies', 'HTTP Error', '403', '401'))
                if need_cookies:
                    print("Download failed without cookies. Retrying with browser cookies...")
                    cookies_ok, cookies_err = _ydl_download_with_cookies_retry(
                        ydl_opts, normalized,
                        lambda y: y.download([normalized])
                    )
                    if cookies_ok is None:
                        return None, f"Download failed (with and without cookies): {cookies_err or error_msg}"
                else:
                    raise

        if os.path.exists(output_path):
            print(f"Video downloaded: {output_path}")
            return output_path, title
        for ext in [".mp4", ".mkv", ".webm"]:
            alt = output_path.replace(".mp4", ext)
            if os.path.exists(alt):
                print(f"Video downloaded: {alt}")
                return alt, title
        return None, "Downloaded file not found"
    except yt_dlp.utils.DownloadError as e:
        msg = str(e)
        if "Unsupported URL" in msg:
            return None, "URL is not supported by yt-dlp"
        if "Video unavailable" in msg or "Private video" in msg:
            return None, "Video is unavailable"
        if "is not a valid URL" in msg:
            return None, "Invalid URL"
        return None, f"Download error: {msg}"
    except Exception as e:
        return None, f"Download error: {str(e)}"


def is_youtube_url(url):
    return is_supported_url(url)


def download_youtube_audio(url, temp_dir=None):
    return download_url_audio(url, temp_dir=temp_dir)


def download_youtube_video(url, temp_dir=None):
    return download_url_video(url, temp_dir=temp_dir)


_COOKERS_BROWSERS = ['chrome', 'brave', 'edge']


def _ydl_download_with_cookies_retry(ydl_opts_base, url, download_fn):
    last_err = None
    for browser in [None] + _COOKERS_BROWSERS:
        opts = dict(ydl_opts_base)
        if browser is not None:
            opts['cookiesfrombrowser'] = (browser,)
            print(f"Retrying with cookies from {browser}...")
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(opts) as ydl:
                result = download_fn(ydl)
            return result, None
        except yt_dlp.utils.DownloadError as e:
            last_err = str(e)
            if browser is None and any(k in last_err for k in ('Sign in', 'login', 'Login', 'restricted', 'age', 'private', 'cookies')):
                continue
            if browser is not None and browser != _COOKERS_BROWSERS[-1]:
                continue
            return None, last_err
        except Exception as e:
            last_err = str(e)
            if browser is not None and browser != _COOKERS_BROWSERS[-1]:
                continue
            return None, last_err
    return None, last_err or "All download attempts failed"


def download_url_image(url, temp_dir=None):
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)
    platform_id = detect_platform(url)
    pname = platform_name(platform_id) if platform_id else "public_net"
    if platform_id is None:
        print(f"WARNING: This platform is not officially supported. Results may vary — they are untested and we do not know what you may face.")
    print(f"Downloading image from {pname}: {_normalize_url(url)}")
    target_dir = os.path.join(temp_dir, f"voder_img_{int(time.time())}")
    os.makedirs(target_dir, exist_ok=True)
    cmd_base = ['gallery-dl', '--directory', target_dir]
    browsers = [None] + _COOKERS_BROWSERS
    last_err = None
    for browser in browsers:
        cmd = list(cmd_base)
        if browser is not None:
            cmd.extend(['--cookies-from-browser', browser])
            print(f"Retrying with cookies from {browser}...")
        cmd.append(_normalize_url(url))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                found = []
                for root, dirs, files in os.walk(target_dir):
                    for f in files:
                        ext = os.path.splitext(f)[1].lower()
                        if ext in {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}:
                            found.append(os.path.join(root, f))
                if found:
                    found.sort(key=os.path.getmtime)
                    print(f"Image downloaded successfully: {found[0]}")
                    return found[0], None
                last_err = "gallery-dl succeeded but no image file was found"
            else:
                err = r.stderr.strip() if r.stderr else r.stdout.strip()
                last_err = err[-500:] if err else "gallery-dl failed"
                if 'No extractor found' in str(last_err):
                    return None, f"URL is not supported by gallery-dl: {last_err}"
                if browser is not None and browser != _COOKERS_BROWSERS[-1]:
                    continue
                return None, f"gallery-dl error: {last_err}"
        except FileNotFoundError:
            return None, "gallery-dl is not installed. Run: pip install gallery-dl"
        except subprocess.TimeoutExpired:
            last_err = "gallery-dl timed out (120s)"
            if browser is not None and browser != _COOKERS_BROWSERS[-1]:
                continue
            return None, last_err
        except Exception as e:
            last_err = str(e)
            if browser is not None and browser != _COOKERS_BROWSERS[-1]:
                continue
            return None, last_err
    return None, last_err or "Image download failed"


def get_url_media_info(url):
    normalized = _normalize_url(url)
    try:
        import yt_dlp
        ydl_opts = {'quiet': True, 'no_warnings': True, 'skip_download': True, 'noplaylist': True}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(normalized, download=False)
            if info:
                clean = ydl.sanitize_info(info) if hasattr(ydl, 'sanitize_info') else info
                itype = clean.get('_type')
                if itype in ('playlist', 'multi_video'):
                    return False, None, None, f"This {platform_name(detect_platform(url))} link points to a playlist, not a single video"
                formats = clean.get('formats')
                direct_url = clean.get('url')
                if formats or direct_url:
                    media_type = 'video'
                    return True, media_type, {
                        'title': clean.get('title', 'Unknown'),
                        'duration': clean.get('duration'),
                        'extractor': clean.get('extractor') or clean.get('extractor_key'),
                        'platform': detect_platform(url) or 'public_net',
                        'uploader': clean.get('uploader'),
                    }, None
        except yt_dlp.utils.UnsupportedError:
            pass
        except yt_dlp.utils.DownloadError as e:
            err = str(e)
            if 'Unsupported URL' not in err and 'No extractor' not in err:
                return False, None, None, err
    except ImportError:
        pass

    try:
        cmd = ['gallery-dl', '-j', normalized]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and r.stdout.strip():
            import json
            metadata = json.loads(r.stdout)
            for item in metadata:
                if isinstance(item, list) and len(item) >= 3 and item[0] == 3:
                    info_dict = item[2] if isinstance(item[2], dict) else {}
                    return True, 'image', {
                        'title': info_dict.get('title') or info_dict.get('filename') or 'Unknown',
                        'duration': None,
                        'extractor': info_dict.get('category') or 'gallery-dl',
                        'platform': detect_platform(url) or 'public_net',
                        'uploader': info_dict.get('user') or info_dict.get('account'),
                    }, None
            if metadata:
                return True, 'image', {
                    'title': 'Unknown',
                    'duration': None,
                    'extractor': 'gallery-dl',
                    'platform': detect_platform(url) or 'public_net',
                    'uploader': None,
                }, None
        err = (r.stderr or '').strip()
        if 'No extractor found' in err:
            return False, None, None, "URL is not supported by yt-dlp or gallery-dl"
    except FileNotFoundError:
        return False, None, None, "Neither yt-dlp nor gallery-dl could process this URL"
    except Exception as e:
        return False, None, None, f"Info check error: {e}"

    return False, None, None, "URL is not supported by yt-dlp or gallery-dl"


_MODE_PREFIXES = {
    'tts': 'tts', 'sts': 'sts', 'ttm': 'ttm', 'stt': 'stt',
    'se': 'se', 'sfx': 'sfx', 'svs': 'svs', 'ss': 'ss',
    'quest': 'quest', 'chains': 'chains',
}


def organize_results(results_dir=None):
    if results_dir is None:
        results_dir = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(results_dir):
        return
    for fname in os.listdir(results_dir):
        fpath = os.path.join(results_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.startswith("voder_"):
            continue
        after = fname[len("voder_"):]
        mode = None
        for prefix in sorted(_MODE_PREFIXES.keys(), key=len, reverse=True):
            if after.startswith(prefix + "_") or after.startswith(prefix):
                mode = prefix
                break
        if mode is None:
            continue
        mode_dir = os.path.join(results_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        dest = os.path.join(mode_dir, fname)
        if not os.path.exists(dest):
            try:
                shutil.copy2(fpath, dest)
            except Exception:
                pass


def setup_hf_token():
    if not os.path.exists(HF_TOKEN_FILE):
        with open(HF_TOKEN_FILE, 'w') as f:
            f.write("# Paste your HuggingFace token here\n")
            f.write("# Get your token from: https://huggingface.co/settings/tokens\n")
            f.write("# Some models may require a token for gated repositories\n")
        return None
    with open(HF_TOKEN_FILE, 'r') as f:
        content = f.read().strip()
        lines = [line for line in content.split('\n') if line and not line.startswith('#')]
        if lines:
            return lines[0]
    return None

hf_token = setup_hf_token()
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
else:
    possible_paths = [
        "HF_TOKEN.txt",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "HF_TOKEN.txt"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "1", "HF_TOKEN.txt"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read().strip()
                lines = [line for line in content.split('\n') if line and not line.startswith('#')]
                if lines:
                    hf_token = lines[0]
                    os.environ["HF_TOKEN"] = hf_token
                    break
    if not hf_token and not os.environ.get("VODER_HF_WARNING_SHOWN"):
        os.environ["VODER_HF_WARNING_SHOWN"] = "1"
        print("\n" + "="*60)
        print("WARNING: HuggingFace token not found!")
        print("="*60)
        print("To use pyannote speaker diarization, you need to:")
        print("1. Get a token from: https://huggingface.co/settings/tokens")
        print("2. Create a file called 'HF_TOKEN.txt' with your token")
        print("3. Make sure the token has access to pyannote models")
        print("   (Accept conditions at: https://huggingface.co/pyannote/speaker-diarization-community-1)")
        print("="*60 + "\n")

def load_custom_model_from_hf(repo_id, model_filename="pytorch_model.bin", config_filename=None, target_dir=None):
    if target_dir is None:
        target_dir = SEED_VC_V2_DIR
    os.makedirs(target_dir, exist_ok=True)
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename, cache_dir=target_dir)
    if config_filename is None:
        return model_path
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename, cache_dir=target_dir)
    return model_path, config_path

class WhisperSTT:
    def __init__(self, model_dir=None, skip_turbo=False):
        self.model_dir = WHISPER_DIR if model_dir is None else model_dir
        self.model = None
        self.checkpoint_path = os.path.join(self.model_dir, "whisper-turbo.pt")
        self.translate_model = None
        self.translate_checkpoint_path = os.path.join(self.model_dir, "whisper-large-v3.pt")
        if not skip_turbo:
            self.ensure_model()

    def _save_checkpoint(self, model, path):
        import torch
        checkpoint = {
            "dims": {
                "n_mels": model.dims.n_mels,
                "n_audio_ctx": model.dims.n_audio_ctx,
                "n_audio_state": model.dims.n_audio_state,
                "n_audio_head": model.dims.n_audio_head,
                "n_audio_layer": model.dims.n_audio_layer,
                "n_vocab": model.dims.n_vocab,
                "n_text_ctx": model.dims.n_text_ctx,
                "n_text_state": model.dims.n_text_state,
                "n_text_head": model.dims.n_text_head,
                "n_text_layer": model.dims.n_text_layer,
            },
            "model_state_dict": model.state_dict(),
        }
        torch.save(checkpoint, path)

    def _load_model(self, model_name, checkpoint_path):
        import whisper
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            if os.path.exists(checkpoint_path):
                return whisper.load_model(checkpoint_path)
            else:
                model = whisper.load_model(model_name)
                self._save_checkpoint(model, checkpoint_path)
                return model
        except Exception as e:
            print(f"Error loading Whisper: {e}")
            return None

    def ensure_model(self):
        if self.model is None:
            self.model = self._load_model("large-v3-turbo", self.checkpoint_path)

    def ensure_translate_model(self):
        if self.translate_model is None:
            self.translate_model = self._load_model("large-v3", self.translate_checkpoint_path)

    def transcribe(self, audio_path):
        if self.model is None:
            return None
        try:
            result = self.model.transcribe(audio_path, word_timestamps=True)
            return result
        except Exception as e:
            print(f"Transcription error: {e}")
            return None

    def translate(self, audio_path):
        self.ensure_translate_model()
        if self.translate_model is None:
            return None
        try:
            result = self.translate_model.transcribe(audio_path, task="translate", word_timestamps=True)
            return result
        except Exception as e:
            print(f"Translation error: {e}")
            return None

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.translate_model is not None:
            del self.translate_model
            self.translate_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class EasyOCRReader:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.easyocr_dir = os.path.join(self.model_dir, "easyocr")
        self.model = None
        self.reader = None
        os.makedirs(self.easyocr_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.easyocr_dir, exist_ok=True)
        if self.reader is None:
            try:
                import easyocr
                print("Loading EasyOCR model...")
                self.reader = easyocr.Reader(
                    ['en'],
                    model_storage_directory=self.easyocr_dir,
                    download_enabled=True,
                    gpu=False
                )
                print("EasyOCR model loaded successfully")
            except Exception as e:
                print(f"Error loading EasyOCR: {e}")
                print("Note: EasyOCR will use CPU for text recognition.")

    def read_text(self, image_path):
        if self.reader is None:
            return None
        try:
            result = self.reader.readtext(image_path)
            return result
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return None

    def extract_text_from_image(self, image_path):
        if self.reader is None:
            return False, None, "EasyOCR model not loaded"

        try:
            result = self.read_text(image_path)
            if not result:
                return False, None, "No text found in image"

            texts = []
            for detection in result:
                text = detection[1].strip()
                if text:
                    texts.append(text)

            if not texts:
                return False, None, "No text found in image"

            full_text = ' '.join(texts)
            return True, full_text, None

        except Exception as e:
            return False, None, f"Error extracting text: {str(e)}"

    def cleanup(self):
        self.reader = None
        gc.collect()

class SpeakerDiarization:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.diarization_dir = os.path.join(self.model_dir, "pyannote")
        self.model = None
        self.pipeline = None
        os.makedirs(self.diarization_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.diarization_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                libs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libs')
                if libs_path not in sys.path:
                    sys.path.insert(0, libs_path)

                os.environ["PYANNOTE_SKIP_DEPENDENCY_CHECK"] = "1"

                from pyannote.audio import Pipeline
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                print("Loading pyannote speaker diarization model...")

                token = os.environ.get("HF_TOKEN")
                if not token:
                    print("Error: HuggingFace token is required for pyannote")
                    return

                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-community-1",
                    cache_dir=self.diarization_dir,
                    token=token
                )
                self.pipeline = self.pipeline.to(device)
                print("Speaker diarization model loaded successfully")
            except ImportError as e:
                print(f"Error: pyannote.audio not available in local libs")
                print(f"Import error: {e}")
            except Exception as e:
                error_str = str(e).lower()
                if 'audio_metadata' in error_str or 'torchaudio' in error_str:
                    print(f"Error loading speaker diarization model: torchaudio compatibility issue")
                    print("Note: Your torchaudio version may be incompatible with pyannote.audio")
                    print("Try upgrading: pip install --upgrade torchaudio")
                elif 'token' in error_str or 'auth' in error_str:
                    print(f"Error loading speaker diarization model: Authentication failed")
                    print("Make sure your HF_TOKEN is valid and has accepted the model conditions")
                    print("Visit: https://huggingface.co/pyannote/speaker-diarization-community-1")
                else:
                    print(f"Error loading speaker diarization model: {e}")
                print("Note: pyannote requires authentication token.")
                print("Set HF_TOKEN in HF_TOKEN.txt file with your HuggingFace token.")

    def diarize(self, audio_path):
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline(audio_path, min_speakers=1)
            if hasattr(result, 'speaker_diarization'):
                return result.speaker_diarization
            return result
        except Exception as e:
            print(f"Diarization error: {e}")
            return None

    def diarize_full(self, audio_path):
        if self.pipeline is None:
            return None
        try:
            result = self.pipeline(audio_path, min_speakers=1)
            if hasattr(result, 'speaker_diarization'):
                return result
            return result
        except Exception as e:
            print(f"Diarization error: {e}")
            return None

    def format_diarization(self, diarization, transcription_result):
        if diarization is None or transcription_result is None:
            return []
        try:
            segments = transcription_result.get("segments", [])
            if not segments:
                return []
            transcription_segments = []
            for seg in segments:
                words = seg.get("words", [])
                if words:
                    for word in words:
                        transcription_segments.append({
                            "start": word.get("start", seg.get("start", 0)),
                            "end": word.get("end", seg.get("end", 0)),
                            "text": word.get("word", "").strip()
                        })
                else:
                    transcription_segments.append({
                        "start": seg.get("start", 0),
                        "end": seg.get("end", 0),
                        "text": seg.get("text", "").strip()
                    })
            if not transcription_segments:
                return []
            diarization_turns = []
            for turn in diarization.itertracks(yield_label=True):
                segment, track, speaker = turn
                start_time = float(segment.start)
                end_time = float(segment.end)
                diarization_turns.append({
                    "start": start_time,
                    "end": end_time,
                    "speaker": speaker
                })
            diarization_turns.sort(key=lambda x: x["start"])
            result = []
            last_assigned_speaker = None
            for t_seg in transcription_segments:
                best_speaker = None
                best_overlap = 0
                for turn in diarization_turns:
                    overlap_start = max(t_seg["start"], turn["start"])
                    overlap_end = min(t_seg["end"], turn["end"])
                    overlap_duration = max(0, overlap_end - overlap_start)
                    if overlap_duration > 0:
                        if t_seg["start"] >= turn["start"] and t_seg["end"] <= turn["end"]:
                            if overlap_duration + 1 > best_overlap:
                                best_speaker = turn["speaker"]
                                best_overlap = overlap_duration + 1
                        elif overlap_duration > best_overlap:
                            best_speaker = turn["speaker"]
                            best_overlap = overlap_duration
                if best_speaker is not None:
                    last_assigned_speaker = best_speaker
                elif last_assigned_speaker is not None:
                    best_speaker = last_assigned_speaker
                if best_speaker is not None:
                    result.append({
                        "speaker": best_speaker,
                        "start": t_seg["start"],
                        "end": t_seg["end"],
                        "text": t_seg["text"]
                    })
            return result
        except Exception as e:
            print(f"Error formatting diarization: {e}")
            return []

def get_system_resources():
    vram_gb = 0
    single_gpu_gb = 0
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_mem = gpu_props.total_memory / (1024 ** 3)
            vram_gb += gpu_mem
            if gpu_mem > single_gpu_gb:
                single_gpu_gb = gpu_mem

    total_sys_gb = 0
    try:
        import psutil
        total_sys_gb = psutil.virtual_memory().total / (1024 ** 3)
        swap = psutil.swap_memory()
        total_sys_gb += swap.total / (1024 ** 3)
    except:
        try:
            with open('/proc/meminfo', 'r') as f:
                mem_total = 0
                swap_total = 0
                for line in f:
                    if line.startswith('MemTotal:'):
                        mem_total = int(line.split()[1])
                    elif line.startswith('SwapTotal:'):
                        swap_total = int(line.split()[1])
                total_sys_gb = (mem_total + swap_total) / (1024 * 1024)
        except:
            pass
    return single_gpu_gb, total_sys_gb

class VibeVoiceASR:
    def __init__(self, model_dir=None):
        self.model_dir = VIBEVOICE_DIR if model_dir is None else model_dir
        self.processor = None
        self.model = None
        self.device = None
        self._loaded = False

    def _check_resources(self):
        single_gpu_gb, total_sys_gb = get_system_resources()
        if single_gpu_gb >= 24.0:
            self.device = torch.device("cuda:0")
            print(f"Single GPU has {single_gpu_gb:.1f} GB - loading entire model on GPU")
            return True
        if total_sys_gb >= 48.0:
            self.device = torch.device("cpu")
            print(f"CPU mode: {total_sys_gb:.1f} GB RAM+Swap/Pagefile available - loading on CPU")
            return True
        return False

    def ensure_model(self):
        if self._loaded:
            return
        try:
            import sys
            asr_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asr')
            if asr_path not in sys.path:
                sys.path.insert(0, asr_path)

            os.makedirs(self.model_dir, exist_ok=True)

            if not self._check_resources():
                print("Error: VibeVoice ASR requires 24GB+ VRAM or 48GB+ combined GPU+RAM")
                print("Falling back to Whisper + pyannote")
                return

            hf_token = os.environ.get("HF_TOKEN")
            download_kwargs = {}
            if hf_token:
                download_kwargs["token"] = hf_token

            print("Loading VibeVoice ASR model...")

            from asr.vibevoice_asr_processor import VibeVoiceASRProcessor
            from asr.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

            self.processor = VibeVoiceASRProcessor.from_pretrained(
                "microsoft/VibeVoice-ASR",
                language_model_pretrained_name="Qwen/Qwen2.5-7B",
                **download_kwargs
            )

            model_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16 if torch.cuda.is_available() else torch.float32

            self.model = VibeVoiceASRForConditionalGeneration.from_pretrained(
                "microsoft/VibeVoice-ASR",
                dtype=model_dtype,
                attn_implementation="sdpa",
                trust_remote_code=True,
                cache_dir=self.model_dir,
                **download_kwargs
            )
            self.model.eval()
            self.model.to(self.device)
            self._loaded = True
            print("VibeVoice ASR model loaded successfully")
        except Exception as e:
            print(f"Error loading VibeVoice ASR model: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def transcribe(self, audio_path):
        if not self._loaded:
            self.ensure_model()
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            segments = self.processor.post_process_transcription(generated_text)

            result = []
            for seg in segments:
                raw_text = seg.get("text", seg.get("Content", ""))
                clean = re.sub(r'^\[(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]\s*', '', raw_text, flags=re.IGNORECASE).strip()
                if not clean:
                    continue
                result.append({
                    "start": seg.get("start_time", seg.get("Start", seg.get("Start time", 0))),
                    "end": seg.get("end_time", seg.get("End", seg.get("End time", 0))),
                    "speaker": seg.get("speaker_id", seg.get("Speaker", seg.get("Speaker ID", 0))),
                    "text": clean
                })
            return result
        except Exception as e:
            print(f"VibeVoice ASR transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def transcribe_with_events(self, audio_path):
        if not self._loaded:
            self.ensure_model()
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            segments = self.processor.post_process_transcription(generated_text)

            result = []
            event_re = re.compile(r'^\[([^\]]+)\]\s*', re.IGNORECASE)
            event_tags = {'lyric', 'silence', 'music', 'noise', 'applause', 'laughter', 'cough', 'breath'}

            for seg in segments:
                raw_text = seg.get("text", seg.get("Content", ""))
                ev_match = event_re.match(raw_text)
                event_type = None
                is_event = False
                text = raw_text

                if ev_match:
                    tag = ev_match.group(1).lower()
                    remainder = raw_text[ev_match.end():].strip()
                    if tag in event_tags:
                        event_type = tag
                        if not remainder:
                            is_event = True
                            text = ""
                        else:
                            text = remainder

                result.append({
                    "start": seg.get("start_time", seg.get("Start", seg.get("Start time", 0))),
                    "end": seg.get("end_time", seg.get("End", seg.get("End time", 0))),
                    "speaker": seg.get("speaker_id", seg.get("Speaker", seg.get("Speaker ID", 0))),
                    "text": text,
                    "is_event": is_event,
                    "event_type": event_type
                })
            return result
        except Exception as e:
            print(f"VibeVoice ASR transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    @torch.no_grad()
    def transcribe_plain_text(self, audio_path):
        if not self._loaded:
            self.ensure_model()
        if self.model is None or self.processor is None:
            return None
        try:
            inputs = self.processor(
                audio=audio_path,
                return_tensors="pt",
                add_generation_prompt=True,
                context_info="Please transcribe this audio without timestamps or speaker labels."
            )

            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=32768,
                pad_token_id=self.processor.pad_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                do_sample=False,
            )

            input_length = inputs['input_ids'].shape[1]
            generated_ids = output_ids[0, input_length:]
            generated_text = self.processor.decode(generated_ids, skip_special_tokens=True)

            try:
                segments = self.processor.post_process_transcription(generated_text)
                if segments:
                    return " ".join(seg.get("text", "") for seg in segments)
            except:
                pass

            return generated_text.strip()
        except Exception as e:
            print(f"VibeVoice ASR transcription error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

import re

TRANSLATE_GEMMA_SUPPORTED_LANGS = {
    'af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'bs', 'ca', 'cs', 'cy', 'da',
    'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr', 'ga', 'gl', 'gu',
    'ha', 'he', 'hi', 'hr', 'hu', 'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk',
    'km', 'kn', 'ko', 'lo', 'lt', 'lv', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt',
    'my', 'ne', 'nl', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'si', 'sk',
    'sl', 'so', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'tk', 'tl',
    'tr', 'uk', 'ur', 'uz', 'vi', 'yo', 'zh'
}

class TranslateGemma:
    def __init__(self, model_dir=None):
        self.model_dir = TRANSLATE_GEMMA_DIR if model_dir is None else model_dir
        self.pipe = None
        self._loaded = False

    def ensure_model(self):
        if self._loaded and self.pipe is not None:
            return True
        try:
            from transformers import pipeline
            model_id = "google/translategemma-12b-it"
            print("Loading TranslateGemma 12B model...")
            use_cuda = torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / (1024**3) >= 24.0
            dtype = torch.bfloat16 if use_cuda else torch.float32
            model_kwargs = {"cache_dir": self.model_dir, "device_map": "auto" if use_cuda else "cpu"}
            pipe_kwargs = {
                "model": model_id,
                "torch_dtype": dtype,
                "model_kwargs": model_kwargs,
            }
            hf_token = os.environ.get("HF_TOKEN")
            if hf_token:
                pipe_kwargs["token"] = hf_token
            self.pipe = pipeline("image-text-to-text", **pipe_kwargs)
            self._loaded = True
            print(f"TranslateGemma loaded ({'GPU bfloat16' if use_cuda else 'CPU float32'})")
            return True
        except Exception as e:
            print(f"Error loading TranslateGemma: {e}")
            self.cleanup()
            return False

    def translate(self, text, source_lang, target_lang):
        if not self._loaded or self.pipe is None:
            return None
        if source_lang != 'auto':
            detected = _detect_lang_from_text(text)
            if detected != source_lang:
                print(f"Wrong source language input, auto-detected {detected}, overriding")
                source_lang = detected
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "source_lang_code": source_lang,
                            "target_lang_code": target_lang,
                            "text": text,
                        }
                    ],
                }
            ]
            output = self.pipe(text=messages, max_new_tokens=2048, generate_kwargs={"do_sample": False})
            translated = output[0]["generated_text"][-1]["content"]
            return translated.strip() if translated else None
        except Exception as e:
            print(f"TranslateGemma translation error: {e}")
            return None

    def translate_segments(self, segments, source_lang, target_lang):
        if source_lang != 'auto':
            all_text = ' '.join((seg if isinstance(seg, str) else seg.get('text', '')) for seg in segments)
            detected = _detect_lang_from_text(all_text)
            if detected != source_lang:
                print(f"Wrong source language input, auto-detected {detected}, overriding")
                source_lang = detected
        results = []
        for seg in segments:
            text = seg if isinstance(seg, str) else seg.get('text', '')
            translated = self.translate(text, source_lang, target_lang)
            results.append(translated if translated else text)
        return results

    def cleanup(self):
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class AudioSREnhancer:
    def __init__(self, model_dir=None):
        self.model_dir = AUDIOSR_DIR if model_dir is None else model_dir
        self.model = None
        self.model_name = "basic"
        self._loaded = False

    def ensure_model(self, model_name="basic"):
        if self._loaded and self.model is not None:
            if self.model_name == model_name:
                return True
            print(f"Switching AudioSR from {self.model_name} to {model_name}...")
            self.cleanup()
        try:
            os.makedirs(self.model_dir, exist_ok=True)
            self.model_name = model_name
            print(f"Loading AudioSR {model_name} model...")
            from audiosr.pipeline import build_model
            self.model = build_model(model_name=model_name, cache_dir=self.model_dir)
            self._loaded = True
            print(f"AudioSR {model_name} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading AudioSR model: {e}")
            self.cleanup()
            return False

    def enhance(self, input_path, output_path, ddim_steps=50, guidance_scale=3.5, seed=42):
        if not self._loaded or self.model is None:
            return False
        try:
            duration = _get_audio_duration(input_path)
            if duration <= 10.24:
                from audiosr.pipeline import super_resolution
                waveform = super_resolution(
                    self.model, input_path,
                    seed=seed, ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                )
                if isinstance(waveform, np.ndarray):
                    sf.write(output_path, waveform[0].T, samplerate=48000)
                elif isinstance(waveform, torch.Tensor):
                    wav_np = waveform.squeeze().cpu().numpy()
                    if wav_np.ndim == 1:
                        sf.write(output_path, wav_np, samplerate=48000)
                    else:
                        sf.write(output_path, wav_np.T, samplerate=48000)
                else:
                    return False
            else:
                from audiosr.pipeline import super_resolution_long_audio
                waveform = super_resolution_long_audio(
                    self.model, input_path,
                    seed=seed, ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                )
                if isinstance(waveform, torch.Tensor):
                    wav_np = waveform.squeeze().cpu().numpy()
                    if wav_np.ndim == 1:
                        sf.write(output_path, wav_np, samplerate=48000)
                    else:
                        sf.write(output_path, wav_np.T, samplerate=48000)
                elif isinstance(waveform, np.ndarray):
                    sf.write(output_path, waveform[0].T, samplerate=48000)
                else:
                    return False
            return os.path.exists(output_path)
        except Exception as e:
            print(f"AudioSR enhancement error: {e}")
            return False

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_lang_spec(spec_str):
    m = re.match(r'^\(([a-zA-Z]{2,})-([a-zA-Z]{2,})\)$', spec_str)
    if m:
        return {'source': m.group(1).lower(), 'target': m.group(2).lower()}
    m = re.match(r'^\(([a-zA-Z]{2,})\)$', spec_str)
    if m:
        lang = m.group(1).lower()
        if lang == 'auto':
            return None
        return {'source': 'auto', 'target': lang}
    return None


def _detect_lang_from_text(text):
    if not text or not text.strip():
        return 'en'
    try:
        from langdetect import detect
        detected = detect(text.strip()[:2000])
        return detected[:2].lower() if detected else 'en'
    except Exception:
        return 'en'


def _translate_with_gemma(text, source_lang, target_lang):
    translator = TranslateGemma()
    if not translator.ensure_model():
        print("Error: Failed to load TranslateGemma for translation")
        return None
    result = translator.translate(text, source_lang, target_lang)
    translator.cleanup()
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _translate_segments_with_gemma(segments, source_lang, target_lang):
    translator = TranslateGemma()
    if not translator.ensure_model():
        print("Error: Failed to load TranslateGemma for translation")
        return None
    results = translator.translate_segments(segments, source_lang, target_lang)
    translator.cleanup()
    del translator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def _mix_audio_at_target_sr(vocals_path, music_path, output_path, target_sr=48000, original_audio_path=None):
    try:
        if not os.path.exists(vocals_path):
            print(f"Error: Vocals file not found: {vocals_path}")
            return False
        if not os.path.exists(music_path):
            print(f"Error: Music file not found: {music_path}")
            return False
        voc_wav, voc_sr = torchaudio.load(vocals_path)
        mus_wav, mus_sr = torchaudio.load(music_path)
        if voc_sr != target_sr:
            voc_wav = torchaudio.functional.resample(voc_wav, orig_freq=voc_sr, new_freq=target_sr)
        if mus_sr != target_sr:
            mus_wav = torchaudio.functional.resample(mus_wav, orig_freq=mus_sr, new_freq=target_sr)
        if voc_wav.shape[0] > 1:
            voc_wav = torch.mean(voc_wav, dim=0, keepdim=True)
        if mus_wav.shape[0] > 1:
            mus_wav = torch.mean(mus_wav, dim=0, keepdim=True)
        max_len = max(voc_wav.shape[-1], mus_wav.shape[-1])
        if voc_wav.shape[-1] < max_len:
            voc_wav = torch.nn.functional.pad(voc_wav, (0, max_len - voc_wav.shape[-1]))
        if mus_wav.shape[-1] < max_len:
            mus_wav = torch.nn.functional.pad(mus_wav, (0, max_len - mus_wav.shape[-1]))
        mixed = voc_wav + mus_wav
        peak = torch.max(torch.abs(mixed))
        if peak > 1.0:
            mixed = mixed * (0.95 / peak)
        sf.write(output_path, mixed.squeeze().numpy(), samplerate=target_sr)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Audio mixing error: {e}")
        return False

def _adjust_audio_speed(input_path, target_duration, output_path):
    try:
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                     '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if probe_result.returncode != 0:
            return False
        current_duration = float(probe_result.stdout.strip())
        if current_duration <= 0 or target_duration <= 0:
            return False
        speed_factor = current_duration / target_duration
        if speed_factor < 0.25 or speed_factor > 4.0:
            return False
        atempo_filters = []
        remaining = speed_factor
        while remaining < 0.5 or remaining > 2.0:
            if remaining < 0.5:
                chunk = 0.5
                remaining = remaining / chunk
                atempo_filters.append(f"atempo={chunk}")
            elif remaining > 2.0:
                chunk = 2.0
                remaining = remaining / chunk
                atempo_filters.append(f"atempo={chunk}")
        atempo_filters.append(f"atempo={remaining:.4f}")
        filter_str = ",".join(atempo_filters)
        cmd = ['ffmpeg', '-i', input_path, '-filter:a', filter_str, '-y', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False


def _get_audio_duration(path):
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        try:
            info = torchaudio.info(path)
            return info.num_frames / info.sample_rate
        except Exception:
            return 30

def _overlay_segment_on_base(base_path, segment_path, start_time, output_path):
    try:
        cmd = [
            'ffmpeg', '-i', base_path, '-i', segment_path,
            '-filter_complex',
            f'[1:a]adelay={int(start_time * 1000)}|{int(start_time * 1000)}[delayed];'
            f'[0:a][delayed]amix=inputs=2:duration=first:dropout_transition=0,volume=2.0[out]',
            '-map', '[out]', '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False

def _measure_rms(audio_path, start=None, end=None):
    try:
        data, sr = sf.read(audio_path, dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if start is not None and end is not None:
            s = int(start * sr)
            e = int(end * sr)
            data = data[s:e]
        if len(data) == 0:
            return 0.0
        return float(np.sqrt(np.mean(data ** 2)))
    except Exception:
        return 0.0

def _build_speaker_timeline_audio(spk_parts, total_duration, output_path, target_sr=44100):
    try:
        import numpy as np
        total_samples = int(total_duration * target_sr) + target_sr
        output = np.zeros(total_samples, dtype=np.float64)
        for p in sorted(spk_parts, key=lambda x: x['start']):
            seg_path = p.get('path')
            if not seg_path or not os.path.exists(seg_path):
                continue
            try:
                data, seg_sr = sf.read(seg_path, dtype='float32')
            except Exception:
                try:
                    wav, seg_sr = torchaudio.load(seg_path)
                    data = wav.squeeze().float().numpy()
                except Exception:
                    continue
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if seg_sr != target_sr:
                try:
                    resample_dir = tempfile.mkdtemp()
                    resampled = os.path.join(resample_dir, 'resampled.wav')
                    cmd = ['ffmpeg', '-i', seg_path, '-ar', str(target_sr), '-ac', '1', '-y', resampled]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    data, seg_sr = sf.read(resampled, dtype='float32')
                    shutil.rmtree(resample_dir, ignore_errors=True)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                except Exception:
                    continue
            start_sample = int(p['start'] * target_sr)
            end_sample = start_sample + len(data)
            if end_sample > total_samples:
                new_total = end_sample + target_sr
                output = np.resize(output, new_total)
                total_samples = new_total
            actual_len = min(len(data), total_samples - start_sample)
            if actual_len > 0:
                output[start_sample:start_sample + actual_len] += data[:actual_len]
        sf.write(output_path, output.astype(np.float32), target_sr)
        return os.path.exists(output_path)
    except Exception:
        return False

def _assemble_dubbed_audio(parts, total_duration, output_path, target_sr=44100, original_audio_path=None):
    try:
        import numpy as np
        total_samples = int(total_duration * target_sr) + target_sr
        output = np.zeros(total_samples, dtype=np.float64)
        for part in sorted(parts, key=lambda x: x['start']):
            seg_path = part['path']
            if not seg_path or not os.path.exists(seg_path):
                continue
            try:
                data, seg_sr = sf.read(seg_path, dtype='float32')
            except Exception:
                try:
                    wav, seg_sr = torchaudio.load(seg_path)
                    data = wav.squeeze().float().numpy()
                except Exception:
                    continue
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if seg_sr != target_sr:
                try:
                    resample_dir = tempfile.mkdtemp()
                    resampled = os.path.join(resample_dir, 'resampled.wav')
                    cmd = ['ffmpeg', '-i', seg_path, '-ar', str(target_sr), '-ac', '1', '-y', resampled]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    data, seg_sr = sf.read(resampled, dtype='float32')
                    shutil.rmtree(resample_dir, ignore_errors=True)
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                except Exception:
                    continue
            if original_audio_path and os.path.exists(original_audio_path):
                seg_start = part.get('start', 0)
                seg_end = part.get('end', 0)
                orig_rms = _measure_rms(original_audio_path, seg_start, seg_end)
                tts_rms = float(np.sqrt(np.mean(data ** 2))) if len(data) > 0 else 0.0
                if tts_rms > 1e-6 and orig_rms > 1e-6:
                    scale = orig_rms / tts_rms
                    scale = min(scale, 5.0)
                    data = data * scale
            start_sample = int(part['start'] * target_sr)
            end_sample = start_sample + len(data)
            if end_sample > total_samples:
                new_total = end_sample + target_sr
                output = np.resize(output, new_total)
                total_samples = new_total
            actual_len = min(len(data), total_samples - start_sample)
            if actual_len > 0:
                output[start_sample:start_sample + actual_len] += data[:actual_len]
        peak = np.max(np.abs(output))
        if peak > 1.0:
            output = output * (0.95 / peak)
        sf.write(output_path, output.astype(np.float32), target_sr)
        return os.path.exists(output_path)
    except Exception:
        return False

def _extract_audio_segment(input_path, start, end, output_path):
    try:
        duration = end - start
        if duration <= 0:
            return False
        cmd = [
            'ffmpeg', '-i', input_path, '-ss', str(start), '-t', str(duration),
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(output_path)
    except Exception:
        return False

def _validate_stems_for_svtype(stem_names, sv_type):
    if not stem_names:
        return stem_names
    if sv_type == 'voice':
        valid = ACESTEP_VOICE_TRACKS
        invalid = [s for s in stem_names if s not in valid]
        if invalid:
            print(f"Error: Instrument stems ({', '.join(invalid)}) are not available with voice-only reference. Voice references only support: {', '.join(sorted(valid))}")
            return None
    elif sv_type == 'music':
        valid = ACESTEP_INSTRUMENT_TRACKS
        invalid = [s for s in stem_names if s not in valid]
        if invalid:
            print(f"Error: Vocal stems ({', '.join(invalid)}) are not available with music-only reference. Music references only support: {', '.join(sorted(valid))}")
            return None
    if 'everything' in stem_names:
        print("Error: 'everything' stem keyword is not valid in references (as-is mode already provides the full audio). Specify individual stems instead.")
        return None
    unrecognized = [s for s in stem_names if s not in VALID_ACESTEP_TRACKS]
    if unrecognized:
        recognized = [s for s in stem_names if s in VALID_ACESTEP_TRACKS]
        if unrecognized:
            display = unrecognized[:5]
            extra = len(unrecognized) - 5
            if extra > 0:
                print(f"Warning: Unrecognized keywords: {', '.join(display)} + {extra} others")
            else:
                print(f"Warning: Unrecognized keywords: {', '.join(display)}")
            print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
        if not recognized:
            return None
        stem_names = recognized
    return stem_names

def _parse_ref_time_spec(spec):
    if '(' not in spec or not spec.endswith(')'):
        return None, spec, None
    paren_start = spec.find('(')
    path = spec[paren_start + 1:-1].strip()
    if not path:
        return None, spec, None
    time_part = spec[:paren_start].strip()
    stem_names = None
    if '/' in time_part:
        first_slash = time_part.find('/')
        potential_stems = time_part[:first_slash]
        remaining_after_slash = time_part[first_slash + 1:]
        stem_candidates = [s.strip().lower() for s in potential_stems.split('-')]
        if stem_candidates:
            recognized = [s for s in stem_candidates if s in VALID_ACESTEP_TRACKS]
            unrecognized = [s for s in stem_candidates if s not in VALID_ACESTEP_TRACKS and s != 'everything']
            has_everything = 'everything' in stem_candidates
            if has_everything:
                print("Error: 'everything' stem keyword is not valid in references (as-is mode already provides the full audio). Specify individual stems instead.")
                stem_names = None
            elif unrecognized:
                if unrecognized:
                    display = unrecognized[:5]
                    extra = len(unrecognized) - 5
                    if extra > 0:
                        print(f"Warning: Unrecognized keywords: {', '.join(display)} + {extra} others")
                    else:
                        print(f"Warning: Unrecognized keywords: {', '.join(display)}")
                    print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
                if recognized:
                    stem_names = recognized
                else:
                    stem_names = None
            elif recognized:
                stem_names = recognized
            else:
                stem_names = None
            if stem_names is not None:
                time_part = remaining_after_slash
    stem_prefix = None
    if ':' in time_part:
        colon_idx = time_part.find(':')
        after_colon = time_part[colon_idx + 1:]
        if after_colon and after_colon[0].isdigit():
            stem_prefix = time_part[:colon_idx]
            time_part = after_colon
        elif after_colon and after_colon[0] == '/':
            return None, spec, None
        else:
            before_colon = time_part[:colon_idx]
            if before_colon:
                stem_prefix = before_colon
                time_part = after_colon
    if not time_part:
        if stem_prefix:
            return None, f"{stem_prefix}:{path}", stem_names
        return None, path, stem_names
    segments = time_part.split('/')
    ranges = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if '-' in seg:
            parts = seg.split('-', 1)
            try:
                start = float(parts[0])
                end = float(parts[1])
                if start < 0 or end < 0 or start >= end:
                    return None, spec, None
                ranges.append((start, end))
            except (ValueError, IndexError):
                return None, spec, None
        else:
            try:
                start = float(seg)
                if start < 0:
                    return None, spec, None
                ranges.append((start, None))
            except ValueError:
                return None, spec, None
    if not ranges:
        return None, spec, None
    if stem_prefix:
        path = f"{stem_prefix}:{path}"
    return ranges, path, stem_names

def _extract_ref_segments(audio_path, time_ranges, slot_max, cleanup_list):
    duration = _get_audio_duration(audio_path)
    if duration <= 0:
        return audio_path
    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 2:
        wav = wav[:2, :]
    elif wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    total_samples = wav.shape[-1]
    if len(time_ranges) == 1:
        start, end = time_ranges[0]
        if end is None:
            end = start + slot_max
        if start > duration:
            print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), adjusting")
            start = max(0, duration - slot_max)
        if end > duration:
            if time_ranges[0][1] is not None:
                print(f"Warning: Time spec end ({time_ranges[0][1]}s) exceeds audio duration ({duration:.1f}s), clamping")
            end = duration
        if end - start < slot_max:
            needed = slot_max - (end - start)
            start = max(0, start - needed)
            if end - start < slot_max:
                end = min(duration, start + slot_max)
        start_sample = int(max(0, start) * sr)
        end_sample = int(min(end, duration) * sr)
        start_sample = max(0, min(start_sample, total_samples))
        end_sample = max(start_sample, min(end_sample, total_samples))
        combined = wav[:, start_sample:end_sample]
    else:
        adjusted = []
        for start, end in time_ranges:
            if end is None:
                if start > duration:
                    print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), adjusting")
                    start = max(0, duration - slot_max)
                end = start + slot_max
            else:
                if start > duration:
                    print(f"Warning: Time spec start ({start}s) exceeds audio duration ({duration:.1f}s), skipping segment")
                    continue
                if end > duration:
                    print(f"Warning: Time spec end ({end}s) exceeds audio duration ({duration:.1f}s), clamping")
                    end = duration
            if start < end:
                adjusted.append((start, end))
        if not adjusted:
            return audio_path
        combined_dur = sum(e - s for s, e in adjusted)
        if combined_dur < slot_max:
            scale = slot_max / combined_dur
            slid = []
            for s, e in adjusted:
                seg_dur = e - s
                new_dur = seg_dur * scale
                mid = (s + e) / 2.0
                ns = mid - new_dur / 2.0
                ne = mid + new_dur / 2.0
                ns = max(0, ne - new_dur)
                ne = min(duration, ns + new_dur)
                ns = max(0, ne - new_dur)
                slid.append((ns, ne))
            adjusted = slid
        extracted = []
        for start, end in adjusted:
            start_sample = int(max(0, start) * sr)
            end_sample = int(min(end, duration) * sr)
            start_sample = max(0, min(start_sample, total_samples))
            end_sample = max(start_sample, min(end_sample, total_samples))
            if end_sample > start_sample:
                extracted.append(wav[:, start_sample:end_sample])
        if not extracted:
            return audio_path
        combined = torch.cat(extracted, dim=-1)
    target_samples = int(slot_max * sr)
    if combined.shape[-1] < target_samples:
        reps = math.ceil(target_samples / combined.shape[-1])
        combined = combined.repeat(1, reps)
        combined = combined[:, :target_samples]
    out_path = os.path.join(tempfile.gettempdir(), f"voder_ref_seg_{int(time.time())}_{len(cleanup_list)}.wav")
    torchaudio.save(out_path, combined, sr)
    cleanup_list.append(out_path)
    return out_path

def _extract_acestep_stems(audio_path, stem_names, cleanup_list, ace_step=None):
    own_model = ace_step is None
    if own_model:
        ace_step = AceStepWrapper(use_overdose=False, complete_mode=True)
        if ace_step.handler is None:
            print("Warning: Failed to load ACE-Step model for stem extraction, using original audio")
            del ace_step
            gc.collect()
            return audio_path
    try:
        _ts = time.strftime("%Y%m%d_%H%M%S")
        generated_files = []
        for stem in stem_names:
            print(f"  Extracting stem: {stem}...")
            temp_output = os.path.join(tempfile.gettempdir(), f"voder_stem_{stem}_{_ts}.wav")
            success = ace_step.extract(
                src_audio=audio_path,
                track_name=stem,
                output_path=temp_output
            )
            if success:
                generated_files.append(temp_output)
            else:
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass
        if not generated_files:
            return audio_path
        if len(generated_files) == 1:
            cleanup_list.append(generated_files[0])
            return generated_files[0]
        mix_output = os.path.join(tempfile.gettempdir(), f"voder_stem_mix_{_ts}.wav")
        input_list = " ".join(f'-i "{f}"' for f in generated_files)
        ret = os.system(f'ffmpeg -y {input_list} -filter_complex amix=inputs={len(generated_files)}:duration=longest "{mix_output}" 2>/dev/null')
        for f in generated_files:
            if os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        if ret != 0 or not os.path.exists(mix_output):
            return audio_path
        cleanup_list.append(mix_output)
        return mix_output
    finally:
        if own_model:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del ace_step
            gc.collect()

def _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup_list, time_ranges=None, slot_max=30, stems=None, ace_step=None):
    resolved = None
    if not os.path.exists(raw_path) and not is_youtube_url(raw_path):
        print(f"Warning: Audio path not found: {raw_path}, skipping")
        return None
    if is_youtube_url(raw_path):
        print(f"Downloading audio from URL: {raw_path}")
        res, cl = resolve_target_to_audio(raw_path)
        if res is None:
            print("Warning: Could not download audio, skipping")
            return None
        cleanup_list.extend(cl)
        resolved = res
    else:
        r_ext = os.path.splitext(raw_path)[1].lower()
        if r_ext in VIDEO_EXTENSIONS:
            tmp = os.path.join(results_dir, f'_vid_{timestamp}_{len(cleanup_list)}.wav')
            ret = os.system(f'ffmpeg -y -i "{raw_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{tmp}" 2>/dev/null')
            if ret != 0 or not os.path.exists(tmp):
                print("Warning: Failed to extract audio from video, skipping")
                return None
            cleanup_list.append(tmp)
            resolved = tmp
        else:
            valid, msg = validate_audio_file(raw_path)
            if not valid:
                print(f"Warning: Invalid audio file: {msg}, skipping")
                return None
            resolved = raw_path
    if sv_type == 'voice':
        print("Extracting vocals via SVS...")
        processed = svs_extract_vocals(resolved)
    elif sv_type == 'music':
        print("Extracting music (removing vocals) via SVS...")
        processed = svs_extract_music(resolved)
    else:
        processed = resolved
    if processed and processed != resolved and processed not in cleanup_list:
        cleanup_list.append(processed)
    if stems:
        validated = _validate_stems_for_svtype(stems, sv_type)
        if validated:
            print(f"Extracting stem(s): {', '.join(validated)}...")
            processed = _extract_acestep_stems(processed, validated, cleanup_list, ace_step=ace_step)
        else:
            print("Warning: Stem extraction skipped due to validation errors")
    if time_ranges:
        processed = _extract_ref_segments(processed, time_ranges, slot_max, cleanup_list)
    return processed

def _compose_refs(ref_entries, results_dir):
    if not ref_entries:
        return None, []
    cleanup = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_entries = len(ref_entries)
    slot_max = 30 // max(1, num_entries)
    has_time_spec = any(len(e) > 2 and e[2] is not None for e in ref_entries)
    any_stems = any(len(e) > 3 and e[3] is not None for e in ref_entries)
    ace_step_shared = None
    if any_stems:
        print("Loading ACE-Step XL-Base model (stem extraction)...")
        ace_step_shared = AceStepWrapper(use_overdose=False, complete_mode=True)
        if ace_step_shared.handler is None:
            print("Warning: Failed to load ACE-Step model for stem extraction")
            del ace_step_shared
            ace_step_shared = None
            gc.collect()
    processed = []
    for entry in ref_entries:
        sv_type = entry[0]
        raw_path = entry[1]
        tr = entry[2] if len(entry) > 2 else None
        entry_stems = entry[3] if len(entry) > 3 else None
        if has_time_spec and tr is None:
            tr = [(0, None)]
        if entry_stems:
            audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup, time_ranges=tr, slot_max=slot_max, stems=entry_stems, ace_step=ace_step_shared)
        else:
            audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup, time_ranges=tr, slot_max=slot_max)
        if audio_path is None:
            continue
        processed.append(audio_path)
    if not processed:
        if ace_step_shared is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            del ace_step_shared
            gc.collect()
        return None, cleanup
    if ace_step_shared is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step_shared
        gc.collect()
    if len(processed) == 1:
        return processed[0], cleanup
    if has_time_spec:
        tensors = []
        for p in processed:
            wav, sr = torchaudio.load(p)
            if sr != 48000:
                wav = torchaudio.transforms.Resample(sr, 48000)(wav)
            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            elif wav.shape[0] > 2:
                wav = wav[:2, :]
            tensors.append(wav)
        composed = torch.cat(tensors, dim=-1)
        out_path = os.path.join(results_dir, f'_composed_ref_{timestamp}.wav')
        torchaudio.save(out_path, composed, 48000)
        cleanup.append(out_path)
        return out_path, cleanup
    print(f"Composing {len(processed)} references into 30s composite...")
    tensors = []
    for p in processed:
        wav, sr = torchaudio.load(p)
        if sr != 48000:
            wav = torchaudio.transforms.Resample(sr, 48000)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        tensors.append(wav)
    sr = 48000
    seg10 = 10 * sr
    seg5 = 5 * sr
    composed = None
    if len(tensors) == 2:
        t1, t2 = tensors[0], tensors[1]
        for idx, t in enumerate([t1, t2]):
            if t.shape[-1] < seg10:
                reps = math.ceil(seg10 / t.shape[-1])
                if idx == 0:
                    t1 = t.repeat(1, reps)
                else:
                    t2 = t.repeat(1, reps)
        third1 = t1.shape[-1] // 3
        third2 = t2.shape[-1] // 3
        off1m = random.randint(0, max(0, third1 - seg5))
        off2m = random.randint(0, max(0, third2 - seg5))
        front = t1[:, :seg10]
        mid1 = t1[:, third1 + off1m:third1 + off1m + seg5]
        mid2 = t2[:, third2 + off2m:third2 + off2m + seg5]
        end2_start = max(0, t2.shape[-1] - seg10)
        end2 = t2[:, end2_start:end2_start + seg10]
        composed = torch.cat([front, mid1, mid2, end2], dim=-1)
    else:
        for idx, t in enumerate(tensors):
            if t.shape[-1] < seg10:
                reps = math.ceil(seg10 / t.shape[-1])
                tensors[idx] = t.repeat(1, reps)
        t1, t2, t3 = tensors[0], tensors[1], tensors[2]
        third2 = t2.shape[-1] // 3
        off2 = random.randint(0, max(0, third2 - seg10))
        front = t1[:, :seg10]
        mid = t2[:, third2 + off2:third2 + off2 + seg10]
        end3_start = max(0, t3.shape[-1] - seg10)
        end = t3[:, end3_start:end3_start + seg10]
        composed = torch.cat([front, mid, end], dim=-1)
    out_path = os.path.join(results_dir, f'_composed_ref_{timestamp}.wav')
    torchaudio.save(out_path, composed, sr)
    cleanup.append(out_path)
    return out_path, cleanup

def _compose_sources(source_entries, results_dir):
    if not source_entries:
        return None, []
    cleanup = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    any_stems = any(len(e) > 3 and e[3] is not None for e in source_entries)
    ace_step_shared = None
    if any_stems:
        print("Loading ACE-Step XL-Base model (stem extraction)...")
        ace_step_shared = AceStepWrapper(use_overdose=False, complete_mode=True)
        if ace_step_shared.handler is None:
            print("Warning: Failed to load ACE-Step model for stem extraction")
            del ace_step_shared
            ace_step_shared = None
            gc.collect()
    processed = []
    for entry in source_entries:
        sv_type = entry[0]
        raw_path = entry[1]
        entry_tr = entry[2] if len(entry) > 2 else None
        entry_stems = entry[3] if len(entry) > 3 else None
        if entry_stems:
            audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup, time_ranges=entry_tr, stems=entry_stems, ace_step=ace_step_shared)
        else:
            audio_path = _resolve_audio_entry(sv_type, raw_path, results_dir, timestamp, cleanup, time_ranges=entry_tr)
        if audio_path is None:
            continue
        processed.append(audio_path)
    if ace_step_shared is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step_shared
        gc.collect()
    if not processed:
        return None, cleanup
    print(f"Composing {len(processed)} sources into composite...")
    tensors = []
    durations = []
    for p in processed:
        wav, sr = torchaudio.load(p)
        if sr != 48000:
            wav = torchaudio.transforms.Resample(sr, 48000)(wav)
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        elif wav.shape[0] > 2:
            wav = wav[:2, :]
        tensors.append(wav)
        durations.append(wav.shape[-1] / 48000.0)
    total_dur = sum(durations)
    per_source = total_dur / len(tensors)
    per_source_frames = int(per_source * 48000)
    segments = []
    for t in tensors:
        if t.shape[-1] < per_source_frames:
            reps = math.ceil(per_source_frames / t.shape[-1])
            t = t.repeat(1, reps)
        seg = t[:, :per_source_frames]
        segments.append(seg)
    composed = torch.cat(segments, dim=-1)
    out_path = os.path.join(results_dir, f'_composed_src_{timestamp}.wav')
    torchaudio.save(out_path, composed, 48000)
    cleanup.append(out_path)
    return out_path, cleanup

def _parse_multi_refs(text):
    import re
    matches = re.findall(r'\(([^)]+)\)', text)
    if not matches:
        return None
    return [m.strip() for m in matches if m.strip()]

def _concat_audio_files(file_list, output_path):
    if len(file_list) == 1:
        shutil.copy(file_list[0], output_path)
        return True
    inputs = ' '.join(f'-i "{f}"' for f in file_list)
    filter_parts = ''.join(f'[{i}:a]' for i in range(len(file_list)))
    filter_str = f'{filter_parts}concat=n={len(file_list)}:v=0:a=1[out]'
    cmd = f'ffmpeg -y {inputs} -filter_complex "{filter_str}" -map "[out]" "{output_path}" 2>/dev/null'
    ret = os.system(cmd)
    return ret == 0 and os.path.exists(output_path)

def _extract_target_speaker_from_audio(source_path, target_voice_path, cleanup_list):
    try:
        from unise import UniSEEnhancer
        tse = UniSEEnhancer(UNISE_DIR)
        tse.ensure_model()
        if tse.model is None:
            print("Warning: Could not load TSE model for 'first' pipe, using original audio")
            tse.cleanup()
            del tse
            return source_path
        temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_out.close()
        ok = tse.tse_extract(source_path, target_voice_path, temp_out.name)
        tse.cleanup()
        del tse
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if ok and os.path.exists(temp_out.name):
            cleanup_list.append(temp_out.name)
            return temp_out.name
        return source_path
    except Exception as _e:
        print(f"Warning: Target speaker extraction failed: {_e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return source_path

def _resolve_multi_refs(ref_paths, cleanup_list, use_first=False):
    clean_vocals = []
    for ref_path in ref_paths:
        resolved_audio, _cl = resolve_target_to_audio(ref_path.strip())
        if not resolved_audio:
            return None
        cleanup_list.extend(_cl)
        cv = svs_extract_vocals(resolved_audio)
        if cv and cv != resolved_audio:
            cleanup_list.append(cv)
        else:
            cv = resolved_audio
        if resolved_audio not in cleanup_list:
            cleanup_list.append(resolved_audio)
        clean_vocals.append(cv)
    if len(clean_vocals) > 1 and use_first:
        print("Applying 'first' pipe: extracting target speaker from additional references...")
        target_voice = clean_vocals[0]
        for idx in range(1, len(clean_vocals)):
            print(f"  Extracting target speaker from reference {idx + 1}...")
            extracted = _extract_target_speaker_from_audio(clean_vocals[idx], target_voice, cleanup_list)
            clean_vocals[idx] = extracted
    if len(clean_vocals) == 1:
        return clean_vocals[0]
    concat_path = os.path.join(tempfile.gettempdir(), f"voder_multi_ref_{int(time.time())}.wav")
    if _concat_audio_files(clean_vocals, concat_path):
        cleanup_list.append(concat_path)
        return concat_path
    return clean_vocals[0]

def _parse_repaint_pass_spec(spec):
    parts = []
    current = ''
    depth = 0
    for ch in spec:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == '/' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    if current:
        parts.append(current)
    if not parts:
        return None, 'empty pass spec'
    time_part = parts[0]
    time_parts = time_part.split('-')
    if len(time_parts) != 2:
        return None, f'invalid time range format: {time_part}'
    try:
        start_sec = float(time_parts[0].strip())
        end_sec = float(time_parts[1].strip())
    except ValueError:
        return None, f'time values must be numbers: {time_part}'
    if start_sec < 0:
        return None, 'start time cannot be negative'
    if end_sec <= 0:
        return None, 'end time must be greater than 0'
    if start_sec >= end_sec:
        return None, 'start time must be less than end time'
    lyrics = None
    styling = None
    references = []
    bias = None
    j = 1
    while j < len(parts):
        part = parts[j]
        if part.startswith('lyrics(') and part.endswith(')'):
            lyrics = part[7:-1].replace('\\n', '\n')
        elif part.startswith('styling(') and part.endswith(')'):
            styling = part[8:-1].replace('\\n', '\n')
        elif part.startswith('reference-voice(') and part.endswith(')'):
            inner = part[16:-1]
            tr, rp, ref_stems = _parse_ref_time_spec(inner)
            references.append(('voice', rp, tr, ref_stems))
        elif part.startswith('reference-music(') and part.endswith(')'):
            inner = part[15:-1]
            tr, rp, ref_stems = _parse_ref_time_spec(inner)
            references.append(('music', rp, tr, ref_stems))
        elif part.startswith('reference(') and part.endswith(')'):
            inner = part[9:-1]
            tr, rp, ref_stems = _parse_ref_time_spec(inner)
            references.append(('asis', rp, tr, ref_stems))
        elif part == 'bias' and j + 1 < len(parts):
            bias = parts[j + 1]
            j += 1
        j += 1
    if references and len(references) > 3:
        print(f"Warning: repaint pass supports up to 3 references, using first 3")
        references = references[:3]
    return {
        'start': start_sec,
        'end': end_sec,
        'lyrics': lyrics,
        'styling': styling,
        'references': references,
        'bias': bias
    }, None

def _parse_script_directives(text):
    tokens = text.split()
    directives = {}
    content_end = len(tokens)
    for i in range(len(tokens) - 1, -1, -1):
        token = tokens[i]
        if token.startswith('/time:'):
            directives['time_raw'] = token[6:]
            content_end = i
        elif token.startswith('/level:'):
            directives['level_raw'] = token[7:]
            content_end = i
        elif token.startswith('/duration:'):
            directives['duration_raw'] = token[10:]
            content_end = i
        else:
            break
    clean_text = ' '.join(tokens[:content_end])
    return clean_text.strip(), directives

def _validate_time_directive(time_str):
    time_str = time_str.strip()
    if not time_str:
        return 0, 0, 0, None
    if not re.match(r'^[+-]?\d+([+-]\d+)*$', time_str):
        return 0, 0, 0, "Invalid time format"
    tokens = re.findall(r'[+-]?\d+', time_str)
    start_pad = 0
    cut_start = 0
    cut_end = 0
    position_set = False
    for token in tokens:
        if token.startswith('+'):
            cut_start += int(token[1:])
        elif token.startswith('-'):
            cut_end += int(token[1:])
        else:
            if not position_set:
                start_pad = int(token)
                position_set = True
            else:
                cut_end += int(token)
    return start_pad, cut_start, cut_end, None

def _validate_level_directive(level_str):
    level_str = level_str.strip()
    if not re.match(r'^\d+$', level_str):
        return None, "Invalid level: must be a number"
    val = int(level_str)
    if val < 0 or val > 100:
        return None, "Invalid level: must be 0-100"
    return val, None

def _validate_duration_directive(dur_str):
    dur_str = dur_str.strip()
    if not re.match(r'^\d+$', dur_str):
        return None, "Invalid duration: must be a number"
    val = int(dur_str)
    if val < 1 or val > 30:
        return None, "Invalid duration: must be 1-30"
    return val, None

def _parse_sfx_specs(sfx_args, max_duration):
    parsed = []
    for raw in sfx_args:
        raw = raw.strip()
        if not raw:
            continue
        if not raw.startswith('sfx:'):
            return None, f"Invalid SFX spec (must start with sfx:): {raw}"
        body = raw[4:]
        if not body:
            return None, "SFX spec requires a prompt after sfx:"
        parts = body.split('/')
        prompt = parts[0].strip()
        if not prompt:
            return None, "SFX prompt cannot be empty"

        sfx_dur = 5
        sfx_pos = None
        sfx_level = 50

        if len(parts) >= 2:
            dp = parts[1].strip()
            if not dp:
                return None, f"SFX duration-position is empty in: {raw}"
            dp_parts = dp.split('-')
            if len(dp_parts) != 2:
                return None, f"SFX duration-position must be duration-position (e.g. 10-5), got: {dp}"
            dur_str = dp_parts[0].strip()
            pos_str = dp_parts[1].strip()
            if not dur_str or not pos_str:
                return None, f"SFX duration and position must both be numbers, got: {dp}"
            dur_str = dur_str.lstrip('-')
            if not dur_str or not dur_str.isdigit():
                return None, f"SFX duration must be a number, got: {dp_parts[0]}"
            sfx_dur = int(dur_str)
            if sfx_dur < 5:
                sfx_dur = 5
            if sfx_dur > 30:
                print(f"Warning: SFX duration {sfx_dur}s exceeds 30s, clamping to 30")
                sfx_dur = 30
            if not pos_str.isdigit():
                return None, f"SFX position must be a non-negative number, got: {dp_parts[1]}"
            sfx_pos = int(pos_str)
            if sfx_pos < 0:
                return None, f"SFX position cannot be negative: {sfx_pos}"
            if sfx_pos > max_duration:
                return None, f"SFX position {sfx_pos}s exceeds source duration {max_duration:.1f}s"
            if sfx_pos + sfx_dur > max_duration:
                new_dur = max_duration - sfx_pos
                new_dur = max(1, int(new_dur))
                print(f"Warning: SFX at {sfx_pos}s with duration {sfx_dur}s exceeds source duration {max_duration:.1f}s, auto-cutting to {new_dur}s")
                sfx_dur = new_dur

        if len(parts) >= 3:
            lv_str = parts[2].strip()
            if not lv_str:
                return None, f"SFX level is empty in: {raw}"
            lv_str = lv_str.lstrip('-')
            if not lv_str or not lv_str.isdigit():
                return None, f"SFX level must be a number, got: {parts[2]}"
            sfx_level = int(lv_str)
            if sfx_level < 1:
                print(f"Warning: SFX level {sfx_level} is below 1, setting to 1")
                sfx_level = 1
            if sfx_level > 100:
                print(f"Warning: SFX level {sfx_level} exceeds 100, setting to 100")
                sfx_level = 100

        if sfx_pos is None:
            return None, f'SFX spec requires duration-position (e.g. "sfx:thunder/10-5"): {raw}'

        parsed.append({
            'prompt': prompt,
            'duration': sfx_dur,
            'position': sfx_pos,
            'level': sfx_level
        })
    return parsed, None

def _parse_directives_for_line(directives):
    result = {'time_end': 0, 'time_start': 0, 'time_pad': 0, 'level': 100, 'duration': None, 'has_time': False}
    errors = []
    if 'time_raw' in directives:
        start_pad, cut_start, cut_end, err = _validate_time_directive(directives['time_raw'])
        if err:
            errors.append(f"/time: {err}")
        else:
            result['time_pad'] = start_pad
            result['time_end'] = cut_end
            result['time_start'] = cut_start
            result['has_time'] = True
    if 'level_raw' in directives:
        val, err = _validate_level_directive(directives['level_raw'])
        if err:
            errors.append(f"/level: {err}")
        else:
            result['level'] = val
    if 'duration_raw' in directives:
        val, err = _validate_duration_directive(directives['duration_raw'])
        if err:
            errors.append(f"/duration: {err}")
        else:
            result['duration'] = val
    return result, errors

def _apply_clip_effects(input_path, output_path, cut_start=0, cut_end=0, level=100):
    clip_duration = _get_audio_duration(input_path)
    effective_duration = clip_duration - cut_start - cut_end
    if effective_duration <= 0.01:
        cmd = [
            'ffmpeg', '-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=mono',
            '-t', '0.01', '-y', output_path
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        return
    filters = []
    if level != 100:
        filters.append(f"volume={level / 100.0}")
    filter_str = ",".join(filters)
    cmd = ['ffmpeg', '-ss', str(cut_start), '-i', input_path, '-t', str(effective_duration)]
    if filter_str:
        cmd.extend(['-af', filter_str])
    cmd.extend(['-y', output_path])
    subprocess.run(cmd, capture_output=True, text=True)

def _parse_music_level_spec(spec_str):
    if not spec_str or not spec_str.strip():
        return []
    spec_str = spec_str.strip()
    segments = []
    parts = spec_str.split()
    for part in parts:
        m = re.match(r'^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)-(\d+)$', part)
        if m:
            from_sec = float(m.group(1))
            to_sec = float(m.group(2))
            level_pct = int(m.group(3))
        else:
            m = re.match(r'^(\d+(?:\.\d+)?)-(\d+)$', part)
            if m:
                from_sec = float(m.group(1))
                to_sec = None
                level_pct = int(m.group(2))
            else:
                m = re.match(r'^(\d+)$', part)
                if m:
                    from_sec = 0.0
                    to_sec = None
                    level_pct = int(m.group(1))
                else:
                    return None
        if to_sec is not None and from_sec >= to_sec:
            return None
        if level_pct < 0:
            level_pct = 0
        if level_pct > 100:
            level_pct = 100
        segments.append((from_sec, to_sec, level_pct))
    return segments

def _build_music_volume_expression(segments, total_duration, default_vol=0.35, fade_dur=1.0):
    if not segments:
        return f"volume={default_vol}"
    default_v = f"{default_vol:.6f}"
    expr = default_v
    for from_sec, to_sec, level_pct in reversed(segments):
        if to_sec is None:
            to_sec = total_duration
        if to_sec > total_duration:
            to_sec = total_duration
        if from_sec >= to_sec:
            continue
        vol = level_pct / 100.0
        v = f"{vol:.6f}"
        seg_dur = to_sec - from_sec
        actual_fade = min(fade_dur, seg_dur / 2.0) if seg_dur > 0.01 else 0.01
        af = f"{actual_fade:.2f}"
        fi = max(0, from_sec - actual_fade)
        fo = max(fi, to_sec - actual_fade)
        expr = (
            f"if(between(t,{fi:.3f},{from_sec:.3f}),"
            f"{default_v}+({v}-{default_v})*(t-{fi:.3f})/{af},"
            f"if(between(t,{from_sec:.3f},{fo:.3f}),"
            f"{v},"
            f"if(between(t,{fo:.3f},{to_sec:.3f}),"
            f"{v}+({default_v}-{v})*(t-{fo:.3f})/{af},"
            f"{expr}"
            f")))"
        )
    return f"volume='{expr}':eval=frame"

def _mix_dialogue_with_music(dialogue_path, music_path, output_path, music_level_spec=None):
    duration = _get_audio_duration(dialogue_path)
    segments = _parse_music_level_spec(music_level_spec)
    if segments is None:
        print("Warning: Invalid music level spec, using default 35%")
        segments = []
    vol_filter = _build_music_volume_expression(segments, duration)
    mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    mixed_temp.close()
    cmd = [
        'ffmpeg', '-i', dialogue_path, '-i', music_path,
        '-filter_complex', f'[1:a]{vol_filter}[music];[0:a][music]amix=inputs=2:duration=longest',
        '-y', mixed_temp.name
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg mixing failed: {result.stderr}")
            return False
        shutil.move(mixed_temp.name, output_path)
        return True
    finally:
        if os.path.exists(mixed_temp.name):
            try:
                os.unlink(mixed_temp.name)
            except:
                pass

def _generate_music_and_mix(ace, music_description, dialogue_path, output_path, music_level_spec=None, reference_audio=None):
    duration = _get_audio_duration(dialogue_path)
    print(f"Dialogue duration: {duration:.2f}s")
    print("Generating background music...")
    music_result = generate_background_music(ace, music_description, duration, reference_audio=reference_audio)
    if music_result is None:
        print("Error: Background music generation failed")
        return False
    music_temp_path, music_temp_dir = music_result
    print("Mixing dialogue with music...")
    success = _mix_dialogue_with_music(dialogue_path, music_temp_path, output_path, music_level_spec)
    if music_temp_dir is not None:
        shutil.rmtree(music_temp_dir, ignore_errors=True)
    return success

def _generate_and_overlay_sfx(source_path, sfx_specs, output_path):
    from tangoflux import TangoFluxGenerator
    sfx_gen = TangoFluxGenerator(TANGOFLUX_DIR)
    sfx_gen.ensure_model()
    if sfx_gen.model is None:
        print("Error: Failed to load TangoFlux SFX model")
        sfx_gen.cleanup()
        del sfx_gen
        return False

    sfx_temp_dir = tempfile.mkdtemp()
    sfx_files = []
    try:
        for idx, spec in enumerate(sfx_specs):
            print(f"  Generating SFX [{idx+1}/{len(sfx_specs)}]: \"{spec['prompt']}\" ({spec['duration']}s at {spec['position']}s, level {spec['level']}%)")
            sfx_wav = os.path.join(sfx_temp_dir, f"sfx_{idx}.wav")
            audio = sfx_gen.generate(spec['prompt'], spec['duration'])
            if audio is None:
                print(f"  Warning: SFX generation failed for \"{spec['prompt']}\", skipping")
                continue
            sfx_gen.save(audio, sfx_wav)
            if not os.path.exists(sfx_wav):
                print(f"  Warning: SFX file not saved for \"{spec['prompt']}\", skipping")
                continue
            sfx_files.append((sfx_wav, spec['position'], spec['level'] / 100.0))

        sfx_gen.cleanup()
        del sfx_gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not sfx_files:
            print("Warning: No SFX files generated, copying source as-is")
            shutil.copy2(source_path, output_path)
            return True

        filter_parts = []
        input_idx = 0
        cmd = ['ffmpeg', '-i', source_path]
        for sfx_path, pos, vol in sfx_files:
            cmd.extend(['-i', sfx_path])
            delay_ms = int(pos * 1000)
            label = f"sfx{input_idx}"
            filter_parts.append(f"[{input_idx + 1}:a]adelay={delay_ms}|{delay_ms},volume={vol:.2f}[{label}]")
            input_idx += 1

        mix_inputs = "[0:a]" + "".join(f"[sfx{i}]" for i in range(len(sfx_files)))
        filter_parts.append(f"{mix_inputs}amix=inputs={len(sfx_files) + 1}:duration=first:dropout_transition=0[out]")
        filter_str = ";".join(filter_parts)
        cmd.extend(['-filter_complex', filter_str, '-map', '[out]', '-y', output_path])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: SFX overlay failed: {result.stderr}, copying source as-is")
            shutil.copy2(source_path, output_path)
        return True
    finally:
        try:
            shutil.rmtree(sfx_temp_dir)
        except Exception:
            pass

def _assemble_enhanced_dialogue(dialogue_items, voice_data, tts_design_obj=None, tts_vc_obj=None, vc_voice_data=None, output_path=None, mode='tts', sts_refs=None, use_extreme=False, fish_voice_data=None):
    temp_dir = tempfile.mkdtemp()
    try:
        clips = []
        sfx_generator = None
        design_audio_tracker = {}
        design_cloned_prompts = {}
        sts_vc_obj = None
        if sts_refs:
            sts_vc_obj = SeedVCV2()
            if sts_vc_obj.model is None:
                print("Warning: Seed-VC v2 model failed to load, STS passes will be skipped")
                del sts_vc_obj
                sts_vc_obj = None
        for i, item in enumerate(dialogue_items):
            num = item[0]
            char = item[1]
            text = item[2]
            directives = item[3] if len(item) > 3 else {}
            cut_start = directives.get('time_start', 0)
            cut_end = directives.get('time_end', 0)
            start_pad = directives.get('time_pad', 0)
            level = directives.get('level', 100)
            raw_file = os.path.join(temp_dir, f"raw_{i:03d}.wav")
            processed_file = os.path.join(temp_dir, f"processed_{i:03d}.wav")
            if char.lower() == 'sfx':
                duration = directives.get('duration')
                if duration is None:
                    print(f"Error: SFX line {num} requires /duration:nn (1-30)")
                    return False, "Missing duration for SFX line"
                sfx_valid, _ = _validate_text_language(text, SUPPORTED_TANGOFLUX_LANGS, "SFX")
                if not sfx_valid:
                    return False, f"Unsupported language for SFX line {num}"
                if sfx_generator is None:
                    from tangoflux import TangoFluxGenerator
                    sfx_generator = TangoFluxGenerator(TANGOFLUX_DIR)
                    sfx_generator.ensure_model()
                    if sfx_generator.model is None:
                        return False, "Failed to load TangoFlux model"
                print(f"  Generating SFX line {num}: \"{text[:50]}\" ({duration}s)")
                audio = sfx_generator.generate(text, duration)
                if audio is None:
                    return False, f"SFX generation failed for line {num}"
                sfx_generator.save(audio, raw_file)
            else:
                char_lower = char.lower()
                is_vc = vc_voice_data is not None and char_lower in vc_voice_data
                is_tts = char_lower in voice_data
                if not is_vc and not is_tts:
                    print(f"Error: No voice data for '{char}'")
                    return False, f"Missing voice data for '{char}'"
                if is_vc:
                    if tts_vc_obj is None:
                        return False, "TTS+VC object not provided for cloned voice character"
                    if use_extreme and isinstance(tts_vc_obj, FishTTS) and fish_voice_data and char_lower in fish_voice_data:
                        tts_vc_obj.encoded_refs = fish_voice_data[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                    else:
                        tts_vc_obj.voice_prompt = vc_voice_data[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                    if not success:
                        return False, f"Failed to synthesize line {num}"
                else:
                    if use_extreme and isinstance(tts_vc_obj, FishTTS) and fish_voice_data and char_lower in fish_voice_data:
                        tts_vc_obj.encoded_refs = fish_voice_data[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                        if not success:
                            return False, f"Failed to synthesize line {num}"
                    elif char_lower in design_cloned_prompts and tts_vc_obj is not None:
                        tts_vc_obj.voice_prompt = design_cloned_prompts[char_lower]
                        success = tts_vc_obj.synthesize(text, raw_file)
                        if not success:
                            return False, f"Failed to synthesize line {num}"
                    else:
                        if tts_design_obj is None:
                            return False, "TTS design object not provided"
                        voice_instruct = voice_data[char_lower]
                        success = tts_design_obj.synthesize(text, voice_instruct, raw_file)
                        if not success:
                            return False, f"Failed to synthesize line {num}"
                        if tts_vc_obj is not None and char_lower not in design_cloned_prompts:
                            if char_lower not in design_audio_tracker:
                                design_audio_tracker[char_lower] = []
                            design_audio_tracker[char_lower].append(raw_file)
                            if len(design_audio_tracker[char_lower]) >= 3:
                                concat_path = os.path.join(temp_dir, f"design_clone_{char_lower}_{int(time.time())}.wav")
                                if _concat_audio_files(design_audio_tracker[char_lower], concat_path):
                                    extracted = svs_extract_vocals(concat_path)
                                    clone_src = extracted if extracted and extracted != concat_path else concat_path
                                    try:
                                        clone_ref_text = _transcribe_for_qwen_ref(clone_src)
                                        clone_success = tts_vc_obj.extract_voice(clone_src, ref_text=clone_ref_text if clone_ref_text else None)
                                        if clone_success and tts_vc_obj.voice_prompt is not None:
                                            design_cloned_prompts[char_lower] = tts_vc_obj.voice_prompt
                                    except:
                                        pass
                                    if extracted and extracted != concat_path:
                                        try:
                                            os.unlink(extracted)
                                        except:
                                            pass
                if sts_vc_obj is not None and sts_refs and char_lower in sts_refs:
                    sts_ref_path = sts_refs[char_lower]
                    if os.path.exists(raw_file) and sts_ref_path and os.path.exists(sts_ref_path):
                        sts_out_file = os.path.join(temp_dir, f"sts_{i:03d}.wav")
                        try:
                            svs_raw = svs_extract_vocals(raw_file)
                            vc_source = svs_raw if svs_raw and svs_raw != raw_file else raw_file
                            sts_ok = sts_vc_obj.convert(vc_source, sts_ref_path, sts_out_file)
                            if sts_ok and os.path.exists(sts_out_file):
                                try:
                                    os.unlink(raw_file)
                                except:
                                    pass
                                shutil.move(sts_out_file, raw_file)
                                print(f"  STS pass applied for '{char}' line {num}")
                            else:
                                print(f"  Warning: STS pass failed for '{char}' line {num}, using TTS output")
                            if svs_raw and svs_raw != raw_file:
                                try:
                                    os.unlink(svs_raw)
                                except:
                                    pass
                        except Exception as e:
                            print(f"  Warning: STS pass error for '{char}' line {num}: {e}")
            if not os.path.exists(raw_file):
                return False, f"Audio file not generated for line {num}"
            if cut_start > 0 or cut_end > 0 or level != 100:
                _apply_clip_effects(raw_file, processed_file, cut_start, cut_end, level)
                try:
                    os.unlink(raw_file)
                except:
                    pass
            else:
                shutil.move(raw_file, processed_file)
            has_time = directives.get('has_time', False)
            clips.append((has_time, start_pad, processed_file))
        if len(clips) < 1:
            return False, "No audio segments generated"
        tracks = []
        cursor = 0
        for has_time, orig_pad, fpath in clips:
            if has_time:
                pos = orig_pad
            else:
                pos = cursor
            tracks.append((pos, fpath))
            dur = _get_audio_duration(fpath)
            end = pos + dur
            if end > cursor:
                cursor = end
        if len(tracks) == 1:
            pad_ms, fpath = tracks[0]
            if pad_ms > 0:
                cmd = [
                    'ffmpeg', '-i', fpath,
                    '-af', f'adelay={int(pad_ms * 1000)}|{int(pad_ms * 1000)}',
                    '-y', output_path
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    return False, f"FFmpeg delay failed: {result.stderr}"
            else:
                shutil.copy(fpath, output_path)
            return True, "Dialogue assembled successfully"
        total_duration = 0
        for pad_sec, fpath in tracks:
            d = _get_audio_duration(fpath)
            end = pad_sec + d
            if end > total_duration:
                total_duration = end
        if total_duration <= 0:
            return False, "Total duration is zero"
        cmd = ['ffmpeg']
        for _, fpath in tracks:
            cmd.extend(['-i', fpath])
        filter_parts = []
        for idx, (pad_sec, _) in enumerate(tracks):
            if pad_sec > 0:
                delay_ms = int(pad_sec * 1000)
                filter_parts.append(f"[{idx}:a]adelay={delay_ms}|{delay_ms}[d{idx}]")
            else:
                filter_parts.append(f"[{idx}:a]acopy[d{idx}]")
        input_labels = "".join(f"[d{i}]" for i in range(len(tracks)))
        filter_parts.append(f"{input_labels}amix=inputs={len(tracks)}:duration=longest:dropout_transition=0[out]")
        filter_str = ";".join(filter_parts)
        cmd.extend(['-filter_complex', filter_str, '-map', '[out]', '-y', output_path])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return False, f"FFmpeg mix failed: {result.stderr}"
        return True, "Dialogue assembled successfully"
    finally:
        if sfx_generator:
            sfx_generator.cleanup()
        if sts_vc_obj:
            del sts_vc_obj
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

class QwenTTSVoiceDesign:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.model_dir_full = QWEN_TTS_VOICEDESIGN_DIR if model_dir is None else os.path.join(model_dir, "qwen_tts_voice_design")
        self.model = None
        os.makedirs(self.model_dir_full, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.model_dir_full, exist_ok=True)
        if self.model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                if not os.path.exists(os.path.join(self.model_dir_full, "config.json")):
                    print("Downloading Qwen-TTS VoiceDesign from HuggingFace...")
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        local_dir=self.model_dir_full,
                        local_dir_use_symlinks=False
                    )
                print("Loading Qwen-TTS VoiceDesign model...")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_dir_full,
                    device_map=device,
                    dtype=dtype
                )
            except Exception as e:
                print(f"Error loading Qwen-TTS VoiceDesign: {e}")

    def synthesize(self, text, voice_instruct, output_path, language="Auto"):
        if self.model is None:
            return False
        try:
            import soundfile as sf
            import torch
            wavs, sr = self.model.generate_voice_design(
                text=text,
                language=language,
                instruct=voice_instruct
            )
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"VoiceDesign synthesis error: {e}")
            return False

    def synthesize_dialogue(self, dialogue_items, voice_prompts, output_path, language="Auto"):
        if self.model is None:
            return False, "Model not loaded"
        temp_dir = tempfile.mkdtemp()
        temp_files = []
        try:
            for i, (num, char, script_text) in enumerate(dialogue_items):
                char_lower = char.lower()
                voice_instruct = voice_prompts.get(char_lower, voice_prompts.get(char, ""))
                if not voice_instruct:
                    return False, f"Missing voice prompt for character '{char}'"
                temp_file = os.path.join(temp_dir, f"segment_{i+1:03d}.wav")
                temp_files.append(temp_file)
                success = self.synthesize(script_text, voice_instruct, temp_file, language)
                if not success:
                    return False, f"Failed to synthesize segment {i+1}"
            if len(temp_files) < 2:
                if temp_files:
                    shutil.copy(temp_files[0], output_path)
                return len(temp_files) > 0, "Single segment processed" if temp_files else "No segments generated"
            concat_list = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_list, 'w') as f:
                for tf in temp_files:
                    f.write(f"file '{tf}'\n")
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list, '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                return False, f"FFmpeg concatenation failed: {result.stderr}"
            return True, "Dialogue compiled successfully"
        except Exception as e:
            return False, f"Dialogue processing error: {str(e)}"
        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

class QwenTTS:
    def __init__(self, model_dir=None):
        self.model_dir = MODELS_CHECKPOINTS_DIR if model_dir is None else model_dir
        self.model_dir_base = QWEN_TTS_BASE_DIR if model_dir is None else os.path.join(model_dir, "qwen_tts_base")
        self.model = None
        self.voice_prompt = None
        os.makedirs(self.model_dir_base, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        if self.model is None:
            try:
                from qwen_tts import Qwen3TTSModel
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                if not os.path.exists(os.path.join(self.model_dir_base, "config.json")):
                    print("Downloading Qwen-TTS Base from HuggingFace...")
                    from huggingface_hub import snapshot_download
                    snapshot_download(
                        repo_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        local_dir=self.model_dir_base,
                        local_dir_use_symlinks=False
                    )
                print("Loading Qwen-TTS model...")
                self.model = Qwen3TTSModel.from_pretrained(
                    self.model_dir_base,
                    device_map=device,
                    dtype=dtype
                )
            except Exception as e:
                print(f"Error loading Qwen-TTS: {e}")

    def extract_voice(self, audio_path, ref_text=None):
        if self.model is None:
            return None
        try:
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            waveform_np = waveform.cpu().numpy().flatten()
            use_icl = ref_text is not None and ref_text.strip() != ""
            self.voice_prompt = self.model.create_voice_clone_prompt(
                ref_audio=(waveform_np, sample_rate),
                ref_text=ref_text if use_icl else None,
                x_vector_only_mode=not use_icl
            )
            return True
        except Exception as e:
            print(f"Voice extraction error: {e}")
            return None

    def synthesize(self, text, output_path, language="Auto"):
        if self.model is None or self.voice_prompt is None:
            return False
        try:
            import soundfile as sf
            import torch
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=self.voice_prompt
            )
            sf.write(output_path, wavs[0], sr)
            return True
        except Exception as e:
            print(f"Synthesis error: {e}")
            return False

class FishTTS:
    def __init__(self, model_dir=None):
        self.model_dir = FISH_S2PRO_DIR if model_dir is None else model_dir
        self.model = None
        self.codec = None
        self.decode_one_token = None
        self.device = None
        self.dtype = None
        self.encoded_refs = None
        os.makedirs(self.model_dir, exist_ok=True)

    def ensure_model(self):
        if self.model is not None:
            return True
        try:
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            os.makedirs(self.model_dir, exist_ok=True)
            if not os.path.exists(os.path.join(self.model_dir, "config.json")):
                print("Downloading Fish-S2Pro from HuggingFace...")
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="fishaudio/s2-pro",
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )
            print("Loading Fish-S2Pro model...")
            from fish_speech.models.text2semantic.inference import init_model, load_codec_model
            self.model, self.decode_one_token = init_model(
                checkpoint_path=self.model_dir,
                device=self.device,
                precision=self.dtype,
                compile=False
            )
            codec_path = os.path.join(self.model_dir, "codec.pth")
            self.codec = load_codec_model(codec_path, self.device, self.dtype)
            return True
        except Exception as e:
            print(f"Error loading Fish-S2Pro: {e}")
            return False

    def encode_voice(self, audio_path, ref_text=None):
        if not self.ensure_model():
            return None
        try:
            import torch
            from fish_speech.models.text2semantic.inference import encode_audio
            prompt_tokens = encode_audio(audio_path, self.codec, self.device)
            self.encoded_refs = {
                "tokens": prompt_tokens.cpu(),
                "text": ref_text or ""
            }
            return True
        except Exception as e:
            print(f"Voice encoding error: {e}")
            return None

    def synthesize(self, text, output_path, temperature=1.0, top_p=0.9, top_k=30):
        if not self.ensure_model():
            return False
        if self.encoded_refs is None:
            print("Error: No voice reference encoded for Fish TTS")
            return False
        try:
            import torch
            import soundfile as sf
            from fish_speech.models.text2semantic.inference import generate_long, decode_to_audio
            prompt_tokens = self.encoded_refs["tokens"].to(self.device)
            prompt_text = self.encoded_refs["text"]
            all_audio = []
            for response in generate_long(
                model=self.model,
                device=self.device,
                decode_one_token=self.decode_one_token,
                text=text,
                prompt_text=[prompt_text],
                prompt_tokens=[prompt_tokens],
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=0,
            ):
                if response.action == "sample":
                    audio = decode_to_audio(response.codes.to(self.device), self.codec)
                    if audio is not None:
                        all_audio.append(audio.cpu())
            if not all_audio:
                return False
            final_audio = torch.cat(all_audio, dim=-1)
            audio_np = final_audio.squeeze().float().numpy()
            sf.write(output_path, audio_np, 44100)
            return True
        except Exception as e:
            print(f"Fish TTS synthesis error: {e}")
            return False

    def cleanup(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.codec is not None:
            del self.codec
            self.codec = None
        self.decode_one_token = None
        self.encoded_refs = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

class SeedVCV2:
    def __init__(self):
        self.model = None
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.checkpoints_dir = SEED_VC_V2_DIR
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from hf_utils import load_custom_model_from_hf
                from modules.v2.vc_wrapper import (
                    DEFAULT_CE_REPO_ID, DEFAULT_CE_NARROW_CHECKPOINT,
                    DEFAULT_CE_WIDE_CHECKPOINT, DEFAULT_SE_REPO_ID, DEFAULT_SE_CHECKPOINT
                )
                cfm_path = self.download_checkpoint(
                    repo_id="Plachta/Seed-VC",
                    filename="v2/cfm_small.pth",
                    local_name="cfm_small.pth"
                )
                ar_path = self.download_checkpoint(
                    repo_id="Plachta/Seed-VC",
                    filename="v2/ar_base.pth",
                    local_name="ar_base.pth"
                )
                if not all([cfm_path, ar_path]):
                    return
                config_path = os.path.join(os.path.dirname(__file__), "configs", "v2", "vc_wrapper.yaml")
                cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
                self.model = instantiate(cfg)
                try:
                    from modules.bigvgan import bigvgan
                    self.model.vocoder = bigvgan.BigVGAN.from_pretrained(
                        "nvidia/bigvgan_v2_22khz_80band_256x",
                        use_cuda_kernel=False
                    )
                    print("Vocoder loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load vocoder: {e}")
                self.model.load_checkpoints(
                    cfm_checkpoint_path=cfm_path,
                    ar_checkpoint_path=ar_path
                )
                ce_narrow_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_NARROW_CHECKPOINT,
                    local_name="bsq32_light.pth"
                )
                if ce_narrow_path:
                    ce_narrow_checkpoint = torch.load(ce_narrow_path, map_location="cpu")
                    self.model.content_extractor_narrow.load_state_dict(ce_narrow_checkpoint, strict=False)
                ce_wide_path = self.download_checkpoint(
                    repo_id=DEFAULT_CE_REPO_ID,
                    filename=DEFAULT_CE_WIDE_CHECKPOINT,
                    local_name="bsq2048_light.pth"
                )
                if ce_wide_path:
                    ce_wide_checkpoint = torch.load(ce_wide_path, map_location="cpu")
                    self.model.content_extractor_wide.load_state_dict(ce_wide_checkpoint, strict=False)
                se_path = self.download_checkpoint(
                    repo_id=DEFAULT_SE_REPO_ID,
                    filename=DEFAULT_SE_CHECKPOINT,
                    local_name="campplus_cn_common.bin"
                )
                if se_path:
                    se_checkpoint = torch.load(se_path, map_location="cpu")
                    self.model.style_encoder.load_state_dict(se_checkpoint, strict=False)
                self.model.to(self.device)
                self.model.eval()
                self.model.setup_ar_caches(
                    max_batch_size=1,
                    max_seq_len=8192,
                    dtype=self.dtype,
                    device=self.device
                )
            except ImportError as e:
                print(f"Missing dependency for Seed-VC: {e}")
            except Exception as e:
                print(f"Error loading Seed-VC v2: {e}")

    def download_checkpoint(self, repo_id, filename, local_name):
        local_path = os.path.join(self.checkpoints_dir, local_name)
        if os.path.exists(local_path):
            return local_path
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.checkpoints_dir,
                force_filename=local_name
            )
            return downloaded_path if os.path.exists(downloaded_path) else local_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def convert(self, source_path, reference_path, output_path, convert_style=False):
        if self.model is None:
            return False
        try:
            generator = self.model.convert_voice_with_streaming(
                source_audio_path=source_path,
                target_audio_path=reference_path,
                diffusion_steps=30,
                length_adjust=1.0,
                intelligebility_cfg_rate=0.7,
                similarity_cfg_rate=0.7,
                top_p=0.9,
                temperature=1.0,
                repetition_penalty=1.0,
                convert_style=convert_style,
                anonymization_only=False,
                device=torch.device(self.device),
                dtype=self.dtype,
                stream_output=True
            )
            full_audio = None
            for _, audio in generator:
                full_audio = audio
            if full_audio is not None:
                save_sr, audio_data = full_audio
                sf.write(output_path, audio_data, save_sr)
                return True
            return False
        except Exception as e:
            print(f"Seed-VC conversion error: {e}")
            return False

class SeedVCV1:
    def __init__(self):
        self.model = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.dtype = torch.float16
        self.checkpoints_dir = SEED_VC_V1_DIR
        self.whisper_model = None
        self.whisper_feature_extractor = None
        self.campplus_model = None
        self.bigvgan_model = None
        self.rmvpe = None
        self.to_mel = None
        self.sr = 44100
        self.hop_length = 512
        self.ensure_model()

    def ensure_model(self):
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if self.model is None:
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from hf_utils import load_custom_model_from_hf
                from modules.commons import build_model, load_checkpoint, recursive_munch
                from modules.campplus.DTDNN import CAMPPlus
                from modules.bigvgan import bigvgan
                from modules.audio import mel_spectrogram
                from modules.rmvpe import RMVPE
                from transformers import WhisperModel, AutoFeatureExtractor

                dit_checkpoint_path, dit_config_path = load_custom_model_from_hf(
                    "Plachta/Seed-VC",
                    "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth",
                    "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
                    target_dir=SEED_VC_V1_DIR
                )
                config = yaml.safe_load(open(dit_config_path, 'r'))
                model_params = recursive_munch(config['model_params'])
                self.model = build_model(model_params, stage='DiT')
                self.hop_length = config['preprocess_params']['spect_params']['hop_length']
                self.sr = config['preprocess_params']['sr']

                self.model, _, _, _ = load_checkpoint(
                    self.model, None, dit_checkpoint_path,
                    load_only_params=True, ignore_modules=[], is_distributed=False
                )
                for key in self.model:
                    self.model[key].eval()
                    self.model[key].to(self.device)
                self.model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

                mel_fn_args = {
                    "n_fft": config['preprocess_params']['spect_params']['n_fft'],
                    "win_size": config['preprocess_params']['spect_params']['win_length'],
                    "hop_size": config['preprocess_params']['spect_params']['hop_length'],
                    "num_mels": config['preprocess_params']['spect_params']['n_mels'],
                    "sampling_rate": self.sr,
                    "fmin": 0,
                    "fmax": None,
                    "center": False
                }
                self.to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)

                whisper_name = "openai/whisper-small"
                self.whisper_model = WhisperModel.from_pretrained(whisper_name, torch_dtype=torch.float16).to(self.device)
                del self.whisper_model.decoder
                self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_name)

                campplus_ckpt_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", target_dir=SEED_VC_V1_DIR)
                self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
                self.campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
                self.campplus_model.eval()
                self.campplus_model.to(self.device)

                self.bigvgan_model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
                self.bigvgan_model.remove_weight_norm()
                self.bigvgan_model = self.bigvgan_model.eval().to(self.device)

                rmvpe_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", target_dir=SEED_VC_V1_DIR)
                self.rmvpe = RMVPE(rmvpe_path, is_half=False, device=self.device)

                print("Seed-VC v1 (seed-uvit-whisper-base-f0-44k) loaded successfully")
            except ImportError as e:
                print(f"Missing dependency for Seed-VC v1: {e}")
            except Exception as e:
                print(f"Error loading Seed-VC v1: {e}")

    def download_checkpoint(self, repo_id, filename, local_name):
        local_path = os.path.join(self.checkpoints_dir, local_name)
        if os.path.exists(local_path):
            return local_path
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=self.checkpoints_dir,
                force_filename=local_name
            )
            return downloaded_path if os.path.exists(downloaded_path) else local_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return None

    def _process_whisper_features(self, audio_16k):
        if audio_16k.size(-1) <= 16000 * 30:
            inputs = self.whisper_feature_extractor(
                [audio_16k.squeeze(0).cpu().numpy()],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=16000
            )
            input_features = self.whisper_model._mask_input_features(
                inputs.input_features, attention_mask=inputs.attention_mask
            ).to(self.device)
            outputs = self.whisper_model.encoder(
                input_features.to(self.whisper_model.encoder.dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            features = outputs.last_hidden_state.to(torch.float32)
            features = features[:, :audio_16k.size(-1) // 320 + 1]
        else:
            overlapping_time = 5
            features_list = []
            buffer = None
            traversed_time = 0
            while traversed_time < audio_16k.size(-1):
                if buffer is None:
                    chunk = audio_16k[:, traversed_time:traversed_time + 16000 * 30]
                else:
                    chunk = torch.cat([
                        buffer,
                        audio_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]
                    ], dim=-1)
                inputs = self.whisper_feature_extractor(
                    [chunk.squeeze(0).cpu().numpy()],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000
                )
                input_features = self.whisper_model._mask_input_features(
                    inputs.input_features, attention_mask=inputs.attention_mask
                ).to(self.device)
                outputs = self.whisper_model.encoder(
                    input_features.to(self.whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                chunk_features = outputs.last_hidden_state.to(torch.float32)
                chunk_features = chunk_features[:, :chunk.size(-1) // 320 + 1]
                if traversed_time == 0:
                    features_list.append(chunk_features)
                else:
                    features_list.append(chunk_features[:, 50 * overlapping_time:])
                buffer = chunk[:, -16000 * overlapping_time:]
                traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
            features = torch.cat(features_list, dim=1)
        return features

    def convert(self, source_path, reference_path, output_path, extract_vocals=False):
        if self.model is None:
            return False
        try:
            import librosa
            source_audio = librosa.load(source_path, sr=self.sr)[0]
            actual_reference_path = reference_path

            if extract_vocals:
                print("Extracting clean vocals from target audio...")
                import tempfile as _tf
                temp_vocals = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_vocals.close()
                try:
                    _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
                    if _bs_roformer_lib not in sys.path:
                        sys.path.insert(0, _bs_roformer_lib)
                    _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
                    if _bs_roformer_pkg not in sys.path:
                        sys.path.insert(0, _bs_roformer_pkg)
                    from bs_roformer import BSRoformerSeparator
                    _separator = BSRoformerSeparator(SVS_DIR)
                    _separator.ensure_model(stem='voice')
                    if _separator.vocals_model is not None:
                        _success = _separator.separate(reference_path, 'voice', temp_vocals.name)
                        if _success:
                            actual_reference_path = temp_vocals.name
                            print("Clean vocals extracted from target successfully")
                        else:
                            print("Warning: Vocal extraction failed, using original target")
                    else:
                        print("Warning: Could not load SVS model, using original target")
                    _separator.cleanup()
                    del _separator
                    _separator = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as _e:
                    print(f"Warning: Vocal extraction error: {_e}, using original target")

            ref_audio = librosa.load(actual_reference_path, sr=self.sr)[0]

            source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(self.device)
            ref_audio = torch.tensor(ref_audio[:self.sr * 25]).unsqueeze(0).float().to(self.device)

            ref_waves_16k = torchaudio.functional.resample(ref_audio, self.sr, 16000)
            converted_waves_16k = torchaudio.functional.resample(source_audio, self.sr, 16000)

            S_alt = self._process_whisper_features(converted_waves_16k)
            S_ori = self._process_whisper_features(ref_waves_16k)

            mel = self.to_mel(source_audio.to(self.device).float())
            mel2 = self.to_mel(ref_audio.to(self.device).float())

            target_lengths = torch.LongTensor([int(mel.size(2))]).to(mel.device)
            target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

            feat2 = torchaudio.compliance.kaldi.fbank(
                ref_waves_16k,
                num_mel_bins=80,
                dither=0,
                sample_frequency=16000
            )
            feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
            style2 = self.campplus_model(feat2.unsqueeze(0))

            F0_ori = self.rmvpe.infer_from_audio(ref_waves_16k[0], thred=0.03)
            F0_alt = self.rmvpe.infer_from_audio(converted_waves_16k[0], thred=0.03)

            if self.device.type == "mps":
                F0_ori = torch.from_numpy(F0_ori).float().to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).float().to(self.device)[None]
            else:
                F0_ori = torch.from_numpy(F0_ori).to(self.device)[None]
                F0_alt = torch.from_numpy(F0_alt).to(self.device)[None]

            voiced_F0_ori = F0_ori[F0_ori > 1]
            voiced_F0_alt = F0_alt[F0_alt > 1]

            log_f0_alt = torch.log(F0_alt + 1e-5)
            voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
            voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
            median_log_f0_ori = torch.median(voiced_log_f0_ori)
            median_log_f0_alt = torch.median(voiced_log_f0_alt)

            shifted_log_f0_alt = log_f0_alt.clone()
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
            shifted_f0_alt = torch.exp(shifted_log_f0_alt)

            cond, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_alt, ylens=target_lengths, n_quantizers=3, f0=shifted_f0_alt
            )
            prompt_condition, _, codes, commitment_loss, codebook_loss = self.model.length_regulator(
                S_ori, ylens=target2_lengths, n_quantizers=3, f0=F0_ori
            )

            max_context_window = self.sr // self.hop_length * 30
            max_source_window = max_context_window - mel2.size(2)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                vc_target = self.model.cfm.inference(
                    torch.cat([prompt_condition, cond], dim=1),
                    torch.LongTensor([torch.cat([prompt_condition, cond], dim=1).size(1)]).to(mel2.device),
                    mel2, style2, None, 10,
                    inference_cfg_rate=0.7
                )
            vc_target = vc_target[:, :, mel2.size(-1):]

            del self.whisper_model, self.whisper_feature_extractor
            del self.campplus_model, self.rmvpe, self.model
            self.whisper_model = None
            self.whisper_feature_extractor = None
            self.campplus_model = None
            self.rmvpe = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            vc_wave = self.bigvgan_model(vc_target.clone().float())[0]

            output_audio = vc_wave[0].cpu().numpy()
            sf.write(output_path, output_audio, self.sr)
            if extract_vocals and actual_reference_path != reference_path:
                try:
                    os.remove(actual_reference_path)
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"Seed-VC v1 conversion error: {e}")
            import traceback
            traceback.print_exc()
            return False

class AceStepWrapper:
    def __init__(self, use_overdose=False, complete_mode=False):
        self.checkpoints_dir = ACESTEP_DIR
        self.handler = None
        self.use_overdose = use_overdose
        self.complete_mode = complete_mode
        if complete_mode:
            self.config_path = "acestep-v15-xl-base"
            self.lm_model = "acestep-5Hz-lm-1.7B"
            self.shift = 1.0
        else:
            self.config_path = "acestep-v15-xl-turbo" if use_overdose else "acestep-v15-turbo"
            self.lm_model = "acestep-5Hz-lm-4B" if use_overdose else "acestep-5Hz-lm-1.7B"
            self.shift = 3.0 if use_overdose else 1.0
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.ensure_model()

    def ensure_model(self):
        if self.handler is None:
            if self.complete_mode:
                single_gpu_gb, total_sys_gb = get_system_resources()
                if single_gpu_gb < 32.0 and total_sys_gb < 48.0:
                    print(f"Error: ACE-Step Complete (XL-Base) requires 32GB+ VRAM or 48GB+ combined System Memory (RAM+Swap/Pagefile)")
                    print(f"Detected: {single_gpu_gb:.1f}GB VRAM, {total_sys_gb:.1f}GB System Memory")
                    print("Cannot proceed with complete task.")
                    return
            elif self.use_overdose:
                single_gpu_gb, total_sys_gb = get_system_resources()
                if single_gpu_gb < 32.0 and total_sys_gb < 48.0:
                    print(f"Error: ACE-Step Overdose requires 32GB+ VRAM or 48GB+ combined System Memory (RAM+Swap/Pagefile)")
                    print(f"Detected: {single_gpu_gb:.1f}GB VRAM, {total_sys_gb:.1f}GB System Memory")
                    print("Falling back to Standard ACE-Step model...")
                    self.use_overdose = False
                    self.config_path = "acestep-v15-turbo"
                    self.lm_model = "acestep-5Hz-lm-1.7B"
                    self.shift = 1.0
            try:
                import sys
                sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
                from acestep.handler import AceStepHandler
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                print(f"Loading ACE-Step model ({self.config_path})...")
                self.handler = AceStepHandler()
                status, success = self.handler.initialize_service(
                    project_root=self.checkpoints_dir,
                    config_path=self.config_path,
                    device=device
                )
                if not success:
                    print(f"Error initializing ACE-Step: {status}")
                    self.handler = None
            except Exception as e:
                print(f"Error loading ACE-Step model: {e}")
                self.handler = None

    def generate(self, lyrics, style_prompt, output_path, duration=10, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "text2music",
                "shift": self.shift,
            }
            if reference_audio is not None:
                _gen_kwargs["reference_audio"] = reference_audio
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step generation error: {e}")
            return False

    def cover(self, src_audio, style_prompt, output_path, cover_strength=0.4, reference_audio=None, lyrics="..."):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": 30,
                "batch_size": 1,
                "task_type": "cover",
                "shift": self.shift,
                "src_audio": src_audio,
                "audio_cover_strength": cover_strength,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step cover error: {e}")
            return False

    def repaint(self, src_audio, style_prompt, output_path, repaint_start, repaint_end, lyrics="...", cover_strength=0.4, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            _gen_kwargs = {
                "captions": style_prompt,
                "lyrics": lyrics,
                "vocal_language": "unknown",
                "inference_steps": 8,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": 30,
                "batch_size": 1,
                "task_type": "repaint",
                "shift": self.shift,
                "src_audio": src_audio,
                "repainting_start": repaint_start,
                "repainting_end": repaint_end,
                "audio_cover_strength": cover_strength,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step repaint error: {e}")
            return False

    def complete(self, src_audio, track_classes, output_path, styling=None, duration=None, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = "Complete the input track with " + " | ".join(t.upper() for t in track_classes) + ":"
            _gen_kwargs = {
                "captions": styling if styling else "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "complete",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
                "audio_cover_strength": 0.2,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step complete error: {e}")
            return False

    def extract(self, src_audio, track_name, output_path, duration=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = f"Extract the {track_name.upper()} track from the audio:"
            _gen_kwargs = {
                "captions": "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "extract",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
            }
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step extract error: {e}")
            return False

    def lego(self, src_audio, track_name, output_path, styling=None, duration=None, reference_audio=None):
        if self.handler is None:
            return False
        try:
            import soundfile as sf
            if duration is None:
                try:
                    info = sf.info(src_audio)
                    duration = info.duration
                except:
                    duration = 30
            instruction = f"Generate the {track_name.upper()} track based on the audio context:"
            _gen_kwargs = {
                "captions": styling if styling else "",
                "lyrics": "",
                "vocal_language": "unknown",
                "inference_steps": 50,
                "guidance_scale": 7.0,
                "use_random_seed": True,
                "seed": -1,
                "audio_duration": duration,
                "batch_size": 1,
                "task_type": "lego",
                "shift": 1.0,
                "src_audio": src_audio,
                "instruction": instruction,
                "audio_cover_strength": 0.2,
                "reference_audio": reference_audio,
            }
            result = self.handler.generate_music(**_gen_kwargs)
            if result.get("success", False) and result.get("audios"):
                audio_dict = result["audios"][0]
                audio_tensor = audio_dict.get("tensor")
                sample_rate = audio_dict.get("sample_rate", 48000)
                if audio_tensor is not None:
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_array = audio_tensor.cpu().numpy()
                    else:
                        audio_array = audio_tensor
                    if len(audio_array.shape) == 2:
                        audio_array = audio_array.transpose(1, 0)
                    sf.write(output_path, audio_array, sample_rate)
                    return True
            return False
        except Exception as e:
            print(f"ACE-Step lego error: {e}")
            return False

VALID_ACESTEP_TRACKS = {"woodwinds", "brass", "fx", "synth", "strings", "percussion",
                        "keyboard", "guitar", "bass", "drums", "backing_vocals", "vocals"}
ACESTEP_INSTRUMENT_TRACKS = {"woodwinds", "brass", "fx", "synth", "strings", "percussion",
                              "keyboard", "guitar", "bass", "drums"}
ACESTEP_VOICE_TRACKS = {"vocals", "backing_vocals"}

def resolve_acestep_tracks(instruments_raw):
    track_classes = []
    use_everything = False
    use_instruments = False
    use_voices = False
    for t in instruments_raw.strip().split():
        tl = t.lower()
        if tl == 'everything':
            use_everything = True
        elif tl == 'instruments':
            use_instruments = True
        elif tl == 'voices':
            use_voices = True
        elif use_everything:
            continue
        else:
            track_classes.append(tl)
    if use_everything:
        return sorted(VALID_ACESTEP_TRACKS), None
    if use_instruments and use_voices:
        return sorted(VALID_ACESTEP_TRACKS), None
    if use_instruments:
        expansion = sorted(ACESTEP_INSTRUMENT_TRACKS)
        track_classes = [t for t in track_classes if t in ACESTEP_VOICE_TRACKS]
        track_classes = expansion + track_classes
    if use_voices:
        expansion = sorted(ACESTEP_VOICE_TRACKS)
        track_classes = [t for t in track_classes if t in ACESTEP_INSTRUMENT_TRACKS]
        track_classes = expansion + track_classes
    track_classes = list(dict.fromkeys(track_classes))
    unknown = [t for t in track_classes if t not in VALID_ACESTEP_TRACKS]
    if unknown:
        return None, unknown
    if not track_classes:
        return None, []
    return track_classes, None

def parse_ref_raw(raw):
    colon_idx = raw.find(':')
    if colon_idx == -1:
        return None, raw.strip()
    prefix = raw[:colon_idx].strip().lower()
    rest = raw[colon_idx + 1:].strip()
    if not rest:
        return None, raw.strip()
    if prefix in VALID_ACESTEP_TRACKS or prefix in ('everything', 'instruments', 'voices'):
        return prefix, rest
    return None, raw.strip()

def generate_background_music(ace_wrapper, music_description, total_duration, progress_callback=None, reference_audio=None):
    min_duration = 10

    if total_duration < min_duration:
        total_duration = min_duration

    if total_duration <= 250:
        music_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        music_temp.close()
        success = ace_wrapper.generate(
            lyrics="...",
            style_prompt=music_description,
            output_path=music_temp.name,
            duration=int(total_duration),
            reference_audio=reference_audio
        )
        if not success:
            os.unlink(music_temp.name)
            return None
        return music_temp.name, None

    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    chunk_size = 250

    num_chunks = math.ceil(total_duration / chunk_size)

    for i in range(num_chunks):
        if progress_callback:
            progress_callback(i, num_chunks)

        chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        chunk_files.append(chunk_file)

        if i == num_chunks - 1:
            current_duration = total_duration - (i * chunk_size)
            if current_duration < min_duration:
                current_duration = min_duration
        else:
            current_duration = chunk_size

        success = ace_wrapper.generate(
            lyrics="...",
            style_prompt=music_description,
            output_path=chunk_file,
            duration=int(current_duration),
            reference_audio=reference_audio
        )

        if not success:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    if progress_callback:
        progress_callback(num_chunks, num_chunks)

    concat_file = os.path.join(temp_dir, "concat_list.txt")
    with open(concat_file, 'w') as f:
        for chunk in chunk_files:
            f.write(f"file '{chunk}'\n")

    output_file = os.path.join(temp_dir, "music.wav")
    cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file, '-y', output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    for chunk_file in chunk_files:
        if os.path.exists(chunk_file):
            os.unlink(chunk_file)

    if result.returncode != 0:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None

    return output_file, temp_dir

def validate_file_exists(path):
    if os.path.exists(path):
        return True
    print(f"Error: File not found: {path}")
    return False

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv'}

VOICE_PROFILE_EXTENSIONS = {'.tts', '.ttse'}

_AUDIO_VIDEO_URL = 'audio file / video file / supported platform URL'

MODE_INPUT_FORMATS = {
    'tts':   f'{_AUDIO_VIDEO_URL} / .tts or .ttse voice profile (only at voice slots or target slots using sts: prefix)',
    'sts':   _AUDIO_VIDEO_URL,
    'ttm':   f'{_AUDIO_VIDEO_URL} / text file (.txt)',
    'stt':   _AUDIO_VIDEO_URL,
    'se':    _AUDIO_VIDEO_URL,
    'sfx':   '(no file input — uses inline text prompt via "sound <text>")',
    'svs':   _AUDIO_VIDEO_URL,
    'ss':    _AUDIO_VIDEO_URL,
    'train': _AUDIO_VIDEO_URL,
    'quest': 'varies by quest type',
}


def slot_accepts_voice_profile(mode, content_tokens, slot_pos):
    if mode != 'tts':
        return False
    if slot_pos < 0 or slot_pos >= len(content_tokens):
        return False
    if content_tokens[slot_pos] != 'input':
        return False
    prev = content_tokens[slot_pos - 1] if slot_pos > 0 else None
    if prev == 'voice':
        return True
    if prev == 'target':
        return True
    if prev is None:
        return True
    return False


def describe_input_slot(mode, content_tokens, slot_pos):
    if mode == 'sfx':
        return '(no file input — sfx uses inline sound <text> prompt)'
    if mode == 'quest':
        return 'varies by quest type — see quest documentation'
    base = _AUDIO_VIDEO_URL
    if slot_accepts_voice_profile(mode, content_tokens, slot_pos):
        prev = content_tokens[slot_pos - 1] if slot_pos > 0 else None
        if prev == 'voice':
            return f'{base} / .tts or .ttse voice profile (voice slot — engine resolves trained voice refs)'
        if prev == 'target':
            return f'{base} / .tts or .ttse voice profile (target slot — only when value starts with "sts:" prefix)'
        return f'{base} / .tts or .ttse voice profile'
    return base

SUPPORTED_TTS_LANGUAGES = {
    "zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
    "de": "German", "fr": "French", "ru": "Russian", "pt": "Portuguese",
    "es": "Spanish", "it": "Italian"
}

SUPPORTED_ACESTEP_LANGS = {
    'ar', 'az', 'bg', 'bn', 'ca', 'cs', 'da', 'de', 'el', 'en', 'es', 'fa',
    'fi', 'fr', 'he', 'hi', 'hr', 'ht', 'hu', 'id', 'is', 'it', 'ja', 'ko',
    'la', 'lt', 'ms', 'ne', 'nl', 'no', 'pa', 'pl', 'pt', 'ro', 'ru', 'sa',
    'sk', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'uk', 'ur', 'vi',
    'yue', 'zh'
}

SUPPORTED_FISH_LANGS = {
    'af', 'am', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'bo', 'br', 'bs', 'ca',
    'cs', 'cy', 'da', 'de', 'el', 'en', 'es', 'et', 'eu', 'fa', 'fi', 'fr',
    'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'id', 'is', 'it', 'ja', 'jw',
    'ka', 'kk', 'km', 'kn', 'ko', 'lt', 'lv', 'mi', 'ml', 'mn', 'mr', 'ms',
    'my', 'ne', 'nl', 'nn', 'no', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru', 'sd',
    'sk', 'sl', 'sn', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr',
    'uk', 'ur', 'vi', 'yo', 'zh'
}

SUPPORTED_TANGOFLUX_LANGS = {'en'}

def _validate_text_language(text, supported_langs, context_name):
    detected = _detect_lang_from_text(text)
    if detected not in supported_langs:
        lang_name = SUPPORTED_TTS_LANGUAGES.get(detected) or detected.upper()
        print(f"Unsupported language ({lang_name}) for {context_name}")
        return False, detected
    return True, detected

def validate_audio_file(path):
    if not os.path.exists(path):
        return False, "File does not exist."
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return True, "video"
    try:
        torchaudio.load(path)
        return True, "audio"
    except Exception as e:
        return False, f"Unsupported or corrupt audio/video format: {str(e)}"

def extract_audio_from_video_cli(video_path):
    try:
        temp_dir = tempfile.gettempdir()
        audio_path = os.path.join(temp_dir, f"voder_cli_{int(time.time())}.wav")
        cmd = ['ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(audio_path):
            return audio_path
        return None
    except Exception as e:
        print(f"FFmpeg error: {e}")
        return None

def _replace_audio_in_video(video_path, audio_path, output_path):
    try:
        cmd = [
            'ffmpeg', '-i', video_path, '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0', '-shortest',
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Error replacing audio in video: {e}")
        return False

def svs_extract_vocals(audio_path):
    try:
        print("Cleaning target audio through SVS voice pipe...")
        import tempfile as _tf
        temp_vocals = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_vocals.close()
        try:
            _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
            if _bs_roformer_lib not in sys.path:
                sys.path.insert(0, _bs_roformer_lib)
            _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
            if _bs_roformer_pkg not in sys.path:
                sys.path.insert(0, _bs_roformer_pkg)
            from bs_roformer import BSRoformerSeparator
            _separator = BSRoformerSeparator(SVS_DIR)
            _separator.ensure_model(stem='voice')
            _result = audio_path
            try:
                if _separator.vocals_model is not None:
                    _success = _separator.separate(audio_path, 'voice', temp_vocals.name)
                    if _success:
                        print("Target vocals cleaned successfully")
                        _result = temp_vocals.name
                    else:
                        print("Warning: SVS vocal extraction failed, using original target")
                else:
                    print("Warning: Could not load SVS model, using original target")
            finally:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _result
        except Exception as _e:
            print(f"Warning: SVS vocal extraction error: {_e}, using original target")
            try:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        try:
            os.unlink(temp_vocals.name)
        except:
            pass
        return audio_path
    except Exception as _e:
        print(f"Warning: SVS error: {_e}, using original target")
        return audio_path

def svs_extract_music(audio_path):
    try:
        print("Cleaning target audio through SVS music pipe...")
        import tempfile as _tf
        temp_music = _tf.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_music.close()
        try:
            _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
            if _bs_roformer_lib not in sys.path:
                sys.path.insert(0, _bs_roformer_lib)
            _bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
            if _bs_roformer_pkg not in sys.path:
                sys.path.insert(0, _bs_roformer_pkg)
            from bs_roformer import BSRoformerSeparator
            _separator = BSRoformerSeparator(SVS_DIR)
            _separator.ensure_model(stem='music')
            _result = audio_path
            try:
                if _separator.inst_model is not None:
                    _success = _separator.separate(audio_path, 'music', temp_music.name)
                    if _success:
                        print("Target music cleaned successfully")
                        _result = temp_music.name
                    else:
                        print("Warning: SVS music extraction failed, using original target")
                else:
                    print("Warning: Could not load SVS music model, using original target")
            finally:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return _result
        except Exception as _e:
            print(f"Warning: SVS music extraction error: {_e}, using original target")
            try:
                _separator.cleanup()
                del _separator
                _separator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        try:
            os.unlink(temp_music.name)
        except:
            pass
        return audio_path
    except Exception as _e:
        print(f"Warning: SVS error: {_e}, using original target")
        return audio_path

def ss_extract_speakers(audio_path, use_overdose=False):
    try:
        print("Running SS pipe for speaker separation...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        original_name = os.path.splitext(os.path.basename(audio_path))[0]
        temp_results = tempfile.mkdtemp()
        outputs = _ss_run_pipeline(
            audio_path, False, temp_results, original_name, timestamp,
            target_path=None, use_overdose=use_overdose
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not outputs:
            try:
                shutil.rmtree(temp_results)
            except:
                pass
            return {}, None
        speaker_clips = {}
        for out_path in outputs:
            fname = os.path.basename(out_path)
            match = re.search(r'speaker(\d+)', fname)
            if match:
                spk_num = str(int(match.group(1)))
                speaker_clips[spk_num] = out_path
        return speaker_clips, temp_results
    except Exception as _e:
        print(f"Warning: SS pipe error: {_e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {}, None

def resolve_target_to_audio(path):
    cleanup_files = []
    if is_supported_url(path):
        platform_id = detect_platform(path)
        pname = platform_name(platform_id)
        is_vid, verify_err, _pid = is_video_url(path, verify=True)
        if not is_vid:
            print(f"Error: {verify_err or 'This link is not a video'}")
            return None, cleanup_files
        print(f"Downloading audio from {pname}: {path}")
        success_dl, error_msg, audio_path = download_url_audio(path, skip_verify=True)
        if not success_dl:
            print(f"Error: {error_msg}")
            return None, cleanup_files
        cleanup_files.append(audio_path)
        return audio_path, cleanup_files
    if not os.path.exists(path):
        print(f"Error: Target not found: {path}")
        return None, cleanup_files
    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        print("Extracting audio from video target...")
        extracted = extract_audio_from_video_cli(path)
        if not extracted:
            print("Error: Could not extract audio from video")
            return None, cleanup_files
        cleanup_files.append(extracted)
        return extracted, cleanup_files
    valid, msg = validate_audio_file(path)
    if not valid:
        print(f"Error: {msg}")
        return None, cleanup_files
    return path, cleanup_files

def validate_dialogue_source_file(file_path):
    if is_supported_url(file_path):
        platform_id = detect_platform(file_path)
        return True, "url", platform_id

    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}", None

    ext = file_path.lower()
    if ext.endswith(('.mp4', '.avi', '.mov', '.mkv', '.wav', '.mp3', '.flac', '.m4a')):
        return True, "audio", None
    elif ext.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')):
        return True, "image", None
    elif ext.endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                return False, "Empty text file", None

            lines = content.split('\n')
            dialogue_items = []
            mode_detected = None
            auto_formatted = False

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                has_colon = ':' in line
                if mode_detected is None:
                    mode_detected = 'dialogue' if has_colon else 'single'

                if mode_detected == 'single':
                    if len(lines) == 1:
                        dialogue_items.append((1, 'text', line))
                    else:
                        dialogue_items.append((len(dialogue_items) + 1, 'text', line))
                        auto_formatted = True
                else:
                    if ':' not in line:
                        dialogue_items.append((len(dialogue_items) + 1, 'text', line))
                        auto_formatted = True
                    else:
                        parts = line.split(':', 1)
                        speaker = parts[0].strip()
                        text = parts[1].strip()
                        if not speaker or not text:
                            dialogue_items.append((len(dialogue_items) + 1, 'text', line.split(':', 1)[1].strip() if ':' in line else line))
                            auto_formatted = True
                        else:
                            dialogue_items.append((len(dialogue_items) + 1, speaker, text))

            if not dialogue_items:
                return False, "No valid dialogue found in file", None

            if auto_formatted:
                print(f"\n[Auto-format] TXT file has been reformatted for compatibility:")
                print("  - Lines without speaker name are prefixed with 'text:'")
                print("  - Empty lines have been removed")
                print(f"  - Total lines after formatting: {len(dialogue_items)}")

            return True, "txt", dialogue_items

        except Exception as e:
            return False, f"Error reading file: {str(e)}", None
    else:
        return False, f"Unsupported file format: {file_path}", None

def analyze_dialogue_source(file_path, source_type="audio", use_overdose=False):
    if source_type == "txt":
        return True, None, None, None, None

    if source_type == "image":
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            return False, "Failed to load EasyOCR model", None, None, None

        print(f"Extracting text from image: {os.path.basename(file_path)}")
        success, text, error_msg = ocr.extract_text_from_image(file_path)

        ocr.cleanup()
        del ocr
        gc.collect()

        if not success:
            return False, error_msg or "Failed to extract text from image", None, None, None

        if not text:
            return False, "No text found in image", None, None, None

        dialogue_items = [(1, 'text', text, {})]
        return True, None, dialogue_items, None, None

    if source_type == "url":
        platform_id = detect_platform(file_path)
        pname = platform_name(platform_id)
        is_vid, verify_err, _pid = is_video_url(file_path, verify=True)
        if not is_vid:
            return False, verify_err or "This link is not a video", None, None, None
        print(f"Downloading audio from {pname}...")
        success, error_msg, audio_path = download_url_audio(file_path, skip_verify=True)
        if not success:
            return False, error_msg, None, None, None

        file_path = audio_path

    audio_path = file_path
    needs_cleanup = False
    if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        audio_path = extract_audio_from_video_cli(file_path)
        if not audio_path:
            return False, "Failed to extract audio from video", None, None, None
        needs_cleanup = True
    elif not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        return False, f"Unsupported audio format: {file_path}", None, None, None

    try:
        speech_segments = None
        if use_overdose:
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper + pyannote")
                asr.cleanup()
                del asr
                use_overdose = False
            else:
                print("Transcribing with VibeVoice ASR...")
                speech_segments = asr.transcribe(audio_path)
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not speech_segments:
                    return False, "VibeVoice ASR transcription returned no segments", None, None, None

        if speech_segments is None:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                return False, "Failed to load Whisper model", None, None, None

            print("Transcribing audio...")
            result = stt.transcribe(audio_path)

            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not result:
                return False, "Transcription failed", None, None, None

            print("Performing speaker diarization...")
            diarization = SpeakerDiarization()

            if diarization.pipeline is None:
                text = result.get("text", "").strip()
                if not text:
                    return False, "No text transcribed", None, None, None
                dialogue_items = [(1, 'text', text, {})]
                return True, None, dialogue_items, audio_path, None

            diar_result = diarization.diarize(audio_path)
            formatted_segments = diarization.format_diarization(diar_result, result)

            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not formatted_segments:
                text = result.get("text", "").strip()
                dialogue_items = [(1, 'text', text, {})]
                return True, None, dialogue_items, audio_path, None

            speech_segments = []
            for seg in formatted_segments:
                speech_segments.append({
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0)
                })

        original_speakers = []
        for seg in speech_segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            if speaker not in original_speakers:
                original_speakers.append(speaker)

        if len(original_speakers) < 2:
            content = " ".join(seg.get("text", "") for seg in speech_segments)
            dialogue_items = [(1, 'text', content.strip(), {})]
            return True, None, dialogue_items, audio_path, None

        print(f"Detected {len(original_speakers)} speakers, running TSE per-speaker extraction...")
        speaker_extraction = _extract_speakers_for_subtitles(audio_path)
        if speaker_extraction is None:
            speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}
            dialogue_items = []
            current_speaker = None
            current_text_parts = []
            for seg in speech_segments:
                spk_label = speaker_mapping.get(seg.get("speaker", "SPEAKER_00"), 1)
                text = seg.get("text", "")
                if current_speaker is None:
                    current_speaker = spk_label
                    current_text_parts = [text]
                elif spk_label == current_speaker:
                    current_text_parts.append(text)
                else:
                    if current_text_parts:
                        dialogue_items.append((len(dialogue_items) + 1, str(current_speaker), " ".join(current_text_parts), {}))
                    current_speaker = spk_label
                    current_text_parts = [text]
            if current_text_parts:
                dialogue_items.append((len(dialogue_items) + 1, str(current_speaker), " ".join(current_text_parts), {}))
            return True, None, dialogue_items, audio_path, None

        speaker_files = speaker_extraction.get("speaker_files", {})
        diar_speakers = sorted(speaker_extraction.get("speaker_segments", {}).keys(),
                               key=lambda spk: speaker_extraction["speaker_segments"][spk][0]["start"])
        asr_speakers_sorted = sorted(original_speakers,
                                     key=lambda spk: next((seg.get('start', 0) for seg in speech_segments if seg.get('speaker') == spk), 0))
        asr_to_diar = {}
        for idx, asr_spk in enumerate(asr_speakers_sorted):
            asr_to_diar[asr_spk] = diar_speakers[idx] if idx < len(diar_speakers) else diar_speakers[-1]

        spk_texts = {}
        spk_segments_data = {}
        for seg in speech_segments:
            spk = seg.get("speaker", "SPEAKER_00")
            if spk not in spk_texts:
                spk_texts[spk] = []
                spk_segments_data[spk] = []
            spk_texts[spk].append(seg.get("text", "").strip())
            spk_segments_data[spk].append({"start": seg.get("start", 0), "end": seg.get("end", 0)})

        print("Running per-speaker forced alignment...")
        all_aligned_segments = []
        for asr_spk in asr_speakers_sorted:
            diar_spk = asr_to_diar.get(asr_spk)
            spk_audio = speaker_files.get(diar_spk) if diar_spk else None
            spk_text = " ".join(spk_texts.get(asr_spk, []))
            if not spk_text or not spk_audio or not os.path.exists(spk_audio):
                for seg_data in spk_segments_data.get(asr_spk, []):
                    all_aligned_segments.append({
                        "speaker": asr_spk,
                        "text": "",
                        "start": seg_data["start"],
                        "end": seg_data["end"]
                    })
                continue

            lang_code = _detect_lang_from_text(spk_text)
            iso3 = _LANG_TO_ISO3.get(lang_code, "eng")
            word_ts = _forced_align_words(spk_audio, spk_text, language=iso3)
            if word_ts:
                grouped = _group_words_to_segments(word_ts, chunk_size=8, speaker=asr_spk)
                all_aligned_segments.extend(grouped)
            else:
                for seg_data in spk_segments_data.get(asr_spk, []):
                    all_aligned_segments.append({
                        "speaker": asr_spk,
                        "text": "",
                        "start": seg_data["start"],
                        "end": seg_data["end"]
                    })

        all_aligned_segments.sort(key=lambda s: s.get("start", 0))

        dialogue_items = []
        current_speaker = None
        current_text_parts = []
        seg_start = 0
        seg_end = 0
        for seg in all_aligned_segments:
            spk = seg.get("speaker", "SPEAKER_00")
            text = seg.get("text", "").strip()
            if not text:
                continue
            if current_speaker is None:
                current_speaker = spk
                current_text_parts = [text]
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)
            elif spk == current_speaker:
                current_text_parts.append(text)
                seg_end = seg.get("end", 0)
            else:
                if current_text_parts:
                    directives = {"time_pad": seg_start, "has_time": True}
                    dialogue_items.append((len(dialogue_items) + 1, current_speaker, " ".join(current_text_parts), directives))
                current_speaker = spk
                current_text_parts = [text]
                seg_start = seg.get("start", 0)
                seg_end = seg.get("end", 0)

        if current_text_parts:
            directives = {"time_pad": seg_start, "has_time": True}
            dialogue_items.append((len(dialogue_items) + 1, current_speaker, " ".join(current_text_parts), directives))

        if not dialogue_items:
            _cleanup_speaker_extraction(speaker_extraction)
            content = " ".join(seg.get("text", "") for seg in speech_segments)
            dialogue_items = [(1, 'text', content.strip(), {})]
            return True, None, dialogue_items, audio_path, None

        speaker_name_mapping = {}
        for idx, asr_spk in enumerate(asr_speakers_sorted):
            speaker_name_mapping[asr_spk] = str(idx + 1)
        mapped_items = []
        for item_idx, spk, text, directives in dialogue_items:
            mapped_spk = speaker_name_mapping.get(spk, spk)
            mapped_items.append((item_idx, mapped_spk, text, directives))

        return True, None, mapped_items, audio_path, speaker_extraction

    except Exception as e:
        return False, f"Error analyzing audio: {str(e)}", None, None, None





def parse_oneline_args(args):
    if not args:
        return {'error': 'No arguments provided'}
    mode = args[0].lower()
    result = {'mode': mode, 'params': {}, 'error': None, 'is_music': False, 'is_mimic': False, 'nomusic': False}
    valid_keywords = ['script', 'voice', 'lyrics', 'styling', 'base', 'target', 'duration', 'timestamp', 'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr', 'reference', 'music']
    i = 1
    current_keyword = None
    result_path = None

    if mode == 'overdose' and len(args) > 1 and args[1].lower() == 'vadar':
        mode = 'vadar'
        result['mode'] = 'vadar'
        result['params']['use_lite'] = False
        i = 2
        prompt_parts = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                prompt_parts.append(arg)
                i += 1
        result['params']['prompt'] = ' '.join(prompt_parts)
        result['params']['result_path'] = result_path
        return result

    if mode == 'vadar':
        prompt_parts = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                prompt_parts.append(arg)
                i += 1
        result['params']['prompt'] = ' '.join(prompt_parts)
        result['params']['result_path'] = result_path
        result['params']['use_lite'] = True
        return result

    if mode == 'stt':
        file_paths = []
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower in ['timestamp', 'dialogue', 'se', 'overdose', 'subtitle']:
                result['params'][arg_lower] = True
                i += 1
            elif arg_lower == 'translate':
                result['params']['translate'] = True
                i += 1
                if i < len(args):
                    lang_spec = _parse_lang_spec(args[i])
                    if lang_spec:
                        result['params']['translate_langs'] = lang_spec
                        i += 1
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif os.path.exists(arg) or is_youtube_url(arg):
                file_paths.append(arg)
                i += 1
            else:
                result['error'] = f'File not found: {arg}'
                return result

        if not file_paths:
            result['error'] = 'STT mode requires at least one audio/video file path'
            return result

        if result['params'].get('subtitle'):
            result['params']['overdose'] = True
        if result['params'].get('translate_langs'):
            pass
        elif result['params'].get('subtitle') and result['params'].get('translate'):
            result['error'] = 'STT subtitle cannot be used with bare translate. Use translate (source-target) for any-to-any with subtitle'
            return result
        elif result['params'].get('overdose') and result['params'].get('translate'):
            result['error'] = 'STT overdose cannot be used with bare translate. Use translate (source-target) for any-to-any with overdose'
            return result
        if result['params'].get('translate') and 'translate_langs' not in result['params']:
            result['params']['translate_langs'] = {'source': 'auto', 'target': 'en'}

        result['params']['files'] = file_paths
        result['params']['result_path'] = result_path
        return result

    if mode == 'se':
        file_paths = []
        se_sub = None
        se_blend = False
        se_video = False
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif arg_lower == 'video':
                se_video = True
                i += 1
            elif arg_lower == 'voice':
                if se_sub is None:
                    se_sub = 'voice'
                elif se_sub == 'sr':
                    se_sub = 'sr_voice'
                else:
                    result['error'] = 'Invalid SE syntax: unexpected voice keyword'
                    return result
                i += 1
            elif arg_lower == 'sr':
                if se_sub is not None:
                    result['error'] = 'Invalid SE syntax: sr already specified'
                    return result
                se_sub = 'sr'
                i += 1
            elif arg_lower == 'blend':
                if se_sub not in ('voice', 'sr_music', 'sr_voice'):
                    result['error'] = 'blend only valid after voice, sr music, or sr voice'
                    return result
                se_blend = True
                i += 1
            elif arg_lower == 'music':
                if se_sub == 'sr':
                    se_sub = 'sr_music'
                elif se_sub == 'sr_voice':
                    se_sub = 'sr_voice_music'
                else:
                    result['error'] = 'music keyword requires sr or sr voice prefix'
                    return result
                i += 1
            elif os.path.exists(arg) or is_youtube_url(arg):
                file_paths.append(arg)
                i += 1
            else:
                result['error'] = f'File not found: {arg}'
                return result

        if not file_paths:
            result['error'] = 'SE mode requires at least one audio/video file path'
            return result

        result['params']['files'] = file_paths
        result['params']['result_path'] = result_path
        result['params']['se_sub'] = se_sub
        result['params']['se_blend'] = se_blend
        result['params']['se_video'] = se_video
        return result

    if mode == 'svs':
        stem = None
        file_path = None
        svs_video = False
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif arg_lower == 'video':
                svs_video = True
                i += 1
            elif arg_lower in ('voice', 'music', 'both'):
                stem = arg_lower
                i += 1
            elif file_path is None and (os.path.exists(arg) or is_youtube_url(arg)):
                file_path = arg
                i += 1
            else:
                result['error'] = f'Invalid argument: {arg}'
                return result
        if stem is None:
            result['error'] = 'SVS mode requires stem: voice or music'
            return result
        if file_path is None:
            result['error'] = 'SVS mode requires an audio file path'
            return result
        result['params']['stem'] = stem
        result['params']['file_path'] = file_path
        result['params']['result_path'] = result_path
        result['params']['svs_video'] = svs_video
        return result

    if mode == 'ss':
        use_se = False
        file_path = None
        target_path = None
        use_overdose = False
        use_blend = False
        use_video = False
        speaker_num = None
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'se':
                use_se = True
                i += 1
            elif arg_lower == 'overdose':
                use_overdose = True
                i += 1
            elif arg_lower == 'blend':
                use_blend = True
                i += 1
            elif arg_lower == 'video':
                use_video = True
                i += 1
            elif arg_lower == 'target':
                if i + 1 < len(args):
                    target_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'target keyword requires a path argument'
                    return result
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            elif arg.isdigit() and speaker_num is None:
                speaker_num = int(arg)
                i += 1
            elif file_path is None and (os.path.exists(arg) or is_youtube_url(arg)):
                file_path = arg
                i += 1
            else:
                result['error'] = f'Invalid argument: {arg}'
                return result
        if file_path is None:
            result['error'] = 'SS mode requires an audio/video file path or URL'
            return result
        if speaker_num is not None:
            if speaker_num < 0:
                result['error'] = 'SS speaker number must be 0 or higher (0 resolves to 1)'
                return result
            if speaker_num == 0:
                speaker_num = 1
        result['params']['use_se'] = use_se
        result['params']['file_path'] = file_path
        result['params']['target_path'] = target_path
        result['params']['overdose'] = use_overdose
        result['params']['use_blend'] = use_blend
        result['params']['use_video'] = use_video
        result['params']['speaker_num'] = speaker_num
        result['params']['result_path'] = result_path
        return result

    if mode == 'sfx':
        prompt = None
        duration = None
        steps = None
        guide = None
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'sound':
                if i + 1 < len(args):
                    prompt = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'sound keyword requires a prompt argument'
                    return result
            elif arg_lower == 'duration':
                if i + 1 < len(args):
                    try:
                        duration = int(args[i + 1])
                        i += 2
                    except ValueError:
                        result['error'] = 'duration must be a number between 1 and 30'
                        return result
                else:
                    result['error'] = 'duration keyword requires a number argument'
                    return result
            elif arg_lower == 'steps':
                if i + 1 < len(args):
                    try:
                        steps = int(args[i + 1])
                        i += 2
                    except ValueError:
                        print("Warning: Invalid steps value, using default (30).")
                        steps = None
                        i += 2
                else:
                    print("Warning: steps keyword requires a number, using default (30).")
                    i += 1
            elif arg_lower == 'guide':
                if i + 1 < len(args):
                    try:
                        guide = float(args[i + 1])
                        i += 2
                    except ValueError:
                        print("Warning: Invalid guide value, using default (4.5).")
                        guide = None
                        i += 2
                else:
                    print("Warning: guide keyword requires a number, using default (4.5).")
                    i += 1
            elif arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                result['error'] = f'Unknown parameter: {arg}'
                return result

        if not prompt:
            result['error'] = 'SFX mode requires a sound prompt (sound "your prompt")'
            return result
        if duration is None:
            result['error'] = 'SFX mode requires a duration (duration <1-30>)'
            return result
        if duration < 1:
            result['error'] = 'Duration must be at least 1 second'
            return result

        use_steps = 30
        use_guide = 4.5
        if steps is not None:
            if 1 <= steps <= 100:
                use_steps = steps
            else:
                print("Warning: steps must be between 1-100, using default (30).")
        if guide is not None:
            guide = round(guide * 2) / 2
            if 1.0 <= guide <= 10.0:
                use_guide = guide
            else:
                print("Warning: guide must be between 1.0-10.0, using default (4.5).")

        result['params']['prompt'] = prompt
        result['params']['duration'] = duration
        result['params']['steps'] = use_steps
        result['params']['guide'] = use_guide
        result['params']['result_path'] = result_path
        return result

    if mode == 'train':
        sub_type = None
        voice_name = None
        ref_paths = []
        test_script = None
        has_test = False
        use_first = False
        use_extreme = False
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower.startswith('voice:'):
                sub_type = 'voice'
                voice_name = arg[6:].strip()
                i += 1
            elif arg_lower == 'extreme':
                use_extreme = True
                i += 1
            elif arg_lower == 'first':
                use_first = True
                i += 1
            elif arg_lower == 'test':
                has_test = True
                if i + 1 < len(args) and not args[i + 1].lower().startswith('voice:') and not args[i + 1].lower() == 'test' and args[i + 1].lower() != 'first':
                    test_script = args[i + 1]
                    i += 2
                else:
                    test_script = None
                    i += 1
            elif os.path.exists(arg) or is_youtube_url(arg):
                ref_paths.append(arg)
                i += 1
            else:
                ref_paths.append(arg)
                i += 1
        if sub_type != 'voice':
            result['error'] = 'Train mode requires voice: sub-type (e.g., train voice:character-name "path")'
            return result
        if not voice_name:
            result['error'] = 'Train mode requires a character name after voice: (e.g., voice:james)'
            return result
        if not ref_paths:
            result['error'] = 'Train mode requires at least one reference audio path'
            return result
        result['params']['sub_type'] = sub_type
        result['params']['voice_name'] = voice_name
        result['params']['ref_paths'] = ref_paths
        result['params']['has_test'] = has_test
        result['params']['test_script'] = test_script
        result['params']['use_first'] = use_first
        result['params']['extreme'] = use_extreme
        return result

    if mode == 'quest':
        quest_name = None
        quest_args = []
        result_path = None
        if i >= len(args):
            result['params']['list_quests'] = True
            return result
        quest_name = args[i].lower()
        i += 1
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                quest_args.append(arg)
                i += 1
        result['params']['quest_name'] = quest_name
        result['params']['quest_args'] = quest_args
        result['params']['result_path'] = result_path
        return result

    if mode == 'chains':
        chains_args = []
        result_path = None
        chains_subcmd = None
        if i < len(args) and args[i].lower() in ('build', 'load', 'comment', 'journey', 'decompile', 'compile'):
            chains_subcmd = args[i].lower()
            i += 1
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            if arg_lower == 'result':
                if i + 1 < len(args):
                    result_path = args[i + 1]
                    i += 2
                else:
                    result['error'] = 'result keyword requires a path argument'
                    return result
            else:
                chains_args.append(arg)
                i += 1
        result['params']['chains_args'] = chains_args
        result['params']['chains_subcmd'] = chains_subcmd
        result['params']['result_path'] = result_path
        return result

    while i < len(args):
        arg = args[i]
        arg_lower = arg.lower()
        if arg_lower == 'result':
            if i + 1 < len(args):
                result_path = args[i + 1]
                i += 2
            else:
                result['error'] = 'result keyword requires a path argument'
                return result
        elif arg_lower == 'overdose':
            result['params']['overdose'] = True
            i += 1
        elif arg_lower == 'extreme':
            if mode in ('tts', 'sts'):
                result['params']['extreme'] = True
            else:
                print("Warning: 'extreme' keyword is only valid in TTS and STS modes, ignoring")
            i += 1
        elif mode == 'ttm' and arg_lower == 'complete':
            result['params']['complete'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'lego':
            result['params']['lego'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'extract':
            result['params']['extract'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'bgm':
            if i + 1 >= len(args):
                result['error'] = 'bgm requires a source path (audio/video file or URL)'
                return result
            result['params']['bgm'] = True
            result['params']['bgm_source'] = args[i + 1]
            i += 2
        elif mode == 'ttm' and arg_lower == 'voice':
            if result['params'].get('use_music'):
                result['error'] = 'voice and music cannot be used together, use one or the other'
                return result
            if 'complete' in result['params'] or 'lego' in result['params']:
                result['params']['use_vocals'] = True
            else:
                result['params']['ttm_voice'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'music' and 'bgm' not in result['params']:
            if 'complete' not in result['params'] and 'lego' not in result['params']:
                result['error'] = 'music keyword is only valid with complete/lego/bgm task'
                return result
            if result['params'].get('use_vocals'):
                result['error'] = 'voice and music cannot be used together, use one or the other'
                return result
            result['params']['use_music'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'video':
            if 'complete' not in result['params'] and 'bgm' not in result['params']:
                result['error'] = 'video keyword is only valid with complete/bgm task'
                return result
            result['params']['want_video'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'noblend':
            if 'complete' not in result['params']:
                result['error'] = 'noblend keyword is only valid with complete task'
                return result
            result['params']['noblend'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'usrc':
            if 'complete' not in result['params']:
                result['error'] = 'usrc keyword is only valid with complete task'
                return result
            result['params']['use_source'] = True
            i += 1
        elif mode == 'ttm' and (arg.startswith('sfx:') or (arg.startswith('"sfx:') and arg.endswith('"'))):
            if 'bgm' not in result['params'] and 'complete' not in result['params']:
                result['error'] = '"sfx:" specs are only valid with bgm or complete task'
                return result
            if 'complete' in result['params'] and result['params'].get('noblend'):
                result['error'] = '"sfx:" cannot be used with noblend'
                return result
            if 'sfx_specs' not in result['params']:
                result['params']['sfx_specs'] = []
            sfx_val = arg.strip('"') if arg.startswith('"') and arg.endswith('"') else arg
            result['params']['sfx_specs'].append(sfx_val)
            i += 1
        elif mode == 'ttm' and arg_lower == 'stems':
            if 'extract' not in result['params']:
                result['error'] = 'stems keyword is only valid with extract task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'stems keyword requires instruments (e.g., stems "drums bass" or stems "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'add':
            if 'complete' not in result['params']:
                result['error'] = 'add keyword is only valid with complete task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'add keyword requires instruments (e.g., add "drums bass guitar" or add "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'reference':
            if ('complete' not in result['params'] and 'lego' not in result['params']
                and 'is_remix' not in result and 'is_repaint' not in result
                and 'bgm' not in result['params']):
                result['error'] = 'reference keyword is only valid with complete/lego/remix/repaint/bgm task'
                return result
            i += 1
            ref_entries = []
            while i < len(args):
                peek_lower = args[i].lower()
                if peek_lower in ('mix', 'blend', 'result', 'make', 'add', 'overdose',
                                  'complete', 'lego', 'video', 'extract', 'stems', 'only',
                                  'remix', 'repaint', 'bias', 'vc', 'clone', 'noblend', 'usrc',
                                  'script', 'lyrics', 'styling', 'base', 'target', 'duration',
                                  'timestamp', 'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr',
                                  'reference', 'sfx:', 'bgm', 'music'):
                    break
                if peek_lower in ('voice', 'music'):
                    sv_type = peek_lower
                    i += 1
                    if i >= len(args):
                        result['error'] = 'reference requires a path after voice/music'
                        return result
                    tr, rp, ref_stems = _parse_ref_time_spec(args[i])
                    ref_entries.append((sv_type, rp, tr, ref_stems))
                    i += 1
                else:
                    tr, rp, ref_stems = _parse_ref_time_spec(args[i])
                    ref_entries.append(('asis', rp, tr, ref_stems))
                    i += 1
            if not ref_entries:
                result['error'] = 'reference requires at least one path'
                return result
            if len(ref_entries) > 3:
                print("Warning: reference supports up to 3 entries, using first 3")
                ref_entries = ref_entries[:3]
            result['params']['ref_entries'] = ref_entries
        elif mode == 'ttm' and arg_lower == 'make':
            if 'lego' not in result['params']:
                result['error'] = 'make keyword is only valid with lego task'
                return result
            if i + 1 < len(args):
                result['params']['instruments_raw'] = args[i + 1]
                i += 2
            else:
                result['error'] = 'make keyword requires instruments (e.g., make "drums bass" or make "everything")'
                return result
        elif mode == 'ttm' and arg_lower == 'only':
            if 'extract' not in result['params']:
                result['error'] = 'only keyword is only valid with extract task'
                return result
            if result['params'].get('extract_mix'):
                result['error'] = 'only and mix cannot be used together, use one or the other'
                return result
            result['params']['extract_only'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'mix':
            if 'lego' not in result['params'] and 'extract' not in result['params']:
                result['error'] = 'mix keyword is only valid with lego/extract task'
                return result
            if result['params'].get('blend_mode'):
                result['error'] = 'mix and blend cannot be used together, use one or the other'
                return result
            if result['params'].get('extract_only'):
                result['error'] = 'only and mix cannot be used together, use one or the other'
                return result
            if 'extract' in result['params']:
                result['params']['extract_mix'] = True
            else:
                result['params']['mix_mode'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'blend':
            if 'lego' not in result['params']:
                result['error'] = 'blend keyword is only valid with lego task'
                return result
            if result['params'].get('mix_mode'):
                result['error'] = 'mix and blend cannot be used together, use one or the other'
                return result
            result['params']['blend_mode'] = True
            i += 1
        elif arg_lower in valid_keywords:
            current_keyword = arg_lower
            result['params'].setdefault(current_keyword, [])
            i += 1
        elif mode == 'sts' and arg_lower == 'music':
            if result['is_mimic']:
                result['error'] = 'music and mimic cannot be used together'
                return result
            result['is_music'] = True
            current_keyword = None
            i += 1
        elif mode == 'sts' and arg_lower == 'mimic':
            if result['is_music']:
                result['error'] = 'music and mimic cannot be used together'
                return result
            result['is_mimic'] = True
            current_keyword = None
            i += 1
        elif mode == 'sts' and arg_lower == 'nomusic':
            if result['is_music']:
                result['error'] = 'nomusic cannot be used with music'
                return result
            result['nomusic'] = True
            current_keyword = None
            i += 1
        elif mode == 'sts' and arg_lower == 'original':
            result['use_original'] = True
            current_keyword = None
            i += 1
        elif mode == 'tts' and arg_lower == 'slc':
            result['params']['slc'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                peek_lower = peek.lower()
                if peek_lower == 'translate':
                    result['params']['slc_translate'] = True
                    i += 2
                    if i < len(args):
                        lang_spec = _parse_lang_spec(args[i])
                        if lang_spec:
                            result['params']['slc_translate_langs'] = lang_spec
                            i += 1
                        else:
                            result['params']['slc_translate_langs'] = {'source': 'auto', 'target': 'en'}
                elif peek_lower != 'music' and peek_lower != 'overdose' and peek_lower not in valid_keywords and peek_lower != 'dub':
                    result['params']['slc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'tts' and arg_lower == 'music' and result['params'].get('slc'):
            result['params']['slc_music'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                if peek.lower() != 'overdose' and peek.lower() not in valid_keywords:
                    result['params']['slc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'tts' and arg_lower == 'svc':
            result['params']['svc'] = True
            if i + 1 < len(args):
                peek = args[i + 1]
                if peek.lower() != 'overdose' and peek.lower() not in valid_keywords:
                    result['params']['svc_path'] = peek
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        elif mode == 'tts' and arg_lower == 'se':
            result['params']['dub_se'] = True
            i += 1
        elif mode == 'tts' and arg_lower == 'dub':
            result['params']['dub'] = True
            result['params']['extreme'] = True
            i += 1
            while i < len(args):
                dub_arg = args[i]
                dub_lower = dub_arg.lower()
                if dub_lower == 'subtitle':
                    result['params']['dub_subtitle'] = True
                    i += 1
                    if i < len(args) and args[i].lower() == 'original':
                        result['params']['dub_subtitle_original'] = True
                        i += 1
                    if i < len(args):
                        lang_spec = _parse_lang_spec(args[i])
                        if lang_spec:
                            result['params']['dub_subtitle_langs'] = lang_spec
                            i += 1
                elif dub_lower == 'translate':
                    i += 1
                    if i < len(args):
                        lang_spec = _parse_lang_spec(args[i])
                        if lang_spec:
                            result['params']['dub_translate_langs'] = lang_spec
                            i += 1
                        else:
                            result['params']['dub_translate_langs'] = {'source': 'auto', 'target': 'en'}
                elif dub_lower == 'se':
                    result['params']['dub_se'] = True
                    i += 1
                elif dub_lower == 'video':
                    result['params']['dub_video'] = True
                    i += 1
                    if i < len(args) and not args[i].lower().startswith(('translate', 'subtitle', 'result', 'overdose', 'se')):
                        result['params']['dub_source'] = args[i]
                        i += 1
                elif dub_lower == 'result':
                    if i + 1 < len(args):
                        result_path = args[i + 1]
                        i += 2
                    else:
                        result['error'] = 'result keyword requires a path argument'
                        return result
                elif dub_lower == 'overdose':
                    result['params']['overdose'] = True
                    i += 1
                elif os.path.exists(dub_arg) or is_youtube_url(dub_arg):
                    if 'dub_source' not in result['params']:
                        result['params']['dub_source'] = dub_arg
                    i += 1
                else:
                    if 'dub_source' not in result['params']:
                        result['params']['dub_source'] = dub_arg
                    i += 1
            if 'dub_translate_langs' not in result['params']:
                result['params']['dub_translate_langs'] = {'source': 'auto', 'target': 'en'}
            break
        elif mode == 'ttm' and arg_lower == 'vc':
            result['params']['vc'] = True
            i += 1
        elif mode == 'ttm' and arg_lower == 'clone':
            if i + 1 >= len(args):
                result['error'] = 'clone requires a source path'
                return result
            _ci = i + 1
            if args[_ci].lower() == 'first':
                result['clone_first'] = True
                _ci += 1
                if _ci >= len(args):
                    result['error'] = 'clone requires a source path after first'
                    return result
            result['clone_path'] = args[_ci]
            i = _ci + 1
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'remix':
            result['is_remix'] = True
            i += 1
            remix_entries = []
            while i < len(args):
                peek_lower = args[i].lower()
                if peek_lower in ('script', 'lyrics', 'styling', 'base', 'target', 'duration', 'timestamp',
                                  'dialogue', 'sound', 'steps', 'guide', 'level', 'ocr',
                                  'complete', 'lego', 'video', 'extract', 'stems', 'only',
                                  'noblend', 'usrc', 'remix', 'repaint', 'bias', 'vc', 'clone',
                                  'reference', 'sfx:', 'add', 'make', 'mix', 'blend', 'result', 'overdose',
                                  'music', 'bgm'):
                    break
                if peek_lower in ('voice', 'music'):
                    sv_type = peek_lower
                    i += 1
                    if i >= len(args):
                        result['error'] = 'remix source requires a path after voice/music'
                        return result
                    _r_tr, _r_rp, _r_stems = _parse_ref_time_spec(args[i])
                    remix_entries.append((sv_type, _r_rp, _r_tr, _r_stems))
                    i += 1
                else:
                    _r_tr, _r_rp, _r_stems = _parse_ref_time_spec(args[i])
                    remix_entries.append(('asis', _r_rp, _r_tr, _r_stems))
                    i += 1
            if not remix_entries:
                result['error'] = 'remix requires at least one source path'
                return result
            if len(remix_entries) > 3:
                print("Warning: remix supports up to 3 sources, using first 3")
                remix_entries = remix_entries[:3]
            result['remix_entries'] = remix_entries
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'bias':
            if i + 1 >= len(args):
                result['error'] = 'bias requires a value'
                return result
            result['bias_val'] = args[i + 1]
            i += 2
            current_keyword = None
        elif mode == 'ttm' and arg_lower == 'repaint':
            if i + 1 >= len(args):
                result['error'] = 'repaint requires a source path'
                return result
            result['is_repaint'] = True
            _rp_src_idx = i + 1
            if args[_rp_src_idx].lower() in ('voice', 'music'):
                result['repaint_source_prefix'] = args[_rp_src_idx].lower()
                _rp_src_idx += 1
                if _rp_src_idx >= len(args):
                    result['error'] = f'repaint requires a source path after {result["repaint_source_prefix"]}'
                    return result
            result['repaint_path'] = args[_rp_src_idx]
            i = _rp_src_idx + 1
            _mp_pattern = re.compile(r'^\d+\.?\d*-\d+\.?\d*/')
            if i < len(args) and _mp_pattern.match(args[i]):
                result['repaint_multipass'] = []
                while i < len(args) and _mp_pattern.match(args[i]):
                    result['repaint_multipass'].append(args[i])
                    i += 1
            current_keyword = None
        elif mode == 'ttm' and arg.startswith('time:'):
            result['time_range'] = arg[5:]
            i += 1
        elif current_keyword is not None and arg_lower == 'first':
            result['use_first'] = True
            i += 1
        elif current_keyword is not None:
            try:
                duration_val = int(arg)
                remaining = args[i+1:]
                is_duration = all(is_num(x) for x in remaining)
                if is_duration:
                    result['params']['duration'] = duration_val
                elif current_keyword == 'duration':
                    result['params']['duration'] = duration_val
                else:
                    result['params'][current_keyword].append(arg)
                i += 1
            except ValueError:
                result['params'][current_keyword].append(arg)
                i += 1
        else:
                if mode == 'ttm' and (result['params'].get('complete') or result['params'].get('lego')
                                   or result['params'].get('extract')):
                    if not result['params'].get('task_source_args'):
                        result['params']['task_source_args'] = [arg]
                        i += 1
                    else:
                        if mode == 'sts':
                            result['error'] = f'invalid parameter "{arg}" only next parameter should be music or mimic or empty'
                        else:
                            result['error'] = f'Unknown parameter: {arg}'
                        return result
                else:
                    try:
                        duration = int(arg)
                        result['params']['duration'] = duration
                        i += 1
                    except ValueError:
                        if mode == 'sts':
                            result['error'] = f'invalid parameter "{arg}" only next parameter should be music or mimic or empty'
                        else:
                            result['error'] = f'Unknown parameter: {arg}'
                        return result

    result['params']['result_path'] = result_path
    if mode == 'ttm' and 'clone_path' in result and not result['params'].get('vc'):
        print("Warning: 'clone' specified without 'vc' flag — clone will be ignored. Use 'vc' to enable voice cloning in TTM mode.")
    return result

def is_num(s):
    try:
        int(s)
        return True
    except (ValueError, TypeError):
        return False

VOICES_DIR = os.path.join(os.getcwd(), "voices")

def _ensure_voices_dir():
    os.makedirs(VOICES_DIR, exist_ok=True)

def _save_voice_prompt(voice_prompt_items, character_name):
    _ensure_voices_dir()
    character_name = character_name.lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"voder_tts_{character_name}_{timestamp}.tts"
    filepath = os.path.join(VOICES_DIR, filename)
    import torch
    from dataclasses import asdict
    payload = {"items": [asdict(it) for it in voice_prompt_items], "character": character_name, "timestamp": timestamp}
    torch.save(payload, filepath)
    return filepath

def _load_voice_prompt(filepath):
    import torch
    from qwen_tts import VoiceClonePromptItem
    if not os.path.exists(filepath):
        return None
    payload = torch.load(filepath, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "items" not in payload:
        return None
    items_raw = payload["items"]
    if not isinstance(items_raw, list) or len(items_raw) == 0:
        return None
    items = []
    for d in items_raw:
        if not isinstance(d, dict):
            return None
        ref_code = d.get("ref_code", None)
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)
        ref_spk = d.get("ref_spk_embedding", None)
        if ref_spk is None:
            return None
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)
        items.append(VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=ref_spk,
            x_vector_only_mode=bool(d.get("x_vector_only_mode", False)),
            icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode", False)))),
            ref_text=d.get("ref_text", None),
        ))
    return items

def _find_voice_file(name):
    _ensure_voices_dir()
    if os.path.exists(name) and name.endswith('.tts'):
        return name
    name = name.lower()
    matches = []
    for f in os.listdir(VOICES_DIR):
        if not f.endswith('.tts'):
            continue
        if f.startswith(f"voder_tts_{name}_"):
            matches.append(os.path.join(VOICES_DIR, f))
    if not matches:
        return None
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0]

def _resolve_voice_ref(value):
    if ':' not in value:
        voice_file = _find_voice_file(value.lower())
        if voice_file:
            return voice_file
        return None
    parts = value.split(':', 1)
    first = parts[0].strip()
    second = parts[1].strip().lower()
    if second.endswith('.tts'):
        if os.path.exists(second):
            return second
        voice_file = _find_voice_file(second)
        if voice_file:
            return voice_file
        return None
    voice_file = _find_voice_file(second)
    if voice_file:
        return voice_file
    return None

def _is_trained_voice_ref(value):
    if os.path.exists(value) and value.endswith('.tts'):
        return True
    if _find_voice_file(value.lower()) is not None:
        return True
    if ':' in value:
        parts = value.split(':', 1)
        second = parts[1].strip()
        if second.endswith('.tts'):
            if os.path.exists(second):
                return True
            if _find_voice_file(second.lower()) is not None:
                return True
        if _find_voice_file(second.lower()) is not None:
            return True
    return False

def _save_fish_voice(encoded_refs, character_name, ref_text=None):
    _ensure_voices_dir()
    character_name = character_name.lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"voder_ttse_{character_name}_{timestamp}.ttse"
    filepath = os.path.join(VOICES_DIR, filename)
    import torch
    payload = {
        "tokens": encoded_refs["tokens"],
        "text": ref_text or encoded_refs.get("text", ""),
        "character": character_name,
        "timestamp": timestamp
    }
    torch.save(payload, filepath)
    return filepath

def _load_fish_voice(filepath):
    import torch
    if not os.path.exists(filepath):
        return None
    payload = torch.load(filepath, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "tokens" not in payload:
        return None
    return payload

def _find_fish_voice_file(name):
    _ensure_voices_dir()
    if os.path.exists(name) and name.endswith('.ttse'):
        return name
    name = name.lower()
    matches = []
    for f in os.listdir(VOICES_DIR):
        if not f.endswith('.ttse'):
            continue
        if f.startswith(f"voder_ttse_{name}_"):
            matches.append(os.path.join(VOICES_DIR, f))
    if not matches:
        return None
    matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return matches[0]

def _is_fish_voice_ref(value):
    if os.path.exists(value) and value.endswith('.ttse'):
        return True
    if _find_fish_voice_file(value.lower()) is not None:
        return True
    if ':' in value:
        parts = value.split(':', 1)
        second = parts[1].strip()
        if second.endswith('.ttse'):
            if os.path.exists(second):
                return True
            if _find_fish_voice_file(second.lower()) is not None:
                return True
        if _find_fish_voice_file(second.lower()) is not None:
            return True
    return False

def _resolve_fish_voice_ref(value):
    if ':' not in value:
        voice_file = _find_fish_voice_file(value.lower())
        if voice_file:
            return voice_file
        return None
    parts = value.split(':', 1)
    first = parts[0].strip()
    second = parts[1].strip().lower()
    if second.endswith('.ttse'):
        if os.path.exists(second):
            return second
        voice_file = _find_fish_voice_file(second)
        if voice_file:
            return voice_file
        return None
    voice_file = _find_fish_voice_file(second)
    if voice_file:
        return voice_file
    return None

def _check_voice_extreme_mismatch(voice_path, use_extreme):
    if not voice_path:
        return False
    basename = os.path.basename(voice_path)
    if use_extreme and basename.endswith('.tts'):
        print("Error: .tts voice files are for standard TTS mode, .ttse files are for extreme mode. Use 'voder.py train extreme' to create a .ttse file.")
        return True
    if not use_extreme and basename.endswith('.ttse'):
        print("Error: .ttse voice files are for extreme mode, .tts files are for standard TTS mode. Remove 'extreme' keyword or use a .tts file instead.")
        return True
    return False

def validate_oneline_mode(mode_name):
    if mode_name.lower() == 'overdose':
        return 'overdose'
    valid_modes = ['tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train', 'quest', 'chains', 'vadar', 'overdose']
    if mode_name.lower() in valid_modes:
        return mode_name.lower()
    return None

def show_oneline_usage():
    print("VODER One-Line Command Usage:")
    print("=" * 60)
    print()
    print("Available modes:")
    print("  tts      - Text-to-Speech")
    print("  sts      - Speech-to-Speech (Voice Conversion)")
    print("  ttm      - Text-to-Music (use 'vc' flag for voice cloning)")
    print("  stt      - Speech-to-Text (Transcription with optional diarization)")
    print("  se       - Sound Enhancement (denoise, dereverb, restore, super-resolution)")
    print("  sfx      - Sound Effects (text prompt + duration → audio)")
    print("  svs      - Song Voice Separate (extract vocals/music from song)")
    print("  ss       - Speakers Separator (extract all speakers, or a specific one by number)")
    print("  train    - Train and save voice clones")
    print("  quest    - Side-quests (utility tasks): download + Media Manipulation (Sound Effects / Audio Editing / Format & File). Run `quest` with no args to list them all as a tree.")
    print("  chains   - Chain multiple voder tasks: each chain's output feeds later chains")
    print("  vadar    - VADAR agent: give it a natural-language request and it decides what to run")
    print()
    print("Train examples:")
    print('  python voder.py train voice:james "ref1.wav" "ref2.wav"')
    print('  python voder.py train voice:sarah "ref.wav" test')
    print('  python voder.py train voice:narrator "ref.wav" test "Custom test script here"')
    print()
    print("SVS examples (Song Voice Separate):")
    print('  python voder.py svs voice "path/to/song.mp3"')
    print('  python voder.py svs music "path/to/song.mp3"')
    print('  python voder.py svs voice "path/to/song.mp3" result "output.wav"')
    print('  python voder.py svs music "path/to/song.mp3" result "output.wav"')
    print('  python voder.py svs voice video "https://youtube.com/watch?v=..."')
    print()
    print("SS examples (Speakers Separator):")
    print('  python voder.py ss "path/to/audio.wav"')
    print('  python voder.py ss 1 "path/to/audio.wav"')
    print('  python voder.py ss 999 "path/to/video.mp4"')
    print('  python voder.py ss 1 "https://youtube.com/watch?v=..."')
    print('  python voder.py ss se 1 "path/to/audio.wav"')
    print('  python voder.py ss overdose 1 se blend "path/to/audio.wav"')
    print('  python voder.py ss overdose 3 "path/to/audio.wav"')
    print('  python voder.py ss target "ref.wav" blend "path/to/audio.wav"')
    print('  python voder.py ss video 1 "path/to/video.mp4"')
    print('  python voder.py ss target "ref.wav" video "path/to/video.mp4"')
    print()
    print("Single mode examples:")
    print('  python voder.py tts script "hello world" voice "male voice"')
    print('  python voder.py tts script "hello" target "voice.wav"')
    print('  python voder.py tts ocr "path/to/image.png" voice "text: female voice"')
    print('  python voder.py tts ocr "path/to/image.png" target "text: voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav"')
    print('  python voder.py sts base "input.wav" target "voice.wav" music')
    print('  python voder.py ttm lyrics "song" styling "pop" 30')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target voice "ref.wav"')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target music "ref.wav"')
    print('  python voder.py ttm lyrics "song" styling "pop" 30 target voice "https://youtu.be/..."')
    print('  python voder.py ttm vc lyrics "song" styling "pop" 30 clone "voice.wav"')
    print()
    print("TTM voice examples (generate song then extract vocals):")
    print('  python voder.py ttm voice lyrics "song" styling "pop" 30')
    print('  python voder.py ttm voice lyrics "song" styling "rock" 30 target voice "ref.wav"')
    print()
    print("STT examples (Speech-to-Text transcription):")
    print('  python voder.py stt "path/to/audio.wav"')
    print('  python voder.py stt "audio1.wav" "audio2.wav"')
    print('  python voder.py stt "audio.wav" timestamp')
    print('  python voder.py stt "audio.wav" dialogue')
    print('  python voder.py stt "audio.wav" timestamp dialogue')
    print('  python voder.py stt "audio.wav" translate')
    print('  python voder.py stt "audio.wav" translate dialogue')
    print('  python voder.py stt "audio.wav" translate timestamp dialogue')
    print('  python voder.py stt "https://youtube.com/watch?v=..."')
    print('  python voder.py stt overdose subtitle "video.mp4"')
    print('  python voder.py stt overdose subtitle se "noisy_video.mp4"')
    print('  python voder.py stt overdose subtitle "https://youtube.com/watch?v=..."')
    print('  python voder.py stt translate "(auto-en)" "audio.wav"')
    print('  python voder.py stt translate "(auto-ar)" "audio.wav"')
    print('  python voder.py stt translate "(ja-en)" "audio.wav" dialogue')
    print('  python voder.py stt overdose translate "(auto-fr)" "audio.wav"')
    print('  python voder.py stt overdose subtitle translate "(auto-ar)" "video.mp4"')
    print()
    print("SE examples (Sound Enhancement):")
    print('  python voder.py se "path/to/audio.wav"')
    print('  python voder.py se "audio1.wav" "audio2.wav"')
    print('  python voder.py se "path/to/video.mp4"')
    print('  python voder.py se voice "path/to/audio.wav"')
    print('  python voder.py se voice blend "path/to/audio.wav"')
    print('  python voder.py se sr "path/to/audio.wav"')
    print('  python voder.py se sr music "path/to/audio.wav"')
    print('  python voder.py se sr music blend "path/to/audio.wav"')
    print('  python voder.py se sr voice "path/to/audio.wav"')
    print('  python voder.py se sr voice blend "path/to/audio.wav"')
    print('  python voder.py se sr voice music "path/to/audio.wav"')
    print('  python voder.py se "https://youtube.com/watch?v=..."')
    print('  python voder.py se voice video "https://youtube.com/watch?v=..."')
    print()
    print("SFX examples (Sound Effects Generation):")
    print('  python voder.py sfx sound "thunder cracking" duration 5')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 result "output.wav"')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 steps 50')
    print('  python voder.py sfx sound "rain on a tin roof" duration 10 steps 50 guide 3.5 result "output.wav"')
    print()
    print("Dialogue mode examples:")
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano"')
    print('  python voder.py tts script "James: Hello" script "sfx: thunder /duration:3" voice "James: deep male" music "soft piano" level "10:20-50"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "soft piano" reference "ref_song.mp3"')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" music "epic orchestral" reference "https://youtube.com/watch?v=..."')
    print('  python voder.py tts script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female" music "chill lo-fi" reference "ref_video.mp4"')
    print('  python voder.py tts overdose script "James: Hello" script "Sarah: Hi" voice "James: deep male" voice "Sarah: cheerful female"')
    print('  python voder.py tts overdose script "James: Hello" script "Sarah: Hi" target "James: james.wav" target "Sarah: sarah.wav" music "soft piano"')
    print()
    print("Parameters (can appear multiple times):")
    print("  script   - Dialogue line in 'Character: text' format, or plain text for single mode")
    print("  voice    - Voice prompt in 'Character: description' format (TTS)")
    print("  target   - Audio file path in 'Character: path' format (voice clone) or single path (STS)")
    print("  lyrics   - Song lyrics for TTM / remix (single, optional for remix)")
    print("  styling  - Style prompt for TTM (single)")
    print("  base     - Base audio/video path")
    print("  music    - Music flag for STS mode (uses 44.1kHz v1 model)")
    print("  timestamp - Keep Whisper timestamps in output (STT mode)")
    print("  dialogue - Enable speaker diarization (STT mode)")
    print("  translate - Translate to English (STT mode, uses large-v3 model), or translate (source-target) for any-to-any via TranslateGemma")
    print("  sound    - Sound effect prompt (SFX mode)")
    print("  duration - Duration in seconds (10-300 for TTM, 1-30 for SFX)")
    print("  steps    - Inference steps (1-100, SFX mode, default: 30)")
    print("  guide    - Guidance scale (1.0-10.0, SFX mode, default: 4.5)")
    print("  music    - Background music description (dialogue/bgm modes)")
    print("  level    - Music volume levels e.g. \"10:20-50 30:60-80\" (dialogue modes) or 0-100 (bgm mode, default: 35)")
    print("  reference - Music reference audio/video path or URL (up to 3 with voice/music prefix, optional time spec: \"start(ref)\" or \"start-end/ref2(ref)\")")
    print("  ocr      - Image file path for OCR text extraction (TTS modes)")
    print("  overdose - Use VibeVoice ASR for dialogue source and enhanced music (TTS/TTM modes)")
    print("  bgm      - Add or replace background music on an audio/video (TTM mode)")
    print("  voice    - Extract vocals only from TTM output (TTM mode), or isolate voice (complete/lego)")
    print("  usrc     - Blend with original source instead of isolated voice/music (complete)")
    print('  "sfx:"   - Sound effect spec for bgm/complete tasks: "sfx:prompt/duration-position/level"')
    print("  video    - Download URL as video (and output MP4) for SE / SVS / SS / TTS dub / TTM complete / TTM bgm (default for URL is audio download); also used by quest download")
    print("  <number> - Duration in seconds (10-300, for TTM modes)")
    print()
    print("Side-Quests (utility tasks): grouped by category in the listing.")
    print("  Run `python voder.py quest` (no args) to see all available side-quests as a tree.")
    print("  Standalone:                              download (URL fetch as audio or video)")
    print("  Media Manipulation:")
    print("    Sound Effects:                         bassboost, fade, loudnorm, pitch, reverb, soundlevel, speed")
    print("    Audio Editing:                         cut, merge, remove, reverse, silence")
    print("    Format & File:                         compress, convert, glue, noframes")
    print()
    print("  quest download \"<url>\"            - Download a URL as audio (default) → results/")
    print("  quest download video \"<url>\"      - Download a URL as video → results/")
    print("  quest remove \"10-20\" \"song.wav\"   - Inverse of cut: drop ranges, keep the rest")
    print("  quest soundlevel 2.00 \"song.wav\"  - Linear level multiplier (1.00 = original, 0.25 = 25%, 10.00 = 10×)")
    print("  quest loudnorm \"episode.wav\"      - Auto perceptual loudness to -16 LUFS broadcast standard")
    print()
    print("Chains (pipeline of voder tasks; later chains can reference earlier chain names as input paths):")
    print('  chains "name1" <voder command...> / "name2" <voder command that references "name1"> / ...')
    print("  Intermediate chain outputs are stored in temp_chains/ — only the last chain's output reaches results/.")
    print("  Empty chains are skipped (their names remain available for reuse); duplicate names are an error.")
    print()
    print("TTS SLC examples (Speaker Language Conversion):")
    print('  python voder.py tts slc "path/to/audio.wav"')
    print('  python voder.py tts slc music "path/to/audio.wav"')
    print('  python voder.py tts slc "path/to/audio.wav" target "voice_ref.wav"')
    print('  python voder.py tts overdose slc "path/to/audio.wav"')
    print('  python voder.py tts overdose slc music "path/to/audio.wav"')
    print('  python voder.py tts slc translate "(auto-ar)" "path/to/audio.wav"')
    print('  python voder.py tts slc translate "(en-ja)" "path/to/audio.wav"')
    print()
    print("TTS dub examples (Video/Audio Dubbing):")
    print('  python voder.py tts dub "video.mp4"')
    print('  python voder.py tts dub subtitle "video.mp4"')
    print('  python voder.py tts dub subtitle original "video.mp4"')
    print('  python voder.py tts dub translate "(auto-ar)" "video.mp4"')
    print('  python voder.py tts dub translate "(auto-ar)" subtitle "video.mp4"')
    print('  python voder.py tts dub "audio.wav"')
    print('  python voder.py tts dub translate "(en-ja)" "audio.wav"')
    print('  python voder.py tts dub "https://youtube.com/watch?v=..."')
    print('  python voder.py tts dub video "https://youtube.com/watch?v=..."')
    print()
    print("TTS SVC examples (Speaker Voice Change):")
    print('  python voder.py tts svc "speech.wav" target "voice_ref.wav"')
    print('  python voder.py tts svc "speech.wav" voice "deep male, authoritative"')
    print('  python voder.py tts overdose svc "speech.wav" target "voice.wav"')
    print('  python voder.py tts svc "speech.wav" target "sts:voice_ref.wav"')
    print('  python voder.py tts svc "speech.wav" target "(ref1.wav)(ref2.wav)(ref3.wav)"')
    print('  python voder.py tts svc "speech.wav" target "sts:(ref1.wav)(ref2.wav)"')
    print()
    print("BGM examples (add/replace background music on audio or video):")
    print('  python voder.py ttm bgm "path/to/audio.wav" music "soft piano"')
    print('  python voder.py ttm bgm "path/to/video.mp4" music "epic orchestral" level 50')
    print('  python voder.py ttm overdose bgm "path/to/audio.wav" music "lo-fi chill" level 25 reference "ref_song.mp3"')
    print('  python voder.py ttm bgm "https://youtube.com/watch?v=..." music "ambient synth" level 40')
    print('  python voder.py ttm bgm video "https://youtube.com/watch?v=..." music "cinematic" level 30 reference "ref.mp3"')
    print('  python voder.py ttm bgm "audio.wav" music "piano" "sfx:thunder/10-5/50"')
    print('  python voder.py ttm bgm "audio.wav" "sfx:rain/8-22" "sfx:thunder/10-5/60"')
    print('  python voder.py ttm bgm "audio.wav" music "piano" reference "30-60(ref.wav)"')
    print()
    print("Complete + SFX examples (add instruments and/or sound effects):")
    print('  python voder.py ttm complete "source.wav" add "drums bass" "sfx:thunder/10-5/50"')
    print('  python voder.py ttm complete "source.wav" "sfx:rain/8-22"')
    print('  python voder.py ttm complete video "source.mp4" add "everything" "sfx:boom/12-30/40"')
    print()
    print("Complete with voice/music isolation examples:")
    print('  python voder.py ttm complete voice "song.wav" add "drums bass"')
    print('  python voder.py ttm complete music "song.wav" add "everything"')
    print('  python voder.py ttm complete voice usrc "song.wav" add "drums bass guitar"')
    print('  python voder.py ttm complete music usrc "song.wav" add "everything"')
    print('  python voder.py ttm complete voice "podcast.wav" "sfx:bell/5-10/40"')
    print()
    print("Remix examples (remix a song with new style and optional lyrics):")
    print('  python voder.py ttm remix "song.wav" styling "electronic dance"')
    print('  python voder.py ttm remix "song.wav" lyrics "new words here" styling "pop rock"')
    print('  python voder.py ttm remix voice "song.wav" styling "lo-fi chill"')
    print('  python voder.py ttm remix music "song.wav" lyrics "verse lyrics" styling "jazz"')
    print('  python voder.py ttm remix "song.wav" lyrics "custom lyrics" styling "hip hop" bias 70')
    print('  python voder.py ttm remix "song.wav" styling "ambient" reference "ref_song.mp3"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference voice "ref1.wav" music "ref2.wav"')
    print('  python voder.py ttm remix voice "vocal.wav" music "inst.wav" styling "funk" reference "ref.wav"')
    print('  python voder.py ttm overdose remix "song.wav" lyrics "dreamy verse" styling "synthwave"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference "30-60(ref.wav)"')
    print('  python voder.py ttm remix "song.wav" styling "pop" reference voice "0-15/40-55(ref.wav)" music "ref2.wav"')
    print()
    print("Repaint examples (restyle a specific time range of a song):")
    print('  python voder.py ttm repaint "song.wav" time:20-80 styling "more energetic"')
    print('  python voder.py ttm repaint "song.wav" time:20-80 styling "orchestral" bias 80 reference "ref.wav"')
    print('  python voder.py ttm overdose repaint "song.wav" time:20-80 styling "jazz" reference voice "ref.wav"')
    print('  python voder.py ttm repaint voice "song.wav" time:20-80 styling "funk"')
    print('  python voder.py ttm repaint music "song.wav" time:20-80 styling "ambient"')
    print()
    print("Multi-pass repaint examples (multiple edits, each pass builds on the previous):")
    print('  python voder.py ttm repaint "song.wav" "20-80/styling(orchestral)" "10-30/styling(jazz)/bias/70"')
    print('  python voder.py ttm repaint "song.wav" "0-30/styling(funk)/lyrics(new words\\nhere)" "15-30/styling(ambient)/reference(ref.wav)"')
    print('  python voder.py ttm overdose repaint "song.wav" "0-15/styling(lo-fi)" "10-25/styling(drum and bass)/bias/80/reference-voice(vocals.wav)"')
    print('  python voder.py ttm repaint "song.wav" "20-80/styling(jazz)/reference-voice(30-60(vocals.wav))"')
    print('  python voder.py ttm repaint music "song.wav" "0-30/styling(chill)" "20-30/styling(epic)/reference-music(inst.wav)"')
    print()
    print('SFX spec format: "sfx:prompt/duration-position/level"')
    print("  prompt    - SFX description text (required)")
    print("  duration  - SFX length 5-30 seconds (clamped, auto-cut if exceeds source)")
    print("  position  - Place at N seconds into source (required, cannot exceed source length)")
    print("  level     - Volume 1-100% (optional, default: 50)")
    print("  Multiple SFX specs can be specified")
    print()
    print("Script directives (per line, at end of text):")
    print("  /time:nn-nn+nn  - Cut nn seconds from end (-nn) and/or start (+nn)")
    print("  /level:0-100     - Volume level for that line (default: 100)")
    print("  /duration:1-30    - SFX duration (required for sfx: lines)")
    print("  sfx: prompt      - Special character: generates SFX via TangoFlux")
    print()
    print("Side-Quest examples (utility tasks; run `quest` with no args to list them all):")
    print('  python voder.py quest download "https://youtube.com/watch?v=..."          # audio download (default)')
    print('  python voder.py quest download video "https://youtube.com/watch?v=..."   # video download')
    print('  python voder.py quest remove "10-20" "30-35" "song.wav"                    # drop ranges, keep the rest')
    print('  python voder.py quest soundlevel 2.00 "song.wav"                          # 2× louder (1.00 = original, 10.00 = max)')
    print('  python voder.py quest loudnorm "podcast.wav"                              # auto-normalize to -16 LUFS broadcast standard')
    print()
    print("Chain examples (pipeline of voder tasks; intermediate outputs stay in temp_chains/, only the last chain reaches results/):")
    print('  python voder.py chains "vocals" svs voice "song.wav" / "enhanced" se voice "vocals"')
    print('  python voder.py chains "song" ttm lyrics "la la la" styling "pop" 30 / "voice" svs voice "song" / "cover" sts base "voice" target "ref.wav"')
    print('  python voder.py chains "1" tts script "hi" voice "male" / "2" se "1" / "3" stt "2" timestamp')
    print('  # Empty chains are skipped and their names remain available for reuse; duplicate names are an error.')
    print('  # Use " / " (space slash space) to separate chains. The last non-empty chain\'s output is exported to results/.')

def execute_oneline_command(parsed):
    mode = parsed['mode']
    params = parsed['params']
    if 'is_music' in parsed:
        params['is_music'] = parsed['is_music']
    if 'is_mimic' in parsed:
        params['is_mimic'] = parsed['is_mimic']
    if 'nomusic' in parsed:
        params['nomusic'] = parsed['nomusic']
    if 'is_remix' in parsed:
        params['is_remix'] = parsed['is_remix']
    if 'remix_entries' in parsed:
        params['remix_entries'] = parsed['remix_entries']
    if 'bias_val' in parsed:
        params['bias_val'] = parsed['bias_val']
    if 'is_repaint' in parsed:
        params['is_repaint'] = parsed['is_repaint']
    if 'repaint_path' in parsed:
        params['repaint_path'] = parsed['repaint_path']
    if 'repaint_source_prefix' in parsed:
        params['repaint_source_prefix'] = parsed['repaint_source_prefix']
    if 'repaint_multipass' in parsed:
        params['repaint_multipass'] = parsed['repaint_multipass']
    if 'time_range' in parsed:
        params['time_range'] = parsed['time_range']
    if 'clone_path' in parsed:
        params['clone_path'] = parsed['clone_path']
    if 'use_first' in parsed:
        params['use_first'] = parsed['use_first']
    if 'clone_first' in parsed:
        params['clone_first'] = parsed['clone_first']

    success = False
    if mode == 'tts':
        success = oneline_tts(params)
    elif mode == 'sts':
        success = oneline_sts(params)
    elif mode == 'ttm':
        if params.get('bgm'):
            success = oneline_ttm_bgm(params)
        elif params.get('complete'):
            success = oneline_ttm_complete(params)
        elif params.get('lego'):
            success = oneline_ttm_lego(params)
        elif params.get('extract'):
            success = oneline_ttm_extract(params)
        elif params.get('ttm_voice'):
            success = oneline_ttm_voice(params)
        else:
            success = oneline_ttm(params)
    elif mode == 'stt':
        if params.get('subtitle'):
            success = oneline_stt_subtitle(params)
        else:
            success = oneline_stt(params)
    elif mode == 'se':
        success = oneline_se(params)
    elif mode == 'sfx':
        success = oneline_sfx(params)
    elif mode == 'svs':
        success = oneline_svs(params)
    elif mode == 'ss':
        success = oneline_ss(params)
    elif mode == 'train':
        success = oneline_train(params)
    elif mode == 'quest':
        success = oneline_quest(params)
    elif mode == 'chains':
        success = oneline_chains(params)
    elif mode == 'vadar':
        from voders.vadars.vadar import run_vadar_oneline
        success = run_vadar_oneline(params.get('prompt', ''), result_path=params.get('result_path'), use_lite=params.get('use_lite', True))
    else:
        print(f"Error: Unknown mode '{mode}'")
        show_oneline_usage()
        return False

    if success and params.get('result_path'):
        copy_result_to_path(params['result_path'])

    if mode != 'vadar':
        try:
            organize_results()
        except Exception:
            pass

    return success

def copy_result_to_path(result_path):
    if result_path is None:
        return
    try:
        results_dir = os.path.join(os.getcwd(), "results")
        if not os.path.exists(results_dir):
            return
        files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if os.path.isfile(os.path.join(results_dir, f))]
        if not files:
            return
        latest_file = max(files, key=os.path.getmtime)

        result_dir = os.path.dirname(result_path)
        result_filename = os.path.basename(result_path)

        if not result_dir:
            destination = os.path.join(".", result_filename)
            os.makedirs(".", exist_ok=True)
        else:
            os.makedirs(result_dir, exist_ok=True)
            destination = result_path

        shutil.copy2(latest_file, destination)
        print(f"Result copied to: {destination}")
    except Exception as e:
        print(f"Note: Could not copy to result path: {e}")

TRAIN_TEST_SCRIPT = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor. In the distance, you can hear the distant rumble of thunder rolling across the wide open plains, signaling that a summer storm is approaching fast."

def oneline_train(params):
    sub_type = params.get('sub_type')
    voice_name = params.get('voice_name', '').lower()
    ref_paths = params.get('ref_paths', [])
    has_test = params.get('has_test', False)
    test_script = params.get('test_script')
    use_first = params.get('use_first', False)
    use_extreme = params.get('extreme', False)

    if sub_type != 'voice':
        print("Error: Only 'voice' training is supported")
        return False

    _cleanup = []
    try:
        print(f"Training voice '{voice_name}' from {len(ref_paths)} reference(s)...")
        clean_vocal = None
        if len(ref_paths) > 1:
            clean_vocal = _resolve_multi_refs(ref_paths, _cleanup, use_first=use_first)
            if not clean_vocal:
                print("Error: Failed to resolve reference audios")
                return False
        else:
            if use_first:
                print("Warning: 'first' keyword ignored (only one reference provided)")
            resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
            if not resolved_audio:
                print(f"Error: Failed to resolve reference: {ref_paths[0]}")
                return False
            _cleanup.extend(_cl)
            clean_vocal = svs_extract_vocals(resolved_audio)
            if clean_vocal and clean_vocal != resolved_audio:
                _cleanup.append(clean_vocal)
            if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                _cleanup.append(resolved_audio)

        if use_extreme:
            print("Loading Fish-S2Pro model (extreme)...")
            fish_tts = FishTTS()
            if not fish_tts.ensure_model():
                print("Error: Failed to load Fish-S2Pro model")
                return False
            print("Transcribing voice reference...")
            ref_text = _transcribe_for_fish_ref(clean_vocal)
            print("Encoding voice reference...")
            success = fish_tts.encode_voice(clean_vocal, ref_text=ref_text)
            if not success:
                print("Error: Voice encoding failed")
                return False
            saved_path = _save_fish_voice(fish_tts.encoded_refs, voice_name)
            print(f"Extreme voice '{voice_name}' saved to: {saved_path}")
            if has_test:
                script = test_script if test_script else (ref_text if ref_text else TRAIN_TEST_SCRIPT)
                print(f"Testing trained voice (extreme)...")
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                test_output = os.path.join(results_dir, f"voder_tts_extreme_{voice_name}_test_{timestamp}.wav")
                success = fish_tts.synthesize(script, test_output)
                if success:
                    print(f"Test output saved to: {test_output}")
                else:
                    print("Warning: Test synthesis failed (voice was still saved)")
            fish_tts.cleanup()
        else:
            print("Loading Qwen-TTS model...")
            tts = QwenTTS()
            if tts.model is None:
                print("Error: Failed to load Qwen-TTS model")
                return False

            print("Extracting voice characteristics...")
            ref_text = _transcribe_for_qwen_ref(clean_vocal)
            success = tts.extract_voice(clean_vocal, ref_text=ref_text if ref_text else None)
            if not success:
                print("Error: Voice extraction failed")
                return False

            voice_prompt = tts.voice_prompt
            if voice_prompt is None:
                print("Error: Voice prompt extraction returned None")
                return False

            if not isinstance(voice_prompt, list):
                voice_prompt = [voice_prompt]

            saved_path = _save_voice_prompt(voice_prompt, voice_name)
            print(f"Voice '{voice_name}' saved to: {saved_path}")

            if has_test:
                script = test_script if test_script else TRAIN_TEST_SCRIPT
                print(f"Testing trained voice...")
                results_dir = os.path.join(os.getcwd(), "results")
                os.makedirs(results_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                test_output = os.path.join(results_dir, f"voder_tts_{voice_name}_test_{timestamp}.wav")
                success = tts.synthesize(script, test_output)
                if success:
                    print(f"Test output saved to: {test_output}")
                else:
                    print("Warning: Test synthesis failed (voice was still saved)")

        return True
    finally:
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def _transcribe_for_fish_ref(audio_path):
    asr = VibeVoiceASR()
    asr.ensure_model()
    if asr.model is None:
        return ""
    try:
        text = asr.transcribe_plain_text(audio_path)
        if text is None:
            return ""
        text = re.sub(r'\[?(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]?\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\(?(?:silence|music|noise|applause|laughter|cough|breath)\)?\s*', '', text, flags=re.IGNORECASE).strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""
    finally:
        asr.cleanup()
        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _transcribe_for_qwen_ref(audio_path):
    stt = WhisperSTT()
    if stt.model is None:
        return ""
    try:
        result = stt.transcribe(audio_path)
        if result is None:
            return ""
        text = result.get("text", "").strip()
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception:
        return ""
    finally:
        stt.cleanup()
        del stt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

_ALIGNER_ONNX_PATH = os.path.join(ALIGNER_DIR, "model.onnx")
_ALIGNER_MODEL_URL = 'https://huggingface.co/deskpai/ctc_forced_aligner/resolve/main/04ac86b67129634da93aea76e0147ef3.onnx'
_ALIGNER_SESSION = None

def _get_aligner_model():
    global _ALIGNER_SESSION
    if _ALIGNER_SESSION is not None:
        return _ALIGNER_SESSION
    try:
        if not os.path.exists(_ALIGNER_ONNX_PATH):
            print("Downloading aligner model...")
            import requests
            response = requests.get(_ALIGNER_MODEL_URL, stream=True)
            response.raise_for_status()
            with open(_ALIGNER_ONNX_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Aligner model downloaded")
        import onnxruntime
        _ALIGNER_SESSION = onnxruntime.InferenceSession(_ALIGNER_ONNX_PATH)
        return _ALIGNER_SESSION
    except Exception as e:
        print(f"Warning: Failed to load aligner model: {e}")
        return None

def _cleanup_aligner_model():
    global _ALIGNER_SESSION
    if _ALIGNER_SESSION is not None:
        del _ALIGNER_SESSION
        _ALIGNER_SESSION = None
        gc.collect()

_LANG_TO_ISO3 = {
    "en": "eng", "ar": "ara", "fr": "fra", "de": "deu", "es": "spa",
    "it": "ita", "pt": "por", "ru": "rus", "ja": "jpn", "ko": "kor",
    "zh": "zho", "hi": "hin", "tr": "tur", "pl": "pol", "nl": "nld",
    "sv": "swe", "da": "dan", "fi": "fin", "el": "ell", "cs": "ces",
    "ro": "ron", "hu": "hun", "uk": "ukr", "id": "ind", "ms": "msa",
    "th": "tha", "vi": "vie", "he": "heb", "fa": "fas", "bn": "ben",
    "ta": "tam", "te": "tel", "ml": "mal", "ur": "urd"
}

def _forced_align_words(audio_path, text, language="eng"):
    if not text or not text.strip():
        return []
    if language == "auto":
        detected = _detect_lang_from_text(text)
        language = _LANG_TO_ISO3.get(detected, "eng")
    session = _get_aligner_model()
    if session is None:
        return []
    try:
        from ctc_forced_aligner import (
            load_audio, generate_emissions, preprocess_text,
            get_alignments, get_spans, postprocess_results, Tokenizer
        )
        waveform = load_audio(audio_path)
        emissions, stride = generate_emissions(session, waveform, batch_size=4)
        tokenizer = Tokenizer()
        tokens_starred, text_starred = preprocess_text(text, romanize=True, language=language)
        segments, scores, blank_token = get_alignments(emissions, tokens_starred, tokenizer)
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        return [{"text": w["text"], "start": w["start"], "end": w["end"]} for w in word_timestamps]
    except Exception as e:
        print(f"Warning: Forced alignment failed: {e}")
        return []

def _group_words_to_segments(word_timestamps, chunk_size=8, speaker=None):
    if not word_timestamps:
        return []
    segments = []
    i = 0
    while i < len(word_timestamps):
        end_idx = min(i + chunk_size, len(word_timestamps))
        for k in range(i + 1, end_idx):
            if word_timestamps[k]["start"] - word_timestamps[k - 1]["end"] > 1.5:
                end_idx = k
                break
        if end_idx - i == chunk_size and end_idx < len(word_timestamps):
            chunk_preview = word_timestamps[i:end_idx]
            duration = chunk_preview[-1]["end"] - chunk_preview[0]["start"]
            if duration > 0 and len(chunk_preview) / duration > 3.0:
                fast_limit = min(i + 12, len(word_timestamps))
                for k in range(end_idx, fast_limit):
                    if word_timestamps[k]["start"] - word_timestamps[k - 1]["end"] > 1.5:
                        break
                    end_idx = k + 1
        chunk = word_timestamps[i:end_idx]
        if not chunk:
            break
        text = " ".join(w["text"] for w in chunk)
        start = chunk[0]["start"]
        end = chunk[-1]["end"]
        seg = {"text": text, "start": start, "end": end, "word_count": len(chunk)}
        if speaker is not None:
            seg["speaker"] = speaker
        segments.append(seg)
        i = end_idx
    return segments

def _extract_speakers_for_subtitles(audio_path):
    clean_source = audio_path
    svs_temp_dir = None
    try:
        _bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
        if _bs_roformer_lib not in sys.path:
            sys.path.insert(0, _bs_roformer_lib)
        from bs_roformer import BSRoformerSeparator
        svs_sep = BSRoformerSeparator(SVS_DIR)
        svs_sep.ensure_model(stem='voice')
        svs_temp_dir = tempfile.mkdtemp()
        svs_temp = os.path.join(svs_temp_dir, 'svs_vocals.wav')
        svs_ok = svs_sep.separate(audio_path, 'voice', svs_temp)
        svs_sep.cleanup()
        del svs_sep
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if svs_ok and os.path.exists(svs_temp):
            clean_source = svs_temp
    except Exception:
        pass
    try:
        diarization = SpeakerDiarization()
        if diarization.pipeline is None:
            return None
        diar_full = diarization.diarize_full(clean_source)
        diarization.pipeline = None
        del diarization
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return None
    if diar_full is None:
        return None
    if hasattr(diar_full, 'exclusive_speaker_diarization'):
        inclusive_diar = diar_full.speaker_diarization
        exclusive_diar = diar_full.exclusive_speaker_diarization
    else:
        inclusive_diar = diar_full
        exclusive_diar = diar_full
    speaker_segments = {}
    for turn in inclusive_diar.itertracks(yield_label=True):
        segment, _, speaker = turn
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append({"start": float(segment.start), "end": float(segment.end)})
    for spk in speaker_segments:
        speaker_segments[spk].sort(key=lambda x: x["start"])
        merged = []
        for s in speaker_segments[spk]:
            if merged and s["start"] - merged[-1]["end"] < 0.3:
                merged[-1]["end"] = s["end"]
            else:
                merged.append({"start": s["start"], "end": s["end"]})
        speaker_segments[spk] = merged
    sorted_speakers = sorted(speaker_segments.keys(), key=lambda spk: speaker_segments[spk][0]["start"])
    if len(sorted_speakers) < 2:
        return None
    exclusive_segments = {}
    for turn in exclusive_diar.itertracks(yield_label=True):
        segment, _, speaker = turn
        if speaker not in exclusive_segments:
            exclusive_segments[speaker] = []
        exclusive_segments[speaker].append({
            "start": float(segment.start),
            "end": float(segment.end),
            "duration": float(segment.end) - float(segment.start)
        })
    for spk in exclusive_segments:
        exclusive_segments[spk].sort(key=lambda x: x["duration"], reverse=True)
    overlap_regions = []
    try:
        overlap_tl = inclusive_diar.get_overlap()
        for seg in overlap_tl:
            overlap_regions.append({"start": float(seg.start), "end": float(seg.end)})
    except Exception:
        pass
    from unise import UniSEEnhancer
    tse_enhancer = UniSEEnhancer(UNISE_DIR)
    tse_enhancer.ensure_model()
    if tse_enhancer.model is None:
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    temp_dir = tempfile.mkdtemp()
    speaker_files = {}
    speaker_to_idx = {}
    for idx, spk in enumerate(sorted_speakers, 1):
        speaker_to_idx[spk] = idx
    for spk in sorted_speakers:
        spk_num = speaker_to_idx[spk]
        clean_segs = exclusive_segments.get(spk, [])
        enroll_parts = []
        collected = 0.0
        target_enroll = 5.0
        for seg in clean_segs:
            if collected >= target_enroll:
                break
            remaining = target_enroll - collected
            take_dur = min(seg["duration"], remaining)
            enroll_parts.append({"start": seg["start"], "duration": take_dur})
            collected += take_dur
        if not enroll_parts:
            segs = speaker_segments[spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            start_t = longest["start"]
            dur_t = longest["end"] - longest["start"]
            if dur_t > 5.0:
                mid = start_t + dur_t / 2.0
                start_t = mid - 2.5
                dur_t = 5.0
                if start_t < 0:
                    start_t = 0.0
            enroll_parts.append({"start": start_t, "duration": dur_t})
        enroll_clip = os.path.join(temp_dir, f"enroll_spk{spk_num}.wav")
        if len(enroll_parts) == 1:
            part = enroll_parts[0]
            cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                   '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', enroll_clip]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0 or not os.path.exists(enroll_clip):
                continue
        else:
            part_files = []
            for pi, part in enumerate(enroll_parts):
                part_file = os.path.join(temp_dir, f"enroll_spk{spk_num}_part{pi}.wav")
                cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                       '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', part_file]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(part_file):
                    continue
                part_files.append(part_file)
            if not part_files:
                continue
            concat_list = os.path.join(temp_dir, f"enroll_spk{spk_num}_concat.txt")
            with open(concat_list, 'w') as f:
                for pf in part_files:
                    f.write(f"file '{pf}'\n")
            cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list,
                   '-ar', '16000', '-ac', '1', '-y', enroll_clip]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0 or not os.path.exists(enroll_clip):
                continue
        spk_output = os.path.join(temp_dir, f"spk{spk_num}_p1.wav")
        tse_ok = tse_enhancer.tse_extract(clean_source, enroll_clip, spk_output)
        if tse_ok and os.path.exists(spk_output):
            speaker_files[spk] = spk_output

    if speaker_files:
        print("Pass 2: Refining TSE with per-speaker aligned enrollment...")
        p2_asr = VibeVoiceASR()
        p2_asr.ensure_model()
        speaker_transcriptions = {}
        for spk in list(speaker_files.keys()):
            spk_num = speaker_to_idx[spk]
            p1_audio = speaker_files[spk]
            spk_text = ""
            if p2_asr.model is not None:
                try:
                    raw_text = p2_asr.transcribe_plain_text(p1_audio)
                    if raw_text:
                        spk_text = re.sub(r'\[?(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]?\s*', '', raw_text, flags=re.IGNORECASE).strip()
                        spk_text = re.sub(r'\(?(?:silence|music|noise|applause|laughter|cough|breath)\)?\s*', '', spk_text, flags=re.IGNORECASE).strip()
                        spk_text = re.sub(r'\s+', ' ', spk_text).strip()
                except Exception:
                    pass
            if spk_text:
                speaker_transcriptions[spk] = spk_text
            if not spk_text:
                continue
            detected_lang = _detect_lang_from_text(spk_text)
            lang_iso3 = _LANG_TO_ISO3.get(detected_lang, "eng")
            word_ts = _forced_align_words(p1_audio, spk_text, language=lang_iso3)
            if not word_ts:
                continue
            best_segs = []
            for w in word_ts:
                ws, we = w["start"], w["end"]
                in_overlap = False
                for ov in overlap_regions:
                    if ws < ov["end"] and we > ov["start"]:
                        in_overlap = True
                        break
                if not in_overlap:
                    best_segs.append({"start": ws, "end": we, "duration": we - ws})
            best_segs.sort(key=lambda x: x["duration"], reverse=True)
            enroll_parts2 = []
            collected2 = 0.0
            for seg in best_segs:
                if collected2 >= 5.0:
                    break
                remaining = 5.0 - collected2
                take_dur = min(seg["duration"], remaining)
                enroll_parts2.append({"start": seg["start"], "duration": take_dur})
                collected2 += take_dur
            if not enroll_parts2:
                continue
            enroll_clip2 = os.path.join(temp_dir, f"enroll_spk{spk_num}_p2.wav")
            if len(enroll_parts2) == 1:
                part = enroll_parts2[0]
                cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                       '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', enroll_clip2]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip2):
                    continue
            else:
                part_files2 = []
                for pi, part in enumerate(enroll_parts2):
                    part_file = os.path.join(temp_dir, f"enroll_spk{spk_num}_p2_part{pi}.wav")
                    cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                           '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', part_file]
                    ret = subprocess.run(cmd, capture_output=True, text=True)
                    if ret.returncode != 0 or not os.path.exists(part_file):
                        continue
                    part_files2.append(part_file)
                if not part_files2:
                    continue
                concat_list2 = os.path.join(temp_dir, f"enroll_spk{spk_num}_p2_concat.txt")
                with open(concat_list2, 'w') as f:
                    for pf in part_files2:
                        f.write(f"file '{pf}'\n")
                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list2,
                       '-ar', '16000', '-ac', '1', '-y', enroll_clip2]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip2):
                    continue
            spk_output2 = os.path.join(temp_dir, f"spk{spk_num}_p2.wav")
            tse_ok2 = tse_enhancer.tse_extract(clean_source, enroll_clip2, spk_output2)
            if tse_ok2 and os.path.exists(spk_output2):
                try:
                    os.remove(p1_audio)
                except Exception:
                    pass
                speaker_files[spk] = spk_output2
        for spk in speaker_files:
            if spk not in speaker_transcriptions and p2_asr.model is not None:
                try:
                    raw_text = p2_asr.transcribe_plain_text(speaker_files[spk])
                    if raw_text:
                        t = re.sub(r'\[?(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]?\s*', '', raw_text, flags=re.IGNORECASE).strip()
                        t = re.sub(r'\(?(?:silence|music|noise|applause|laughter|cough|breath)\)?\s*', '', t, flags=re.IGNORECASE).strip()
                        t = re.sub(r'\s+', ' ', t).strip()
                        if t:
                            speaker_transcriptions[spk] = t
                except Exception:
                    pass
        p2_asr.cleanup()
        del p2_asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cleanup_aligner_model()
    else:
        speaker_transcriptions = {}

    tse_enhancer.cleanup()
    del tse_enhancer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not speaker_files:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
        return None
    return {
        "speaker_files": speaker_files,
        "speaker_segments": speaker_segments,
        "overlap_regions": overlap_regions,
        "temp_dir": temp_dir,
        "svs_temp_dir": svs_temp_dir,
        "speaker_to_idx": speaker_to_idx,
        "speaker_transcriptions": speaker_transcriptions
    }

def _cleanup_speaker_extraction(extraction_result):
    if extraction_result is None:
        return
    temp_dir = extraction_result.get("temp_dir")
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass
    svs_temp_dir = extraction_result.get("svs_temp_dir")
    if svs_temp_dir and os.path.exists(svs_temp_dir):
        try:
            shutil.rmtree(svs_temp_dir)
        except Exception:
            pass

def _align_subtitle_segments(audio_path, segments, language="auto"):
    if not segments:
        return segments
    speakers = {}
    for seg in segments:
        spk = seg.get("speaker")
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append(seg)
    if len(speakers) < 2:
        all_text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
        if not all_text:
            return segments
        speaker = segments[0].get("speaker") if segments else None
        word_timestamps = _forced_align_words(audio_path, all_text, language=language)
        if not word_timestamps:
            return segments
        result = _group_words_to_segments(word_timestamps, chunk_size=8, speaker=speaker)
        for seg in result:
            seg["overlap"] = False
        return result
    extraction = _extract_speakers_for_subtitles(audio_path)
    if extraction is None:
        all_text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
        if not all_text:
            return segments
        word_timestamps = _forced_align_words(audio_path, all_text, language=language)
        if not word_timestamps:
            return segments
        result = _group_words_to_segments(word_timestamps, chunk_size=8, speaker=segments[0].get("speaker"))
        for seg in result:
            seg["overlap"] = False
        return result
    try:
        speaker_files = extraction["speaker_files"]
        speaker_segs = extraction["speaker_segments"]
        overlap_regions = extraction["overlap_regions"]
        diar_speakers = sorted(speaker_segs.keys(), key=lambda spk: speaker_segs[spk][0]["start"])
        asr_speakers = sorted(speakers.keys())
        speaker_map = {}
        for i, asr_spk in enumerate(asr_speakers):
            speaker_map[asr_spk] = diar_speakers[i] if i < len(diar_speakers) else diar_speakers[-1]
        all_aligned = []
        for asr_spk, spk_segs in speakers.items():
            diar_spk = speaker_map.get(asr_spk)
            if diar_spk is None:
                continue
            spk_text = " ".join(seg.get("text", "").strip() for seg in spk_segs if seg.get("text", "").strip())
            if not spk_text:
                continue
            separated_audio = speaker_files.get(diar_spk)
            if separated_audio and os.path.exists(separated_audio):
                word_ts = _forced_align_words(separated_audio, spk_text, language=language)
            else:
                word_ts = _forced_align_words(audio_path, spk_text, language=language)
            if not word_ts:
                for seg in spk_segs:
                    seg_copy = dict(seg)
                    seg_copy["overlap"] = False
                    all_aligned.append(seg_copy)
                continue
            windows = speaker_segs.get(diar_spk, [])
            if windows:
                filtered = []
                for w in word_ts:
                    for win in windows:
                        if w["start"] >= win["start"] - 0.3 and w["end"] <= win["end"] + 0.3:
                            filtered.append(w)
                            break
                if filtered:
                    word_ts = filtered
            chunked = _group_words_to_segments(word_ts, chunk_size=8, speaker=asr_spk)
            for chunk in chunked:
                is_overlap = False
                for ov in overlap_regions:
                    if chunk["start"] < ov["end"] and chunk["end"] > ov["start"]:
                        is_overlap = True
                        break
                chunk["overlap"] = is_overlap
            all_aligned.extend(chunked)
        all_aligned.sort(key=lambda x: x["start"])
        if not all_aligned:
            all_text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
            if all_text:
                word_timestamps = _forced_align_words(audio_path, all_text, language=language)
                if word_timestamps:
                    result = _group_words_to_segments(word_timestamps, chunk_size=8, speaker=segments[0].get("speaker"))
                    for seg in result:
                        seg["overlap"] = False
                    return result
            return segments
        return all_aligned
    except Exception as e:
        print(f"Warning: Per-speaker alignment failed: {e}")
        all_text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text", "").strip())
        if not all_text:
            return segments
        word_timestamps = _forced_align_words(audio_path, all_text, language=language)
        if not word_timestamps:
            return segments
        result = _group_words_to_segments(word_timestamps, chunk_size=8, speaker=segments[0].get("speaker"))
        for seg in result:
            seg["overlap"] = False
        return result
    finally:
        _cleanup_speaker_extraction(extraction)

def _tts_extract_voice(engine, audio_path, use_extreme=False, ref_text=None):
    if use_extreme and isinstance(engine, FishTTS):
        return engine.encode_voice(audio_path, ref_text=ref_text)
    return engine.extract_voice(audio_path, ref_text=ref_text)

def _tts_synthesize(engine, text, output_path, language="Auto", use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        return engine.synthesize(text, output_path)
    return engine.synthesize(text, output_path, language=language)

def _tts_load_voice(engine, voice_path, use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        payload = _load_fish_voice(voice_path)
        if payload is None:
            return False
        engine.encoded_refs = payload
        return True
    items = _load_voice_prompt(voice_path)
    if items is None:
        return False
    engine.voice_prompt = items
    return True

def _tts_cleanup(engine, use_extreme=False):
    if use_extreme and isinstance(engine, FishTTS):
        engine.cleanup()
    else:
        import gc
        try:
            del engine
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

def _sts_extreme_pass(audio_path):
    fish_tts = FishTTS()
    if not fish_tts.ensure_model():
        print("Error: Failed to load Fish-S2Pro model for STS extreme pass")
        return None
    print("Transcribing target reference (extreme)...")
    ref_text = _transcribe_for_fish_ref(audio_path)
    if not ref_text:
        print("Warning: STS extreme transcription empty, falling back to original reference")
        fish_tts.cleanup()
        return None
    print("Encoding target reference (extreme)...")
    success = fish_tts.encode_voice(audio_path, ref_text=ref_text)
    if not success:
        print("Warning: STS extreme voice encoding failed, falling back to original reference")
        fish_tts.cleanup()
        return None
    temp_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_output.close()
    print("Synthesizing clean reference (extreme)...")
    success = fish_tts.synthesize(ref_text, temp_output.name)
    fish_tts.cleanup()
    if not success or not os.path.exists(temp_output.name):
        print("Warning: STS extreme synthesis failed, falling back to original reference")
        if os.path.exists(temp_output.name):
            os.unlink(temp_output.name)
        return None
    return temp_output.name

def oneline_tts(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    scripts = params.get('script', [])
    voices = params.get('voice', [])
    targets = params.get('target', [])
    music_params = params.get('music', [])
    music_description = music_params[0] if music_params else None
    level_params = params.get('level', [])
    music_level_spec = level_params[0] if level_params else None
    reference_params = params.get('reference', [])
    reference_source = reference_params[0] if reference_params else None
    ocr_param = params.get('ocr', [])
    use_overdose = params.get('overdose', False)
    use_extreme = params.get('extreme', False)

    if params.get('slc'):
        slc_path = params.get('slc_path')
        if not slc_path:
            for t in targets:
                if t:
                    slc_path = t
                    break
        if not slc_path and voices:
            for v in voices:
                if os.path.exists(v) or is_youtube_url(v):
                    slc_path = v
                    break
        if not slc_path:
            for s in scripts:
                if os.path.exists(s) or is_youtube_url(s):
                    slc_path = s
                    break
        if not slc_path:
            print("Error: TTS SLC requires an audio/video source path")
            return False

        _slc_cleanup = []
        audio_path = slc_path
        needs_youtube_dl = is_youtube_url(slc_path)

        if needs_youtube_dl:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(slc_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _slc_cleanup.append(dl_path)
        elif slc_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(slc_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _slc_cleanup.append(audio_path)
        elif not os.path.exists(slc_path):
            print(f"Error: File not found: {slc_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _slc_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        slc_music = params.get('slc_music', False)
        music_track = None
        if slc_music:
            print("Extracting music track via SVS music...")
            music_track = svs_extract_music(audio_path)
            if music_track and music_track != audio_path:
                _slc_cleanup.append(music_track)
            else:
                music_track = None

        print("Loading Whisper model (large-v3)...")
        stt = WhisperSTT(skip_turbo=True)
        stt.ensure_translate_model()
        if stt.translate_model is None:
            print("Error: Failed to load Whisper large-v3 model")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print("Transcribing audio...")
        try:
            result = stt.translate_model.transcribe(clean_vocal, word_timestamps=True)
        except Exception as e:
            print(f"Transcription error: {e}")
            result = None
        if not result:
            print("Error: Transcription failed")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        detected_lang = result.get("language", "en")
        transcribed_text = result.get("text", "").strip()
        if not transcribed_text:
            print("Error: No speech detected in audio")
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print(f"Detected language: {detected_lang}")
        print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        slc_translate_langs = params.get('slc_translate_langs')
        tts_lang = "English"
        final_text = transcribed_text

        if slc_translate_langs:
            tgt_lang = slc_translate_langs['target']
            src_lang = slc_translate_langs['source']
            if src_lang == 'auto':
                src_lang = detected_lang[:2].lower() if detected_lang else 'en'

            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if src_lang == tgt_lang:
                print(f"Source and target are both {tgt_lang}, no translation needed")
            else:
                print(f"Translating with TranslateGemma ({src_lang}->{tgt_lang})...")
                translated = _translate_with_gemma(transcribed_text, src_lang, tgt_lang)
                if translated:
                    final_text = translated
                    print(f"Translated text: {final_text[:100]}{'...' if len(final_text) > 100 else ''}")
                else:
                    print("Warning: TranslateGemma translation failed, using original transcription")

            lang_name_map = {'en': 'English', 'ar': 'Arabic', 'fr': 'French', 'de': 'German',
                           'es': 'Spanish', 'it': 'Italian', 'ja': 'Japanese', 'ko': 'Korean',
                           'zh': 'Chinese', 'pt': 'Portuguese', 'ru': 'Russian', 'hi': 'Hindi',
                           'tr': 'Turkish', 'nl': 'Dutch', 'pl': 'Polish', 'sv': 'Swedish'}
            tts_lang = lang_name_map.get(tgt_lang, tgt_lang.upper())
        elif detected_lang == "en":
            print("Audio is already in English")
        else:
            print("Translating to English...")
            try:
                trans_result = stt.translate_model.transcribe(clean_vocal, task="translate", word_timestamps=True)
            except Exception as e:
                print(f"Translation error: {e}")
                trans_result = None
            if trans_result and trans_result.get("text", "").strip():
                final_text = trans_result["text"].strip()
                print(f"Translated text: {final_text[:100]}{'...' if len(final_text) > 100 else ''}")
            else:
                print("Warning: Translation failed, using original transcription")

            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"TTS language: {tts_lang}")
        slc_lang_set = SUPPORTED_FISH_LANGS if use_extreme else set(SUPPORTED_TTS_LANGUAGES.keys())
        slc_lang_ctx = "TTS (extreme)" if use_extreme else "TTS"
        slc_valid, _ = _validate_text_language(final_text, slc_lang_set, slc_lang_ctx)
        if not slc_valid:
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        if use_extreme:
            print("Loading Fish-S2Pro model (extreme)...")
            tts = FishTTS()
            if not tts.ensure_model():
                print("Error: Failed to load Fish-S2Pro model")
                for f in _slc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
        else:
            print("Loading Qwen-TTS model...")
            tts = QwenTTS()
        print("Extracting voice characteristics...")
        ref_text = _transcribe_for_fish_ref(clean_vocal) if use_extreme else (_transcribe_for_qwen_ref(clean_vocal) if isinstance(tts, QwenTTS) else None)
        success = _tts_extract_voice(tts, clean_vocal, use_extreme=use_extreme, ref_text=ref_text)
        if not success:
            print("Error: Voice extraction failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        print("Generating speech...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_slc_{timestamp}.wav")
        success = _tts_synthesize(tts, final_text, output_path, language=tts_lang, use_extreme=use_extreme)
        if not success:
            print("Error: Synthesis failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _slc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        _tts_cleanup(tts, use_extreme=use_extreme)
        tts = None

        print(f"✓ SLC output saved to: {output_path}")

        if use_overdose:
            print("\nRunning overdose pass (STS v2 non-mimic)...")
            vc = SeedVCV2()
            if vc.model is None:
                print("Warning: Seed-VC v2 model failed to load, skipping overdose pass")
            else:
                svs_out = svs_extract_vocals(output_path)
                if svs_out and svs_out != output_path:
                    _slc_cleanup.append(svs_out)
                    vc_input = svs_out
                else:
                    vc_input = output_path
                try:
                    od_timestamp = time.strftime("%Y%m%d_%H%M%S")
                    od_output = os.path.join(results_dir, f"voder_tts_slc_od_{od_timestamp}.wav")
                    od_success = vc.convert(vc_input, clean_vocal, od_output)
                    if od_success:
                        print(f"✓ Overdose output saved to: {od_output}")
                        output_path = od_output
                    else:
                        print("Warning: Overdose STS pass failed, using standard SLC output")
                finally:
                    del vc
                    vc = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        if music_track and os.path.exists(music_track):
            print("\nBlending voice output with music track...")
            blend_timestamp = time.strftime("%Y%m%d_%H%M%S")
            blend_output = os.path.join(results_dir, f"voder_tts_slc_music_{blend_timestamp}.wav")
            blend_cmd = [
                'ffmpeg', '-i', output_path, '-i', music_track,
                '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]',
                '-map', '[out]', '-y', blend_output
            ]
            blend_result = subprocess.run(blend_cmd, capture_output=True, text=True)
            if blend_result.returncode == 0 and os.path.exists(blend_output):
                print(f"✓ Blended output saved to: {blend_output}")
                output_path = blend_output
            else:
                print("Warning: Music blending failed, voice-only output preserved")

        for f in _slc_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return True

    if params.get('dub'):
        return oneline_tts_dub(params)

    if params.get('svc'):
        svc_path = params.get('svc_path')
        if not svc_path:
            for s in scripts:
                if os.path.exists(s) or is_youtube_url(s):
                    svc_path = s
                    break
        if not svc_path:
            for t in targets:
                if t and not t.lower().startswith('sts:'):
                    if os.path.exists(t) or is_youtube_url(t):
                        svc_path = t
                        break
        if not svc_path and voices:
            for v in voices:
                if v and not v.lower().startswith('sts:'):
                    if os.path.exists(v) or is_youtube_url(v):
                        svc_path = v
                        break
        if not svc_path:
            print("Error: TTS SVC requires an audio/video source path")
            return False

        svc_target = None
        for t in targets:
            if t:
                svc_target = t
                break
        if not svc_target:
            for v in voices:
                if v:
                    svc_target = v
                    break
        if not svc_target:
            print("Error: TTS SVC requires a target voice reference (target or voice parameter)")
            return False

        _svc_cleanup = []
        audio_path = svc_path
        needs_youtube_dl = is_youtube_url(svc_path)

        if needs_youtube_dl:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(svc_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _svc_cleanup.append(dl_path)
        elif svc_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(svc_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _svc_cleanup.append(audio_path)
        elif not os.path.exists(svc_path):
            print(f"Error: File not found: {svc_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _svc_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        if use_overdose:
            print("Loading VibeVoice ASR (overdose mode)...")
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                use_overdose = False
            else:
                try:
                    transcribed_text = asr.transcribe_plain_text(clean_vocal)
                except Exception as e:
                    print(f"VibeVoice transcription error: {e}")
                    transcribed_text = ""
                del asr
                asr = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not transcribed_text or not transcribed_text.strip():
                    print("Error: No speech detected in audio (VibeVoice)")
                    for f in _svc_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                transcribed_text = transcribed_text.strip()
                detected_lang = None
                print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        if not use_overdose:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            stt.ensure_model()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print("Transcribing audio...")
            try:
                result = stt.model.transcribe(clean_vocal, word_timestamps=True)
            except Exception as e:
                print(f"Transcription error: {e}")
                result = None
            if not result:
                print("Error: Transcription failed")
                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            detected_lang = result.get("language", "en")
            transcribed_text = result.get("text", "").strip()
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not transcribed_text:
                print("Error: No speech detected in audio")
                for f in _svc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            print(f"Detected language: {detected_lang}")
            print(f"Transcribed text ({len(transcribed_text)} chars): {transcribed_text[:100]}{'...' if len(transcribed_text) > 100 else ''}")

        tts_lang = "Auto"
        final_text = transcribed_text

        sts_target = svc_target.lower().startswith('sts:')
        target_voice_path = None
        if sts_target:
            sts_ref_raw = svc_target[4:]
            trained_file = _resolve_voice_ref(sts_ref_raw)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                print(f"Loading trained voice for STS pass: {trained_file}")
                if use_extreme:
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        tts.voice_prompt = voice_items
                        use_extreme = False
                else:
                    voice_items = _load_voice_prompt(trained_file)
                    if voice_items is None:
                        print(f"Error: Failed to load trained voice: {trained_file}")
                        return False
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    tts.voice_prompt = voice_items
            else:
                multi = _parse_multi_refs(sts_ref_raw)
                if multi:
                    target_voice_path = _resolve_multi_refs(multi, _svc_cleanup)
                    if not target_voice_path:
                        print("Error: Could not resolve STS target multi-reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                else:
                    resolved_audio, _cl = resolve_target_to_audio(sts_ref_raw)
                    if not resolved_audio:
                        print("Error: Could not resolve STS target reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    _svc_cleanup.extend(_cl)
                    target_voice_path = svs_extract_vocals(resolved_audio)
                    if target_voice_path and target_voice_path != resolved_audio:
                        _svc_cleanup.append(target_voice_path)
                    if resolved_audio not in _svc_cleanup and resolved_audio != target_voice_path:
                        _svc_cleanup.append(resolved_audio)
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                else:
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                print("Extracting voice characteristics from STS target...")
                ref_text = _transcribe_for_fish_ref(target_voice_path) if use_extreme else (_transcribe_for_qwen_ref(target_voice_path) if isinstance(tts, QwenTTS) else None)
                success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme, ref_text=ref_text)
                if not success:
                    print("Error: Voice extraction from STS target failed")
                    del tts
                    tts = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for f in _svc_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
        else:
            trained_file = _resolve_voice_ref(svc_target)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                print(f"Loading trained voice from: {trained_file}")
                if use_extreme:
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        tts.voice_prompt = voice_items
                        use_extreme = False
                else:
                    voice_items = _load_voice_prompt(trained_file)
                    if voice_items is None:
                        print(f"Error: Failed to load trained voice: {trained_file}")
                        return False
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    tts.voice_prompt = voice_items
            else:
                multi = _parse_multi_refs(svc_target)
                if multi:
                    target_voice_path = _resolve_multi_refs(multi, _svc_cleanup)
                    if not target_voice_path:
                        print("Error: Could not resolve target multi-reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    if use_extreme:
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            return False
                    else:
                        print("Loading Qwen-TTS model...")
                        tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    ref_text = _transcribe_for_fish_ref(target_voice_path) if use_extreme else (_transcribe_for_qwen_ref(target_voice_path) if isinstance(tts, QwenTTS) else None)
                    success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme, ref_text=ref_text)
                    if not success:
                        print("Error: Voice extraction failed")
                        del tts
                        tts = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                elif os.path.exists(svc_target) or is_youtube_url(svc_target):
                    resolved_audio, _cl = resolve_target_to_audio(svc_target)
                    if not resolved_audio:
                        print("Error: Could not resolve target audio reference")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                    _svc_cleanup.extend(_cl)
                    target_voice_path = svs_extract_vocals(resolved_audio)
                    if target_voice_path and target_voice_path != resolved_audio:
                        _svc_cleanup.append(target_voice_path)
                    if resolved_audio not in _svc_cleanup and resolved_audio != target_voice_path:
                        _svc_cleanup.append(resolved_audio)
                    if use_extreme:
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            return False
                    else:
                        print("Loading Qwen-TTS model...")
                        tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    ref_text = _transcribe_for_fish_ref(target_voice_path) if use_extreme else (_transcribe_for_qwen_ref(target_voice_path) if isinstance(tts, QwenTTS) else None)
                    success = _tts_extract_voice(tts, target_voice_path, use_extreme=use_extreme, ref_text=ref_text)
                    if not success:
                        print("Error: Voice extraction failed")
                        del tts
                        tts = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return False
                else:
                    if use_extreme:
                        print("Loading Qwen-TTS VoiceDesign model (extreme: generating placeholder voice)...")
                        tts_design = QwenTTSVoiceDesign()
                        if tts_design.model is None:
                            print("Error: Failed to load VoiceDesign model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        placeholder_text = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor."
                        placeholder_path = os.path.join(tempfile.gettempdir(), f"voder_extreme_placeholder_{int(time.time())}.wav")
                        success = tts_design.synthesize(placeholder_text, svc_target, placeholder_path, language="English")
                        del tts_design
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if not success or not os.path.exists(placeholder_path):
                            print("Error: Failed to generate voice design placeholder for extreme mode")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        _svc_cleanup.append(placeholder_path)
                        print("Loading Fish-S2Pro model (extreme)...")
                        tts = FishTTS()
                        if not tts.ensure_model():
                            print("Error: Failed to load Fish-S2Pro model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Encoding voice design audio as Fish reference...")
                        success = tts.encode_voice(placeholder_path, ref_text=placeholder_text)
                        if not success:
                            print("Error: Fish voice encoding failed")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Generating speech (extreme)...")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(results_dir, f"voder_tts_svc_extreme_{timestamp}.wav")
                        success = tts.synthesize(final_text, output_path)
                        if not success:
                            print("Error: Fish TTS synthesis failed")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        _tts_cleanup(tts, use_extreme=True)
                        tts = None
                        print(f"✓ SVC extreme output saved to: {output_path}")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return True
                    else:
                        print(f"Loading Qwen-TTS VoiceDesign model...")
                        tts_design = QwenTTSVoiceDesign()
                        if tts_design.model is None:
                            print("Error: Failed to load VoiceDesign model")
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        print("Generating speech with voice design...")
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(results_dir, f"voder_tts_svc_{timestamp}.wav")
                        success = tts_design.synthesize(final_text, svc_target, output_path, language=tts_lang)
                        if not success:
                            print("Error: VoiceDesign synthesis failed")
                            del tts_design
                            for f in _svc_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        del tts_design
                        print(f"✓ SVC output saved to: {output_path}")
                        for f in _svc_cleanup:
                            if f and os.path.exists(f):
                                try:
                                    os.unlink(f)
                                except:
                                    pass
                        return True

        print("Generating speech...")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_tts_svc_{timestamp}.wav")
        success = _tts_synthesize(tts, final_text, output_path, language=tts_lang, use_extreme=use_extreme)
        if not success:
            print("Error: Synthesis failed")
            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for f in _svc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False

        del tts
        tts = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✓ SVC output saved to: {output_path}")

        if sts_target:
            print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
            if target_voice_path and os.path.exists(target_voice_path):
                vc = SeedVCV2()
                if vc.model is None:
                    print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                else:
                    svs_out = svs_extract_vocals(output_path)
                    if svs_out and svs_out != output_path:
                        _svc_cleanup.append(svs_out)
                        vc_input = svs_out
                    else:
                        vc_input = output_path
                    try:
                        sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sts_output = os.path.join(results_dir, f"voder_tts_svc_sts_{sts_timestamp}.wav")
                        sts_success = vc.convert(vc_input, target_voice_path, sts_output)
                        if sts_success:
                            print(f"✓ STS-converted output saved to: {sts_output}")
                            output_path = sts_output
                        else:
                            print("Warning: STS voice conversion pass failed, using TTS output")
                    finally:
                        del vc
                        vc = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                print("Warning: No target voice path available for STS pass (trained voice used)")

        for f in _svc_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return True

    if ocr_param:
        ocr_path = ocr_param[0]
        if not os.path.exists(ocr_path):
            print(f"Error: Image file not found: {ocr_path}")
            return False
        ext = os.path.splitext(ocr_path)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']:
            print(f"Error: Input must be an image file. Supported formats: PNG, JPG, JPEG, BMP, GIF, TIFF, WebP")
            return False
        print("Loading EasyOCR model...")
        ocr = EasyOCRReader()
        if ocr.reader is None:
            print("Error: Failed to load EasyOCR model")
            return False
        print(f"Extracting text from image...")
        success, extracted_text, error_msg = ocr.extract_text_from_image(ocr_path)
        ocr.cleanup()
        del ocr
        gc.collect()
        if not success:
            print(f"Error: {error_msg or 'Failed to extract text from image'}")
            return False
        if not extracted_text:
            print("Error: No text found in image")
            return False
        scripts = [f"text: {extracted_text}"]
        print(f"Extracted text: {extracted_text[:100]}{'...' if len(extracted_text) > 100 else ''}")

    if not scripts:
        print("Error: TTS mode requires at least one 'script' parameter")
        return False

    _is_all_sfx = all(s.strip().lower().startswith('sfx:') for s in scripts)
    if not voices and not targets and not _is_all_sfx:
        print("Error: TTS mode requires at least one 'voice' or 'target' parameter")
        return False

    has_colon_script = any(':' in s for s in scripts)
    has_colon_voice = any(':' in v for v in voices) if voices else False
    has_colon_target = any(':' in t for t in targets) if targets else False

    if not has_colon_script and not has_colon_voice and not has_colon_target:
        if len(scripts) != 1:
            print("Error: Single mode expects exactly one script argument")
            return False
        if not voices and not targets:
            print("Error: Single mode expects one voice or target argument")
            return False
        if music_description:
            print("Warning: Background music is only supported for dialogue mode. Ignoring music parameter.")
        if reference_source:
            print("Warning: Music reference is only supported for dialogue mode. Ignoring reference parameter.")
        script = scripts[0].replace('\\n', '\n')
        tts_lang_set = SUPPORTED_FISH_LANGS if use_extreme else set(SUPPORTED_TTS_LANGUAGES.keys())
        tts_lang_ctx = "TTS (extreme)" if use_extreme else "TTS"
        tts_valid, _ = _validate_text_language(script, tts_lang_set, tts_lang_ctx)
        if not tts_valid:
            return False
        if voices:
            voice_value = voices[0]
            trained_file = _resolve_voice_ref(voice_value)
            if trained_file:
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    if not _tts_load_voice(tts, trained_file, use_extreme=True):
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        tts = QwenTTS()
                        if tts.model is None:
                            print("Error: Failed to load Qwen-TTS model")
                            return False
                        tts.voice_prompt = voice_items
                        use_extreme = False
                    print("Generating speech with trained voice (extreme)...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(results_dir, f"voder_tts_extreme_{timestamp}.wav")
                    success = _tts_synthesize(tts, script, output_path, use_extreme=use_extreme)
                    if not success:
                        print("Error: Synthesis failed")
                        return False
                    _tts_cleanup(tts, use_extreme=use_extreme)
                    print(f"✓ Success! Output saved to: {output_path}")
                    return True
                print(f"Loading trained voice from: {trained_file}")
                voice_items = _load_voice_prompt(trained_file)
                if voice_items is None:
                    print(f"Error: Failed to load trained voice: {trained_file}")
                    return False
                print("Loading Qwen-TTS model...")
                tts = QwenTTS()
                if tts.model is None:
                    print("Error: Failed to load Qwen-TTS model")
                    return False
                tts.voice_prompt = voice_items
                print("Generating speech with trained voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts.synthesize(script, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                print(f"✓ Success! Output saved to: {output_path}")
                return True
            voice_prompt = voice_value
            if use_extreme:
                print("Loading Qwen-TTS VoiceDesign model (extreme: generating placeholder voice)...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False
                placeholder_text = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor."
                placeholder_path = os.path.join(tempfile.gettempdir(), f"voder_extreme_placeholder_{int(time.time())}.wav")
                success = tts_design.synthesize(placeholder_text, voice_prompt, placeholder_path, language="English")
                del tts_design
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not success or not os.path.exists(placeholder_path):
                    print("Error: Failed to generate voice design placeholder for extreme mode")
                    return False
                try:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    print("Encoding voice design audio as Fish reference...")
                    success = tts.encode_voice(placeholder_path, ref_text=placeholder_text)
                    if not success:
                        print("Error: Fish voice encoding failed")
                        return False
                    print("Generating speech (extreme)...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(results_dir, f"voder_tts_extreme_{timestamp}.wav")
                    success = tts.synthesize(script, output_path)
                    if not success:
                        print("Error: Fish TTS synthesis failed")
                        return False
                    _tts_cleanup(tts, use_extreme=True)
                    print(f"✓ Extreme output saved to: {output_path}")
                    return True
                finally:
                    if os.path.exists(placeholder_path):
                        try:
                            os.unlink(placeholder_path)
                        except:
                            pass
            else:
                print("Loading Qwen-TTS VoiceDesign model...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False
                print("Generating speech...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts_design.synthesize(script, voice_prompt, output_path)
                if not success:
                    print("Error: VoiceDesign synthesis failed")
                    return False
                print(f"✓ Success! Output saved to: {output_path}")
                return True
        else:
            target_value = targets[0]
            is_sts_target = target_value.lower().startswith('sts:')
            if is_sts_target:
                target_value = target_value[4:]
            use_first = params.get('use_first', False)
            multi = _parse_multi_refs(target_value)
            _cleanup = []
            try:
                if multi:
                    clean_vocal = _resolve_multi_refs(multi, _cleanup, use_first=use_first)
                    if not clean_vocal:
                        return False
                else:
                    if use_first:
                        print("Warning: 'first' keyword ignored (only one reference provided)")
                    resolved_audio, _cl = resolve_target_to_audio(target_value)
                    if not resolved_audio:
                        return False
                    _cleanup.extend(_cl)
                    clean_vocal = svs_extract_vocals(resolved_audio)
                    if clean_vocal and clean_vocal != resolved_audio:
                        _cleanup.append(clean_vocal)
                    if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                        _cleanup.append(resolved_audio)
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts = FishTTS()
                    if not tts.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                else:
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                print("Extracting voice characteristics...")
                ref_text = _transcribe_for_fish_ref(clean_vocal) if use_extreme else (_transcribe_for_qwen_ref(clean_vocal) if isinstance(tts, QwenTTS) else None)
                success = _tts_extract_voice(tts, clean_vocal, use_extreme=use_extreme, ref_text=ref_text)
                if not success:
                    print("Error: Voice extraction failed")
                    return False
                print("Generating speech with cloned voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{'extreme_' if use_extreme else ''}{timestamp}.wav")
                success = _tts_synthesize(tts, script, output_path, use_extreme=use_extreme)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                del tts
                tts = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if is_sts_target:
                    print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
                    vc = SeedVCV2()
                    if vc.model is None:
                        print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                    else:
                        svs_out = svs_extract_vocals(output_path)
                        if svs_out and svs_out != output_path:
                            _cleanup.append(svs_out)
                            vc_input = svs_out
                        else:
                            vc_input = output_path
                        try:
                            sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                            sts_output = os.path.join(results_dir, f"voder_tts_sts_{sts_timestamp}.wav")
                            sts_success = vc.convert(vc_input, clean_vocal, sts_output)
                            if sts_success:
                                print(f"✓ STS-converted output saved to: {sts_output}")
                                output_path = sts_output
                            else:
                                print("Warning: STS voice conversion pass failed, using TTS output")
                        finally:
                            del vc
                            vc = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                else:
                    print(f"✓ Success! Output saved to: {output_path}")
                return True
            finally:
                for f in _cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
    else:
        if not has_colon_script:
            print("Error: Dialogue script must be in format 'Character: text', got: {s}")
            return False
        if not _is_all_sfx and not has_colon_voice and not has_colon_target:
            print("Error: Dialogue mode requires voice or target parameters in 'Character: value' format.")
            return False
        dialogue_items = []
        for idx, s in enumerate(scripts, start=1):
            if ':' not in s:
                print(f"Error: Dialogue script must be in format 'Character: text', got: {s}")
                return False
            char, text = s.split(':', 1)
            char = char.strip()
            text = text.strip().replace('\\n', '\n')
            if not char:
                print(f"Error: Empty character in script: {s}")
                return False
            if char.lower() == 'sfx' and not text:
                print(f"Error: Empty SFX prompt in script: {s}")
                return False
            clean_text, directives_raw = _parse_script_directives(text)
            parsed_directives, errors = _parse_directives_for_line(directives_raw)
            if errors:
                print(f"Error in script line {idx}: {'; '.join(errors)}")
                return False
            if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                print(f"Error: SFX line {idx} requires /duration:nn (1-30)")
                return False
            if not clean_text and char.lower() != 'sfx':
                print(f"Error: Empty text in script: {s}")
                return False
            dialogue_items.append((idx, char, clean_text, parsed_directives))

        tts_lang_set = SUPPORTED_FISH_LANGS if use_extreme else set(SUPPORTED_TTS_LANGUAGES.keys())
        tts_lang_ctx = "TTS (extreme)" if use_extreme else "TTS"
        all_dial_text = " ".join(di[2] for di in dialogue_items if di[1].lower() != 'sfx')
        if all_dial_text.strip():
            tts_valid, _ = _validate_text_language(all_dial_text, tts_lang_set, tts_lang_ctx)
            if not tts_valid:
                return False

        voice_prompts = {}
        trained_voice_refs = {}
        for v in voices:
            if ':' not in v:
                print(f"Error: Voice prompt must be in format 'Character: prompt', got: {v}")
                return False
            char, prompt = v.split(':', 1)
            char = char.strip()
            prompt = prompt.strip()
            if not char or not prompt:
                print(f"Error: Empty character or prompt in voice: {v}")
                return False
            trained_file = _resolve_voice_ref(prompt)
            if trained_file:
                trained_voice_refs[char.lower()] = trained_file
            else:
                voice_prompts[char.lower()] = prompt

        try:
            target_assignments = {}
            sts_refs = {}
            all_target_cleanup = []
            use_first = params.get('use_first', False)
            for t in targets:
                if ':' not in t:
                    print(f"Error: Target assignment must be in format 'Character: path', got: {t}")
                    return False
                char, path = t.split(':', 1)
                char = char.strip()
                path = path.strip()
                if not char or not path:
                    print(f"Error: Empty character or path in target: {t}")
                    return False
                is_sts = path.lower().startswith('sts:')
                if is_sts:
                    path = path[4:]
                multi = _parse_multi_refs(path)
                if multi:
                    clean_vocal = _resolve_multi_refs(multi, all_target_cleanup, use_first=use_first)
                    if not clean_vocal:
                        return False
                    target_assignments[char.lower()] = clean_vocal
                    if is_sts:
                        sts_refs[char.lower()] = clean_vocal
                else:
                    resolved_audio, _cleanup = resolve_target_to_audio(path)
                    if not resolved_audio:
                        return False
                    all_target_cleanup.extend(_cleanup)
                    clean_vocal = svs_extract_vocals(resolved_audio)
                    if clean_vocal and clean_vocal != resolved_audio:
                        all_target_cleanup.append(clean_vocal)
                    target_assignments[char.lower()] = clean_vocal
                    if is_sts:
                        sts_ref_path = clean_vocal
                        sts_refs[char.lower()] = sts_ref_path

            overlap = set(voice_prompts.keys()) & (set(target_assignments.keys()) | set(trained_voice_refs.keys()))
            if overlap:
                print(f"Error: Character(s) specified in both voice and target/trained: {', '.join(overlap)}")
                return False

            trained_voice_overlap = set(trained_voice_refs.keys()) & set(target_assignments.keys())
            if trained_voice_overlap:
                print(f"Error: Character(s) specified in both trained voice and target: {', '.join(trained_voice_overlap)}")
                return False

            script_chars = set()
            for _, char, _, _ in dialogue_items:
                if char.lower() != 'sfx':
                    script_chars.add(char.lower())
            all_assigned = set(voice_prompts.keys()) | set(target_assignments.keys()) | set(trained_voice_refs.keys())
            missing = script_chars - all_assigned
            if missing:
                print(f"Error: Missing voice/target for characters: {', '.join(missing)}")
                return False

            has_tts_chars = len(voice_prompts) > 0
            has_vc_chars = len(target_assignments) > 0 or len(trained_voice_refs) > 0

            for char_lower, trained_file in trained_voice_refs.items():
                if _check_voice_extreme_mismatch(trained_file, use_extreme):
                    return False

            if music_description and music_description.strip() == "":
                music_description = None

            if music_level_spec and not music_description:
                print("Warning: Level spec ignored (no music description provided)")

            if reference_source and not music_description:
                print("Warning: Music reference ignored (no music description provided)")

            reference_audio = None
            if reference_source and music_description:
                _ref_is_video = False
                _ref_is_link = is_youtube_url(reference_source)
                if not _ref_is_link and os.path.exists(reference_source):
                    _ref_ext = os.path.splitext(reference_source)[1].lower()
                    _ref_is_video = _ref_ext in VIDEO_EXTENSIONS
                if _ref_is_video:
                    print("Reference is a video file, extracting audio...")
                elif _ref_is_link:
                    print("Reference is a URL, downloading audio...")
                else:
                    print("Resolving music reference source...")
                resolved_ref_audio, ref_cleanup = resolve_target_to_audio(reference_source)
                if not resolved_ref_audio:
                    print("Error: Could not resolve music reference source")
                    return False
                print("Cleaning reference through SVS music pipe...")
                reference_audio = svs_extract_music(resolved_ref_audio)
                all_target_cleanup.extend(ref_cleanup)
                if reference_audio and reference_audio != resolved_ref_audio and reference_audio not in all_target_cleanup:
                    all_target_cleanup.append(reference_audio)

            tts_design = None
            vc_voice_prompts = None
            fish_voice_data = None
            tts_obj = None

            if use_extreme:
                fish_voice_data = {}
                placeholder_text = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore, while the crystal clear waves gently lap against the warm golden sand. Every morning, the old lighthouse keeper climbs the winding stone stairs to check the beam that guides ships safely through the foggy harbor."
                _design_cleanup = []
                _placeholder_paths = {}
                if has_tts_chars:
                    print("Loading Qwen-TTS VoiceDesign model (extreme)...")
                    tts_design = QwenTTSVoiceDesign()
                    if tts_design.model is None:
                        print("Error: Failed to load VoiceDesign model")
                        return False
                    for char_lower, voice_instruct in voice_prompts.items():
                        print(f"Generating voice design placeholder for '{char_lower}' (extreme)...")
                        placeholder_path = os.path.join(tempfile.gettempdir(), f"voder_extreme_placeholder_{char_lower}_{int(time.time())}.wav")
                        success = tts_design.synthesize(placeholder_text, voice_instruct, placeholder_path, language="English")
                        if not success or not os.path.exists(placeholder_path):
                            print(f"Error: Failed to generate voice design placeholder for '{char_lower}'")
                            for f in _design_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        _design_cleanup.append(placeholder_path)
                        _placeholder_paths[char_lower] = placeholder_path
                    del tts_design
                    tts_design = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                print("Loading Fish-S2Pro model (extreme)...")
                tts_obj = FishTTS()
                if not tts_obj.ensure_model():
                    print("Error: Failed to load Fish-S2Pro model")
                    for f in _design_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                if has_vc_chars:
                    for char_lower, audio_path in target_assignments.items():
                        print(f"Transcribing voice for '{char_lower}' (extreme)...")
                        ref_text = _transcribe_for_fish_ref(audio_path)
                        print(f"Encoding voice for '{char_lower}' (extreme)...")
                        success = tts_obj.encode_voice(audio_path, ref_text=ref_text)
                        if not success:
                            print(f"Error: Failed to encode voice from {audio_path}")
                            for f in _design_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        fish_voice_data[char_lower] = {
                            "tokens": tts_obj.encoded_refs["tokens"].clone(),
                            "text": tts_obj.encoded_refs.get("text", "")
                        }
                    for char_lower, trained_file in trained_voice_refs.items():
                        print(f"Loading trained voice for '{char_lower}' (extreme) from: {trained_file}")
                        payload = _load_fish_voice(trained_file)
                        if payload is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            for f in _design_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        fish_voice_data[char_lower] = payload
                if has_tts_chars:
                    for char_lower, placeholder_path in _placeholder_paths.items():
                        print(f"Encoding voice design for '{char_lower}' (extreme)...")
                        success = tts_obj.encode_voice(placeholder_path, ref_text=placeholder_text)
                        if not success:
                            print(f"Error: Failed to encode voice design for '{char_lower}'")
                            for f in _design_cleanup:
                                if f and os.path.exists(f):
                                    try:
                                        os.unlink(f)
                                    except:
                                        pass
                            return False
                        fish_voice_data[char_lower] = {
                            "tokens": tts_obj.encoded_refs["tokens"].clone(),
                            "text": tts_obj.encoded_refs.get("text", "")
                        }
                for f in _design_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
            else:
                if has_tts_chars:
                    print("Loading Qwen-TTS VoiceDesign model...")
                    tts_design = QwenTTSVoiceDesign()
                    if tts_design.model is None:
                        print("Error: Failed to load VoiceDesign model")
                        return False
                if has_vc_chars:
                    print("Loading Qwen-TTS model...")
                    tts_obj = QwenTTS()
                    vc_voice_prompts = {}
                    for char_lower, audio_path in target_assignments.items():
                        print(f"Extracting voice for '{char_lower}'...")
                        ref_text = _transcribe_for_qwen_ref(audio_path)
                        success = tts_obj.extract_voice(audio_path, ref_text=ref_text if ref_text else None)
                        if not success:
                            print(f"Error: Failed to extract voice from {audio_path}")
                            return False
                        vc_voice_prompts[char_lower] = tts_obj.voice_prompt
                    for char_lower, trained_file in trained_voice_refs.items():
                        print(f"Loading trained voice for '{char_lower}' from: {trained_file}")
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        vc_voice_prompts[char_lower] = voice_items
                if has_tts_chars and tts_obj is None:
                    print("Loading Qwen-TTS model for voice stabilization...")
                    tts_obj = QwenTTS()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"voder_tts_dialogue_{timestamp}"
            if music_description:
                base_name += "_m"
            output_path = os.path.join(results_dir, f"{base_name}.wav")

            dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            dialogue_temp.close()

            has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
            has_effects = any(
                item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
                for item in dialogue_items
            ) if len(dialogue_items) > 0 else False

            if has_sfx or has_effects or has_vc_chars or has_tts_chars:
                success, msg = _assemble_enhanced_dialogue(
                    dialogue_items, voice_prompts, tts_design_obj=tts_design,
                    tts_vc_obj=tts_obj, vc_voice_data=vc_voice_prompts,
                    output_path=dialogue_temp.name, mode='tts',
                    sts_refs=sts_refs if sts_refs else None,
                    use_extreme=use_extreme, fish_voice_data=fish_voice_data
                )
                if not success:
                    print(f"Error: {msg}")
                    return False
            else:
                simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
                success, msg = tts_design.synthesize_dialogue(simple_items, voice_prompts, dialogue_temp.name)
                if not success:
                    print(f"Error: {msg}")
                    return False

            if music_description:
                if tts_design is not None:
                    del tts_design
                    tts_design = None
                if tts_obj is not None:
                    del tts_obj
                    tts_obj = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ace = AceStepWrapper(use_overdose=use_overdose)
                if ace.handler is None:
                    print("Error: Failed to load ACE-Step model")
                    return False
                success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec, reference_audio=reference_audio)
                del ace
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not success:
                    return False
                os.unlink(dialogue_temp.name)
            else:
                shutil.move(dialogue_temp.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            if tts_design is not None:
                del tts_design
            if tts_obj is not None:
                del tts_obj
            return True
        finally:
            for f in all_target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            if 'dialogue_temp' in dir() and os.path.exists(dialogue_temp.name):
                try:
                    os.unlink(dialogue_temp.name)
                except:
                    pass

def oneline_sts(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    is_music = params.get('is_music', False)
    is_mimic = params.get('is_mimic', False)
    no_music = params.get('nomusic', False)
    use_original = params.get('use_original', False)
    use_extreme = params.get('extreme', False)

    if 'base' not in params or len(params['base']) != 1:
        print("Error: STS mode requires exactly one 'base' parameter")
        return False
    if 'target' not in params or len(params['target']) != 1:
        print("Error: STS mode requires exactly one 'target' parameter")
        return False
    base_path = params['base'][0]
    target_value = params['target'][0]
    if not os.path.exists(base_path) and not is_youtube_url(base_path):
        print(f"Error: Base file not found: {base_path}")
        return False
    _target_cleanup = []
    use_first = params.get('use_first', False)
    target_multi = _parse_multi_refs(target_value)
    target_pre_cleaned = False
    if target_multi:
        resolved_target = _resolve_multi_refs(target_multi, _target_cleanup, use_first=use_first)
        if not resolved_target:
            return False
        target_pre_cleaned = True
    else:
        if use_first:
            print("Warning: 'first' keyword ignored (only one reference provided)")
        resolved_target, _cl = resolve_target_to_audio(target_value)
        if not resolved_target:
            return False
        _target_cleanup.extend(_cl)
    base_is_video = os.path.splitext(base_path)[1].lower() in VIDEO_EXTENSIONS
    base_original = base_path
    temp_base_extracted = None
    if base_is_video:
        if is_music or is_mimic:
            print("Error: Base input must be audio for this mode")
            return False
        print("Extracting audio from video...")
        temp_base_extracted = os.path.join(tempfile.gettempdir(), f"voder_cli_{int(time.time())}.wav")
        cmd = ['ffmpeg', '-i', base_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2', '-y', temp_base_extracted]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(temp_base_extracted):
            print("Error: Could not extract audio from video")
            return False
        base_path = temp_base_extracted
    target_path = resolved_target
    if is_music:
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        if use_original:
            print("Using original source audio (no SVS split)...")
            base_vocals = base_path
            base_music = None
        else:
            print("Extracting vocals from source...")
            base_vocals = svs_extract_vocals(base_path)
            _target_cleanup.append(base_vocals)
            print("Extracting music from source...")
            base_music = svs_extract_music(base_path)
            _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        if target_pre_cleaned:
            clean_vocal_target = target_path
        else:
            clean_vocal_target = svs_extract_vocals(target_path)
            _target_cleanup.append(clean_vocal_target)
        _extreme_cleanup = None
        if use_extreme:
            print("Running STS extreme pass on target reference...")
            extreme_result = _sts_extreme_pass(clean_vocal_target)
            if extreme_result:
                _extreme_cleanup = extreme_result
                clean_vocal_target = extreme_result
        print("Resampling inputs to 44100Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 44100:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if _extreme_cleanup and os.path.exists(_extreme_cleanup):
                os.unlink(_extreme_cleanup)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    else:
        if use_original:
            print("Using original source audio (no SVS split)...")
            base_vocals = base_path
            base_music = None
        else:
            print("Extracting vocals from source...")
            base_vocals = svs_extract_vocals(base_path)
            _target_cleanup.append(base_vocals)
            base_music = None
            if not no_music:
                print("Extracting music from source...")
                base_music = svs_extract_music(base_path)
                _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        if target_pre_cleaned:
            clean_vocal_target = target_path
        else:
            clean_vocal_target = svs_extract_vocals(target_path)
            _target_cleanup.append(clean_vocal_target)
        _extreme_cleanup = None
        if use_extreme:
            print("Running STS extreme pass on target reference...")
            extreme_result = _sts_extreme_pass(clean_vocal_target)
            if extreme_result:
                _extreme_cleanup = extreme_result
                clean_vocal_target = extreme_result
        print("Loading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Resampling inputs to 22050Hz...")
        import torchaudio
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 22050:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 22050)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_22k.name,
                convert_style=is_mimic
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            print("Upsampling output to 44100Hz...")
            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
            if sr_out != 44100:
                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                waveform_out = resampler_out(waveform_out)
            temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_output_44k.name, waveform_out, 44100)
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not no_music and base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            elif base_is_video:
                print("Merging converted audio with video...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.mp4")
                ret = os.system(f'ffmpeg -y -i "{base_original}" -i "{temp_output_44k.name}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Error: Failed to merge audio with video")
                    return False
            else:
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_22k.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if _extreme_cleanup and os.path.exists(_extreme_cleanup):
                os.unlink(_extreme_cleanup)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

def oneline_ttm(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    _is_remix = params.get('is_remix', False)
    use_overdose = params.get('overdose', False)
    use_vc = params.get('vc', False)

    if use_vc:
        if _is_remix:
            print("Error: VC cannot be used with remix")
            return False
        if params.get('is_repaint', False):
            print("Error: VC cannot be used with repaint")
            return False
        if 'lyrics' not in params or len(params['lyrics']) != 1:
            print("Error: TTM VC requires exactly one 'lyrics' parameter")
            return False
        if 'styling' not in params or len(params['styling']) != 1:
            print("Error: TTM VC requires exactly one 'styling' parameter")
            return False
        if 'duration' not in params:
            print("Error: TTM VC requires duration (10-300 seconds)")
            return False
        duration = params['duration']
        if not (10 <= duration <= 300):
            print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
            return False
        clone_path = params.get('clone_path')
        if not clone_path:
            print("Error: TTM VC requires clone source path")
            return False
        lyrics = params['lyrics'][0].replace('\\n', '\n')
        style = params['styling'][0].replace('\\n', '\n')
        ttm_valid, _ = _validate_text_language(lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
        if not ttm_valid:
            return False
        _vc_cleanup = []
        use_first = params.get('clone_first', False)
        clone_multi = _parse_multi_refs(clone_path)
        clone_pre_cleaned = False
        if clone_multi:
            clean_vocal = _resolve_multi_refs(clone_multi, _vc_cleanup, use_first=use_first)
            if not clean_vocal:
                return False
            clone_pre_cleaned = True
        else:
            if use_first:
                print("Warning: 'first' keyword ignored (only one reference provided)")
            if not os.path.exists(clone_path) and not is_youtube_url(clone_path):
                print(f"Error: Clone source not found: {clone_path}")
                return False
            resolved_audio, cleanup = resolve_target_to_audio(clone_path)
            if resolved_audio is None:
                print("Error: Could not resolve clone source")
                return False
            _vc_cleanup.extend(cleanup)
            clean_vocal = svs_extract_vocals(resolved_audio)
            if clean_vocal != resolved_audio and clean_vocal not in _vc_cleanup:
                _vc_cleanup.append(clean_vocal)
            if resolved_audio not in _vc_cleanup and resolved_audio != clean_vocal:
                _vc_cleanup.append(resolved_audio)
        reference_audio = None
        _target_vals = params.get('target', [])
        if _target_vals:
            if len(_target_vals) >= 2:
                ref_type = _target_vals[0].lower()
                ref_path_raw = _target_vals[1]
                if ref_type not in ('voice', 'music'):
                    ref_type = 'asis'
                    ref_path_raw = _target_vals[0]
            else:
                ref_type = 'asis'
                ref_path_raw = _target_vals[0]
            _vc_tr, ref_path, _vc_stems = _parse_ref_time_spec(ref_path_raw)
            if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
                print(f"Error: Reference target not found: {ref_path}")
                for f in _vc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            resolved_ref, ref_cleanup = resolve_target_to_audio(ref_path)
            if resolved_ref is None:
                print("Error: Could not resolve reference target")
                for f in _vc_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            _vc_cleanup.extend(ref_cleanup)
            if ref_type == 'voice':
                processed_ref = svs_extract_vocals(resolved_ref)
            elif ref_type == 'music':
                processed_ref = svs_extract_music(resolved_ref)
            else:
                processed_ref = resolved_ref
            if processed_ref and processed_ref != resolved_ref:
                if processed_ref not in _vc_cleanup:
                    _vc_cleanup.append(processed_ref)
            if _vc_stems:
                print(f"Extracting stem(s): {', '.join(_vc_stems)}...")
                processed_ref = _extract_acestep_stems(processed_ref, _vc_stems, _vc_cleanup)
            if _vc_tr:
                processed_ref = _extract_ref_segments(processed_ref, _vc_tr, 30, _vc_cleanup)
            reference_audio = processed_ref
            if ref_type == 'asis':
                print("Using reference audio for music generation (as-is)")
            else:
                print(f"Using reference audio for music generation: {ref_type}")
        print("Loading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_ttm_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_clone_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            print(f"Generating music ({duration}s duration)...")
            if reference_audio:
                print(f"Using reference audio: {reference_audio}")
            success = ace_step.generate(
                lyrics=lyrics,
                style_prompt=style,
                output_path=temp_ttm_output.name,
                duration=duration,
                reference_audio=reference_audio
            )
            if not success:
                print("Error: Music generation failed")
                return False
            print("Extracting vocals from TTM output...")
            ttm_vocals = svs_extract_vocals(temp_ttm_output.name)
            if ttm_vocals and ttm_vocals != temp_ttm_output.name:
                _vc_cleanup.append(ttm_vocals)
            else:
                ttm_vocals = temp_ttm_output.name
            print("Extracting music from TTM output...")
            ttm_music = svs_extract_music(temp_ttm_output.name)
            if ttm_music and ttm_music != temp_ttm_output.name:
                _vc_cleanup.append(ttm_music)
            else:
                ttm_music = None
            print("Resampling TTM vocals to 44100Hz...")
            waveform_vocals, sr_vocals = torchaudio.load(ttm_vocals)
            if sr_vocals != 44100:
                resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
                waveform_vocals = resampler_vocals(waveform_vocals)
            torchaudio.save(temp_ttm_44k.name, waveform_vocals, 44100)
            print("Resampling clone voice to 44100Hz...")
            waveform_clone, sr_clone = torchaudio.load(clean_vocal)
            if sr_clone != 44100:
                resampler_clone = torchaudio.transforms.Resample(sr_clone, 44100)
                waveform_clone = resampler_clone(waveform_clone)
            torchaudio.save(temp_clone_44k.name, waveform_clone, 44100)
            print("Clearing ACE-Step from memory...")
            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Loading Seed-VC v1 model...")
            seed_vc = SeedVCV1()
            if seed_vc.model is None:
                print("Error: Failed to load Seed-VC v1 model")
                return False
            print("Converting voice...")
            vc_success = seed_vc.convert(
                source_path=temp_ttm_44k.name,
                reference_path=temp_clone_44k.name,
                output_path=temp_vc_output.name
            )
            if not vc_success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ttm_music:
                print("Mixing converted vocals with TTM music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_vc_output.name}" -i "{ttm_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_vc_output.name, output_path)
            else:
                print("Saving output...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                shutil.copy(temp_vc_output.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_ttm_output.name, temp_ttm_44k.name, temp_clone_44k.name, temp_vc_output.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    if _is_remix:
        _remix_entries = params.get('remix_entries', [])
        if not _remix_entries:
            print("Error: Remix requires at least one source path")
            return False
        if 'styling' not in params or len(params['styling']) != 1:
            print("Error: TTM remix requires 'styling' parameter")
            return False
        style = params['styling'][0].replace('\\n', '\n')
        _remix_lyrics = "..."
        if 'lyrics' in params and len(params['lyrics']) == 1:
            _remix_lyrics = params['lyrics'][0].replace('\\n', '\n')
            if _remix_lyrics.strip() and _remix_lyrics.strip() != '...':
                ttm_valid, _ = _validate_text_language(_remix_lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
                if not ttm_valid:
                    return False
        _remix_cleanup = []
        resolved_audio, src_cl = _compose_sources(_remix_entries, results_dir)
        if resolved_audio is None:
            print("Error: Could not resolve remix source(s)")
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        _remix_cleanup.extend(src_cl)
        _remix_ref_entries = params.get('ref_entries', [])
        _remix_ref_audio = None
        if _remix_ref_entries:
            _remix_ref_audio, ref_cl = _compose_refs(_remix_ref_entries, results_dir)
            _remix_cleanup.extend(ref_cl)
        _cover_strength = 0.4
        _bias_raw = params.get('bias_val')
        if _bias_raw is not None:
            try:
                _bv = int(_bias_raw)
                if 0 <= _bv <= 100:
                    if _bv == 0 or _bv == 100:
                        _cover_strength = _bv / 100.0
                    elif _bv % 10 == 5:
                        _cover_strength = (_bv - 5) / 100.0
                    else:
                        _cover_strength = (round(_bv / 10) * 10) / 100.0
            except (ValueError, TypeError):
                pass
        original_name = os.path.splitext(os.path.basename(_remix_entries[0][1]))[0]
        original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
        print("Loading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        try:
            print("Generating remix...")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(results_dir, f"voder_ttm_remix_{original_name}_{timestamp}.wav")
            success = ace_step.cover(
                src_audio=resolved_audio,
                style_prompt=style,
                output_path=output_path,
                cover_strength=_cover_strength,
                reference_audio=_remix_ref_audio,
                lyrics=_remix_lyrics
            )
            if not success:
                print("Error: Remix generation failed")
                return False
            print(f"\n✓ Success! Output saved to: {output_path}")
            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for f in _remix_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    _is_repaint = params.get('is_repaint', False)
    _repaint_path = params.get('repaint_path')
    _time_range = params.get('time_range')
    _repaint_multipass = params.get('repaint_multipass')
    _repaint_source_prefix = params.get('repaint_source_prefix')

    if _is_repaint:
        if not _repaint_path:
            print("Error: Repaint requires a source path")
            return False
        if not os.path.exists(_repaint_path) and not is_youtube_url(_repaint_path):
            print(f"Error: Repaint source not found: {_repaint_path}")
            return False
        _repaint_cleanup = []
        resolved_audio, cleanup = resolve_target_to_audio(_repaint_path)
        if resolved_audio is None:
            print("Error: Could not resolve repaint source")
            return False
        _repaint_cleanup.extend(cleanup)
        if _repaint_source_prefix:
            _svs_timestamp = time.strftime("%Y%m%d_%H%M%S")
            if _repaint_source_prefix == 'voice':
                _svs_result = svs_extract_vocals(resolved_audio)
            else:
                _svs_result = svs_extract_music(resolved_audio)
            if _svs_result and _svs_result != resolved_audio:
                _repaint_cleanup.append(_svs_result)
                resolved_audio = _svs_result
        if _repaint_multipass:
            _parsed_passes = []
            for _pi, _spec in enumerate(_repaint_multipass):
                _parsed, _err = _parse_repaint_pass_spec(_spec)
                if _err:
                    print(f"Error: Repaint pass {_pi + 1}: {_err}")
                    for f in _repaint_cleanup:
                        if f and os.path.exists(f):
                            try: os.unlink(f)
                            except: pass
                    return False
                _parsed_passes.append(_parsed)
            original_name = os.path.splitext(os.path.basename(_repaint_path))[0]
            original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
            print("Loading ACE-Step model...")
            ace_step = AceStepWrapper(use_overdose=use_overdose)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            _current_source = resolved_audio
            _last_output = None
            _intermediate_files = []
            _total_passes = len(_parsed_passes)
            try:
                for _pi, _pass in enumerate(_parsed_passes):
                    _start_sec = _pass['start']
                    _end_sec = _pass['end']
                    try:
                        import soundfile as sf
                        _audio_info = sf.info(_current_source)
                        _max_duration = _audio_info.duration
                    except Exception:
                        print(f"Error: Could not read audio duration for pass {_pi + 1}")
                        return False
                    if _start_sec > _max_duration:
                        print(f"Error: Pass {_pi + 1}: Start time {_start_sec}s exceeds audio duration {_max_duration:.1f}s")
                        return False
                    if _end_sec > _max_duration:
                        print(f"Pass {_pi + 1}: End time {_end_sec}s exceeds audio duration, clamping to {_max_duration:.1f}s")
                        _end_sec = _max_duration
                    if _start_sec >= _end_sec:
                        print(f"Error: Pass {_pi + 1}: Start time must be less than end time after clamping")
                        return False
                    _pass_style = (_pass.get('styling') or '...')
                    _pass_lyrics = (_pass.get('lyrics') or '...')
                    _cover_strength = 0.4
                    if _pass.get('bias') is not None:
                        try:
                            _bv = int(_pass['bias'])
                            if 0 <= _bv <= 100:
                                if _bv == 0 or _bv == 100:
                                    _cover_strength = _bv / 100.0
                                elif _bv % 10 == 5:
                                    _cover_strength = (_bv - 5) / 100.0
                                else:
                                    _cover_strength = (round(_bv / 10) * 10) / 100.0
                        except (ValueError, TypeError):
                            pass
                    _pass_ref_audio = None
                    if _pass.get('references'):
                        _pass_ref_audio, _pass_ref_cl = _compose_refs(_pass['references'], results_dir)
                        _repaint_cleanup.extend(_pass_ref_cl)
                    _start_int = int(_start_sec)
                    _end_int = int(_end_sec)
                    _pass_num = _pi + 1
                    print(f"Repainting pass {_pass_num}/{_total_passes}: {_start_int}s - {_end_int}s...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    if _pi < _total_passes - 1:
                        output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_pass{_pass_num}_{timestamp}.wav")
                    else:
                        output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_{timestamp}.wav")
                    success = ace_step.repaint(
                        src_audio=_current_source,
                        style_prompt=_pass_style,
                        output_path=output_path,
                        repaint_start=_start_sec,
                        repaint_end=_end_sec,
                        lyrics=_pass_lyrics,
                        cover_strength=_cover_strength,
                        reference_audio=_pass_ref_audio
                    )
                    if not success:
                        print(f"Error: Repaint pass {_pass_num} failed")
                        return False
                    if _last_output:
                        _intermediate_files.append(_last_output)
                    _last_output = output_path
                    _current_source = output_path
                for f in _intermediate_files:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                print(f"\n✓ Success! Output saved to: {_last_output}")
                del ace_step
                ace_step = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            finally:
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
        else:
            if _time_range is None:
                print("Error: Repaint requires time range (e.g., time:20-80)")
                return False
            if 'styling' not in params or len(params['styling']) != 1:
                print("Error: TTM repaint requires 'styling' parameter")
                return False
            _time_parts = _time_range.split('-')
            if len(_time_parts) != 2:
                print(f"Error: Invalid time format '{_time_range}', expected time:start-end")
                return False
            try:
                _start_sec = float(_time_parts[0].strip())
                _end_sec = float(_time_parts[1].strip())
            except ValueError:
                print(f"Error: Time values must be numbers, got '{_time_range}'")
                return False
            if _start_sec < 0:
                print("Error: Start time cannot be negative")
                return False
            if _end_sec <= 0:
                print("Error: End time must be greater than 0")
                return False
            if _start_sec == _end_sec:
                print("Error: Start and end time cannot be the same")
                return False
            if _start_sec >= _end_sec:
                print("Error: Start time must be less than end time")
                return False
            style = params['styling'][0].replace('\\n', '\n')
            _lyrics_content = "..."
            if 'lyrics' in params and len(params['lyrics']) == 1:
                _lyrics_content = params['lyrics'][0].replace('\\n', '\n')
                if _lyrics_content.strip() and _lyrics_content.strip() != '...':
                    ttm_valid, _ = _validate_text_language(_lyrics_content, SUPPORTED_ACESTEP_LANGS, "TTM")
                    if not ttm_valid:
                        return False
            _rp_ref_entries = params.get('ref_entries', [])
            _repaint_ref_audio = None
            if _rp_ref_entries:
                _repaint_ref_audio, ref_cl = _compose_refs(_rp_ref_entries, results_dir)
                _repaint_cleanup.extend(ref_cl)
            try:
                import soundfile as sf
                _audio_info = sf.info(resolved_audio)
                _max_duration = _audio_info.duration
            except Exception:
                print("Error: Could not read audio duration")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            if _start_sec > _max_duration:
                print(f"Error: Start time {_start_sec}s exceeds audio duration {_max_duration:.1f}s")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            if _end_sec > _max_duration:
                print(f"End time {_end_sec}s exceeds audio duration, clamping to {_max_duration:.1f}s")
                _end_sec = _max_duration
            _cover_strength = 0.4
            _bias_raw = params.get('bias_val')
            if _bias_raw is not None:
                try:
                    _bv = int(_bias_raw)
                    if 0 <= _bv <= 100:
                        if _bv == 0 or _bv == 100:
                            _cover_strength = _bv / 100.0
                        elif _bv % 10 == 5:
                            _cover_strength = (_bv - 5) / 100.0
                        else:
                            _cover_strength = (round(_bv / 10) * 10) / 100.0
                except (ValueError, TypeError):
                    pass
            original_name = os.path.splitext(os.path.basename(_repaint_path))[0]
            original_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', original_name)
            print("Loading ACE-Step model...")
            ace_step = AceStepWrapper(use_overdose=use_overdose)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model")
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass
                return False
            try:
                _start_int = int(_start_sec)
                _end_int = int(_end_sec)
                print(f"Repainting {_start_int}s - {_end_int}s...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_repaint_{original_name}_{_start_int}-{_end_int}_{timestamp}.wav")
                success = ace_step.repaint(
                    src_audio=resolved_audio,
                    style_prompt=style,
                    output_path=output_path,
                    repaint_start=_start_sec,
                    repaint_end=_end_sec,
                    lyrics=_lyrics_content,
                    cover_strength=_cover_strength,
                    reference_audio=_repaint_ref_audio
                )
                if not success:
                    print("Error: Repaint generation failed")
                    return False
                print(f"\n✓ Success! Output saved to: {output_path}")
                del ace_step
                ace_step = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
            finally:
                for f in _repaint_cleanup:
                    if f and os.path.exists(f):
                        try: os.unlink(f)
                        except: pass

    if 'lyrics' not in params or len(params['lyrics']) != 1:
        print("Error: TTM mode requires exactly one 'lyrics' parameter")
        return False
    if 'styling' not in params or len(params['styling']) != 1:
        print("Error: TTM mode requires exactly one 'styling' parameter")
        return False
    if 'duration' not in params:
        print("Error: TTM mode requires duration (10-300 seconds)")
        return False
    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False
    lyrics = params['lyrics'][0].replace('\\n', '\n')
    style = params['styling'][0].replace('\\n', '\n')
    ttm_valid, _ = _validate_text_language(lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
    if not ttm_valid:
        return False
    _ttm_cleanup = []
    reference_audio = None
    _target_vals = params.get('target', [])
    if _target_vals:
        if len(_target_vals) >= 2:
            ref_type = _target_vals[0].lower()
            ref_path_raw = _target_vals[1]
            if ref_type not in ('voice', 'music'):
                ref_type = 'asis'
                ref_path_raw = _target_vals[0]
        else:
            ref_type = 'asis'
            ref_path_raw = _target_vals[0]
        _ttm_tr, ref_path, _ttm_stems = _parse_ref_time_spec(ref_path_raw)
        if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
            print(f"Error: Reference target not found: {ref_path}")
            return False
        resolved_audio, cleanup = resolve_target_to_audio(ref_path)
        if resolved_audio is None:
            print("Error: Could not resolve reference target")
            return False
        _ttm_cleanup.extend(cleanup)
        if ref_type == 'voice':
            processed = svs_extract_vocals(resolved_audio)
        elif ref_type == 'music':
            processed = svs_extract_music(resolved_audio)
        else:
            processed = resolved_audio
        if processed != resolved_audio and processed not in _ttm_cleanup:
            _ttm_cleanup.append(processed)
        if _ttm_stems:
            print(f"Extracting stem(s): {', '.join(_ttm_stems)}...")
            processed = _extract_acestep_stems(processed, _ttm_stems, _ttm_cleanup)
        if _ttm_tr:
            processed = _extract_ref_segments(processed, _ttm_tr, 30, _ttm_cleanup)
        reference_audio = processed
    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper(use_overdose=use_overdose)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    try:
        print(f"Generating music ({duration}s duration)...")
        if reference_audio:
            print(f"Using reference audio: {reference_audio}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=output_path,
            duration=duration,
            reference_audio=reference_audio
        )
        if not success:
            print("Error: Music generation failed")
            return False
        print(f"\n✓ Success! Output saved to: {output_path}")
        return True
    finally:
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_voice(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    use_overdose = params.get('overdose', False)

    if 'lyrics' not in params or len(params['lyrics']) != 1:
        print("Error: TTM voice requires exactly one 'lyrics' parameter")
        return False
    if 'styling' not in params or len(params['styling']) != 1:
        print("Error: TTM voice requires exactly one 'styling' parameter")
        return False
    if 'duration' not in params:
        print("Error: TTM voice requires duration (10-300 seconds)")
        return False
    duration = params['duration']
    if not (10 <= duration <= 300):
        print(f"Error: Duration must be between 10 and 300 seconds, got {duration}")
        return False
    lyrics = params['lyrics'][0].replace('\\n', '\n')
    style = params['styling'][0].replace('\\n', '\n')
    ttm_valid, _ = _validate_text_language(lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
    if not ttm_valid:
        return False
    _cleanup = []
    reference_audio = None
    _target_vals = params.get('target', [])
    if _target_vals:
        if len(_target_vals) >= 2:
            ref_type = _target_vals[0].lower()
            ref_path_raw = _target_vals[1]
            if ref_type not in ('voice', 'music'):
                ref_type = 'asis'
                ref_path_raw = _target_vals[0]
        else:
            ref_type = 'asis'
            ref_path_raw = _target_vals[0]
        _tr, ref_path, _ref_stems = _parse_ref_time_spec(ref_path_raw)
        if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
            print(f"Error: Reference target not found: {ref_path}")
            return False
        resolved_audio, cleanup = resolve_target_to_audio(ref_path)
        if resolved_audio is None:
            print("Error: Could not resolve reference target")
            return False
        _cleanup.extend(cleanup)
        if ref_type == 'voice':
            processed = svs_extract_vocals(resolved_audio)
        elif ref_type == 'music':
            processed = svs_extract_music(resolved_audio)
        else:
            processed = resolved_audio
        if processed != resolved_audio and processed not in _cleanup:
            _cleanup.append(processed)
        if _ref_stems:
            print(f"Extracting stem(s): {', '.join(_ref_stems)}...")
            processed = _extract_acestep_stems(processed, _ref_stems, _cleanup)
        if _tr:
            processed = _extract_ref_segments(processed, _tr, 30, _cleanup)
        reference_audio = processed
    print("Loading ACE-Step model...")
    ace_step = AceStepWrapper(use_overdose=use_overdose)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    try:
        print(f"Generating music ({duration}s duration)...")
        if reference_audio:
            print(f"Using reference audio: {reference_audio}")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=temp_ttm_output.name,
            duration=duration,
            reference_audio=reference_audio
        )
        if not success:
            print("Error: Music generation failed")
            return False
        del ace_step
        ace_step = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Extracting vocals from TTM output...")
        vocals_path = svs_extract_vocals(temp_ttm_output.name)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_voice_{timestamp}.wav")
        if vocals_path and vocals_path != temp_ttm_output.name:
            shutil.copy(vocals_path, output_path)
            if vocals_path not in _cleanup:
                _cleanup.append(vocals_path)
        else:
            shutil.copy(temp_ttm_output.name, output_path)
        print(f"\n✓ Success! Voice output saved to: {output_path}")
        return True
    finally:
        if os.path.exists(temp_ttm_output.name):
            os.remove(temp_ttm_output.name)
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_complete(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    use_vocals = params.get('use_vocals', False)
    use_music = params.get('use_music', False)
    use_source = params.get('use_source', False)
    want_video = params.get('want_video', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Complete task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    sfx_specs_raw = params.get('sfx_specs', [])
    has_instruments = bool(instruments_raw)
    has_sfx = bool(sfx_specs_raw)

    if not has_instruments and not has_sfx:
        print('Error: Complete task requires instruments and/or "sfx:" specs (e.g., add "drums bass" or "sfx:thunder/10-5")')
        return False

    track_classes = None
    if has_instruments:
        track_classes, unknown = resolve_acestep_tracks(instruments_raw)
        if unknown is not None and len(unknown) > 0:
            print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
            print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
            print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
            return False
        if track_classes is None:
            print("Error: No valid tracks specified")
            return False

    noblend = params.get('noblend', False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    video_path = None
    downloaded_video = None
    is_video_source = False

    ext = os.path.splitext(source_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        is_video_source = True

    if is_youtube_url(source_path):
        if want_video:
            print(f"Downloading video from URL: {source_path}")
            downloaded_video, video_title = download_youtube_video(source_path, results_dir)
            if downloaded_video is None:
                print(f"Error: {video_title}")
                return False
            video_path = downloaded_video
            is_video_source = True
            original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
            temp_audio = os.path.join(results_dir, f'_ttm_complete_dl_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from downloaded video")
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            print(f"Downloading audio from URL: {source_path}")
            resolved, cleanup = resolve_target_to_audio(source_path)
            if resolved is None:
                print("Error: Could not download audio from URL")
                return False
            _cleanup.extend(cleanup)
            source_audio = resolved
    elif is_video_source:
        video_path = source_path
        temp_audio = os.path.join(results_dir, f'_ttm_complete_vid_{timestamp}.wav')
        ret = os.system(f'ffmpeg -y -i "{video_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
        if ret != 0 or not os.path.exists(temp_audio):
            print("Error: Failed to extract audio from video")
            return False
        source_audio = temp_audio
        _cleanup.append(temp_audio)
    else:
        valid, msg = validate_audio_file(source_path)
        if not valid:
            print(f"Error: {msg}")
            return False
        source_audio = source_path

    source_duration = _get_audio_duration(source_audio)

    sfx_specs = None
    if sfx_specs_raw:
        sfx_specs, sfx_err = _parse_sfx_specs(sfx_specs_raw, source_duration)
        if sfx_err:
            print(f"Error: {sfx_err}")
            return False

    actual_source = source_audio
    if use_vocals:
        print("Extracting vocals via SVS...")
        vocals = svs_extract_vocals(source_audio)
        if vocals != source_audio and vocals not in _cleanup:
            _cleanup.append(vocals)
        actual_source = vocals
    elif use_music:
        print("Extracting music (removing vocals) via SVS...")
        music = svs_extract_music(source_audio)
        if music != source_audio and music not in _cleanup:
            _cleanup.append(music)
        actual_source = music
    else:
        print("Using source audio as-is")

    if use_source and not use_vocals and not use_music:
        print("Warning: usrc has no effect without voice or music (only one source to blend with)")
        use_source = False

    reference_audio = None
    _ref_entries = params.get('ref_entries', [])
    if _ref_entries:
        reference_audio, ref_cl = _compose_refs(_ref_entries, results_dir)
        _cleanup.extend(ref_cl)

    try:
        output_ext = '.wav'
        if want_video and video_path:
            output_ext = '.mp4'
        elif want_video and not video_path:
            print("Warning: 'video' specified but source is an audio file (not video). Outputting as WAV.")
        _noblend_tag = '_noblend_' if noblend else ''
        _usrc_tag = '_usrc_' if use_source else ''
        output_filename = f'voder_ttm_complete_{original_name}{_noblend_tag}{_usrc_tag}{timestamp}{output_ext}'
        output_path = os.path.join(results_dir, output_filename)

        blended_path = actual_source

        if has_instruments:
            print(f"Tracks to add: {', '.join(track_classes)}")
            print("Loading ACE-Step XL-Base model (complete task)...")
            print("Note: Complete task uses the base model (50 inference steps), this may take a while...")

            ace_step = AceStepWrapper(use_overdose=True, complete_mode=True)
            if ace_step.handler is None:
                print("Error: Failed to load ACE-Step model for complete task")
                for f in _cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            temp_gen_wav = os.path.join(results_dir, f'_ttm_complete_gen_{timestamp}.wav')
            print(f"Completing track (adding {len(track_classes)} instruments)...")
            _styling = None
            if 'styling' in params and len(params['styling']) == 1:
                _styling = params['styling'][0].replace('\\n', '\n')
            success = ace_step.complete(
                src_audio=actual_source,
                track_classes=track_classes,
                output_path=temp_gen_wav,
                styling=_styling,
                reference_audio=reference_audio
            )

            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not success or not os.path.exists(temp_gen_wav):
                if os.path.exists(temp_gen_wav):
                    try:
                        os.unlink(temp_gen_wav)
                    except:
                        pass
                print("Error: Complete generation failed")
                return False

            if noblend:
                blended_path = temp_gen_wav
                _cleanup.append(temp_gen_wav)
            else:
                blend_source = source_audio if use_source else actual_source
                if use_source:
                    print("Blending completed audio with original source (usrc)...")
                else:
                    print("Blending completed audio with source...")
                temp_blend_wav = os.path.join(results_dir, f'_ttm_complete_blend_{timestamp}.wav')
                ret = os.system(f'ffmpeg -y -i "{temp_gen_wav}" -i "{blend_source}" -filter_complex amix=inputs=2:duration=longest "{temp_blend_wav}" 2>/dev/null')
                if os.path.exists(temp_gen_wav):
                    try:
                        os.unlink(temp_gen_wav)
                    except:
                        pass
                if ret == 0 and os.path.exists(temp_blend_wav):
                    blended_path = temp_blend_wav
                    _cleanup.append(temp_blend_wav)
                else:
                    print("Warning: Blend failed, using generated audio as-is")
                    blended_path = temp_gen_wav
                    _cleanup.append(temp_gen_wav)
                    if os.path.exists(temp_blend_wav):
                        try:
                            os.unlink(temp_blend_wav)
                        except:
                            pass

        if sfx_specs:
            print(f"Applying {len(sfx_specs)} SFX overlay(s)...")
            sfx_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sfx_temp.close()
            sfx_ok = _generate_and_overlay_sfx(blended_path, sfx_specs, sfx_temp.name)
            if sfx_ok and os.path.exists(sfx_temp.name):
                blended_path = sfx_temp.name
                _cleanup.append(sfx_temp.name)
            else:
                if os.path.exists(sfx_temp.name):
                    try:
                        os.unlink(sfx_temp.name)
                    except:
                        pass
                print("Warning: SFX overlay failed, using audio without SFX")

        if want_video and video_path:
            print("Merging audio with video...")
            mux_cmd = ['ffmpeg', '-i', video_path, '-i', blended_path,
                        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                        '-shortest', '-y', output_path]
            mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
            if mux_result.returncode != 0:
                print(f"Error: Video muxing failed: {mux_result.stderr}")
                return False
        else:
            shutil.copy2(blended_path, output_path)

        print(f"\nSuccess! Output saved to: {output_path}")
        return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        if downloaded_video and os.path.exists(downloaded_video):
            try:
                os.remove(downloaded_video)
            except:
                pass

def oneline_ttm_lego(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    use_vocals = params.get('use_vocals', False)
    use_music = params.get('use_music', False)
    mix_mode = params.get('mix_mode', False)
    blend_mode = params.get('blend_mode', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Lego task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    if not instruments_raw:
        print('Error: Lego task requires instruments (e.g., make "drums bass" or make "everything")')
        return False

    track_classes, unknown = resolve_acestep_tracks(instruments_raw)
    if unknown is not None and len(unknown) > 0:
        print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
        print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
        print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
        return False
    if track_classes is None:
        print("Error: No valid tracks specified")
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    if is_youtube_url(source_path):
        print(f"Downloading audio from URL: {source_path}")
        resolved, cleanup = resolve_target_to_audio(source_path)
        if resolved is None:
            print("Error: Could not download audio from URL")
            return False
        _cleanup.extend(cleanup)
        source_audio = resolved
    else:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            temp_audio = os.path.join(results_dir, f'_lego_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{source_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from video")
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            valid, msg = validate_audio_file(source_path)
            if not valid:
                print(f"Error: {msg}")
                return False
            source_audio = source_path

    actual_source = source_audio
    if use_vocals:
        print("Extracting vocals via SVS...")
        vocals = svs_extract_vocals(source_audio)
        if vocals != source_audio and vocals not in _cleanup:
            _cleanup.append(vocals)
        actual_source = vocals
    elif use_music:
        print("Extracting music (removing vocals) via SVS...")
        music = svs_extract_music(source_audio)
        if music != source_audio and music not in _cleanup:
            _cleanup.append(music)
        actual_source = music
    else:
        print("Using source audio as-is")

    _ref_entries = params.get('ref_entries', [])
    track_set = set(track_classes)
    stem_refs = {}
    fallback_ref = None
    if _ref_entries:
        ref_cache = {}
        num_ref_entries = len(_ref_entries)
        lego_slot_max = 30 // max(1, num_ref_entries)
        for entry in _ref_entries:
            sv_type = entry[0]
            raw = entry[1]
            lego_tr = entry[2] if len(entry) > 2 else None
            stem_name, ref_path = parse_ref_raw(raw)
            cache_key = (ref_path, sv_type)
            if cache_key not in ref_cache:
                if not os.path.exists(ref_path) and not is_youtube_url(ref_path):
                    print(f"Warning: Reference not found: {ref_path}, skipping")
                    continue
                if is_youtube_url(ref_path):
                    print(f"Downloading reference audio from URL: {ref_path}")
                    resolved_ref, ref_cl = resolve_target_to_audio(ref_path)
                    if resolved_ref is None:
                        print(f"Warning: Could not download reference, skipping")
                        continue
                    _cleanup.extend(ref_cl)
                    ref_audio = resolved_ref
                else:
                    r_ext = os.path.splitext(ref_path)[1].lower()
                    if r_ext in VIDEO_EXTENSIONS:
                        ref_temp = os.path.join(results_dir, f'_lego_ref_vid_{timestamp}.wav')
                        ret = os.system(f'ffmpeg -y -i "{ref_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{ref_temp}" 2>/dev/null')
                        if ret != 0 or not os.path.exists(ref_temp):
                            print("Warning: Failed to extract audio from reference video, skipping")
                            continue
                        ref_audio = ref_temp
                        _cleanup.append(ref_temp)
                    else:
                        valid, msg = validate_audio_file(ref_path)
                        if not valid:
                            print(f"Warning: Invalid reference file: {msg}")
                            continue
                        ref_audio = ref_path
                if sv_type == 'voice':
                    ref_processed = svs_extract_vocals(ref_audio)
                elif sv_type == 'music':
                    ref_processed = svs_extract_music(ref_audio)
                else:
                    ref_processed = ref_audio
                if ref_processed and ref_processed != ref_audio:
                    if ref_processed not in _cleanup:
                        _cleanup.append(ref_processed)
                if lego_tr:
                    ref_processed = _extract_ref_segments(ref_processed, lego_tr, lego_slot_max, _cleanup)
                if ref_audio not in _cleanup and ref_audio != ref_processed:
                    _cleanup.append(ref_audio)
                ref_cache[cache_key] = ref_processed
            resolved_audio = ref_cache[cache_key]
            if stem_name is None:
                fallback_ref = resolved_audio
            else:
                if stem_name == 'everything':
                    stems = sorted(VALID_ACESTEP_TRACKS)
                elif stem_name == 'instruments':
                    stems = sorted(ACESTEP_INSTRUMENT_TRACKS)
                elif stem_name == 'voices':
                    stems = sorted(ACESTEP_VOICE_TRACKS)
                elif stem_name in track_set:
                    stems = [stem_name]
                else:
                    continue
                for s in stems:
                    if s in track_set:
                        stem_refs[s] = resolved_audio
        if stem_refs or fallback_ref is not None:
            refd = list(stem_refs.keys())
            if fallback_ref is not None:
                unrefd = [t for t in track_classes if t not in stem_refs]
                if unrefd:
                    print(f"References loaded: {len(refd)} specific, fallback for {len(unrefd)} more")
                else:
                    print(f"References loaded: {len(refd)} specific")
            else:
                print(f"References loaded: {len(refd)} specific, no fallback")

    print(f"Tracks to generate ({len(track_classes)}): {', '.join(track_classes)}")
    print("Loading ACE-Step XL-Base model (lego task)...")

    ace_step = AceStepWrapper(use_overdose=False, complete_mode=True)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model for lego task")
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False

    _styling = None
    if 'styling' in params and len(params['styling']) == 1:
        _styling = params['styling'][0].replace('\\n', '\n')

    try:
        generated_files = []
        all_succeeded = True
        for idx, track in enumerate(track_classes):
            track_ref = stem_refs.get(track, fallback_ref)
            if track_ref:
                print(f"\n[{idx+1}/{len(track_classes)}] Generating {track} (with reference)...")
            else:
                print(f"\n[{idx+1}/{len(track_classes)}] Generating {track}...")
            temp_output = os.path.join(results_dir, f'_lego_tmp_{track}_{timestamp}.wav')
            success = ace_step.lego(
                src_audio=actual_source,
                track_name=track,
                output_path=temp_output,
                styling=_styling,
                reference_audio=track_ref
            )
            if success:
                generated_files.append(temp_output)
                print(f"  {track} generated successfully")
            else:
                print(f"  Failed to generate {track}")
                all_succeeded = False
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass

        if not generated_files:
            print("Error: No tracks were generated successfully")
            return False

        if not all_succeeded:
            print(f"Warning: {len(track_classes) - len(generated_files)}/{len(track_classes)} tracks failed to generate")

        if mix_mode or blend_mode:
            mode_label = "blend" if blend_mode else "mix"
            print(f"\n{mode_label.capitalize()}ing {len(generated_files)} tracks...")
            mix_output = os.path.join(results_dir, f'_lego_{mode_label}_raw_{timestamp}.wav')
            input_list = " ".join(f'-i "{f}"' for f in generated_files)
            ret = os.system(f'ffmpeg -y {input_list} -filter_complex amix=inputs={len(generated_files)}:duration=longest "{mix_output}" 2>/dev/null')
            if ret != 0 or not os.path.exists(mix_output):
                print(f"Error: Failed to {mode_label} tracks")
                for f in generated_files:
                    if os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                return False

            if blend_mode:
                blend_output = os.path.join(results_dir, f'_lego_blend_src_{timestamp}.wav')
                ret = os.system(f'ffmpeg -y -i "{mix_output}" -i "{actual_source}" -filter_complex amix=inputs=2:duration=longest "{blend_output}" 2>/dev/null')
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                if ret != 0 or not os.path.exists(blend_output):
                    print("Error: Failed to blend mixed tracks with source")
                    for f in generated_files:
                        if os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    if os.path.exists(blend_output):
                        try:
                            os.unlink(blend_output)
                        except:
                            pass
                    return False
                output_filename = f'voder_ttm_lego_blend_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(blend_output, output_path)
            else:
                output_filename = f'voder_ttm_lego_mix_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(mix_output, output_path)

            for f in generated_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

            print(f"\nSuccess! Output saved to: {output_path}")
            return True
        else:
            exported = []
            for idx, f in enumerate(generated_files):
                track = track_classes[idx]
                output_filename = f'voder_ttm_lego_{track}_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(f, output_path)
                exported.append(output_path)

            print(f"\nSuccess! {len(exported)} track(s) exported:")
            for p in exported:
                print(f"  {p}")
            return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step
        ace_step = None
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_extract(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    extract_mix = params.get('extract_mix', False)
    extract_only = params.get('extract_only', False)
    _cleanup = []

    _task_source_args = params.get('task_source_args', [])
    if not _task_source_args:
        print("Error: Extract task requires a source path (audio/video file or URL)")
        return False

    source_path = _task_source_args[0]
    if not os.path.exists(source_path) and not is_youtube_url(source_path):
        print(f"Error: Source not found: {source_path}")
        return False

    instruments_raw = params.get('instruments_raw', '')
    if not instruments_raw:
        print('Error: Extract task requires stems (e.g., stems "drums bass" or stems "everything")')
        return False

    track_classes, unknown = resolve_acestep_tracks(instruments_raw)
    if unknown is not None and len(unknown) > 0:
        print(f"Error: Unknown stem name(s): {', '.join(unknown)}")
        print(f"Valid stems: {', '.join(sorted(VALID_ACESTEP_TRACKS))}")
        print(f'Shortcuts: everything, instruments (non-vocal), voices (vocal only)')
        return False
    if track_classes is None:
        print("Error: No valid tracks specified")
        return False

    if extract_only:
        if len(track_classes) >= 12:
            print("Error: 'only' cannot be used with 'everything' or all 12 stems (nothing would remain)")
            return False
        specified_set = set(track_classes)
        all_tracks = sorted(VALID_ACESTEP_TRACKS)
        track_classes = [t for t in all_tracks if t not in specified_set]
        removed_names = sorted(specified_set)
        print(f"Only mode: removing {', '.join(removed_names)}, extracting {len(track_classes)} remaining stems")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    original_name = os.path.splitext(os.path.basename(source_path))[0].replace(' ', '_')[:50]

    if is_youtube_url(source_path):
        print(f"Downloading audio from URL: {source_path}")
        resolved, cleanup = resolve_target_to_audio(source_path)
        if resolved is None:
            print("Error: Could not download audio from URL")
            return False
        _cleanup.extend(cleanup)
        source_audio = resolved
    else:
        ext = os.path.splitext(source_path)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            temp_audio = os.path.join(results_dir, f'_extract_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{source_path}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from video")
                return False
            source_audio = temp_audio
            _cleanup.append(temp_audio)
        else:
            valid, msg = validate_audio_file(source_path)
            if not valid:
                print(f"Error: {msg}")
                return False
            source_audio = source_path

    print(f"Tracks to extract ({len(track_classes)}): {', '.join(track_classes)}")
    print("Loading ACE-Step XL-Base model (extract task)...")

    ace_step = AceStepWrapper(use_overdose=False, complete_mode=True)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model for extract task")
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False

    try:
        generated_files = []
        all_succeeded = True
        for idx, track in enumerate(track_classes):
            print(f"\n[{idx+1}/{len(track_classes)}] Extracting {track}...")
            temp_output = os.path.join(results_dir, f'_extract_tmp_{track}_{timestamp}.wav')
            success = ace_step.extract(
                src_audio=source_audio,
                track_name=track,
                output_path=temp_output
            )
            if success:
                generated_files.append(temp_output)
                print(f"  {track} extracted successfully")
            else:
                print(f"  Failed to extract {track}")
                all_succeeded = False
                if os.path.exists(temp_output):
                    try:
                        os.unlink(temp_output)
                    except:
                        pass

        if not generated_files:
            print("Error: No tracks were extracted successfully")
            return False

        if not all_succeeded:
            print(f"Warning: {len(track_classes) - len(generated_files)}/{len(track_classes)} tracks failed to extract")

        if extract_mix or extract_only:
            mode_label = "only" if extract_only else "mix"
            print(f"\nMixing {len(generated_files)} extracted tracks ({mode_label})...")
            mix_output = os.path.join(results_dir, f'_extract_{mode_label}_raw_{timestamp}.wav')
            input_list = " ".join(f'-i "{f}"' for f in generated_files)
            ret = os.system(f'ffmpeg -y {input_list} -filter_complex amix=inputs={len(generated_files)}:duration=longest "{mix_output}" 2>/dev/null')
            if ret != 0 or not os.path.exists(mix_output):
                print(f"Error: Failed to mix extracted tracks")
                for f in generated_files:
                    if os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                if os.path.exists(mix_output):
                    try:
                        os.unlink(mix_output)
                    except:
                        pass
                return False

            if extract_only:
                output_filename = f'voder_ttm_extract_{original_name}_{timestamp}.wav'
            else:
                output_filename = f'voder_ttm_extract_mix_{original_name}_{timestamp}.wav'
            output_path = os.path.join(results_dir, output_filename)
            shutil.move(mix_output, output_path)

            for f in generated_files:
                if os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

            print(f"\nSuccess! Output saved to: {output_path}")
            return True
        else:
            exported = []
            for idx, f in enumerate(generated_files):
                track = track_classes[idx]
                output_filename = f'voder_ttm_extract_{track}_{original_name}_{timestamp}.wav'
                output_path = os.path.join(results_dir, output_filename)
                shutil.move(f, output_path)
                exported.append(output_path)

            print(f"\nSuccess! {len(exported)} track(s) extracted:")
            for p in exported:
                print(f"  {p}")
            return True
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        del ace_step
        ace_step = None
        gc.collect()
        for f in _cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_ttm_bgm(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    bgm_source = params.get('bgm_source', '')
    if not bgm_source:
        print("Error: bgm requires a source path (audio/video file or URL)")
        return False

    music_params_list = params.get('music', [])
    music_description = None
    if music_params_list:
        music_description = music_params_list[-1]
        if music_description:
            music_description = music_description.strip()

    sfx_specs_raw = params.get('sfx_specs', [])

    if not music_description and not sfx_specs_raw:
        print('Error: bgm requires a music description and/or "sfx:" specs')
        return False

    level = 35
    level_list = params.get('level', [])
    if level_list:
        try:
            lv = int(level_list[-1])
            if lv < 0 or lv > 100:
                print("Error: level must be between 0 and 100")
                return False
            level = lv
        except (ValueError, TypeError):
            print("Error: level must be a number between 0 and 100")
            return False

    use_overdose = params.get('overdose', False)
    want_video = params.get('want_video', False)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cleanup_files = []
    original_video_path = None
    downloaded_video = None
    video_title = None

    try:
        is_link = is_youtube_url(bgm_source)
        is_video_file = False
        if not is_link and os.path.exists(bgm_source):
            ext = os.path.splitext(bgm_source)[1].lower()
            is_video_file = ext in VIDEO_EXTENSIONS
            if is_video_file:
                original_video_path = bgm_source

        if is_link and want_video:
            print(f"Downloading video from URL: {bgm_source}")
            downloaded_video, video_title = download_youtube_video(bgm_source, results_dir)
            if downloaded_video is None:
                print(f"Error: {video_title}")
                return False
            original_video_path = downloaded_video
            temp_audio = os.path.join(results_dir, f'_bgm_vid_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{downloaded_video}" -vn -acodec pcm_s16le -ar 48000 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print("Error: Failed to extract audio from downloaded video")
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False
            source_audio = temp_audio
            cleanup_files.append(temp_audio)
        elif want_video and not original_video_path:
            print("Warning: 'video' specified but source is an audio file (not video). Outputting as WAV.")
            print(f"Resolving source: {bgm_source}")
            source_audio, source_cleanup = resolve_target_to_audio(bgm_source)
            if source_audio is None:
                print("Error: Could not resolve source to audio")
                return False
            cleanup_files.extend(source_cleanup)
        else:
            print(f"Resolving source: {bgm_source}")
            source_audio, source_cleanup = resolve_target_to_audio(bgm_source)
            if source_audio is None:
                print("Error: Could not resolve source to audio")
                return False
            cleanup_files.extend(source_cleanup)

        source_duration = _get_audio_duration(source_audio)

        sfx_specs = None
        if sfx_specs_raw:
            sfx_specs, sfx_err = _parse_sfx_specs(sfx_specs_raw, source_duration)
            if sfx_err:
                print(f"Error: {sfx_err}")
                return False

        print("Cleaning source audio through SVS voice pipe...")
        clean_voice = svs_extract_vocals(source_audio)
        if clean_voice != source_audio:
            cleanup_files.append(clean_voice)

        voice_duration = _get_audio_duration(clean_voice)
        print(f"Clean voice duration: {voice_duration:.2f}s")

        reference_audio = None
        _bgm_ref_entries = params.get('ref_entries', [])
        if _bgm_ref_entries:
            reference_audio, ref_cl = _compose_refs(_bgm_ref_entries, results_dir)
            cleanup_files.extend(ref_cl)

        mixed_path = None

        if music_description:
            print("Loading ACE-Step model...")
            ace = AceStepWrapper(use_overdose=use_overdose)
            if ace.handler is None:
                print("Error: Failed to load ACE-Step model")
                return False

            print(f"Generating background music (description: \"{music_description}\")...")
            music_result = generate_background_music(ace, music_description, voice_duration, reference_audio=reference_audio)
            del ace
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if music_result is None:
                print("Error: Background music generation failed")
                return False
            music_temp_path, music_temp_dir = music_result

            vol = level / 100.0
            print(f"Mixing clean voice with music (volume: {level}%)...")
            mixed_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            mixed_temp.close()
            cmd = [
                'ffmpeg', '-i', clean_voice, '-i', music_temp_path,
                '-filter_complex', f'[1:a]volume={vol:.2f}[music];[0:a][music]amix=inputs=2:duration=longest',
                '-y', mixed_temp.name
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if music_temp_dir is not None:
                shutil.rmtree(music_temp_dir, ignore_errors=True)
            if result.returncode != 0:
                print(f"Error: FFmpeg mixing failed: {result.stderr}")
                try:
                    os.unlink(mixed_temp.name)
                except:
                    pass
                return False
            mixed_path = mixed_temp.name
            cleanup_files.append(mixed_path)
        else:
            mixed_path = clean_voice

        if sfx_specs:
            print(f"Applying {len(sfx_specs)} SFX overlay(s)...")
            sfx_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            sfx_temp.close()
            sfx_ok = _generate_and_overlay_sfx(mixed_path, sfx_specs, sfx_temp.name)
            if sfx_ok and os.path.exists(sfx_temp.name):
                mixed_path = sfx_temp.name
                cleanup_files.append(sfx_temp.name)
            else:
                if os.path.exists(sfx_temp.name):
                    try:
                        os.unlink(sfx_temp.name)
                    except:
                        pass
                print("Warning: SFX overlay failed, using audio without SFX")

        if original_video_path and os.path.exists(original_video_path):
            if downloaded_video and video_title:
                name = video_title.replace(' ', '_').replace('/', '_')[:50]
            else:
                name = os.path.splitext(os.path.basename(original_video_path))[0]
            out_ext = os.path.splitext(original_video_path)[1]
            if not out_ext:
                out_ext = '.mp4'
            output_path = os.path.join(results_dir, f"voder_ttm_bgm_{name}_{timestamp}{out_ext}")
            print("Muxing mixed audio into video...")
            final_temp = tempfile.NamedTemporaryFile(suffix=out_ext, delete=False)
            final_temp.close()
            mux_cmd = [
                'ffmpeg', '-i', original_video_path, '-i', mixed_path,
                '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                '-shortest', '-y', final_temp.name
            ]
            mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
            if mux_result.returncode != 0:
                print(f"Error: Video muxing failed: {mux_result.stderr}")
                try:
                    os.unlink(final_temp.name)
                except:
                    pass
                return False
            shutil.move(final_temp.name, output_path)
        else:
            if is_link:
                name = "audio"
            else:
                name = os.path.splitext(os.path.basename(bgm_source))[0]
                if not name:
                    name = "audio"
            output_path = os.path.join(results_dir, f"voder_ttm_bgm_{name}_{timestamp}.wav")
            shutil.copy2(mixed_path, output_path)

        print(f"✓ Success! Output saved to: {output_path}")
        return True
    finally:
        if downloaded_video and os.path.exists(downloaded_video):
            try:
                os.remove(downloaded_video)
            except:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        for f in cleanup_files:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass

def oneline_tts_dub(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    dub_source = params.get('dub_source')
    dub_subtitle = params.get('dub_subtitle', False)
    dub_subtitle_original = params.get('dub_subtitle_original', False)
    dub_subtitle_langs = params.get('dub_subtitle_langs')
    dub_translate_langs = params.get('dub_translate_langs')
    dub_video = params.get('dub_video', False)
    use_se = params.get('dub_se', False)

    if not dub_translate_langs:
        dub_translate_langs = {'source': 'auto', 'target': 'en'}

    if not dub_source:
        print("Error: TTS dub requires a source audio/video path")
        return False

    is_url = is_youtube_url(dub_source)
    has_video_path = is_url or dub_source.lower().endswith(tuple(VIDEO_EXTENSIONS))

    if dub_subtitle and not has_video_path and not dub_video:
        print("Warning: subtitle requested but no video available — subtitles will be skipped")

    video_path = None
    audio_path = None
    downloaded_video = None
    extracted_audio = None
    svs_vocal = None
    svs_music_track = None
    _dub_cleanup = []
    _dub_cleanup_dirs = []
    speaker_extraction = None

    try:
        if is_url:
            if dub_video or dub_subtitle:
                print("Downloading video from URL...")
                downloaded_video, video_title = download_youtube_video(dub_source)
                if not downloaded_video:
                    print("Error: Failed to download video")
                    return False
                video_path = downloaded_video
                _dub_cleanup.append(downloaded_video)
                print("Extracting audio from video...")
                extracted_audio = extract_audio_from_video_cli(video_path)
                if not extracted_audio:
                    print("Error: Could not extract audio from video")
                    return False
                audio_path = extracted_audio
                _dub_cleanup.append(extracted_audio)
            else:
                print("Downloading audio from URL...")
                ok, err, dl_audio = download_youtube_audio(dub_source)
                if not ok:
                    print(f"Error: {err}")
                    return False
                audio_path = dl_audio
                _dub_cleanup.append(dl_audio)
        elif dub_source.lower().endswith(tuple(VIDEO_EXTENSIONS)):
            video_path = dub_source
            print("Extracting audio from video...")
            extracted_audio = extract_audio_from_video_cli(video_path)
            if not extracted_audio:
                print("Error: Could not extract audio from video")
                return False
            audio_path = extracted_audio
            _dub_cleanup.append(extracted_audio)
        elif os.path.exists(dub_source):
            audio_path = dub_source
        else:
            print(f"Error: File not found: {dub_source}")
            return False

        bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
        if bs_roformer_lib not in sys.path:
            sys.path.insert(0, bs_roformer_lib)
        bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
        if bs_roformer_pkg not in sys.path:
            sys.path.insert(0, bs_roformer_pkg)

        print("Stage 1: SVS voice isolation (BS-RoFormer)...")
        from bs_roformer import BSRoformerSeparator
        svs_separator = BSRoformerSeparator(SVS_DIR)
        svs_separator.ensure_model(stem='voice')
        if svs_separator.vocals_model is None:
            print("Error: Failed to load BS-RoFormer vocals model")
            svs_separator.cleanup()
            del svs_separator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False
        ts = time.strftime("%Y%m%d_%H%M%S")
        svs_vocal_dir = tempfile.mkdtemp()
        _dub_cleanup_dirs.append(svs_vocal_dir)
        svs_vocal = os.path.join(svs_vocal_dir, f'_dub_vocal_{ts}.wav')
        svs_ok = svs_separator.separate(audio_path, 'voice', svs_vocal)
        svs_separator.cleanup()
        del svs_separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not (svs_ok and os.path.exists(svs_vocal)):
            print("Warning: SVS voice isolation failed, using original audio")
            svs_vocal = audio_path
        else:
            _dub_cleanup.append(svs_vocal)

        print("Extracting music track via SVS music...")
        svs_sep2 = BSRoformerSeparator(SVS_DIR)
        svs_sep2.ensure_model(stem='music')
        svs_music_dir = tempfile.mkdtemp()
        _dub_cleanup_dirs.append(svs_music_dir)
        svs_music_track = os.path.join(svs_music_dir, f'_dub_music_{ts}.wav')
        svs_m_ok = svs_sep2.separate(audio_path, 'music', svs_music_track)
        svs_sep2.cleanup()
        del svs_sep2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if svs_m_ok and os.path.exists(svs_music_track):
            _dub_cleanup.append(svs_music_track)
        else:
            svs_music_track = None

        asr_input = svs_vocal
        if use_se:
            print("Stage 1.5: Sound Enhancement (UniSE SE)...")
            from unise import UniSEEnhancer
            se_enhancer = UniSEEnhancer(UNISE_DIR)
            se_enhancer.ensure_model()
            if se_enhancer.model is None:
                print("Warning: Failed to load UniSE SE model, skipping enhancement")
                se_enhancer.cleanup()
                del se_enhancer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                se_temp_dir = tempfile.mkdtemp()
                _dub_cleanup_dirs.append(se_temp_dir)
                se_temp = os.path.join(se_temp_dir, f'_dub_se_{ts}.wav')
                se_ok = se_enhancer.enhance(svs_vocal, se_temp)
                se_enhancer.cleanup()
                del se_enhancer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if se_ok and os.path.exists(se_temp):
                    asr_input = se_temp
                    _dub_cleanup.append(se_temp)
                else:
                    print("Warning: Sound Enhancement failed, using previous audio")
                    if se_temp_dir:
                        shutil.rmtree(se_temp_dir, ignore_errors=True)

        print("Stage 2: Transcribing with VibeVoice ASR (with audio events)...")
        asr = VibeVoiceASR()
        asr.ensure_model()
        if asr.model is None:
            print("Error: VibeVoice ASR failed to load. Dub requires VibeVoice (24GB+ VRAM or 48GB+ RAM)")
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        asr_segments = asr.transcribe_with_events(asr_input)
        asr.cleanup()
        del asr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not asr_segments:
            print("Error: ASR transcription returned no segments")
            return False

        speech_segments = [s for s in asr_segments if not s.get('is_event', False) and s.get('text', '').strip()]
        event_segments = [s for s in asr_segments if s.get('is_event', False)]

        print(f"Transcribed {len(speech_segments)} speech segments, {len(event_segments)} audio events")

        src_lang = dub_translate_langs['source']
        if src_lang == 'auto':
            all_speech_text = ' '.join(seg.get('text', '').strip() for seg in speech_segments if seg.get('text', '').strip())
            src_lang = _detect_lang_from_text(all_speech_text)
            print(f"Auto-detected source language: {src_lang}")

        tgt_lang = dub_translate_langs['target']

        need_dub_translate = dub_translate_langs is not None and src_lang != tgt_lang

        sub_src_lang = tgt_lang if need_dub_translate else src_lang
        sub_tgt_lang = None
        need_sub_translate = False
        if dub_subtitle and dub_subtitle_langs:
            if dub_subtitle_langs['target'] == tgt_lang:
                dub_subtitle_langs = None
            else:
                sub_tgt_lang = dub_subtitle_langs['target']
                if dub_subtitle_langs['source'] != 'auto':
                    sub_src_lang = dub_subtitle_langs['source']
                need_sub_translate = sub_src_lang != sub_tgt_lang

        translator = None
        if need_dub_translate or need_sub_translate:
            print("Loading TranslateGemma for translation...")
            translator = TranslateGemma()
            if not translator.ensure_model():
                print("Warning: Failed to load TranslateGemma, translation will be skipped")
                translator.cleanup()
                del translator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                translator = None

        num_speakers = len(set(seg.get('speaker', 'SPEAKER_00') for seg in speech_segments))
        print(f"Detected {num_speakers} speaker(s), source language: {src_lang}, target language: {tgt_lang}")

        speaker_voices = {}
        if num_speakers >= 2:
            print("Stage 2.5: Per-speaker separation for voice cloning and overlap detection...")
            speaker_extraction = _extract_speakers_for_subtitles(audio_path)
            if speaker_extraction:
                speaker_files = speaker_extraction.get("speaker_files", {})
                speaker_transcriptions = speaker_extraction.get("speaker_transcriptions", {})
                diar_speakers = sorted(speaker_extraction.get("speaker_segments", {}).keys(),
                                       key=lambda spk: speaker_extraction["speaker_segments"][spk][0]["start"])
                asr_speakers_sorted = sorted(set(seg.get('speaker', 'SPEAKER_00') for seg in speech_segments),
                                             key=lambda spk: next((seg.get('start', 0) for seg in speech_segments if seg.get('speaker') == spk), 0))
                asr_to_diar = {}
                for idx, asr_spk in enumerate(asr_speakers_sorted):
                    asr_to_diar[asr_spk] = diar_speakers[idx] if idx < len(diar_speakers) else diar_speakers[-1]

                spk_texts = {}
                for seg in speech_segments:
                    spk = seg.get('speaker', 'SPEAKER_00')
                    if spk not in spk_texts:
                        spk_texts[spk] = []
                    spk_texts[spk].append(seg.get('text', '').strip())

                print("Stage 2.5: Loading Fish-S2Pro model (extreme)...")
                tts = FishTTS()
                if not tts.ensure_model():
                    print("Error: Failed to load Fish-S2Pro model")
                    _cleanup_speaker_extraction(speaker_extraction)
                    speaker_extraction = None
                    if translator:
                        translator.cleanup()
                        del translator
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return False

                for asr_spk in asr_speakers_sorted:
                    diar_spk = asr_to_diar.get(asr_spk)
                    spk_audio = speaker_files.get(diar_spk) if diar_spk else None
                    if spk_audio and os.path.exists(spk_audio):
                        spk_ref_text = speaker_transcriptions.get(diar_spk, "")
                        if not spk_ref_text:
                            spk_ref_text = " ".join(spk_texts.get(asr_spk, []))
                        if not spk_ref_text:
                            spk_ref_text = _transcribe_for_fish_ref(spk_audio)
                        voice_ok = _tts_extract_voice(tts, spk_audio, use_extreme=True, ref_text=spk_ref_text)
                        if voice_ok:
                            speaker_voices[asr_spk] = {
                                "tokens": tts.encoded_refs["tokens"].cpu().clone(),
                                "text": tts.encoded_refs["text"]
                            }
                            print(f"Encoded voice for {asr_spk}")

                tts.cleanup()
                del tts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not speaker_voices:
                    print("Warning: Per-speaker voice encoding failed, falling back to single voice")
                    _cleanup_speaker_extraction(speaker_extraction)
                    speaker_extraction = None

        print("Stage 3: Loading Fish-S2Pro model (extreme)...")
        tts = FishTTS()
        if not tts.ensure_model():
            print("Error: Failed to load Fish-S2Pro model")
            if translator:
                translator.cleanup()
                del translator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return False

        if not speaker_voices:
            print("Extracting voice from source audio...")
            ref_text = _transcribe_for_fish_ref(svs_vocal)
            voice_ok = _tts_extract_voice(tts, svs_vocal, use_extreme=True, ref_text=ref_text)
            if not voice_ok:
                print("Error: Voice extraction failed")
                tts.cleanup()
                del tts
                if translator:
                    translator.cleanup()
                    del translator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False

        segment_tts_parts = []

        for seg_idx, seg in enumerate(speech_segments):
            seg_text = seg.get('text', '').strip()
            if not seg_text:
                continue

            seg_start = seg.get('start', 0)
            seg_end = seg.get('end', 0)
            seg_duration = seg_end - seg_start
            seg_speaker = seg.get('speaker', 'SPEAKER_00')
            original_text = seg_text

            if translator and need_dub_translate:
                print(f"Translating segment {seg_idx+1}/{len(speech_segments)} with TranslateGemma ({src_lang}->{tgt_lang})...")
                translated = translator.translate(seg_text, src_lang, tgt_lang)
                if translated:
                    seg_text = translated
                else:
                    print(f"Warning: Translation failed for segment {seg_idx+1}, using original text")

            if speaker_voices and seg_speaker in speaker_voices:
                tts.encoded_refs = {
                    "tokens": speaker_voices[seg_speaker]["tokens"].to(tts.device),
                    "text": speaker_voices[seg_speaker]["text"]
                }

            print(f"Generating speech for segment {seg_idx+1}/{len(speech_segments)} ({len(seg_text)} chars)...")
            seg_output_dir = tempfile.mkdtemp()
            _dub_cleanup_dirs.append(seg_output_dir)
            seg_output = os.path.join(seg_output_dir, f'_dub_seg{seg_idx}_{ts}.wav')
            syn_ok = _tts_synthesize(tts, seg_text, seg_output, language=None, use_extreme=True)
            if not syn_ok or not os.path.exists(seg_output):
                print(f"Warning: TTS synthesis failed for segment {seg_idx+1}")
                continue
            _dub_cleanup.append(seg_output)

            if seg_duration > 0:
                tts_duration = _get_audio_duration(seg_output)
                if tts_duration > 0:
                    speed_ratio = tts_duration / seg_duration
                    if speed_ratio > 1.05 or speed_ratio < 0.5:
                        print(f"Adjusting speed for segment {seg_idx+1} (TTS: {tts_duration:.1f}s, target: {seg_duration:.1f}s)...")
                        adjusted_dir = tempfile.mkdtemp()
                        _dub_cleanup_dirs.append(adjusted_dir)
                        adjusted_output = os.path.join(adjusted_dir, f'_dub_seg{seg_idx}_adj_{ts}.wav')
                        adj_ok = _adjust_audio_speed(seg_output, seg_duration, adjusted_output)
                        if adj_ok and os.path.exists(adjusted_output):
                            _dub_cleanup.append(adjusted_output)
                            seg_output = adjusted_output
                        else:
                            print("Warning: Speed adjustment failed, using original TTS output")

            segment_tts_parts.append({
                'path': seg_output,
                'start': seg_start,
                'end': seg_end,
                'speaker': seg_speaker,
                'text': seg_text,
                'original_text': original_text,
                'duration': seg_duration
            })

        _tts_cleanup(tts, use_extreme=True)
        tts = None

        if not segment_tts_parts:
            print("Error: No TTS parts generated")
            if translator:
                translator.cleanup()
                del translator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return False

        print("Stage 4: Assembling dubbed audio on timeline...")
        orig_duration = _get_audio_duration(audio_path)

        assemble_dir = tempfile.mkdtemp()
        _dub_cleanup_dirs.append(assemble_dir)
        dubbed_audio = os.path.join(assemble_dir, f'_dub_assembled_{ts}.wav')
        assemble_ok = _assemble_dubbed_audio(segment_tts_parts, orig_duration, dubbed_audio, original_audio_path=audio_path)
        if not assemble_ok or not os.path.exists(dubbed_audio):
            print("Warning: Numpy assembly failed, falling back to sequential overlay...")
            import numpy as np
            timeline_dir = tempfile.mkdtemp()
            _dub_cleanup_dirs.append(timeline_dir)
            silent_base = os.path.join(timeline_dir, f'_dub_silent_{ts}.wav')
            sr = 44100
            silence_samples = int(orig_duration * sr)
            silence_data = np.zeros((1, silence_samples), dtype=np.float32)
            sf.write(silent_base, silence_data.T, sr)
            _dub_cleanup.append(silent_base)
            current_base = silent_base
            for part_idx, part in enumerate(sorted(segment_tts_parts, key=lambda x: x['start'])):
                overlay_dir = tempfile.mkdtemp()
                _dub_cleanup_dirs.append(overlay_dir)
                overlay_output = os.path.join(overlay_dir, f'_dub_overlay_{part_idx}_{ts}.wav')
                overlay_ok = _overlay_segment_on_base(current_base, part['path'], part['start'], overlay_output)
                if overlay_ok and os.path.exists(overlay_output):
                    _dub_cleanup.append(overlay_output)
                    current_base = overlay_output
                else:
                    print(f"Warning: Failed to overlay segment {part_idx+1}")
            dubbed_audio = current_base

        final_audio = dubbed_audio
        if svs_music_track and os.path.exists(svs_music_track):
            print("Mixing dubbed vocals with original music track...")
            mix_dir = tempfile.mkdtemp()
            _dub_cleanup_dirs.append(mix_dir)
            mix_output = os.path.join(mix_dir, f'_dub_mixed_{ts}.wav')
            mix_ok = _mix_audio_at_target_sr(dubbed_audio, svs_music_track, mix_output, target_sr=44100)
            if mix_ok and os.path.exists(mix_output):
                final_audio = mix_output
                _dub_cleanup.append(mix_output)
            else:
                print("Warning: Music mixing failed, using vocals-only output")

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        lang_tag = f"_{tgt_lang}" if tgt_lang != 'en' else ""

        subtitle_segments = None
        if dub_subtitle and video_path and os.path.exists(video_path):
            if dub_subtitle_original:
                subtitle_segments = []
                for part in segment_tts_parts:
                    subtitle_segments.append({
                        'start': part['start'],
                        'end': part['end'],
                        'text': part.get('original_text', part['text']),
                        'speaker': part['speaker'],
                        'overlap': False
                    })
                if speaker_extraction:
                    spk_texts_orig = {}
                    for part in segment_tts_parts:
                        spk = part['speaker']
                        if spk not in spk_texts_orig:
                            spk_texts_orig[spk] = []
                        spk_texts_orig[spk].append(part.get('original_text', part['text']))

                    speaker_files = speaker_extraction.get("speaker_files", {})
                    diar_speakers = sorted(speaker_extraction.get("speaker_segments", {}).keys(),
                                           key=lambda spk: speaker_extraction["speaker_segments"][spk][0]["start"])
                    asr_speakers_sorted = sorted(set(seg.get('speaker', 'SPEAKER_00') for seg in speech_segments),
                                                 key=lambda spk: next((seg.get('start', 0) for seg in speech_segments if seg.get('speaker') == spk), 0))
                    asr_to_diar = {}
                    for idx, asr_spk in enumerate(asr_speakers_sorted):
                        asr_to_diar[asr_spk] = diar_speakers[idx] if idx < len(diar_speakers) else diar_speakers[-1]

                    orig_align_lang = _LANG_TO_ISO3.get(src_lang, "eng")

                    all_aligned_subs = []
                    for asr_spk in asr_speakers_sorted:
                        diar_spk = asr_to_diar.get(asr_spk)
                        spk_audio = speaker_files.get(diar_spk) if diar_spk else None
                        spk_text = " ".join(spk_texts_orig.get(asr_spk, []))
                        if not spk_text:
                            continue
                        align_audio = spk_audio if spk_audio and os.path.exists(spk_audio) else audio_path
                        word_ts = _forced_align_words(align_audio, spk_text, language=orig_align_lang)
                        if word_ts:
                            chunked = _group_words_to_segments(word_ts, chunk_size=8, speaker=asr_spk)
                            for chunk in chunked:
                                chunk["overlap"] = False
                            all_aligned_subs.extend(chunked)
                        else:
                            for seg in speech_segments:
                                if seg.get('speaker') == asr_spk:
                                    all_aligned_subs.append({
                                        'start': seg.get('start', 0),
                                        'end': seg.get('end', 0),
                                        'text': seg.get('text', '').strip(),
                                        'speaker': asr_spk,
                                        'overlap': False
                                    })

                    _cleanup_aligner_model()

                    if all_aligned_subs:
                        all_aligned_subs.sort(key=lambda x: x["start"])
                        for i, seg_a in enumerate(all_aligned_subs):
                            for j, seg_b in enumerate(all_aligned_subs):
                                if i == j:
                                    continue
                                if seg_a.get("speaker") == seg_b.get("speaker"):
                                    continue
                                if seg_a["start"] < seg_b["end"] and seg_a["end"] > seg_b["start"]:
                                    if seg_a["start"] <= seg_b["start"]:
                                        seg_b["overlap"] = True
                                    else:
                                        seg_a["overlap"] = True
                        subtitle_segments = all_aligned_subs
                else:
                    original_audio_for_align = audio_path if audio_path and os.path.exists(audio_path) else None
                    if original_audio_for_align:
                        print("Running forced alignment on original audio for accurate subtitle timing...")
                        aligned_subs = _align_subtitle_segments(original_audio_for_align, subtitle_segments, language=orig_align_lang)
                        _cleanup_aligner_model()
                        if aligned_subs:
                            subtitle_segments = aligned_subs
                            print(f"Forced alignment produced {len(subtitle_segments)} subtitle segments")
                        else:
                            print("Warning: Forced alignment failed, using original segment timings")

                if translator and need_sub_translate:
                    print(f"Translating subtitles with TranslateGemma ({sub_src_lang}->{sub_tgt_lang})...")
                    for sub_idx, sub_seg in enumerate(subtitle_segments):
                        sub_text = sub_seg.get('text', '').strip()
                        if sub_text:
                            translated = translator.translate(sub_text, sub_src_lang, sub_tgt_lang)
                            if translated:
                                sub_seg['text'] = translated
                            else:
                                print(f"Warning: Subtitle translation failed for segment {sub_idx+1}")
            else:
                subtitle_segments = []
                for part in segment_tts_parts:
                    subtitle_segments.append({
                        'start': part['start'],
                        'end': part['end'],
                        'text': part['text'],
                        'speaker': part['speaker'],
                        'overlap': False
                    })

                spk_segments_map = {}
                for part in segment_tts_parts:
                    spk = part['speaker']
                    if spk not in spk_segments_map:
                        spk_segments_map[spk] = []
                    spk_segments_map[spk].append(part)

                all_aligned_subs = []
                for spk, spk_parts in spk_segments_map.items():
                    spk_text = " ".join(p.get('text', '').strip() for p in sorted(spk_parts, key=lambda x: x['start']) if p.get('text', '').strip())
                    if not spk_text:
                        continue

                    has_tts = all(p.get('path') and os.path.exists(p['path']) for p in spk_parts if p.get('text', '').strip())
                    if has_tts:
                        spk_timeline_dir = tempfile.mkdtemp()
                        _dub_cleanup_dirs.append(spk_timeline_dir)
                        spk_timeline_path = os.path.join(spk_timeline_dir, f'_spk_timeline_{spk}_{ts}.wav')
                        build_ok = _build_speaker_timeline_audio(spk_parts, orig_duration, spk_timeline_path)
                        if build_ok and os.path.exists(spk_timeline_path):
                            _dub_cleanup.append(spk_timeline_path)
                            word_ts = _forced_align_words(spk_timeline_path, spk_text, language="auto")
                            if word_ts:
                                chunked = _group_words_to_segments(word_ts, chunk_size=8, speaker=spk)
                                for chunk in chunked:
                                    chunk["overlap"] = False
                                all_aligned_subs.extend(chunked)
                                continue

                    for p in sorted(spk_parts, key=lambda x: x['start']):
                        tts_path = p.get('path')
                        tts_text = p.get('text', '').strip()
                        seg_start = p.get('start', 0)
                        if not tts_path or not os.path.exists(tts_path) or not tts_text:
                            all_aligned_subs.append({
                                'text': tts_text,
                                'start': seg_start,
                                'end': p.get('end', seg_start),
                                'speaker': spk,
                                'overlap': False
                            })
                            continue
                        word_ts = _forced_align_words(tts_path, tts_text, language="auto")
                        if word_ts:
                            for w in word_ts:
                                all_aligned_subs.append({
                                    'text': w['text'],
                                    'start': w['start'] + seg_start,
                                    'end': w['end'] + seg_start,
                                    'speaker': spk,
                                    'overlap': False
                                })
                        else:
                            all_aligned_subs.append({
                                'text': tts_text,
                                'start': seg_start,
                                'end': p.get('end', seg_start),
                                'speaker': spk,
                                'overlap': False
                            })

                _cleanup_aligner_model()

                if all_aligned_subs:
                    all_aligned_subs.sort(key=lambda x: x["start"])
                    for i, seg_a in enumerate(all_aligned_subs):
                        for j, seg_b in enumerate(all_aligned_subs):
                            if i == j:
                                continue
                            if seg_a.get("speaker") == seg_b.get("speaker"):
                                continue
                            if seg_a["start"] < seg_b["end"] and seg_a["end"] > seg_b["start"]:
                                if seg_a["start"] <= seg_b["start"]:
                                    seg_b["overlap"] = True
                                else:
                                    seg_a["overlap"] = True
                    subtitle_segments = all_aligned_subs

                if dub_subtitle_langs:
                    sub_tgt_lang_new = dub_subtitle_langs['target']
                    sub_src_lang_new = dub_subtitle_langs.get('source', 'auto')
                    if sub_src_lang_new == 'auto':
                        sub_src_lang_new = tgt_lang
                    if sub_src_lang_new != sub_tgt_lang_new:
                        if not translator:
                            print("Loading TranslateGemma for subtitle translation...")
                            translator = TranslateGemma()
                            if not translator.ensure_model():
                                print("Warning: Failed to load TranslateGemma, subtitle translation skipped")
                                translator.cleanup()
                                del translator
                                translator = None
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        if translator:
                            print(f"Translating subtitles with TranslateGemma ({sub_src_lang_new}->{sub_tgt_lang_new})...")
                            for sub_idx, sub_seg in enumerate(subtitle_segments):
                                sub_text = sub_seg.get('text', '').strip()
                                if sub_text:
                                    translated = translator.translate(sub_text, sub_src_lang_new, sub_tgt_lang_new)
                                    if translated:
                                        sub_seg['text'] = translated

        if speaker_extraction:
            _cleanup_speaker_extraction(speaker_extraction)
            speaker_extraction = None

        if translator:
            translator.cleanup()
            del translator
            translator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if video_path and os.path.exists(video_path):
            if dub_subtitle and subtitle_segments:
                base_name = os.path.splitext(os.path.basename(dub_source))[0] if not is_url else "youtube_dub"
                output_filename = f"voder_tts_dub{lang_tag}_{timestamp_str}_{base_name}.mp4"
                output_path = os.path.join(results_dir, output_filename)
                if dub_subtitle_original:
                    burn_ok = _burn_subtitles_on_video(video_path, subtitle_segments, output_path)
                else:
                    burn_ok = _burn_subtitles_with_audio(video_path, final_audio, subtitle_segments, output_path)
                if burn_ok:
                    print(f"✓ Success! Dubbed+subtitled video saved to: {output_path}")
                else:
                    mux_cmd = [
                        'ffmpeg', '-i', video_path, '-i', final_audio,
                        '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                        '-shortest', '-y', output_path
                    ]
                    mux_result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=300)
                    if mux_result.returncode == 0:
                        print(f"✓ Success! Dubbed video (no subtitles) saved to: {output_path}")
                    else:
                        print("Error: Failed to create output video")
                        return False
            else:
                base_name = os.path.splitext(os.path.basename(dub_source))[0] if not is_url else "youtube_dub"
                output_filename = f"voder_tts_dub{lang_tag}_{timestamp_str}_{base_name}.mp4"
                output_path = os.path.join(results_dir, output_filename)
                mux_cmd = [
                    'ffmpeg', '-i', video_path, '-i', final_audio,
                    '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                    '-shortest', '-y', output_path
                ]
                mux_result = subprocess.run(mux_cmd, capture_output=True, text=True, timeout=300)
                if mux_result.returncode == 0:
                    print(f"✓ Success! Dubbed video saved to: {output_path}")
                else:
                    print("Error: Failed to mux dubbed audio into video")
                    return False
        else:
            base_name = os.path.splitext(os.path.basename(dub_source))[0]
            output_filename = f"voder_tts_dub{lang_tag}_{timestamp_str}_{base_name}.wav"
            output_path = os.path.join(results_dir, output_filename)
            shutil.copy2(final_audio, output_path)
            print(f"✓ Success! Dubbed audio saved to: {output_path}")

        return True

    except Exception as e:
        print(f"Error in dub pipeline: {e}")
        traceback.print_exc()
        if speaker_extraction:
            _cleanup_speaker_extraction(speaker_extraction)
        if translator:
            translator.cleanup()
            del translator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False

    finally:
        for f in _dub_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        for d in _dub_cleanup_dirs:
            if d and os.path.isdir(d):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except:
                    pass

def oneline_stt(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])
    keep_timestamp = params.get('timestamp', False)
    enable_dialogue = params.get('dialogue', False)
    enable_translate = params.get('translate', False)
    use_overdose = params.get('overdose', False)
    translate_langs = params.get('translate_langs')

    if not files:
        print("Error: STT mode requires at least one audio/video/image file path or supported platform URL")
        return False

    for file_path in files:
        if not os.path.exists(file_path) and not is_youtube_url(file_path):
            print(f"Error: File not found or unsupported URL: {file_path}")
            return False

    success_count = 0
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        print("=" * 60)

        do_translate = enable_translate
        has_lang_spec = translate_langs is not None
        is_image = file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp'))

        if is_image:
            try:
                print("Loading EasyOCR model...")
                ocr = EasyOCRReader()
                if ocr.reader is None:
                    print("Error: Failed to load EasyOCR model")
                    continue

                print(f"Extracting text from image...")
                success, text, error_msg = ocr.extract_text_from_image(file_path)

                ocr.cleanup()
                del ocr
                gc.collect()

                if not success:
                    print(f"Error: {error_msg or 'Failed to extract text from image'}")
                    continue

                if not text:
                    print(f"Error: No text found in image")
                    continue

                formatted_text = f"image: {text}"

                timestamp = time.strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(os.path.basename(file_path))[0]

                suffix = "stt_ocr"

                output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
                output_path = os.path.join(results_dir, output_filename)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_text)

                print(f"\u2713 Success! Output saved to: {output_path}")
                success_count += 1

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
            continue

        audio_path = file_path
        needs_youtube_download = is_youtube_url(file_path)
        needs_extraction = False

        if needs_youtube_download:
            print("Downloading audio from YouTube...")
            success, error_msg, audio_path = download_youtube_audio(file_path)
            if not success:
                print(f"Error: {error_msg}")
                continue
        elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            extracted = extract_audio_from_video_cli(file_path)
            if not extracted:
                print(f"Error: Could not extract audio from {file_path}")
                continue
            audio_path = extracted
            needs_extraction = True

        use_se = params.get('se', False)

        bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
        if bs_roformer_lib not in sys.path:
            sys.path.insert(0, bs_roformer_lib)
        bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
        if bs_roformer_pkg not in sys.path:
            sys.path.insert(0, bs_roformer_pkg)

        svs_temp = None
        if not is_image:
            print("Stage 1: SVS voice isolation (BS-RoFormer)...")
            from bs_roformer import BSRoformerSeparator
            svs_separator = BSRoformerSeparator(SVS_DIR)
            svs_separator.ensure_model(stem='voice')
            if svs_separator.vocals_model is None:
                print("Error: Failed to load BS-RoFormer vocals model")
                svs_separator.cleanup()
                del svs_separator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            svs_temp_dir = tempfile.mkdtemp()
            svs_temp = os.path.join(svs_temp_dir, f'_stt_svs_{timestamp}.wav')
            svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
            svs_separator.cleanup()
            del svs_separator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if svs_ok and os.path.exists(svs_temp):
                audio_path = svs_temp
            else:
                print("Warning: SVS voice isolation failed, using original audio")
                if svs_temp_dir:
                    shutil.rmtree(svs_temp_dir, ignore_errors=True)

        se_temp = None
        if use_se and not is_image:
            print("Stage 2: Sound Enhancement (UniSE SE)...")
            from unise import UniSEEnhancer
            se_enhancer = UniSEEnhancer(UNISE_DIR)
            se_enhancer.ensure_model()
            if se_enhancer.model is None:
                print("Error: Failed to load UniSE SE model")
                se_enhancer.cleanup()
                del se_enhancer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if svs_temp and os.path.exists(svs_temp):
                    shutil.rmtree(os.path.dirname(svs_temp), ignore_errors=True)
                continue
            se_temp_dir = tempfile.mkdtemp()
            se_temp = os.path.join(se_temp_dir, f'_stt_se_{timestamp}.wav')
            se_ok = se_enhancer.enhance(audio_path, se_temp)
            se_enhancer.cleanup()
            del se_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if se_ok and os.path.exists(se_temp):
                audio_path = se_temp
            else:
                print("Warning: Sound Enhancement failed, using previous audio")
                if se_temp_dir:
                    shutil.rmtree(se_temp_dir, ignore_errors=True)

        try:
            can_overdose_with_translate = has_lang_spec and do_translate

            if use_overdose and (not do_translate or can_overdose_with_translate) and not is_image:
                asr = VibeVoiceASR()
                asr.ensure_model()
                if asr.model is None:
                    print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                    asr.cleanup()
                    del asr
                    use_overdose = False

            if use_overdose and (not do_translate or can_overdose_with_translate) and not is_image:
                print("Transcribing with VibeVoice ASR...")
                asr_segments = asr.transcribe(audio_path)
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if not asr_segments:
                    print(f"Error: ASR transcription returned no segments for {file_path}")
                    continue

                if do_translate and has_lang_spec:
                    src_lang = translate_langs['source']
                    tgt_lang = translate_langs['target']
                    print(f"Translating with TranslateGemma ({src_lang}->{tgt_lang})...")
                    texts_to_translate = [seg.get('text', '') for seg in asr_segments]
                    translated_texts = _translate_segments_with_gemma(texts_to_translate, src_lang, tgt_lang)
                    if translated_texts:
                        for idx, seg in enumerate(asr_segments):
                            if idx < len(translated_texts):
                                seg['text'] = translated_texts[idx]

                def format_time_range(start, end):
                    def format_single(seconds):
                        if seconds is None:
                            seconds = 0
                        minutes = int(seconds // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds % 1) * 100)
                        return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                    return f"[{format_single(start)}-{format_single(end)}]"

                if enable_dialogue:
                    original_speakers = []
                    for seg in asr_segments:
                        speaker = seg["speaker"]
                        if speaker not in original_speakers:
                            original_speakers.append(speaker)
                    speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}
                    lines = []
                    current_speaker_num = None
                    current_text_parts = []
                    current_first_time = None
                    current_last_time = None
                    for seg in asr_segments:
                        speaker_num = speaker_mapping[seg["speaker"]]
                        text = seg.get("text", "")
                        seg_start = seg.get("start", 0) or 0
                        seg_end = seg.get("end", 0) or 0
                        if current_speaker_num is None:
                            current_speaker_num = speaker_num
                            current_text_parts = [text]
                            current_first_time = seg_start
                            current_last_time = seg_end
                        elif speaker_num == current_speaker_num:
                            current_text_parts.append(text)
                            current_last_time = seg_end
                        else:
                            if current_text_parts:
                                content = " ".join(current_text_parts)
                                if len(original_speakers) == 1:
                                    if keep_timestamp:
                                        lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content}")
                                    else:
                                        lines.append(f"text: {content}")
                                else:
                                    if keep_timestamp:
                                        lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                    else:
                                        lines.append(f"{current_speaker_num}: {content}")
                            current_speaker_num = speaker_num
                            current_text_parts = [text]
                            current_first_time = seg_start
                            current_last_time = seg_end
                    if current_text_parts:
                        content = " ".join(current_text_parts)
                        if len(original_speakers) == 1:
                            if keep_timestamp:
                                lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content}")
                            else:
                                lines.append(f"text: {content}")
                        else:
                            if keep_timestamp:
                                lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                            else:
                                lines.append(f"{current_speaker_num}: {content}")
                    formatted_text = "\n".join(lines)
                elif keep_timestamp:
                    lines = []
                    for seg in asr_segments:
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "").strip()
                        if text:
                            lines.append(f"{format_time_range(start, end)} text: {text}")
                    if lines:
                        formatted_text = "\n".join(lines)
                    else:
                        formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
                else:
                    formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
            else:
                print("Loading Whisper model...")
                stt = WhisperSTT()
                if stt.model is None:
                    print("Error: Failed to load Whisper model")
                    continue

                if do_translate and enable_dialogue:
                    print("Transcribing audio (for diarization)...")
                    original_result = stt.transcribe(audio_path)
                    if not original_result:
                        print(f"Error: Transcription failed for {file_path}")
                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    if has_lang_spec:
                        src_lang = translate_langs['source']
                        tgt_lang = translate_langs['target']
                        if src_lang == 'auto':
                            src_lang = original_result.get('language', 'en')[:2].lower()
                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        print(f"Translating with TranslateGemma ({src_lang}->{tgt_lang})...")
                        original_text = original_result.get('text', '').strip()
                        translated_text = _translate_with_gemma(original_text, src_lang, tgt_lang)
                        if translated_text:
                            translated_segments = original_result.get('segments', [])
                            seg_texts = [seg.get('text', '') for seg in translated_segments]
                            translated_seg_texts = _translate_segments_with_gemma(seg_texts, src_lang, tgt_lang)
                            if translated_seg_texts:
                                for idx, seg in enumerate(translated_segments):
                                    if idx < len(translated_seg_texts):
                                        seg['text'] = translated_seg_texts[idx]
                            result = {"text": translated_text, "segments": translated_segments, "language": tgt_lang}
                        else:
                            print("Warning: TranslateGemma translation failed, using original transcription")
                            result = original_result
                            do_translate = False
                    else:
                        print("Translating audio to English...")
                        result = stt.translate(audio_path)
                        if not result:
                            print("Error: Translation failed, using original transcription")
                            result = original_result
                            do_translate = False

                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                elif do_translate:
                    if has_lang_spec:
                        print("Transcribing audio...")
                        original_result = stt.transcribe(audio_path)
                        if not original_result:
                            print(f"Error: Transcription failed for {file_path}")
                            del stt
                            stt = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        src_lang = translate_langs['source']
                        tgt_lang = translate_langs['target']
                        if src_lang == 'auto':
                            src_lang = original_result.get('language', 'en')[:2].lower()

                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                        print(f"Translating with TranslateGemma ({src_lang}->{tgt_lang})...")
                        original_text = original_result.get('text', '').strip()
                        translated_text = _translate_with_gemma(original_text, src_lang, tgt_lang)
                        if translated_text:
                            translated_segments = original_result.get('segments', [])
                            seg_texts = [seg.get('text', '') for seg in translated_segments]
                            translated_seg_texts = _translate_segments_with_gemma(seg_texts, src_lang, tgt_lang)
                            if translated_seg_texts:
                                for idx, seg in enumerate(translated_segments):
                                    if idx < len(translated_seg_texts):
                                        seg['text'] = translated_seg_texts[idx]
                            result = {"text": translated_text, "segments": translated_segments, "language": tgt_lang}
                        else:
                            print("Warning: TranslateGemma translation failed, using original transcription")
                            result = original_result
                            do_translate = False
                    else:
                        print("Translating audio to English...")
                        result = stt.translate(audio_path)
                        if not result:
                            print(f"Error: Translation failed for {file_path}")
                            del stt
                            stt = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue

                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    print("Transcribing audio...")
                    result = stt.transcribe(audio_path)
                    if not result:
                        print(f"Error: Transcription failed for {file_path}")
                        del stt
                        stt = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    del stt
                    stt = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                def format_time_range(start, end):
                    def format_single(seconds):
                        if seconds is None:
                            seconds = 0
                        minutes = int(seconds // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds % 1) * 100)
                        return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                    return f"[{format_single(start)}-{format_single(end)}]"

                def format_time(seconds):
                    if seconds is None:
                        seconds = 0
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 100)
                    return f"[{minutes:02d}:{secs:02d}:{millis:02d}]"

            if not use_overdose:
                if enable_dialogue:
                    print("Performing speaker diarization...")
                    diarization = SpeakerDiarization()
                    if diarization.pipeline is None:
                        print("Warning: Speaker diarization model not available, proceeding without it")
                        if keep_timestamp and result.get("segments"):
                            lines = []
                            for seg in result.get("segments", []):
                                start = seg.get("start", 0)
                                end = seg.get("end", 0)
                                text = seg.get("text", "").strip()
                                if text:
                                    lines.append(f"{format_time_range(start, end)} text: {text}")
                            if lines:
                                formatted_text = "\n".join(lines)
                            else:
                                formatted_text = result.get("text", "").strip()
                        else:
                            formatted_text = result.get("text", "").strip()
                    else:
                        diar_result = diarization.diarize(audio_path)
                        if do_translate:
                            diarization_segments = diarization.format_diarization(diar_result, original_result)
                        else:
                            diarization_segments = diarization.format_diarization(diar_result, result)

                        formatted_segments = None
                        if diarization_segments:
                            if do_translate:
                                translated_segments = result.get("segments", [])
                                speaker_time_map = []
                                for ds in diarization_segments:
                                    speaker_time_map.append({
                                        "speaker": ds["speaker"],
                                        "start": ds.get("start", 0),
                                        "end": ds.get("end", 0),
                                        "text": ds["text"]
                                    })

                                merged_segments = []
                                for ts in translated_segments:
                                    ts_start = ts.get("start", 0)
                                    ts_end = ts.get("end", 0)
                                    ts_text = ts.get("text", "").strip()
                                    if not ts_text:
                                        continue
                                    best_speaker = None
                                    best_overlap = 0
                                    for sm in speaker_time_map:
                                        overlap_start = max(ts_start, sm["start"])
                                        overlap_end = min(ts_end, sm["end"])
                                        overlap = max(0, overlap_end - overlap_start)
                                        if overlap > best_overlap:
                                            best_overlap = overlap
                                            best_speaker = sm["speaker"]
                                    if best_speaker is not None:
                                        merged_segments.append({
                                            "speaker": best_speaker,
                                            "start": ts_start,
                                            "end": ts_end,
                                            "text": ts_text
                                        })
                                formatted_segments = merged_segments if merged_segments else None
                            else:
                                formatted_segments = diarization_segments

                        if formatted_segments:
                            original_speakers = []
                            for seg in formatted_segments:
                                speaker = seg["speaker"]
                                if speaker not in original_speakers:
                                    original_speakers.append(speaker)

                            speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

                            if len(original_speakers) == 1:
                                content = " ".join(seg["text"] for seg in formatted_segments)
                                if keep_timestamp:
                                    first_time = formatted_segments[0]["start"]
                                    last_time = formatted_segments[-1]["end"]
                                    formatted_text = f"{format_time_range(first_time, last_time)} text: {content}"
                                else:
                                    formatted_text = f"text: {content}"
                            else:
                                lines = []
                                current_speaker_num = None
                                current_text_parts = []
                                current_first_time = None
                                current_last_time = None

                                for seg in formatted_segments:
                                    speaker_num = speaker_mapping[seg["speaker"]]
                                    text = seg["text"]
                                    seg_start = seg.get("start", 0) or 0
                                    seg_end = seg.get("end", 0) or 0

                                    if current_speaker_num is None:
                                        current_speaker_num = speaker_num
                                        current_text_parts = [text]
                                        current_first_time = seg_start
                                        current_last_time = seg_end
                                    elif speaker_num == current_speaker_num:
                                        current_text_parts.append(text)
                                        current_last_time = seg_end
                                    else:
                                        if current_text_parts:
                                            content = " ".join(current_text_parts)
                                            if keep_timestamp:
                                                lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                            else:
                                                lines.append(f"{current_speaker_num}: {content}")
                                        current_speaker_num = speaker_num
                                        current_text_parts = [text]
                                        current_first_time = seg_start
                                        current_last_time = seg_end

                                if current_text_parts:
                                    content = " ".join(current_text_parts)
                                    if keep_timestamp:
                                        lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content}")
                                    else:
                                        lines.append(f"{current_speaker_num}: {content}")

                                formatted_text = "\n".join(lines)

                            del diarization
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        else:
                            if keep_timestamp and result.get("segments"):
                                lines = []
                                for seg in result.get("segments", []):
                                    start = seg.get("start", 0)
                                    end = seg.get("end", 0)
                                    text = seg.get("text", "").strip()
                                    if text:
                                        lines.append(f"{format_time_range(start, end)} text: {text}")
                                if lines:
                                    formatted_text = "\n".join(lines)
                                else:
                                    formatted_text = result.get("text", "").strip()
                            else:
                                formatted_text = result.get("text", "").strip()
                else:
                    formatted_text = result.get("text", "").strip()

                    if keep_timestamp and result.get("segments"):
                        lines = []
                        for seg in result.get("segments", []):
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            text = seg.get("text", "").strip()
                            if text:
                                lines.append(f"{format_time_range(start, end)} text: {text}")
                        if lines:
                            formatted_text = "\n".join(lines)
                        else:
                            formatted_text = result.get("text", "").strip()

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            if is_youtube_url(file_path):
                base_name = f"youtube_{len(files)}_{success_count + 1}"
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]

            suffix_parts = ["stt"]
            if do_translate:
                if has_lang_spec:
                    suffix_parts.append(f"translate_{translate_langs['target']}")
                else:
                    suffix_parts.append("translate")
            if keep_timestamp:
                suffix_parts.append("timestamp")
            if enable_dialogue:
                suffix_parts.append("dialogue")
            suffix = "_".join(suffix_parts)

            output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
            output_path = os.path.join(results_dir, output_filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)

            print(f"✓ Success! Output saved to: {output_path}")
            success_count += 1

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

        finally:
            if file_path != audio_path and os.path.exists(audio_path):
                try:
                    parent_dir = os.path.dirname(audio_path)
                    os.unlink(audio_path)
                    if os.path.exists(parent_dir) and os.path.basename(parent_dir).startswith('_'):
                        shutil.rmtree(parent_dir, ignore_errors=True)
                except:
                    pass

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {success_count}/{len(files)} files successful")
    return success_count > 0

def _get_video_resolution_ffmpeg(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
    except:
        pass
    return 1920, 1080

def _format_ass_time(seconds):
    if seconds is None:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def _build_ass_subtitles(segments, video_width, video_height):
    font_size = max(16, int(video_height * 0.035))
    margin_v = max(20, int(video_height * 0.03))
    outline_width = max(1, int(font_size * 0.04))
    shadow_offset = max(1, int(font_size * 0.08))
    line_gap = font_size + 8
    max_slots = 4

    speaker_outline_colors = [
        "&H00DD5500",
        "&H0000AAFF",
        "&H0000CC44",
        "&H00CC66FF",
        "&H00FFCC00",
        "&H00CC00BB",
    ]

    valid_segments = []
    for seg in segments:
        start = seg.get("start", 0) or 0
        end = seg.get("end", 0) or 0
        text = seg.get("text", "").strip()
        if not text or end <= start:
            continue
        valid_segments.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": seg.get("speaker"),
            "word_count": seg.get("word_count", len(text.split())),
        })

    valid_segments.sort(key=lambda x: x["start"])

    speaker_order = []
    seen_speakers = set()
    for seg in valid_segments:
        spk = seg.get("speaker")
        if spk is not None and spk not in seen_speakers:
            speaker_order.append(spk)
            seen_speakers.add(spk)

    speaker_color_map = {}
    for idx, spk in enumerate(speaker_order):
        speaker_color_map[spk] = speaker_outline_colors[idx % len(speaker_outline_colors)]

    speaker_by_spk = {}
    for seg in valid_segments:
        spk = seg.get("speaker")
        if spk not in speaker_by_spk:
            speaker_by_spk[spk] = []
        speaker_by_spk[spk].append(seg)
    for spk in speaker_by_spk:
        speaker_by_spk[spk].sort(key=lambda x: x["start"])

    for spk, segs in speaker_by_spk.items():
        for idx, seg in enumerate(segs):
            if idx < len(segs) - 1:
                gap = segs[idx + 1]["start"] - seg["end"]
                gap_extension = min(3.0, max(0, gap))
            else:
                gap_extension = 3.0
            speech_duration = seg["end"] - seg["start"]
            min_reading = seg["word_count"] * 0.4
            display_end = max(seg["end"] + gap_extension, seg["start"] + min_reading)
            if idx < len(segs) - 1:
                display_end = min(display_end, segs[idx + 1]["start"])
            seg["display_end"] = display_end

    active_slots = {}
    slot_segments = {}
    for seg in valid_segments:
        expired = [s for s, t in active_slots.items() if t <= seg["start"]]
        for s in expired:
            del active_slots[s]
            del slot_segments[s]
        assigned = None
        for s in range(1, max_slots + 1):
            if s not in active_slots:
                assigned = s
                break
        if assigned is None:
            earliest_slot = min(active_slots, key=active_slots.get)
            evicted = slot_segments[earliest_slot]
            evicted["display_end"] = min(evicted["display_end"], seg["start"])
            del active_slots[earliest_slot]
            del slot_segments[earliest_slot]
            assigned = earliest_slot
        seg["slot"] = assigned
        active_slots[assigned] = seg["display_end"]
        slot_segments[assigned] = seg

    style_lines = []
    for s in range(1, max_slots + 1):
        slot_margin_v = margin_v + (s - 1) * line_gap
        style_lines.append(
            f"Style: Slot{s},Noto Sans,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},{shadow_offset},2,10,10,{slot_margin_v},1"
        )
    style_lines.append(
        f"Style: Default,Noto Sans,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{outline_width},{shadow_offset},2,10,10,{margin_v},1"
    )

    ass_header = f"""[Script Info]
Title: VODER Transcription
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
""" + "\n".join(style_lines) + f"""

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []
    for seg in valid_segments:
        slot = seg.get("slot", 1)
        style_name = f"Slot{slot}" if seg.get("speaker") is not None else "Default"
        spk = seg.get("speaker")
        outline_color = speaker_color_map.get(spk, "&H00000000")
        text_escaped = seg["text"].replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", "\\N")
        if spk is not None:
            text_escaped = f"{{\\3c{outline_color}}}{text_escaped}"
        events.append(f"Dialogue: 0,{_format_ass_time(seg['start'])},{_format_ass_time(seg['display_end'])},{style_name},,0,0,0,,{text_escaped}")

    return ass_header + "\n".join(events) + "\n"

def _burn_subtitles_on_video(video_path, segments, output_path):
    video_width, video_height = _get_video_resolution_ffmpeg(video_path)
    ass_content = _build_ass_subtitles(segments, video_width, video_height)
    temp_dir = tempfile.mkdtemp()
    ass_path = os.path.join(temp_dir, "subtitles.ass")
    try:
        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        cmd = ['ffmpeg', '-i', video_path, '-vf', f"ass={escaped_ass}",
               '-c:a', 'copy', '-y', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            cmd = ['ffmpeg', '-i', video_path, '-vf', f"subtitles={escaped_ass}",
                   '-c:a', 'copy', '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                return False
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Error burning subtitles: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def _burn_subtitles_with_audio(video_path, audio_path, segments, output_path):
    video_width, video_height = _get_video_resolution_ffmpeg(video_path)
    ass_content = _build_ass_subtitles(segments, video_width, video_height)
    temp_dir = tempfile.mkdtemp()
    ass_path = os.path.join(temp_dir, "subtitles.ass")
    try:
        with open(ass_path, 'w', encoding='utf-8') as f:
            f.write(ass_content)
        escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
        cmd = ['ffmpeg', '-i', video_path, '-i', audio_path,
               '-vf', f"ass={escaped_ass}",
               '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
               '-shortest', '-y', output_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            cmd = ['ffmpeg', '-i', video_path, '-i', audio_path,
                   '-vf', f"subtitles={escaped_ass}",
                   '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                   '-shortest', '-y', output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                return False
        return os.path.exists(output_path)
    except Exception as e:
        print(f"Error burning subtitles with audio: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def oneline_stt_subtitle(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])
    use_se = params.get('se', False)
    translate_langs = params.get('translate_langs')

    if not files:
        print("Error: STT subtitle requires a video file path or URL")
        return False

    for file_path in files:
        if not os.path.exists(file_path) and not is_youtube_url(file_path):
            print(f"Error: File not found or invalid URL: {file_path}")
            return False

    translator = None
    if translate_langs:
        print("Loading TranslateGemma for translation...")
        translator = TranslateGemma()
        if not translator.ensure_model():
            print("Warning: Failed to load TranslateGemma, translation will be skipped")
            translator.cleanup()
            del translator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            translator = None

    success_count = 0
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        print("=" * 60)

        is_url = is_youtube_url(file_path)
        ext = os.path.splitext(file_path)[1].lower() if not is_url else ""

        if not is_url and ext not in VIDEO_EXTENSIONS:
            print(f"Error: STT subtitle only accepts video files and URLs. Got: {ext if ext else 'non-video file'}")
            continue

        video_path = None
        audio_path = None
        downloaded_video = None
        extracted_audio = None
        svs_temp = None
        se_temp = None
        svs_temp_dir = None
        se_temp_dir = None
        stt_speaker_extraction = None

        try:
            if is_url:
                print("Downloading video from URL...")
                downloaded_video, video_title = download_youtube_video(file_path)
                if not downloaded_video:
                    print("Error: Failed to download video")
                    continue
                video_path = downloaded_video
                print("Extracting audio from video...")
                extracted_audio = extract_audio_from_video_cli(video_path)
                if not extracted_audio:
                    print("Error: Could not extract audio from downloaded video")
                    continue
                audio_path = extracted_audio
            elif file_path.lower().endswith(tuple(VIDEO_EXTENSIONS)):
                video_path = file_path
                print("Extracting audio from video...")
                extracted_audio = extract_audio_from_video_cli(video_path)
                if not extracted_audio:
                    print("Error: Could not extract audio from video")
                    continue
                audio_path = extracted_audio
            else:
                print("Error: STT subtitle only accepts video files and URLs")
                continue

            bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
            if bs_roformer_lib not in sys.path:
                sys.path.insert(0, bs_roformer_lib)
            bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
            if bs_roformer_pkg not in sys.path:
                sys.path.insert(0, bs_roformer_pkg)

            print("Stage 1: SVS voice isolation (BS-RoFormer)...")
            from bs_roformer import BSRoformerSeparator
            svs_separator = BSRoformerSeparator(SVS_DIR)
            svs_separator.ensure_model(stem='voice')
            if svs_separator.vocals_model is None:
                print("Error: Failed to load BS-RoFormer vocals model")
                svs_separator.cleanup()
                del svs_separator
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            svs_temp_dir = tempfile.mkdtemp()
            svs_temp = os.path.join(svs_temp_dir, f'_stt_svs_{timestamp}.wav')
            svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
            svs_separator.cleanup()
            del svs_separator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if svs_ok and os.path.exists(svs_temp):
                audio_path = svs_temp
            else:
                print("Warning: SVS voice isolation failed, using original audio")
                if svs_temp_dir:
                    shutil.rmtree(svs_temp_dir, ignore_errors=True)
                    svs_temp_dir = None

            if use_se:
                print("Stage 2: Sound Enhancement (UniSE SE)...")
                from unise import UniSEEnhancer
                se_enhancer = UniSEEnhancer(UNISE_DIR)
                se_enhancer.ensure_model()
                if se_enhancer.model is None:
                    print("Warning: Failed to load UniSE SE model, skipping enhancement")
                    se_enhancer.cleanup()
                    del se_enhancer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    se_temp_dir = tempfile.mkdtemp()
                    se_temp = os.path.join(se_temp_dir, f'_stt_se_{timestamp}.wav')
                    se_ok = se_enhancer.enhance(audio_path, se_temp)
                    se_enhancer.cleanup()
                    del se_enhancer
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if se_ok and os.path.exists(se_temp):
                        audio_path = se_temp
                    else:
                        print("Warning: Sound Enhancement failed, using previous audio")
                        if se_temp_dir:
                            shutil.rmtree(se_temp_dir, ignore_errors=True)
                            se_temp_dir = None

            print("Transcribing with VibeVoice ASR...")
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Error: VibeVoice ASR failed to load. Subtitle requires VibeVoice (24GB+ VRAM or 48GB+ RAM)")
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            asr_segments = asr.transcribe(audio_path)
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not asr_segments:
                print("Error: ASR transcription returned no segments")
                continue

            num_asr_speakers = len(set(seg.get('speaker', 'SPEAKER_00') for seg in asr_segments))
            if num_asr_speakers >= 2:
                print(f"Detected {num_asr_speakers} speakers from ASR")
                print("Stage 1.5: Per-speaker separation for overlap-aware subtitling...")
                stt_speaker_extraction = _extract_speakers_for_subtitles(extracted_audio)
                if stt_speaker_extraction:
                    speaker_files = stt_speaker_extraction.get("speaker_files", {})
                    diar_speakers = sorted(stt_speaker_extraction.get("speaker_segments", {}).keys(),
                                           key=lambda spk: stt_speaker_extraction["speaker_segments"][spk][0]["start"])
                    asr_speakers_sorted = sorted(set(seg.get('speaker', 'SPEAKER_00') for seg in asr_segments),
                                                 key=lambda spk: next((seg.get('start', 0) for seg in asr_segments if seg.get('speaker') == spk), 0))
                    asr_to_diar = {}
                    for idx, asr_spk in enumerate(asr_speakers_sorted):
                        asr_to_diar[asr_spk] = diar_speakers[idx] if idx < len(diar_speakers) else diar_speakers[-1]

                    spk_texts = {}
                    for seg in asr_segments:
                        spk = seg.get('speaker', 'SPEAKER_00')
                        if spk not in spk_texts:
                            spk_texts[spk] = []
                        spk_texts[spk].append(seg.get('text', '').strip())

                    all_aligned_subs = []
                    for asr_spk in asr_speakers_sorted:
                        diar_spk = asr_to_diar.get(asr_spk)
                        spk_audio = speaker_files.get(diar_spk) if diar_spk else None
                        spk_text = " ".join(spk_texts.get(asr_spk, []))
                        if not spk_text:
                            continue
                        align_audio = spk_audio if spk_audio and os.path.exists(spk_audio) else audio_path
                        word_ts = _forced_align_words(align_audio, spk_text, language="auto")
                        if word_ts:
                            chunked = _group_words_to_segments(word_ts, chunk_size=8, speaker=asr_spk)
                            for chunk in chunked:
                                chunk["overlap"] = False
                            all_aligned_subs.extend(chunked)
                        else:
                            for seg in asr_segments:
                                if seg.get('speaker') == asr_spk:
                                    all_aligned_subs.append({
                                        'start': seg.get('start', 0),
                                        'end': seg.get('end', 0),
                                        'text': seg.get('text', '').strip(),
                                        'speaker': asr_spk,
                                        'overlap': False
                                    })

                    _cleanup_aligner_model()

                    if all_aligned_subs:
                        all_aligned_subs.sort(key=lambda x: x["start"])
                        for i, seg_a in enumerate(all_aligned_subs):
                            for j, seg_b in enumerate(all_aligned_subs):
                                if i == j:
                                    continue
                                if seg_a.get("speaker") == seg_b.get("speaker"):
                                    continue
                                if seg_a["start"] < seg_b["end"] and seg_a["end"] > seg_b["start"]:
                                    if seg_a["start"] <= seg_b["start"]:
                                        seg_b["overlap"] = True
                                    else:
                                        seg_a["overlap"] = True
                        asr_segments = all_aligned_subs
                        print(f"Per-speaker alignment produced {len(asr_segments)} subtitle segments")

                    _cleanup_speaker_extraction(stt_speaker_extraction)
                    stt_speaker_extraction = None
                else:
                    print("Warning: Speaker extraction failed, falling back to single-speaker alignment")
                    print("Running forced alignment for accurate subtitle timing...")
                    aligned_segments = _align_subtitle_segments(extracted_audio, asr_segments, language="auto")
                    _cleanup_aligner_model()
                    if aligned_segments:
                        asr_segments = aligned_segments
                        print(f"Forced alignment produced {len(asr_segments)} subtitle segments")
                    else:
                        print("Warning: Forced alignment failed, using original ASR segment timings")
            else:
                print("Running forced alignment for accurate subtitle timing...")
                aligned_segments = _align_subtitle_segments(extracted_audio, asr_segments, language="auto")
                _cleanup_aligner_model()
                if aligned_segments:
                    asr_segments = aligned_segments
                    print(f"Forced alignment produced {len(asr_segments)} subtitle segments")
                else:
                    print("Warning: Forced alignment failed, using original ASR segment timings")

            if translate_langs and translator:
                src_lang = translate_langs['source']
                tgt_lang = translate_langs['target']
                if src_lang == 'auto':
                    all_text = ' '.join(seg.get('text', '').strip() for seg in asr_segments if seg.get('text', '').strip())
                    src_lang = _detect_lang_from_text(all_text)
                    print(f"Auto-detected source language: {src_lang}")
                print(f"Translating with TranslateGemma ({src_lang}->{tgt_lang})...")
                translation_ok = True
                for idx, seg in enumerate(asr_segments):
                    seg_text = seg.get('text', '').strip()
                    if seg_text:
                        translated = translator.translate(seg_text, src_lang, tgt_lang)
                        if translated:
                            seg['text'] = translated
                        else:
                            print(f"Warning: Translation failed for segment {idx+1}")
                            translation_ok = False
                if not translation_ok:
                    print("Warning: Some translations failed, subtitles may contain mixed languages")
            elif translate_langs and not translator:
                print("Warning: Translation was requested but TranslateGemma could not be loaded — generating subtitles with original text")

            print(f"Transcribed {len(asr_segments)} segments. Burning subtitles onto video...")

            timestamp_str = time.strftime("%Y%m%d_%H%M%S")
            if is_url:
                base_name = "youtube_subtitle"
            else:
                base_name = os.path.splitext(os.path.basename(file_path))[0]

            lang_suffix = f"_{translate_langs['target']}" if translate_langs else ""
            output_filename = f"voder_stt_subtitle{lang_suffix}_{timestamp_str}_{base_name}.mp4"
            output_path = os.path.join(results_dir, output_filename)

            burn_ok = _burn_subtitles_on_video(video_path, asr_segments, output_path)
            if burn_ok:
                print(f"✓ Success! Subtitled video saved to: {output_path}")
                success_count += 1
            else:
                print("Error: Failed to burn subtitles onto video")
                continue

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()

        finally:
            if stt_speaker_extraction:
                _cleanup_speaker_extraction(stt_speaker_extraction)
            for temp_path in [extracted_audio, downloaded_video, svs_temp, se_temp]:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            for temp_dir in [svs_temp_dir, se_temp_dir]:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except:
                        pass

    if translator:
        translator.cleanup()
        del translator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Processing complete: {success_count}/{len(files)} files successful")
    return success_count > 0

def oneline_se(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    files = params.get('files', [])
    se_sub = params.get('se_sub')
    se_blend = params.get('se_blend', False)
    se_video = params.get('se_video', False)

    if not files:
        print("Error: SE mode requires at least one audio/video file path")
        return False

    if se_blend and se_sub not in ('voice', 'sr_music', 'sr_voice'):
        print("Warning: blend only applies to se voice blend, se sr music blend, or se sr voice blend. Ignoring blend.")
        se_blend = False

    if se_sub not in (None, 'voice', 'sr', 'sr_music', 'sr_voice', 'sr_voice_music'):
        print(f"Error: Unknown SE sub-mode '{se_sub}'. Valid: (none), voice, sr, sr music, sr voice, sr voice music")
        return False

    resolved_files = []
    _se_cleanup = []
    for file_path in files:
        if is_youtube_url(file_path):
            if se_video:
                print(f"Downloading video from URL: {file_path}")
                dl_path, dl_err = download_youtube_video(file_path)
                if not dl_path:
                    print(f"Error: {dl_err}")
                    for f in _se_cleanup:
                        if f and os.path.exists(f):
                            try: os.unlink(f)
                            except: pass
                    return False
                _se_cleanup.append(dl_path)
                resolved_files.append(dl_path)
            else:
                print(f"Downloading audio from URL: {file_path}")
                ok, err, dl_path = download_youtube_audio(file_path)
                if not ok:
                    print(f"Error: {err}")
                    for f in _se_cleanup:
                        if f and os.path.exists(f):
                            try: os.unlink(f)
                            except: pass
                    return False
                _se_cleanup.append(dl_path)
                resolved_files.append(dl_path)
        elif os.path.exists(file_path):
            resolved_files.append(file_path)
        else:
            print(f"Error: File not found: {file_path}")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

    def _se_resolve_audio(file_path, _se_cleanup):
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        audio_path = file_path
        if is_video:
            print("Extracting audio from video...")
            extracted = extract_audio_from_video_cli(file_path)
            if extracted:
                _se_cleanup.append(extracted)
                audio_path = extracted
            else:
                print("Warning: Could not extract audio from video, using file as-is")
        return audio_path, is_video

    def _se_track_svs_temp(result_path, original_path, _se_cleanup):
        if result_path != original_path:
            _se_cleanup.append(result_path)

    def _se_output(final_audio_path, is_video, file_path, timestamp, tag, results_dir, _se_cleanup):
        if is_video:
            output_filename = f"voder_se_{tag}_{timestamp}.mp4"
            output_path = os.path.join(results_dir, output_filename)
            print("Merging enhanced audio back into video...")
            ok = _replace_audio_in_video(file_path, final_audio_path, output_path)
            if not ok:
                print("Warning: Failed to merge audio into video, saving audio only")
                output_filename = f"voder_se_{tag}_{timestamp}.wav"
                output_path = os.path.join(results_dir, output_filename)
                shutil.copy2(final_audio_path, output_path)
        else:
            output_filename = f"voder_se_{tag}_{timestamp}.wav"
            output_path = os.path.join(results_dir, output_filename)
            shutil.copy2(final_audio_path, output_path)
        return output_path

    if se_sub is None:
        print("Loading UniSE Sound Enhancement model...")
        from unise import UniSEEnhancer
        enhancer = UniSEEnhancer(UNISE_DIR)
        enhancer.ensure_model()
        if enhancer.model is None:
            print("Error: Failed to load UniSE model")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                ext = os.path.splitext(file_path)[1].lower()
                is_video = ext in VIDEO_EXTENSIONS
                if is_video:
                    output_filename = f"voder_se_{timestamp}.mp4"
                    output_path = os.path.join(results_dir, output_filename)
                    print("Enhancing audio in video...")
                    success = enhancer.enhance_video(file_path, output_path)
                else:
                    output_filename = f"voder_se_{timestamp}.wav"
                    output_path = os.path.join(results_dir, output_filename)
                    print("Enhancing audio...")
                    success = enhancer.enhance(file_path, output_path)
                if success:
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Enhancement failed for {file_path}")
            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        enhancer.cleanup()
        del enhancer
        enhancer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try: os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    elif se_sub == 'voice':
        print("SE Voice: Extracting vocals via SVS, then enhancing via UniSE...")
        from unise import UniSEEnhancer
        se_enh = UniSEEnhancer(UNISE_DIR)
        se_enh.ensure_model()
        if se_enh.model is None:
            print("Error: Failed to load UniSE model")
            se_enh.cleanup()
            del se_enh
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.mkdtemp()
                _se_cleanup.append(temp_dir)

                audio_path, is_video = _se_resolve_audio(file_path, _se_cleanup)

                svs_vocals = svs_extract_vocals(audio_path)
                _se_track_svs_temp(svs_vocals, audio_path, _se_cleanup)

                se_out = os.path.join(temp_dir, f"se_voice_{timestamp}.wav")
                se_ok = se_enh.enhance(svs_vocals, se_out)

                if not se_ok or not os.path.exists(se_out):
                    print("Warning: UniSE enhancement failed, using SVS vocals as-is")
                    se_out = svs_vocals

                if se_blend:
                    svs_music = svs_extract_music(audio_path)
                    _se_track_svs_temp(svs_music, audio_path, _se_cleanup)
                    blend_out = os.path.join(temp_dir, f"se_voice_blend_{timestamp}.wav")
                    print("Blending enhanced vocals with original music...")
                    mix_ok = _mix_audio_at_target_sr(se_out, svs_music, blend_out, target_sr=48000)
                    if not mix_ok:
                        print("Warning: Blend failed, saving enhanced vocals only")
                        output_path = _se_output(se_out, is_video, file_path, timestamp, "voice_blend", results_dir, _se_cleanup)
                    else:
                        output_path = _se_output(blend_out, is_video, file_path, timestamp, "voice_blend", results_dir, _se_cleanup)
                else:
                    output_path = _se_output(se_out, is_video, file_path, timestamp, "voice", results_dir, _se_cleanup)

                if os.path.exists(output_path):
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Failed to produce output for {file_path}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        se_enh.cleanup()
        del se_enh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    elif se_sub == 'sr':
        print("SE SR: Upsampling full audio via AudioSR (basic model)...")

        audiosr = AudioSREnhancer(AUDIOSR_DIR)
        if not audiosr.ensure_model(model_name="basic"):
            print("Error: Failed to load AudioSR basic model")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.mkdtemp()
                _se_cleanup.append(temp_dir)

                audio_path, is_video = _se_resolve_audio(file_path, _se_cleanup)

                sr_out = os.path.join(temp_dir, f"sr_{timestamp}.wav")
                print("Upsampling audio via AudioSR (basic model)...")
                sr_ok = audiosr.enhance(audio_path, sr_out)

                if not sr_ok or not os.path.exists(sr_out):
                    print(f"Error: AudioSR upsampling failed for {file_path}")
                    continue

                output_path = _se_output(sr_out, is_video, file_path, timestamp, "sr", results_dir, _se_cleanup)

                if os.path.exists(output_path):
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Failed to produce output for {file_path}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        audiosr.cleanup()
        del audiosr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    elif se_sub == 'sr_music':
        print("SE SR Music: Upsampling non-vocals via AudioSR (basic model)...")

        audiosr = AudioSREnhancer(AUDIOSR_DIR)
        if not audiosr.ensure_model(model_name="basic"):
            print("Error: Failed to load AudioSR basic model")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.mkdtemp()
                _se_cleanup.append(temp_dir)

                audio_path, is_video = _se_resolve_audio(file_path, _se_cleanup)

                svs_music = svs_extract_music(audio_path)
                _se_track_svs_temp(svs_music, audio_path, _se_cleanup)

                sr_out = os.path.join(temp_dir, f"sr_music_{timestamp}.wav")
                print("Upsampling non-vocals via AudioSR (basic model)...")
                sr_ok = audiosr.enhance(svs_music, sr_out)

                if not sr_ok or not os.path.exists(sr_out):
                    print("Warning: AudioSR upsampling failed on non-vocals, using original")
                    sr_out = svs_music

                if se_blend:
                    svs_vocals = svs_extract_vocals(audio_path)
                    _se_track_svs_temp(svs_vocals, audio_path, _se_cleanup)
                    se_voice_out = os.path.join(temp_dir, f"se_voice_{timestamp}.wav")
                    print("Enhancing vocals via UniSE for blend...")
                    from unise import UniSEEnhancer
                    se_enh = UniSEEnhancer(UNISE_DIR)
                    se_enh.ensure_model()
                    if se_enh.model is not None:
                        se_ok = se_enh.enhance(svs_vocals, se_voice_out)
                        se_enh.cleanup()
                        del se_enh
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if not se_ok or not os.path.exists(se_voice_out):
                            print("Warning: UniSE enhancement failed, using SVS vocals")
                            se_voice_out = svs_vocals
                    else:
                        se_enh.cleanup()
                        del se_enh
                        se_voice_out = svs_vocals

                    blend_out = os.path.join(temp_dir, f"se_sr_music_blend_{timestamp}.wav")
                    print("Blending upsampled music with enhanced vocals at 48kHz...")
                    mix_ok = _mix_audio_at_target_sr(se_voice_out, sr_out, blend_out, target_sr=48000)
                    if not mix_ok:
                        print("Warning: Blend failed, saving upsampled music only")
                        output_path = _se_output(sr_out, is_video, file_path, timestamp, "sr_music_blend", results_dir, _se_cleanup)
                    else:
                        output_path = _se_output(blend_out, is_video, file_path, timestamp, "sr_music_blend", results_dir, _se_cleanup)
                else:
                    output_path = _se_output(sr_out, is_video, file_path, timestamp, "sr_music", results_dir, _se_cleanup)

                if os.path.exists(output_path):
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Failed to produce output for {file_path}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        audiosr.cleanup()
        del audiosr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    elif se_sub == 'sr_voice':
        if se_blend:
            print("SE SR Voice Blend: Upsampling vocals via AudioSR (speech model), blending with music...")
        else:
            print("SE SR Voice: Upsampling vocals via AudioSR (speech model)...")

        audiosr = AudioSREnhancer(AUDIOSR_DIR)
        if not audiosr.ensure_model(model_name="speech"):
            print("Error: Failed to load AudioSR speech model")
            for f in _se_cleanup:
                if f and os.path.exists(f):
                    try: os.unlink(f)
                    except: pass
            return False

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.mkdtemp()
                _se_cleanup.append(temp_dir)

                audio_path, is_video = _se_resolve_audio(file_path, _se_cleanup)

                svs_vocals = svs_extract_vocals(audio_path)
                _se_track_svs_temp(svs_vocals, audio_path, _se_cleanup)

                sr_voice_out = os.path.join(temp_dir, f"sr_voice_{timestamp}.wav")
                print("Upsampling vocals via AudioSR (speech model)...")
                sr_ok = audiosr.enhance(svs_vocals, sr_voice_out)

                if not sr_ok or not os.path.exists(sr_voice_out):
                    print("Warning: AudioSR upsampling failed on vocals, using SVS vocals as-is")
                    sr_voice_out = svs_vocals

                if se_blend:
                    svs_music = svs_extract_music(audio_path)
                    _se_track_svs_temp(svs_music, audio_path, _se_cleanup)
                    blend_out = os.path.join(temp_dir, f"se_sr_voice_blend_{timestamp}.wav")
                    print("Blending upsampled vocals with music at 48kHz...")
                    mix_ok = _mix_audio_at_target_sr(sr_voice_out, svs_music, blend_out, target_sr=48000)
                    if not mix_ok:
                        print("Warning: Blend failed, saving upsampled vocals only")
                        output_path = _se_output(sr_voice_out, is_video, file_path, timestamp, "sr_voice_blend", results_dir, _se_cleanup)
                    else:
                        output_path = _se_output(blend_out, is_video, file_path, timestamp, "sr_voice_blend", results_dir, _se_cleanup)
                else:
                    output_path = _se_output(sr_voice_out, is_video, file_path, timestamp, "sr_voice", results_dir, _se_cleanup)

                if os.path.exists(output_path):
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Failed to produce output for {file_path}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        audiosr.cleanup()
        del audiosr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    elif se_sub == 'sr_voice_music':
        print("SE SR Voice Music: Upsampling vocals (speech model) + non-vocals (basic model), auto-blending...")

        audiosr = AudioSREnhancer(AUDIOSR_DIR)

        success_count = 0
        for file_path in resolved_files:
            print(f"\nProcessing: {file_path}")
            print("=" * 60)
            try:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                temp_dir = tempfile.mkdtemp()
                _se_cleanup.append(temp_dir)

                audio_path, is_video = _se_resolve_audio(file_path, _se_cleanup)

                svs_vocals = svs_extract_vocals(audio_path)
                _se_track_svs_temp(svs_vocals, audio_path, _se_cleanup)
                svs_music = svs_extract_music(audio_path)
                _se_track_svs_temp(svs_music, audio_path, _se_cleanup)

                sr_voice_out = os.path.join(temp_dir, f"sr_voice_{timestamp}.wav")
                print("Upsampling vocals via AudioSR (speech model)...")
                if not audiosr.ensure_model(model_name="speech"):
                    print("Error: Failed to load AudioSR speech model for vocals")
                    continue
                sr_voice_ok = audiosr.enhance(svs_vocals, sr_voice_out)

                sr_music_out = os.path.join(temp_dir, f"sr_music_{timestamp}.wav")
                print("Upsampling non-vocals via AudioSR (basic model)...")
                if not audiosr.ensure_model(model_name="basic"):
                    print("Error: Failed to load AudioSR basic model for non-vocals")
                    continue
                sr_music_ok = audiosr.enhance(svs_music, sr_music_out)

                if not sr_voice_ok or not os.path.exists(sr_voice_out):
                    print("Warning: AudioSR failed on vocals, using SVS vocals as-is")
                    sr_voice_out = svs_vocals
                if not sr_music_ok or not os.path.exists(sr_music_out):
                    print("Warning: AudioSR failed on non-vocals, using SVS music as-is")
                    sr_music_out = svs_music

                blend_out = os.path.join(temp_dir, f"se_sr_voice_music_{timestamp}.wav")
                print("Auto-blending SR'd vocals and SR'd non-vocals at 48kHz...")
                mix_ok = _mix_audio_at_target_sr(sr_voice_out, sr_music_out, blend_out, target_sr=48000)
                if not mix_ok:
                    print("Warning: Blend failed, saving SR'd vocals only")
                    output_path = _se_output(sr_voice_out, is_video, file_path, timestamp, "sr_voice_music", results_dir, _se_cleanup)
                else:
                    output_path = _se_output(blend_out, is_video, file_path, timestamp, "sr_voice_music", results_dir, _se_cleanup)

                if os.path.exists(output_path):
                    print(f"\nSuccess! Output saved to: {output_path}")
                    success_count += 1
                else:
                    print(f"Error: Failed to produce output for {file_path}")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {file_path}: {e}")

        audiosr.cleanup()
        del audiosr
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for f in _se_cleanup:
            if f and os.path.exists(f):
                try:
                    if os.path.isdir(f):
                        shutil.rmtree(f)
                    else:
                        os.unlink(f)
                except: pass

        print(f"\n{'=' * 60}")
        print(f"Processing complete: {success_count}/{len(resolved_files)} files successful")
        return success_count > 0

    print(f"Error: Unrecognized SE sub-mode '{se_sub}'")
    for f in _se_cleanup:
        if f and os.path.exists(f):
            try: os.unlink(f)
            except: pass
    return False

def _ss_resolve_input(file_path, results_dir, timestamp):
    audio_path = None
    cleanup_list = []
    original_name = None
    is_url = is_youtube_url(file_path)

    if is_url:
        print(f"Downloading audio from URL...")
        success_dl, error_msg, downloaded_path = download_youtube_audio(file_path)
        if not success_dl:
            return None, None, None, cleanup_list, f"Download failed: {error_msg}"
        audio_path = downloaded_path
        cleanup_list.append(audio_path)
        original_name = "download"
    elif os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in VIDEO_EXTENSIONS
        original_name = os.path.splitext(os.path.basename(file_path))[0][:50]
        if is_video:
            temp_audio = os.path.join(results_dir, f'_ss_input_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{file_path}" -vn -acodec pcm_s16le -ar 16000 -ac 1 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                return None, None, None, cleanup_list, "Failed to extract audio from video"
            audio_path = temp_audio
            cleanup_list.append(audio_path)
        else:
            try:
                torchaudio.load(file_path)
                audio_path = file_path
            except Exception:
                return None, None, None, cleanup_list, f"Could not read audio file: {file_path}"
    else:
        return None, None, None, cleanup_list, f"File not found: {file_path}"

    return audio_path, original_name, is_url, cleanup_list, None

def _ss_run_pipeline(audio_path, use_se, results_dir, original_name, timestamp, target_path=None, use_overdose=False, use_blend=False, speaker_num=None):
    all_outputs = []
    temp_dirs = []

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)
    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    print("Stage 1: SVS voice isolation (BS-RoFormer)...")
    from bs_roformer import BSRoformerSeparator
    svs_separator = BSRoformerSeparator(SVS_DIR)
    svs_separator.ensure_model(stem='voice')
    if svs_separator.vocals_model is None:
        print("Error: Failed to load BS-RoFormer vocals model")
        svs_separator.cleanup()
        del svs_separator
        return None

    svs_temp_dir = tempfile.mkdtemp()
    temp_dirs.append(svs_temp_dir)
    svs_temp = os.path.join(svs_temp_dir, f'_ss_svs_{timestamp}.wav')
    svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
    svs_separator.cleanup()
    del svs_separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not svs_ok or not os.path.exists(svs_temp):
        print("Error: SVS voice isolation failed")
        return None

    clean_source = svs_temp

    blend_music_path = None
    if use_blend:
        print("Stage 1b: SVS music extraction for blend...")
        blend_music_path = svs_extract_music(audio_path)
        if not blend_music_path or not os.path.exists(blend_music_path):
            print("Warning: SVS music extraction failed, blend will be skipped")
            blend_music_path = None

    if target_path and os.path.exists(target_path):
        print("Stage 2: Target-based extraction (UniSE TSE)...")
        from unise import UniSEEnhancer
        tse_enhancer = UniSEEnhancer(UNISE_DIR)
        tse_enhancer.ensure_model()
        if tse_enhancer.model is None:
            print("Error: Failed to load UniSE TSE model")
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        output_filename = f"voder_ss_{original_name}_{timestamp}_extracted.wav"
        tse_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(tse_temp_dir)
        tse_temp_path = os.path.join(tse_temp_dir, output_filename)

        print(f"  Extracting target voice from source using reference...")
        tse_ok = tse_enhancer.tse_extract(clean_source, target_path, tse_temp_path)
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_se and tse_ok and os.path.exists(tse_temp_path):
            print("Applying Sound Enhancement to extracted voice...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp = os.path.join(tse_temp_dir, f"se_{output_filename}")
                se_ok = se_enh.enhance(tse_temp_path, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, tse_temp_path)
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if tse_ok and os.path.exists(tse_temp_path):
            if blend_music_path:
                blend_filename = f"voder_ss_{original_name}_{timestamp}_extracted_blend.wav"
                blend_out = os.path.join(tse_temp_dir, blend_filename)
                print("Blending extracted voice with non-vocals...")
                mix_ok = _mix_audio_at_target_sr(tse_temp_path, blend_music_path, blend_out, target_sr=48000)
                if mix_ok:
                    final_path = os.path.join(results_dir, blend_filename)
                    shutil.copy2(blend_out, final_path)
                    all_outputs.append(final_path)
                    print(f"  Blended voice saved to: {final_path}")
                else:
                    print("Warning: Blend failed, saving extracted voice only")
                    final_path = os.path.join(results_dir, output_filename)
                    shutil.copy2(tse_temp_path, final_path)
                    all_outputs.append(final_path)
                    print(f"  Extracted voice saved to: {final_path}")
            else:
                final_path = os.path.join(results_dir, output_filename)
                shutil.copy2(tse_temp_path, final_path)
                all_outputs.append(final_path)
                print(f"  Extracted voice saved to: {final_path}")
        else:
            print(f"  Warning: TSE extraction failed for target voice")

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        return all_outputs if all_outputs else None

    if use_overdose:
        print("Stage 2: Transcription + Speaker Diarization (VibeVoice ASR)...")
        asr = VibeVoiceASR()
        asr.ensure_model()
        if asr.model is None:
            print("Error: Failed to load VibeVoice ASR model")
            asr.cleanup()
            del asr
            return None

        asr_segments = asr.transcribe(clean_source)

        if not asr_segments:
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Error: VibeVoice ASR transcription returned no segments")
            return None

        formatted = asr_segments
    else:
        print("Stage 2: Speaker Diarization (pyannote)...")
        diarization = SpeakerDiarization()
        if diarization.pipeline is None:
            print("Error: Speaker diarization model not available (HF_TOKEN required)")
            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        diar_full = diarization.diarize_full(clean_source)
        if diar_full is None:
            print("Error: Speaker diarization failed")
            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        if hasattr(diar_full, 'exclusive_speaker_diarization'):
            exclusive_diar = diar_full.exclusive_speaker_diarization
            inclusive_diar = diar_full.speaker_diarization
        else:
            exclusive_diar = diar_full
            inclusive_diar = diar_full

        formatted = []
        for turn in inclusive_diar.itertracks(yield_label=True):
            segment, track, speaker = turn
            formatted.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "speaker": speaker,
                "text": ""
            })

        exclusive_segments = {}
        for turn in exclusive_diar.itertracks(yield_label=True):
            segment, track, speaker = turn
            if speaker not in exclusive_segments:
                exclusive_segments[speaker] = []
            exclusive_segments[speaker].append({
                "start": float(segment.start),
                "end": float(segment.end),
                "duration": float(segment.end) - float(segment.start)
            })

        for spk in exclusive_segments:
            exclusive_segments[spk].sort(key=lambda x: x["duration"], reverse=True)

    if not formatted:
        print("Error: No speaker segments found")
        return None

    speaker_segments = {}
    for seg in formatted:
        spk = seg["speaker"]
        if spk not in speaker_segments:
            speaker_segments[spk] = []
        speaker_segments[spk].append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})

    for spk in speaker_segments:
        segs = speaker_segments[spk]
        segs.sort(key=lambda x: x["start"])
        merged = []
        for s in segs:
            if merged and s["start"] - merged[-1]["end"] < 0.3:
                merged[-1]["end"] = s["end"]
                merged[-1]["text"] += " " + s["text"]
            else:
                merged.append({"start": s["start"], "end": s["end"], "text": s["text"]})
        speaker_segments[spk] = merged

    first_speaker_order = sorted(speaker_segments.keys(), key=lambda spk: speaker_segments[spk][0]["start"])
    sorted_speakers = first_speaker_order
    num_speakers = len(sorted_speakers)
    print(f"Detected {num_speakers} speaker(s)")

    if speaker_num is not None and speaker_num > num_speakers:
        speaker_num = num_speakers

    overlap_regions = []
    if use_overdose and formatted:
        for i in range(len(formatted)):
            for j in range(i + 1, len(formatted)):
                if formatted[i]["speaker"] != formatted[j]["speaker"]:
                    ov_start = max(formatted[i]["start"], formatted[j]["start"])
                    ov_end = min(formatted[i]["end"], formatted[j]["end"])
                    if ov_start < ov_end:
                        overlap_regions.append({"start": ov_start, "end": ov_end})
    elif not use_overdose:
        try:
            if hasattr(diar_full, 'exclusive_speaker_diarization'):
                inclusive_for_overlap = diar_full.speaker_diarization
            else:
                inclusive_for_overlap = diar_full
            overlap_tl = inclusive_for_overlap.get_overlap()
            for seg in overlap_tl:
                overlap_regions.append({"start": float(seg.start), "end": float(seg.end)})
        except Exception:
            pass

    if num_speakers < 2:
        print("Only one speaker detected. No separation needed.")
        if use_overdose:
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            diarization.pipeline = None
            del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for spk in sorted_speakers:
            segs = speaker_segments[spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            dur = longest["end"] - longest["start"]
            print(f"  Speaker 1: {len(segs)} segments, longest: {dur:.1f}s")
        output_filename = f"voder_ss_{original_name}_{timestamp}_speaker1.wav"
        single_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(single_temp_dir)
        single_temp = os.path.join(single_temp_dir, output_filename)
        shutil.copy2(clean_source, single_temp)

        if use_se:
            print("Applying Sound Enhancement to extracted voice...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp = os.path.join(single_temp_dir, f"se_{output_filename}")
                se_ok = se_enh.enhance(single_temp, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, single_temp)
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if blend_music_path:
            blend_filename = f"voder_ss_{original_name}_{timestamp}_speaker1_blend.wav"
            blend_out = os.path.join(single_temp_dir, blend_filename)
            print("Blending extracted voice with non-vocals...")
            mix_ok = _mix_audio_at_target_sr(single_temp, blend_music_path, blend_out, target_sr=48000)
            if mix_ok:
                final_path = os.path.join(results_dir, blend_filename)
                shutil.copy2(blend_out, final_path)
                all_outputs.append(final_path)
                print(f"Blended output saved to: {final_path}")
            else:
                print("Warning: Blend failed, saving extracted voice only")
                final_path = os.path.join(results_dir, output_filename)
                shutil.copy2(single_temp, final_path)
                all_outputs.append(final_path)
                print(f"Output saved to: {final_path}")
        else:
            final_path = os.path.join(results_dir, output_filename)
            shutil.copy2(single_temp, final_path)
            all_outputs.append(final_path)
            print(f"Output saved to: {final_path}")

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        return all_outputs

    speaker_to_num = {}
    for idx, spk in enumerate(sorted_speakers, 1):
        speaker_to_num[spk] = idx

    for spk in sorted_speakers:
        segs = speaker_segments[spk]
        longest = max(segs, key=lambda x: x["end"] - x["start"])
        dur = longest["end"] - longest["start"]
        print(f"  Speaker {speaker_to_num[spk]}: {len(segs)} segments, longest: {dur:.1f}s")

    if speaker_num is not None:
        target_spk = sorted_speakers[speaker_num - 1]
        target_spk_num = speaker_to_num[target_spk]
        print(f"Extracting speaker {target_spk_num} only...")

        from unise import UniSEEnhancer
        tse_enhancer = UniSEEnhancer(UNISE_DIR)
        tse_enhancer.ensure_model()
        if tse_enhancer.model is None:
            print("Error: Failed to load UniSE TSE model")
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if use_overdose:
                asr.cleanup()
                del asr
            else:
                diarization.pipeline = None
                del diarization
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None

        tse_temp_dir = tempfile.mkdtemp()
        temp_dirs.append(tse_temp_dir)

        if use_overdose:
            segs = speaker_segments[target_spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            start_t = longest["start"]
            dur_t = longest["end"] - longest["start"]
            if dur_t > 5.0:
                mid = start_t + dur_t / 2.0
                start_t = mid - 2.5
                dur_t = 5.0
                if start_t < 0:
                    start_t = 0.0
            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_pass0.wav")
            cmd = ['ffmpeg', '-i', clean_source, '-ss', str(start_t),
                   '-t', str(dur_t), '-ar', '16000', '-ac', '1', '-y', enroll_clip]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0 or not os.path.exists(enroll_clip):
                print(f"  Error: Failed to cut enrollment clip for speaker {target_spk_num}")
                tse_enhancer.cleanup()
                del tse_enhancer
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for td in temp_dirs:
                    try:
                        shutil.rmtree(td)
                    except Exception:
                        pass
                return None

            current_enroll = enroll_clip
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{target_spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None

            pass_output = os.path.join(tse_temp_dir, f"spk{target_spk_num}_pass1.wav")
            print(f"  Speaker {target_spk_num} — Pass 1: extracting with enrollment...")
            tse_ok = tse_enhancer.tse_extract(clean_source, current_enroll, pass_output)
            if not tse_ok or not os.path.exists(pass_output):
                print(f"  Error: TSE extraction failed for speaker {target_spk_num}")
                tse_enhancer.cleanup()
                del tse_enhancer
                asr.cleanup()
                del asr
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for td in temp_dirs:
                    try:
                        shutil.rmtree(td)
                    except Exception:
                        pass
                return None

            last_good_pass = pass_output

            print(f"  Speaker {target_spk_num} — Forced alignment refinement...")
            spk_text = ""
            try:
                raw_text = asr.transcribe_plain_text(pass_output)
                if raw_text:
                    spk_text = re.sub(r'\[?(?:Lyric|Silence|Music|Noise|Applause|Laughter|Cough|Breath)\]?\s*', '', raw_text, flags=re.IGNORECASE).strip()
                    spk_text = re.sub(r'\(?(?:silence|music|noise|applause|laughter|cough|breath)\)?\s*', '', spk_text, flags=re.IGNORECASE).strip()
                    spk_text = re.sub(r'\s+', ' ', spk_text).strip()
            except Exception:
                pass

            if spk_text:
                detected_lang = _detect_lang_from_text(spk_text)
                lang_iso3 = _LANG_TO_ISO3.get(detected_lang, "eng")
                word_ts = _forced_align_words(pass_output, spk_text, language=lang_iso3)
                if word_ts:
                    best_segs = []
                    for w in word_ts:
                        ws, we = w["start"], w["end"]
                        in_overlap = False
                        for ov in overlap_regions:
                            if ws < ov["end"] and we > ov["start"]:
                                in_overlap = True
                                break
                        if not in_overlap:
                            best_segs.append({"start": ws, "end": we, "duration": we - ws})
                    best_segs.sort(key=lambda x: x["duration"], reverse=True)

                    enroll_parts = []
                    collected = 0.0
                    for seg in best_segs:
                        if collected >= 5.0:
                            break
                        remaining = 5.0 - collected
                        take_dur = min(seg["duration"], remaining)
                        enroll_parts.append({"start": seg["start"], "duration": take_dur})
                        collected += take_dur

                    if enroll_parts:
                        enroll_clip2 = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_aligned.wav")
                        if len(enroll_parts) == 1:
                            part = enroll_parts[0]
                            cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                                   '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', enroll_clip2]
                            ret = subprocess.run(cmd, capture_output=True, text=True)
                            if ret.returncode != 0 or not os.path.exists(enroll_clip2):
                                enroll_clip2 = None
                        else:
                            part_files = []
                            for pi, part in enumerate(enroll_parts):
                                part_file = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_aligned_part{pi}.wav")
                                cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                                       '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', part_file]
                                ret = subprocess.run(cmd, capture_output=True, text=True)
                                if ret.returncode != 0 or not os.path.exists(part_file):
                                    continue
                                part_files.append(part_file)
                            if part_files:
                                concat_list = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_aligned_concat.txt")
                                with open(concat_list, 'w') as fw:
                                    for pf in part_files:
                                        fw.write(f"file '{pf}'\n")
                                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list,
                                       '-ar', '16000', '-ac', '1', '-y', enroll_clip2]
                                ret = subprocess.run(cmd, capture_output=True, text=True)
                                if ret.returncode != 0 or not os.path.exists(enroll_clip2):
                                    enroll_clip2 = None
                            else:
                                enroll_clip2 = None

                        if enroll_clip2:
                            pass_output2 = os.path.join(tse_temp_dir, f"spk{target_spk_num}_pass2_aligned.wav")
                            print(f"  Speaker {target_spk_num} — Pass 2 (aligned): extracting with refined enrollment...")
                            tse_ok2 = tse_enhancer.tse_extract(clean_source, enroll_clip2, pass_output2)
                            if tse_ok2 and os.path.exists(pass_output2):
                                last_good_pass = pass_output2

            _cleanup_aligner_model()

            recheck = asr.transcribe(last_good_pass)
            if recheck is not None:
                recheck_speakers = set()
                for seg in recheck:
                    recheck_speakers.add(seg.get("speaker"))
                if len(recheck_speakers) > 1:
                    print(f"  Speaker {target_spk_num} — Still {len(recheck_speakers)} speakers, multi-pass refinement...")
                    current_enroll_refined = last_good_pass
                    for pass_idx in range(3, 6):
                        recheck_segs = sorted(recheck, key=lambda x: x["end"] - x["start"], reverse=True)
                        longest_seg = recheck_segs[0]
                        ls = longest_seg["start"]
                        ld = longest_seg["end"] - longest_seg["start"]
                        if ld > 5.0:
                            mid = ls + ld / 2.0
                            ls = mid - 2.5
                            ld = 5.0
                            if ls < 0:
                                ls = 0.0
                        next_enroll = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_pass{pass_idx}.wav")
                        cmd = ['ffmpeg', '-i', current_enroll_refined, '-ss', str(ls),
                               '-t', str(ld), '-ar', '16000', '-ac', '1', '-y', next_enroll]
                        ret = subprocess.run(cmd, capture_output=True, text=True)
                        if ret.returncode != 0 or not os.path.exists(next_enroll):
                            break
                        pass_output_n = os.path.join(tse_temp_dir, f"spk{target_spk_num}_pass{pass_idx}.wav")
                        print(f"  Speaker {target_spk_num} — Pass {pass_idx}: refining...")
                        tse_ok_n = tse_enhancer.tse_extract(clean_source, next_enroll, pass_output_n)
                        if not tse_ok_n or not os.path.exists(pass_output_n):
                            break
                        last_good_pass = pass_output_n
                        current_enroll_refined = pass_output_n
                        recheck_n = asr.transcribe(pass_output_n)
                        if recheck_n is not None:
                            rs = set()
                            for seg in recheck_n:
                                rs.add(seg.get("speaker"))
                            if len(rs) <= 1:
                                print(f"  Speaker {target_spk_num} — Clean after pass {pass_idx}")
                                break
                else:
                    print(f"  Speaker {target_spk_num} — Clean after alignment refinement")

            shutil.copy2(last_good_pass, speaker_temp)

            asr.cleanup()
            del asr
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            clean_segs = exclusive_segments.get(target_spk, [])
            enroll_parts = []
            collected = 0.0
            target_enroll = 5.0
            for seg in clean_segs:
                if collected >= target_enroll:
                    break
                remaining = target_enroll - collected
                take_dur = min(seg["duration"], remaining)
                enroll_parts.append({"start": seg["start"], "duration": take_dur})
                collected += take_dur
            if not enroll_parts:
                segs = speaker_segments[target_spk]
                longest = max(segs, key=lambda x: x["end"] - x["start"])
                start_t = longest["start"]
                dur_t = longest["end"] - longest["start"]
                if dur_t > 5.0:
                    mid = start_t + dur_t / 2.0
                    start_t = mid - 2.5
                    dur_t = 5.0
                    if start_t < 0:
                        start_t = 0.0
                enroll_parts.append({"start": start_t, "duration": dur_t})

            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_pass0.wav")
            if len(enroll_parts) == 1:
                part = enroll_parts[0]
                cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                       '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', enroll_clip]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Error: Failed to cut enrollment clip for speaker {target_spk_num}")
                    tse_enhancer.cleanup()
                    del tse_enhancer
                    diarization.pipeline = None
                    del diarization
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for td in temp_dirs:
                        try:
                            shutil.rmtree(td)
                        except Exception:
                            pass
                    return None
            else:
                part_files = []
                for pi, part in enumerate(enroll_parts):
                    part_file = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_part{pi}.wav")
                    cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                           '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', part_file]
                    ret = subprocess.run(cmd, capture_output=True, text=True)
                    if ret.returncode != 0 or not os.path.exists(part_file):
                        continue
                    part_files.append(part_file)
                if not part_files:
                    print(f"  Error: Failed to cut enrollment clips for speaker {target_spk_num}")
                    tse_enhancer.cleanup()
                    del tse_enhancer
                    diarization.pipeline = None
                    del diarization
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for td in temp_dirs:
                        try:
                            shutil.rmtree(td)
                        except Exception:
                            pass
                    return None
                concat_list = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_concat.txt")
                with open(concat_list, 'w') as fw:
                    for pf in part_files:
                        fw.write(f"file '{pf}'\n")
                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list,
                       '-ar', '16000', '-ac', '1', '-y', enroll_clip]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Error: Failed to concatenate enrollment for speaker {target_spk_num}")
                    tse_enhancer.cleanup()
                    del tse_enhancer
                    diarization.pipeline = None
                    del diarization
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    for td in temp_dirs:
                        try:
                            shutil.rmtree(td)
                        except Exception:
                            pass
                    return None

            current_enroll = enroll_clip
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{target_spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None
            max_passes = 3

            for pass_idx in range(1, max_passes + 1):
                pass_output = os.path.join(tse_temp_dir, f"spk{target_spk_num}_pass{pass_idx}.wav")
                print(f"  Speaker {target_spk_num} — Pass {pass_idx}: extracting...")
                tse_ok = tse_enhancer.tse_extract(clean_source, current_enroll, pass_output)
                if not tse_ok or not os.path.exists(pass_output):
                    print(f"  Warning: TSE extraction failed for speaker {target_spk_num} pass {pass_idx}")
                    if last_good_pass:
                        shutil.copy2(last_good_pass, speaker_temp)
                    break

                last_good_pass = pass_output

                recheck = diarization.diarize_full(pass_output)
                if recheck is None:
                    print(f"  Speaker {target_spk_num} — Re-check failed, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                if hasattr(recheck, 'exclusive_speaker_diarization'):
                    recheck_excl = recheck.exclusive_speaker_diarization
                else:
                    recheck_excl = recheck

                recheck_speakers = set()
                for turn in recheck_excl.itertracks(yield_label=True):
                    _, _, speaker = turn
                    recheck_speakers.add(speaker)

                if len(recheck_speakers) <= 1:
                    print(f"  Speaker {target_spk_num} — Clean! Single speaker confirmed after pass {pass_idx}")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                print(f"  Speaker {target_spk_num} — Still {len(recheck_speakers)} speakers detected, refining...")
                recheck_segs = []
                for turn in recheck_excl.itertracks(yield_label=True):
                    segment, _, speaker = turn
                    dur = float(segment.end) - float(segment.start)
                    recheck_segs.append({"start": float(segment.start), "duration": dur, "speaker": speaker})
                recheck_segs.sort(key=lambda x: x["duration"], reverse=True)

                best_seg = recheck_segs[0]
                ls = best_seg["start"]
                ld = best_seg["duration"]
                if ld > 5.0:
                    mid = ls + ld / 2.0
                    ls = mid - 2.5
                    ld = 5.0
                    if ls < 0:
                        ls = 0.0

                next_enroll = os.path.join(tse_temp_dir, f"enroll_{target_spk_num}_pass{pass_idx}.wav")
                cmd = ['ffmpeg', '-i', pass_output, '-ss', str(ls),
                       '-t', str(ld), '-ar', '16000', '-ac', '1', '-y', next_enroll]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(next_enroll):
                    print(f"  Speaker {target_spk_num} — Failed to cut refined enrollment, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                current_enroll = next_enroll

                if pass_idx == max_passes:
                    print(f"  Speaker {target_spk_num} — Max passes reached, using final result")
                    shutil.copy2(pass_output, speaker_temp)

            diarization.pipeline = None
            del diarization
            tse_enhancer.cleanup()
            del tse_enhancer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not os.path.exists(speaker_temp):
            print(f"  Error: No output for speaker {target_spk_num}")
            for td in temp_dirs:
                try:
                    shutil.rmtree(td)
                except Exception:
                    pass
            return None

        if use_se:
            print("Applying Sound Enhancement to extracted voice...")
            from unise import UniSEEnhancer
            se_enh = UniSEEnhancer(UNISE_DIR)
            se_enh.ensure_model()
            if se_enh.model is not None:
                se_tmp = os.path.join(tse_temp_dir, f"se_{output_filename}")
                se_ok = se_enh.enhance(speaker_temp, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, speaker_temp)
            se_enh.cleanup()
            del se_enh
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if blend_music_path:
            blend_filename = output_filename.replace('.wav', '_blend.wav')
            blend_out = os.path.join(tse_temp_dir, blend_filename)
            print(f"  Blending speaker {target_spk_num} with non-vocals...")
            mix_ok = _mix_audio_at_target_sr(speaker_temp, blend_music_path, blend_out, target_sr=48000)
            if mix_ok:
                final_path = os.path.join(results_dir, blend_filename)
                shutil.copy2(blend_out, final_path)
                all_outputs.append(final_path)
            else:
                print(f"  Warning: Blend failed for speaker {target_spk_num}, saving voice only")
                final_path = os.path.join(results_dir, output_filename)
                shutil.copy2(speaker_temp, final_path)
                all_outputs.append(final_path)
        else:
            final_path = os.path.join(results_dir, output_filename)
            shutil.copy2(speaker_temp, final_path)
            all_outputs.append(final_path)

        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass

        print(f"\n{'=' * 60}")
        print(f"Separated 1 speaker successfully:")
        for p in all_outputs:
            print(f"  {os.path.basename(p)}")

        return all_outputs

    print("Stage 3: Target Speaker Extraction (UniSE TSE)...")
    from unise import UniSEEnhancer
    tse_enhancer = UniSEEnhancer(UNISE_DIR)
    tse_enhancer.ensure_model()
    if tse_enhancer.model is None:
        print("Error: Failed to load UniSE TSE model")
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None

    tse_temp_dir = tempfile.mkdtemp()
    temp_dirs.append(tse_temp_dir)
    speaker_temp_files = {}

    if use_overdose:
        for spk in sorted_speakers:
            spk_num = speaker_to_num[spk]
            segs = speaker_segments[spk]
            longest = max(segs, key=lambda x: x["end"] - x["start"])
            start_t = longest["start"]
            dur_t = longest["end"] - longest["start"]
            if dur_t > 5.0:
                mid = start_t + dur_t / 2.0
                start_t = mid - 2.5
                dur_t = 5.0
                if start_t < 0:
                    start_t = 0.0
            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass0.wav")
            cmd = ['ffmpeg', '-i', clean_source, '-ss', str(start_t),
                   '-t', str(dur_t), '-ar', '16000', '-ac', '1', '-y', enroll_clip]
            ret = subprocess.run(cmd, capture_output=True, text=True)
            if ret.returncode != 0 or not os.path.exists(enroll_clip):
                print(f"  Warning: Failed to cut enrollment clip for speaker {spk_num}, skipping")
                continue

            current_enroll = enroll_clip
            current_source = clean_source
            max_passes = 3
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None

            for pass_idx in range(1, max_passes + 1):
                pass_output = os.path.join(tse_temp_dir, f"spk{spk_num}_pass{pass_idx}.wav")
                print(f"  Speaker {spk_num} — Pass {pass_idx}: extracting with enrollment from {os.path.basename(current_enroll)}...")
                tse_ok = tse_enhancer.tse_extract(current_source, current_enroll, pass_output)
                if not tse_ok or not os.path.exists(pass_output):
                    print(f"  Warning: TSE extraction failed for speaker {spk_num} pass {pass_idx}")
                    if last_good_pass:
                        shutil.copy2(last_good_pass, speaker_temp)
                    break

                last_good_pass = pass_output

                recheck = asr.transcribe(pass_output)
                if recheck is None:
                    print(f"  Speaker {spk_num} — VibeVoice re-check failed, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                recheck_speakers = set()
                for seg in recheck:
                    recheck_speakers.add(seg.get("speaker"))

                if len(recheck_speakers) <= 1:
                    print(f"  Speaker {spk_num} — Clean! VibeVoice confirms single speaker after pass {pass_idx}")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                print(f"  Speaker {spk_num} — Still {len(recheck_speakers)} speakers detected, refining...")
                longest_seg = max(recheck, key=lambda x: x["end"] - x["start"])
                ls = longest_seg["start"]
                ld = longest_seg["end"] - longest_seg["start"]
                if ld > 5.0:
                    mid = ls + ld / 2.0
                    ls = mid - 2.5
                    ld = 5.0
                    if ls < 0:
                        ls = 0.0

                next_enroll = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass{pass_idx}.wav")
                cmd = ['ffmpeg', '-i', pass_output, '-ss', str(ls),
                       '-t', str(ld), '-ar', '16000', '-ac', '1', '-y', next_enroll]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(next_enroll):
                    print(f"  Speaker {spk_num} — Failed to cut refined enrollment, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                current_enroll = next_enroll
                current_source = clean_source

                if pass_idx == max_passes:
                    print(f"  Speaker {spk_num} — Max passes reached, using final result")
                    shutil.copy2(pass_output, speaker_temp)

            if os.path.exists(speaker_temp):
                speaker_temp_files[spk_num] = (speaker_temp, output_filename)
            else:
                print(f"  Warning: No output for speaker {spk_num}")

        asr.cleanup()
        del asr
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    else:
        for spk in sorted_speakers:
            spk_num = speaker_to_num[spk]
            clean_segs = exclusive_segments.get(spk, [])
            enroll_parts = []
            collected = 0.0
            target_enroll = 5.0
            for seg in clean_segs:
                if collected >= target_enroll:
                    break
                remaining = target_enroll - collected
                take_dur = min(seg["duration"], remaining)
                enroll_parts.append({"start": seg["start"], "duration": take_dur})
                collected += take_dur

            if not enroll_parts:
                segs = speaker_segments[spk]
                longest = max(segs, key=lambda x: x["end"] - x["start"])
                start_t = longest["start"]
                dur_t = longest["end"] - longest["start"]
                if dur_t > 5.0:
                    mid = start_t + dur_t / 2.0
                    start_t = mid - 2.5
                    dur_t = 5.0
                    if start_t < 0:
                        start_t = 0.0
                enroll_parts.append({"start": start_t, "duration": dur_t})

            enroll_clip = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass0.wav")
            if len(enroll_parts) == 1:
                part = enroll_parts[0]
                cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                       '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', enroll_clip]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Warning: Failed to cut enrollment clip for speaker {spk_num}, skipping")
                    continue
            else:
                part_files = []
                for pi, part in enumerate(enroll_parts):
                    part_file = os.path.join(tse_temp_dir, f"enroll_{spk_num}_part{pi}.wav")
                    cmd = ['ffmpeg', '-i', clean_source, '-ss', str(part["start"]),
                           '-t', str(part["duration"]), '-ar', '16000', '-ac', '1', '-y', part_file]
                    ret = subprocess.run(cmd, capture_output=True, text=True)
                    if ret.returncode != 0 or not os.path.exists(part_file):
                        continue
                    part_files.append(part_file)
                if not part_files:
                    print(f"  Warning: Failed to cut enrollment clips for speaker {spk_num}, skipping")
                    continue
                concat_list = os.path.join(tse_temp_dir, f"enroll_{spk_num}_concat.txt")
                with open(concat_list, 'w') as fw:
                    for pf in part_files:
                        fw.write(f"file '{pf}'\n")
                cmd = ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_list,
                       '-ar', '16000', '-ac', '1', '-y', enroll_clip]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(enroll_clip):
                    print(f"  Warning: Failed to concatenate enrollment for speaker {spk_num}, skipping")
                    continue

            current_enroll = enroll_clip
            current_source = clean_source
            max_passes = 3
            output_filename = f"voder_ss_{original_name}_{timestamp}_speaker{spk_num}.wav"
            speaker_temp = os.path.join(tse_temp_dir, output_filename)
            last_good_pass = None

            for pass_idx in range(1, max_passes + 1):
                pass_output = os.path.join(tse_temp_dir, f"spk{spk_num}_pass{pass_idx}.wav")
                print(f"  Speaker {spk_num} — Pass {pass_idx}: extracting...")
                tse_ok = tse_enhancer.tse_extract(current_source, current_enroll, pass_output)
                if not tse_ok or not os.path.exists(pass_output):
                    print(f"  Warning: TSE extraction failed for speaker {spk_num} pass {pass_idx}")
                    if last_good_pass:
                        shutil.copy2(last_good_pass, speaker_temp)
                    break

                last_good_pass = pass_output

                recheck = diarization.diarize_full(pass_output)
                if recheck is None:
                    print(f"  Speaker {spk_num} — Re-check failed, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                if hasattr(recheck, 'exclusive_speaker_diarization'):
                    recheck_excl = recheck.exclusive_speaker_diarization
                else:
                    recheck_excl = recheck

                recheck_speakers = set()
                for turn in recheck_excl.itertracks(yield_label=True):
                    _, _, speaker = turn
                    recheck_speakers.add(speaker)

                if len(recheck_speakers) <= 1:
                    print(f"  Speaker {spk_num} — Clean! Single speaker confirmed after pass {pass_idx}")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                print(f"  Speaker {spk_num} — Still {len(recheck_speakers)} speakers detected, refining...")
                recheck_segs = []
                for turn in recheck_excl.itertracks(yield_label=True):
                    segment, _, speaker = turn
                    dur = float(segment.end) - float(segment.start)
                    recheck_segs.append({"start": float(segment.start), "duration": dur, "speaker": speaker})
                recheck_segs.sort(key=lambda x: x["duration"], reverse=True)

                best_seg = recheck_segs[0]
                ls = best_seg["start"]
                ld = best_seg["duration"]
                if ld > 5.0:
                    mid = ls + ld / 2.0
                    ls = mid - 2.5
                    ld = 5.0
                    if ls < 0:
                        ls = 0.0

                next_enroll = os.path.join(tse_temp_dir, f"enroll_{spk_num}_pass{pass_idx}.wav")
                cmd = ['ffmpeg', '-i', pass_output, '-ss', str(ls),
                       '-t', str(ld), '-ar', '16000', '-ac', '1', '-y', next_enroll]
                ret = subprocess.run(cmd, capture_output=True, text=True)
                if ret.returncode != 0 or not os.path.exists(next_enroll):
                    print(f"  Speaker {spk_num} — Failed to cut refined enrollment, using pass {pass_idx} result")
                    shutil.copy2(pass_output, speaker_temp)
                    break

                current_enroll = next_enroll
                current_source = clean_source

                if pass_idx == max_passes:
                    print(f"  Speaker {spk_num} — Max passes reached, using final result")
                    shutil.copy2(pass_output, speaker_temp)

            if os.path.exists(speaker_temp):
                speaker_temp_files[spk_num] = (speaker_temp, output_filename)
            else:
                print(f"  Warning: No output for speaker {spk_num}")

        diarization.pipeline = None
        del diarization
        tse_enhancer.cleanup()
        del tse_enhancer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not speaker_temp_files:
        print("Error: Failed to extract any speakers")
        for td in temp_dirs:
            try:
                shutil.rmtree(td)
            except Exception:
                pass
        return None

    if use_se and speaker_temp_files:
        print("Applying Sound Enhancement to extracted voices...")
        from unise import UniSEEnhancer
        se_enh = UniSEEnhancer(UNISE_DIR)
        se_enh.ensure_model()
        if se_enh.model is not None:
            se_tmp_dir = tempfile.mkdtemp()
            for spk_num, (temp_f, fname) in speaker_temp_files.items():
                se_tmp = os.path.join(se_tmp_dir, f"se_{fname}")
                se_ok = se_enh.enhance(temp_f, se_tmp)
                if se_ok and os.path.exists(se_tmp):
                    shutil.copy2(se_tmp, temp_f)
            try:
                shutil.rmtree(se_tmp_dir)
            except Exception:
                pass
        se_enh.cleanup()
        del se_enh
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for spk_num, (temp_f, fname) in speaker_temp_files.items():
        if blend_music_path:
            blend_fname = fname.replace('.wav', '_blend.wav')
            blend_out = os.path.join(tse_temp_dir, blend_fname)
            print(f"  Blending speaker {spk_num} with non-vocals...")
            mix_ok = _mix_audio_at_target_sr(temp_f, blend_music_path, blend_out, target_sr=48000)
            if mix_ok:
                final_path = os.path.join(results_dir, blend_fname)
                shutil.copy2(blend_out, final_path)
                all_outputs.append(final_path)
            else:
                print(f"  Warning: Blend failed for speaker {spk_num}, saving voice only")
                final_path = os.path.join(results_dir, fname)
                shutil.copy2(temp_f, final_path)
                all_outputs.append(final_path)
        else:
            final_path = os.path.join(results_dir, fname)
            shutil.copy2(temp_f, final_path)
            all_outputs.append(final_path)

    for td in temp_dirs:
        try:
            shutil.rmtree(td)
        except Exception:
            pass

    print(f"\n{'=' * 60}")
    print(f"Separated {len(all_outputs)} speaker(s) successfully:")
    for p in all_outputs:
        print(f"  {os.path.basename(p)}")

    return all_outputs


def oneline_ss(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    file_path = params.get('file_path', '')
    use_se = params.get('use_se', False)
    target_path = params.get('target_path')
    use_overdose = params.get('overdose', False)
    use_blend = params.get('use_blend', False)
    use_video = params.get('use_video', False)
    speaker_num = params.get('speaker_num')

    if not file_path:
        print("Error: SS mode requires an audio/video file path or URL")
        return False

    if target_path and not os.path.exists(target_path) and not is_youtube_url(target_path):
        print(f"Error: Target file not found or invalid: {target_path}")
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("VODER SS - Speakers Separator")
    print("=" * 60)

    target_audio = target_path
    if target_path:
        if is_youtube_url(target_path):
            print("Downloading target audio from URL...")
            success_dl, error_msg, target_audio = download_youtube_audio(target_path)
            if not success_dl:
                print(f"Error: Target download failed: {error_msg}")
                return False
        elif target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from target video...")
            extracted = extract_audio_from_video_cli(target_path)
            if extracted:
                target_audio = extracted
            else:
                print("Error: Could not extract audio from target video")
                return False

    source_video_path = None
    video_cleanup = []

    if use_video:
        if is_youtube_url(file_path):
            print("Downloading video from URL for video output...")
            dl_video, dl_title = download_youtube_video(file_path, results_dir)
            if dl_video and os.path.exists(dl_video):
                source_video_path = dl_video
                video_cleanup.append(dl_video)
            else:
                print("Warning: Video download failed, continuing without video output")
        elif os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                source_video_path = file_path
            else:
                print("Info: Input is audio, 'video' keyword ignored")

    audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
    if err:
        print(f"Error: {err}")
        if target_audio and target_audio != target_path and os.path.exists(target_audio):
            try:
                os.unlink(target_audio)
            except:
                pass
        for vf in video_cleanup:
            if vf and os.path.exists(vf):
                try:
                    os.unlink(vf)
                except:
                    pass
        return False

    try:
        pipeline_outputs = _ss_run_pipeline(audio_path, use_se, results_dir, original_name, timestamp, target_audio, use_overdose, use_blend, speaker_num)
        if pipeline_outputs is None:
            print("SS pipeline failed")
            return False

        if source_video_path and pipeline_outputs:
            print("Muxing separated audio with video...")
            for wav_path in pipeline_outputs:
                mp4_name = os.path.splitext(os.path.basename(wav_path))[0] + '.mp4'
                mp4_path = os.path.join(results_dir, mp4_name)
                mux_cmd = ['ffmpeg', '-i', source_video_path, '-i', wav_path,
                            '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                            '-shortest', '-y', mp4_path]
                mux_result = subprocess.run(mux_cmd, capture_output=True, text=True)
                if mux_result.returncode == 0 and os.path.exists(mp4_path):
                    print(f"  Video saved to: {mp4_path}")
                else:
                    print(f"  Warning: Video muxing failed for {os.path.basename(wav_path)}")

        return True
    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return False
    finally:
        for f in cleanup_list:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except Exception:
                    pass
        if target_audio and target_audio != target_path and os.path.exists(target_audio):
            try:
                os.unlink(target_audio)
            except:
                pass
        for vf in video_cleanup:
            if vf and os.path.exists(vf):
                try:
                    os.unlink(vf)
                except:
                    pass



def oneline_svs(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    stem = params.get('stem', 'voice')
    file_path = params.get('file_path', '')
    svs_video = params.get('svs_video', False)

    if not file_path:
        print('Error: SVS mode requires an audio file path or URL')
        return False

    is_url = is_youtube_url(file_path)
    if not is_url and not os.path.exists(file_path):
        print(f'Error: File not found: {file_path}')
        return False

    stems_to_run = ['voice', 'music'] if stem == 'both' else [stem]
    stem_labels = {'voice': 'vocals', 'music': 'instruments', 'both': 'vocals and instruments'}
    print(f'Song Voice Separate - extracting {stem_labels.get(stem, stem)}')
    print(f'  Input: {file_path}')
    print('Loading BS-RoFormer Resurrection model...')

    bs_roformer_lib = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)

    bs_roformer_pkg = os.path.dirname(os.path.abspath(__file__))
    if bs_roformer_pkg not in sys.path:
        sys.path.insert(0, bs_roformer_pkg)

    from bs_roformer import BSRoformerSeparator
    separator = BSRoformerSeparator(SVS_DIR)
    for s in stems_to_run:
        separator.ensure_model(stem=s)
    if 'voice' in stems_to_run and separator.vocals_model is None:
        print('Error: Failed to load vocals model')
        return False
    if 'music' in stems_to_run and separator.inst_model is None:
        print('Error: Failed to load instrumental model')
        return False

    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
        downloaded_video = None
        downloaded_audio = None
        actual_file_path = file_path

        if is_url:
            if svs_video:
                downloaded_video, video_title = download_youtube_video(file_path, results_dir)
                if downloaded_video is None:
                    print(f'Error: {video_title}')
                    return False
                actual_file_path = downloaded_video
                original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
                is_video = True
            else:
                ok, err, dl_audio = download_youtube_audio(file_path, results_dir)
                if not ok:
                    print(f'Error: {err}')
                    return False
                downloaded_audio = dl_audio
                actual_file_path = dl_audio
                info_title = os.path.splitext(os.path.basename(dl_audio))[0]
                original_name = info_title.replace(' ', '_').replace('/', '_')[:50]
                is_video = False
        else:
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            input_ext = os.path.splitext(file_path)[1].lower()
            is_video = input_ext in video_exts

        temp_audio = None
        if is_video:
            print('Video detected, extracting audio...')
            temp_audio = os.path.join(results_dir, f'_svs_temp_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print('Error: Failed to extract audio from video')
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False

        audio_source = temp_audio if is_video else actual_file_path
        output_paths = []
        all_ok = True
        for s in stems_to_run:
            suffix = 'vocals' if s == 'voice' else 'instruments'
            output_filename = f'voder_svs_{original_name}_{timestamp}_{suffix}.mp4' if is_video else f'voder_svs_{original_name}_{timestamp}_{suffix}.wav'
            output_path = os.path.join(results_dir, output_filename)

            if is_video:
                temp_wav = os.path.join(results_dir, f'_svs_temp_{timestamp}_{suffix}.wav')
                success = separator.separate(audio_source, s, temp_wav)
                if success:
                    print(f'Merging {suffix} back into video...')
                    ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -i "{temp_wav}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                    if ret != 0 or not os.path.exists(output_path):
                        print(f'Error: Failed to merge {suffix} with video')
                        success = False
                        all_ok = False
                    else:
                        os.remove(temp_wav)
                        output_paths.append(output_path)
                else:
                    all_ok = False
                    if os.path.exists(temp_wav):
                        os.remove(temp_wav)
            else:
                success = separator.separate(audio_source, s, output_path)
                if success:
                    output_paths.append(output_path)
                else:
                    all_ok = False

        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)
        if downloaded_video and os.path.exists(downloaded_video):
            os.remove(downloaded_video)
        if downloaded_audio and os.path.exists(downloaded_audio):
            os.remove(downloaded_audio)

        if output_paths:
            print(f'\nSuccess! {len(output_paths)} file(s) saved:')
            for p in output_paths:
                print(f'  {p}')
            return True
        else:
            print('Error: All separations failed')
            return False
    except Exception as e:
        traceback.print_exc()
        print(f'Error: {e}')
        return False
    finally:
        separator.cleanup()
        del separator
        separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def oneline_sfx(params):
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    prompt = params.get('prompt', '')
    duration = params.get('duration', 10)
    steps = params.get('steps', 30)
    guide = params.get('guide', 4.5)

    if prompt:
        sfx_valid, _ = _validate_text_language(prompt, SUPPORTED_TANGOFLUX_LANGS, "SFX")
        if not sfx_valid:
            return False

    if duration > 30:
        print("Warning: Duration >30s clamped to 30s (model maximum).")
        duration = 30

    print(f"SFX Generation")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: {duration}s")
    print(f"  Steps: {steps}")
    print(f"  Guidance: {guide}")

    print("Loading TangoFlux SFX model...")
    from tangoflux import TangoFluxGenerator
    generator = TangoFluxGenerator(TANGOFLUX_DIR)
    generator.ensure_model()
    if generator.model is None:
        print("Error: Failed to load TangoFlux model")
        return False

    try:
        print(f"\nGenerating sound effect...")
        audio = generator.generate(prompt, duration, steps=steps, guidance_scale=guide)
        if audio is None:
            print("Error: Generation failed")
            return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"voder_sfx_{timestamp}.wav"
        output_path = os.path.join(results_dir, output_filename)

        if generator.save(audio, output_path):
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        else:
            print("Error: Failed to save output")
            return False

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}")
        return False
    finally:
        generator.cleanup()
        del generator
        generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()




def parse_and_execute_oneline(args):
    parsed = parse_oneline_args(args)
    if parsed.get('error'):
        print(f"Error: {parsed['error']}")
        show_oneline_usage()
        return False
    mode = validate_oneline_mode(parsed['mode'])
    if mode is None:
        print(f"Error: Invalid mode '{parsed['mode']}'")
        show_oneline_usage()
        return False
    parsed['mode'] = mode
    return execute_oneline_command(parsed)



class ChainPipeline:
    CHAIN_SEPARATOR = '/'

    def __init__(self):
        self.index = {}

    def split_segments(self, args):
        segments = []
        current = []
        for arg in args:
            if arg == self.CHAIN_SEPARATOR:
                segments.append(current)
                current = []
            else:
                current.append(arg)
        segments.append(current)
        return segments

    def parse_chain_segment(self, seg):
        if not seg:
            return None, None
        first = seg[0]
        if len(first) >= 2 and first.startswith('"') and first.endswith('"'):
            name = first[1:-1]
        else:
            name = first
        if not name:
            return None, "chain name cannot be empty"
        command_args = seg[1:]
        return name, command_args

    def validate(self, parsed_chains):
        seen = set()
        valid = []
        for name, command_args in parsed_chains:
            if not command_args:
                continue
            if name in seen:
                return None, f"Duplicate chain name: '{name}'"
            seen.add(name)
            valid.append((name, command_args))
        return valid, None

    def substitute_refs(self, command_args):
        out = []
        for a in command_args:
            if a in self.index:
                out.append(self.index[a])
            else:
                out.append(a)
        return out

    def _snapshot(self, directory):
        if not os.path.isdir(directory):
            return {}
        snap = {}
        for f in os.listdir(directory):
            p = os.path.join(directory, f)
            if os.path.isfile(p):
                snap[f] = os.path.getmtime(p)
        return snap

    def _new_files(self, directory, before):
        if not os.path.isdir(directory):
            return []
        new = []
        for f in os.listdir(directory):
            p = os.path.join(directory, f)
            if not os.path.isfile(p):
                continue
            if f not in before:
                new.append(p)
            elif os.path.getmtime(p) > before[f]:
                new.append(p)
        return new

    def execute(self, chains_args, result_path=None):
        segments = self.split_segments(chains_args)
        parsed_chains = []
        for seg in segments:
            name, command_args = self.parse_chain_segment(seg)
            if name is None and command_args is None:
                continue
            if name is None:
                return False, command_args
            parsed_chains.append((name, command_args))

        valid_chains, err = self.validate(parsed_chains)
        if err:
            print(f"Error: {err}")
            return False, err
        if not valid_chains:
            print("Error: no valid chains to execute (all chains were empty)")
            return False, "no valid chains"

        chains_temp_dir = os.path.join(os.getcwd(), "temp_chains")
        os.makedirs(chains_temp_dir, exist_ok=True)
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        voices_dir = os.path.join(os.getcwd(), "voices")

        total = len(valid_chains)
        print(f"Executing {total} chain(s)...")
        for idx, (name, command_args) in enumerate(valid_chains, start=1):
            is_last = (idx == total)
            substituted = self.substitute_refs(command_args)
            display_cmd = ' '.join(substituted)
            print(f"\n[Chain {idx}/{total}] name=\"{name}\"  >>>  {display_cmd}")

            results_before = self._snapshot(results_dir)
            voices_before = self._snapshot(voices_dir)

            success = parse_and_execute_oneline(substituted)
            if not success:
                print(f"Error: chain '{name}' failed")
                return False, f"chain '{name}' failed"

            new_results = self._new_files(results_dir, results_before)
            new_voices = self._new_files(voices_dir, voices_before)
            all_new = new_results + new_voices
            if not all_new:
                print(f"Warning: chain '{name}' produced no output file")
                continue
            all_new.sort(key=lambda p: os.path.getmtime(p))
            chain_output = all_new[0]
            ts = time.strftime("%Y%m%d_%H%M%S")
            if is_last:
                self.index[name] = chain_output
                print(f"[Chain '{name}'] final output retained: {chain_output}")
                if result_path:
                    try:
                        shutil.copy2(chain_output, result_path)
                        print(f"Result copied to: {result_path}")
                    except Exception as e:
                        print(f"Note: could not copy to result path: {e}")
            else:
                ext = os.path.splitext(chain_output)[1] or '.bin'
                safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:40] or 'chain'
                temp_path = os.path.join(chains_temp_dir, f"voder_chain_{safe_name}_{ts}{ext}")
                shutil.move(chain_output, temp_path)
                for extra in all_new[1:]:
                    try:
                        os.remove(extra)
                    except Exception:
                        pass
                self.index[name] = temp_path
                print(f"[Chain '{name}'] intermediate output stored: {temp_path}")
        return True, None


def oneline_chains(params):
    chains_args = params.get('chains_args', [])
    result_path = params.get('result_path')
    subcmd = params.get('chains_subcmd')
    if subcmd == 'build':
        return handle_build(chains_args)
    if subcmd == 'load':
        return handle_load(chains_args, result_path=result_path)
    if subcmd == 'journey':
        return handle_journey(chains_args)
    if subcmd == 'comment':
        return handle_comment(chains_args)
    if subcmd == 'decompile':
        return handle_decompile(chains_args)
    if subcmd == 'compile':
        return handle_compile(chains_args)
    if not chains_args:
        print("Error: chains mode requires at least one chain")
        return False
    pipeline = ChainPipeline()
    ok, _err = pipeline.execute(chains_args, result_path=result_path)
    return ok


CHAIN_FILE_MAGIC = "# VODER_CHAIN v1"
CHAIN_FILE_EXT = ".chain"
PREBUILT_CHAINS_DIR = os.path.join(_src_dir, "chains")
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_VALID_CONTENT_MODES = {'tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train', 'quest'}

def _err(step_index, step_name, category, message, fix=""):
    return {
        "step_index": step_index,
        "step_name": step_name,
        "category": category,
        "message": message,
        "fix": fix,
    }


def _resolve_linear_index(user_value, total, kind, context_label):
    if not isinstance(user_value, int) or user_value < 1:
        valid = list(range(1, total + 1)) if total > 0 else []
        likely = ", ".join(str(v) for v in valid[:10]) if valid else "(none — no %s exist)" % kind
        return None, (f"failed to resolve '{user_value}' {context_label} — {kind} index must be a positive integer. "
                      f"Likely meant: {likely}.")
    if user_value > total:
        valid = list(range(1, total + 1)) if total > 0 else []
        likely = ", ".join(str(v) for v in valid[:10]) if valid else "(none — no %s exist)" % kind
        return None, (f"failed to resolve '{user_value}' {context_label} — chain has {total} {kind}(s). "
                      f"Likely meant: {likely}.")
    return user_value - 1, None


def build_chain_text(name, timestamp, title, description, steps):
    lines = [f"{CHAIN_FILE_MAGIC} {timestamp} {name}"]
    if title:
        lines.append(f"title: {title}")
    else:
        lines.append("title:")
    if description:
        lines.append(f"description: {description}")
    else:
        lines.append("description:")
    for step in steps:
        lines.append("---")
        lines.append(f"chain: {step['name']}")
        if step.get("comment"):
            lines.append(f"comment: {step['comment']}")
        else:
            lines.append("comment:")
        lines.append(f"content: {step['content']}")
        input_comments = step.get("input_comments") or {}
        for input_idx in sorted(input_comments.keys()):
            lines.append(f"comment.input.{input_idx}: {input_comments[input_idx]}")
    lines.append("---")
    return "\n".join(lines) + "\n"


def parse_chain_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        return None, [_err(None, None, "format", f"Could not read file: {e}",
                            "Check the file path and permissions.")]
    return _parse_chain_text(raw)


def _parse_chain_text(raw):
    errors = []
    lines = raw.splitlines()
    if not lines:
        errors.append(_err(None, None, "format", "File is empty",
                           "Add the magic header line and at least one chain step."))
        return None, errors

    magic_line = lines[0].strip()
    magic_parts = magic_line.split()
    if len(magic_parts) != 5 or " ".join(magic_parts[:3]) != CHAIN_FILE_MAGIC:
        errors.append(_err(None, None, "format",
                           f"First line must be exactly '{CHAIN_FILE_MAGIC} <timestamp> <name>' (5 whitespace-separated tokens)",
                           "Fix line 1 — name cannot contain spaces."))
        return None, errors

    timestamp = magic_parts[3]
    name = magic_parts[4]
    if not _TIMESTAMP_RE.match(timestamp):
        errors.append(_err(None, None, "format",
                           f"Timestamp '{timestamp}' does not match YYYYMMDD_HHMMSS",
                           "Generate with time.strftime('%Y%m%d_%H%M%S')."))
    if not _NAME_RE.match(name):
        errors.append(_err(None, None, "naming",
                           f"Chain name '{name}' contains invalid characters",
                           "Use only letters, digits, underscores, hyphens. No spaces."))

    blocks = []
    current = {}
    in_header = True
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "---":
            if current:
                blocks.append(current)
                current = {}
            in_header = False
            continue
        if ":" not in stripped:
            errors.append(_err(None, None, "format",
                               f"Line does not match 'key: value' format: {stripped}",
                               "Use 'key: value' on each non-separator line."))
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if in_header:
            if key in ("title", "description"):
                current[key] = value
            else:
                errors.append(_err(None, None, "format",
                                   f"Unknown header key '{key}'",
                                   "Header only allows 'title' and 'description'."))
        else:
            if key in ("chain", "comment", "content"):
                current[key] = value
            elif key.startswith("comment.input."):
                idx_str = key[len("comment.input."):]
                if not idx_str.isdigit():
                    errors.append(_err(None, None, "format",
                                       f"Invalid input comment key '{key}:' — input index must be a positive integer",
                                       "Use 'comment.input.N:' where N is the 1-indexed input slot number."))
                else:
                    input_idx = int(idx_str)
                    if input_idx < 1:
                        errors.append(_err(None, None, "format",
                                           f"Invalid input comment key '{key}:' — input index must be >= 1",
                                           "Input slots are 1-indexed; use 'comment.input.1:' for the first input."))
                    else:
                        current.setdefault("_input_comments", {})[input_idx] = value
            else:
                errors.append(_err(None, None, "format",
                                   f"Unknown step key '{key}'",
                                   "Step blocks only allow 'chain', 'comment', 'content', and 'comment.input.N'."))
    if current:
        blocks.append(current)

    if not blocks:
        errors.append(_err(None, None, "format",
                           "No header block found",
                           "Add 'title:' and 'description:' lines after the magic line."))
        return None, errors

    header = blocks[0]
    title = header.get("title", "")
    description = header.get("description", "")
    step_blocks = blocks[1:]

    chains = []
    for idx, blk in enumerate(step_blocks, start=1):
        if "chain" not in blk:
            errors.append(_err(idx, None, "format",
                               f"Step {idx} missing 'chain:' key",
                               "Add 'chain: <name>' to this step."))
            continue
        cname = blk["chain"]
        if not _NAME_RE.match(cname):
            errors.append(_err(idx, cname, "naming",
                               f"Chain name '{cname}' has invalid characters",
                               "Use only letters, digits, underscores, hyphens."))
        if "content" not in blk:
            errors.append(_err(idx, cname, "format",
                               f"Step {idx} '{cname}' missing 'content:' key",
                               "Add 'content: <oneline command>' to this step."))
            continue
        content = blk["content"]
        if not content.strip():
            errors.append(_err(idx, cname, "format",
                               f"Step {idx} '{cname}' has empty content",
                               "Add the oneline command for this step."))
            continue
        comment = blk.get("comment", "")
        input_comments = blk.get("_input_comments", {})
        chains.append({
            "name": cname,
            "comment": comment,
            "content": content,
            "content_tokens": content.split(),
            "input_comments": input_comments,
        })

    if not chains and not any(e["category"] == "format" for e in errors):
        errors.append(_err(None, None, "format",
                           "No chain steps found",
                           "Add at least one step block after the header."))

    if not chains:
        return None, errors

    parsed = {
        "name": name,
        "timestamp": timestamp,
        "title": title,
        "description": description,
        "chains": chains,
    }
    return parsed, errors


def verify_chain_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        return False, [_err(None, None, "format", f"Could not read file: {e}",
                            "Check the file path and permissions.")], []
    return verify_chain_text(raw)


def verify_chain_text(raw):
    parsed, errors = _parse_chain_text(raw)
    warnings = []
    if parsed is None:
        return False, errors, warnings

    chain_names = [c["name"] for c in parsed["chains"]]
    seen = set()
    for idx, c in enumerate(parsed["chains"], start=1):
        if c["name"] in seen:
            errors.append(_err(idx, c["name"], "naming",
                               f"Duplicate chain name '{c['name']}'",
                               "Rename this step to be unique within the file."))
        seen.add(c["name"])

    for idx, c in enumerate(parsed["chains"], start=1):
        errors.extend(_verify_content_syntax(idx, c, chain_names))

    for idx, c in enumerate(parsed["chains"], start=1):
        errors.extend(_verify_references(idx, c, chain_names))

    for idx, c in enumerate(parsed["chains"], start=1):
        manual_count = sum(1 for t in c["content_tokens"] if t == "input")
        auto_count = sum(1 for t in c["content_tokens"] if t in chain_names and t != c["name"])
        if manual_count == 0 and auto_count == 0:
            warnings.append(f"Step {idx} '{c['name']}' has no 'input' placeholders and no chain references — it will run without any external input (OK for modes like sfx, but unusual for tts/sts/etc.).")
        if manual_count > 5:
            warnings.append(f"Step {idx} '{c['name']}' has {manual_count} manual inputs — consider splitting.")

    if not parsed["title"]:
        warnings.append("Title is empty — consider adding a short title for users.")
    if not parsed["description"]:
        warnings.append("Description is empty — consider adding a description.")
    for idx, c in enumerate(parsed["chains"], start=1):
        if not c["comment"]:
            warnings.append(f"Step {idx} '{c['name']}' has no comment — users won't know what input to provide.")

    return len(errors) == 0, errors, warnings


def _verify_content_syntax(step_idx, chain_step, all_chain_names=None):
    errors = []
    tokens = chain_step["content_tokens"]
    if not tokens:
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           "Content is empty",
                           "Add the oneline command."))
        return errors
    mode = tokens[0].lower()
    if mode not in _VALID_CONTENT_MODES:
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           f"Unknown oneline mode '{mode}'",
                           "Use one of: tts, sts, ttm, stt, se, sfx, svs, ss, train, quest."))
        return errors
    names_set = set(all_chain_names or [])
    placeholder_path = None
    try:
        fd, placeholder_path = tempfile.mkstemp(suffix=".wav", prefix="_voder_chain_input_")
        os.close(fd)
    except Exception:
        placeholder_path = None
    try:
        verify_tokens = []
        for t in tokens:
            if placeholder_path and (t == "input" or t in names_set):
                verify_tokens.append(placeholder_path)
            else:
                verify_tokens.append(t)
        parsed = parse_oneline_args(verify_tokens)
    except Exception:
        if placeholder_path and os.path.exists(placeholder_path):
            try:
                os.remove(placeholder_path)
            except Exception:
                pass
        return errors
    finally:
        if placeholder_path and os.path.exists(placeholder_path):
            try:
                os.remove(placeholder_path)
            except Exception:
                pass
    if parsed.get("error"):
        msg = parsed['error']
        if placeholder_path:
            msg = msg.replace(placeholder_path, "input")
        errors.append(_err(step_idx, chain_step["name"], "syntax",
                           f"Oneline parser error: {msg}",
                           "Fix the oneline syntax for this step."))
    return errors


def _verify_references(step_idx, chain_step, all_chain_names):
    errors = []
    tokens = chain_step["content_tokens"]
    step_name = chain_step["name"]
    prior_names = set()
    for n in all_chain_names:
        if n == step_name:
            break
        prior_names.add(n)
    for tok in tokens:
        if tok == "input":
            continue
        if tok == step_name:
            continue
        if tok in all_chain_names and tok not in prior_names:
            errors.append(_err(step_idx, step_name, "reference",
                               f"Forward reference: '{tok}' is defined later in the file (won't be available when this step runs)",
                               f"Move step '{tok}' before step '{step_name}', or remove the reference."))
    return errors


def classify_chain_step(chain_step, prior_chain_names):
    tokens = chain_step["content_tokens"]
    manual_count = sum(1 for t in tokens if t == "input")
    auto_count = sum(1 for t in tokens if t in prior_chain_names)
    if manual_count == 0 and auto_count > 0:
        return "automated", manual_count, auto_count
    if manual_count > 0 and auto_count == 0:
        return "manual", manual_count, auto_count
    if manual_count > 0 and auto_count > 0:
        return "semi-automated", manual_count, auto_count
    return "error", manual_count, auto_count


def find_chain_by_name(name):
    if not os.path.isdir(PREBUILT_CHAINS_DIR):
        return None
    ext_escaped = re.escape(CHAIN_FILE_EXT)
    pattern = re.compile(rf"^VODER_{re.escape(name)}_\d{{8}}_\d{{6}}{ext_escaped}$")
    matches = [os.path.join(PREBUILT_CHAINS_DIR, f)
               for f in os.listdir(PREBUILT_CHAINS_DIR) if pattern.match(f)]
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def list_chains():
    if not os.path.isdir(PREBUILT_CHAINS_DIR):
        return []
    out = []
    for entry in sorted(os.listdir(PREBUILT_CHAINS_DIR)):
        if not entry.endswith(CHAIN_FILE_EXT):
            continue
        if not entry.startswith("VODER_"):
            continue
        path = os.path.join(PREBUILT_CHAINS_DIR, entry)
        parsed, _ = parse_chain_file(path)
        if parsed is None:
            out.append({"path": path, "name": "", "timestamp": "",
                        "title": "", "description": "", "valid": False})
            continue
        out.append({
            "path": path,
            "name": parsed["name"],
            "timestamp": parsed["timestamp"],
            "title": parsed["title"],
            "description": parsed["description"],
            "valid": True,
        })
    out.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return out


def resolve_chain_path(name_or_path):
    if os.path.isfile(name_or_path):
        if not name_or_path.endswith(CHAIN_FILE_EXT):
            return None, f"File must end in '{CHAIN_FILE_EXT}': {name_or_path}"
        return name_or_path, None
    if "/" in name_or_path or "\\" in name_or_path:
        return None, f"File not found: {name_or_path}"
    path = find_chain_by_name(name_or_path)
    if path is None:
        return None, f"No prebuilt chain found with name '{name_or_path}' in {PREBUILT_CHAINS_DIR}"
    return path, None


def _parse_build_args(args):
    if len(args) < 2:
        return None, "Usage: chains build <name> description <title-desc> [chain <name> <comment> <content>]..."
    name = args[0]
    if not _NAME_RE.match(name):
        return None, f"Chain name '{name}' is invalid — use only letters, digits, underscores, hyphens. No spaces."
    if args[1].lower() != 'description':
        return None, f"Expected 'description' keyword after chain name, got '{args[1]}'"
    if len(args) < 3:
        return None, "Description text required after 'description' keyword (can be empty string \"\")"
    title_desc = args[2]
    rest = args[3:]
    steps = []
    i = 0
    while i < len(rest):
        if rest[i].lower() != 'chain':
            return None, f"Expected 'chain' keyword at position {i+4}, got '{rest[i]}'"
        if i + 3 >= len(rest):
            return None, f"'chain' keyword at position {i+4} must be followed by <name> <comment> <content> (3 quoted strings)"
        sname = rest[i + 1]
        scomment = rest[i + 2]
        scontent = rest[i + 3]
        if not _NAME_RE.match(sname):
            return None, f"Step name '{sname}' is invalid — use only letters, digits, underscores, hyphens. No spaces."
        if not scontent.strip():
            return None, f"Step '{sname}' has empty content"
        steps.append({"name": sname, "comment": scomment, "content": scontent})
        i += 4
    if not steps:
        return None, "At least one 'chain <name> <comment> <content>' block is required"
    seen = set()
    for s in steps:
        if s["name"] in seen:
            return None, f"Duplicate step name '{s['name']}' — each step must have a unique name"
        seen.add(s["name"])
    return {"name": name, "title_desc": title_desc, "steps": steps}, None


def handle_build(args):
    parsed, err = _parse_build_args(args)
    if err:
        print(f"Error: {err}")
        return False
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw = build_chain_text(parsed["name"], timestamp, parsed["title_desc"],
                           "", parsed["steps"])
    ok, errors, warnings = verify_chain_text(raw)
    print("Build verification:")
    if errors:
        for e in errors:
            loc = "file"
            if e["step_index"]:
                loc = f"step {e['step_index']} '{e['step_name']}'"
            print(f"  [ERROR] [{loc}] {e['category']}: {e['message']}")
            if e["fix"]:
                print(f"          fix: {e['fix']}")
        print(f"\n{len(errors)} error(s) found. Chain file was NOT saved.")
        return False
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    print(f"  [OK] All checks passed ({len(parsed['steps'])} step(s), 0 errors, {len(warnings)} warning(s)).")
    os.makedirs(PREBUILT_CHAINS_DIR, exist_ok=True)
    filename = f"VODER_{parsed['name']}_{timestamp}{CHAIN_FILE_EXT}"
    out_path = os.path.join(PREBUILT_CHAINS_DIR, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"\nSaved: {out_path}")
    total_manual = sum(1 for s in parsed["steps"] for t in s["content"].split() if t == "input")
    chain_names = [s["name"] for s in parsed["steps"]]
    total_auto = sum(1 for s in parsed["steps"] for t in s["content"].split()
                     if t in chain_names and t != s["name"])
    print(f"Summary: {len(parsed['steps'])} chain(s), {total_manual} manual input(s), {total_auto} automated reference(s).")
    print(f"\nTest it with:  python voder.py chains load \"{parsed['name']}\"")
    print(f"Journey it with:  python voder.py chains journey \"{parsed['name']}\"")
    return True


def handle_journey(args):
    if not args:
        print("Error: 'chains journey' requires at least one chain name or path.")
        print("Usage: python voder.py chains journey <chain-name-or-path> [<another> ...]")
        return False
    targets = []
    for arg in args:
        path, err = resolve_chain_path(arg)
        if err:
            print(f"Error: {err}")
            return False
        targets.append(path)
    chain_results = []
    for idx, path in enumerate(targets, start=1):
        parsed, _ = parse_chain_file(path)
        ok, errors, warnings = verify_chain_file(path)
        chain_results.append({"path": path, "parsed": parsed, "ok": ok,
                              "errors": errors, "warnings": warnings})
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_lines = _journey_report(chain_results, ts)
    safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', chain_results[0]["parsed"]["name"] if chain_results[0]["parsed"] else "unknown")[:60] or "unknown"
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"voder_journey_{safe_name}_{ts}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"Journey report saved to: {out_path}")
    any_errors = any(not cr["ok"] for cr in chain_results)
    if any_errors:
        print(f"{sum(len(cr['errors']) for cr in chain_results)} error(s) found — see report for the full journey.")
        return False
    print("All checks passed — the journey is complete.")
    return True


def _journey_report(chain_results, ts):
    lines = []
    lines.extend(_journey_opening(chain_results, ts))
    lines.append("")
    lines.append("## Cast of Chains")
    lines.append("")
    lines.append("| # | Name | Path | Steps | Status |")
    lines.append("|---|------|------|-------|--------|")
    for idx, cr in enumerate(chain_results, start=1):
        parsed = cr["parsed"]
        cname = parsed["name"] if parsed else os.path.basename(cr["path"])
        nsteps = len(parsed["chains"]) if parsed else 0
        status = "OK" if cr["ok"] else f"{len(cr['errors'])} error(s)"
        lines.append(f"| {idx} | {cname} | `{cr['path']}` | {nsteps} | {status} |")
    lines.append("")
    for idx, cr in enumerate(chain_results, start=1):
        lines.extend(_journey_one_chain(idx, cr, len(chain_results)))
        lines.append("")
    if len(chain_results) > 1:
        lines.extend(_journey_saga(chain_results))
        lines.append("")
    lines.extend(_journey_statistics(chain_results))
    lines.append("")
    lines.extend(_journey_epilogue(chain_results))
    lines.append("")
    return lines


_MODE_PERSONA = {
    'tts':   {'name': 'the Voice Weaver',    'verb': 'weaves spoken words from text, designing or cloning the speaker\'s voice'},
    'sts':   {'name': 'the Shape Shifter',   'verb': 'transforms one voice into another, preserving words and emotion'},
    'ttm':   {'name': 'the Song Smith',      'verb': 'forging music from lyrics and style descriptions'},
    'stt':   {'name': 'the Scribe',          'verb': 'transcribes speech to text, optionally identifying speakers'},
    'se':    {'name': 'the Restorer',        'verb': 'cleanses noise, dereverberates, and restores clarity to degraded audio'},
    'sfx':   {'name': 'the Sound Conjurer',  'verb': 'conjures sound effects from text descriptions'},
    'svs':   {'name': 'the Separator',       'verb': 'isolates vocals from music, or extracts the instrumental'},
    'ss':    {'name': 'the Crowd Sorter',    'verb': 'extracts individual speakers from multi-speaker recordings'},
    'train': {'name': 'the Voice Keeper',    'verb': 'trains and saves a voice clone for later reuse'},
    'quest': {'name': 'the Errand Runner',   'verb': 'runs a lightweight utility task outside the main engine'},
    'chains':{'name': 'the Chain Master',    'verb': 'orchestrates a pipeline of voder tasks'},
}


def _mode_persona(mode):
    return _MODE_PERSONA.get(mode, {'name': 'the Unknown Artisan', 'verb': 'performs an unrecognized operation'})


_CLASSIFICATION_NARRATIVE = {
    'manual':         'The traveler must provide {n} offering(s) to proceed — without them, this step cannot begin.',
    'automated':      'This step requires no offerings from the traveler; it draws entirely from what came before.',
    'semi-automated': 'This step blends fate and choice — {n} offering(s) from the traveler, plus the fruits of prior steps.',
    'error':          'This step stands at a crossroads with no clear path — neither offerings nor prior outputs guide it.',
}


def _journey_opening(chain_results, ts):
    lines = []
    lines.append("# VODER Chain Journey")
    lines.append("")
    readable_ts = _human_readable_timestamp(ts)
    lines.append(f"*The journey began on {readable_ts}.*")
    lines.append("")
    total_chains = len(chain_results)
    any_errors = any(not cr["ok"] for cr in chain_results)
    if total_chains == 1:
        parsed = chain_results[0]["parsed"]
        name = parsed["name"] if parsed else "an unknown chain"
        nsteps = len(parsed["chains"]) if parsed else 0
        if any_errors:
            lines.append(f"> In a world full of complexity and many of the unknowns, someone decided to build a chain called **{name}** to make their path easier. But did they? We shall find out.")
        else:
            lines.append(f"> In a world full of complexity and many of the unknowns, someone decided to build a chain called **{name}** to make their path easier — and so the journey of {nsteps} step(s) unfolds.")
    else:
        names = [cr["parsed"]["name"] if cr["parsed"] else f"chain {i+1}" for i, cr in enumerate(chain_results)]
        names_str = ", ".join(f"**{n}**" for n in names)
        if any_errors:
            lines.append(f"> In a world full of complexity and many of the unknowns, someone decided to build not one but {total_chains} chains — {names_str} — to make their path easier. But did they? We shall find out as the saga unfolds, chapter by chapter.")
        else:
            lines.append(f"> In a world full of complexity and many of the unknowns, someone decided to build {total_chains} chains — {names_str} — to make their path easier. The saga unfolds, chapter by chapter.")
    lines.append("")
    return lines


def _human_readable_timestamp(ts):
    try:
        dt = time.strptime(ts, "%Y%m%d_%H%M%S")
        return time.strftime("%B %d, %Y at %H:%M:%S", dt)
    except Exception:
        return ts


def _journey_one_chain(chain_idx, chain_result, total_chains):
    parsed = chain_result["parsed"]
    errors = chain_result["errors"]
    warnings = chain_result["warnings"]
    path = chain_result["path"]
    if parsed is None:
        lines = []
        lines.append(f"## Chapter {chain_idx}: The Broken Scroll")
        lines.append("")
        lines.append(f"> The scroll at `{path}` could not be read. Its runes are too corrupted to parse.")
        lines.append("")
        for e in errors:
            lines.append(f"- **[{e['category']}]** {e['message']}")
            if e["fix"]:
                lines.append(f"  - _Fix:_ {e['fix']}")
        lines.append("")
        return lines
    lines = []
    chapter_word = "Chapter" if total_chains > 1 else "Act"
    lines.append(f"## {chapter_word} {chain_idx}: The Chain of **{parsed['name']}**")
    lines.append("")
    readable_file_ts = _human_readable_timestamp(parsed["timestamp"])
    lines.append(f"> The journey of chain **{parsed['name']}** began on {readable_file_ts}, when it was first forged.")
    lines.append("")
    lines.append(f"- **Scroll:** `{path}`")
    lines.append(f"- **Forged:** {readable_file_ts}")
    lines.append(f"- **Title:** {parsed['title'] or '_(untitled)_'}")
    lines.append(f"- **Purpose:** {parsed['description'] or '_(unstated)_'}")
    lines.append(f"- **Steps in this chain:** {len(parsed['chains'])}")
    total_manual = sum(1 for c in parsed["chains"] for t in c["content_tokens"] if t == "input")
    total_auto = sum(1 for c in parsed["chains"] for t in c["content_tokens"]
                     if t in [cc["name"] for cc in parsed["chains"]] and t != c["name"])
    lines.append(f"- **Offerings required (manual inputs):** {total_manual}")
    lines.append(f"- **Echoes from prior steps (automated references):** {total_auto}")
    lines.append("")
    lines.append("### The Waypoints")
    lines.append("")
    lines.append("| # | Name | Type | Manual | Auto | Input comments | Step comment |")
    lines.append("|---|------|------|--------|------|----------------|--------------|")
    chain_names = [c["name"] for c in parsed["chains"]]
    for si, c in enumerate(parsed["chains"], start=1):
        prior = set(chain_names[:si-1])
        ctype, m, a = classify_chain_step(c, prior)
        ic_count = len([k for k, v in (c.get("input_comments") or {}).items() if v])
        ic_display = str(ic_count) if ic_count else "0"
        comment_excerpt = (c["comment"][:50] + "...") if len(c["comment"]) > 50 else (c["comment"] or "_(empty)_")
        comment_excerpt = comment_excerpt.replace("|", "\\|")
        lines.append(f"| {si} | {c['name']} | {ctype} | {m} | {a} | {ic_display} | {comment_excerpt} |")
    lines.append("")
    lines.append("### The Path Walked")
    lines.append("")
    lines.append("> Walk each step in execution order. Where a step falters, the error is shown — and a glimpse of an alternate dimension, where the error was corrected, reveals what could have been.")
    lines.append("")
    for si, c in enumerate(parsed["chains"], start=1):
        prior = set(chain_names[:si-1])
        ctype, m_count, a_count = classify_chain_step(c, prior)
        lines.append(f"#### Waypoint {si}: `{c['name']}` — {ctype}")
        lines.append("")
        if c["comment"]:
            lines.append(f"**The step's intent:** {c['comment']}")
        else:
            lines.append("**The step's intent:** _(none written — the traveler must infer the purpose from the content below)_")
        lines.append("")
        tokens = c["content_tokens"]
        mode = tokens[0].lower() if tokens else ""
        persona = _mode_persona(mode)
        if mode in _VALID_CONTENT_MODES:
            lines.append(f"**The artisan:** {persona['name']} — {persona['verb']}.")
        else:
            lines.append(f"**The artisan:** {persona['name']} — the engine does not recognize this mode (`{mode}`).")
        lines.append("")
        lines.append(f"**Content (raw):** `{c['content']}`")
        lines.append("")
        resolved_tokens = []
        slot_index_in_step = 0
        for tok in tokens:
            if tok == "input":
                slot_index_in_step += 1
                resolved_tokens.append(f"`<manual input {slot_index_in_step}>`")
            elif tok in prior:
                prior_idx = chain_names.index(tok) + 1
                resolved_tokens.append(f"`<output of step {prior_idx} '{tok}'>`")
            else:
                resolved_tokens.append(f"`{tok}`")
        lines.append(f"**Content (resolved):** {' '.join(resolved_tokens)}")
        lines.append("")
        narr = _CLASSIFICATION_NARRATIVE.get(ctype, _CLASSIFICATION_NARRATIVE['error'])
        narr_text = narr.format(n=m_count) if '{n}' in narr else narr
        lines.append(f"> _{narr_text}_")
        lines.append("")
        manual_positions = [(pos, tok) for pos, tok in enumerate(tokens) if tok == "input"]
        if manual_positions:
            lines.append("**Offerings awaited at this step:**")
            lines.append("")
            input_comments = c.get("input_comments") or {}
            for slot_i, (pos, _) in enumerate(manual_positions, start=1):
                step_mode = tokens[0].lower() if tokens else ""
                desc = describe_input_slot(step_mode, tokens, pos)
                vp_marker = ""
                if _is_voice_profile_position(tokens, pos):
                    vp_marker = " \u2014 **voice-profile eligible**"
                lines.append(f"- offering {slot_i} (position {pos}): {desc}{vp_marker}")
                if slot_i in input_comments and input_comments[slot_i]:
                    lines.append(f"  - _guidance:_ {input_comments[slot_i]}")
            lines.append("")
        step_errors = [e for e in errors if e["step_index"] == si]
        if step_errors:
            lines.extend(_journey_alternate_dimension(si, c, chain_names, step_errors))
            lines.append("")
            lines.append("> _The path continues, assuming this step had succeeded..._")
            lines.append("")
        else:
            if ctype == "manual":
                lines.append(f"> **The step holds.** It will ask the traveler for {m_count} offering(s).")
            elif ctype == "automated":
                lines.append(f"> **The step holds.** It is fully automated — the traveler need only press onward.")
            elif ctype == "semi-automated":
                lines.append(f"> **The step holds.** Semi-automated: {m_count} offering(s) from the traveler, {a_count} echo(es) from prior steps.")
            else:
                lines.append("> **The step holds.** It carries no external inputs — it will run on its inline arguments alone.")
            lines.append("")
    if warnings:
        lines.append("### Whispers Along the Path")
        lines.append("")
        lines.append("> Not all is amiss, but the path whispers of things to watch:")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")
    return lines


def _journey_alternate_dimension(step_idx, chain_step, all_chain_names, step_errors):
    lines = []
    lines.append("> **But the step falters.** Errors are found:")
    lines.append("")
    for e in step_errors:
        lines.append(f"> - **[{e['category']}]** {e['message']}")
        if e["fix"]:
            lines.append(f">   - _Fix:_ {e['fix']}")
    lines.append("")
    lines.append("> **In another dimension** — where the chain took another path, a valid path — what could have happened if the error were the correct thing?")
    lines.append("")
    whatif = _what_if_dimension(step_idx, chain_step, all_chain_names, step_errors)
    if whatif:
        lines.append(f"> {whatif}")
        lines.append("")
    else:
        lines.append("> _(The alternate dimension for this error category has not been charted yet.)_")
        lines.append("")
    return lines


def _what_if_dimension(step_idx, chain_step, all_chain_names, step_errors):
    tokens = chain_step["content_tokens"]
    if not tokens:
        return None
    mode = tokens[0].lower() if tokens[0] else ""
    error_categories = set(e["category"] for e in step_errors)
    parts = []
    if "reference" in error_categories:
        forward_refs = []
        prior = set()
        for n in all_chain_names:
            if n == chain_step["name"]:
                break
            prior.add(n)
        for tok in tokens:
            if tok in all_chain_names and tok not in prior and tok != chain_step["name"]:
                ref_step = all_chain_names.index(tok) + 1
                forward_refs.append(f"step {ref_step} '{tok}' would have been placed before this step")
        if forward_refs:
            parts.append("If " + "; and ".join(forward_refs) + ", the automated reference would have resolved to that step's output file at runtime, and the path would have continued unbroken.")
    if "syntax" in error_categories:
        persona = _mode_persona(mode)
        if mode not in _VALID_CONTENT_MODES:
            parts.append(f"If the mode had been a recognized one (tts, sts, ttm, stt, se, sfx, svs, ss, train, or quest), {persona['name']} would have taken the stage and the step would have executed that mode's pipeline.")
        else:
            parts.append(f"If the oneline syntax had been correct, {persona['name']} would have executed as a `{mode}` command with the provided arguments, and the step would have produced its output for the next waypoint.")
    if "naming" in error_categories:
        parts.append("If the name had matched `[A-Za-z0-9_-]+` and been unique within the file, the step would have been registered under that name and available for later steps to reference.")
    if "format" in error_categories:
        parts.append("If the format had been correct, the step block would have parsed cleanly and taken its place in the chain's sequence.")
    if not parts:
        return None
    return " ".join(parts)


def _journey_saga(chain_results):
    lines = []
    lines.append("## The Saga: How the Chapters Connect")
    lines.append("")
    lines.append("> When multiple prebuilt chains are loaded in one `chains load` command or one interactive CLI session, they execute in the order told here. Each prebuilt's final output is registered under its main name. Subsequent chapters can reference prior chapters' main names as manual input values — the runner resolves the name to that chapter's final output at runtime.")
    lines.append("")
    lines.append("**The order of the saga:**")
    lines.append("")
    for idx, cr in enumerate(chain_results, start=1):
        parsed = cr["parsed"]
        if not parsed:
            lines.append(f"{idx}. _(could not be read — see the chapter above)_")
            continue
        name = parsed["name"]
        manual_count = sum(1 for c in parsed["chains"] for t in c["content_tokens"] if t == "input")
        step_count = len(parsed["chains"])
        if idx == 1:
            lines.append(f"{idx}. **Chapter {idx}: {name}** — {step_count} step(s), {manual_count} offering(s). The first chapter; no prior chapters exist to echo from.")
        else:
            prior_names = [cr2["parsed"]["name"] for cr2 in chain_results[:idx-1] if cr2["parsed"]]
            prior_str = ", ".join(f"'{n}'" for n in prior_names) if prior_names else "(none)"
            lines.append(f"{idx}. **Chapter {idx}: {name}** — {step_count} step(s), {manual_count} offering(s). Can echo from prior chapters: {prior_str}.")
    lines.append("")
    lines.append("> **The linearity rule of the saga:** chapters execute strictly in order. A chapter cannot echo from a later chapter — that chapter's output does not exist yet at this point in the story. If you need chapter B's output in chapter A, tell chapter B's story first.")
    lines.append("")
    return lines


def _journey_statistics(chain_results):
    lines = []
    lines.append("## The Ledger of the Journey")
    lines.append("")
    total_chains = len(chain_results)
    total_steps = sum(len(cr["parsed"]["chains"]) for cr in chain_results if cr["parsed"])
    total_manual = sum(1 for cr in chain_results if cr["parsed"]
                       for c in cr["parsed"]["chains"]
                       for t in c["content_tokens"] if t == "input")
    total_auto = 0
    for cr in chain_results:
        if not cr["parsed"]:
            continue
        chain_names = [c["name"] for c in cr["parsed"]["chains"]]
        for c in cr["parsed"]["chains"]:
            for t in c["content_tokens"]:
                if t in chain_names and t != c["name"]:
                    total_auto += 1
    total_errors = sum(len(cr["errors"]) for cr in chain_results)
    total_warnings = sum(len(cr["warnings"]) for cr in chain_results)
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Chapters (prebuilt chains) | {total_chains} |")
    lines.append(f"| Waypoints (steps) | {total_steps} |")
    lines.append(f"| Offerings awaited (manual inputs) | {total_manual} |")
    lines.append(f"| Echoes from prior steps (automated references) | {total_auto} |")
    lines.append(f"| Errors found | {total_errors} |")
    lines.append(f"| Whispers (warnings) | {total_warnings} |")
    lines.append("")
    mode_counts = {}
    for cr in chain_results:
        if not cr["parsed"]:
            continue
        for c in cr["parsed"]["chains"]:
            tokens = c["content_tokens"]
            if tokens:
                mode = tokens[0].lower()
                mode_counts[mode] = mode_counts.get(mode, 0) + 1
    if mode_counts:
        lines.append("**Artisans summoned (by mode):**")
        lines.append("")
        lines.append("| Mode | Artisan | Steps |")
        lines.append("|------|---------|-------|")
        for mode in sorted(mode_counts.keys()):
            persona = _mode_persona(mode)
            lines.append(f"| `{mode}` | {persona['name']} | {mode_counts[mode]} |")
        lines.append("")
    if total_errors > 0:
        lines.append("### All Errors")
        lines.append("")
        lines.append("| Chapter | Waypoint | Category | Message | Fix |")
        lines.append("|---------|----------|----------|---------|-----|")
        for ci, cr in enumerate(chain_results, start=1):
            cname = cr["parsed"]["name"] if cr["parsed"] else f"chain {ci}"
            for e in cr["errors"]:
                step = f"{e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                msg = e["message"].replace("|", "\\|")
                fix = (e["fix"] or "").replace("|", "\\|")
                lines.append(f"| {cname} | {step} | {e['category']} | {msg} | {fix} |")
        lines.append("")
    return lines


def _journey_epilogue(chain_results):
    lines = []
    lines.append("## Epilogue")
    lines.append("")
    any_errors = any(not cr["ok"] for cr in chain_results)
    total_errors = sum(len(cr["errors"]) for cr in chain_results)
    total_chains = len(chain_results)
    if not any_errors:
        if total_chains == 1:
            lines.append("> The journey of this chain is whole. No errors were found. The path is clear — the traveler may now walk it with `chains load`.")
        else:
            lines.append(f"> The saga of {total_chains} chapters is whole. No errors were found. The path is clear — the traveler may now walk it with `chains load`.")
    else:
        if total_chains == 1:
            lines.append(f"> The journey falters at {total_errors} point(s). The errors above must be mended before this chain can be walked. Tend to them, and the path will open.")
        else:
            lines.append(f"> The saga falters at {total_errors} point(s) across its chapters. The errors above must be mended before these chains can be walked. Tend to them, and the path will open.")
    lines.append("")
    lines.append("> *The journey ends here. For now.*")
    lines.append("")
    return lines


def handle_load(args, result_path=None):
    sections, err = _parse_load_args(args)
    if err:
        print(f"Error: {err}")
        return False
    resolved_sections = []
    all_prebuilt_names_in_order = []
    for sec in sections:
        path, err = resolve_chain_path(sec["name_or_path"])
        if err:
            print(f"Error: {err}")
            return False
        ok, errors, _ = verify_chain_file(path)
        if not ok:
            print(f"Error: chain '{sec['name_or_path']}' failed verification:")
            for e in errors:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                print(f"  [{loc}] {e['category']}: {e['message']}")
            return False
        parsed, _ = parse_chain_file(path)
        all_prebuilt_names_in_order.append(parsed["name"])
        resolved_sections.append((sec, path, parsed))
    pipeline = ChainPipeline()
    prior_prebuilt_names = set()
    for sec_idx, (sec, path, parsed) in enumerate(resolved_sections, start=1):
        chain_names = [c["name"] for c in parsed["chains"]]
        total_steps_this = len(parsed["chains"])
        for step_num in sec["markers"]:
            _, idx_err = _resolve_linear_index(step_num, total_steps_this, "step", f"in '{parsed['name']}'")
            if idx_err:
                print(f"Error: {idx_err}")
                return False
        later_prebuilt_names = set(all_prebuilt_names_in_order[sec_idx:])
        print(f"\n[Prebuilt {sec_idx}/{len(resolved_sections)}] Loading '{parsed['name']}' ({total_steps_this} steps)")
        if parsed["title"]:
            print(f"  Title: {parsed['title']}")
        if parsed["description"]:
            print(f"  Description: {parsed['description']}")
        if prior_prebuilt_names:
            print(f"  Available prior prebuilt outputs to reference by name: {', '.join(sorted(prior_prebuilt_names))}")
        chains_args = []
        for step_idx, c in enumerate(parsed["chains"], start=1):
            tokens = list(c["content_tokens"])
            manual_slots = _find_manual_slots(tokens)
            auto_slots = _find_auto_slots(tokens, set(chain_names), pipeline.index, c["name"])
            user_values = sec["markers"].get(step_idx)
            if user_values is not None:
                if len(user_values) != len(manual_slots):
                    print(f"  Error: step {step_idx} '{c['name']}' has {len(manual_slots)} manual input slot(s) but marker provides {len(user_values)} value(s)")
                    print(f"         (automated slots are auto-resolved and never take values)")
                    return False
                substituted = list(tokens)
                for (pos, _), value in zip(manual_slots, user_values):
                    if value in later_prebuilt_names and value not in prior_prebuilt_names:
                        print(f"  Error: step {step_idx} '{c['name']}' marker value '{value}' is a forward reference — prebuilt '{value}' is loaded later in this command (position {all_prebuilt_names_in_order.index(value) + 1}) but hasn't run yet")
                        print(f"         Reorder: load '{value}' before '{parsed['name']}', or provide a file path/URL instead.")
                        return False
                    resolved = _resolve_manual_value(value, prior_prebuilt_names, pipeline)
                    if resolved is None:
                        print(f"    [step {step_idx}] manual slot -> file/URL: {value}")
                        substituted[pos] = value
                    else:
                        print(f"    [step {step_idx}] manual slot -> prior prebuilt '{value}' output: {resolved}")
                        substituted[pos] = resolved
            else:
                substituted = list(tokens)
                if manual_slots:
                    print(f"  Error: step {step_idx} '{c['name']}' has {len(manual_slots)} manual input(s) but no marker was provided")
                    print(f"         Provide a marker: {step_idx}:\"(value1/value2/...)\"")
                    return False
            if auto_slots:
                for pos, slot_name in auto_slots:
                    if substituted[pos] == slot_name:
                        if slot_name in pipeline.index:
                            print(f"    [step {step_idx}] auto slot '{slot_name}' -> auto-resolved: {pipeline.index[slot_name]}")
                        else:
                            print(f"    [step {step_idx}] auto slot '{slot_name}' -> NOT YET RESOLVED (will resolve at runtime)")
            if step_idx > 1:
                chains_args.append(ChainPipeline.CHAIN_SEPARATOR)
            chains_args.append(c["name"])
            chains_args.extend(substituted)
        ok, err = pipeline.execute(chains_args, result_path=result_path if sec_idx == len(resolved_sections) else None)
        if not ok:
            err_msg = (err or "unknown error")[:500]
            print()
            print("=" * 60)
            print("Something went further than expected.")
            print(f"Error (at prebuilt {sec_idx} '{parsed['name']}'): {err_msg}")
            print("=" * 60)
            return False
        final_step = parsed["chains"][-1]["name"]
        if final_step in pipeline.index:
            pipeline.index[parsed["name"]] = pipeline.index[final_step]
            prior_prebuilt_names.add(parsed["name"])
            print(f"\n[Prebuilt {sec_idx}] '{parsed['name']}' completed. Final output registered under name '{parsed['name']}'.")
    return True


def _resolve_manual_value(value, prior_prebuilt_names, pipeline):
    if not value:
        return None
    if value in prior_prebuilt_names:
        if value in pipeline.index:
            return pipeline.index[value]
        return None
    return None


def _find_manual_slots(tokens):
    return [(pos, "input") for pos, tok in enumerate(tokens) if tok == "input"]


def _find_auto_slots(tokens, all_chain_names, global_index, current_step_name):
    slots = []
    for pos, tok in enumerate(tokens):
        if tok == "input":
            continue
        if tok == current_step_name:
            continue
        if tok in all_chain_names or tok in global_index:
            slots.append((pos, tok))
    return slots


def _is_voice_profile_position(content_tokens, slot_pos):
    if not content_tokens:
        return False
    mode = content_tokens[0].lower()
    return slot_accepts_voice_profile(mode, content_tokens, slot_pos)


def get_input_formats_for_step(content_tokens):
    if not content_tokens:
        return "(unknown — content is empty)"
    mode = content_tokens[0].lower()
    return MODE_INPUT_FORMATS.get(mode, f"(unknown mode '{mode}')")


def is_voice_profile_value(value):
    if not value:
        return False
    if ':' in value:
        _, _, rest = value.partition(':')
        value = rest.strip()
    if not value:
        return False
    lower = value.lower()
    return lower.endswith('.tts') or lower.endswith('.ttse')


def _parse_load_args(args):
    if not args:
        return None, "Usage: chains load <chain-name-or-path> [N:(v1/v2/...)]... [<another-chain> [N:(...)]...]..."
    sections = []
    current = None
    marker_re = re.compile(r'^(\d+):\((.*)\)$')
    for arg in args:
        m = marker_re.match(arg)
        if m:
            if current is None:
                return None, f"Marker '{arg}' appears before any chain name"
            step_num = int(m.group(1))
            values_raw = m.group(2)
            if not values_raw.strip():
                values = []
            else:
                values = values_raw.split('/')
            if step_num in current["markers"]:
                return None, f"Duplicate marker for step {step_num} in chain '{current['name_or_path']}'"
            current["markers"][step_num] = values
        else:
            if current:
                sections.append(current)
            current = {"name_or_path": arg, "markers": {}}
    if current:
        sections.append(current)
    if not sections:
        return None, "At least one chain name or path is required"
    return sections, None


def _parse_comment_args(args):
    if not args:
        return None, "Usage: chains comment <chain-name-or-path> [N:\"<new chain comment>\"]... [N:(I1:<input comment>/I2:<input comment>/...)]..."
    name_or_path = args[0]
    rest = args[1:]
    chain_comment_edits = {}
    input_comment_edits = {}
    chain_comment_re = re.compile(r'^(\d+):"(.*)"$')
    input_block_re = re.compile(r'^(\d+):\((.*)\)$')
    for arg in rest:
        m = chain_comment_re.match(arg)
        if m:
            step_num = int(m.group(1))
            new_comment = m.group(2)
            if step_num in chain_comment_edits:
                return None, f"Duplicate chain-comment edit for step {step_num}"
            chain_comment_edits[step_num] = new_comment
            continue
        m = input_block_re.match(arg)
        if m:
            step_num = int(m.group(1))
            body = m.group(2)
            if step_num in input_comment_edits:
                return None, f"Duplicate input-comment edit for step {step_num}"
            slot_edits = {}
            if body.strip():
                parts = body.split('/')
                for part in parts:
                    if ':' not in part:
                        return None, f"Malformed input comment '{part}' — expected 'I:<comment>' (input index, colon, comment text)"
                    idx_str, _, comment_text = part.partition(':')
                    idx_str = idx_str.strip()
                    if not idx_str.isdigit():
                        return None, f"Malformed input comment '{part}' — input index must be a positive integer"
                    input_idx = int(idx_str)
                    if input_idx < 1:
                        return None, f"Invalid input index '{input_idx}' — input slots are 1-indexed; use 1 for the first input"
                    if input_idx in slot_edits:
                        return None, f"Duplicate input index {input_idx} in step {step_num}'s input-comment block"
                    slot_edits[input_idx] = comment_text
            input_comment_edits[step_num] = slot_edits
            continue
        return None, f"Unrecognized argument '{arg}' — expected N:\"<chain comment>\" or N:(I1:<comment>/I2:<comment>/...)"
    if not chain_comment_edits and not input_comment_edits:
        return None, "At least one edit is required — provide N:\"<chain comment>\" or N:(I:<input comment>/...)"
    return {
        "name_or_path": name_or_path,
        "chain_comment_edits": chain_comment_edits,
        "input_comment_edits": input_comment_edits,
    }, None


def handle_comment(args):
    parsed, err = _parse_comment_args(args)
    if err:
        print(f"Error: {err}")
        return False
    path, err = resolve_chain_path(parsed["name_or_path"])
    if err:
        print(f"Error: {err}")
        return False
    chain_parsed, parse_errs = parse_chain_file(path)
    if chain_parsed is None:
        print(f"Error: chain file could not be parsed:")
        for e in parse_errs:
            loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
            print(f"  [{loc}] {e['category']}: {e['message']}")
        return False
    chains = chain_parsed["chains"]
    total_steps = len(chains)
    chain_comment_edits = parsed["chain_comment_edits"]
    input_comment_edits = parsed["input_comment_edits"]
    for step_num in chain_comment_edits:
        _, err = _resolve_linear_index(step_num, total_steps, "step", f"in '{chain_parsed['name']}'")
        if err:
            print(f"Error: {err}")
            return False
    for step_num, slot_edits in input_comment_edits.items():
        zero_idx, err = _resolve_linear_index(step_num, total_steps, "step", f"in '{chain_parsed['name']}'")
        if err:
            print(f"Error: {err}")
            return False
        step = chains[zero_idx]
        manual_count = sum(1 for t in step["content_tokens"] if t == "input")
        for input_idx in slot_edits:
            _, err = _resolve_linear_index(input_idx, manual_count, "input slot", f"in step {step_num} '{step['name']}'")
            if err:
                print(f"Error: {err}")
                return False
    for step_num, new_comment in chain_comment_edits.items():
        zero_idx = step_num - 1
        chains[zero_idx]["comment"] = new_comment
    for step_num, slot_edits in input_comment_edits.items():
        zero_idx = step_num - 1
        step = chains[zero_idx]
        existing = dict(step.get("input_comments") or {})
        for input_idx, comment_text in slot_edits.items():
            existing[input_idx] = comment_text
        step["input_comments"] = existing
    raw = build_chain_text(chain_parsed["name"], chain_parsed["timestamp"],
                           chain_parsed["title"], chain_parsed["description"], chains)
    ok, errors, warnings = verify_chain_text(raw)
    if not ok:
        print("Error: verification failed after applying comment edits — file was NOT saved:")
        for e in errors:
            loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
            print(f"  [{loc}] {e['category']}: {e['message']}")
            if e["fix"]:
                print(f"          fix: {e['fix']}")
        return False
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"Updated: {path}")
    print(f"Summary: {len(chain_comment_edits)} chain comment(s) edited, "
          f"{sum(len(v) for v in input_comment_edits.values())} input comment(s) edited across "
          f"{len(input_comment_edits)} step(s).")
    if warnings:
        for w in warnings:
            print(f"  [WARN] {w}")
    return True


def handle_decompile(args):
    if not args:
        print("Error: 'chains decompile' requires at least one chain name or path.")
        print("Usage: python voder.py chains decompile <chain-name-or-path> [<another> ...]")
        return False
    targets = []
    for arg in args:
        path, err = resolve_chain_path(arg)
        if err:
            print(f"Error: {err}")
            return False
        targets.append(path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(results_dir, exist_ok=True)
    any_errors = False
    for idx, path in enumerate(targets, start=1):
        parsed, parse_errs = parse_chain_file(path)
        ok, errors, warnings = verify_chain_file(path)
        if parsed is None:
            print(f"\n[{idx}/{len(targets)}] Could not parse: {path}")
            for e in parse_errs:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                print(f"  [{loc}] {e['category']}: {e['message']}")
                if e["fix"]:
                    print(f"          fix: {e['fix']}")
            any_errors = True
            continue
        txt_lines = []
        txt_lines.append(f"# VODER decompiled chain: {parsed['name']}")
        txt_lines.append(f"# Source: {path}")
        txt_lines.append(f"# Decompiled: {_human_readable_timestamp(ts)}")
        txt_lines.append(f"# Title: {parsed['title'] or '(empty)'}")
        txt_lines.append(f"# Description: {parsed['description'] or '(empty)'}")
        txt_lines.append(f"# Steps: {len(parsed['chains'])}")
        txt_lines.append("#")
        txt_lines.append("# This file contains the raw chains oneline command that produces the same")
        txt_lines.append("# pipeline as the source .chain file. Edit the command below, then recompile with:")
        txt_lines.append(f"#   python voder.py chains compile \"{os.path.basename(path).replace(CHAIN_FILE_EXT, '.txt')}\"")
        txt_lines.append("#")
        txt_lines.append("# Each chain step is quoted-named, followed by its oneline command.")
        txt_lines.append("# Steps are separated by ' / ' (space slash space).")
        txt_lines.append("# The literal token 'input' marks a manual file input slot.")
        txt_lines.append("# Prior chain names referenced verbatim are automated references.")
        txt_lines.append("")
        segments = []
        for c in parsed["chains"]:
            tokens = c["content_tokens"]
            content_str = " ".join(tokens)
            segments.append(f'"{c["name"]}" {content_str}')
        oneline_command = " / ".join(segments)
        txt_lines.append(oneline_command)
        if errors:
            txt_lines.append("")
            txt_lines.append("# --- VERIFICATION ERRORS (commented out — fix the source chain to clear these) ---")
            for e in errors:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                txt_lines.append(f"# [{loc}] {e['category']}: {e['message']}")
                if e["fix"]:
                    txt_lines.append(f"#   fix: {e['fix']}")
        if warnings:
            txt_lines.append("")
            txt_lines.append("# --- WARNINGS ---")
            for w in warnings:
                txt_lines.append(f"# {w}")
        safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', parsed["name"])[:60] or "unknown"
        out_path = os.path.join(results_dir, f"VODER_chains_{safe_name}_decompiled_{ts}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(txt_lines) + "\n")
        status_tag = "OK" if ok else f"{len(errors)} error(s) commented"
        print(f"\n[{idx}/{len(targets)}] Decompiled '{parsed['name']}' ({len(parsed['chains'])} step(s), {status_tag})")
        print(f"  Source: {path}")
        print(f"  Output: {out_path}")
        if errors:
            any_errors = True
            print(f"  {len(errors)} error(s) found — commented out at the bottom of the .txt file.")
        if warnings:
            print(f"  {len(warnings)} warning(s) — commented out at the bottom of the .txt file.")
    print(f"\nDecompiled {len(targets)} chain(s) to results/.")
    if any_errors:
        print("Some chain(s) had errors — see the commented-out sections in the .txt file(s).")
        return False
    return True


def handle_compile(args):
    if not args:
        print("Error: 'chains compile' requires at least one .txt file path.")
        print("Usage: python voder.py chains compile <txt-path> [<another> ...]")
        return False
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(PREBUILT_CHAINS_DIR, exist_ok=True)
    all_ok = True
    for idx, txt_path in enumerate(args, start=1):
        if not os.path.isfile(txt_path):
            print(f"\n[{idx}/{len(args)}] Error: file not found: {txt_path}")
            all_ok = False
            continue
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                raw_txt = f.read()
        except Exception as e:
            print(f"\n[{idx}/{len(args)}] Error: could not read {txt_path}: {e}")
            all_ok = False
            continue
        compiled = _compile_txt_to_chain(raw_txt, txt_path)
        if compiled is None:
            print(f"\n[{idx}/{len(args)}] Error: could not parse .txt structure: {txt_path}")
            print("  Expected a header line '# VODER decompiled chain: <name>' and a command line.")
            all_ok = False
            continue
        name = compiled["name"]
        title = compiled["title"]
        description = compiled["description"]
        steps = compiled["steps"]
        new_ts = time.strftime("%Y%m%d_%H%M%S")
        chain_raw = build_chain_text(name, new_ts, title, description, steps)
        ok, errors, warnings = verify_chain_text(chain_raw)
        print(f"\n[{idx}/{len(args)}] Compile verification for '{name}':")
        if errors:
            for e in errors:
                loc = f"step {e['step_index']} '{e['step_name']}'" if e["step_index"] else "file"
                print(f"  [ERROR] [{loc}] {e['category']}: {e['message']}")
                if e["fix"]:
                    print(f"          fix: {e['fix']}")
            print(f"  {len(errors)} error(s) found. Chain file was NOT saved.")
            all_ok = False
            continue
        if warnings:
            for w in warnings:
                print(f"  [WARN] {w}")
        print(f"  [OK] All checks passed ({len(steps)} step(s), 0 errors, {len(warnings)} warning(s)).")
        filename = f"VODER_{name}_{new_ts}{CHAIN_FILE_EXT}"
        out_path = os.path.join(PREBUILT_CHAINS_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(chain_raw)
        print(f"  Saved: {out_path}")
        total_manual = sum(1 for s in steps for t in s["content"].split() if t == "input")
        chain_names = [s["name"] for s in steps]
        total_auto = sum(1 for s in steps for t in s["content"].split()
                         if t in chain_names and t != s["name"])
        print(f"  Summary: {len(steps)} chain(s), {total_manual} manual input(s), {total_auto} automated reference(s).")
        print(f"  Test it with:  python voder.py chains load \"{name}\"")
        print(f"  Journey it with:  python voder.py chains journey \"{name}\"")
    return all_ok


def _compile_txt_to_chain(raw_txt, source_path):
    lines = raw_txt.splitlines()
    name = None
    title = ""
    description = ""
    command_line = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            if stripped.lower().startswith("# voder decompiled chain:"):
                rest = stripped[len("# voder decompiled chain:"):].strip()
                name = rest
            elif stripped.lower().startswith("# title:"):
                title = stripped[len("# title:"):].strip()
                if title == "(empty)":
                    title = ""
            elif stripped.lower().startswith("# description:"):
                description = stripped[len("# description:"):].strip()
                if description == "(empty)":
                    description = ""
            continue
        if command_line is None:
            command_line = stripped
            break
    if name is None or command_line is None:
        return None
    if not _NAME_RE.match(name):
        return None
    segments = _split_oneline_segments(command_line)
    if segments is None:
        return None
    steps = []
    seen_names = set()
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.startswith('"'):
            end_quote = seg.find('"', 1)
            if end_quote == -1:
                return None
            step_name = seg[1:end_quote]
            content = seg[end_quote + 1:].strip()
        else:
            tokens = seg.split(None, 1)
            step_name = tokens[0] if tokens else ""
            content = tokens[1] if len(tokens) > 1 else ""
        if not step_name or not _NAME_RE.match(step_name):
            return None
        if step_name in seen_names:
            return None
        seen_names.add(step_name)
        if not content.strip():
            return None
        steps.append({
            "name": step_name,
            "comment": "",
            "content": content,
            "content_tokens": content.split(),
            "input_comments": {},
        })
    if not steps:
        return None
    return {"name": name, "title": title, "description": description, "steps": steps}


def _split_oneline_segments(command_line):
    segments = []
    current = []
    i = 0
    n = len(command_line)
    while i < n:
        if command_line[i] == '"':
            end = command_line.find('"', i + 1)
            if end == -1:
                return None
            current.append(command_line[i:end + 1])
            i = end + 1
        elif command_line[i] == ' ' and i + 2 < n and command_line[i + 1] == '/' and command_line[i + 2] == ' ':
            segments.append("".join(current))
            current = []
            i += 3
        else:
            current.append(command_line[i])
            i += 1
    if current:
        segments.append("".join(current))
    return segments


_vadar_model = None
_vadar_processor = None
_vadar_model_loading_attempted = False


def vadar_check_model_downloaded():
    if not os.path.isdir(VADAR_MODEL_DIR):
        return False
    has_weights = any(
        f.endswith('.safetensors') or f.endswith('.bin')
        for f in os.listdir(VADAR_MODEL_DIR)
    )
    has_config = os.path.exists(os.path.join(VADAR_MODEL_DIR, 'config.json'))
    return has_weights and has_config


def vadar_load_model(force_reload=False):
    global _vadar_model, _vadar_processor, _vadar_model_loading_attempted
    if _vadar_model is not None and not force_reload:
        return _vadar_model, _vadar_processor, None
    if _vadar_model_loading_attempted and not force_reload:
        return None, None, "Model loading was already attempted and failed."
    _vadar_model_loading_attempted = True

    try:
        import torch
    except ImportError:
        return None, None, "torch is not installed. Run: pip install torch"

    try:
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except ImportError:
        try:
            from transformers import AutoModelForCausalLM as AutoModelForMultimodalLM
            from transformers import AutoTokenizer as AutoProcessor
        except ImportError:
            return None, None, "transformers is not installed. Run: pip install transformers"

    if not vadar_check_model_downloaded():
        print(f"VADAR: model not found at {VADAR_MODEL_DIR}. Downloading from {VADAR_MODEL_REPO}...")
        print("VADAR: this is a ~24GB download. It may take a while depending on your connection.")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            return None, None, "huggingface-hub is not installed. Run: pip install huggingface-hub"
        os.makedirs(VADAR_MODEL_DIR, exist_ok=True)
        try:
            snapshot_download(
                repo_id=VADAR_MODEL_REPO,
                local_dir=VADAR_MODEL_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            return None, None, f"Model download failed: {e}"
        if not vadar_check_model_downloaded():
            return None, None, "Download completed but model files are missing."
        print(f"VADAR: model downloaded to {VADAR_MODEL_DIR}")

    print(f"VADAR: loading model from {VADAR_MODEL_DIR}...")
    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        _vadar_model = AutoModelForMultimodalLM.from_pretrained(
            VADAR_MODEL_DIR,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        _vadar_processor = AutoProcessor.from_pretrained(VADAR_MODEL_DIR, trust_remote_code=True)
        _vadar_model.eval()
        print("VADAR: model loaded successfully.")
        return _vadar_model, _vadar_processor, None
    except Exception as e:
        _vadar_model = None
        _vadar_processor = None
        return None, None, f"Failed to load model: {e}"


def vadar_run_inference(messages, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=64):
    model, processor, err = vadar_load_model()
    if err:
        return None, err
    try:
        import torch
    except ImportError:
        return None, "torch not available"

    try:
        if hasattr(processor, 'apply_chat_template'):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = '\n'.join(f"{m['role']}: {m['content']}" for m in messages) + '\nassistant: '

        inputs = processor(text=text, return_tensors='pt').to(
            model.device if hasattr(model, 'device') else 'cpu'
        )

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
            )

        input_len = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        new_tokens = output[0][input_len:]
        response = processor.decode(new_tokens, skip_special_tokens=True)
        return response, None
    except Exception as e:
        return None, str(e)


def vadar_run_inference_streamed(messages, max_new_tokens=1024, temperature=0.8, top_p=0.95, top_k=64):
    model, processor, err = vadar_load_model()
    if err:
        return None, err
    try:
        import torch
    except ImportError:
        return None, "torch not available"

    try:
        from transformers import TextIteratorStreamer
    except ImportError:
        return vadar_run_inference(messages, max_new_tokens, temperature, top_p, top_k)

    try:
        if hasattr(processor, 'apply_chat_template'):
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = '\n'.join(f"{m['role']}: {m['content']}" for m in messages) + '\nassistant: '

        inputs = processor(text=text, return_tensors='pt').to(
            model.device if hasattr(model, 'device') else 'cpu'
        )

        streamer = TextIteratorStreamer(
            processor.tokenizer if hasattr(processor, 'tokenizer') else processor,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
            streamer=streamer,
        )

        thread = threading.Thread(target=lambda: model.generate(**generation_kwargs))
        thread.start()

        collected = []
        _stream_state = {'buffer': '', 'in_tag': False, 'tag_name': '', 'tag_content': [], 'char_count': 0, 'last_label': '', 'last_print': 0}
        _TAG_PATTERN = re.compile(r'<(/?)(thinking|decide|reply|act|tool_call|eval|EOS_REPLY|EOS_ACT|EOS_DONE)([^>]*)>')

        def _process_stream_chunk(chunk):
            _stream_state['buffer'] += chunk
            collected.append(chunk)
            while True:
                buf = _stream_state['buffer']
                if _stream_state['in_tag']:
                    end_idx = buf.find(f'</{_stream_state["tag_name"]}>')
                    if end_idx == -1:
                        content_part = buf
                        _stream_state['tag_content'].append(content_part)
                        _stream_state['char_count'] += len(content_part)
                        _stream_state['buffer'] = ''
                        label_map = {'thinking': '[THINKING]', 'decide': '[DECIDE]', 'reply': '[REPLY]', 'act': '[ACT]', 'tool_call': '[TOOL]', 'eval': '[EVAL]'}
                        label = label_map.get(_stream_state['tag_name'], f'[{_stream_state["tag_name"].upper()}]')
                        if label != _stream_state['last_label']:
                            sys.stdout.write(f'\n{label}: ')
                            sys.stdout.flush()
                            _stream_state['last_label'] = label
                            _stream_state['char_count'] = 0
                            _stream_state['last_print'] = time.time()
                        now = time.time()
                        if now - _stream_state['last_print'] > 0.5:
                            sys.stdout.write(f'\r{label}: {_stream_state["char_count"]} chars...  ')
                            sys.stdout.flush()
                            _stream_state['last_print'] = now
                        break
                    else:
                        content_part = buf[:end_idx]
                        _stream_state['tag_content'].append(content_part)
                        _stream_state['char_count'] += len(content_part)
                        label_map = {'thinking': '[THINKING]', 'decide': '[DECIDE]', 'reply': '[REPLY]', 'act': '[ACT]', 'tool_call': '[TOOL]', 'eval': '[EVAL]'}
                        label = label_map.get(_stream_state['tag_name'], f'[{_stream_state["tag_name"].upper()}]')
                        if label != _stream_state['last_label']:
                            sys.stdout.write(f'\n{label}: ')
                            sys.stdout.flush()
                            _stream_state['last_label'] = label
                        full_content = ''.join(_stream_state['tag_content'])
                        if _stream_state['tag_name'] == 'reply':
                            sys.stdout.write(f'\r[REPLY]: {full_content}\n')
                        elif _stream_state['tag_name'] == 'act':
                            sys.stdout.write(f'\r[ACT]: {full_content}\n')
                        elif _stream_state['tag_name'] == 'tool_call':
                            sys.stdout.write(f'\r[TOOL]: {full_content}\n')
                        else:
                            sys.stdout.write(f'\r{label}: {full_content}\n')
                        sys.stdout.flush()
                        _stream_state['buffer'] = buf[end_idx + len(f'</{_stream_state["tag_name"]}>'):]
                        _stream_state['in_tag'] = False
                        _stream_state['tag_content'] = []
                        _stream_state['char_count'] = 0
                        _stream_state['last_label'] = ''
                else:
                    m = _TAG_PATTERN.search(buf)
                    if m is None:
                        if '<' in buf and not buf.endswith('<'):
                            pass
                        else:
                            text_before = buf
                            if text_before.strip():
                                pass
                            _stream_state['buffer'] = ''
                        break
                    text_before = buf[:m.start()]
                    if text_before.strip():
                        pass
                    tag_name = m.group(2)
                    is_close = m.group(1) == '/'
                    if not is_close:
                        _stream_state['in_tag'] = True
                        _stream_state['tag_name'] = tag_name
                        _stream_state['tag_content'] = []
                        _stream_state['char_count'] = 0
                        _stream_state['buffer'] = buf[m.end():]
                    else:
                        if tag_name in ('EOS_REPLY', 'EOS_ACT', 'EOS_DONE'):
                            sys.stdout.write(f'\n[{tag_name}]\n')
                            sys.stdout.flush()
                        _stream_state['buffer'] = buf[m.end():]

        for chunk in streamer:
            _process_stream_chunk(chunk)
        thread.join()

        if _stream_state['in_tag'] and _stream_state['tag_content']:
            label_map = {'thinking': '[THINKING]', 'decide': '[DECIDE]', 'reply': '[REPLY]', 'act': '[ACT]', 'tool_call': '[TOOL]', 'eval': '[EVAL]'}
            label = label_map.get(_stream_state['tag_name'], f'[{_stream_state["tag_name"].upper()}]')
            full_content = ''.join(_stream_state['tag_content'])
            sys.stdout.write(f'\r{label}: {full_content}\n')
            sys.stdout.flush()

        full = ''.join(collected)
        sys.stdout.write('\n')
        sys.stdout.flush()
        return full, None
    except Exception as e:
        return None, str(e)


_lite_vadar_llm = None
_lite_vadar_template = None


def lite_vadar_check_model_downloaded():
    gguf_path = os.path.join(LITE_VADAR_MODEL_DIR, LITE_VADAR_GGUF_FILENAME)
    return os.path.exists(gguf_path) and os.path.getsize(gguf_path) > 0


_LITE_MODEL_SIZE_GB = 7.5
_LITE_OVERHEAD_GB = 4.0
_LITE_KV_PER_TOKEN_MB = 0.4
_LITE_MIN_CONTEXT = 2048
_LITE_MAX_CONTEXT = 262144


def _calculate_dynamic_context(config):
    manual = config.get('lite_context_length', 0)
    if manual and manual > 0:
        return min(manual, _LITE_MAX_CONTEXT)

    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        available_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        ram_gb = 16
        available_gb = 12

    vram_gb = 0
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb += props.total_memory / (1024**3)
    except Exception:
        pass

    total_pool = available_gb + vram_gb
    usable = total_pool - _LITE_MODEL_SIZE_GB - _LITE_OVERHEAD_GB
    if usable <= 0:
        return _LITE_MIN_CONTEXT

    context = int((usable * 1024) / _LITE_KV_PER_TOKEN_MB)
    context = max(_LITE_MIN_CONTEXT, min(context, _LITE_MAX_CONTEXT))
    rounded = 512
    context = (context // rounded) * rounded

    print(f"VADAR LITE: dynamic context — RAM: {available_gb:.1f}GB available, VRAM: {vram_gb:.1f}GB, model: {_LITE_MODEL_SIZE_GB}GB, overhead: {_LITE_OVERHEAD_GB}GB → {context} tokens ({usable:.1f}GB for KV cache at {_LITE_KV_PER_TOKEN_MB}MB/token)")
    return context


def lite_vadar_load_model(force_reload=False):
    global _lite_vadar_llm, _lite_vadar_template
    if _lite_vadar_llm is not None and not force_reload:
        return _lite_vadar_llm, _lite_vadar_template, None

    try:
        from llama_cpp import Llama
    except ImportError:
        return None, None, "llama-cpp-python not installed. Run: pip install llama-cpp-python"

    if not lite_vadar_check_model_downloaded():
        print(f"VADAR LITE: model not found at {LITE_VADAR_MODEL_DIR}. Downloading from {LITE_VADAR_MODEL_REPO}...")
        os.makedirs(LITE_VADAR_MODEL_DIR, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            gguf_path = hf_hub_download(
                repo_id=LITE_VADAR_MODEL_REPO,
                filename=LITE_VADAR_GGUF_FILENAME,
                local_dir=LITE_VADAR_MODEL_DIR,
            )
        except Exception as e:
            return None, None, f"Model download failed: {e}"
        if not lite_vadar_check_model_downloaded():
            return None, None, "Download completed but model files are missing."
        print(f"VADAR LITE: model downloaded to {LITE_VADAR_MODEL_DIR}")

    print(f"VADAR LITE: loading model from {LITE_VADAR_MODEL_DIR}...")
    try:
        config = vadar_load_config()
        ctx_len = _calculate_dynamic_context(config)
        gpu_layers = config.get('lite_gpu_layers', -1)
        n_threads = config.get('lite_n_threads', -1)
        verbose = config.get('lite_verbose', False)

        gguf_path = os.path.join(LITE_VADAR_MODEL_DIR, LITE_VADAR_GGUF_FILENAME)
        _lite_vadar_llm = Llama(
            model_path=gguf_path,
            n_ctx=ctx_len,
            n_gpu_layers=gpu_layers,
            n_threads=n_threads,
            verbose=verbose,
        )

        print("VADAR LITE: model loaded successfully.")
        return _lite_vadar_llm, _lite_vadar_template, None
    except Exception as e:
        _lite_vadar_llm = None
        _lite_vadar_template = None
        return None, None, f"Failed to load lite model: {e}"


_LITE_THINK_START = "<|channel>thought\n"
_LITE_THINK_END = "<channel|>"


def _parse_lite_thinking(text):
    thinking_parts = []
    remaining = text
    while True:
        start = remaining.find(_LITE_THINK_START)
        if start == -1:
            break
        end = remaining.find(_LITE_THINK_END, start)
        if end == -1:
            break
        thought = remaining[start + len(_LITE_THINK_START):end].strip()
        if thought:
            thinking_parts.append(thought)
        remaining = remaining[:start] + remaining[end + len(_LITE_THINK_END):]
    return thinking_parts, remaining.strip()


def lite_vadar_run_inference(messages, max_new_tokens=1024):
    llm, template, err = lite_vadar_load_model()
    if err:
        return None, err
    try:
        config = vadar_load_config()
        temperature = config.get('temperature', 0.8)
        top_p = config.get('top_p', 0.95)
        top_k = config.get('top_k', 64)
        repeat_penalty = config.get('lite_repeat_penalty', 1.1)

        resp = llm.create_chat_completion(
            messages=messages,
            stream=False,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        )
        raw = resp["choices"][0]["message"]["content"]
        thoughts, clean = _parse_lite_thinking(raw)
        return clean, None
    except Exception as e:
        return None, str(e)


def lite_vadar_run_inference_streamed(messages, max_new_tokens=1024):
    llm, template, err = lite_vadar_load_model()
    if err:
        return None, err
    try:
        config = vadar_load_config()
        temperature = config.get('temperature', 0.8)
        top_p = config.get('top_p', 0.95)
        top_k = config.get('top_k', 64)
        repeat_penalty = config.get('lite_repeat_penalty', 1.1)

        collected = []
        char_count = 0
        last_print = 0

        for chunk in llm.create_chat_completion(
            messages=messages,
            stream=True,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
        ):
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if not content:
                continue
            collected.append(content)
            char_count += len(content)
            now = time.time()
            if now - last_print > 0.5:
                sys.stdout.write(f'\r[VADAR]: generating... {char_count} chars  ')
                sys.stdout.flush()
                last_print = now

        sys.stdout.write('\n')
        sys.stdout.flush()

        full = ''.join(collected)
        thoughts, clean = _parse_lite_thinking(full)
        return clean, None
    except Exception as e:
        return None, str(e)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "gui" and len(sys.argv) == 2:
            from voders.gui import launch
            launch()
            sys.exit(0)
        if sys.argv[1] == "cli" and len(sys.argv) == 2:
            from voders.interactiveCLI import interactive_cli_mode
            interactive_cli_mode()
            sys.exit(0)
        arg_offset = 1
        if sys.argv[1] == "cli":
            arg_offset = 2
        if len(sys.argv) > arg_offset:
            result = parse_and_execute_oneline(sys.argv[arg_offset:])
            sys.exit(0 if result else 1)
    print("VODER — Local voice processing toolkit")
    print("=" * 60)
    print()
    print("Usage:")
    print("  python voder.py                       Show this help message")
    print("  python voder.py gui                   Launch the GUI")
    print("  python voder.py cli                   Interactive CLI mode")
    print("  python voder.py <mode> [args...]      Run a one-line command")
    print()
    print("Available modes: tts, sts, ttm, stt, se, sfx, svs, ss, train, quest, chains")
    print()
    print("Run 'python voder.py <mode>' with no further args for mode-specific help,")
    print("or see docs/COMMAND_CATALOG.md for the full reference.")
    print()
    print("Documentation: https://github.com/HAKORADev/VODER")
