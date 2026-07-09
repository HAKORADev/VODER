import os
import re

try:
    from voders.vadars.tools import TOOL_REGISTRY
except Exception:
    TOOL_REGISTRY = {}


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}

_NO_ARG_TOOLS = {'read_role', 'delete_role', 'read_role_extras', 'delete_role_extras', 'read_catalog_general'}
_VALID_PLATFORMS = {'youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x', 'reddit'}
_VALID_CATALOG_MODES = {'tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train', 'quest', 'chains', 'prebuilt_chains', 'general'}


def _is_url(s):
    return isinstance(s, str) and (s.startswith('http://') or s.startswith('https://'))


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _strip_quotes(s):
    s = s.strip()
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
    return s


def validate_tool_basic(tool_name, tool_args, allowed_paths=None):
    args = (tool_args or '').strip()
    allowed_paths = allowed_paths or set()

    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(TOOL_REGISTRY.keys()))}"

    if tool_name in _NO_ARG_TOOLS:
        if args:
            return False, f"Tool '{tool_name}' takes no arguments."
        return True, None

    if not args:
        return False, f"Tool '{tool_name}' requires arguments."

    if tool_name in ('look', 'listen', 'watch'):
        tokens = args.split(None, 1)
        first = _strip_quotes(tokens[0])
        if not first:
            return False, f"Tool '{tool_name}' needs a file path or URL."
        if not _is_url(first):
            if not os.path.exists(first):
                return False, f"File not found: {first}"
            if not _is_within_project(first) and first not in allowed_paths:
                return False, f"Path '{first}' is outside the VODER project directory. Only paths inside the project or paths the user explicitly provided are allowed."
        if tool_name == 'look':
            ext = os.path.splitext(first)[1].lower()
            if ext and ext not in _IMAGE_EXTENSIONS and not _is_url(first):
                return False, f"'{first}' does not appear to be an image file."
        if tool_name == 'watch':
            ext = os.path.splitext(first)[1].lower()
            if ext and ext not in _VIDEO_EXTENSIONS and not _is_url(first):
                return False, f"'{first}' does not appear to be a video file."

    if tool_name == 'read':
        tokens = args.split(None, 1)
        first = _strip_quotes(tokens[0])
        if not first:
            return False, "read needs a file path or act title."
        if os.path.isabs(first) or '/' in first or '\\' in first:
            if not _is_url(first) and not _is_within_project(first) and first not in allowed_paths:
                if os.path.exists(first):
                    return False, f"Path '{first}' is outside the VODER project directory. Only paths inside the project or paths the user explicitly provided are allowed."

    if tool_name in ('memory_read', 'memory_write', 'memory_edit', 'memory_delete'):
        parts = args.split(None, 1)
        if len(parts) < 2:
            return False, f"{tool_name} needs <vadar|user> and more arguments."
        if parts[0].lower() not in ('vadar', 'user'):
            return False, f"Category must be 'vadar' or 'user', got '{parts[0]}'."

    if tool_name == 'search_media':
        parts = args.split(None, 2)
        if len(parts) < 3:
            return False, "search_media needs <platform> <query> <number>."
        platform = parts[0].lower()
        if platform not in _VALID_PLATFORMS:
            return False, f"Platform '{platform}' not supported. Use: {', '.join(sorted(_VALID_PLATFORMS))}"
        try:
            n = int(parts[2])
            if n < 1 or n > 50:
                return False, f"Number must be 1-50, got {n}."
        except ValueError:
            return False, f"Number must be an integer, got '{parts[2]}'."

    if tool_name == 'search':
        if 'path' not in args.lower():
            return False, "search needs 'path' keyword in the arguments."

    if tool_name == 'read_catalog_mode':
        mode = args.strip().lower()
        if not mode:
            return False, f"read_catalog_mode needs a mode. Available: {', '.join(sorted(_VALID_CATALOG_MODES))}"
        if mode not in _VALID_CATALOG_MODES:
            return False, f"Unknown mode '{mode}'. Available: {', '.join(sorted(_VALID_CATALOG_MODES))}"

    return True, None
