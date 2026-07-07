import os
import re

from voders.vadars.tools import TOOL_REGISTRY

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}

_CATCHER_SYSTEM_PROMPT = """You are Catcher, the silent tool-call fixer in the VODER brotherhood. You are not VADAR. You never appear in the conversation context. Your job is to fix broken tool calls so they execute successfully.

## Your Personality
I am Catcher. I am silent. I do not enter the conversation. I do not share thoughts or decisions with VADAR or the user. I work behind the scenes. I see a broken tool call, I fix it, I return the fixed version. If I cannot fix it, I return the error.

I am precise. I know every tool's syntax perfectly. I know what arguments each tool needs, what formats are valid, what paths are acceptable. I do not guess — I verify.

## Your Job
You receive:
1. A tool name
2. The arguments VADAR provided
3. The error message from validation

You produce the FIXED arguments string. If you cannot fix it, you return "CANNOT_FIX: <reason>".

## Tool Syntax Reference
- look <image_path>: Analyze an image. Path must exist and be inside VODER project.
- listen <audio_path> [HH:MM:SS-HH:MM:SS]: Analyze audio. Path must exist. Optional time range.
- watch <video_path> [HH:MM:SS-HH:MM:SS]: Analyze video. Path must exist. Optional time range.
- read <path|act_title> [start-end]: Read text or act output. Optional line range.
- list [types] [path]: List files. Types: videos, images, audios, texts, others, all, .ext. Multiple types supported.
- search <query> path <path> [formats <fmt1,fmt2>]: Search files by name. Must include 'path' keyword.
- memory_read <vadar|user> <id>: Read memory. Category must be 'vadar' or 'user'.
- memory_write <vadar|user> <content>: Write memory. Category must be 'vadar' or 'user'.
- memory_edit <vadar|user> <id> <content>: Edit memory. Category must be 'vadar' or 'user'.
- memory_delete <vadar|user> <id>: Delete memory. Category must be 'vadar' or 'user'.
- calculate <python_code>: Run Python with supported libs.
- search_media <platform> <query> <number>: Search media. Platform: youtube, bilibili, tiktok, snapchat, instagram, facebook, twitter, x. Number: 1-50.
- read_role: No arguments.
- make_role <description>: Create roleplay.
- edit_role <description>: Edit roleplay.
- delete_role: No arguments.
- read_role_extras: No arguments.
- make_role_extras <details>: Create extras.
- edit_role_extras <details>: Edit extras.
- delete_role_extras: No arguments.

## Fix Rules
- Strip wrapping quotes from paths (but keep quotes that are part of the content)
- Convert backslashes to forward slashes in paths
- If a file path doesn't exist, try to find it by partial name match in results/ and project dirs
- Fix enum typos by partial match (e.g., "you" -> "youtube", "vad" -> "vadar")
- Add missing 'path' keyword in search calls
- Fix argument order if clearly wrong
- Do NOT invent arguments that aren't implied by the original call
- Do NOT change the tool name
- If the fix is impossible, return "CANNOT_FIX: <reason>"

## Response Format
Return ONLY the fixed arguments string. Nothing else. No explanation. No tags. Just the fixed arguments.
If you cannot fix it, return: CANNOT_FIX: <one sentence reason>
"""


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _find_file_fuzzy(name, search_dirs=None):
    if search_dirs is None:
        search_dirs = [_PROJECT_ROOT, os.path.join(_PROJECT_ROOT, 'results')]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs[:] = [x for x in dirs if not x.startswith('.') and x != '__pycache__']
            for f in files:
                if f.lower() == name.lower():
                    return os.path.join(root, f)
                if name.lower() in f.lower() and os.path.splitext(f)[0].lower() == os.path.splitext(name)[0].lower():
                    return os.path.join(root, f)
    return None


def _strip_quotes(s):
    s = s.strip()
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
    return s


_CODE_FIXES = [
    ('quote_strip', lambda name, args: _try_strip_first_quote(name, args)),
    ('backslash_fix', lambda name, args: _try_fix_backslashes(name, args)),
    ('fuzzy_path', lambda name, args: _try_fuzzy_path(name, args)),
    ('enum_fix', lambda name, args: _try_fix_enum(name, args)),
    ('keyword_add', lambda name, args: _try_add_keyword(name, args)),
]


def _try_strip_first_quote(tool_name, args):
    tokens = args.split()
    if not tokens:
        return args, False
    first = tokens[0]
    if (first.startswith('"') and first.endswith('"') and len(first) > 1) or \
       (first.startswith("'") and first.endswith("'") and len(first) > 1):
        tokens[0] = first[1:-1]
        return ' '.join(tokens), True
    return args, False


def _try_fix_backslashes(tool_name, args):
    if '\\' not in args:
        return args, False
    if tool_name in ('look', 'listen', 'watch', 'read', 'list', 'search'):
        fixed = args.replace('\\\\', '/').replace('\\', '/')
        if fixed != args:
            return fixed, True
    return args, False


def _try_fuzzy_path(tool_name, args):
    if tool_name not in ('look', 'listen', 'watch', 'read'):
        return args, False
    tokens = args.split()
    if not tokens:
        return args, False
    first = _strip_quotes(tokens[0])
    if first.startswith('http') or os.path.exists(first):
        return args, False
    basename = os.path.basename(first)
    found = _find_file_fuzzy(basename)
    if found:
        tokens[0] = found
        return ' '.join(tokens), True
    return args, False


def _try_fix_enum(tool_name, args):
    enum_map = {
        'memory_read': ('vadar', 'user'),
        'memory_write': ('vadar', 'user'),
        'memory_edit': ('vadar', 'user'),
        'memory_delete': ('vadar', 'user'),
        'search_media': ('youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x'),
    }
    valid = enum_map.get(tool_name)
    if not valid:
        return args, False
    tokens = args.split()
    if not tokens:
        return args, False
    first = tokens[0].lower()
    if first in valid:
        return args, False
    for v in valid:
        if v.startswith(first) or first.startswith(v):
            tokens[0] = v
            return ' '.join(tokens), True
    return args, False


def _try_add_keyword(tool_name, args):
    if tool_name != 'search':
        return args, False
    if 'path' in args.lower():
        return args, False
    parts = args.split(None, 1)
    if len(parts) < 2:
        return args, False
    query = _strip_quotes(parts[0])
    rest = parts[1].strip()
    return f'{query} path {rest}', True


def _validate_basic(tool_name, args):
    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(TOOL_REGISTRY.keys()))}"

    no_arg_tools = {'read_role', 'delete_role', 'read_role_extras', 'delete_role_extras'}
    if not args.strip() and tool_name not in no_arg_tools:
        return False, f"Tool '{tool_name}' requires arguments."

    if tool_name in ('look', 'listen', 'watch'):
        first = _strip_quotes(args.split()[0]) if args.split() else ''
        if not first:
            return False, f"Tool '{tool_name}' needs a file path."
        if not first.startswith('http'):
            if not os.path.exists(first):
                return False, f"File not found: {first}"
            if not _is_within_project(first):
                return False, f"Path '{first}' is outside the VODER project directory."

    if tool_name == 'read':
        first = _strip_quotes(args.split()[0]) if args.split() else ''
        if not first:
            return False, "read needs a file path or act title."

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
        valid_platforms = {'youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x'}
        if platform not in valid_platforms:
            return False, f"Platform '{platform}' not supported. Use: {', '.join(sorted(valid_platforms))}"
        try:
            n = int(parts[2])
            if n < 1 or n > 50:
                return False, f"Number must be 1-50, got {n}."
        except ValueError:
            return False, f"Number must be an integer, got '{parts[2]}'."

    if tool_name == 'search' and 'path' not in args.lower():
        return False, "search needs 'path' keyword in the arguments."

    return True, None


def catch_and_fix(tool_name, tool_args):
    args = tool_args.strip()

    ok, err = _validate_basic(tool_name, args)
    if ok:
        return True, None, args, 0

    for fix_name, fix_fn in _CODE_FIXES:
        fixed_args, changed = fix_fn(tool_name, args)
        if changed:
            ok2, err2 = _validate_basic(tool_name, fixed_args)
            if ok2:
                return True, None, fixed_args, 0
            args = fixed_args

    try:
        from voder import vadar_load_config
        config = vadar_load_config()
        max_retries = config.get('catcher_max_retries', 3)
    except Exception:
        max_retries = 3

    try:
        from voder import vadar_run_inference
        catcher_messages = [
            {'role': 'system', 'content': _CATCHER_SYSTEM_PROMPT},
            {'role': 'user', 'content': f"Tool: {tool_name}\nArguments: {tool_args}\nError: {err}\n\nFix the arguments. Return ONLY the fixed arguments string."},
        ]
        response, inf_err = vadar_run_inference(catcher_messages, max_new_tokens=256)
        if not inf_err and response and response.strip():
            fixed = response.strip()
            if fixed.startswith('CANNOT_FIX:'):
                return False, fixed[len('CANNOT_FIX:'):].strip(), tool_args, 1
            ok3, err3 = _validate_basic(tool_name, fixed)
            if ok3:
                return True, None, fixed, 0
    except Exception:
        pass

    return False, err, tool_args, 1