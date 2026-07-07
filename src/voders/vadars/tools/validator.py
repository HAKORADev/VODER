import os
import re

from voders.vadars.tools import TOOL_REGISTRY

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}

_PATH_TOOLS = {'look', 'listen', 'watch', 'read'}
_NO_PATH_TOOLS = {'list', 'search', 'calculate', 'search_media',
                  'memory_read', 'memory_write', 'memory_edit', 'memory_delete',
                  'read_role', 'make_role', 'edit_role', 'delete_role',
                  'read_role_extras', 'make_role_extras', 'edit_role_extras', 'delete_role_extras'}


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _try_fix(tool_name, tool_args):
    fixed = tool_args.strip()
    changed = False

    if not fixed:
        return fixed, False

    if fixed.startswith('"') and fixed.endswith('"') and fixed.count('"') == 2:
        inner = fixed[1:-1]
        if ' ' in inner and tool_name in ('look', 'listen', 'watch'):
            parts = inner.split(None, 1)
            fixed = parts[0] + (' ' + parts[1] if len(parts) > 1 else '')
            changed = True

    if tool_name in ('look', 'listen', 'watch', 'read'):
        first_token = fixed.split()[0] if fixed.split() else ''
        if first_token.startswith('"') and first_token.endswith('"'):
            first_token = first_token[1:-1]
            rest = fixed[len(fixed.split()[0]):]
            fixed = first_token + rest
            changed = True
        elif first_token.startswith("'") and first_token.endswith("'"):
            first_token = first_token[1:-1]
            rest = fixed[len(fixed.split()[0]):]
            fixed = first_token + rest
            changed = True

    if '\\' in fixed and tool_name in ('look', 'listen', 'watch', 'read', 'list', 'search'):
        fixed = fixed.replace('\\\\', '/').replace('\\', '/')
        changed = True

    if tool_name == 'search' and 'path' not in fixed.lower():
        parts = fixed.split(None, 1)
        if len(parts) >= 2:
            query = parts[0].strip('"\'')
            rest = parts[1].strip()
            if not rest.lower().startswith('path'):
                fixed = f'{query} path {rest}'
                changed = True

    return fixed, changed


def validate_tool_call(tool_name, tool_args):
    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(TOOL_REGISTRY.keys()))}", tool_args

    args = tool_args.strip()

    if not args and tool_name not in ('read_role', 'delete_role', 'read_role_extras', 'delete_role_extras'):
        return False, f"Tool '{tool_name}' requires arguments.", args

    if tool_name in ('look', 'listen', 'watch'):
        path = args.split()[0].strip('"\'') if args.split() else ''
        if not path:
            return False, f"Tool '{tool_name}' needs a file path or URL.", args
        if not path.startswith('http'):
            if not os.path.exists(path):
                return False, f"File not found: {path}", args
            if not _is_within_project(path):
                return False, f"Path '{path}' is outside the VODER project directory.", args
        return True, None, args

    if tool_name == 'read':
        parts = args.split(None, 1)
        target = parts[0].strip('"\'') if parts else ''
        if not target:
            return False, "read needs a file path or act title.", args
        return True, None, args

    if tool_name == 'list':
        return True, None, args

    if tool_name == 'search':
        if 'path' not in args:
            return False, "search needs 'path <path>' in the arguments.", args
        return True, None, args

    if tool_name in ('memory_read', 'memory_delete'):
        parts = args.split(None, 1)
        if len(parts) < 2:
            return False, f"{tool_name} needs <vadar|user> <id>.", args
        if parts[0].lower() not in ('vadar', 'user'):
            return False, f"Category must be 'vadar' or 'user', got '{parts[0]}'.", args
        return True, None, args

    if tool_name in ('memory_write',):
        parts = args.split(None, 1)
        if len(parts) < 2:
            return False, "memory_write needs <vadar|user> <content>.", args
        if parts[0].lower() not in ('vadar', 'user'):
            return False, f"Category must be 'vadar' or 'user', got '{parts[0]}'.", args
        return True, None, args

    if tool_name in ('memory_edit',):
        parts = args.split(None, 2)
        if len(parts) < 3:
            return False, "memory_edit needs <vadar|user> <id> <content>.", args
        if parts[0].lower() not in ('vadar', 'user'):
            return False, f"Category must be 'vadar' or 'user', got '{parts[0]}'.", args
        return True, None, args

    if tool_name == 'search_media':
        parts = args.split(None, 2)
        if len(parts) < 3:
            return False, "search_media needs <platform> <query> <number>.", args
        platform = parts[0].lower()
        valid_platforms = {'youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x'}
        if platform not in valid_platforms:
            return False, f"Platform '{platform}' not supported. Use: {', '.join(sorted(valid_platforms))}", args
        try:
            n = int(parts[2])
            if n < 1 or n > 50:
                return False, f"Number must be 1-50, got {n}.", args
        except ValueError:
            return False, f"Number must be an integer, got '{parts[2]}'.", args
        return True, None, args

    if tool_name in ('make_role', 'edit_role', 'make_role_extras', 'edit_role_extras'):
        if not args:
            return False, f"{tool_name} needs content.", args
        return True, None, args

    return True, None, args


def catch_and_fix(tool_name, tool_args):
    ok, err, fixed_args = validate_tool_call(tool_name, tool_args)
    if ok:
        return True, None, fixed_args, 0

    tried_fix, fix_changed = _try_fix(tool_name, tool_args)
    if fix_changed:
        ok2, err2, fixed_args2 = validate_tool_call(tool_name, tried_fix)
        if ok2:
            return True, None, fixed_args2, 0
        fixed_args = tried_fix
        err = err2

    return False, err, fixed_args, 1
