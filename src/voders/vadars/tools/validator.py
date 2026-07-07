import os
import re

from voders.vadars.tools import TOOL_REGISTRY

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}

_TOOL_SYNTAX = {
    'look': {'min_args': 1, 'first_arg_is_path': True, 'optional_second': 'time_range'},
    'listen': {'min_args': 1, 'first_arg_is_path': True, 'optional_second': 'time_range'},
    'watch': {'min_args': 1, 'first_arg_is_path': True, 'optional_second': 'time_range'},
    'read': {'min_args': 1, 'first_arg_is_path_or_title': True, 'optional_second': 'line_range'},
    'list': {'min_args': 0, 'optional_types': True, 'optional_path': True},
    'search': {'min_args': 2, 'requires_keyword': 'path'},
    'memory_read': {'min_args': 2, 'first_arg_enum': ('vadar', 'user')},
    'memory_write': {'min_args': 2, 'first_arg_enum': ('vadar', 'user')},
    'memory_edit': {'min_args': 3, 'first_arg_enum': ('vadar', 'user')},
    'memory_delete': {'min_args': 2, 'first_arg_enum': ('vadar', 'user')},
    'calculate': {'min_args': 1},
    'search_media': {'min_args': 3, 'first_arg_enum': ('youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x')},
    'read_role': {'min_args': 0},
    'make_role': {'min_args': 1},
    'edit_role': {'min_args': 1},
    'delete_role': {'min_args': 0},
    'read_role_extras': {'min_args': 0},
    'make_role_extras': {'min_args': 1},
    'edit_role_extras': {'min_args': 1},
    'delete_role_extras': {'min_args': 0},
}


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


def _fix_tool_call(tool_name, tool_args):
    syntax = _TOOL_SYNTAX.get(tool_name, {})
    fixed = tool_args.strip()
    changes = []

    tokens = fixed.split()
    if not tokens and syntax.get('min_args', 0) > 0:
        return fixed, changes

    if syntax.get('first_arg_is_path') or syntax.get('first_arg_is_path_or_title'):
        if tokens:
            first = _strip_quotes(tokens[0])
            if first != tokens[0]:
                tokens[0] = first
                fixed = ' '.join(tokens)
                changes.append('stripped quotes from first arg')

            if '\\' in first:
                first_fixed = first.replace('\\\\', '/').replace('\\', '/')
                if first_fixed != first:
                    tokens[0] = first_fixed
                    fixed = ' '.join(tokens)
                    changes.append('fixed backslashes to forward slashes')

            if syntax.get('first_arg_is_path') and not first.startswith('http'):
                if not os.path.exists(first):
                    found = _find_file_fuzzy(os.path.basename(first))
                    if found:
                        tokens[0] = found
                        fixed = ' '.join(tokens)
                        changes.append(f'resolved path: {first} -> {found}')

    if syntax.get('first_arg_enum'):
        valid = syntax['first_arg_enum']
        if tokens and tokens[0].lower() not in valid:
            for v in valid:
                if v.startswith(tokens[0].lower()) or tokens[0].lower().startswith(v):
                    tokens[0] = v
                    fixed = ' '.join(tokens)
                    changes.append(f'fixed enum: -> {v}')
                    break

    if syntax.get('requires_keyword'):
        kw = syntax['requires_keyword']
        if kw not in fixed.lower():
            parts = fixed.split(None, 1)
            if len(parts) >= 2:
                query = _strip_quotes(parts[0])
                rest = parts[1].strip()
                if not rest.lower().startswith(kw):
                    fixed = f'{query} {kw} {rest}'
                    changes.append(f'added missing "{kw}" keyword')

    return fixed, changes


def validate_tool_call(tool_name, tool_args):
    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(TOOL_REGISTRY.keys()))}", tool_args

    syntax = _TOOL_SYNTAX.get(tool_name, {})
    args = tool_args.strip()

    if syntax.get('min_args', 0) > 0:
        tokens = args.split()
        if len(tokens) < syntax['min_args']:
            return False, f"Tool '{tool_name}' needs at least {syntax['min_args']} argument(s), got {len(tokens)}.", args

    if not args and syntax.get('min_args', 0) == 0:
        return True, None, args

    if not args and tool_name not in ('read_role', 'delete_role', 'read_role_extras', 'delete_role_extras'):
        return False, f"Tool '{tool_name}' requires arguments.", args

    if syntax.get('first_arg_is_path') or syntax.get('first_arg_is_path_or_title'):
        first = _strip_quotes(args.split()[0]) if args.split() else ''
        if not first:
            return False, f"Tool '{tool_name}' needs a file path or act title.", args
        if syntax.get('first_arg_is_path'):
            if not first.startswith('http'):
                if not os.path.exists(first):
                    return False, f"File not found: {first}", args
                if not _is_within_project(first):
                    return False, f"Path '{first}' is outside the VODER project directory.", args
        return True, None, args

    if syntax.get('first_arg_enum'):
        valid = syntax['first_arg_enum']
        parts = args.split(None, 1)
        if parts and parts[0].lower() not in valid:
            return False, f"First argument must be one of: {', '.join(valid)}, got '{parts[0]}'.", args
        return True, None, args

    if syntax.get('requires_keyword'):
        kw = syntax['requires_keyword']
        if kw not in args.lower():
            return False, f"Tool '{tool_name}' needs '{kw}' keyword in the arguments.", args
        return True, None, args

    if tool_name == 'search_media':
        parts = args.split(None, 2)
        if len(parts) >= 3:
            try:
                n = int(parts[2])
                if n < 1 or n > 50:
                    return False, f"Number must be 1-50, got {n}.", args
            except ValueError:
                return False, f"Number must be an integer, got '{parts[2]}'.", args

    return True, None, args


def catch_and_fix(tool_name, tool_args):
    ok, err, fixed_args = validate_tool_call(tool_name, tool_args)
    if ok:
        return True, None, fixed_args, 0

    tried_fix, changes = _fix_tool_call(tool_name, tool_args)
    if changes:
        ok2, err2, fixed_args2 = validate_tool_call(tool_name, tried_fix)
        if ok2:
            return True, None, fixed_args2, 0
        fixed_args = tried_fix
        err = err2

    return False, err, fixed_args, 1
