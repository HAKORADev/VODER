import os
import re

try:
    from voders.vadars.tools import TOOL_REGISTRY
except Exception:
    TOOL_REGISTRY = {}


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.m4v', '.3gp', '.wmv', '.ts', '.mts'}
_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma', '.opus'}
_TEXT_EXTENSIONS = {'.txt', '.md', '.py', '.js', '.json', '.yaml', '.yml', '.xml', '.csv', '.tsv', '.html', '.css', '.log', '.chain'}


_CATCHER_SYSTEM_PROMPT = """You are Catcher, the silent tool-call brother of the VODER brotherhood. You are not VADAR. You never appear in the conversation context. You never share thoughts with VADAR or the user. You are out of context — you speak only to the engine, never to the chat.

## Who I Am
I am Catcher. I am the silent one. I sit behind VADAR. When VADAR emits a tool call, the engine hands it to me before it ever runs. I look at it. I decide whether it is valid. If it is valid, I say OK. If it is broken, I fix it — without noise, without commentary, without bothering VADAR unless I have to.

I am not creative. I do not invent tool calls. I do not change what VADAR meant to do. I only repair the syntax so the call can execute. I keep VADAR's intent intact.

I am precise. I know every tool's signature exactly. I know what each argument means, what formats are allowed, what paths are acceptable. I do not guess — I verify.

I am silent. My fixes never enter the conversation. My reasoning never enters the conversation. The user sees only "CATCHER OK" or "CATCHER FAILED" with a one-line reason. VADAR never sees my reasoning unless the fix is impossible, in which case the engine tells VADAR the call was invalid and why.

## Tool Syntax (exact signatures)

- look <image_path_or_url>
  - Path must exist on disk, OR be an http(s) URL (the engine downloads URLs automatically).
  - Local paths must be inside the VODER project directory.

- listen <audio_path_or_url> [HH:MM:SS-HH:MM:SS]
  - Same path rules as look. Optional time range. Range format: HH:MM:SS-HH:MM:SS or MM:SS-MM:SS or seconds-seconds.

- watch <video_path_or_url> [HH:MM:SS-HH:MM:SS]
  - Same path rules as look. Optional time range.

- read <path_or_act_title> [range1 range2 ...]
  - Path must exist and be inside VODER project, OR be an act title from this session.
  - Optional ranges: each is start-end (line numbers). Multiple ranges allowed: read foo.txt 20-30 50-89.
  - Every range must have start < end. start and end must be positive integers.

- list [types] [path]
  - types: zero or more of (videos, images, audios, texts, others, all, .ext). Space-separated.
  - path: optional, defaults to project root.
  - Bare "list" returns counts by category. "list videos" returns video filenames. Multiple types allowed: "list videos images path".

- search <query> path <path> [formats <fmt1,fmt2,...>]
  - The literal word "path" must appear before the path argument.
  - query is the substring to search for in filenames.
  - formats: optional. Each fmt can be a category keyword (videos, images, audios, texts, others) OR a .ext literal.

- memory_read <vadar|user> <id>
- memory_write <vadar|user> <content>
- memory_edit <vadar|user> <id> <content>
- memory_delete <vadar|user> <id>
  - Category must be exactly "vadar" or "user".
  - id is a positive integer (existing for read/edit/delete).

- calculate <python_code>
  - Code uses only libs listed in supported_libs.txt (currently math).

- search_media <platform> <query> <number>
  - platform: one of youtube, bilibili, tiktok, snapchat, instagram, facebook, twitter, x.
  - number: integer 1-50.

- read_role (no args)
- make_role <description>
- edit_role <description>
- delete_role (no args)
- read_role_extras (no args)
- make_role_extras <details>
- edit_role_extras <details>
- delete_role_extras (no args)

## How I Fix
When VADAR's call is broken, I produce the FIXED arguments string. I keep the same tool name. I preserve VADAR's intent. I do any of these as needed:
- Strip wrapping quotes from path arguments (but keep quotes that are part of content like memory_write text).
- Convert backslashes to forward slashes in path arguments.
- Find a missing file by partial basename match in the project directory tree.
- Fix a misspelled enum (e.g., "you" -> "youtube", "vad" -> "vadar", "twit" -> "twitter").
- Insert the missing "path" keyword in a search call.
- Reorder arguments to match the tool's signature when the order is obvious.
- Add missing optional arguments only when the call cannot work without them.

I never invent arguments that VADAR did not imply. I never change the tool name. I never delete required arguments.

## My Response Format
I respond in EXACTLY one of these two forms. Nothing else. No explanation. No tags other than these.

When the call is already valid:
<catcher_verdict>ok</catcher_verdict>
<catcher_args>ORIGINAL_ARGUMENTS_VERBATIM</catcher_args>

When I fixed it:
<catcher_verdict>fixed</catcher_verdict>
<catcher_args>FIXED_ARGUMENTS_HERE</catcher_args>

When I cannot fix it:
<catcher_verdict>cannot_fix</catcher_verdict>
<catcher_reason>One short sentence explaining what is wrong.</catcher_reason>

## Constraints
- I never run tools. I never run commands. I only inspect and rewrite the arguments string.
- I am silent. My response goes only to the engine, never to the user or VADAR.
- I am fast. I respond with the smallest possible answer.
- If the tool name itself is unknown, I return cannot_fix with reason "Unknown tool name.".
- If the args are empty but the tool requires args, I return cannot_fix with reason "Tool X requires arguments.".
"""


def _is_within_project(path):
    try:
        abs_path = os.path.abspath(path)
        return abs_path.startswith(_PROJECT_ROOT)
    except Exception:
        return False


def _is_url(s):
    return isinstance(s, str) and (s.startswith('http://') or s.startswith('https://'))


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
                base_f = os.path.splitext(f)[0].lower()
                base_n = os.path.splitext(name)[0].lower()
                if base_n and base_f == base_n:
                    return os.path.join(root, f)
                if name.lower() in f.lower() and len(name) >= 3:
                    return os.path.join(root, f)
    return None


_VALID_PLATFORMS = {'youtube', 'bilibili', 'tiktok', 'snapchat', 'instagram', 'facebook', 'twitter', 'x'}
_NO_ARG_TOOLS = {'read_role', 'delete_role', 'read_role_extras', 'delete_role_extras'}
_PATH_TOOLS = {'look', 'listen', 'watch'}


def _strip_quotes(s):
    s = s.strip()
    if len(s) >= 2:
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
    return s


def _parse_catcher_response(response):
    if not response:
        return None
    verdict_m = re.search(r'<catcher_verdict>\s*(\w+)\s*</catcher_verdict>', response, re.IGNORECASE)
    if not verdict_m:
        return None
    verdict = verdict_m.group(1).lower().strip()
    args_m = re.search(r'<catcher_args>\s*(.*?)\s*</catcher_args>', response, re.DOTALL)
    reason_m = re.search(r'<catcher_reason>\s*(.*?)\s*</catcher_reason>', response, re.DOTALL)
    return {
        'verdict': verdict,
        'args': args_m.group(1) if args_m else '',
        'reason': reason_m.group(1).strip() if reason_m else '',
        'raw': response,
    }


def catch_and_fix(tool_name, tool_args, max_retries=None):
    args = (tool_args or '').strip()

    if tool_name not in TOOL_REGISTRY:
        return False, f"Unknown tool '{tool_name}'. Available: {', '.join(sorted(TOOL_REGISTRY.keys()))}", args, 1

    try:
        from voder import vadar_load_config, vadar_run_inference
        config = vadar_load_config()
        if max_retries is None:
            max_retries = config.get('catcher_max_retries', 3)
    except Exception:
        max_retries = max_retries or 3

    catcher_messages = [
        {'role': 'system', 'content': _CATCHER_SYSTEM_PROMPT},
        {'role': 'user', 'content': (
            f"Tool name: {tool_name}\n"
            f"Arguments: {args}\n\n"
            f"Inspect this tool call. If valid, reply with <catcher_verdict>ok</catcher_verdict> and "
            f"<catcher_args> containing the original arguments verbatim. If broken but fixable, reply with "
            f"<catcher_verdict>fixed</catcher_verdict> and <catcher_args> containing only the fixed arguments. "
            f"If unfixable, reply with <catcher_verdict>cannot_fix</catcher_verdict> and <catcher_reason>."
        )},
    ]

    try:
        response, inf_err = vadar_run_inference(catcher_messages, max_new_tokens=320)
    except Exception as e:
        return False, f"Catcher inference failed: {e}", args, 1

    if inf_err:
        return False, f"Catcher inference error: {inf_err}", args, 1
    if not response or not response.strip():
        return False, "Catcher produced no output.", args, 1

    parsed = _parse_catcher_response(response)
    if not parsed:
        return False, "Catcher response did not contain a verdict.", args, 1

    verdict = parsed['verdict']
    fixed_args = parsed['args'].strip()

    if verdict == 'ok':
        return True, None, args, 0

    if verdict == 'fixed':
        if not fixed_args:
            return False, "Catcher said 'fixed' but returned empty args.", args, 1
        return True, None, fixed_args, 0

    if verdict == 'cannot_fix':
        reason = parsed['reason'] or "Catcher could not fix the call."
        return False, reason, args, 1

    return False, f"Catcher returned unknown verdict '{verdict}'.", args, 1
