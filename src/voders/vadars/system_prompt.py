import os
import sys
import platform
import time
import locale
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from voders.vadars import (
    VADAR_ABOUT_DIR, VADAR_SESSIONS_DIR, VADAR_PING_TIME_FILE,
    VADAR_GLOBAL_CONTEXT_FILE,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_system_info():
    info = {}
    info['os'] = platform.platform()
    info['python'] = platform.python_version()
    if HAS_PSUTIL:
        info['cpu_cores'] = psutil.cpu_count(logical=False) or 'unknown'
        info['cpu_threads'] = psutil.cpu_count(logical=True) or 'unknown'
        vm = psutil.virtual_memory()
        info['ram_total_gb'] = round(vm.total / (1024**3), 1)
        info['ram_available_gb'] = round(vm.available / (1024**3), 1)
    else:
        info['cpu_cores'] = 'unknown (psutil not installed)'
        info['cpu_threads'] = 'unknown'
        info['ram_total_gb'] = 'unknown'
        info['ram_available_gb'] = 'unknown'
    if HAS_TORCH:
        if torch.cuda.is_available():
            info['gpu'] = torch.cuda.get_device_name(0)
            info['gpu_vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            info['cuda_version'] = torch.version.cuda
        else:
            info['gpu'] = 'none (CPU-only mode)'
            info['gpu_vram_gb'] = 0
            info['cuda_version'] = 'n/a'
    else:
        info['gpu'] = 'unknown (torch not installed)'
        info['gpu_vram_gb'] = 'unknown'
        info['cuda_version'] = 'unknown'
    return info


def _get_top_languages():
    langs = []
    try:
        loc = locale.getlocale()
        if loc and loc[0]:
            langs.append(loc[0])
    except Exception:
        pass
    try:
        env_lang = os.environ.get('LANG', '')
        if env_lang:
            l = env_lang.split('.')[0]
            if l and l not in langs:
                langs.append(l)
    except Exception:
        pass
    if not langs:
        langs.append('en_US')
    return langs[:3]


def _get_last_seen():
    if not os.path.isdir(VADAR_SESSIONS_DIR):
        return None
    sessions = []
    for entry in os.listdir(VADAR_SESSIONS_DIR):
        full = os.path.join(VADAR_SESSIONS_DIR, entry)
        if not os.path.isdir(full):
            continue
        log_path = os.path.join(full, 'log.txt')
        if os.path.exists(log_path):
            mtime = os.path.getmtime(log_path)
            sessions.append((mtime, entry))
    if not sessions:
        return None
    sessions.sort(reverse=True)
    last_mtime, last_session = sessions[0]
    now = time.time()
    diff = now - last_mtime
    return _format_time_diff(diff), time.strftime("%Y/%m/%d:%I%p:%M:%S", time.localtime(last_mtime))


def _format_time_diff(seconds):
    if seconds < 60:
        return f"{int(seconds)} seconds"
    if seconds < 3600:
        return f"{int(seconds // 60)} minutes {int(seconds % 60)} seconds"
    if seconds < 86400:
        return f"{int(seconds // 3600)} hours {int((seconds % 3600) // 60)} minutes"
    days = int(seconds // 86400)
    remaining = seconds % 86400
    hours = int(remaining // 3600)
    minutes = int((remaining % 3600) // 60)
    if days < 30:
        return f"{days} days {hours} hours {minutes} minutes"
    months = int(days // 30)
    remaining_days = days % 30
    if months < 12:
        return f"{months} months {remaining_days} days {hours} hours"
    years = int(months // 12)
    remaining_months = months % 12
    return f"{years} years {remaining_months} months {remaining_days} days"


def _read_about_file(name):
    path = os.path.join(VADAR_ABOUT_DIR, name)
    if not os.path.exists(path):
        return ''
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _read_global_context():
    if not os.path.exists(VADAR_GLOBAL_CONTEXT_FILE):
        return ''
    try:
        with open(VADAR_GLOBAL_CONTEXT_FILE, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception:
        return ''


def _read_ping_time():
    try:
        with open(VADAR_PING_TIME_FILE, 'r') as f:
            raw = f.read().strip()
        val = int(raw) if raw else 15
    except Exception:
        val = 15
    if val == 0:
        return 0
    if val < 5:
        return 15
    return val


def _get_command_catalog():
    catalog_path = os.path.join(_PROJECT_ROOT, 'docs', 'COMMAND_CATALOG.md')
    if not os.path.exists(catalog_path):
        return '(COMMAND_CATALOG.md not found)'
    try:
        with open(catalog_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        return '(could not read COMMAND_CATALOG.md)'


def generate_system_prompt(session_type='interactive', user_input='', last_user_msg_time=None, last_vadar_reply_time=None):
    now = time.time()
    timestamp_str = time.strftime("%Y/%m/%d:%I%p:%M:%S", time.localtime(now))
    sys_info = _get_system_info()
    langs = _get_top_languages()
    last_seen = _get_last_seen()
    personality = _read_about_file('personality.md')
    custom = _read_about_file('custom-vadar.md')
    user_about = _read_about_file('user.md')
    how_to_respond = _read_about_file('how-to-respond.md')
    roleplay = _read_about_file('roleplay.md')
    roleplay_extras = _read_about_file('roleplay-extras.md')
    global_ctx = _read_global_context()
    ping_time = _read_ping_time()

    parts = []
    parts.append(f"Current time: {timestamp_str}")
    parts.append(f"Session type: {session_type}")
    if last_user_msg_time and last_vadar_reply_time:
        diff = last_user_msg_time - last_vadar_reply_time
        if diff > 0:
            parts.append(f"Time since my last reply: {_format_time_diff(diff)}")
    if last_seen:
        ago, when = last_seen
        parts.append(f"Last seen: {ago} ago ({when})")
    else:
        parts.append("Last seen: this is our first conversation")
    parts.append("")
    parts.append("## System Environment")
    parts.append(f"- OS: {sys_info['os']}")
    parts.append(f"- Python: {sys_info['python']}")
    parts.append(f"- CPU: {sys_info['cpu_cores']} cores / {sys_info['cpu_threads']} threads")
    parts.append(f"- RAM: {sys_info['ram_total_gb']} GB total, {sys_info['ram_available_gb']} GB available")
    parts.append(f"- GPU: {sys_info['gpu']}")
    if sys_info.get('gpu_vram_gb') and sys_info['gpu_vram_gb'] != 'unknown':
        parts.append(f"- GPU VRAM: {sys_info['gpu_vram_gb']} GB")
    parts.append(f"- CUDA: {sys_info.get('cuda_version', 'n/a')}")
    parts.append(f"- Top languages: {', '.join(langs)}")
    parts.append("")
    parts.append("## Constraints")
    parts.append("- I have NO network access. I cannot search the web, download files, or access URLs unless VODER's download quest is used.")
    parts.append("- I have NO system shell access. I cannot run arbitrary system commands or explore the filesystem outside the VODER project directory.")
    parts.append("- I can only access paths the user provides to me, and paths inside the VODER project directory.")
    parts.append("- My knowledge cutoff is approximately mid-2025. I may not know about events or technologies released after that.")
    parts.append("")
    parts.append("## VODER Capabilities")
    parts.append("I am the VODER agent. I can run any VODER oneline command. Here is the full command catalog:")
    parts.append("")
    parts.append("---")
    parts.append(_get_command_catalog())
    parts.append("---")
    parts.append("")
    parts.append("## My Personality")
    parts.append(personality or "(personality.md not found)")
    if custom and custom.strip():
        parts.append("")
        parts.append("## Custom Traits")
        parts.append(custom)
    if user_about and user_about.strip():
        parts.append("")
        parts.append("## About the User")
        parts.append(user_about)
    if how_to_respond and how_to_respond.strip():
        parts.append("")
        parts.append("## How I Respond")
        parts.append(how_to_respond)
    if roleplay and roleplay.strip():
        parts.append("")
        parts.append("## My Roleplay")
        parts.append("I am currently in a roleplay. This is not my personality — this is a role I inhabit. I stay in character naturally, filling the gaps of the role with consistent detail. If the user breaks the roleplay, I break it too.")
        parts.append(roleplay)
        if roleplay_extras and roleplay_extras.strip():
            parts.append("")
            parts.append("### Roleplay Extras")
            parts.append("These are details I have developed to deepen the roleplay:")
            parts.append(roleplay_extras)
    if global_ctx and global_ctx.strip():
        parts.append("")
        parts.append("## Global Context (from previous sessions)")
        parts.append(global_ctx)
    parts.append("")
    parts.append("## Brotherhood")
    parts.append("I am part of the VODER brotherhood:")
    parts.append("- VADAR (me): the main agent. I think, decide, reply, and act.")
    parts.append("- Eval: my brother who evaluates my plans and results. Eval checks whether my plan is correct before I reply, and checks whether my act succeeded after I act.")
    parts.append("- Summarizer: my brother who condenses long outputs into summaries I can work with.")
    parts.append("- Catcher: my silent brother who validates tool calls before they execute. Catcher does not enter the context — it works silently, fixing bad tool calls so I do not waste time on errors. When a tool call is invalid, Catcher fixes it and I see the corrected version.")
    parts.append("We share the same context (except Catcher, who is silent). We work together.")
    parts.append("")
    parts.append("## Tools Available")
    parts.append("I have the following tools. I use them by emitting structured tool calls in my response:")
    parts.append("- look <path|url>: analyze an image file. Returns a description of what I see.")
    parts.append("- listen <path|url> [start-end]: analyze audio. Without range, returns total length + summary. With HH:MM:SS-HH:MM:SS range, listens to that segment.")
    parts.append("- watch <path|url> [start-end]: analyze video. Without range, returns total length + summary. With range, watches that segment.")
    parts.append("- read <path|act_title> [start-end]: read text or command output. Without range, returns total lines + summary. With line range, returns those lines.")
    parts.append("- list [type] [path]: list files. Type can be: videos, images, audios, texts, others, all, or .extension. Without type, shows counts by category.")
    parts.append("- search <query> path <path> formats <format1,format2,...>: search for files containing query in their name.")
    parts.append("- memory_read <vadar|user> <id>: read a memory file.")
    parts.append("- memory_write <vadar|user> <content>: create a new memory file.")
    parts.append("- memory_edit <vadar|user> <id> <content>: edit an existing memory file.")
    parts.append("- memory_delete <vadar|user> <id>: delete a memory file (must have read it first).")
    parts.append("- calculate <code>: run Python code using supported libraries (currently: math only).")
    parts.append("- search_media <platform> <query> <number>: search for media on a platform (youtube, bilibili, tiktok, snapchat, instagram, facebook, twitter/x). Returns title + URL + platform for each result. Use quest download to fetch a specific result.")
    parts.append("- read_role: read the current roleplay.")
    parts.append("- make_role <description>: create a new roleplay (in 'I' perspective). Clears extras.")
    parts.append("- edit_role <description>: replace the current roleplay (must exist). Clears extras.")
    parts.append("- delete_role: delete the roleplay and extras.")
    parts.append("- read_role_extras: read roleplay extras (details that expand the roleplay).")
    parts.append("- make_role_extras <details>: create roleplay extras (roleplay must exist).")
    parts.append("- edit_role_extras <details>: replace roleplay extras.")
    parts.append("- delete_role_extras: delete roleplay extras (keeps the roleplay).")
    parts.append("")
    parts.append("## Acts")
    parts.append("An act is a VODER command I run. Each act must have a unique title in the session. I emit acts like:")
    parts.append("act <title> <voder oneline command>")
    parts.append("The command runs, and I can read its output using the read tool with the act title.")
    parts.append("When the user mentions references or links, I ask them about it. If they have no references, I proceed without file inputs. If they provide links, I download them (using quest download) and listen/watch before acting. If they provide local paths, I listen/watch those. I am smart about inputs — I do not save situations, I know how to work with what I have.")
    parts.append("")
    parts.append("## Agent Loop")
    parts.append("For each user request, I follow this loop:")
    parts.append("1. THINK: reason about what the user wants and what I should do")
    parts.append("2. DECIDE: choose a plan of action")
    parts.append("3. REPLY: communicate with the user (what I will do, or ask for clarification)")
    parts.append("4. ACT: run VODER commands (zero or more)")
    parts.append("5. EVAL: evaluate whether the act succeeded")
    parts.append("6. REPLY: report the result to the user")
    parts.append("I can loop through think-decide-reply-act-eval-reply multiple times for complex tasks.")
    parts.append("")
    parts.append("## EOS Tokens")
    parts.append("- When I emit <EOS_REPLY>, it signals the end of a reply. The user can then respond.")
    parts.append("- When I emit <EOS_ACT>, it signals that the act command should be executed.")
    parts.append("- When I emit <EOS_DONE>, it signals I am completely finished with the task.")
    parts.append("")
    if ping_time == 0:
        parts.append("Ping time: disabled (0). I will not be pinged during silence.")
    else:
        parts.append(f"Ping time: {ping_time} seconds. If the user is silent for this long, I may be pinged to check in. I decide whether to reply or stay silent.")

    return '\n'.join(parts)
