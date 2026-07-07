import os

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

VADAR_DIR = _PKG_DIR

VADAR_SESSIONS_DIR = os.path.join(VADAR_DIR, 'sessions')
VADAR_MEMORIES_DIR = os.path.join(VADAR_DIR, 'memories')
VADAR_MEMORIES_VADAR_DIR = os.path.join(VADAR_MEMORIES_DIR, 'vadar')
VADAR_MEMORIES_USER_DIR = os.path.join(VADAR_MEMORIES_DIR, 'user')
VADAR_ABOUT_DIR = os.path.join(VADAR_DIR, 'about')
VADAR_PING_TIME_FILE = os.path.join(VADAR_DIR, 'ping-time.txt')
VADAR_SUPPORTED_LIBS_FILE = os.path.join(VADAR_DIR, 'supported_libs.txt')
VADAR_GLOBAL_CONTEXT_FILE = os.path.join(VADAR_SESSIONS_DIR, 'context.txt')

for _d in [VADAR_SESSIONS_DIR, VADAR_MEMORIES_DIR,
           VADAR_MEMORIES_VADAR_DIR, VADAR_MEMORIES_USER_DIR,
           VADAR_ABOUT_DIR]:
    os.makedirs(_d, exist_ok=True)

for _f, _default in [
    (VADAR_PING_TIME_FILE, '15\n'),
    (VADAR_SUPPORTED_LIBS_FILE, 'math\n'),
]:
    if not os.path.exists(_f):
        with open(_f, 'w', encoding='utf-8') as _fh:
            _fh.write(_default)

for _fname in ['personality.md', 'custom-vadar.md', 'user.md', 'how-to-respond.md']:
    _fpath = os.path.join(VADAR_ABOUT_DIR, _fname)
    if not os.path.exists(_fpath):
        with open(_fpath, 'w', encoding='utf-8') as _fh:
            _fh.write('')
