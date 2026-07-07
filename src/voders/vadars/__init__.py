VADAR_DIR = None

import os
_voders_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_voders_dir)
_project_root = os.path.dirname(_src_dir)
VADAR_DIR = os.path.join(_project_root, 'vadars')

try:
    from voder import VADAR_MODEL_DIR
except ImportError:
    VADAR_MODEL_DIR = os.path.join(_src_dir, 'models', 'checkpoints', 'vadar')

VADAR_SESSIONS_DIR = os.path.join(VADAR_DIR, 'sessions')
VADAR_MEMORIES_DIR = os.path.join(VADAR_DIR, 'memories')
VADAR_ABOUT_DIR = os.path.join(VADAR_DIR, 'about')
VADAR_PING_TIME_FILE = os.path.join(VADAR_DIR, 'ping-time.txt')
VADAR_SUPPORTED_LIBS_FILE = os.path.join(VADAR_DIR, 'supported_libs.txt')
VADAR_GLOBAL_CONTEXT_FILE = os.path.join(VADAR_DIR, 'context.txt')

for _d in [VADAR_DIR, VADAR_SESSIONS_DIR, VADAR_MEMORIES_DIR,
           os.path.join(VADAR_MEMORIES_DIR, 'vadar'),
           os.path.join(VADAR_MEMORIES_DIR, 'user'),
           VADAR_ABOUT_DIR]:
    os.makedirs(_d, exist_ok=True)

for _f, _default in [
    (VADAR_PING_TIME_FILE, '15\n'),
    (VADAR_SUPPORTED_LIBS_FILE, 'math\n'),
    (os.path.join(VADAR_ABOUT_DIR, 'personality.md'), None),
    (os.path.join(VADAR_ABOUT_DIR, 'custom-vadar.md'), None),
    (os.path.join(VADAR_ABOUT_DIR, 'user.md'), None),
    (os.path.join(VADAR_ABOUT_DIR, 'how-to-respond.md'), None),
]:
    if not os.path.exists(_f) and _default is not None:
        with open(_f, 'w', encoding='utf-8') as _fh:
            _fh.write(_default)
