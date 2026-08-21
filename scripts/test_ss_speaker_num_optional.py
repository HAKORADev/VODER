"""
AST-based smoke test for the SS speaker_num optional behavior.

Pulls the URL helpers + PLATFORMS table + parse_oneline_args out of
src/voder.py and execs them in an isolated namespace, so we can call
parse_oneline_args() directly without importing torch / torchaudio /
huggingface_hub / etc.

Verifies:
  1. ss "x.wav"            -> no error, speaker_num is None, file_path set
  2. ss 1 "x.wav"          -> no error, speaker_num normalized to 1
  3. ss 2 "x.wav"          -> no error, speaker_num == 2
  4. ss 999 "x.wav"        -> no error, speaker_num == 999 (left as-is, resolved later)
  5. ss 0 "x.wav"          -> no error, speaker_num normalized to 1
  6. ss target r.wav x.wav -> no error, target_path set, speaker_num is None
  7. ss se blend "x.wav"   -> no error, speaker_num is None, use_se=True, use_blend=True
  8. ss overdose 3 "x.wav" -> no error, speaker_num == 3, use_overdose=True
  9. ss "x.wav" extra      -> error (invalid positional after file_path)  [sanity]
"""
import ast
import os
import sys
import re
from urllib.parse import urlparse

VODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'voder.py')
VODER = os.path.abspath(VODER)
src = open(VODER).read()
tree = ast.parse(src)

KEEP_NAMES = {
    'PLATFORMS', '_normalize_url', '_host_of', 'detect_platform',
    'is_supported_url', 'is_youtube_url', 'parse_oneline_args',
    '_URL_DOMAIN_INDEX', '_URL_SHORT_DOMAIN_INDEX',
}

filtered = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name in KEEP_NAMES:
            filtered.append(node)
    elif isinstance(node, ast.ClassDef):
        continue
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        continue
    elif isinstance(node, ast.Assign):
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(t in KEEP_NAMES for t in targets):
            filtered.append(node)
    elif isinstance(node, ast.For):
        if any(isinstance(n, ast.Name) and n.id in KEEP_NAMES for n in ast.walk(node)):
            filtered.append(node)
    elif isinstance(node, ast.Expr):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        else:
            continue

mod = ast.Module(body=filtered, type_ignores=[])
ns = {'os': os, 'sys': sys, 're': re, 'urlparse': urlparse}
exec(compile(mod, VODER, 'exec'), ns)

parse = ns['parse_oneline_args']

import tempfile
tmpdir = tempfile.mkdtemp()
fake_files = {}
for name in ['x.wav', 'r.wav', 'interview.mp4', 'vlog.wav', 'noisy_conversation.wav', 'conversation.wav']:
    p = os.path.join(tmpdir, name)
    open(p, 'w').close()
    fake_files[name] = p

def F(name):
    return fake_files[name]

cases = [
    # (label, argv, expected_no_error, expected_speaker_num, expected_target_path, expected_extras)
    ('blind no number',           [F('x.wav')],                                  True,  None, None, {}),
    ('blind number 1',            ['1', F('x.wav')],                             True,  1,    None, {}),
    ('blind number 2',            ['2', F('x.wav')],                             True,  2,    None, {}),
    ('blind number 999',          ['999', F('x.wav')],                           True,  999,  None, {}),
    ('blind number 0 -> 1',       ['0', F('x.wav')],                             True,  1,    None, {}),
    ('target mode',               ['target', F('r.wav'), F('x.wav')],            True,  None, F('r.wav'), {}),
    ('flags only, no number',     ['se', 'blend', F('x.wav')],                   True,  None, None, {'use_se': True, 'use_blend': True}),
    ('overdose + number',         ['overdose', '3', F('x.wav')],                 True,  3,    None, {'overdose': True}),
    ('invalid positional extra',  [F('x.wav'), 'extra'],                         False, None, None, {}),
]

passed = 0
failed = 0
for label, argv, expect_ok, expect_spk, expect_target, expect_extras in cases:
    res = parse(['ss'] + argv)
    err = res.get('error')
    ok = True
    msgs = []
    if expect_ok:
        if err is not None:
            ok = False
            msgs.append(f'unexpected error: {err!r}')
        else:
            p = res.get('params', {})
            if p.get('speaker_num') != expect_spk:
                ok = False
                msgs.append(f'speaker_num = {p.get("speaker_num")!r}, expected {expect_spk!r}')
            if expect_target is not None and p.get('target_path') != expect_target:
                ok = False
                msgs.append(f'target_path = {p.get("target_path")!r}, expected {expect_target!r}')
            for k, v in expect_extras.items():
                if p.get(k) != v:
                    ok = False
                    msgs.append(f'params[{k!r}] = {p.get(k)!r}, expected {v!r}')
    else:
        if err is None:
            ok = False
            msgs.append('expected an error but parse succeeded')
    status = 'PASS' if ok else 'FAIL'
    if ok:
        passed += 1
    else:
        failed += 1
    print(f'[{status}] {label}: argv={argv!r}')
    for m in msgs:
        print(f'         - {m}')
    if err is not None and not expect_ok:
        print(f'         (expected error: {err!r})')

print()
print(f'Total: {passed} passed, {failed} failed')
sys.exit(0 if failed == 0 else 1)

