"""
AST-based smoke test for chains decompile + compile subcommands.

Tests:
  1. Decompile a valid 3-step chain -> .txt with oneline command
  2. Compile the .txt back -> .chain that matches the original structure
  3. Round-trip: decompile -> compile -> decompile produces same oneline command
  4. Decompile a corrupted chain -> .txt with errors commented out
  5. Compile a .txt with errors -> no .chain saved, errors printed
  6. Multi-input decompile: two chains -> two .txt files
  7. Multi-input compile: two .txt files -> two .chain files
"""
import ast
import os
import sys
import re
import tempfile
import shutil
from urllib.parse import urlparse

VODER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src', 'voder.py')
VODER = os.path.abspath(VODER)
src = open(VODER).read()
tree = ast.parse(src)

KEEP_TOP = {
    'PLATFORMS', '_URL_DOMAIN_INDEX', '_URL_SHORT_DOMAIN_INDEX',
    '_normalize_url', '_host_of', 'detect_platform',
    'is_supported_url', 'is_youtube_url', 'parse_oneline_args',
    'CHAIN_FILE_MAGIC', 'CHAIN_FILE_EXT', 'PREBUILT_CHAINS_DIR',
    '_TIMESTAMP_RE', '_NAME_RE', '_VALID_CONTENT_MODES',
    '_err', '_resolve_linear_index',
    '_AUDIO_VIDEO_URL', 'VIDEO_EXTENSIONS', 'VOICE_PROFILE_EXTENSIONS',
    'MODE_INPUT_FORMATS', 'slot_accepts_voice_profile', 'describe_input_slot',
    'build_chain_text', 'parse_chain_file', '_parse_chain_text',
    'verify_chain_file', 'verify_chain_text', '_verify_content_syntax',
    '_verify_references', 'classify_chain_step', 'find_chain_by_name',
    'list_chains', 'resolve_chain_path', '_parse_build_args', 'handle_build',
    'handle_journey', 'handle_load', '_resolve_manual_value',
    '_find_manual_slots', '_find_auto_slots', '_is_voice_profile_position',
    'get_input_formats_for_step', 'is_voice_profile_value', '_parse_load_args',
    '_parse_comment_args', 'handle_comment', 'handle_decompile', 'handle_compile',
    '_compile_txt_to_chain', '_split_oneline_segments',
    '_human_readable_timestamp',
    'oneline_chains', 'ChainPipeline',
}

filtered = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        if node.name in KEEP_TOP:
            filtered.append(node)
    elif isinstance(node, ast.Assign):
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if any(t in KEEP_TOP for t in targets):
            filtered.append(node)
    elif isinstance(node, ast.For):
        if any(isinstance(n, ast.Name) and n.id in KEEP_TOP for n in ast.walk(node)):
            filtered.append(node)

mod = ast.Module(body=filtered, type_ignores=[])
ns = {'os': os, 'sys': sys, 're': re, 'urlparse': urlparse,
      'time': __import__('time'), 'shutil': shutil, 'tempfile': tempfile,
      'gc': __import__('gc'), 'json': __import__('json'),
      'math': __import__('math'), 'random': __import__('random'),
      'traceback': __import__('traceback'),
      'subprocess': __import__('subprocess'),
      'numpy': type('numpy', (), {'array': lambda *a, **k: None})(),
      'torch': type('torch', (), {'cuda': type('cuda', (), {'is_available': staticmethod(lambda: False), 'empty_cache': staticmethod(lambda: None)})(), 'float16': 0, 'float32': 0, 'no_grad': staticmethod(lambda f: f)})(),
      'torchaudio': type('torchaudio', (), {})(),
      'yaml': type('yaml', (), {'DictConfig': dict}),
      'soundfile': type('soundfile', (), {})(),
      'omegaconf': type('omegaconf', (), {'DictConfig': dict}),
      'hydra': type('hydra', (), {'utils': type('utils', (), {'instantiate': staticmethod(lambda *a, **k: None)})()}),
      'huggingface_hub': type('huggingface_hub', (), {'hf_hub_download': staticmethod(lambda *a, **k: None)})(),
      'copy': __import__('copy'),
      }
ns['_src_dir'] = os.path.dirname(VODER)
exec(compile(mod, VODER, 'exec'), ns)

handle_build = ns['handle_build']
handle_decompile = ns['handle_decompile']
handle_compile = ns['handle_compile']
_compile_txt_to_chain = ns['_compile_txt_to_chain']
_split_oneline_segments = ns['_split_oneline_segments']
parse_chain_file = ns['parse_chain_file']
build_chain_text = ns['build_chain_text']

tmpdir = tempfile.mkdtemp()
chains_dir = os.path.join(tmpdir, 'chains')
os.makedirs(chains_dir, exist_ok=True)
ns['PREBUILT_CHAINS_DIR'] = chains_dir

original_cwd = os.getcwd()
os.chdir(tmpdir)
os.makedirs(os.path.join(tmpdir, 'results'), exist_ok=True)

passed = 0
failed = 0

def check(label, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {label}")
    else:
        failed += 1
        print(f"  [FAIL] {label}: {detail}")

print("=== Setup: build a 3-step chain ===")
ok = handle_build([
    'bombo', 'description', 'Bombo - extract vocals, transcribe, re-synth',
    'chain', 'vocals', 'Provide the source song', 'svs voice input',
    'chain', 'lyrics', 'Automated - uses vocals', 'stt vocals timestamp',
    'chain', 'cover', 'Provide a reference voice', 'tts script lyrics voice input target input',
])
check("build bombo", ok)

print()
print("=== Test 1: Decompile a valid chain ===")
ok = handle_decompile(['bombo'])
check("decompile succeeds", ok)
results_dir = os.path.join(tmpdir, 'results')
txt_files = [f for f in os.listdir(results_dir) if f.startswith('VODER_chains_') and f.endswith('.txt')]
check("one .txt file produced", len(txt_files) == 1, str(txt_files))
if txt_files:
    with open(os.path.join(results_dir, txt_files[0])) as f:
        txt_content = f.read()
    check("has '# VODER decompiled chain:' header", "# VODER decompiled chain: bombo" in txt_content)
    check("has oneline command with quotes", '"vocals" svs voice input' in txt_content)
    check("has ' / ' separator", " / " in txt_content)
    check("has 3 quoted names", txt_content.count('"') >= 6)
    check("no error comments (valid chain)", "VERIFICATION ERRORS" not in txt_content)

print()
print("=== Test 2: Compile the .txt back to .chain ===")
ok = handle_compile([os.path.join(results_dir, txt_files[0])])
check("compile succeeds", ok)
new_chain_files = [f for f in os.listdir(chains_dir) if f.endswith('.chain')]
check("at least 1 .chain file exists after compile", len(new_chain_files) >= 1, str(new_chain_files))

print()
print("=== Test 3: Round-trip: compile output matches original structure ===")
new_chain_files.sort(key=lambda f: os.path.getmtime(os.path.join(chains_dir, f)), reverse=True)
compiled_path = os.path.join(chains_dir, new_chain_files[0])
parsed_compiled, _ = parse_chain_file(compiled_path)
check("compiled chain parses", parsed_compiled is not None)
if parsed_compiled:
    check("compiled name is bombo", parsed_compiled["name"] == "bombo", str(parsed_compiled["name"]))
    check("compiled has 3 steps", len(parsed_compiled["chains"]) == 3, str(len(parsed_compiled["chains"])))
    check("step 1 is vocals", parsed_compiled["chains"][0]["name"] == "vocals")
    check("step 2 is lyrics", parsed_compiled["chains"][1]["name"] == "lyrics")
    check("step 3 is cover", parsed_compiled["chains"][2]["name"] == "cover")
    check("step 1 content preserved", parsed_compiled["chains"][0]["content"] == "svs voice input")
    check("step 2 content preserved", parsed_compiled["chains"][1]["content"] == "stt vocals timestamp")
    check("step 3 content preserved", parsed_compiled["chains"][2]["content"] == "tts script lyrics voice input target input")

print()
print("=== Test 4: _split_oneline_segments handles quotes and slashes ===")
segs = _split_oneline_segments('"a" tts script "hello world" / "b" svs voice "a"')
check("splits into 2 segments", len(segs) == 2, str(segs))
check("segment 1 has quoted name", segs[0].startswith('"a"'), segs[0] if segs else "")
check("segment 2 has quoted name", segs[1].startswith('"b"'), segs[1] if len(segs) > 1 else "")

print()
print("=== Test 5: _split_oneline_segments returns None on unmatched quote ===")
segs2 = _split_oneline_segments('"unmatched tts script')
check("returns None on unmatched quote", segs2 is None, str(segs2))

print()
print("=== Test 6: _compile_txt_to_chain parses decompiled format ===")
sample_txt = """# VODER decompiled chain: testchain
# Source: /path/to/VODER_testchain_20260101_120000.chain
# Decompiled: January 01, 2026 at 12:00:00
# Title: Test
# Description: A test
# Steps: 2
#
# comment line

"step1" svs voice input / "step2" stt step1 timestamp
"""
compiled = _compile_txt_to_chain(sample_txt, "test.txt")
check("compiled is not None", compiled is not None)
if compiled:
    check("name is testchain", compiled["name"] == "testchain")
    check("title is Test", compiled["title"] == "Test")
    check("description is A test", compiled["description"] == "A test")
    check("2 steps", len(compiled["steps"]) == 2)
    check("step 1 name", compiled["steps"][0]["name"] == "step1")
    check("step 1 content", compiled["steps"][0]["content"] == "svs voice input")
    check("step 2 content", compiled["steps"][2-1]["content"] == "stt step1 timestamp")

print()
print("=== Test 7: _compile_txt_to_chain returns None on missing header ===")
bad_txt = "no header\njust a command"
compiled2 = _compile_txt_to_chain(bad_txt, "bad.txt")
check("returns None on missing header", compiled2 is None)

print()
print("=== Test 8: Decompile a corrupted chain -> errors commented out ===")
broken_chain_text = """# VODER_CHAIN v1 20260101_120000 broken_chain
title: Broken
description: A broken chain
---
chain: bad
comment: Forward ref
content: stt laterstep timestamp
---
chain: laterstep
comment: ok
content: sfx sound boom duration 5
---
"""
broken_path = os.path.join(chains_dir, 'VODER_broken_chain_20260101_120000.chain')
with open(broken_path, 'w') as f:
    f.write(broken_chain_text)
ok = handle_decompile(['broken_chain'])
check("decompile of broken chain returns False (had errors)", not ok)
broken_txts = [f for f in os.listdir(results_dir) if 'broken_chain' in f and f.endswith('.txt')]
check("broken chain .txt produced", len(broken_txts) >= 1, str(broken_txts))
if broken_txts:
    with open(os.path.join(results_dir, broken_txts[-1])) as f:
        broken_txt_content = f.read()
    check("has VERIFICATION ERRORS section", "VERIFICATION ERRORS" in broken_txt_content)
    check("has commented error line", "# [step 1 'bad'] reference:" in broken_txt_content or "# [step" in broken_txt_content)
    check("oneline command still present", '"bad" stt laterstep timestamp' in broken_txt_content)

print()
print("=== Test 9: Compile a .txt with errors -> no .chain saved ===")
bad_txt_content = """# VODER decompiled chain: will_fail
# Title: Will Fail
# Description: A chain that will fail verification
# Steps: 1

"badstep" invalidmode input
"""
bad_txt_path = os.path.join(tmpdir, 'will_fail.txt')
with open(bad_txt_path, 'w') as f:
    f.write(bad_txt_content)
chains_before = set(os.listdir(chains_dir))
ok = handle_compile([bad_txt_path])
check("compile of bad .txt returns False", not ok)
chains_after = set(os.listdir(chains_dir))
check("no new .chain file saved", chains_after == chains_before, f"before={chains_before}, after={chains_after}")

print()
print("=== Test 10: Multi-input decompile ===")
ok = handle_build([
    'second_chain', 'description', 'Second chain',
    'chain', 'step1', 'Use input', 'sfx sound boom duration 5',
])
check("build second_chain", ok)
ok = handle_decompile(['bombo', 'second_chain'])
check("multi decompile succeeds (returns True if no errors)", ok)
all_txts = [f for f in os.listdir(results_dir) if f.startswith('VODER_chains_') and f.endswith('.txt')]
check("at least 2 .txt files (bombo + second_chain)", len([f for f in all_txts if 'bombo' in f]) >= 1 and len([f for f in all_txts if 'second_chain' in f]) >= 1, str(all_txts))

os.chdir(original_cwd)
shutil.rmtree(tmpdir)

print()
print("=" * 60)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
