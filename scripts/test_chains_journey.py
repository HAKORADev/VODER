"""
AST-based smoke test for the RPG-like journey narration.

Verifies the new journey report contains:
  - Opening narrative ("In a world full of complexity...")
  - Human-readable timestamp
  - Per-chain chapter with persona/artisan per mode
  - Per-step waypoint with classification narrative
  - Per-error alternate dimension block
  - Statistics ledger with mode breakdown
  - Multi-chain saga section
  - Epilogue
  - Per-mode persona names (Voice Weaver, Scribe, etc.)
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
    '_parse_comment_args', 'handle_comment', 'oneline_chains', 'ChainPipeline',
    '_journey_report', '_journey_opening', '_journey_one_chain', '_journey_saga',
    '_journey_statistics', '_journey_epilogue', '_journey_alternate_dimension',
    '_what_if_dimension', '_mode_persona', '_human_readable_timestamp',
    '_MODE_PERSONA', '_CLASSIFICATION_NARRATIVE',
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
handle_journey = ns['handle_journey']
_mode_persona = ns['_mode_persona']
_MODE_PERSONA = ns['_MODE_PERSONA']
_CLASSIFICATION_NARRATIVE = ns['_CLASSIFICATION_NARRATIVE']

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

print("=== Setup: build a 3-step chain with multiple modes ===")
ok = handle_build([
    'bombo', 'description', 'Bombo - extract vocals, transcribe, re-synth',
    'chain', 'vocals', 'Provide the source song', 'svs voice input',
    'chain', 'lyrics', 'Automated - uses vocals', 'stt vocals timestamp',
    'chain', 'cover', 'Provide a reference voice', 'tts script lyrics voice input target input',
])
check("build bombo", ok)

print()
print("=== Test 1: _MODE_PERSONA has all 10 valid modes + chains ===")
for mode in ['tts', 'sts', 'ttm', 'stt', 'se', 'sfx', 'svs', 'ss', 'train', 'quest', 'chains']:
    check(f"persona for {mode}", mode in _MODE_PERSONA, f"missing {mode}")

print()
print("=== Test 2: _mode_persona returns Unknown Artisan for unknown mode ===")
p = _mode_persona('nonexistent')
check("unknown mode persona", p['name'] == 'the Unknown Artisan', str(p))

print()
print("=== Test 3: _CLASSIFICATION_NARRATIVE has all 4 types ===")
for ctype in ['manual', 'automated', 'semi-automated', 'error']:
    check(f"narrative for {ctype}", ctype in _CLASSIFICATION_NARRATIVE, f"missing {ctype}")

print()
print("=== Test 4: handle_journey produces report ===")
ok = handle_journey(['bombo'])
check("handle_journey succeeds", ok)
results_dir = os.path.join(tmpdir, 'results')
reports = [f for f in os.listdir(results_dir) if f.startswith('voder_journey_')]
check("journey report file exists", len(reports) >= 1)
if reports:
    with open(os.path.join(results_dir, reports[-1])) as f:
        report = f.read()

    print()
    print("=== Test 5: Opening narrative ===")
    check("has 'In a world'", "In a world" in report)
    check("has 'complexity and many of the unknowns'", "complexity and many of the unknowns" in report)
    check("mentions bombo", "bombo" in report)
    check("has human-readable timestamp", "The journey began on" in report)

    print()
    print("=== Test 6: Chapter/Act structure ===")
    check("has 'Act 1'", "Act 1" in report or "Chapter 1" in report)
    check("has 'The Chain of bombo'", "The Chain of **bombo**" in report)

    print()
    print("=== Test 7: Per-step waypoints ===")
    check("has 'Waypoint 1'", "Waypoint 1" in report)
    check("has 'Waypoint 2'", "Waypoint 2" in report)
    check("has 'Waypoint 3'", "Waypoint 3" in report)

    print()
    print("=== Test 8: Per-mode persona (artisan) ===")
    check("has 'the Separator' (svs)", "the Separator" in report)
    check("has 'the Scribe' (stt)", "the Scribe" in report)
    check("has 'the Voice Weaver' (tts)", "the Voice Weaver" in report)

    print()
    print("=== Test 9: Classification narrative ===")
    check("has 'offering' somewhere", "offering" in report.lower())
    check("has 'echoes' or 'prior steps'", "echo" in report.lower() or "prior step" in report.lower())

    print()
    print("=== Test 10: Statistics ledger ===")
    check("has 'Ledger of the Journey'", "Ledger of the Journey" in report)
    check("has 'Chapters (prebuilt chains)'", "Chapters (prebuilt chains)" in report)
    check("has 'Waypoints (steps)'", "Waypoints (steps)" in report)
    check("has 'Artisans summoned'", "Artisans summoned" in report)

    print()
    print("=== Test 11: Epilogue ===")
    check("has 'Epilogue'", "## Epilogue" in report)
    check("has 'journey ends here'", "journey ends here" in report.lower() or "The journey ends" in report)

    print()
    print("=== Test 12: Per-input comments and offerings ===")
    check("has 'Offerings awaited'", "Offerings awaited" in report or "offering" in report.lower())

print()
print("=== Test 13: Multi-chain saga ===")
ok = handle_build([
    'second_chain', 'description', 'A second chain',
    'chain', 'step1', 'Use bombo output', 'sts base input target input',
])
check("build second_chain", ok)
ok = handle_journey(['bombo', 'second_chain'])
check("multi-chain journey succeeds", ok)
reports = [f for f in os.listdir(results_dir) if f.startswith('voder_journey_')]
if reports:
    with open(os.path.join(results_dir, reports[-1])) as f:
        report2 = f.read()
    check("has 'Saga'", "Saga" in report2)
    check("has 'Chapter 1'", "Chapter 1" in report2)
    check("has 'Chapter 2'", "Chapter 2" in report2)
    check("has 'linearity rule'", "linearity rule" in report2.lower())
    check("has 'echo from prior'", "echo from prior" in report2.lower())
    check("has 2 chapters in ledger", "Chapters (prebuilt chains) | 2" in report2)

print()
print("=== Test 14: Chain with errors produces alternate dimension ===")
broken_chain_text = """# VODER_CHAIN v1 20260101_120000 broken_chain
title: Broken Chain
description: A chain with a forward reference error
---
chain: bad
comment: Forward ref to laterstep
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
ok = handle_journey(['broken_chain'])
check("journey of broken chain returns False", not ok)
reports = [f for f in os.listdir(results_dir) if f.startswith('voder_journey_') and 'broken_chain' in f]
check("broken chain journey report exists", len(reports) >= 1)
if reports:
    with open(os.path.join(results_dir, reports[-1])) as f:
        report3 = f.read()
    check("has 'another dimension'", "another dimension" in report3.lower())
    check("has 'falter'", "falter" in report3.lower())
    check("has error in ledger", "Errors found" in report3)
    check("has 'All Errors' table", "All Errors" in report3)
    check("has 'would have been placed'", "would have been placed" in report3)

os.chdir(original_cwd)
shutil.rmtree(tmpdir)

print()
print("=" * 60)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
