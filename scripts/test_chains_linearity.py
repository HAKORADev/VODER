"""
AST-based smoke test for:
  1. Oneline `chains load` forward-reference detection (Gap 2 fix)
  2. `chains analyze` "what if" dimension + multi-chain narrative
  3. `_verify_references` still catches in-chain forward references (Gap 1 — not a gap)

Pulls the chains subsystem out of src/voder.py without importing torch.
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
    'handle_journey', '_journey_one_chain', '_journey_saga',
    '_journey_report', '_journey_opening', '_journey_one_chain', '_journey_saga',
    '_journey_statistics', '_journey_epilogue', '_journey_alternate_dimension',
    '_what_if_dimension', '_mode_persona', '_human_readable_timestamp',
    '_MODE_PERSONA', '_CLASSIFICATION_NARRATIVE',
    '_what_if_dimension', 'handle_load', '_resolve_manual_value',
    '_find_manual_slots', '_find_auto_slots', '_is_voice_profile_position',
    'get_input_formats_for_step', 'is_voice_profile_value', '_parse_load_args',
    '_parse_comment_args', 'handle_comment', 'oneline_chains', 'ChainPipeline',
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
handle_load = ns['handle_load']
handle_journey = ns['handle_journey']
verify_chain_text = ns['verify_chain_text']
build_chain_text = ns['build_chain_text']
parse_chain_file = ns['parse_chain_file']
_verify_references = ns['_verify_references']
_what_if_dimension = ns['_what_if_dimension']
_journey_saga = ns['_journey_saga']
_err = ns['_err']

tmpdir = tempfile.mkdtemp()
chains_dir = os.path.join(tmpdir, 'chains')
os.makedirs(chains_dir, exist_ok=True)
ns['PREBUILT_CHAINS_DIR'] = chains_dir

def _mock_parse_and_execute_oneline(args):
    mode = args[0].lower() if args else ""
    fake_output = os.path.join(tmpdir, "results", f"voder_{mode}_fake_output.wav")
    with open(fake_output, 'w') as f:
        f.write("")
    return True
ns['parse_and_execute_oneline'] = _mock_parse_and_execute_oneline

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

print("=== Setup: build two chains (vocals_chain, cover_chain) ===")
ok = handle_build([
    'vocals_chain', 'description', 'Extract vocals from a song',
    'chain', 'vocals', 'Provide the source song', 'svs voice input',
])
check("build vocals_chain", ok)

ok = handle_build([
    'cover_chain', 'description', 'Use vocals to make a cover',
    'chain', 'cover', 'Provide reference voice and the vocals', 'sts base input target input',
])
check("build cover_chain", ok)

print()
print("=== Test 1: _verify_references catches in-chain forward reference (Gap 1 — not a gap) ===")
chain_step = {
    "name": "step2",
    "content": "stt step9 timestamp",
    "content_tokens": ["stt", "step9", "timestamp"],
}
all_names = ["step1", "step2", "step9"]
errs = _verify_references(2, chain_step, all_names)
check("forward reference detected", len(errs) == 1 and "Forward reference" in errs[0]["message"], str(errs))

print()
print("=== Test 2: _verify_references does NOT flag backward reference ===")
chain_step2 = {
    "name": "step3",
    "content": "stt step1 timestamp",
    "content_tokens": ["stt", "step1", "timestamp"],
}
errs2 = _verify_references(3, chain_step2, all_names)
check("backward reference OK", len(errs2) == 0, str(errs2))

print()
print("=== Test 3: _verify_references does NOT flag self-reference ===")
chain_step3 = {
    "name": "step1",
    "content": "tts script step1 voice input",
    "content_tokens": ["tts", "script", "step1", "voice", "input"],
}
errs3 = _verify_references(1, chain_step3, all_names)
check("self-reference OK", len(errs3) == 0, str(errs3))

print()
print("=== Test 4: oneline `chains load` forward reference detection (Gap 2 fix) ===")
ok = handle_load(['cover_chain', '1:(vocals_chain/ref.wav)', 'vocals_chain', '1:(song.wav)'])
check("forward reference rejected", not ok, "should have failed but didn't")

print()
print("=== Test 5: oneline `chains load` correct order works ===")
ok = handle_load(['vocals_chain', '1:(song.wav)', 'cover_chain', '1:(vocals_chain/ref.wav)'])
check("correct order accepted", ok, "should have succeeded but didn't")

print()
print("=== Test 6: _what_if_dimension for forward reference error ===")
step_errors = [_err(2, "step2", "reference",
                    "Forward reference: 'step9' is defined later in the file",
                    "Move step 'step9' before step 'step2'.")]
whatif = _what_if_dimension(2, chain_step, all_names, step_errors)
check("what-if mentions placing step", whatif is not None and "placed before" in whatif, str(whatif))

print()
print("=== Test 7: _what_if_dimension for syntax error (invalid mode) ===")
bad_step = {
    "name": "bad",
    "content": "invalidmode input",
    "content_tokens": ["invalidmode", "input"],
}
syntax_errors = [_err(1, "bad", "syntax", "Unknown oneline mode 'invalidmode'", "Use one of: tts, sts, ...")]
whatif2 = _what_if_dimension(1, bad_step, ["bad"], syntax_errors)
check("what-if mentions recognized mode", whatif2 is not None and "recognized" in whatif2, str(whatif2))

print()
print("=== Test 8: _what_if_dimension for syntax error (valid mode, bad args) ===")
ok_mode_step = {
    "name": "okmode",
    "content": "tts script input badarg",
    "content_tokens": ["tts", "script", "input", "badarg"],
}
syntax_errors2 = [_err(1, "okmode", "syntax", "Oneline parser error: Invalid argument: badarg", "Fix the oneline syntax.")]
whatif3 = _what_if_dimension(1, ok_mode_step, ["okmode"], syntax_errors2)
check("what-if mentions mode execution", whatif3 is not None and "`tts`" in whatif3, str(whatif3))

print()
print("=== Test 9: _what_if_dimension returns None for no errors ===")
whatif4 = _what_if_dimension(1, chain_step, all_names, [])
check("what-if None for no errors", whatif4 is None, str(whatif4))

print()
print("=== Test 10: _journey_saga produces content ===")
cr1 = {"parsed": parse_chain_file(os.path.join(chains_dir, [f for f in os.listdir(chains_dir) if 'vocals_chain' in f][0]))[0], "ok": True, "errors": [], "warnings": []}
cr2 = {"parsed": parse_chain_file(os.path.join(chains_dir, [f for f in os.listdir(chains_dir) if 'cover_chain' in f][0]))[0], "ok": True, "errors": [], "warnings": []}
narrative = _journey_saga([cr1, cr2])
narrative_text = "\n".join(narrative)
check("narrative has Saga header", "Saga" in narrative_text)
check("narrative mentions vocals_chain", "vocals_chain" in narrative_text)
check("narrative mentions cover_chain", "cover_chain" in narrative_text)
check("narrative mentions linearity rule", "linearity rule" in narrative_text.lower())
check("narrative mentions prior chapter echo", "echo from prior" in narrative_text.lower())

print()
print("=== Test 11: handle_journey produces report with what-if + multi-chain narrative ===")
ok = handle_journey(['vocals_chain', 'cover_chain'])
check("handle_journey succeeds", ok)
results_dir = os.path.join(tmpdir, 'results')
reports = [f for f in os.listdir(results_dir) if f.startswith('voder_journey_')]
check("journey report exists", len(reports) >= 1)
if reports:
    with open(os.path.join(results_dir, reports[-1])) as f:
        report_text = f.read()
    check("report has Multi-Chain Journey", "Multi-Chain Journey" in report_text or "Saga" in report_text)
    check("report has linearity rule", "Linearity rule" in report_text or "linearity rule" in report_text)

os.chdir(original_cwd)
shutil.rmtree(tmpdir)

print()
print("=" * 60)
print(f"TOTAL: {passed} passed, {failed} failed")
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
