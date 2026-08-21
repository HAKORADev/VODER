"""
AST-based smoke test for `chains comment` + per-input comments.

Pulls the chains subsystem out of src/voder.py without importing torch:
  - PLATFORMS, _URL_DOMAIN_INDEX, _URL_SHORT_DOMAIN_INDEX, _normalize_url,
    _host_of, detect_platform, is_supported_url, is_youtube_url
  - CHAIN_FILE_MAGIC, CHAIN_FILE_EXT, PREBUILT_CHAINS_DIR, _TIMESTAMP_RE,
    _NAME_RE, _VALID_CONTENT_MODES, _err, _resolve_linear_index
  - MODE_INPUT_FORMATS, _AUDIO_VIDEO_URL, VIDEO_EXTENSIONS,
    VOICE_PROFILE_EXTENSIONS, slot_accepts_voice_profile, describe_input_slot
  - parse_oneline_args (for content syntax verification)
  - All chains helpers: build_chain_text, parse_chain_file, _parse_chain_text,
    verify_chain_file, verify_chain_text, _verify_content_syntax,
    _verify_references, classify_chain_step, find_chain_by_name, list_chains,
    resolve_chain_path, _parse_build_args, handle_build, handle_journey,
    _journey_one_chain, handle_load, _resolve_manual_value, _find_manual_slots,
    _find_auto_slots, _is_voice_profile_position, get_input_formats_for_step,
    is_voice_profile_value, _parse_load_args, _parse_comment_args,
    handle_comment, oneline_chains, ChainPipeline

Then creates a temp PREBUILT_CHAINS_DIR and exercises:
  1. Build a 3-step chain (vocals/lyrics/cover)
  2. Edit chain comments only (linear): 1:"...", 3:"..."
  3. Edit input comments only (non-linear): 3:(2:second/1:first)
  4. Edit both kinds at once
  5. Invalid chain number 9 (chain has 3 steps) -> "failed to resolve"
  6. Invalid input number 4 in step 1 (step has 1 input) -> "failed to resolve"
  7. Unmentioned slots are preserved across edits
  8. Round-trip: parse + rebuild + parse again should preserve comments
  9. Bad argv (no edits) -> error
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
    'handle_journey', '_journey_one_chain', 'handle_load',
    '_resolve_manual_value', '_find_manual_slots', '_find_auto_slots',
    '_is_voice_profile_position', 'get_input_formats_for_step',
    'is_voice_profile_value', '_parse_load_args', '_parse_comment_args',
    'handle_comment', 'oneline_chains', 'ChainPipeline',
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
      'torchaudio': type('torchaudio', (), {}),
      'yaml': type('yaml', (), {'DictConfig': dict}),
      'soundfile': type('soundfile', (), {}),
      'omegaconf': type('omegaconf', (), {'DictConfig': dict}),
      'hydra': type('hydra', (), {'utils': type('utils', (), {'instantiate': staticmethod(lambda *a, **k: None)})()}),
      'huggingface_hub': type('huggingface_hub', (), {'hf_hub_download': staticmethod(lambda *a, **k: None)})(),
      'copy': __import__('copy'),
      }

sys.path.insert(0, os.path.join(os.path.dirname(VODER)))
ns['_src_dir'] = os.path.dirname(VODER)

exec(compile(mod, VODER, 'exec'), ns)

build_chain_text = ns['build_chain_text']
parse_chain_file = ns['parse_chain_file']
verify_chain_text = ns['verify_chain_text']
handle_comment = ns['handle_comment']
handle_build = ns['handle_build']
_parse_comment_args = ns['_parse_comment_args']
_resolve_linear_index = ns['_resolve_linear_index']

tmpdir = tempfile.mkdtemp()
chains_dir = os.path.join(tmpdir, 'chains')
os.makedirs(chains_dir, exist_ok=True)
ns['PREBUILT_CHAINS_DIR'] = chains_dir

build_argv = [
    'bombo', 'description', 'Bombo - extract vocals, transcribe, re-synth',
    'chain', 'vocals', 'Provide the source song', 'svs voice input',
    'chain', 'lyrics', 'Automated - uses vocals', 'stt vocals timestamp',
    'chain', 'cover', 'Provide a reference voice', 'tts script lyrics voice input target input',
]

ok = handle_build(build_argv)
assert ok, "handle_build should succeed"
chain_files = [f for f in os.listdir(chains_dir) if f.endswith('.chain')]
assert len(chain_files) == 1, f"Expected 1 chain file, got {chain_files}"
chain_path = os.path.join(chains_dir, chain_files[0])

print(f"Built chain at: {chain_path}")
print()

with open(chain_path) as f:
    print("--- Initial chain file ---")
    print(f.read())
print()

print("=== Test 1: Edit chain comments only (linear 1, 3) ===")
ok = handle_comment(['bombo', '1:"NEW: provide source song (audio/video/URL)"', '3:"NEW: provide reference voice"'])
assert ok, "handle_comment should succeed for linear chain-comment edit"
with open(chain_path) as f:
    content = f.read()
assert "NEW: provide source song" in content
assert "NEW: provide reference voice" in content
parsed, errs = parse_chain_file(chain_path)
assert parsed is not None
assert parsed["chains"][0]["comment"] == "NEW: provide source song (audio/video/URL)"
assert parsed["chains"][1]["comment"] == "Automated - uses vocals"
assert parsed["chains"][2]["comment"] == "NEW: provide reference voice"
print("PASS: chain comments edited, step 2 untouched")
print()

print("=== Test 2: Edit input comments only (non-linear 3, then 1) ===")
ok = handle_comment(['bombo', '3:(2:second input of step 3/1:first input of step 3)', '1:(1:the source song)'])
assert ok, "handle_comment should succeed for non-linear input-comment edit"
parsed, errs = parse_chain_file(chain_path)
assert parsed["chains"][0]["input_comments"][1] == "the source song"
assert parsed["chains"][2]["input_comments"][1] == "first input of step 3"
assert parsed["chains"][2]["input_comments"][2] == "second input of step 3"
assert parsed["chains"][1].get("input_comments", {}) == {}
print("PASS: input comments edited in non-linear order, step 2 still has no input comments")
print()

print("=== Test 3: Both kinds at once ===")
ok = handle_comment(['bombo', '2:"UPDATED step 2 chain comment"', '1:(1:updated source song desc)'])
assert ok
parsed, errs = parse_chain_file(chain_path)
assert parsed["chains"][1]["comment"] == "UPDATED step 2 chain comment"
assert parsed["chains"][0]["input_comments"][1] == "updated source song desc"
assert parsed["chains"][0]["comment"] == "NEW: provide source song (audio/video/URL)"
assert parsed["chains"][2]["input_comments"][1] == "first input of step 3"
print("PASS: both kinds edited at once, unmentioned slots preserved")
print()

print("=== Test 4: Invalid chain number (9, chain has 3 steps) ===")
ok = handle_comment(['bombo', '9:"this should fail"'])
assert not ok, "handle_comment should fail for invalid chain number"
print("PASS: invalid chain number rejected")
print()

print("=== Test 5: Invalid input number (4 in step 1, step 1 has 1 input) ===")
ok = handle_comment(['bombo', '1:(4:this should fail)'])
assert not ok, "handle_comment should fail for invalid input number"
print("PASS: invalid input number rejected")
print()

print("=== Test 6: No edits provided -> error ===")
ok = handle_comment(['bombo'])
assert not ok, "handle_comment should fail when no edits provided"
print("PASS: no-edits case rejected")
print()

print("=== Test 7: Round-trip preservation ===")
parsed_before, _ = parse_chain_file(chain_path)
raw = build_chain_text(parsed_before["name"], parsed_before["timestamp"],
                       parsed_before["title"], parsed_before["description"],
                       parsed_before["chains"])
parsed_after, errs = parse_chain_text = (ns['_parse_chain_text'])(raw)
assert parsed_after is not None
assert parsed_after["chains"][0]["comment"] == parsed_before["chains"][0]["comment"]
assert parsed_after["chains"][0]["input_comments"] == parsed_before["chains"][0]["input_comments"]
assert parsed_after["chains"][2]["input_comments"] == parsed_before["chains"][2]["input_comments"]
print("PASS: round-trip preserves all comments")
print()

print("=== Test 8: Direct path also works (not just name) ===")
ok = handle_comment([chain_path, '1:"via direct path"'])
assert ok
parsed, _ = parse_chain_file(chain_path)
assert parsed["chains"][0]["comment"] == "via direct path"
print("PASS: direct path accepted")
print()

print("=== Test 9: _resolve_linear_index helper unit checks ===")
zero_idx, err = _resolve_linear_index(2, 5, "step", "in 'x'")
assert zero_idx == 1 and err is None
zero_idx, err = _resolve_linear_index(6, 5, "step", "in 'x'")
assert zero_idx is None and "failed to resolve" in err and "Likely meant" in err
zero_idx, err = _resolve_linear_index(0, 5, "step", "in 'x'")
assert zero_idx is None and "failed to resolve" in err
zero_idx, err = _resolve_linear_index(1, 1, "input slot", "in step 1 'y'")
assert zero_idx == 0 and err is None
zero_idx, err = _resolve_linear_index(2, 1, "input slot", "in step 1 'y'")
assert zero_idx is None and "failed to resolve" in err
print("PASS: _resolve_linear_index behaves correctly")
print()

print("=== Test 10: Malformed argv cases ===")
bad_cases = [
    ['bombo', '1:unquoted comment'],
    ['bombo', '1:(no-colon-means-error)'],
    ['bombo', '1:(abc:bad-index)'],
    ['bombo', '1:(0:zero-index)'],
    ['bombo', '1:"unclosed quote'],
    ['bombo', '1:(1:comment with / inside)'],
]
for bad_argv in bad_cases:
    parsed, err = _parse_comment_args(bad_argv)
    assert err is not None, f"Expected error for {bad_argv!r}"
    print(f"  PASS: {bad_argv!r} -> {err}")
print()

print("=== Test 11: Empty input comment clears (1:) ===")
ok = handle_comment(['bombo', '1:(1:)'])
assert ok
parsed, _ = parse_chain_file(chain_path)
assert parsed["chains"][0]["input_comments"][1] == ""
print("PASS: empty input comment accepted (clears)")
print()

shutil.rmtree(tmpdir)
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
