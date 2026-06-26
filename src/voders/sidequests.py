import os
import re
import sys
import time
import shutil
import importlib

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


class SideQuest:
    name = None
    description = ""

    def parse(self, args):
        raise NotImplementedError

    def execute(self, parsed, results_dir, timestamp, result_path=None):
        raise NotImplementedError


SIDE_QUESTS = {}


def _register_side_quest(quest_cls):
    inst = quest_cls()
    SIDE_QUESTS[inst.name] = inst
    return inst


def _discover_quests():
    quests_pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'quests')
    if not os.path.isdir(quests_pkg_dir):
        return
    for entry in sorted(os.listdir(quests_pkg_dir)):
        if not entry.endswith('.py') or entry.startswith('_'):
            continue
        stem = entry[:-3]
        try:
            module = importlib.import_module(f'voders.quests.{stem}')
        except Exception as e:
            print(f"Warning: failed to load quest '{stem}': {e}")
            continue
        quest_cls = getattr(module, 'Quest', None)
        if quest_cls is None or not isinstance(quest_cls, type):
            continue
        if not issubclass(quest_cls, SideQuest):
            continue
        if quest_cls.name is None:
            quest_cls.name = stem
        if quest_cls.name not in SIDE_QUESTS:
            _register_side_quest(quest_cls)


def list_available_quests():
    if not SIDE_QUESTS:
        print("No side-quests are currently registered.")
        print()
        print("Drop a new quest file into src/voders/quests/ to add one.")
        return

    try:
        from voders.quests_categories import CATEGORIES
    except Exception:
        CATEGORIES = []

    categorized = set()
    cat_structure = []
    for cat in CATEGORIES:
        cat_name = (cat.get('name') or '').strip()
        top_quests = []
        for q in cat.get('quests', []) or []:
            if q in SIDE_QUESTS and q not in categorized:
                top_quests.append(q)
                categorized.add(q)
        sub_cats = []
        for sub in cat.get('subcategories', []) or []:
            sub_name = (sub.get('name') or '').strip()
            sub_quests = []
            for q in sub.get('quests', []) or []:
                if q in SIDE_QUESTS and q not in categorized:
                    sub_quests.append(q)
                    categorized.add(q)
            if sub_quests:
                sub_cats.append((sub_name, sub_quests))
        if top_quests or sub_cats:
            cat_structure.append((cat_name, top_quests, sub_cats))

    uncategorized = [q for q in sorted(SIDE_QUESTS.keys()) if q not in categorized]
    max_name = max((len(n) for n in SIDE_QUESTS.keys()), default=0)

    def _quest_line(name, indent):
        quest = SIDE_QUESTS[name]
        desc = (quest.description or '').strip() or '(no description)'
        print(f"{indent}{name:<{max_name}}  -  {desc}")

    print("Available side-quests:")
    print()
    if uncategorized:
        for name in uncategorized:
            _quest_line(name, '  ')
        print()
    for cat_name, top_quests, sub_cats in cat_structure:
        print(f"{cat_name}:")
        for name in top_quests:
            _quest_line(name, '  ')
        for sub_name, sub_quests in sub_cats:
            print(f"  {sub_name}:")
            for name in sub_quests:
                _quest_line(name, '    ')
        print()
    print("Usage:  python voder.py quest <name> [args...]")
    print("(Side-quests can be used directly by name — no prefix needed.)")


def oneline_quest(params):
    if params.get('list_quests'):
        list_available_quests()
        return True
    quest_name = params.get('quest_name')
    quest_args = params.get('quest_args', [])
    result_path = params.get('result_path')
    if not quest_name:
        print("Error: quest mode requires a quest name")
        return False
    quest = SIDE_QUESTS.get(quest_name)
    if quest is None:
        print(f"Error: unknown quest '{quest_name}'. Available quests: {', '.join(sorted(SIDE_QUESTS.keys()))}")
        return False
    parsed, err = quest.parse(quest_args)
    if err:
        print(f"Error: {err}")
        return False
    results_dir = os.path.join(os.getcwd(), "results")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return quest.execute(parsed, results_dir, timestamp, result_path=result_path)


class ChainPipeline:
    CHAIN_SEPARATOR = '/'

    def __init__(self):
        self.index = {}

    def split_segments(self, args):
        segments = []
        current = []
        for arg in args:
            if arg == self.CHAIN_SEPARATOR:
                segments.append(current)
                current = []
            else:
                current.append(arg)
        segments.append(current)
        return segments

    def parse_chain_segment(self, seg):
        if not seg:
            return None, None
        first = seg[0]
        if len(first) >= 2 and first.startswith('"') and first.endswith('"'):
            name = first[1:-1]
        else:
            name = first
        if not name:
            return None, "chain name cannot be empty"
        command_args = seg[1:]
        return name, command_args

    def validate(self, parsed_chains):
        seen = set()
        valid = []
        for name, command_args in parsed_chains:
            if not command_args:
                continue
            if name in seen:
                return None, f"Duplicate chain name: '{name}'"
            seen.add(name)
            valid.append((name, command_args))
        return valid, None

    def substitute_refs(self, command_args):
        out = []
        for a in command_args:
            if a in self.index:
                out.append(self.index[a])
            else:
                out.append(a)
        return out

    def _snapshot(self, directory):
        if not os.path.isdir(directory):
            return {}
        snap = {}
        for f in os.listdir(directory):
            p = os.path.join(directory, f)
            if os.path.isfile(p):
                snap[f] = os.path.getmtime(p)
        return snap

    def _new_files(self, directory, before):
        if not os.path.isdir(directory):
            return []
        new = []
        for f in os.listdir(directory):
            p = os.path.join(directory, f)
            if not os.path.isfile(p):
                continue
            if f not in before:
                new.append(p)
            elif os.path.getmtime(p) > before[f]:
                new.append(p)
        return new

    def execute(self, chains_args, result_path=None):
        from voder import parse_and_execute_oneline
        segments = self.split_segments(chains_args)
        parsed_chains = []
        for seg in segments:
            name, command_args = self.parse_chain_segment(seg)
            if name is None and command_args is None:
                continue
            if name is None:
                return False, command_args
            parsed_chains.append((name, command_args))

        valid_chains, err = self.validate(parsed_chains)
        if err:
            print(f"Error: {err}")
            return False, err
        if not valid_chains:
            print("Error: no valid chains to execute (all chains were empty)")
            return False, "no valid chains"

        chains_temp_dir = os.path.join(os.getcwd(), "temp_chains")
        os.makedirs(chains_temp_dir, exist_ok=True)
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        voices_dir = os.path.join(os.getcwd(), "voices")

        total = len(valid_chains)
        print(f"Executing {total} chain(s)...")
        for idx, (name, command_args) in enumerate(valid_chains, start=1):
            is_last = (idx == total)
            substituted = self.substitute_refs(command_args)
            display_cmd = ' '.join(substituted)
            print(f"\n[Chain {idx}/{total}] name=\"{name}\"  >>>  {display_cmd}")

            results_before = self._snapshot(results_dir)
            voices_before = self._snapshot(voices_dir)

            success = parse_and_execute_oneline(substituted)
            if not success:
                print(f"Error: chain '{name}' failed")
                return False, f"chain '{name}' failed"

            new_results = self._new_files(results_dir, results_before)
            new_voices = self._new_files(voices_dir, voices_before)
            all_new = new_results + new_voices
            if not all_new:
                print(f"Warning: chain '{name}' produced no output file")
                continue
            all_new.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            chain_output = all_new[0]
            ts = time.strftime("%Y%m%d_%H%M%S")
            if is_last:
                self.index[name] = chain_output
                print(f"[Chain '{name}'] final output retained: {chain_output}")
                if result_path:
                    try:
                        shutil.copy2(chain_output, result_path)
                        print(f"Result copied to: {result_path}")
                    except Exception as e:
                        print(f"Note: could not copy to result path: {e}")
            else:
                ext = os.path.splitext(chain_output)[1] or '.bin'
                safe_name = re.sub(r'[^A-Za-z0-9_\-]', '_', name)[:40] or 'chain'
                temp_path = os.path.join(chains_temp_dir, f"voder_chain_{safe_name}_{ts}{ext}")
                shutil.move(chain_output, temp_path)
                for extra in all_new[1:]:
                    try:
                        os.remove(extra)
                    except Exception:
                        pass
                self.index[name] = temp_path
                print(f"[Chain '{name}'] intermediate output stored: {temp_path}")
        return True, None


def oneline_chains(params):
    chains_args = params.get('chains_args', [])
    result_path = params.get('result_path')
    subcmd = params.get('chains_subcmd')
    if subcmd == 'build':
        from voders.prebuilt_chains import handle_build
        return handle_build(chains_args)
    if subcmd == 'load':
        from voders.prebuilt_chains import handle_load
        return handle_load(chains_args, result_path=result_path)
    if subcmd == 'analyze':
        from voders.prebuilt_chains import handle_analyze
        return handle_analyze(chains_args)
    if not chains_args:
        print("Error: chains mode requires at least one chain")
        return False
    pipeline = ChainPipeline()
    ok, _err = pipeline.execute(chains_args, result_path=result_path)
    return ok


_discover_quests()
