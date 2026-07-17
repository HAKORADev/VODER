import os
import sys
import time
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
    return quest.execute(parsed, results_dir, timestamp, result_path=None)


_discover_quests()
