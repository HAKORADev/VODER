import os
import sys

_VODER_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _VODER_SRC not in sys.path:
    sys.path.insert(0, _VODER_SRC)


def print_banner():
    print("""
██    ██  ██████  ██████  ███████ ██████
██    ██ ██    ██ ██   ██ ██      ██   ██
██    ██ ██    ██ ██   ██ █████   ██████
 ██  ██  ██    ██ ██   ██ ██      ██   ██
  ████    ██████  ██████  ███████ ██   ██
""")
    print("=" * 60)
    print("Interactive CLI Mode - Voice Blender Tool")
    print("=" * 60)


_MODE_DISPATCH = None


def _load_dispatch_table():
    global _MODE_DISPATCH
    if _MODE_DISPATCH is not None:
        return _MODE_DISPATCH
    from voders.interactiveCLI.tts import cli_tts_mode
    from voders.interactiveCLI.sts import cli_sts_mode
    from voders.interactiveCLI.ttm import cli_ttm_mode
    from voders.interactiveCLI.se import cli_se_mode
    from voders.interactiveCLI.sfx import cli_sfx_mode
    from voders.interactiveCLI.svs import cli_svs_mode
    from voders.interactiveCLI.stt import cli_stt_mode
    from voders.interactiveCLI.ss import cli_ss_mode
    from voders.interactiveCLI.chains import cli_chains_mode
    _MODE_DISPATCH = {
        '1': cli_tts_mode,
        '2': cli_sts_mode,
        '3': cli_ttm_mode,
        '4': cli_se_mode,
        '5': cli_sfx_mode,
        '6': cli_svs_mode,
        '7': cli_stt_mode,
        '8': cli_ss_mode,
        '9': cli_chains_mode,
        '10': _cli_vadar_mode,
    }
    return _MODE_DISPATCH


def _cli_vadar_mode():
    from voders.vadars.vadar import run_vadar_interactive
    return run_vadar_interactive()


def interactive_cli_mode():
    dispatch = _load_dispatch_table()
    while True:
        print_banner()
        print("\nSelect Mode:")
        print("1. TTS (Text-to-Speech)")
        print("2. STS (Speech-to-Speech / Voice Conversion)")
        print("3. TTM (Text-to-Music)")
        print("4. SE (Sound Enhancement)")
        print("5. SFX (Sound Effects Generation)")
        print("6. SVS (Song Voice Separate)")
        print("7. STT (Speech-to-Text)")
        print("8. SS (Speakers Separator)")
        print("9. Prebuilt Chains (load and run saved chain files)")
        print("10. VADAR (AI agent — talk naturally, it decides what to run)")
        choice = input("\nEnter your choice (1-10): ").strip()
        handler = dispatch.get(choice)
        if handler is None:
            print("Invalid choice. Please enter 1-10.")
            continue
        success = handler()
        print("\n--- What's Next? ---")
        print("1. Blend Again")
        print("2. Exit")
        while True:
            next_choice = input("\nEnter your choice (1-2): ").strip()
            if next_choice == '1':
                print("\n" + "=" * 60 + "\n")
                break
            elif next_choice == '2':
                print("\nThank you for using VODER! Goodbye!")
                print("Results saved to: results/")
                return
            else:
                print("Invalid choice. Please enter 1 or 2.")
