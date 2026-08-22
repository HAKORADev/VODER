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
        '0': _eva_dlc_menu,
    }
    return _MODE_DISPATCH


def _eva_dlc_menu():
    print("\n" + "=" * 60)
    print("DLCs")
    print("=" * 60)
    print("\nAvailable DLCs:")
    print("  1. Eva — Image, Video, Chat, World generation")
    print("  2. Klarify — Upscale, Enhance, Frame Interpolation")
    choice = input("\nSelect DLC (or press Enter to go back): ").strip()
    if choice == '1':
        return _eva_mode_menu()
    elif choice == '2':
        return _klarify_mode_menu()
    return True


def _eva_mode_menu():
    print("\n" + "=" * 60)
    print("Project Eva — Select Mode")
    print("=" * 60)
    print("1. TTI — Text-to-Image (generate, edit, transparent PNG)")
    print("2. TTV — Text-to-Video (generate with audio, edit)")
    print("3. TTT — Text-to-Text / VADAR Chat (uncensored local AI)")
    print("4. TTW — Text-to-World (3D scenes, objects)")
    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == '1':
        return _eva_tti_interactive()
    elif choice == '2':
        return _eva_ttv_interactive()
    elif choice == '3':
        from voders.DLCs.eva.chat.vadar import vadar_interactive
        return vadar_interactive()
    elif choice == '4':
        return _eva_ttw_interactive()
    else:
        print("Invalid choice.")
        return True


def _eva_tti_interactive():
    print("\n--- TTI: Text-to-Image ---")
    print("1. Generate (from text description)")
    print("2. Edit (modify existing image)")
    print("3. NBG (generate transparent PNG)")
    sub = input("Select sub-mode (1-3): ").strip()

    if sub == '1':
        desc = input("Enter image description: ").strip()
        resolution = input("Resolution (e.g. 1024x1024, or press Enter for default): ").strip() or None
        from voders.DLCs.eva.image.flux2 import Flux2Wrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'image'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_tti_gen_{safe}_{ts}.png")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = Flux2Wrapper()
        try:
            w.generate(desc, out, resolution=resolution)
        finally:
            w.cleanup()
    elif sub == '2':
        inp = input("Input image path: ").strip()
        desc = input("Edit description: ").strip()
        from voders.DLCs.eva.image.flux2 import Flux2Wrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'edit'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_tti_edit_{safe}_{ts}.png")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = Flux2Wrapper()
        try:
            w.edit(inp, desc, out)
        finally:
            w.cleanup()
    elif sub == '3':
        desc = input("Enter image description (will generate transparent PNG): ").strip()
        from voders.DLCs.eva.image.flux2 import Flux2Wrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'nbg'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_tti_nbg_{safe}_{ts}.png")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = Flux2Wrapper()
        try:
            w.generate_nbg(desc, out)
        finally:
            w.cleanup()
    else:
        print("Invalid choice.")
    return True


def _eva_ttv_interactive():
    print("\n--- TTV: Text-to-Video ---")
    print("1. Generate (from text description, includes audio)")
    print("2. Edit (modify existing video)")
    sub = input("Select sub-mode (1-2): ").strip()

    if sub == '1':
        desc = input("Enter video description: ").strip()
        from voders.DLCs.eva.video.h3 import H3Wrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'video'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_ttv_gen_{safe}_{ts}.mp4")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = H3Wrapper()
        try:
            w.generate(desc, out)
        finally:
            w.cleanup()
    elif sub == '2':
        inp = input("Input video path: ").strip()
        desc = input("Edit description: ").strip()
        from voders.DLCs.eva.video.vace import VACEWrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'edit'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_ttv_edit_{safe}_{ts}.mp4")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = VACEWrapper()
        try:
            w.edit(inp, desc, out)
        finally:
            w.cleanup()
    else:
        print("Invalid choice.")
    return True


def _eva_ttw_interactive():
    print("\n--- TTW: Text-to-World ---")
    print("1. Generate (create 3D world from description)")
    print("2. Objectify (convert image to 3D object)")
    sub = input("Select sub-mode (1-2): ").strip()

    if sub == '1':
        desc = input("Enter world description: ").strip()
        from voders.DLCs.eva.world.hyworld import HYWorldWrapper
        import time, re, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe = re.sub(r'[^A-Za-z0-9_\-]', '_', desc[:100]) or 'world'
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_ttw_gen_{safe}_{ts}.glb")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = HYWorldWrapper()
        try:
            w.generate(desc, out)
        finally:
            w.cleanup()
    elif sub == '2':
        inp = input("Input image path: ").strip()
        fmt = input("Output format (glb/obj, default glb): ").strip() or "glb"
        from voders.DLCs.eva.world.trellis import Trellis2Wrapper
        import time, os
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join("results", "DLCs", "eva", f"voder_eva_ttw_objectify_{ts}.glb")
        os.makedirs(os.path.join("results", "DLCs", "eva"), exist_ok=True)
        w = Trellis2Wrapper()
        try:
            w.objectify(inp, out, output_format=fmt)
        finally:
            w.cleanup()
    else:
        print("Invalid choice.")
    return True


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
        print("0. DLCs (Eva — image/video/chat/world, Klarify — upscale/enhance/interpolate)")
        choice = input("\nEnter your choice (0-9): ").strip()
        handler = dispatch.get(choice)
        if handler is None:
            print("Invalid choice. Please enter 0-9.")
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


def _klarify_mode_menu():
    print("\n" + "=" * 60)
    print("Klarify DLC — Select Mode")
    print("=" * 60)
    print("1. Upscale (x4 then -2 LANCZOS, images and videos)")
    print("2. Enhance (denoise + deblur, images and videos)")
    print("3. Interpolate (frame interpolation, videos only)")
    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == '1':
        return _klarify_interactive('upscale')
    elif choice == '2':
        return _klarify_interactive('enhance')
    elif choice == '3':
        return _klarify_interactive('interpolate')
    else:
        print("Invalid choice.")
        return True


def _klarify_interactive(mode_name):
    inp = input("Input file path: ").strip()
    if not inp:
        print("No input file provided.")
        return True
    import time, os, re
    from voders.DLCs.klarify.klarify_engine import (
        klarify_upscale, klarify_enhance, klarify_interpolate, klarify_cleanup
    )
    results_dir = os.path.join(os.getcwd(), "results", "DLCs", "klarify")
    os.makedirs(results_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(inp)[1] or '.png'
    if mode_name == 'interpolate':
        ext = '.mp4'
    out = os.path.join(results_dir, f"voder_klarify_{mode_name}_{ts}{ext}")
    try:
        if mode_name == 'upscale':
            klarify_upscale(inp, out)
        elif mode_name == 'enhance':
            klarify_enhance(inp, out)
        elif mode_name == 'interpolate':
            multi_str = input("Interpolation multiplier (default 2): ").strip()
            multi = int(multi_str) if multi_str else 2
            klarify_interpolate(inp, out, multi=multi)
    finally:
        klarify_cleanup()
    return True
