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


def interactive_cli_mode():
    from voder import (cli_tts_mode, cli_sts_mode, cli_ttm_mode, cli_se_mode,
                       cli_sfx_mode, cli_svs_mode, cli_stt_mode, cli_ss_mode)
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
        choice = input("\nEnter your choice (1-8): ").strip()
        success = False
        if choice == '1':
            success = cli_tts_mode()
        elif choice == '2':
            success = cli_sts_mode()
        elif choice == '3':
            success = cli_ttm_mode()
        elif choice == '4':
            success = cli_se_mode()
        elif choice == '5':
            success = cli_sfx_mode()
        elif choice == '6':
            success = cli_svs_mode()
        elif choice == '7':
            success = cli_stt_mode()
        elif choice == '8':
            success = cli_ss_mode()
        else:
            print("Invalid choice. Please enter 1-8.")
            continue
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
