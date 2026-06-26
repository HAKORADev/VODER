import os
import torchaudio

from voder import (
    VIDEO_EXTENSIONS,
    oneline_se,
)


def cli_se_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SE Mode ---")
    print("Sound Enhancement - denoise, dereverb, restore audio")
    print("For sub-modes (voice/sr/sr music/sr voice/sr voice music), use oneline mode.")
    print()

    while True:
        file_path = input("Enter audio/video file path: ").strip()
        if os.path.exists(file_path):
            ext = os.path.splitext(file_path)[1].lower()
            is_valid_audio = False
            is_video = ext in VIDEO_EXTENSIONS
            if is_video:
                break
            try:
                torchaudio.load(file_path)
                is_valid_audio = True
                break
            except Exception:
                pass
            if not is_valid_audio:
                print("Error: Unsupported or corrupt file format.")
        else:
            print("Error: File not found. Please try again.")

    params = {
        'files': [file_path],
        'result_path': None,
        'se_sub': None,
        'se_blend': False,
    }

    return oneline_se(params)
