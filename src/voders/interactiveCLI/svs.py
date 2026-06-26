import os
import sys
import time
import gc
import traceback
import torch
import torchaudio

from voder import (
    is_youtube_url,
    download_youtube_video,
    SVS_DIR,
)


def cli_svs_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SVS Mode ---")
    print("Song Voice Separate - extract vocals or instrumental from a song")
    print("Model: BS-RoFormer Resurrection (best single-pass vocals SDR + instrumental SDR)")
    print()

    while True:
        file_path = input("Enter song, video file path, or supported platform URL: ").strip()
        if not file_path:
            print("Error: File path cannot be empty. Please try again.")
            continue
        if is_youtube_url(file_path):
            break
        if os.path.exists(file_path):
            try:
                torchaudio.load(file_path)
                break
            except Exception:
                video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
                if os.path.splitext(file_path)[1].lower() in video_exts:
                    break
                print("Error: Could not read audio file. Please try again.")
        else:
            print(f"Error: File not found: {file_path}")

    while True:
        choice = input("Separate what? 1: Extract voice  2: Extract music: ").strip()
        if choice == '1':
            stem = 'voice'
            break
        elif choice == '2':
            stem = 'music'
            break
        else:
            print("Error: Please enter 1 or 2.")

    stem_label = 'vocals' if stem == 'voice' else 'instruments'
    if is_youtube_url(file_path):
        print(f"\nExtracting {stem_label} from: {file_path}")
    else:
        print(f"\nExtracting {stem_label} from: {os.path.basename(file_path)}")
    print("Loading BS-RoFormer Resurrection model (first run downloads ~390MB)...")

    _voder_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bs_roformer_lib = os.path.join(_voder_src, 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)

    if _voder_src not in sys.path:
        sys.path.insert(0, _voder_src)

    from bs_roformer import BSRoformerSeparator
    separator = BSRoformerSeparator(SVS_DIR)
    separator.ensure_model(stem=stem)
    if stem == 'voice' and separator.vocals_model is None:
        print("Error: Failed to load vocals model")
        return False
    if stem == 'music' and separator.inst_model is None:
        print("Error: Failed to load instrumental model")
        return False

    try:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        suffix = 'vocals' if stem == 'voice' else 'instruments'

        video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv', '.m4v', '.ts', '.mts'}
        downloaded_video = None
        actual_file_path = file_path
        is_url = is_youtube_url(file_path)

        if is_url:
            downloaded_video, video_title = download_youtube_video(file_path, results_dir)
            if downloaded_video is None:
                print(f'Error: {video_title}')
                return False
            actual_file_path = downloaded_video
            original_name = video_title.replace(' ', '_').replace('/', '_')[:50]
            is_video = True
        else:
            original_name = os.path.splitext(os.path.basename(file_path))[0]
            input_ext = os.path.splitext(file_path)[1].lower()
            is_video = input_ext in video_exts

        output_filename = f'voder_svs_{original_name}_{timestamp}_{suffix}.mp4' if is_video else f'voder_svs_{original_name}_{timestamp}_{suffix}.wav'
        output_path = os.path.join(results_dir, output_filename)

        temp_audio = None
        if is_video:
            print('Video detected, extracting audio...')
            temp_audio = os.path.join(results_dir, f'_svs_temp_{timestamp}.wav')
            ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -vn -acodec pcm_s16le -ar 44100 -ac 2 "{temp_audio}" 2>/dev/null')
            if ret != 0 or not os.path.exists(temp_audio):
                print('Error: Failed to extract audio from video')
                if downloaded_video and os.path.exists(downloaded_video):
                    os.remove(downloaded_video)
                return False

        if is_video:
            temp_wav = os.path.join(results_dir, f'_svs_temp_{timestamp}_{suffix}.wav')
            success = separator.separate(temp_audio, stem, temp_wav)
            if success:
                print('Merging separated audio back into video...')
                ret = os.system(f'ffmpeg -y -i "{actual_file_path}" -i "{temp_wav}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print('Error: Failed to merge audio with video')
                    success = False
                else:
                    os.remove(temp_wav)
                    if temp_audio and os.path.exists(temp_audio):
                        os.remove(temp_audio)
            else:
                if temp_audio and os.path.exists(temp_audio):
                    os.remove(temp_audio)
        else:
            success = separator.separate(actual_file_path, stem, output_path)

        if downloaded_video and os.path.exists(downloaded_video):
            os.remove(downloaded_video)

        if success:
            print(f'\nSuccess! Output saved to: {output_path}')
        else:
            print('Error: Separation failed')
        return success
    except Exception as e:
        traceback.print_exc()
        print(f'Error: {e}')
        return False
    finally:
        separator.cleanup()
        del separator
        separator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
