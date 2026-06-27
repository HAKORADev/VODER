import os
import sys
import time
import gc
import traceback
import torch
import torchaudio

from voder import (
    is_youtube_url,
    download_youtube_audio,
    download_youtube_video,
    extract_audio_from_video_cli,
    _ss_resolve_input,
    _ss_run_pipeline,
)


def cli_ss_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SS Mode ---")
    print("Speakers Separator - extract all speakers from audio one by one")
    print("Pipeline: SVS voice isolation -> STT + diarization -> TSE extraction")
    print("Supports: audio files, video files, and supported platform URLs (YouTube, TikTok, Bilibili, Snapchat, Instagram, Facebook, X/Twitter)")
    print()

    while True:
        file_path = input("Enter audio/video file path or URL: ").strip()
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

    target_path = None
    while True:
        target_input = input("Enter target voice path (audio/video/URL, or Enter to skip): ").strip()
        if not target_input:
            break
        if is_youtube_url(target_input):
            target_path = target_input
            break
        if os.path.exists(target_input):
            target_path = target_input
            break
        print("Error: File not found or invalid path")

    use_overdose = False
    if not target_path:
        while True:
            overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
            if overdose_input in ['y', 'yes']:
                use_overdose = True
                break
            elif overdose_input in ['n', 'no']:
                use_overdose = False
                break
            else:
                print("Please enter Y or N")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    print()

    if target_path:
        target_audio = target_path
        if is_youtube_url(target_path):
            print("Downloading target audio from URL...")
            success_dl, error_msg, target_audio = download_youtube_audio(target_path)
            if not success_dl:
                print(f"Warning: Target download failed, using path as-is: {error_msg}")
                target_audio = target_path
        elif target_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from target video...")
            extracted = extract_audio_from_video_cli(target_path)
            if extracted:
                target_audio = extracted
            else:
                print("Warning: Could not extract audio from target video")

        audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
        if err:
            print(f"Error: {err}")
            if target_audio and target_audio != target_path and os.path.exists(target_audio):
                try:
                    os.unlink(target_audio)
                except:
                    pass
            return False

        try:
            pipeline_outputs = _ss_run_pipeline(audio_path, False, results_dir, original_name, timestamp, target_audio, use_overdose)
            if pipeline_outputs is None:
                print("SS pipeline failed")
                return False
            return True
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            return False
        finally:
            for f in cleanup_list:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
            if target_audio and target_audio != target_path and os.path.exists(target_audio):
                try:
                    os.unlink(target_audio)
                except:
                    pass
    else:
        audio_path, original_name, is_url, cleanup_list, err = _ss_resolve_input(file_path, results_dir, timestamp)
        if err:
            print(f"Error: {err}")
            return False

        try:
            pipeline_outputs = _ss_run_pipeline(audio_path, False, results_dir, original_name, timestamp, None, use_overdose)
            if pipeline_outputs is None:
                print("SS pipeline failed")
                return False
            return True
        except Exception as e:
            traceback.print_exc()
            print(f"Error: {e}")
            return False
        finally:
            for f in cleanup_list:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except Exception:
                        pass
