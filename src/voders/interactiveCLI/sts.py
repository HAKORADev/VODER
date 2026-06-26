import os
import time
import shutil
import tempfile
import gc
import torch
import torchaudio

from voder import (
    validate_file_exists,
    VIDEO_EXTENSIONS,
    extract_audio_from_video_cli,
    resolve_target_to_audio,
    svs_extract_vocals,
    svs_extract_music,
    SeedVCV1,
    SeedVCV2,
)


def cli_sts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- STS Mode ---")
    print("Convert voice from base audio to target voice")
    print()
    base_path = input("Enter base audio/video path: ").strip()
    if not validate_file_exists(base_path):
        return False
    base_is_video = os.path.splitext(base_path)[1].lower() in VIDEO_EXTENSIONS
    base_original = base_path
    temp_base_extracted = None
    if base_is_video:
        print("Extracting audio from video...")
        temp_base_extracted = extract_audio_from_video_cli(base_path)
        if not temp_base_extracted:
            print("Error: Could not extract audio from video")
            return False
        base_path = temp_base_extracted
    print()
    target_path = input("Enter target voice audio/video path or URL: ").strip()
    if not target_path:
        print("Error: No target path provided")
        return False
    resolved_target, _target_cleanup = resolve_target_to_audio(target_path)
    if not resolved_target:
        return False
    target_path = resolved_target
    print()
    no_music = False
    while True:
        music_input = input("Are the inputs musical? (Y/N): ").strip().lower()
        if music_input in ['y', 'yes']:
            is_music = True
            break
        elif music_input in ['n', 'no']:
            is_music = False
            break
        else:
            print("Please enter Y or N")
    if not is_music:
        while True:
            nomusic_input = input("Output voice only without music? (Y/N): ").strip().lower()
            if nomusic_input in ['y', 'yes']:
                no_music = True
                break
            elif nomusic_input in ['n', 'no']:
                no_music = False
                break
            else:
                print("Please enter Y or N")
    if is_music:
        if base_is_video:
            print("Error: Base input must be audio for MSTS mode")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("\nLoading Seed-VC v1 model (44.1kHz)...")
        seed_vc = SeedVCV1()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC v1 model")
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        print("Extracting music from source...")
        base_music = svs_extract_music(base_path)
        _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        clean_vocal_target = svs_extract_vocals(target_path)
        _target_cleanup.append(clean_vocal_target)
        print("Resampling inputs to 44100Hz...")
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 44100:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 44100:
            resampler_target = torchaudio.transforms.Resample(sr_target, 44100)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 44100)
            torchaudio.save(temp_target.name, waveform_target, 44100)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_44k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            else:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_m_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    else:
        print("Extracting vocals from source...")
        base_vocals = svs_extract_vocals(base_path)
        _target_cleanup.append(base_vocals)
        base_music = None
        if not no_music:
            print("Extracting music from source...")
            base_music = svs_extract_music(base_path)
            _target_cleanup.append(base_music)
        print("Extracting clean vocals from target...")
        clean_vocal_target = svs_extract_vocals(target_path)
        _target_cleanup.append(clean_vocal_target)
        print("\nLoading Seed-VC v2 model...")
        seed_vc = SeedVCV2()
        if seed_vc.model is None:
            print("Error: Failed to load Seed-VC model")
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        print("Resampling inputs to 22050Hz...")
        waveform_vocals, sr_vocals = torchaudio.load(base_vocals)
        if sr_vocals != 22050:
            resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 22050)
            waveform_vocals = resampler_vocals(waveform_vocals)
        waveform_target, sr_target = torchaudio.load(clean_vocal_target)
        if sr_target != 22050:
            resampler_target = torchaudio.transforms.Resample(sr_target, 22050)
            waveform_target = resampler_target(waveform_target)
        temp_vocals = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_target = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_output_22k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            torchaudio.save(temp_vocals.name, waveform_vocals, 22050)
            torchaudio.save(temp_target.name, waveform_target, 22050)
            print("Converting voice...")
            success = seed_vc.convert(
                source_path=temp_vocals.name,
                reference_path=temp_target.name,
                output_path=temp_output_22k.name
            )
            if not success:
                print("Error: Voice conversion failed")
                return False
            print("Upsampling output to 44100Hz...")
            waveform_out, sr_out = torchaudio.load(temp_output_22k.name)
            if sr_out != 44100:
                resampler_out = torchaudio.transforms.Resample(sr_out, 44100)
                waveform_out = resampler_out(waveform_out)
            temp_output_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            torchaudio.save(temp_output_44k.name, waveform_out, 44100)
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if not no_music and base_music and os.path.exists(base_music):
                print("Mixing converted vocals with source music...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_output_44k.name}" -i "{base_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_output_44k.name, output_path)
            elif base_is_video:
                print("Merging converted audio with video...")
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.mp4")
                ret = os.system(f'ffmpeg -y -i "{base_original}" -i "{temp_output_44k.name}" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Error: Failed to merge audio with video")
                    return False
            else:
                output_path = os.path.join(results_dir, f"voder_sts_{timestamp}.wav")
                shutil.copy(temp_output_44k.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_vocals.name, temp_target.name, temp_output_22k.name, temp_output_44k.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            if temp_base_extracted and os.path.exists(temp_base_extracted):
                os.remove(temp_base_extracted)
            for f in _target_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
