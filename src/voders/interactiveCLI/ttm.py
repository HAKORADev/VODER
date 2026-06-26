import os
import time
import shutil
import tempfile
import gc
import torch
import torchaudio

from voder import (
    _validate_text_language,
    SUPPORTED_ACESTEP_LANGS,
    is_youtube_url,
    resolve_target_to_audio,
    svs_extract_vocals,
    svs_extract_music,
    AceStepWrapper,
    SeedVCV1,
)


def cli_ttm_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTM Mode ---")
    print("Generate music from lyrics and style")
    print()
    use_overdose = False
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
    use_vc = False
    while True:
        vc_input = input("Want to clone a voice? (Y/N): ").strip().lower()
        if vc_input in ['y', 'yes']:
            use_vc = True
            break
        elif vc_input in ['n', 'no']:
            use_vc = False
            break
        else:
            print("Please enter Y or N")
    if use_vc:
        while True:
            print("Enter song lyrics (use \\n for new lines):")
            lyrics = input("> ").strip()
            if not lyrics:
                print("Error: No lyrics provided")
                return False
            lyrics = lyrics.replace('\\n', '\n')
            lyrics_valid, _ = _validate_text_language(lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
            if lyrics_valid:
                break
            print("Try again with a supported language")
        print()
        print("Enter style prompt (use \\n for new lines, e.g., 'upbeat pop, female vocals'):")
        style = input("> ").strip()
        if not style:
            print("Error: No style prompt provided")
            return False
        style = style.replace('\\n', '\n')
        print()
        print("Enter duration in seconds (10-300, where 300 = 5 minutes max):")
        while True:
            try:
                duration = int(input("> ").strip())
                if 10 <= duration <= 300:
                    break
                else:
                    print("Error: Duration must be between 10 and 300 seconds")
            except ValueError:
                print("Error: Please enter a valid number")
        print()
        clone_input = input("Enter source to clone from (audio/video/URL): ").strip()
        if not clone_input:
            print("Error: No clone source provided")
            return False
        while not (os.path.exists(clone_input) or is_youtube_url(clone_input)):
            print(f"Error: Clone source not found: {clone_input}")
            clone_input = input("Enter source to clone from (audio/video/URL): ").strip()
            if not clone_input:
                print("Error: No clone source provided")
                return False
        _vc_cleanup = []
        resolved_audio, cleanup = resolve_target_to_audio(clone_input)
        if resolved_audio is None:
            print("Error: Could not resolve clone source")
            return False
        _vc_cleanup.extend(cleanup)
        clean_vocal = svs_extract_vocals(resolved_audio)
        if clean_vocal != resolved_audio and clean_vocal not in _vc_cleanup:
            _vc_cleanup.append(clean_vocal)
        if resolved_audio not in _vc_cleanup and resolved_audio != clean_vocal:
            _vc_cleanup.append(resolved_audio)
        print("\nLoading ACE-Step model...")
        ace_step = AceStepWrapper(use_overdose=use_overdose)
        if ace_step.handler is None:
            print("Error: Failed to load ACE-Step model")
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            return False
        temp_ttm_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_ttm_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_clone_44k = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_vc_output = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            print(f"Generating music ({duration}s duration)...")
            success = ace_step.generate(
                lyrics=lyrics,
                style_prompt=style,
                output_path=temp_ttm_output.name,
                duration=duration
            )
            if not success:
                print("Error: Music generation failed")
                return False
            print("Extracting vocals from TTM output...")
            ttm_vocals = svs_extract_vocals(temp_ttm_output.name)
            if ttm_vocals and ttm_vocals != temp_ttm_output.name:
                _vc_cleanup.append(ttm_vocals)
            else:
                ttm_vocals = temp_ttm_output.name
            print("Extracting music from TTM output...")
            ttm_music = svs_extract_music(temp_ttm_output.name)
            if ttm_music and ttm_music != temp_ttm_output.name:
                _vc_cleanup.append(ttm_music)
            else:
                ttm_music = None
            print("Resampling TTM vocals to 44100Hz...")
            waveform_vocals, sr_vocals = torchaudio.load(ttm_vocals)
            if sr_vocals != 44100:
                resampler_vocals = torchaudio.transforms.Resample(sr_vocals, 44100)
                waveform_vocals = resampler_vocals(waveform_vocals)
            torchaudio.save(temp_ttm_44k.name, waveform_vocals, 44100)
            print("Resampling clone voice to 44100Hz...")
            waveform_clone, sr_clone = torchaudio.load(clean_vocal)
            if sr_clone != 44100:
                resampler_clone = torchaudio.transforms.Resample(sr_clone, 44100)
                waveform_clone = resampler_clone(waveform_clone)
            torchaudio.save(temp_clone_44k.name, waveform_clone, 44100)
            print("Clearing ACE-Step from memory...")
            del ace_step
            ace_step = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Loading Seed-VC v1 model...")
            seed_vc = SeedVCV1()
            if seed_vc.model is None:
                print("Error: Failed to load Seed-VC v1 model")
                return False
            print("Converting voice...")
            vc_success = seed_vc.convert(
                source_path=temp_ttm_44k.name,
                reference_path=temp_clone_44k.name,
                output_path=temp_vc_output.name
            )
            if not vc_success:
                print("Error: Voice conversion failed")
                return False
            del seed_vc
            seed_vc = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if ttm_music:
                print("Mixing converted vocals with TTM music...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                ret = os.system(f'ffmpeg -y -i "{temp_vc_output.name}" -i "{ttm_music}" -filter_complex "[0:a]volume=1.0[vc];[1:a]volume=1.0[music];[vc][music]amix=inputs=2:duration=longest" "{output_path}" 2>/dev/null')
                if ret != 0 or not os.path.exists(output_path):
                    print("Warning: Mixing failed, saving converted vocals only")
                    shutil.copy(temp_vc_output.name, output_path)
            else:
                print("Saving output...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_ttm_vc_{timestamp}.wav")
                shutil.copy(temp_vc_output.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for temp_file in [temp_ttm_output.name, temp_ttm_44k.name, temp_clone_44k.name, temp_vc_output.name]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            for f in _vc_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
    while True:
        print("Enter song lyrics (use \\n for new lines):")
        lyrics = input("> ").strip()
        if not lyrics:
            print("Error: No lyrics provided")
            return False
        lyrics = lyrics.replace('\\n', '\n')
        lyrics_valid, _ = _validate_text_language(lyrics, SUPPORTED_ACESTEP_LANGS, "TTM")
        if lyrics_valid:
            break
        print("Try again with a supported language")
    print()
    print("Enter style prompt (use \\n for new lines, e.g., 'upbeat pop, female vocals'):")
    style = input("> ").strip()
    if not style:
        print("Error: No style prompt provided")
        return False
    style = style.replace('\\n', '\n')
    print()
    print("Enter duration in seconds (10-300, where 300 = 5 minutes max):")
    while True:
        try:
            duration = int(input("> ").strip())
            if 10 <= duration <= 300:
                break
            else:
                print("Error: Duration must be between 10 and 300 seconds")
        except ValueError:
            print("Error: Please enter a valid number")
    _ttm_cleanup = []
    reference_audio = None
    print()
    ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
    while ref_input:
        if not os.path.exists(ref_input) and not is_youtube_url(ref_input):
            print(f"Error: Reference target not found: {ref_input}")
            ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
            continue
        while True:
            ref_choice = input("Reference type: 1 for voice, 2 for music: ").strip()
            if ref_choice == '1':
                ref_type = 'voice'
                break
            elif ref_choice == '2':
                ref_type = 'music'
                break
            else:
                print("Error: Please enter 1 (voice) or 2 (music)")
        resolved_audio, cleanup = resolve_target_to_audio(ref_input)
        if resolved_audio is None:
            print("Error: Could not resolve reference target")
            ref_input = input("Enter reference audio path (audio/video/URL, or press Enter to skip): ").strip()
            continue
        _ttm_cleanup.extend(cleanup)
        if ref_type == 'voice':
            processed = svs_extract_vocals(resolved_audio)
        else:
            processed = svs_extract_music(resolved_audio)
        if processed != resolved_audio and processed not in _ttm_cleanup:
            _ttm_cleanup.append(processed)
        if resolved_audio not in _ttm_cleanup and resolved_audio != processed:
            _ttm_cleanup.append(resolved_audio)
        reference_audio = processed
        break
    print("\nLoading ACE-Step model...")
    ace_step = AceStepWrapper(use_overdose=use_overdose)
    if ace_step.handler is None:
        print("Error: Failed to load ACE-Step model")
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
        return False
    try:
        print(f"Generating music ({duration}s duration)...")
        if reference_audio:
            print(f"Using reference audio: {reference_audio}")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(results_dir, f"voder_ttm_{timestamp}.wav")
        success = ace_step.generate(
            lyrics=lyrics,
            style_prompt=style,
            output_path=output_path,
            duration=duration,
            reference_audio=reference_audio
        )
        if not success:
            print("Error: Music generation failed")
            return False
        print(f"\n✓ Success! Output saved to: {output_path}")
        del ace_step
        ace_step = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    finally:
        for f in _ttm_cleanup:
            if f and os.path.exists(f):
                try:
                    os.unlink(f)
                except:
                    pass
