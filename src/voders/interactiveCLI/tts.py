import os
import time
import subprocess
import tempfile
import shutil
import gc
import torch

from voder import (
    is_youtube_url,
    download_youtube_audio,
    extract_audio_from_video_cli,
    svs_extract_vocals,
    svs_extract_music,
    VibeVoiceASR,
    WhisperSTT,
    _validate_text_language,
    SUPPORTED_FISH_LANGS,
    SUPPORTED_TTS_LANGUAGES,
    _parse_multi_refs,
    _resolve_multi_refs,
    resolve_target_to_audio,
    _resolve_voice_ref,
    _load_voice_prompt,
    _transcribe_for_fish_ref,
    _transcribe_for_qwen_ref,
    _parse_script_directives,
    _parse_directives_for_line,
    _validate_duration_directive,
    _parse_music_level_spec,
    _assemble_enhanced_dialogue,
    _generate_music_and_mix,
    _extract_speakers_for_subtitles,
    _tts_extract_voice,
    _load_fish_voice,
    validate_dialogue_source_file,
    analyze_dialogue_source,
    platform_name,
    QwenTTS,
    QwenTTSVoiceDesign,
    FishTTS,
    AceStepWrapper,
    SeedVCV2,
)


def cli_tts_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- TTS Mode ---")

    modify_speech = input("Want to modify speech? (Y/N): ").strip().lower()
    if modify_speech in ['y', 'yes']:
        while True:
            print("\nEnter the path to your audio/video source (file path or supported platform URL):")
            source_path = input("> ").strip()
            if not source_path:
                print("Error: No path provided")
                continue
            break

        _ms_cleanup = []
        audio_path = source_path
        needs_youtube = is_youtube_url(source_path)

        if needs_youtube:
            print("Downloading audio from YouTube...")
            ok, err, dl_path = download_youtube_audio(source_path)
            if not ok:
                print(f"Error: {err}")
                return False
            audio_path = dl_path
            _ms_cleanup.append(dl_path)
        elif source_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print("Extracting audio from video...")
            audio_path = extract_audio_from_video_cli(source_path)
            if not audio_path:
                print("Error: Failed to extract audio from video")
                return False
            _ms_cleanup.append(audio_path)
        elif not os.path.exists(source_path):
            print(f"Error: File not found: {source_path}")
            return False

        print("Isolating vocals via SVS...")
        clean_vocal = svs_extract_vocals(audio_path)
        if clean_vocal and clean_vocal != audio_path:
            _ms_cleanup.append(clean_vocal)
        else:
            clean_vocal = audio_path

        ms_overdose = False
        while True:
            overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
            if overdose_input in ['y', 'yes']:
                ms_overdose = True
                break
            elif overdose_input in ['n', 'no']:
                ms_overdose = False
                break
            else:
                print("Please enter Y or N")

        ms_extreme = False
        while True:
            extreme_input = input("Enable extreme? (Y/N): ").strip().lower()
            if extreme_input in ['y', 'yes']:
                ms_extreme = True
                break
            elif extreme_input in ['n', 'no']:
                ms_extreme = False
                break
            else:
                print("Please enter Y or N")

        if ms_overdose:
            print("Loading VibeVoice ASR (overdose mode)...")
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                asr.cleanup()
                del asr
                ms_overdose = False
            else:
                try:
                    text = asr.transcribe_plain_text(clean_vocal)
                except Exception as e:
                    print(f"VibeVoice transcription error: {e}")
                    text = ""
                del asr
                asr = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not text or not text.strip():
                    print("Error: No speech detected (VibeVoice)")
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                text = text.strip()
                print(f"\nTranscribed text ({len(text)} chars):")
                display_text = text.replace('\n', '\\n').replace('\r', '\\r')
                print(display_text)
                print()

        if not ms_overdose:
            print("\nLoading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print("Transcribing audio...")
            result = stt.transcribe(clean_vocal)
            del stt
            stt = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not result:
                print("Error: Transcription failed")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            text = result.get("text", "").strip()
            if not text:
                print("Error: No speech detected")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False

            print(f"\nTranscribed text ({len(text)} chars):")
            display_text = text.replace('\n', '\\n').replace('\r', '\\r')
            print(display_text)
            print()

        while True:
            edited_text = input("Edit text (or press Enter to keep as is): ").strip()
            if edited_text:
                text = edited_text.replace('\\n', '\n')
            if not text:
                print("Error: No text to synthesize")
                for f in _ms_cleanup:
                    if f and os.path.exists(f):
                        try:
                            os.unlink(f)
                        except:
                            pass
                return False
            ms_lang_set = SUPPORTED_FISH_LANGS if ms_extreme else set(SUPPORTED_TTS_LANGUAGES.keys())
            ms_lang_ctx = "TTS (extreme)" if ms_extreme else "TTS"
            ms_valid, ms_detected = _validate_text_language(text, ms_lang_set, ms_lang_ctx)
            if ms_valid:
                break
            print("Try again with a supported language")

        use_source = input("Want to use source audio as voice reference? (Y/N): ").strip().lower()
        voice_ref = None
        ms_sts = False
        ms_sts_ref = None
        if use_source in ['y', 'yes']:
            voice_ref = clean_vocal
        else:
            while True:
                print("Enter voice reference path (prefix with sts: for enhanced voice conversion, use (path1)(path2) for multi-ref):")
                ref_path = input("> ").strip()
                if not ref_path:
                    print("Error: No path provided")
                    continue
                break
            if ref_path.lower().startswith('sts:'):
                ms_sts = True
                ref_path = ref_path[4:]
            multi = _parse_multi_refs(ref_path)
            if multi:
                voice_ref = _resolve_multi_refs(multi, _ms_cleanup)
                if not voice_ref:
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
            else:
                resolved_ref, _ref_cl = resolve_target_to_audio(ref_path)
                if not resolved_ref:
                    for f in _ms_cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
                    return False
                _ms_cleanup.extend(_ref_cl)
                voice_ref = svs_extract_vocals(resolved_ref)
                if voice_ref and voice_ref != resolved_ref:
                    _ms_cleanup.append(voice_ref)
                if resolved_ref not in _ms_cleanup and resolved_ref != voice_ref:
                    _ms_cleanup.append(resolved_ref)
            if ms_sts and voice_ref:
                ms_sts_ref = voice_ref

        preserve_nonvocals = False
        preserve_input = input("Preserve non-vocals? (Y/N): ").strip().lower()
        if preserve_input in ['y', 'yes']:
            preserve_nonvocals = True

        ms_music_track = None
        if preserve_nonvocals:
            ms_music_track = svs_extract_music(audio_path)
            if ms_music_track and ms_music_track != audio_path:
                _ms_cleanup.append(ms_music_track)
            else:
                ms_music_track = None

        try:
            if ms_extreme:
                print("\nLoading Fish-S2Pro model (extreme)...")
                tts = FishTTS()
                if not tts.ensure_model():
                    print("Error: Fish-S2Pro model failed to load")
                    return False
                print("Transcribing voice reference...")
                ref_text = _transcribe_for_fish_ref(voice_ref)
                print("Encoding voice (extreme)...")
                success = tts.encode_voice(voice_ref, ref_text=ref_text)
                if not success:
                    print("Error: Voice encoding failed")
                    return False
                print("Synthesizing speech (extreme)...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_ms_extreme_{timestamp}.wav")
                success = tts.synthesize(text, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False
            else:
                print("\nLoading Qwen-TTS model...")
                tts = QwenTTS()
                print("Extracting voice characteristics...")
                ref_text = _transcribe_for_qwen_ref(voice_ref)
                success = tts.extract_voice(voice_ref, ref_text=ref_text if ref_text else None)
                if not success:
                    print("Error: Voice extraction failed")
                    return False
                print("Synthesizing speech...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_ms_{timestamp}.wav")
                success = tts.synthesize(text, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False

            del tts
            tts = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not ms_extreme and ms_sts and ms_sts_ref and os.path.exists(ms_sts_ref):
                print("\nRunning STS voice conversion pass (Seed-VC v2 non-mimic)...")
                vc = SeedVCV2()
                if vc.model is None:
                    print("Warning: Seed-VC v2 model failed to load, skipping STS pass")
                else:
                    svs_out = svs_extract_vocals(output_path)
                    if svs_out and svs_out != output_path:
                        _ms_cleanup.append(svs_out)
                        vc_input = svs_out
                    else:
                        vc_input = output_path
                    try:
                        sts_timestamp = time.strftime("%Y%m%d_%H%M%S")
                        sts_output = os.path.join(results_dir, f"voder_tts_ms_sts_{sts_timestamp}.wav")
                        sts_success = vc.convert(vc_input, ms_sts_ref, sts_output)
                        if sts_success:
                            print(f"✓ STS-converted output saved to: {sts_output}")
                            output_path = sts_output
                        else:
                            print("Warning: STS pass failed, using standard output")
                    finally:
                        del vc
                        vc = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if ms_music_track and os.path.exists(ms_music_track):
                print("\nBlending voice output with music track...")
                blend_timestamp = time.strftime("%Y%m%d_%H%M%S")
                blend_output = os.path.join(results_dir, f"voder_tts_ms_music_{blend_timestamp}.wav")
                blend_cmd = [
                    'ffmpeg', '-i', output_path, '-i', ms_music_track,
                    '-filter_complex', '[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=0[out]',
                    '-map', '[out]', '-y', blend_output
                ]
                blend_result = subprocess.run(blend_cmd, capture_output=True, text=True)
                if blend_result.returncode == 0 and os.path.exists(blend_output):
                    print(f"✓ Blended output saved to: {blend_output}")
                    output_path = blend_output
                else:
                    print("Warning: Music blending failed, voice-only output preserved")

            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        finally:
            for f in _ms_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass

    use_overdose = False
    overdose_input = input("Enable overdose? (Y/N): ").strip().lower()
    if overdose_input in ['y', 'yes']:
        use_overdose = True

    use_extreme = False
    extreme_input = input("Enable extreme? (Y/N): ").strip().lower()
    if extreme_input in ['y', 'yes']:
        use_extreme = True

    print("\nDo you have a dialogue source file? (audio/video/txt/image)")
    print("Press Y to provide a file, or N to enter manually")
    has_source = input("> ").strip().lower()

    dialogue_items = None
    mode_detected = None
    resolved_audio_path = None
    _dialogue_speaker_extraction = None

    if has_source in ['y', 'yes']:
        while True:
            print("\nEnter the path to your dialogue source (file path or supported platform URL):")
            file_path = input("> ").strip()
            if not file_path:
                print("Error: No file path provided")
                continue

            success, msg, items = validate_dialogue_source_file(file_path)
            if not success:
                print(f"Error: {msg}")
                retry = input("Try another source? (Y/N): ").strip().lower()
                if retry not in ['y', 'yes']:
                    return False
                continue

            if msg == "txt":
                dialogue_items = []
                for item in items:
                    if len(item) == 3:
                        dialogue_items.append((item[0], item[1], item[2], {}))
                    else:
                        dialogue_items.append(item)
                mode_detected = 'dialogue' if len(dialogue_items) > 1 or (len(dialogue_items) == 1 and dialogue_items[0][1] != 'text') else 'single'
                break
            elif msg == "image":
                print(f"\nAnalyzing image: {os.path.basename(file_path)}...")
                success, error_msg, items, _audio_path, _spk_ext = analyze_dialogue_source(file_path, source_type="image", use_overdose=use_overdose)
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'
                if _spk_ext:
                    _dialogue_speaker_extraction = _spk_ext

                print(f"\nDetected {len(items)} speaker(s):")
                for item in dialogue_items:
                    speaker_num = item[1]
                    content = item[2]
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            elif msg == "url":
                platform_id = items
                pname = platform_name(platform_id)
                print(f"\nProcessing {pname} video...")
                success, error_msg, items, _audio_path, _spk_ext = analyze_dialogue_source(file_path, source_type="url", use_overdose=use_overdose)
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'
                if _audio_path:
                    resolved_audio_path = _audio_path
                if _spk_ext:
                    _dialogue_speaker_extraction = _spk_ext

                print(f"\nDetected {len(items)} speaker(s):")
                for item in dialogue_items:
                    speaker_num = item[1]
                    content = item[2]
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
            else:
                print(f"\nAnalyzing {os.path.basename(file_path)}...")
                success, error_msg, items, _audio_path, _spk_ext = analyze_dialogue_source(file_path, source_type="audio", use_overdose=use_overdose)
                if not success:
                    print(f"Error: {error_msg}")
                    retry = input("Try another source? (Y/N): ").strip().lower()
                    if retry not in ['y', 'yes']:
                        return False
                    continue

                dialogue_items = items
                mode_detected = 'dialogue' if len(items) > 1 else 'single'
                if _audio_path:
                    resolved_audio_path = _audio_path
                if _spk_ext:
                    _dialogue_speaker_extraction = _spk_ext

                print(f"\nDetected {len(items)} speaker(s):")
                for item in dialogue_items:
                    speaker_num = item[1]
                    content = item[2]
                    preview = content[:50] + "..." if len(content) > 50 else content
                    print(f"  {speaker_num}: {preview}")
                break
    else:
        print("\nEnter script lines. Use format 'Character: text' for dialogue, or plain text for single speech.")
        print("Empty line finishes script entry.")
        lines = []
        while True:
            line = input("> ").strip()
            if not line:
                break
            has_colon = ':' in line
            if mode_detected is None:
                mode_detected = 'dialogue' if has_colon else 'single'
            else:
                if (mode_detected == 'dialogue' and not has_colon) or (mode_detected == 'single' and has_colon):
                    print("Error: Inconsistent format. All lines must be either plain text (single mode) or contain 'Character: text' (dialogue mode).")
                    return False
            lines.append(line)

        if not lines:
            print("Error: No script provided")
            return False

        lines = [l.replace('\\n', '\n') for l in lines]

        tts_lang_set = SUPPORTED_FISH_LANGS if use_extreme else set(SUPPORTED_TTS_LANGUAGES.keys())
        tts_lang_ctx = "TTS (extreme)" if use_extreme else "TTS"
        all_script_text = " ".join(l.split(':', 1)[1].strip() if ':' in l else l for l in lines)
        tts_valid, _ = _validate_text_language(all_script_text, tts_lang_set, tts_lang_ctx)
        if not tts_valid:
            print("Try again with a supported language")
            return False

        if mode_detected == 'single':
            script = "\n".join(lines)
            print("Enter voice prompt (or audio/video/URL path to clone a voice, or trained voice name):")
            voice_prompt = input("> ").strip()
            if not voice_prompt:
                print("Error: No voice prompt provided")
                return False
            trained_file = _resolve_voice_ref(voice_prompt)
            if trained_file:
                print(f"Loading trained voice from: {trained_file}")
                voice_items = _load_voice_prompt(trained_file)
                if voice_items is None:
                    print(f"Error: Failed to load trained voice: {trained_file}")
                    return False
                print("Loading Qwen-TTS model...")
                tts = QwenTTS()
                if tts.model is None:
                    print("Error: Failed to load Qwen-TTS model")
                    return False
                tts.voice_prompt = voice_items
                print("Generating speech with trained voice...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts.synthesize(script, output_path)
                if not success:
                    print("Error: Synthesis failed")
                    return False
                print(f"\n✓ Success! Output saved to: {output_path}")
                del tts
                return True
            if os.path.exists(voice_prompt) or is_youtube_url(voice_prompt):
                ref_paths = [voice_prompt]
                while True:
                    more = input("Additional reference? (path/URL, or Enter to finish): ").strip()
                    if not more:
                        break
                    if os.path.exists(more) or is_youtube_url(more):
                        ref_paths.append(more)
                    else:
                        print(f"Warning: File not found: {more}, skipping")
                _cleanup = []
                try:
                    if len(ref_paths) > 1:
                        clean_vocal = _resolve_multi_refs(ref_paths, _cleanup)
                        if not clean_vocal:
                            return False
                    else:
                        resolved_audio, _cl = resolve_target_to_audio(voice_prompt)
                        if not resolved_audio:
                            return False
                        _cleanup.extend(_cl)
                        clean_vocal = svs_extract_vocals(resolved_audio)
                        if clean_vocal and clean_vocal != resolved_audio:
                            _cleanup.append(clean_vocal)
                        if resolved_audio not in _cleanup and resolved_audio != clean_vocal:
                            _cleanup.append(resolved_audio)
                    print("Loading Qwen-TTS model...")
                    tts = QwenTTS()
                    print("Extracting voice characteristics...")
                    ref_text = _transcribe_for_qwen_ref(clean_vocal)
                    success = tts.extract_voice(clean_vocal, ref_text=ref_text if ref_text else None)
                    if not success:
                        print("Error: Voice extraction failed")
                        return False
                    print("Generating speech with cloned voice...")
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                    success = tts.synthesize(script, output_path)
                    if not success:
                        print("Error: Synthesis failed")
                        return False
                    print(f"\n✓ Success! Output saved to: {output_path}")
                    del tts
                    tts = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    return True
                finally:
                    for f in _cleanup:
                        if f and os.path.exists(f):
                            try:
                                os.unlink(f)
                            except:
                                pass
            else:
                print("\nLoading Qwen-TTS VoiceDesign model...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False
                print("Generating speech...")
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(results_dir, f"voder_tts_{timestamp}.wav")
                success = tts_design.synthesize(script, voice_prompt, output_path)
                if not success:
                    print("Error: VoiceDesign synthesis failed")
                    return False
                print(f"\n✓ Success! Output saved to: {output_path}")
                del tts_design
                tts_design = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return True
        else:
            dialogue_items = []
            for i, line in enumerate(lines, start=1):
                if ':' not in line:
                    print(f"Error: Invalid dialogue line (missing ':'): {line}")
                    return False
                char, text = line.split(':', 1)
                char = char.strip()
                text = text.strip().replace('\\n', '\n')
                if not char:
                    print(f"Error: Empty character in line: {line}")
                    return False
                if char.lower() == 'sfx' and not text:
                    print(f"Error: Empty SFX prompt in line: {line}")
                    return False
                clean_text, directives_raw = _parse_script_directives(text)
                parsed_directives, errors = _parse_directives_for_line(directives_raw)
                if errors:
                    print(f"Error in line {i}: {'; '.join(errors)}")
                    print("  Please re-enter this line.")
                    while True:
                        retry_line = input("> ").strip()
                        if not retry_line:
                            print("Error: Line cannot be empty. Please try again.")
                            continue
                        if ':' not in retry_line:
                            print("Error: Line must contain ':'. Please try again.")
                            continue
                        rchar, rtext = retry_line.split(':', 1)
                        rchar = rchar.strip()
                        rtext = rtext.strip()
                        if rchar.lower() != char.lower():
                            print(f"Error: Character must be '{char}'. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and not rtext:
                            print("Error: SFX prompt cannot be empty. Please try again.")
                            continue
                        rclean_text, rdirectives_raw = _parse_script_directives(rtext)
                        rparsed_directives, rerrors = _parse_directives_for_line(rdirectives_raw)
                        if rerrors:
                            print(f"Error: {'; '.join(rerrors)}. Please try again.")
                            continue
                        if rchar.lower() == 'sfx' and rparsed_directives.get('duration') is None:
                            print("Error: SFX line requires /duration:nn (1-30). Please try again.")
                            continue
                        if not rclean_text and rchar.lower() != 'sfx':
                            print("Error: Empty text. Please try again.")
                            continue
                        clean_text = rclean_text
                        parsed_directives = rparsed_directives
                        break
                if char.lower() == 'sfx' and parsed_directives.get('duration') is None:
                    while True:
                        dur_input = input(f"  SFX duration for line {i} (1-30): ").strip()
                        if not dur_input:
                            print("Error: Duration is required for SFX lines.")
                            continue
                        val, err = _validate_duration_directive(dur_input)
                        if err:
                            print(f"Error: {err}. Please enter a number between 1 and 30.")
                            continue
                        parsed_directives['duration'] = val
                        break
                if not clean_text and char.lower() != 'sfx':
                    print(f"Error: Empty text in line: {line}")
                    return False
                dialogue_items.append((i, char, clean_text, parsed_directives))

        chars = set()
        for _, char, _, _ in dialogue_items:
            if char.lower() != 'sfx':
                chars.add(char.lower())

        _is_all_sfx_interactive = len(chars) == 0

        voice_prompts = {}
        target_assignments = {}
        trained_voice_refs = {}
        sts_refs = {}
        _dialogue_cleanup = []

        if _dialogue_speaker_extraction is not None:
            _dialogue_cleanup.append(_dialogue_speaker_extraction.get("temp_dir"))
            svs_temp = _dialogue_speaker_extraction.get("svs_temp_dir")
            if svs_temp and svs_temp != _dialogue_speaker_extraction.get("temp_dir"):
                _dialogue_cleanup.append(svs_temp)

        if not _is_all_sfx_interactive:
            sorted_chars = sorted(chars)

            if _dialogue_speaker_extraction is not None:
                speaker_files = _dialogue_speaker_extraction.get("speaker_files", {})
                diar_speakers = sorted(speaker_files.keys(),
                                       key=lambda spk: _dialogue_speaker_extraction.get("speaker_segments", {}).get(spk, [{"start": 0}])[0]["start"])
                spk_to_char = {}
                for idx, char_lower in enumerate(sorted_chars):
                    if idx < len(diar_speakers):
                        spk_to_char[diar_speakers[idx]] = char_lower

                for diar_spk, char_lower in spk_to_char.items():
                    spk_audio = speaker_files.get(diar_spk)
                    orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                    if spk_audio and os.path.exists(spk_audio):
                        target_assignments[char_lower] = spk_audio
                        print(f"  {orig_char} -> {diar_spk} (TSE extracted)")

                for i, char_lower in enumerate(sorted_chars):
                    if char_lower in target_assignments:
                        continue
                    orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                    print(f"  No TSE audio for {orig_char}, requesting reference...")
                    first_path = None
                    ref_paths = []
                    while True:
                        label = f"{orig_char} reference 1" if first_path is None else f"{orig_char} reference (Enter to finish)"
                        path = input(f"{label}: ").strip()
                        if not path:
                            if first_path is None:
                                print(f"Warning: At least one reference required for {orig_char}")
                                continue
                            break
                        if not os.path.exists(path) and not is_youtube_url(path):
                            print(f"Warning: File not found: {path}, skipping")
                            continue
                        if first_path is None:
                            first_path = path
                            ref_paths = [path]
                        else:
                            ref_paths.append(path)
                    if len(ref_paths) > 1:
                        clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                        if not clean_vocal:
                            return False
                    else:
                        resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
                        if not resolved_audio:
                            return False
                        _dialogue_cleanup.extend(_cl)
                        clean_vocal = svs_extract_vocals(resolved_audio)
                        if clean_vocal and clean_vocal != resolved_audio:
                            _dialogue_cleanup.append(clean_vocal)
                    target_assignments[char_lower] = clean_vocal
                    print(f"  {orig_char} -> manual ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")
            else:
                print(f"\nDo you have a multi-speaker audio source? (for auto voice cloning)")
                print("Press Y to provide a file, or N to enter manually for each character")
                has_multispeaker = input("> ").strip().lower()

                if has_multispeaker in ['y', 'yes']:
                    while True:
                        print("\nEnter the path to your multi-speaker audio source (file path or supported platform URL):")
                        file_path = input("> ").strip()
                        if not file_path:
                            print("Error: No file path provided")
                            continue

                        ss_source_audio = None
                        ss_source_cleanup = []

                        if is_youtube_url(file_path):
                            print(f"Downloading audio from YouTube...")
                            _dl_ok, _dl_err, _dl_path = download_youtube_audio(file_path)
                            if not _dl_ok:
                                print(f"Error: {_dl_err}")
                                retry = input("Try another source? (Y/N): ").strip().lower()
                                if retry not in ['y', 'yes']:
                                    return False
                                continue
                            ss_source_audio = _dl_path
                            ss_source_cleanup.append(_dl_path)
                        elif os.path.exists(file_path):
                            ss_source_audio = file_path
                            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                                print("Extracting audio from video...")
                                ss_source_audio = extract_audio_from_video_cli(file_path)
                                if not ss_source_audio:
                                    print("Error: Failed to extract audio from video")
                                    retry = input("Try another source? (Y/N): ").strip().lower()
                                    if retry not in ['y', 'yes']:
                                        return False
                                    continue
                                ss_source_cleanup.append(ss_source_audio)
                        else:
                            print(f"Error: File not found: {file_path}")
                            retry = input("Try another source? (Y/N): ").strip().lower()
                            if retry not in ['y', 'yes']:
                                return False
                            continue

                        print(f"\nExtracting speakers via TSE...")
                        _ms_speaker_extraction = _extract_speakers_for_subtitles(ss_source_audio)
                        for _cf in ss_source_cleanup:
                            _dialogue_cleanup.append(_cf)

                        if _ms_speaker_extraction:
                            _dialogue_cleanup.append(_ms_speaker_extraction.get("temp_dir"))
                            svs_temp = _ms_speaker_extraction.get("svs_temp_dir")
                            if svs_temp and svs_temp != _ms_speaker_extraction.get("temp_dir"):
                                _dialogue_cleanup.append(svs_temp)
                            ms_speaker_files = _ms_speaker_extraction.get("speaker_files", {})
                            ms_diar_speakers = sorted(ms_speaker_files.keys(),
                                                       key=lambda spk: _ms_speaker_extraction.get("speaker_segments", {}).get(spk, [{"start": 0}])[0]["start"])
                            print(f"\nExtracted {len(ms_diar_speakers)} speaker(s) via TSE.")
                            for diar_spk in ms_diar_speakers:
                                print(f"  {diar_spk}: {os.path.basename(ms_speaker_files[diar_spk])}")

                            spk_to_char = {}
                            for idx, char_lower in enumerate(sorted_chars):
                                if idx < len(ms_diar_speakers):
                                    spk_to_char[ms_diar_speakers[idx]] = char_lower

                            for diar_spk, char_lower in spk_to_char.items():
                                spk_audio = ms_speaker_files.get(diar_spk)
                                orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                                if spk_audio and os.path.exists(spk_audio):
                                    target_assignments[char_lower] = spk_audio
                                    print(f"  {orig_char} -> {diar_spk} (TSE extracted)")

                            for i, char_lower in enumerate(sorted_chars):
                                if char_lower in target_assignments:
                                    continue
                                orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                                print(f"  No TSE audio for {orig_char}, requesting reference...")
                                first_path = None
                                ref_paths = []
                                while True:
                                    label = f"{orig_char} reference 1" if first_path is None else f"{orig_char} reference (Enter to finish)"
                                    path = input(f"{label}: ").strip()
                                    if not path:
                                        if first_path is None:
                                            print(f"Warning: At least one reference required for {orig_char}")
                                            continue
                                        break
                                    if not os.path.exists(path) and not is_youtube_url(path):
                                        print(f"Warning: File not found: {path}, skipping")
                                        continue
                                    if first_path is None:
                                        first_path = path
                                        ref_paths = [path]
                                    else:
                                        ref_paths.append(path)
                                if len(ref_paths) > 1:
                                    clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                                    if not clean_vocal:
                                        return False
                                else:
                                    resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
                                    if not resolved_audio:
                                        return False
                                    _dialogue_cleanup.extend(_cl)
                                    clean_vocal = svs_extract_vocals(resolved_audio)
                                    if clean_vocal and clean_vocal != resolved_audio:
                                        _dialogue_cleanup.append(clean_vocal)
                                target_assignments[char_lower] = clean_vocal
                                print(f"  {orig_char} -> manual ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")
                        else:
                            print("TSE extraction failed, falling back to manual reference for each character.")
                            for i, char_lower in enumerate(sorted_chars):
                                orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                                first_path = None
                                ref_paths = []
                                while True:
                                    label = f"{orig_char} reference 1" if first_path is None else f"{orig_char} reference (Enter to finish)"
                                    path = input(f"{label}: ").strip()
                                    if not path:
                                        if first_path is None:
                                            print(f"Warning: At least one reference required for {orig_char}")
                                            continue
                                        break
                                    if not os.path.exists(path) and not is_youtube_url(path):
                                        print(f"Warning: File not found: {path}, skipping")
                                        continue
                                    if first_path is None:
                                        first_path = path
                                        ref_paths = [path]
                                    else:
                                        ref_paths.append(path)
                                if len(ref_paths) > 1:
                                    clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                                    if not clean_vocal:
                                        return False
                                else:
                                    resolved_audio, _cl = resolve_target_to_audio(ref_paths[0])
                                    if not resolved_audio:
                                        return False
                                    _dialogue_cleanup.extend(_cl)
                                    clean_vocal = svs_extract_vocals(resolved_audio)
                                    if clean_vocal and clean_vocal != resolved_audio:
                                        _dialogue_cleanup.append(clean_vocal)
                                target_assignments[char_lower] = clean_vocal
                                print(f"  {orig_char} -> manual ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")

                        break
                else:
                    print(f"\nVoice prompts or audio file paths for {len(chars)} character(s):")
                    print("(Enter text for voice prompt, a path/URL to clone a voice, or a trained voice name)")
                    for i, char_lower in enumerate(sorted_chars):
                        orig_char = next((c for _, c, _, _ in dialogue_items if c.lower() == char_lower), char_lower)
                        prompt = input(f"{orig_char}: ").strip()
                        if not prompt:
                            print(f"Error: No voice prompt or audio path provided for {orig_char}")
                            return False
                        trained_file = _resolve_voice_ref(prompt)
                        if trained_file:
                            voice_items = _load_voice_prompt(trained_file)
                            if voice_items is None:
                                print(f"Error: Failed to load trained voice: {trained_file}")
                                return False
                            trained_voice_refs[char_lower] = trained_file
                            print(f"  {orig_char} -> trained voice ({os.path.basename(trained_file)})")
                        elif os.path.exists(prompt) or is_youtube_url(prompt):
                            ref_paths = [prompt]
                            ref_num = 2
                            while True:
                                more = input(f"{orig_char} reference {ref_num} (Enter to finish): ").strip()
                                if not more:
                                    break
                                if os.path.exists(more) or is_youtube_url(more):
                                    ref_paths.append(more)
                                    ref_num += 1
                                else:
                                    print(f"Warning: File not found: {more}, skipping")
                            if len(ref_paths) > 1:
                                clean_vocal = _resolve_multi_refs(ref_paths, _dialogue_cleanup)
                                if not clean_vocal:
                                    return False
                            else:
                                resolved_audio, _cl = resolve_target_to_audio(prompt)
                                if not resolved_audio:
                                    return False
                                _dialogue_cleanup.extend(_cl)
                                clean_vocal = svs_extract_vocals(resolved_audio)
                                if clean_vocal and clean_vocal != resolved_audio:
                                    _dialogue_cleanup.append(clean_vocal)
                            target_assignments[char_lower] = clean_vocal
                            print(f"  {orig_char} -> voice clone ({len(ref_paths)} ref{'s' if len(ref_paths) > 1 else ''})")
                        else:
                            voice_prompts[char_lower] = prompt
                        print(f"Progress: {i+1}/{len(chars)} completed")

        has_tts_chars = len(voice_prompts) > 0
        has_vc_chars = len(target_assignments) > 0 or len(trained_voice_refs) > 0

        music_description = None
        music_level_spec = None
        add_music = input("\nAdd background music? (y/N): ").strip().lower()
        if add_music in ('y', 'yes'):
            music_desc = input("Music description: ").strip()
            if music_desc:
                music_description = music_desc
        if music_description:
            level_input = input("Sound level (optional, press Enter for default 35%): ").strip()
            if level_input:
                parsed_level = _parse_music_level_spec(level_input)
                if parsed_level is None:
                    print("Warning: Invalid level format, using default 35%")
                else:
                    music_level_spec = level_input
            ref_input = input("Music reference (path/URL, or press Enter to skip): ").strip()
            if ref_input:
                if not os.path.exists(ref_input) and not is_youtube_url(ref_input):
                    print("Error: Music reference not found: " + ref_input)
                    return False
        else:
            ref_input = None

        music_reference_audio = None
        music_ref_cleanup = []
        if ref_input and music_description:
            print("Resolving music reference source...")
            resolved_ref, ref_cl = resolve_target_to_audio(ref_input)
            if not resolved_ref:
                return False
            music_ref_cleanup.extend(ref_cl)
            print("Extracting clean music from reference via SVS...")
            music_reference_audio = svs_extract_music(resolved_ref)
            if music_reference_audio and music_reference_audio != resolved_ref and music_reference_audio not in music_ref_cleanup:
                music_ref_cleanup.append(music_reference_audio)

        try:
            tts_design = None
            if has_tts_chars:
                print("\nLoading Qwen-TTS VoiceDesign model...")
                tts_design = QwenTTSVoiceDesign()
                if tts_design.model is None:
                    print("Error: Failed to load VoiceDesign model")
                    return False

            tts_obj = None
            vc_voice_prompts = None
            fish_voice_data = None
            if has_vc_chars:
                if use_extreme:
                    print("Loading Fish-S2Pro model (extreme)...")
                    tts_obj = FishTTS()
                    if not tts_obj.ensure_model():
                        print("Error: Failed to load Fish-S2Pro model")
                        return False
                    fish_voice_data = {}
                    for char_lower, audio_path in target_assignments.items():
                        print(f"Encoding voice for '{char_lower}' (extreme)...")
                        ref_text = _transcribe_for_fish_ref(audio_path)
                        voice_ok = _tts_extract_voice(tts_obj, audio_path, use_extreme=True, ref_text=ref_text)
                        if voice_ok:
                            fish_voice_data[char_lower] = {
                                "tokens": tts_obj.encoded_refs["tokens"].cpu().clone(),
                                "text": tts_obj.encoded_refs["text"]
                            }
                            print(f"  Encoded voice for {char_lower}")
                        else:
                            print(f"Warning: Voice encoding failed for '{char_lower}', falling back to Qwen-TTS")
                            tts_obj.cleanup()
                            del tts_obj
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            tts_obj = None
                            fish_voice_data = None
                            break

                    if tts_obj is None and fish_voice_data is None:
                        print("Loading Qwen-TTS model...")
                        tts_obj = QwenTTS()
                        vc_voice_prompts = {}
                        for char_lower, audio_path in target_assignments.items():
                            print(f"Extracting voice for '{char_lower}'...")
                            ref_text = _transcribe_for_qwen_ref(audio_path)
                            success = tts_obj.extract_voice(audio_path, ref_text=ref_text if ref_text else None)
                            if not success:
                                print(f"Error: Failed to extract voice from {audio_path}")
                                return False
                            vc_voice_prompts[char_lower] = tts_obj.voice_prompt

                    for char_lower, trained_file in trained_voice_refs.items():
                        if fish_voice_data is not None:
                            payload = _load_fish_voice(trained_file)
                            if payload is not None:
                                fish_voice_data[char_lower] = payload
                            else:
                                print(f"Warning: Failed to load trained voice for '{char_lower}'")
                        elif vc_voice_prompts is not None:
                            voice_items = _load_voice_prompt(trained_file)
                            if voice_items is None:
                                print(f"Error: Failed to load trained voice: {trained_file}")
                                return False
                            vc_voice_prompts[char_lower] = voice_items
                else:
                    print("Loading Qwen-TTS model...")
                    tts_obj = QwenTTS()
                    vc_voice_prompts = {}
                    for char_lower, audio_path in target_assignments.items():
                        print(f"Extracting voice for '{char_lower}'...")
                        ref_text = _transcribe_for_qwen_ref(audio_path)
                        success = tts_obj.extract_voice(audio_path, ref_text=ref_text if ref_text else None)
                        if not success:
                            print(f"Error: Failed to extract voice from {audio_path}")
                            return False
                        vc_voice_prompts[char_lower] = tts_obj.voice_prompt
                    for char_lower, trained_file in trained_voice_refs.items():
                        print(f"Loading trained voice for '{char_lower}' from: {trained_file}")
                        voice_items = _load_voice_prompt(trained_file)
                        if voice_items is None:
                            print(f"Error: Failed to load trained voice: {trained_file}")
                            return False
                        vc_voice_prompts[char_lower] = voice_items

            if has_tts_chars and tts_obj is None:
                print("Loading Qwen-TTS model for voice stabilization...")
                tts_obj = QwenTTS()

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            base_name = f"voder_tts_dialogue_{timestamp}"
            if music_description:
                base_name += "_m"
            output_path = os.path.join(results_dir, f"{base_name}.wav")

            dialogue_temp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            dialogue_temp.close()

            has_sfx = any(item[1].lower() == 'sfx' for item in dialogue_items)
            has_effects = any(
                item[3].get('time_end', 0) > 0 or item[3].get('time_start', 0) > 0 or item[3].get('time_pad', 0) > 0 or item[3].get('level', 100) != 100
                for item in dialogue_items
            ) if len(dialogue_items) > 0 else False

            if has_sfx or has_effects or has_vc_chars or has_tts_chars:
                success, msg = _assemble_enhanced_dialogue(
                    dialogue_items, voice_prompts, tts_design_obj=tts_design,
                    tts_vc_obj=tts_obj, vc_voice_data=vc_voice_prompts,
                    output_path=dialogue_temp.name, mode='tts',
                    sts_refs=sts_refs if sts_refs else None,
                    use_extreme=use_extreme, fish_voice_data=fish_voice_data
                )
                if not success:
                    print(f"Error: {msg}")
                    return False
            elif len(dialogue_items) == 1:
                _, char, text, _ = dialogue_items[0]
                voice_instruct = voice_prompts[char.lower()]
                success = tts_design.synthesize(text, voice_instruct, dialogue_temp.name)
                if not success:
                    print("Error: VoiceDesign synthesis failed")
                    return False
            else:
                simple_items = [(item[0], item[1], item[2]) for item in dialogue_items]
                success, msg = tts_design.synthesize_dialogue(simple_items, voice_prompts, dialogue_temp.name)
                if not success:
                    print(f"Error: {msg}")
                    return False

            if music_description:
                if tts_design is not None:
                    del tts_design
                    tts_design = None
                if tts_obj is not None:
                    del tts_obj
                    tts_obj = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ace = AceStepWrapper(use_overdose=use_overdose)
                if ace.handler is None:
                    print("Error: Failed to load ACE-Step model")
                    return False
                success = _generate_music_and_mix(ace, music_description, dialogue_temp.name, output_path, music_level_spec, reference_audio=music_reference_audio)
                del ace
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not success:
                    return False
                os.unlink(dialogue_temp.name)
            else:
                shutil.move(dialogue_temp.name, output_path)
            print(f"\n✓ Success! Output saved to: {output_path}")
            if tts_design is not None:
                del tts_design
                tts_design = None
            if tts_obj is not None:
                del tts_obj
                tts_obj = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True
        finally:
            for f in music_ref_cleanup:
                if f and os.path.exists(f):
                    try:
                        os.unlink(f)
                    except:
                        pass
            for f in _dialogue_cleanup:
                if f and os.path.exists(f):
                    try:
                        if os.path.isdir(f):
                            shutil.rmtree(f)
                        else:
                            os.unlink(f)
                    except:
                        pass
            if 'dialogue_temp' in dir() and os.path.exists(dialogue_temp.name):
                try:
                    os.unlink(dialogue_temp.name)
                except:
                    pass
