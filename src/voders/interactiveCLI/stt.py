import os
import sys
import time
import shutil
import tempfile
import gc
import torch
import torchaudio

from voder import (
    is_youtube_url,
    download_youtube_audio,
    extract_audio_from_video_cli,
    SVS_DIR,
    VibeVoiceASR,
    WhisperSTT,
    SpeakerDiarization,
)


def cli_stt_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- STT Mode ---")
    print("Speech-to-Text (Transcription)")
    print()

    while True:
        translate_input = input("Translate to English? (Y/N): ").strip().lower()
        if translate_input in ['y', 'yes']:
            enable_translate = True
            break
        elif translate_input in ['n', 'no']:
            enable_translate = False
            break
        else:
            print("Please enter Y or N")

    while True:
        file_path = input("Enter audio/video file path or supported platform URL: ").strip()
        if not file_path:
            print("Error: No path provided")
            continue
        if os.path.exists(file_path) or is_youtube_url(file_path):
            break
        print("Error: File not found or invalid URL")

    while True:
        timestamp_input = input("Keep timestamps? (Y/N): ").strip().lower()
        if timestamp_input in ['y', 'yes']:
            keep_timestamp = True
            break
        elif timestamp_input in ['n', 'no']:
            keep_timestamp = False
            break
        else:
            print("Please enter Y or N")

    while True:
        dialogue_input = input("Enable speaker diarization? (Y/N): ").strip().lower()
        if dialogue_input in ['y', 'yes']:
            enable_dialogue = True
            break
        elif dialogue_input in ['n', 'no']:
            enable_dialogue = False
            break
        else:
            print("Please enter Y or N")

    use_overdose = False
    if not enable_translate:
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

    audio_path = file_path
    needs_cleanup = False
    is_youtube = is_youtube_url(file_path)

    if is_youtube:
        print("Downloading audio from YouTube...")
        success_dl, error_msg, audio_path = download_youtube_audio(file_path)
        if not success_dl:
            print(f"Error: {error_msg}")
            return False
    elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        print("Extracting audio from video...")
        extracted = extract_audio_from_video_cli(file_path)
        if not extracted:
            print(f"Error: Could not extract audio from {file_path}")
            return False
        audio_path = extracted
        needs_cleanup = True

    _voder_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bs_roformer_lib = os.path.join(_voder_src, 'bs_roformer', 'lib')
    if bs_roformer_lib not in sys.path:
        sys.path.insert(0, bs_roformer_lib)
    if _voder_src not in sys.path:
        sys.path.insert(0, _voder_src)

    print("Stage 1: SVS voice isolation (BS-RoFormer)...")
    from bs_roformer import BSRoformerSeparator
    svs_separator = BSRoformerSeparator(SVS_DIR)
    svs_separator.ensure_model(stem='voice')
    if svs_separator.vocals_model is None:
        print("Error: Failed to load BS-RoFormer vocals model")
        svs_separator.cleanup()
        del svs_separator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    svs_temp_dir = tempfile.mkdtemp()
    svs_temp = os.path.join(svs_temp_dir, f'_cli_stt_svs_{timestamp}.wav')
    svs_ok = svs_separator.separate(audio_path, 'voice', svs_temp)
    svs_separator.cleanup()
    del svs_separator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if svs_ok and os.path.exists(svs_temp):
        if needs_cleanup and audio_path != file_path and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
        audio_path = svs_temp
        needs_cleanup = True
    else:
        print("Warning: SVS voice isolation failed, using original audio")
        shutil.rmtree(svs_temp_dir, ignore_errors=True)

    try:
        if use_overdose and not enable_translate:
            asr = VibeVoiceASR()
            asr.ensure_model()
            if asr.model is None:
                print("Warning: VibeVoice ASR failed to load, falling back to Whisper")
                asr.cleanup()
                del asr
                use_overdose = False

        if use_overdose and not enable_translate:
            print("Transcribing with VibeVoice ASR...")
            asr_segments = asr.transcribe(audio_path)
            asr.cleanup()
            del asr
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if not asr_segments:
                print("Error: ASR transcription returned no segments")
                return False

            def format_time_range(start, end):
                def format_single(seconds):
                    if seconds is None:
                        seconds = 0
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 100)
                    return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                return f"[{format_single(start)}-{format_single(end)}]"

            if enable_dialogue:
                original_speakers = []
                for seg in asr_segments:
                    speaker = seg["speaker"]
                    if speaker not in original_speakers:
                        original_speakers.append(speaker)
                speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}
                lines = []
                current_speaker_num = None
                current_text_parts = []
                current_first_time = None
                current_last_time = None
                for seg in asr_segments:
                    speaker_num = speaker_mapping[seg["speaker"]]
                    text = seg.get("text", "")
                    seg_start = seg.get("start", 0) or 0
                    seg_end = seg.get("end", 0) or 0
                    if current_speaker_num is None:
                        current_speaker_num = speaker_num
                        current_text_parts = [text]
                        current_first_time = seg_start
                        current_last_time = seg_end
                    elif speaker_num == current_speaker_num:
                        current_text_parts.append(text)
                        current_last_time = seg_end
                    else:
                        if current_text_parts:
                            content_out = " ".join(current_text_parts)
                            if len(original_speakers) == 1:
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content_out}")
                                else:
                                    lines.append(f"text: {content_out}")
                            else:
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                else:
                                    lines.append(f"{current_speaker_num}: {content_out}")
                        current_speaker_num = speaker_num
                        current_text_parts = [text]
                        current_first_time = seg_start
                        current_last_time = seg_end
                if current_text_parts:
                    content_out = " ".join(current_text_parts)
                    if len(original_speakers) == 1:
                        if keep_timestamp:
                            lines.append(f"{format_time_range(current_first_time, current_last_time)} text: {content_out}")
                        else:
                            lines.append(f"text: {content_out}")
                    else:
                        if keep_timestamp:
                            lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                        else:
                            lines.append(f"{current_speaker_num}: {content_out}")
                formatted_text = "\n".join(lines)
            elif keep_timestamp:
                lines = []
                for seg in asr_segments:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    text = seg.get("text", "").strip()
                    if text:
                        lines.append(f"{format_time_range(start, end)} text: {text}")
                if lines:
                    formatted_text = "\n".join(lines)
                else:
                    formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
            else:
                formatted_text = " ".join(seg.get("text", "") for seg in asr_segments)
        else:
            print("Loading Whisper model...")
            stt = WhisperSTT()
            if stt.model is None:
                print("Error: Failed to load Whisper model")
                return False

            if enable_translate and enable_dialogue:
                print("Transcribing audio (for diarization)...")
                original_result = stt.transcribe(audio_path)
                if not original_result:
                    print("Error: Transcription failed")
                    return False

                print("Translating audio to English...")
                result = stt.translate(audio_path)
                if not result:
                    print("Error: Translation failed, using original transcription")
                    result = original_result
                    enable_translate = False

                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            elif enable_translate:
                print("Translating audio to English...")
                result = stt.translate(audio_path)
                if not result:
                    print("Error: Translation failed")
                    return False

                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                print("Transcribing audio...")
                result = stt.transcribe(audio_path)
                if not result:
                    print("Error: Transcription failed")
                    return False

                del stt
                stt = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            def format_time_range(start, end):
                def format_single(seconds):
                    if seconds is None:
                        seconds = 0
                    minutes = int(seconds // 60)
                    secs = int(seconds % 60)
                    millis = int((seconds % 1) * 100)
                    return f"{minutes:02d}:{secs:02d}:{millis:02d}"
                return f"[{format_single(start)}-{format_single(end)}]"

            def format_time(seconds):
                if seconds is None:
                    seconds = 0
                minutes = int(seconds // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 100)
                return f"[{minutes:02d}:{secs:02d}:{millis:02d}]"

        if not use_overdose:
            if enable_dialogue:
                print("Performing speaker diarization...")
                diarization = SpeakerDiarization()
                if diarization.pipeline is None:
                    print("Warning: Speaker diarization model not available, proceeding without it")
                    if keep_timestamp and result.get("segments"):
                        lines = []
                        for seg in result.get("segments", []):
                            start = seg.get("start", 0)
                            end = seg.get("end", 0)
                            text = seg.get("text", "").strip()
                            if text:
                                lines.append(f"{format_time_range(start, end)} text: {text}")
                        if lines:
                            formatted_text = "\n".join(lines)
                        else:
                            formatted_text = result.get("text", "").strip()
                    else:
                        formatted_text = result.get("text", "").strip()
                else:
                    diar_result = diarization.diarize(audio_path)
                    if enable_translate:
                        diarization_segments = diarization.format_diarization(diar_result, original_result)
                    else:
                        diarization_segments = diarization.format_diarization(diar_result, result)

                    formatted_segments = None
                    if diarization_segments:
                        if enable_translate:
                            translated_segments = result.get("segments", [])
                            speaker_time_map = []
                            for ds in diarization_segments:
                                speaker_time_map.append({
                                    "speaker": ds["speaker"],
                                    "start": ds.get("start", 0),
                                    "end": ds.get("end", 0),
                                    "text": ds["text"]
                                })

                            merged_segments = []
                            for ts in translated_segments:
                                ts_start = ts.get("start", 0)
                                ts_end = ts.get("end", 0)
                                ts_text = ts.get("text", "").strip()
                                if not ts_text:
                                    continue
                                best_speaker = None
                                best_overlap = 0
                                for sm in speaker_time_map:
                                    overlap_start = max(ts_start, sm["start"])
                                    overlap_end = min(ts_end, sm["end"])
                                    overlap = max(0, overlap_end - overlap_start)
                                    if overlap > best_overlap:
                                        best_overlap = overlap
                                        best_speaker = sm["speaker"]
                                if best_speaker is not None:
                                    merged_segments.append({
                                        "speaker": best_speaker,
                                        "start": ts_start,
                                        "end": ts_end,
                                        "text": ts_text
                                    })
                            formatted_segments = merged_segments if merged_segments else None
                        else:
                            formatted_segments = diarization_segments

                    if formatted_segments:
                        original_speakers = []
                        for seg in formatted_segments:
                            speaker = seg["speaker"]
                            if speaker not in original_speakers:
                                original_speakers.append(speaker)

                        speaker_mapping = {spk: idx for idx, spk in enumerate(original_speakers, 1)}

                        if len(original_speakers) == 1:
                            content_out = " ".join(seg["text"] for seg in formatted_segments)
                            if keep_timestamp:
                                first_time = formatted_segments[0]["start"]
                                last_time = formatted_segments[-1]["end"]
                                formatted_text = f"{format_time_range(first_time, last_time)} text: {content_out}"
                            else:
                                formatted_text = f"text: {content_out}"
                        else:
                            lines = []
                            current_speaker_num = None
                            current_text_parts = []
                            current_first_time = None
                            current_last_time = None

                            for seg in formatted_segments:
                                speaker_num = speaker_mapping[seg["speaker"]]
                                text = seg["text"]
                                seg_start = seg.get("start", 0) or 0
                                seg_end = seg.get("end", 0) or 0

                                if current_speaker_num is None:
                                    current_speaker_num = speaker_num
                                    current_text_parts = [text]
                                    current_first_time = seg_start
                                    current_last_time = seg_end
                                elif speaker_num == current_speaker_num:
                                    current_text_parts.append(text)
                                    current_last_time = seg_end
                                else:
                                    if current_text_parts:
                                        content_out = " ".join(current_text_parts)
                                        if keep_timestamp:
                                            lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                        else:
                                            lines.append(f"{current_speaker_num}: {content_out}")
                                    current_speaker_num = speaker_num
                                    current_text_parts = [text]
                                    current_first_time = seg_start
                                    current_last_time = seg_end

                            if current_text_parts:
                                content_out = " ".join(current_text_parts)
                                if keep_timestamp:
                                    lines.append(f"{format_time_range(current_first_time, current_last_time)} {current_speaker_num}: {content_out}")
                                else:
                                    lines.append(f"{current_speaker_num}: {content_out}")

                            formatted_text = "\n".join(lines)

                        del diarization
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        if keep_timestamp and result.get("segments"):
                            lines = []
                            for seg in result.get("segments", []):
                                start = seg.get("start", 0)
                                end = seg.get("end", 0)
                                text = seg.get("text", "").strip()
                                if text:
                                    lines.append(f"{format_time_range(start, end)} text: {text}")
                            if lines:
                                formatted_text = "\n".join(lines)
                            else:
                                formatted_text = result.get("text", "").strip()
                        else:
                            formatted_text = result.get("text", "").strip()
            else:
                if keep_timestamp and result.get("segments"):
                    lines = []
                    for seg in result.get("segments", []):
                        start = seg.get("start", 0)
                        end = seg.get("end", 0)
                        text = seg.get("text", "").strip()
                        if text:
                            lines.append(f"{format_time_range(start, end)} text: {text}")
                    if lines:
                        formatted_text = "\n".join(lines)
                    else:
                        formatted_text = result.get("text", "").strip()
                else:
                    formatted_text = result.get("text", "").strip()

        print("\n" + formatted_text)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if is_youtube:
            base_name = "youtube_stt"
        else:
            base_name = os.path.splitext(os.path.basename(file_path))[0]

        suffix_parts = ["stt"]
        if enable_translate:
            suffix_parts.append("translate")
        if keep_timestamp:
            suffix_parts.append("timestamp")
        if enable_dialogue:
            suffix_parts.append("dialogue")
        suffix = "_".join(suffix_parts)

        output_filename = f"voder_{suffix}_{timestamp}_{base_name}.txt"
        output_path = os.path.join(results_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)

        print(f"\n\u2713 Success! Output saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        if needs_cleanup and audio_path != file_path and os.path.exists(audio_path):
            try:
                parent_dir = os.path.dirname(audio_path)
                os.unlink(audio_path)
                if os.path.exists(parent_dir) and os.path.basename(parent_dir).startswith('_'):
                    shutil.rmtree(parent_dir, ignore_errors=True)
            except:
                pass
        if is_youtube and os.path.exists(audio_path):
            try:
                os.unlink(audio_path)
            except:
                pass
