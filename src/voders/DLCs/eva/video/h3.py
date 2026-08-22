import os
import sys

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

H3_DEFAULT_RESOLUTION = "1280x720"
H3_DEFAULT_DURATION = 10
H3_MAX_DURATION = 10
H3_SUPPORTED_RESOLUTIONS = ["1280x720", "720x1280", "832x480", "480x832", "1024x1024"]
H3_MAX_DIMENSION = 1280
H3_MAX_IMAGE_REFS = 9
H3_MAX_VIDEO_REFS = 3
H3_MAX_AUDIO_REFS = 3

ENV_KEY = "h3"


class H3Wrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"MiniMax H3 env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def generate(self, prompt, output_path, duration=H3_DEFAULT_DURATION, resolution=None, seed=0,
                 image_refs=None, video_refs=None, audio_refs=None):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path, check_reference_limit
        from voders.DLCs.eva.downscale import check_and_downscale_input
        from voders.DLCs.eva.image.sam import sam_auto_mask_for_edit, sam_apply_mask_to_image
        image_refs = check_reference_limit(image_refs, H3_MAX_IMAGE_REFS, 'H3 (image refs)')
        video_refs = check_reference_limit(video_refs, H3_MAX_VIDEO_REFS, 'H3 (video refs)')
        audio_refs = check_reference_limit(audio_refs, H3_MAX_AUDIO_REFS, 'H3 (audio refs)')
        resolved_image_refs = []
        if image_refs:
            for ref_path in image_refs:
                resolved = resolve_input_path(ref_path, media_type='image')
                if resolved:
                    resolved = check_and_downscale_input(resolved, H3_MAX_DIMENSION, H3_MAX_DIMENSION)
                    if image_refs.index(ref_path) == 0 and len(image_refs) == 1:
                        mask = sam_auto_mask_for_edit(resolved)
                        if mask is not None:
                            masked_path = resolved.rsplit('.', 1)[0] + '_masked.png'
                            sam_apply_mask_to_image(resolved, mask, masked_path)
                            resolved = masked_path
                            print(f"SAM: auto-masked subject in reference image")
                    resolved_image_refs.append(resolved)
        resolved_video_refs = []
        if video_refs:
            for ref_path in video_refs:
                resolved = resolve_input_path(ref_path, media_type='video')
                if resolved:
                    resolved = check_and_downscale_input(resolved, H3_MAX_DIMENSION, H3_MAX_DIMENSION)
                    resolved_video_refs.append(resolved)
        resolved_audio_refs = []
        if audio_refs:
            for ref_path in audio_refs:
                resolved = resolve_input_path(ref_path, media_type='audio')
                if resolved:
                    resolved_audio_refs.append(resolved)
        print(f"Generating video with MiniMax H3 (up to {duration}s)...")
        spec = {
            "action": "generate",
            "prompt": prompt,
            "output_path": output_path,
            "duration": duration,
            "resolution": resolution or H3_DEFAULT_RESOLUTION,
            "seed": seed,
            "image_refs": resolved_image_refs,
            "video_refs": resolved_video_refs,
            "audio_refs": resolved_audio_refs,
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Output saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def cleanup(self):
        pass
