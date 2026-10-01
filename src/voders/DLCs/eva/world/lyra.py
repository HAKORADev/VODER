import os
import sys
import time
import re

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

ENV_KEY = "lyra2"
LYRA_DEFAULT_TRAJECTORY = "horizontal_zoom"
LYRA_DIRECTIONS = ("left", "right", "up", "down")
LYRA_TRAJECTORIES = (
    "original",
    "spiral",
    "spiral_center",
    "spiral_outwards",
    "horizontal",
    "horizontal_noise",
    "horizontal_lift",
    "horizontal_lift_noise",
    "horizontal_zoom",
    "horizontal_zoom_noise",
    "horizontal_zoom_bend",
    "horizontal_zoom_noise_bend",
    "horizontal_zoom_still",
    "horizontal_still",
    "horizontal_simple",
    "vertical_simple",
    "horizontal_outward",
    "back",
    "back_simple",
    "dolly_zoom",
    "horizontal_spiral",
    "orbit_horizontal",
    "orbit_vertical",
    "rotate_zoom_in",
    "rotate_zoom_out",
    "rotate_spot",
    "rotate_spot_noise",
)
LYRA_TTI_PRESET_RESOLUTION = "1280x720"
LYRA_GENERIC_CAPTION = "A detailed photograph of a scenic outdoor environment with clear spatial structure."
LYRA_DEFAULT_DURATION = 20
LYRA_MIN_STRENGTH = 0.1
LYRA_MAX_STRENGTH = 5.0


class Lyra2Wrapper:
    def __init__(self):
        self.pipeline = None

    def ensure_model(self):
        from voders.DLCs.eva._envrunner import venv_exists
        if venv_exists(ENV_KEY):
            return True
        print(f"Lyra 2.0 env not set up. Run: python setup.py --envs {ENV_KEY}")
        return False

    def explorify(self, input_path, output_path, desc=None, seed=0, duration=None,
                  trajectory=None, direction=None, strength=None, fast=False):
        from voders.DLCs.eva._envrunner import run_in_venv
        from voders.DLCs.eva.media_download import resolve_input_path
        if not input_path and not desc:
            print("Error: explorify requires an input image path (or desc to generate the seed image with TTI overdose first)")
            return False
        generated_seed_image = None
        if input_path:
            resolved = resolve_input_path(input_path, media_type='image')
            if resolved is None:
                return False
            input_path = resolved
        else:
            input_path, generated_seed_image = self._generate_seed_image(
                desc, seed, os.path.dirname(os.path.abspath(output_path)))
            if input_path is None:
                return False
        caption = (desc or "").strip() or LYRA_GENERIC_CAPTION
        if trajectory is None:
            trajectory = LYRA_DEFAULT_TRAJECTORY
        if strength is not None:
            if strength <= 0:
                print(f"Warning: camera strength must be positive, got {strength} — using the official default")
                strength = None
            elif strength > LYRA_MAX_STRENGTH:
                print(f"Warning: camera strength {strength} exceeds the official range — clamping to {LYRA_MAX_STRENGTH}")
                strength = LYRA_MAX_STRENGTH
        if duration is not None and duration <= 0:
            print(f"Warning: invalid duration {duration}, using the official default frame counts")
            duration = None
        spec = {
            "action": "generate",
            "input_path": input_path,
            "output_path": output_path,
            "prompt": caption,
            "seed": seed,
            "duration": duration,
            "trajectory": trajectory,
            "direction": direction,
            "strength": strength,
            "fast": bool(fast),
        }
        result = run_in_venv(ENV_KEY, spec)
        if result.get("success"):
            print(f"\n✓ Success! Explorable scene saved to: {result.get('output_path', output_path)}")
            return True
        print(f"Error: {result.get('error', 'unknown')}")
        return False

    def _generate_seed_image(self, desc, seed, results_dir):
        from voders.DLCs.eva.image.qwen import QwenImageUCWrapper
        print("No input image given — generating the seed image with TTI overdose (Qwen-Image-2.1 UC)...")
        print(f"Preset: resolution {LYRA_TTI_PRESET_RESOLUTION} (closest supported aspect to the official 832x480 Lyra canvas), seed {seed}")
        os.makedirs(results_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_desc = re.sub(r'[^A-Za-z0-9_\-]', '_', (desc or 'seed')[:100]) or 'seed'
        seed_image_path = os.path.join(results_dir, f"voder_eva_ttw_explorify_seed_image_{safe_desc}_{timestamp}.png")
        wrapper = QwenImageUCWrapper()
        try:
            ok = wrapper.generate(desc, seed_image_path, resolution=LYRA_TTI_PRESET_RESOLUTION, seed=seed)
        finally:
            wrapper.cleanup()
        if not ok:
            print("Error: TTI overdose failed to produce the seed image for explorify")
            return None, None
        print(f"Seed image ready: {seed_image_path}")
        return seed_image_path, seed_image_path

    def cleanup(self):
        pass
