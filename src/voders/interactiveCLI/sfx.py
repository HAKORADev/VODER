import os
import time
import gc
import torch

from voder import (
    _validate_text_language,
    SUPPORTED_TANGOFLUX_LANGS,
    TANGOFLUX_DIR,
)


def cli_sfx_mode():
    original_cwd = os.getcwd()
    results_dir = os.path.join(original_cwd, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- SFX Mode ---")
    print("Sound Effects Generation - text prompt + duration → audio")
    print("Steps: 30, Guidance: 4.5 (hardcoded for best results)")
    print()

    while True:
        prompt = input("Enter sound prompt: ").strip()
        if not prompt:
            print("Error: Prompt cannot be empty. Please try again.")
            continue
        sfx_valid, _ = _validate_text_language(prompt, SUPPORTED_TANGOFLUX_LANGS, "SFX")
        if sfx_valid:
            break
        print("Try again with English prompts (SFX only supports English)")

    while True:
        duration_input = input("Enter duration in seconds (1-30): ").strip()
        if not duration_input:
            print("Error: Duration cannot be empty. Please try again.")
            continue
        try:
            duration = int(duration_input)
            if duration < 1:
                print("Error: Duration must be at least 1 second. Please try again.")
                continue
            if duration > 30:
                print("Warning: Duration >30s clamped to 30s (model maximum).")
                duration = 30
            break
        except ValueError:
            print("Error: Invalid number. Please enter a number between 1 and 30.")

    print("\nLoading TangoFlux SFX model...")
    from tangoflux import TangoFluxGenerator
    generator = TangoFluxGenerator(TANGOFLUX_DIR)
    generator.ensure_model()
    if generator.model is None:
        print("Error: Failed to load TangoFlux model")
        return False

    try:
        print(f"\nGenerating sound effect...")
        audio = generator.generate(prompt, duration, steps=30, guidance_scale=4.5)
        if audio is None:
            print("Error: Generation failed")
            return False

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"voder_sfx_{timestamp}.wav"
        output_path = os.path.join(results_dir, output_filename)

        if generator.save(audio, output_path):
            print(f"\n✓ Success! Output saved to: {output_path}")
            return True
        else:
            print("Error: Failed to save output")
            return False

    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        generator.cleanup()
        del generator
        generator = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
