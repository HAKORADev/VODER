import os
import sys
import subprocess

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)


def downscale_image(image_path, max_width, max_height, output_path=None):
    try:
        import cv2
        import numpy as np
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            from PIL import Image
            pil_img = Image.open(image_path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        h, w = img.shape[:2]
        if w <= max_width and h <= max_height:
            if output_path:
                cv2.imwrite(output_path, img)
            return image_path
        ratio = min(max_width / w, max_height / h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8
        new_w = max(new_w, 8)
        new_h = max(new_h, 8)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        if output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = base + '_downscaled' + ext
        cv2.imwrite(output_path, resized)
        print(f"Image downscaled from {w}x{h} to {new_w}x{new_h} (LANCZOS)")
        return output_path
    except Exception as e:
        print(f"Downscale error: {e}")
        return image_path


def downscale_video(video_path, max_width, max_height, output_path=None):
    try:
        if output_path is None:
            base, ext = os.path.splitext(video_path)
            output_path = base + '_downscaled' + ext
        scale_filter = f"scale='min({max_width},iw)':'min({max_height},ih)':force_original_aspect_ratio=decrease"
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', scale_filter,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-c:a', 'copy',
            '-y', output_path
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0:
            print(f"Video downscaled to {output_path} (LANCZOS via ffmpeg)")
            return output_path
        print(f"Video downscale failed, using original")
        return video_path
    except Exception as e:
        print(f"Video downscale error: {e}")
        return video_path


def check_and_downscale_input(input_path, max_width, max_height):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.tiff'):
        try:
            import cv2
            img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                h, w = img.shape[:2]
                if w > max_width or h > max_height:
                    return downscale_image(input_path, max_width, max_height)
        except Exception:
            try:
                from PIL import Image
                img = Image.open(input_path)
                w, h = img.size
                if w > max_width or h > max_height:
                    return downscale_image(input_path, max_width, max_height)
            except Exception:
                pass
    elif ext in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'):
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=width,height', '-of', 'csv=p=0', input_path],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split(',')
                w, h = int(parts[0]), int(parts[1])
                if w > max_width or h > max_height:
                    return downscale_video(input_path, max_width, max_height)
        except Exception:
            pass
    return input_path


def validate_resolution(resolution_str, supported_resolutions, default_resolution, max_dimension=None):
    if not resolution_str:
        return default_resolution
    try:
        parts = resolution_str.lower().split('x')
        w, h = int(parts[0]), int(parts[1])
        if max_dimension and max(w, h) > max_dimension:
            ratio = max_dimension / max(w, h)
            w, h = int(w * ratio), int(h * ratio)
            w = (w // 8) * 8
            h = (h // 8) * 8
            print(f"Warning: resolution {resolution_str} exceeds max {max_dimension}, downscaled to {w}x{h}")
            return f"{w}x{h}"
        if supported_resolutions and resolution_str.lower() not in supported_resolutions:
            print(f"Warning: resolution {resolution_str} not in supported list {supported_resolutions}. Using default {default_resolution}")
            return default_resolution
        return resolution_str
    except Exception:
        print(f"Warning: invalid resolution format '{resolution_str}'. Using default {default_resolution}")
        return default_resolution
