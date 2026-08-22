import os
import sys
import time
import tempfile
import subprocess
import shutil

_src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

KLARIFY_MODEL_DIR = os.path.join(_src_dir, "models", "checkpoints", "klarify")
KLARIFY_RESULTS_DIR = os.path.join("results", "DLCs", "klarify")

KLARIFY_GDRIVE_IDS = {
    'deblur': '1S0PVRbyTakYY9a82kujgZLbMihfNBLfC',
    'denoise': '14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR',
    'upscale': '1EioFq5-mKmv1uqta_Byd9cgXp9SU3zjj',
    'rife': '1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg',
}

KLARIFY_MODEL_FILES = {
    'deblur': 'deblur-heavy.pth',
    'denoise': 'denoise-heavy.pth',
    'upscale': 'upscale-heavy.pth',
    'rife': 'framegen-heavy.pkl',
}

KLARIFY_NAFNET_CONFIGS = {
    'deblur': {
        'width': 64,
        'middle_blk_num': 1,
        'enc_blk_nums': [1, 1, 1, 28],
        'dec_blk_nums': [1, 1, 1, 1],
    },
    'denoise': {
        'width': 64,
        'middle_blk_num': 12,
        'enc_blk_nums': [2, 2, 4, 8],
        'dec_blk_nums': [2, 2, 2, 2],
    },
}

_klarify_models = {}
_klarify_device = None


def _get_device():
    global _klarify_device
    if _klarify_device is not None:
        return _klarify_device
    try:
        import torch
        if torch.cuda.is_available():
            _klarify_device = torch.device('cuda:0')
        else:
            _klarify_device = torch.device('cpu')
    except ImportError:
        _klarify_device = 'cpu'
    return _klarify_device


def _download_gdrive(file_id, output_path):
    import urllib.request
    import urllib.parse
    session_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    try:
        req = urllib.request.Request(session_url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
        confirm_url = f'https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t'
        req2 = urllib.request.Request(confirm_url)
        with urllib.request.urlopen(req2, timeout=300) as resp2:
            data = resp2.read()
        with open(output_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"Download error: {e}")
        return False


def _ensure_klarify_model(model_key):
    model_path = os.path.join(KLARIFY_MODEL_DIR, KLARIFY_MODEL_FILES[model_key])
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        return model_path
    os.makedirs(KLARIFY_MODEL_DIR, exist_ok=True)
    file_id = KLARIFY_GDRIVE_IDS.get(model_key)
    if not file_id:
        print(f"Error: no download source for klarify model '{model_key}'")
        return None
    print(f"Downloading {model_key} model from Google Drive...")
    if not _download_gdrive(file_id, model_path):
        print(f"Error: failed to download {model_key} model")
        return None
    print(f"Downloaded: {model_path}")
    return model_path


def _load_deblur_model():
    global _klarify_models
    if 'deblur' in _klarify_models:
        return _klarify_models['deblur']
    import torch
    model_path = _ensure_klarify_model('deblur')
    if not model_path:
        return None
    sys.path.insert(0, os.path.dirname(__file__))
    from voders.DLCs.klarify.nafnet_arch import NAFNetLocal
    config = KLARIFY_NAFNET_CONFIGS['deblur']
    model = NAFNetLocal(
        img_channel=3,
        width=config['width'],
        middle_blk_num=config['middle_blk_num'],
        enc_blk_nums=config['enc_blk_nums'],
        dec_blk_nums=config['dec_blk_nums'],
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params', checkpoint.get('state_dict', checkpoint))
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(_get_device())
    model.eval()
    _klarify_models['deblur'] = model
    print(f"Loaded deblur model (NAFNet-GoPro-width64)")
    return model


def _load_denoise_model():
    global _klarify_models
    if 'denoise' in _klarify_models:
        return _klarify_models['denoise']
    import torch
    model_path = _ensure_klarify_model('denoise')
    if not model_path:
        return None
    sys.path.insert(0, os.path.dirname(__file__))
    from voders.DLCs.klarify.nafnet_arch import NAFNet
    config = KLARIFY_NAFNET_CONFIGS['denoise']
    model = NAFNet(
        img_channel=3,
        width=config['width'],
        middle_blk_num=config['middle_blk_num'],
        enc_blk_nums=config['enc_blk_nums'],
        dec_blk_nums=config['dec_blk_nums'],
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params', checkpoint.get('state_dict', checkpoint))
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(_get_device())
    model.eval()
    _klarify_models['denoise'] = model
    print(f"Loaded denoise model (NAFNet-SIDD-width64)")
    return model


def _load_upscale_model():
    global _klarify_models
    if 'upscale' in _klarify_models:
        return _klarify_models['upscale']
    import torch
    model_path = _ensure_klarify_model('upscale')
    if not model_path:
        return None
    sys.path.insert(0, os.path.dirname(__file__))
    from voders.DLCs.klarify.hat_gan_arch import HAT
    model = HAT(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='pixelshuffle',
        resi_connection='1conv',
    )
    checkpoint = torch.load(model_path, map_location='cpu')
    state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint.get('state_dict', checkpoint)))
    for k in list(state_dict.keys()):
        if k.startswith('module.'):
            state_dict[k[7:]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model = model.to(_get_device())
    model.eval()
    _klarify_models['upscale'] = model
    print(f"Loaded upscale model (Real-HAT-GAN-sharper)")
    return model


def _load_rife_model():
    global _klarify_models
    if 'rife' in _klarify_models:
        return _klarify_models['rife']
    import torch
    model_path = _ensure_klarify_model('rife')
    if not model_path:
        return None
    sys.path.insert(0, os.path.dirname(__file__))
    from voders.DLCs.klarify.rife_arch import RIFE
    model = RIFE(mode='heavy')
    model.load_model(KLARIFY_MODEL_DIR, mode='heavy')
    model.eval()
    model.device()
    _klarify_models['rife'] = model
    print(f"Loaded frame interpolation model (RIFE v4.25)")
    return model


def _img2tensor(img):
    import cv2
    import numpy as np
    import torch
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return img.to(_get_device())


def _tensor2img(tensor):
    import cv2
    import numpy as np
    tensor = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    tensor = cv2.cvtColor(tensor, cv2.COLOR_RGB2BGR)
    return tensor


def _pad_image(img_tensor, modulo=32):
    import torch
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    new_h = ((h - 1) // modulo + 1) * modulo
    new_w = ((w - 1) // modulo + 1) * modulo
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h > 0 or pad_w > 0:
        img_tensor = torch.nn.functional.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
    return img_tensor, (h, w)


def _process_nafnet(model, img_tensor):
    import torch
    with torch.no_grad():
        padded, (h, w) = _pad_image(img_tensor)
        output = model(padded)
        return output[:, :, :h, :w]


def _process_upscale_tensor(model, img_tensor):
    import torch
    with torch.no_grad():
        padded, (h, w) = _pad_image(img_tensor, modulo=16)
        output = model(padded)
        new_h, new_w = h * 4, w * 4
        return output[:, :, :new_h, :new_w]


def klarify_denoise(input_path, output_path):
    import cv2
    import torch
    model = _load_denoise_model()
    if model is None:
        return False
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image: {input_path}")
        return False
    tensor = _img2tensor(img)
    output = _process_nafnet(model, tensor)
    result = _tensor2img(output)
    cv2.imwrite(output_path, result)
    print(f"Denoised: {output_path}")
    return True


def klarify_deblur(input_path, output_path):
    import cv2
    model = _load_deblur_model()
    if model is None:
        return False
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image: {input_path}")
        return False
    tensor = _img2tensor(img)
    output = _process_nafnet(model, tensor)
    result = _tensor2img(output)
    cv2.imwrite(output_path, result)
    print(f"Deblurred: {output_path}")
    return True


def klarify_enhance(input_path, output_path):
    import cv2
    denoise_model = _load_denoise_model()
    if denoise_model is None:
        return False
    deblur_model = _load_deblur_model()
    if deblur_model is None:
        return False
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image: {input_path}")
        return False
    tensor = _img2tensor(img)
    print("Denoising...")
    tensor = _process_nafnet(denoise_model, tensor)
    print("Deblurring...")
    tensor = _process_nafnet(deblur_model, tensor)
    result = _tensor2img(tensor)
    cv2.imwrite(output_path, result)
    print(f"Enhanced (denoise + deblur): {output_path}")
    return True


def klarify_upscale(input_path, output_path):
    import cv2
    import numpy as np
    model = _load_upscale_model()
    if model is None:
        return False
    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: could not read image: {input_path}")
        return False
    tensor = _img2tensor(img)
    print("Upscaling x4 (Real-HAT-GAN-sharper)...")
    output = _process_upscale_tensor(model, tensor)
    result = _tensor2img(output)
    h, w = result.shape[:2]
    new_h, new_w = h // 2, w // 2
    result = cv2.resize(result, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(output_path, result)
    print(f"Upscaled (x4 then -2 LANCZOS): {output_path}")
    return True


def klarify_interpolate(input_path, output_path, multi=2):
    import cv2
    import numpy as np
    if multi < 2:
        print("Error: interpolate multi must be >= 2")
        return False
    model = _load_rife_model()
    if model is None:
        return False
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v'):
        print("Error: interpolate requires a video file")
        return False
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, 'frames')
    output_frames_dir = os.path.join(temp_dir, 'output_frames')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)
    try:
        print("Extracting frames...")
        result = subprocess.run(
            ['ffmpeg', '-i', input_path, '-q:v', '2', os.path.join(frames_dir, 'frame_%06d.png')],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            print("Error: failed to extract frames")
            return False
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if len(frames) < 2:
            print("Error: need at least 2 frames for interpolation")
            return False
        probe_result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', input_path],
            capture_output=True, text=True, timeout=10
        )
        fps_str = probe_result.stdout.strip() if probe_result.returncode == 0 else '30/1'
        try:
            num, den = fps_str.split('/')
            fps = float(num) / float(den)
        except Exception:
            fps = 30.0
        new_fps = fps * multi
        print(f"Interpolating x{multi} (RIFE v4.25)... original FPS: {fps:.1f} -> {new_fps:.1f}")
        output_idx = 0
        for i in range(len(frames) - 1):
            img0 = cv2.imread(os.path.join(frames_dir, frames[i]))
            img1 = cv2.imread(os.path.join(frames_dir, frames[i + 1]))
            h, w = img0.shape[:2]
            divisor = 32
            new_h = ((h - 1) // divisor + 1) * divisor
            new_w = ((w - 1) // divisor + 1) * divisor
            if new_h > h or new_w > w:
                img0 = np.pad(img0, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='edge')
                img1 = np.pad(img1, ((0, new_h - h), (0, new_w - w), (0, 0)), mode='edge')
            img0_tensor = _img2tensor(img0)
            img1_tensor = _img2tensor(img1)
            cv2.imwrite(os.path.join(output_frames_dir, f'{output_idx:08d}.png'), img0[:h, :w])
            output_idx += 1
            for j in range(multi - 1):
                timestep = (j + 1) / multi
                import torch
                with torch.no_grad():
                    mid = model.inference(img0_tensor, img1_tensor, timestep, 1.0)
                mid_img = _tensor2img(mid)
                cv2.imwrite(os.path.join(output_frames_dir, f'{output_idx:08d}.png'), mid_img[:h, :w])
                output_idx += 1
        last_frame = cv2.imread(os.path.join(frames_dir, frames[-1]))
        cv2.imwrite(os.path.join(output_frames_dir, f'{output_idx:08d}.png'), last_frame)
        print("Reassembling video with interpolated frames...")
        temp_video = os.path.join(temp_dir, 'output_no_audio.mp4')
        result = subprocess.run(
            ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(output_frames_dir, '%08d.png'),
             '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
             '-y', temp_video],
            capture_output=True, timeout=300
        )
        if result.returncode != 0:
            print("Error: failed to reassemble video")
            return False
        result = subprocess.run(
            ['ffmpeg', '-i', temp_video, '-i', input_path, '-c:v', 'copy', '-c:a', 'aac',
             '-map', '0:v:0', '-map', '1:a:0?', '-y', output_path],
            capture_output=True, timeout=120
        )
        if result.returncode != 0:
            shutil.move(temp_video, output_path)
        print(f"Interpolated: {output_path} ({new_fps:.1f} FPS)")
        return True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def klarify_cleanup():
    global _klarify_models
    for key in list(_klarify_models.keys()):
        del _klarify_models[key]
    _klarify_models = {}
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
