KLARIFY_DLC_NAME = "klarify"
KLARIFY_DLC_DESCRIPTION = "Klarify DLC — upscale, enhance (denoise+deblur), and frame interpolation"

KLARIFY_MODES = {
    'upscale': {
        'name': 'Upscale',
        'description': 'Upscale x4 then -2 with LANCZOS for double pixels without fine-artifacts. Supports images and videos.',
    },
    'enhance': {
        'name': 'Enhance',
        'description': 'Denoise + deblur in one pass. Supports images and videos.',
    },
    'interpolate': {
        'name': 'Frame Interpolation',
        'description': 'Interpolate video frames with RIFE v4.25 for smooth slow-motion or higher FPS. Video only.',
    },
}
