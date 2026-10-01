from hydra.core.config_store import ConfigStore

from lyra_2._ext.imaginaire.lazy_config import LazyCall as L
from lyra_2._src.callbacks.training import (
    LocalGradClipCallback,
    LocalVideoCallback,
    Lyra2LowPrecisionCallback,
    TensorBoardCallback,
)

cs = ConfigStore.instance()


def register_lyra2():
    """Fully-flattened lyra_2 spatial training experiment.

    Effective config equivalent to:
    two_buffers_dl3dv_image_tokens_correspondence_finetune_kq_only_multibuffer_add_depth_hsg
    in the source repo.
    """
    experiment_config = dict(
        defaults=[
            {"override /model": "fsdp_wan2pt1_lyra2_spatial"},
            {"override /net": "wan2pt1_14B_i2v_lyra2"},
            {"override /conditioner": "lyra2_conditioner"},
            {"override /data_train": "lyra2_sample_data"},
            {"override /data_val": "lyra2_sample_data"},
            {"override /ckpt_type": "dcp"},
            "_self_",
        ],
        job=dict(
            project="lyra_2",
            group="lyra2",
            name="lyra2",
        ),
        model=dict(
            config=dict(
                ema=dict(enabled=False),
                framepack_type="f1k1f4s2f1s1f16k4f2k2f1k1_g20",
                max_segments=13,
                apply_corruption_to_spatial_region="noise_with_sigma",
                augment_sigma_sample_p_mean=-3.0,
                augment_sigma_sample_p_std=2.0,
                augment_sigma_sample_multiplier=1.0,
                self_aug_enabled=True,
                self_aug_steps=1,
                self_aug_guidance=1.0,
                self_aug_scheduler_shift=1.0,
                self_aug_every_k=2,
                self_aug_prob=1.0,
                self_aug_max_T=500,
                self_aug_copy_chunk=True,
                self_aug_encode_gt_with_clean_history=True,
                starting_frame_ratio=0.0,
                use_mp_policy_fsdp=True,
                keep_original_net_dtype=True,
                train_time_weight="reweighting",
                spatial_memory_use_image=True,
                spatial_memory_stride=8,
                spatial_memory_skip_recent=16,
                warp_chunk_size=16,
                framepack_trainable_modules="cam_encoder,buffer_encoder,self_attn,clean_patch_embeddings,patch_embedding",
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        optimizer=dict(
            lr=3e-5,
            weight_decay=1e-3,
        ),
        scheduler=dict(
            warm_up_steps=[100],
            cycle_lengths=[400_000],
            f_start=[1.0e-6],
            f_max=[0.99],
            f_min=[0.4],
        ),
        checkpoint=dict(
            save_iter=100,
            save_to_object_store=dict(enabled=False),
            load_from_object_store=dict(enabled=False),
            load_path="./checkpoints/pretrained/Wan2.1-I2V-14B-480P/Wan2.1-I2V-14B-480P.pth",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=1000000,
            logging_iter=50,
            callbacks=dict(
                low_precision=L(Lyra2LowPrecisionCallback)(),
                grad_clip=L(LocalGradClipCallback)(clip_norm=0.1, force_finite=True),
                tensorboard=L(TensorBoardCallback)(),
                frame_pack_viz_sampling=L(LocalVideoCallback)(
                    every_n=400,
                    num_sampling_step=35,
                    guidance=1.0,
                    fps=16,
                ),
            ),
        ),
        upload_reproducible_setup=False,
    )

    cs.store(
        group="experiment",
        package="_global_",
        name="lyra2",
        node=experiment_config,
    )


register_lyra2()
