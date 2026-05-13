import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from modeling.image_decoder import ImageDecoderMulti
from modeling.image_encoder import ImageEncoderViT
from modeling.losses import MixedL1GradientLoss
from modeling.mript import MRIPT
from modeling.mript_trainer import TrainerMulti
from modeling.prompt_encoder import PromptEncoderMulti
from modeling.transformer import TwoWayTransformer
from utils.data_transform import DataTransform
from utils.help_func import load_train_config, setup_save_dir
from utils.radimgnet_loader_ipt import create_radimgnet_dataloader_multi
from utils.sample_mask import (
    EquiSpaceMask,
    RandomMask,
    RandomMaskGaussian,
    RandomMaskGaussian1D,
)


DEFAULT_CONFIG_PATH = Path("configs/radimagenet_pretrain.yaml")


def parse_args():
    """Parse CLI arguments for YAML-driven training."""
    parser = argparse.ArgumentParser(description="Train MRIPT from a YAML config file")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to training YAML config file",
    )
    return parser.parse_args()


def build_mask_func_list(input_height, input_width, seed=None):
    acc0, frac_c0 = 2.0, 0.1
    acc1, frac_c1 = 4.0, 0.08
    acc2, frac_c2 = 6.0, 0.06
    acc3, frac_c3 = 8.0, 0.04
    acc4, frac_c4 = 10.0, 0.02

    # Cartesian random masks
    random_masks = [
        DataTransform(
            RandomMask(
                center_fraction=frac_c0,
                acceleration=acc0,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMask(
                center_fraction=frac_c1,
                acceleration=acc1,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMask(
                center_fraction=frac_c2,
                acceleration=acc2,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMask(
                center_fraction=frac_c3,
                acceleration=acc3,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMask(
                center_fraction=frac_c4,
                acceleration=acc4,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
    ]

    # Cartesian equi-space masks
    equispace_masks = [
        DataTransform(
            EquiSpaceMask(
                center_fraction=frac_c0,
                acceleration=acc0,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            EquiSpaceMask(
                center_fraction=frac_c1,
                acceleration=acc1,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            EquiSpaceMask(
                center_fraction=frac_c2,
                acceleration=acc2,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            EquiSpaceMask(
                center_fraction=frac_c3,
                acceleration=acc3,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            EquiSpaceMask(
                center_fraction=frac_c4,
                acceleration=acc4,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
    ]

    # 1D Gaussian masks
    gaussian_1d_masks = [
        DataTransform(
            RandomMaskGaussian1D(
                center_fraction=frac_c0,
                acceleration=acc0,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMaskGaussian1D(
                center_fraction=frac_c1,
                acceleration=acc1,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMaskGaussian1D(
                center_fraction=frac_c2,
                acceleration=acc2,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMaskGaussian1D(
                center_fraction=frac_c3,
                acceleration=acc3,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
        DataTransform(
            RandomMaskGaussian1D(
                center_fraction=frac_c4,
                acceleration=acc4,
                size=[1, input_height, input_width],
                seed=seed,
            )
        ),
    ]

    # 2D Gaussian masks
    gaussian_2d_masks = [
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c0**0.5,
                acceleration=acc0,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c1**0.5,
                acceleration=acc1,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c2**0.5,
                acceleration=acc2,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c3**0.5,
                acceleration=acc3,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c4**0.5,
                acceleration=acc4,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
    ]

    scales = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    func_list = [random_masks, equispace_masks, gaussian_1d_masks, gaussian_2d_masks]
    return scales, func_list


def build_model(scales, input_height, mode, device):
    encoder_embed_dim = 1024
    encoder_depth = 24
    encoder_num_heads = 16
    encoder_global_attn_indexes = [5, 11, 17, 23]
    in_feat = 64
    output_dim_factor = 4
    prompt_embed_dim = in_feat * output_dim_factor
    image_size = input_height
    vit_patch_size = 4
    n_colors = 1
    mlp_ratio = 4
    image_embedding_size = image_size // vit_patch_size

    image_encoder = ImageEncoderViT(
        depth=encoder_depth,
        in_chans=in_feat,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )

    prompt_encoder = PromptEncoderMulti(
        num_level=len(scales[0]),
        num_type=len(scales),
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
    )

    image_decoder = ImageDecoderMulti(
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
            attention_downsample_rate=1,
        ),
        transformer_dim=prompt_embed_dim,
        output_dim_factor=output_dim_factor,
    )

    model = MRIPT(
        n_feats=in_feat,
        n_colors=n_colors,
        scale=scales,
        conv_kernel_size=3,
        res_kernel_size=5,
        image_encoder=image_encoder,
        prompt_encoder=prompt_encoder,
        image_decoder=image_decoder,
        mode=mode,
        device=device,
    )
    return model


def load_model_checkpoint(model, resume_from, device):
    """Optionally load model weights from checkpoint path.

    This loads model parameters only. Optimizer, scheduler, epoch counters,
    and all other runtime settings continue to use the current YAML config.
    """
    if resume_from is None:
        return

    checkpoint_path = Path(resume_from)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"resume_from checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    print(f"Loaded model weights from: {checkpoint_path}")


def main():
    args = parse_args()
    config = load_train_config(args.config)
    print(f"Loaded config: {config['config_path']}")

    if not Path(config["dataset_path"]).exists():
        raise FileNotFoundError(f"Dataset path not found: {config['dataset_path']}")

    scales, func_list = build_mask_func_list(
        config["input_height"],
        config["input_width"],
        seed=config["seed"],
    )

    dataloader_train = create_radimgnet_dataloader_multi(
        data_dir=config["dataset_path"],
        random_seed=0,
        val_split=config["val_split"],
        image_size=(config["input_height"], config["input_width"]),
        batch_size=config["batch_size"],
        is_distributed=False,
        is_train=True,
        scales=scales,
        func_list=func_list,
        num_workers=config["num_workers"],
        fix_scale_idx=None,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=config["persistent_workers"],
        use_precomputed_mask=config["use_precomputed_mask"],
    )

    dataloader_test = create_radimgnet_dataloader_multi(
        data_dir=config["dataset_path"],
        random_seed=0,
        val_split=config["val_split"],
        image_size=(config["input_height"], config["input_width"]),
        batch_size=1,
        is_distributed=False,
        is_train=False,
        scales=scales,
        func_list=func_list,
        num_workers=config["num_workers"],
        fix_scale_idx=None,
        prefetch_factor=config["prefetch_factor"],
        persistent_workers=config["persistent_workers"],
        use_precomputed_mask=config["use_precomputed_mask"],
    )

    print(torch.__version__)
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable, fallback from {config['device']} to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(config["device"])
    print("device:", device)

    model = build_model(
        scales=scales,
        input_height=config["input_height"],
        mode=config["mode"],
        device=device,
    )
    model = model.to(device)
    load_model_checkpoint(model, config["resume_from"], device)

    run_path_model = setup_save_dir(config)
    print("PATH_MODEL:", run_path_model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        amsgrad=False,
    )
    if config["loss"] == "mixed_l1_grad":
        criterion = MixedL1GradientLoss(gradient_weight=0.1)
    elif config["loss"] == "l2":
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    print(f"loss: {config['loss']}")

    trainer = TrainerMulti(
        loader_train=dataloader_train,
        loader_test=dataloader_test,
        my_model=model,
        my_loss=criterion,
        optimizer=optimizer,
        PATH_MODEL=run_path_model,
        device=device,
        NUM_EPOCH=config["num_epochs"],
    )

    trainer.train()


if __name__ == "__main__":
    main()
