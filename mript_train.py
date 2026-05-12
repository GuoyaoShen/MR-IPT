import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

from utils.help_func import create_path
from modeling.image_decoder import ImageDecoderMulti
from modeling.image_encoder import ImageEncoderViT
from modeling.mript import MRIPT
from modeling.mript_trainer import TrainerMulti
from modeling.prompt_encoder import PromptEncoderMulti
from modeling.transformer import TwoWayTransformer
from utils.data_transform import DataTransform
from utils.radimgnet_loader_ipt import create_radimgnet_dataloader_multi
from utils.sample_mask import (
    EquiSpaceMask,
    RandomMask,
    RandomMaskGaussian,
    RandomMaskGaussian1D,
)


def build_mask_func_list(input_height, input_width, seed=None):
    acc0, frac_c0 = 2.0, 0.1
    acc1, frac_c1 = 4.0, 0.08
    acc2, frac_c2 = 6.0, 0.06
    acc3, frac_c3 = 8.0, 0.04
    acc4, frac_c4 = 10.0, 0.02

    # Cartesian random masks
    random_masks = [
        DataTransform(RandomMask(center_fraction=frac_c0, acceleration=acc0, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMask(center_fraction=frac_c1, acceleration=acc1, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMask(center_fraction=frac_c2, acceleration=acc2, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMask(center_fraction=frac_c3, acceleration=acc3, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMask(center_fraction=frac_c4, acceleration=acc4, size=[1, input_height, input_width], seed=seed)),
    ]

    # Cartesian equi-space masks
    equispace_masks = [
        DataTransform(EquiSpaceMask(center_fraction=frac_c0, acceleration=acc0, size=[1, input_height, input_width], seed=seed)),
        DataTransform(EquiSpaceMask(center_fraction=frac_c1, acceleration=acc1, size=[1, input_height, input_width], seed=seed)),
        DataTransform(EquiSpaceMask(center_fraction=frac_c2, acceleration=acc2, size=[1, input_height, input_width], seed=seed)),
        DataTransform(EquiSpaceMask(center_fraction=frac_c3, acceleration=acc3, size=[1, input_height, input_width], seed=seed)),
        DataTransform(EquiSpaceMask(center_fraction=frac_c4, acceleration=acc4, size=[1, input_height, input_width], seed=seed)),
    ]

    # 1D Gaussian masks
    gaussian_1d_masks = [
        DataTransform(RandomMaskGaussian1D(center_fraction=frac_c0, acceleration=acc0, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMaskGaussian1D(center_fraction=frac_c1, acceleration=acc1, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMaskGaussian1D(center_fraction=frac_c2, acceleration=acc2, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMaskGaussian1D(center_fraction=frac_c3, acceleration=acc3, size=[1, input_height, input_width], seed=seed)),
        DataTransform(RandomMaskGaussian1D(center_fraction=frac_c4, acceleration=acc4, size=[1, input_height, input_width], seed=seed)),
    ]

    # 2D Gaussian masks
    gaussian_2d_masks = [
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c0 ** 0.5,
                acceleration=acc0,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c1 ** 0.5,
                acceleration=acc1,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c2 ** 0.5,
                acceleration=acc2,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c3 ** 0.5,
                acceleration=acc3,
                size=[1, input_height, input_width],
                seed=seed,
                cov=[[1.5, 0], [0, 1.5]],
            )
        ),
        DataTransform(
            RandomMaskGaussian(
                center_fraction=frac_c4 ** 0.5,
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train MRIPT from script converted from mript_train.ipynb")
    parser.add_argument("--dataset-path", type=str, default="/bigdata/RadImageNet/rin2d/radiology_ai/MR/brain/normal")
    parser.add_argument("--device", type=str, default="cuda:1", help="Torch device string, e.g. cpu, cuda:0, cuda:1")
    parser.add_argument("--input-height", type=int, default=128)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-persistent-workers", action="store_true", help="Disable DataLoader persistent workers")
    parser.add_argument("--use-precomputed-mask", action="store_true", default=False, help="Precompute and reuse masks per type/level to reduce loader overhead")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epoch", type=int, default=5)
    parser.add_argument("--mode", type=str, default="type", choices=["type", "level", "combine"])
    parser.add_argument("--path-model", type=str, default="./saved_models/", help="Directory where model checkpoints are saved")
    parser.add_argument("--no-save", action="store_true", help="Disable checkpoint saving in TrainerMulti")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")

    scales, func_list = build_mask_func_list(args.input_height, args.input_width, seed=args.seed)

    dataloader_train = create_radimgnet_dataloader_multi(
        data_dir=args.dataset_path,
        random_seed=0,
        val_split=args.val_split,
        image_size=(args.input_height, args.input_width),
        batch_size=args.batch_size,
        is_distributed=False,
        is_train=True,
        scales=scales,
        func_list=func_list,
        num_workers=args.num_workers,
        fix_scale_idx=None,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        use_precomputed_mask=args.use_precomputed_mask,
    )

    dataloader_test = create_radimgnet_dataloader_multi(
        data_dir=args.dataset_path,
        random_seed=0,
        val_split=args.val_split,
        image_size=(args.input_height, args.input_width),
        batch_size=1,
        is_distributed=False,
        is_train=False,
        scales=scales,
        func_list=func_list,
        num_workers=args.num_workers,
        fix_scale_idx=None,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=not args.no_persistent_workers,
        use_precomputed_mask=args.use_precomputed_mask,
    )

    print(torch.__version__)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA unavailable, fallback from {args.device} to cpu")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print("device:", device)

    model = build_model(scales=scales, input_height=args.input_height, mode=args.mode, device=device)

    create_path(args.path_model)
    print("PATH_MODEL:", args.path_model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=False)
    criterion = nn.L1Loss()

    trainer = TrainerMulti(
        loader_train=dataloader_train,
        loader_test=dataloader_test,
        my_model=model,
        my_loss=criterion,
        optimizer=optimizer,
        PATH_MODEL=args.path_model,
        device=device,
        NUM_EPOCH=args.num_epoch,
        RESUME_EPOCH=0,
        if_save=not args.no_save,
    )

    trainer.train(show_step=1, show_test=True)


if __name__ == "__main__":
    main()
