import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Ensure repo root is importable when running this script from test/.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# %%
def _to_2d(x):
    if x.ndim == 3:
        return x[0]
    return x


def save_sample_figure(
    image_input, image_target, mask, title, out_path=None, show=False
):
    img_in = _to_2d(image_input.detach().cpu())
    img_tg = _to_2d(image_target.detach().cpu())
    img_mask = _to_2d(mask.detach().cpu())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_in, cmap="gray")
    axes[0].set_title("Input (undersampled)")
    axes[0].axis("off")

    axes[1].imshow(img_tg, cmap="gray")
    axes[1].set_title("Target")
    axes[1].axis("off")

    axes[2].imshow(img_mask, cmap="gray")
    axes[2].set_title("Mask")
    axes[2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, dpi=120)

    if show:
        plt.show()

    plt.close(fig)


# %%
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview MRIPT dataloader samples and save example images"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="/bigdata/RadImageNet/rin2d/radiology_ai/MR/",
    )
    parser.add_argument("--input-height", type=int, default=128)
    parser.add_argument("--input-width", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--max-samples-per-batch", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT_DIR / "tests" / "debug_loader_preview"),
        help="Directory where preview images are saved",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures inline (useful in Interactive Window)",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save preview images to disk"
    )
    if argv is None:
        # In Jupyter/Interactive Window, sys.argv often contains extra kernel flags.
        # Ignore unknown args so this utility can run directly in cells.
        args, _ = parser.parse_known_args()
        return args
    return parser.parse_args(argv)


def get_default_args():
    """Convenience defaults for Interactive Window usage."""
    return parse_args([])


# %%
def run_preview(args):
    from mript_train import build_mask_func_list
    from utils.radimgnet_loader_ipt import create_radimgnet_dataloader_multi

    if args.dataset_path is None:
        raise ValueError("dataset_path is required")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    out_dir = None
    if not args.no_save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    scales, func_list = build_mask_func_list(
        args.input_height, args.input_width, seed=args.seed
    )

    loader = create_radimgnet_dataloader_multi(
        data_dir=str(dataset_path),
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
    )

    print(f"Total train batches: {len(loader)}")
    saved = 0

    for batch_idx, (images, levels, filenames, idx, mask, types) in enumerate(loader):
        if batch_idx >= args.num_batches:
            break

        images_input = images[0]
        images_target = images[1]

        print(
            f"Batch {batch_idx}: input={tuple(images_input.shape)} target={tuple(images_target.shape)} "
            f"levels={tuple(levels.shape)} types={tuple(types.shape)}"
        )

        per_batch = min(images_input.shape[0], args.max_samples_per_batch)
        for i in range(per_batch):
            level_i = int(levels[i].item())
            type_i = int(types[i].item())
            filename_i = filenames[i]

            out_path = None
            if out_dir is not None:
                out_path = (
                    out_dir
                    / f"batch{batch_idx:03d}_sample{i:02d}_type{type_i}_level{level_i}.png"
                )

            title = f"batch={batch_idx} sample={i} type={type_i} level={level_i}"
            save_sample_figure(
                images_input[i],
                images_target[i],
                mask[i],
                title,
                out_path=out_path,
                show=args.show,
            )

            print(f"  sample {i}: type={type_i} level={level_i} file={filename_i}")
            if out_path is not None:
                print(f"  saved: {out_path}")
            saved += 1

    if out_dir is not None:
        print(f"Done. Saved {saved} preview image(s) to {out_dir}")
    else:
        print(f"Done. Rendered {saved} preview image(s) (save disabled)")


# %%
def main(argv=None):
    args = parse_args(argv)
    run_preview(args)


if __name__ == "__main__":
    main()


# %%
# Interactive Window quick start:
# args = get_default_args()
# args.dataset_path = "/bigdata/RadImageNet/rin2d/radiology_ai/MR/"
# args.num_batches = 1
# args.max_samples_per_batch = 4
# args.show = True
# args.no_save = True
# run_preview(args)
