import torch
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset


def _list_images_recursive(root_dir):
    """Return all image file paths under ``root_dir`` recursively."""
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    image_paths = []
    for root, _, files in os.walk(root_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext in image_exts:
                image_paths.append(os.path.join(root, name))
    return image_paths


def create_radimgnet_dataloader_multi(
    data_dir,
    random_seed,
    val_split,
    image_size,
    batch_size,
    is_distributed=False,
    is_train=False,
    scales=[1],
    func_list=[None],
    num_workers=0,
    fix_scale_idx=None,
    prefetch_factor=2,
    persistent_workers=True,
    use_precomputed_mask=False,
):
    """Create a RadImageNet dataloader for MRIPT.

    Args:
        data_dir: Root directory containing images.
        random_seed: Seed used for deterministic train/validation splitting.
        val_split: Fraction of samples reserved for validation.
        image_size: Output image size as ``(height, width)``.
        batch_size: Loader batch size.
        is_distributed: Whether to wrap dataset with ``DistributedSampler``.
        is_train: Whether this loader is for training mode.
        scales: Per-type scale configuration used by MRIPT heads/tails.
        func_list: Per-type/per-level preprocessing callables.
        num_workers: Number of dataloader worker processes.
        fix_scale_idx: Optional fixed ``(type_idx, level_idx)``.
        prefetch_factor: Number of prefetched batches per worker.
        persistent_workers: Keep workers alive between epochs.
        use_precomputed_mask: Reuse one precomputed mask for each
            ``(type_idx, level_idx)`` combination.

    Returns:
        ``torch.utils.data.DataLoader`` with MRIPT trainer-compatible tuples.
    """

    if not data_dir:
        raise ValueError("unspecified dta directory.")

    # create dataset
    dataset = RadimgnetDataSetMulti(
        data_dir=data_dir,
        random_seed=random_seed,
        val_split=val_split,
        image_size=image_size,
        scales=scales,
        func_list=func_list,
        fix_scale_idx=fix_scale_idx,
        is_train=is_train,
        use_precomputed_mask=use_precomputed_mask,
    )

    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(dataset)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    loader = DataLoader(**loader_kwargs)

    # return loader
    return loader


class RadimgnetDataSetMulti(Dataset):
    """RadImageNet dataset used by MRIPT.

    Each sample returns a tuple formatted for ``TrainerMulti``:
    ``((image_input, image_target), level_idx, filename, idx, mask, type_idx)``.
    """

    def __init__(
        self,
        data_dir,
        random_seed,
        val_split,
        image_size,
        scales,
        func_list,
        fix_scale_idx,
        is_train,
        use_precomputed_mask=False,
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.scales = scales
        self.func_list = func_list
        self.fix_scale_idx = fix_scale_idx
        self.is_train = is_train
        self.use_precomputed_mask = use_precomputed_mask

        if not len(self.func_list) == len(self.scales):
            raise ValueError("unequal number of scales and function list.")

        # construct training and validation dataset
        imagePaths = _list_images_recursive(self.data_dir)
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print("train size: " + str(len(self.trainPaths)))
        print("validation size: " + str(len(self.valPaths)))

        # Build reusable transform once instead of re-creating it in __getitem__.
        transform_ops = []
        if self.is_train:
            transform_ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                ]
            )
        transform_ops.append(transforms.Resize(size=self.image_size))
        transform_ops.append(transforms.PILToTensor())
        self.image_transform = transforms.Compose(transform_ops)

        # Optional cache for deterministic masks (same type/level and image size).
        self.precomputed_masks = {}
        if self.use_precomputed_mask:
            self._build_precomputed_masks()

    def _build_precomputed_masks(self):
        """Precompute one deterministic mask per ``(type_idx, level_idx)``.

        Logic:
            - Build a synthetic sample matching the configured image size.
            - For each preprocessing function in ``func_list``, run it once.
            - Cache only the mask in ``self.precomputed_masks``.

        During ``__getitem__``, when ``use_precomputed_mask`` is enabled, the
        cached mask for the selected ``(type_idx, level_idx)`` is reused for all
        samples in that bucket. This improves speed but reduces mask randomness.
        """
        sample = torch.zeros(
            (1, self.image_size[0], self.image_size[1]), dtype=torch.float32
        )
        for i_type in range(len(self.scales)):
            for i_level in range(len(self.scales[i_type])):
                func = self.func_list[i_type][i_level]
                if func is None:
                    continue
                _, _, mask = func(sample)
                self.precomputed_masks[(i_type, i_level)] = mask.to(torch.float32)

    def _apply_precomputed_mask(self, image_process, mask):
        """Apply a cached mask in k-space and return reconstructed magnitude.

        Args:
            image_process: Tensor shaped ``[1, H, W]``.
            mask: Cached mask tensor shaped ``[1, H, W]``.

        Returns:
            Tuple of ``(image_abs, mask)`` where ``image_abs`` is the masked
            reconstruction magnitude with shape ``[1, H, W]``.
        """
        kspace = torch.view_as_real(
            torch.fft.fftshift(torch.fft.fft2(image_process[0]))
        )[None, ...]
        masked_kspace = kspace * mask.unsqueeze(-1)
        image_masked = torch.fft.ifft2(torch.view_as_complex(masked_kspace))
        image_abs = abs(image_masked.squeeze(0)).unsqueeze(0)
        return image_abs, mask

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data
        if self.is_train:
            filename = self.trainPaths[idx]
        else:
            filename = self.valPaths[idx]

        with Image.open(filename) as image:
            image = self.image_transform(image)[0].unsqueeze(0)

        image = image / 255.0  # assume input image scale 0-255

        # use the same normalize as fastmri
        if torch.max(image) > 1e-12:
            image = image / torch.max(image)
        else:
            image = torch.zeros(image.shape)

        # random pick a type of downsample
        if self.fix_scale_idx:
            idx_sample_type = self.fix_scale_idx[0]
        else:
            idx_sample_type = torch.randint(0, len(self.scales), (1,)).squeeze(-1)
        idx_sample_type_i = (
            int(idx_sample_type)
            if not isinstance(idx_sample_type, torch.Tensor)
            else int(idx_sample_type.item())
        )

        # random pick a level of a downsample type
        if self.fix_scale_idx:
            idx_scale = self.fix_scale_idx[1]
        else:
            idx_scale = torch.randint(
                0, len(self.scales[idx_sample_type_i]), (1,)
            ).squeeze(-1)
        idx_scale_i = (
            int(idx_scale)
            if not isinstance(idx_scale, torch.Tensor)
            else int(idx_scale.item())
        )
        scale = self.scales[idx_sample_type_i][idx_scale_i]

        # scale images
        if scale != 1:
            image_process = torch.nn.functional.interpolate(
                image.unsqueeze(0), scale_factor=1 / scale, mode="bicubic"
            )
            image_process = image_process.squeeze(0)
        else:
            image_process = image

        mask = torch.zeros_like(image_process)
        # preprocess images
        cache_key = (idx_sample_type_i, idx_scale_i)
        if self.use_precomputed_mask and cache_key in self.precomputed_masks:
            image_process, mask = self._apply_precomputed_mask(
                image_process, self.precomputed_masks[cache_key]
            )
        elif self.func_list[idx_sample_type_i][idx_scale_i]:
            func = self.func_list[idx_sample_type_i][idx_scale_i]
            image_process, _, mask = func(image_process)  # [1, H, W], complex value

        image_process = abs(image_process.squeeze(0)).unsqueeze(0)  # [1, H, W]

        image_process = image_process.to(torch.float32)
        image = image.to(torch.float32)
        pair = (image_process, image)

        return pair, idx_scale_i, filename, idx, mask, idx_sample_type_i
