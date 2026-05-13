import contextlib

import numpy as np
import torch
import operator
from scipy.stats import norm


@contextlib.contextmanager
def temp_seed(rng, seed):
    """Temporarily set RNG seed within a context.

    Args:
        rng: NumPy random generator module/object.
        seed: Seed value to apply during the context.
    """
    state = rng.get_state()
    rng.seed(seed)
    try:
        yield
    finally:
        rng.set_state(state)


def center_crop_np(img, bounding):
    """Center-crop a NumPy array to the requested bounding shape.

    Args:
        img: Input array with trailing spatial dimensions.
        bounding: Target crop shape tuple.

    Returns:
        Center-cropped array.
    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]


class RandomMask:
    """Random column-wise strip mask."""

    def __init__(self, center_fraction, acceleration, size, seed=None):
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.size = size
        self.seed = seed
        self.rng = np.random

    def __call__(self):
        """Create a random strip mask.

        Returns:
            Tuple ``(mask, mask_fold)`` where:
            - ``mask`` has shape ``(C, H, W)``
            - ``mask_fold`` has shape ``(C, 1, W)``
        """
        with temp_seed(self.rng, self.seed):
            num_cols = self.size[-1]

            # create the mask
            num_low_freqs = int(round(num_cols * self.center_fraction))
            prob = (num_cols / self.acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            pad = (num_cols - num_low_freqs + 1) // 2
            mask = np.ones(self.size, dtype=np.float32)
            mask_fold = np.ones((self.size[0], 1, self.size[2]), dtype=np.float32)

            sequence = np.arange(num_cols)
            mask_sequence = np.concatenate(
                (sequence[0:pad], sequence[pad + num_low_freqs : num_cols]), axis=0
            )
            r = np.random.permutation(mask_sequence.shape[0])
            mask_sequence = np.squeeze(mask_sequence[r[:, None]], axis=-1)
            num_mask_col = int(round((1 - prob) * len(mask_sequence)))
            mask_sequence = mask_sequence[0:num_mask_col]
            mask[..., mask_sequence[:, None]] = 0.0
            mask_fold[..., mask_sequence[:, None]] = 0.0
        return mask, mask_fold


class RandomMask2D:
    """Random pixel-wise 2D mask."""

    def __init__(self, center_fraction, acceleration, size, seed=None):
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.size = size
        self.seed = seed
        self.rng = np.random

    def __call__(self):
        """Create a random pixel-wise mask.

        Returns:
            Tuple ``(mask, mask_fold)`` where both arrays have shape
            ``(C, H, W)``.
        """
        with temp_seed(self.rng, self.seed):
            D, H, W = self.size

            # create the mask
            num_low_freqs = int(round(W * self.center_fraction))
            prob = (W / self.acceleration - num_low_freqs) / (W - num_low_freqs)
            pad = (W - num_low_freqs + 1) // 2
            idx = np.zeros([H * W, 2])
            idx_1d = np.arange(H * W)
            idx[:, 1] = idx_1d % W
            idx[:, 0] = idx_1d // W
            idx = idx.reshape(H, W, 2)
            idx = np.concatenate(
                (idx[:, 0:pad, :], idx[:, pad + num_low_freqs : W, :]), axis=1
            )
            row = np.random.permutation(idx.shape[1])
            num_mask_col = int(round((1 - prob) * len(row)))
            row = row[0:num_mask_col]
            idx = np.squeeze(idx[:, row[:, None], :], axis=-2)
            idx = idx.reshape(H * num_mask_col, 2)
            r = np.random.permutation(idx.shape[0])
            mask_sequence = np.squeeze(idx[r[:, None], :], axis=-2).astype(int)
            mask = np.ones([D, H, W], dtype=np.float32)
            mask[..., mask_sequence[:, 0], mask_sequence[:, 1]] = 0.0
            mask_fold = mask
        return mask, mask_fold


class EquiSpaceMask:
    """Equi-spaced column-wise strip mask."""

    def __init__(self, center_fraction, acceleration, size, seed=None):
        self.center_fraction = center_fraction
        self.acceleration = acceleration
        self.size = size
        self.seed = seed
        self.rng = np.random

    def __call__(self):
        """Create an equi-spaced strip mask.

        Returns:
            Tuple ``(mask, mask_fold)`` where:
            - ``mask`` has shape ``(C, H, W)``
            - ``mask_fold`` has shape ``(C, 1, W)``
        """
        with temp_seed(self.rng, self.seed):
            num_cols = self.size[-1]

            # create the mask
            num_low_freqs = int(round(num_cols * self.center_fraction))
            prob = (num_cols / self.acceleration - num_low_freqs) / (
                num_cols - num_low_freqs
            )
            pad = (num_cols - num_low_freqs + 1) // 2
            mask = np.ones(self.size, dtype=np.float32)
            mask_fold = np.ones((self.size[0], 1, self.size[2]), dtype=np.float32)

            sequence = np.arange(num_cols)
            mask_sequence = np.concatenate(
                (sequence[0:pad], sequence[pad + num_low_freqs : num_cols]), axis=0
            )
            # r = np.random.permutation(mask_sequence.shape[0])
            # mask_sequence = np.squeeze(mask_sequence[r[:, None]], axis=-1)
            num_mask_col = int(round((1 - prob) * len(mask_sequence)))
            idx_remove = np.round(
                np.linspace(0, len(mask_sequence) - 1, num_mask_col)
            ).astype(int)
            # mask_sequence_keep = mask_sequence[0::self.acceleration]
            # mask_sequence_remove = np.setdiff1d(mask_sequence, mask_sequence_keep)
            mask_sequence_remove = mask_sequence[idx_remove]
            mask[..., mask_sequence_remove[:, None]] = 0.0
            mask_fold[..., mask_sequence_remove[:, None]] = 0.0
        return mask, mask_fold


class RandomMaskGaussian1D:
    """1D Gaussian mask sampler wrapper."""

    def __init__(
        self,
        acceleration=4,
        center_fraction=0.08,
        size=(1, 256, 256),
        seed=None,
        mean=[0],
        cov=[[1]],
        concentration=3,
        patch_size=2,
    ):
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.size = size
        self.seed = seed
        self.mean = mean
        self.cov = cov
        self.concentration = concentration
        self.patch_size = patch_size

    def __call__(self):
        return random_mask_gaussian_1D(
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            size=self.size,
            seed=self.seed,
            mean=self.mean,
            cov=self.cov,
            concentration=self.concentration,
            patch_size=self.patch_size,
        )


def random_mask_gaussian_1D(
    acceleration=4,
    center_fraction=0.08,
    size=(16, 320, 320),
    seed=None,
    mean=[0],
    cov=[[1]],
    concentration=3,
    patch_size=4,
):
    """Create a 1D Gaussian subsampling mask.

    Args:
        acceleration: Undersampling acceleration factor.
        center_fraction: Fraction of low-frequency center retained.
        size: Output shape ``(B, H, W)``.
        seed: ``None``, int, or list of ints for deterministic sampling.
        mean: Gaussian mean over width axis.
        cov: Gaussian covariance over width axis.
        concentration: Sampling concentration range in Gaussian std-space.
        patch_size: Upsampling factor from folded mask to image grid.

    Returns:
        Tuple ``(masks, masks_fold)`` where:
        - ``masks`` has shape ``(B, H, W)``
        - ``masks_fold`` has shape ``(B, H // patch_size, W // patch_size)``
    """
    B, H, W = size
    if H != W:
        raise Exception("different height and width of the mask setting")
    if H % patch_size != 0:
        raise Exception("image dimension cannot be fully divided by patch size")

    if isinstance(seed, int):
        seed_list = seed * (np.arange(B) + 1)
    elif isinstance(seed, list):
        if len(seed) != B:
            raise Exception("different seed list length and batch size")
        else:
            seed_list = np.array(seed)
    else:
        seed_list = np.array([None] * B)

    rng = np.random
    crop_size = int(H / patch_size)
    margin = patch_size * 2
    half_size = crop_size / 2

    cdf_lower_limit = norm(mean[0], cov[0][0]).cdf(-concentration)
    cdf_upper_limit = norm(mean[0], cov[0][0]).cdf(concentration)
    probability = cdf_upper_limit - cdf_lower_limit

    num_pts = int(crop_size / (acceleration * probability))
    num_low_freqs = int(round(crop_size * center_fraction))
    pad = (crop_size - num_low_freqs + 1) // 2
    masks = np.zeros((B, H, W))
    masks_fold = np.zeros((B, crop_size, crop_size))

    for i in range(B):
        with temp_seed(rng, seed_list[i]):
            # gaussian distribution index
            gauss_np = rng.multivariate_normal(mean, cov, num_pts)

            gauss_np = gauss_np * (half_size / concentration) + half_size + margin / 2
            gauss_np = np.round(gauss_np).astype(int)
            gauss_np = np.clip(gauss_np, 0, crop_size + int(margin / 2))

            # apply gaussian index on mask
            mask = np.zeros((crop_size + margin, crop_size + margin))
            mask[:, gauss_np.transpose()[0]] = 1.0
            mask = center_crop_np(mask, (crop_size, crop_size))

            # reset center square to unmasked
            mask[:, pad : pad + num_low_freqs] = 1.0
            masks_fold[i] = mask
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, scale_factor=patch_size, mode="nearest"
            )
            mask = mask.squeeze(0).squeeze(0)
            masks[i] = mask.numpy()
    return masks, masks_fold


class RandomMaskGaussian:
    """2D Gaussian mask sampler wrapper."""

    def __init__(
        self,
        acceleration=4,
        center_fraction=0.08,
        size=(1, 256, 256),
        seed=None,
        mean=(0, 0),
        cov=[[1, 0], [0, 1]],
        concentration=3,
        patch_size=4,
    ):
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.size = size
        self.seed = seed
        self.mean = mean
        self.cov = cov
        self.concentration = concentration
        self.patch_size = patch_size

    def __call__(self):
        return random_mask_gaussian(
            acceleration=self.acceleration,
            center_fraction=self.center_fraction,
            size=self.size,
            seed=self.seed,
            mean=self.mean,
            cov=self.cov,
            concentration=self.concentration,
            patch_size=self.patch_size,
        )


def random_mask_gaussian(
    acceleration=4,
    center_fraction=0.08,
    size=(16, 320, 320),
    seed=None,
    mean=(0, 0),
    cov=[[1, 0], [0, 1]],
    concentration=3,
    patch_size=4,
):
    """Create a 2D Gaussian subsampling mask.

    Args:
        acceleration: Undersampling acceleration factor.
        center_fraction: Fraction of low-frequency center retained.
        size: Output shape ``(B, H, W)``.
        seed: ``None``, int, or list of ints for deterministic sampling.
        mean: Gaussian mean over height/width axes.
        cov: Gaussian covariance over height/width axes.
        concentration: Sampling concentration range in Gaussian std-space.
        patch_size: Upsampling factor from folded mask to image grid.

    Returns:
        Tuple ``(masks, masks_fold)`` where:
        - ``masks`` has shape ``(B, H, W)``
        - ``masks_fold`` has shape ``(B, H // patch_size, W // patch_size)``
    """
    B, H, W = size
    if H != W:
        raise Exception("different height and width of the mask setting")
    if H % patch_size != 0:
        raise Exception("image dimension cannot be fully divided by patch size")

    if isinstance(seed, int):
        seed_list = seed * (np.arange(B) + 1)
    elif isinstance(seed, list):
        if len(seed) != B:
            raise Exception("different seed list length and batch size")
        else:
            seed_list = np.array(seed)
    else:
        seed_list = np.array([None] * B)

    rng = np.random
    crop_size = int(H / patch_size)
    margin = patch_size * 2
    half_size = crop_size / 2

    cdf_lower_limit = norm(mean[0], cov[0][0]).cdf(-concentration)
    cdf_upper_limit = norm(mean[1], cov[1][1]).cdf(concentration)
    probability = cdf_upper_limit - cdf_lower_limit

    num_pts = int(crop_size * crop_size / (acceleration * probability * probability))
    num_low_freqs = int(round(crop_size * center_fraction))
    pad = (crop_size - num_low_freqs + 1) // 2
    masks = np.zeros((B, H, W))
    masks_fold = np.zeros((B, crop_size, crop_size))

    for i in range(B):
        with temp_seed(rng, seed_list[i]):
            # gaussian distribution index

            gauss_np = rng.multivariate_normal(mean, cov, num_pts)
            gauss_np = gauss_np * (half_size / concentration) + half_size + margin / 2
            gauss_np = np.round(gauss_np).astype(int)
            gauss_np = np.clip(gauss_np, 0, crop_size + int(margin / 2))

            # apply gaussian index on mask
            mask = np.zeros((crop_size + margin, crop_size + margin))
            mask[gauss_np.transpose()[0], gauss_np.transpose()[1]] = 1.0
            mask = center_crop_np(mask, (crop_size, crop_size))

            # reset center square to unmasked
            mask[pad : pad + num_low_freqs, pad : pad + num_low_freqs] = 1.0
            masks_fold[i] = mask
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(0)
            mask = torch.nn.functional.interpolate(
                mask, scale_factor=patch_size, mode="nearest"
            )
            mask = mask.squeeze(0).squeeze(0)
            masks[i] = mask.numpy()
    return masks, masks_fold
