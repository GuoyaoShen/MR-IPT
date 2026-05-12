import torch
from fastmri.data import subsample, transforms


class DataTransform:
    """Apply k-space masking and return undersampled image data.

    The transform converts a real image to k-space, applies the configured
    mask function, and reconstructs the undersampled image via inverse FFT.
    """

    def __init__(
        self,
        mask_func,
    ):
        """Initialize the transform.

        Args:
            mask_func: Mask function instance used to subsample k-space.
        """
        self.mask_func = mask_func

    def __call__(self, image: torch.tensor):
        """Generate undersampled image and mask from an input image tensor.

        Args:
            image: Tensor with shape ``[1, H, W]``.

        Returns:
            Tuple ``(image_masked, image, mask)`` where:
            - ``image_masked`` is complex undersampled reconstruction,
            - ``image`` is original input tensor,
            - ``mask`` has shape ``[1, H, W]``.
        """
        kspace = torch.view_as_real(torch.fft.fftshift(torch.fft.fft2(image[0])))  # [H, W, 2]
        kspace = kspace[None, ...] # [H,W,2] to [1,H,W,2]

        # ====== Under-sampled ======
        # apply mask
        if isinstance(self.mask_func, subsample.MaskFunc):
            masked_kspace, mask, _ = transforms.apply_mask(kspace, self.mask_func)  # mask [1,1,W,1]
            mask = mask.squeeze(-1).squeeze(0).repeat(kspace.shape[1], 1)  # [H,W]
        else:
            masked_kspace, mask = apply_mask(kspace, self.mask_func)  # mask [1,H,W,1]
            mask = mask.squeeze(-1).squeeze(0)  # [H,W]

        image_masked = torch.fft.ifft2(torch.view_as_complex(masked_kspace))
        return image_masked, image, mask.unsqueeze(0)

def apply_mask(data, mask_func):
    """Apply a custom mask function to k-space data.

    Args:
        data: Tensor with shape ``[Nc, H, W, 2]``.
        mask_func: Callable returning ``(mask, mask_fold)`` with mask shaped
            ``[Nc, H, W]``.

    Returns:
        Tuple ``(masked_data, mask)`` where mask has shape ``[Nc, H, W, 1]``.
    """
    mask, _ = mask_func()
    mask = torch.from_numpy(mask)
    mask = mask[..., None]  # [Nc(1),H,W,1]
    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros
    return masked_data, mask
