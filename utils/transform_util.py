# Code borrowed from https://github.com/facebookresearch/fastMRI
# Code has been customized

import numpy as np
import torch as th
import polarTransform
import operator


# -------- FFT transform --------

def fftc_np(image):
    """
    Orthogonal FFT2 transform image to kspace data, numpy.array to numpy.array.

    :param image: numpy.array of complex with shape of (h, w), mri image.

    :return: numpy.array of complex with shape of (h, w), kspace data with center low-frequency, keep dtype.
    """
    kspace = np.fft.fftshift(np.fft.fft2(image, norm="ortho"))
    kspace = kspace.astype(image.dtype)
    return kspace


def ifftc_np(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, numpy.array to numpy.array.

    :param kspace: numpy.array of complex with shape of (h, w) and center low-frequency, kspace data.

    :return: numpy.array of complex with shape of (h, w), transformed image, keep dtype.
    """
    image = np.fft.ifft2(np.fft.ifftshift(kspace), norm="ortho")
    image = image.astype(kspace.dtype)
    return image


def fftc_th(image):
    """
    Orthogonal FFT2 transform image to kspace data, th.Tensor to th.Tensor.

    :param image: th.Tensor of real with shape of (..., 2, h, w), mri image.

    :return: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency, keep dtype.
    """
    image = image.permute(0, 2, 3, 1).contiguous()
    kspace = th.fft.fftshift(th.fft.fft2(th.view_as_complex(image), norm="ortho"), dim=(-1, -2))
    kspace = th.view_as_real(kspace).permute(0, 3, 1, 2).contiguous()
    return kspace


def ifftc_th(kspace):
    """
    Inverse orthogonal FFT2 transform kspace data to image, th.Tensor to th.Tensor.

    :param kspace: th.Tensor of real with shape of (..., 2, h, w), kspace data with center low-frequency.

    :return: th.Tensor of real with shape of (..., 2, h, w), mri image, keep dtype.
    """
    kspace = kspace.permute(0, 2, 3, 1).contiguous()
    image = th.fft.ifft2(th.fft.ifftshift(th.view_as_complex(kspace), dim=(-1, -2)), norm="ortho")
    image = th.view_as_real(image).permute(0, 3, 1, 2).contiguous()
    return image


# -------- dtype transform --------

def complex2real_np(x):
    """
    Change a complex numpy.array to a real array with two channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: numpy.array of real with shape of (2, h, w).
    """
    return np.stack([x.real, x.imag])


def real2complex_np(x):
    """
    Change a real numpy.array with two channels to a complex array.

    :param x: numpy.array of real with shape of (2, h, w).

    :return: numpy.array of complex64 with shape of (h, w).
    """
    complex_x = np.zeros_like(x[0, ...], dtype=np.complex64)
    complex_x.real, complex_x.imag = x[0], x[1]
    return complex_x


def np2th(x):
    return th.tensor(x)


def th2np(x):
    return x.detach().cpu().numpy()


def np_comlex_to_th_real2c(x):
    """
    Transform numpy.array of complex to th.Tensor of real with 2 channels.

    :param x: numpy.array of complex with shape of (h, w).

    :return: th.Tensor of real with 2 channels with shape of (h, w, 2).
    """
    return np2th(complex2real_np(x).transpose((1, 2, 0)))


def th_real2c_to_np_complex(x):
    """
    Transform th.Tensor of real with 2 channels to numpy.array of complex.

    :param x: th.Tensor of real with 2 channels with shape of (h, w, 2).

    :return: numpy.array of complex with shape of (h, w).
    """
    return real2complex_np(th2np(x.permute(2, 0, 1)))


def th2np_magnitude(x):
    """
    Compute the magnitude of torch.Tensor with shape of (b, 2, h, w).

    :param x: th.Tensor of real with 2 channels with shape of (b, 2, h, w).

    :return: numpy.array of real with shape of (b, h, w).
    """
    x = th2np(x)
    return np.sqrt(x[:, 0, ...] ** 2 + x[:, 1, ...] ** 2)


def pad_to_pool(x: th.tensor, num_layer: int, step_scale: int) -> th.tensor:
    """
    reshape a tensor so that it's dimension fits to up/down pool by given number of layers

    :param x: input tensor, expected shape [D, H, W, ...]
    :param num_layer: number of layers to up/down pool
    :param step_scale: scale coefficient for each up/down scale
    :return: x_rescale, new tensor with shape [D, H_new, W_new, ...], pad extra elements with zero
    """
    H = x.shape[1]
    W = x.shape[2]

    H_new = (step_scale ** num_layer) * round(H / (step_scale ** num_layer))
    W_new = (step_scale ** num_layer) * round(W / (step_scale ** num_layer))

    x_shape = list(x.shape)
    x_shape[1] = H_new
    x_shape[2] = W_new
    x_rescale = th.zeros(x_shape)

    # pad given tensor slightly to make it down-poolable
    x_rescale[:, H_new // 2 - min(H_new, H) // 2: H_new // 2 + min(H_new, H) // 2,
    W_new // 2 - min(W_new, W) // 2: W_new // 2 + min(W_new, W) // 2, :] \
        = x[:, H // 2 - min(H_new, H) // 2: H // 2 + min(H_new, H) // 2,
          W // 2 - min(W_new, W) // 2: W // 2 + min(W_new, W) // 2, :]

    return x_rescale

def center_crop_with_pad(x: th.tensor, center_crop_h, center_crop_w) -> th.tensor:
    """
    reshape a tensor so that it's dimension fits to up/down pool by given number of layers

    :param x: input tensor, expected shape [D, H, W, ...]
    :param num_layer: number of layers to up/down pool
    :param step_scale: scale coefficient for each up/down scale
    :return: x_rescale, new tensor with shape [D, H_new, W_new, ...], pad extra elements with zero
    """
    H = x.shape[1]
    W = x.shape[2]

    H_new = center_crop_h
    W_new = center_crop_w

    x_shape = list(x.shape)
    x_shape[1] = H_new
    x_shape[2] = W_new
    x_rescale = th.zeros(x_shape)

    # pad given tensor slightly to make it down-poolable
    x_rescale[:, H_new // 2 - min(H_new, H) // 2: H_new // 2 + min(H_new, H) // 2,
    W_new // 2 - min(W_new, W) // 2: W_new // 2 + min(W_new, W) // 2] \
        = x[:, H // 2 - min(H_new, H) // 2: H // 2 + min(H_new, H) // 2,
          W // 2 - min(W_new, W) // 2: W // 2 + min(W_new, W) // 2]

    return x_rescale

def normalize_one_to_one(x: th.tensor) -> th.tensor:
    """
    Args: normalize tensor value to range of [-1, 1]
    :param x: input tensor
    :return: x_norm: normalized input tensor
    """
    x_norm = 2 * (x - x.min()) / (x.max() - x.min()) - 1

    return x_norm, x.max(), x.min()


def normalize_zero_to_one(x: th.tensor) -> th.tensor:
    """
    Args: normalize tensor value to range of [0, 1]
    :param x: input tensor
    :return: x_norm: normalized input tensor
    """
    x_norm = (x - x.min()) / (x.max() - x.min()) + 0.0

    return x_norm, x.max(), x.min()


def denormalize_one_to_one(x_norm: th.tensor, x_max, x_min) -> th.tensor:
    """
    Args: denormalize tensor value from range of [-1, 1]

    :param x_norm: input normalized tensor
    :param x_max: upper bound of x
    :param x_min: lower bound of x
    :return: x: denormalized input tensor
    """
    x = (x_norm + 1) * (x_max - x_min) / 2 + x_min

    return x


def denormalize_zero_to_one(x_norm: th.tensor, x_max, x_min) -> th.tensor:
    """
    Args: denormalize tensor value from range of [0, 1]

    :param x_norm: input normalized tensor
    :param x_max: upper bound of x
    :param x_min: lower bound of x
    :return: x: denormalized input tensor
    """
    x = x_norm * (x_max - x_min) + x_min + 0.0

    return x


def add_gaussian_noise(x: th.tensor, coef) -> th.tensor:
    """
    Args: add gaussian noise with scale coef to img, img should have scale [-1, 1]

    :param x: input one_to_one normalized image tensor [C, H, W]
    :param coef: scale coefficient for gaussian noise
    :return: x_noisy: gaussian noise output tensor [C, H, W]
    """

    noise = th.randn_like(x)
    x_noisy = x + coef * noise

    return x_noisy


def cartersianToPolar(img, order=0):
    """
    Args:
        img: [H, W, d] input image tensor
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        polarImage: [H, W, d] polar image of input image
        ptSettings: recorded settings during transform

    """

    H, W, d = img.shape

    # initialize first channel, [H, W]
    polarImage, ptSettings = polarTransform.convertToPolarImage(img[:, :, 0], order=order)
    polarImage = th.tensor(polarImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            polarImageTemp, _ = polarTransform.convertToPolarImage(img[:, :, i], order=order)
            polarImage = th.cat((polarImage, th.tensor(polarImageTemp).unsqueeze(-1)), -1)  # [H, W, d]

    return polarImage, ptSettings


def polarToCartersian(polarImage, order=0):
    """
    Args:
        img: [H, W, d] input image tensor
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        cartesianImage: [H, W, d] polar image of input image
        ptSettings: recorded settings during transform

    """

    H, W, d = polarImage.shape

    # initialize first channel, [H, W]
    cartesianImage, ptSettings = polarTransform.convertToCartesianImage(polarImage[:, :, 0], order=order)
    cartesianImage = th.tensor(cartesianImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            cartesianImageTemp, _ = polarTransform.convertToCartesianImage(polarImage[:, :, i], order=order)
            cartesianImage = th.cat((cartesianImage, th.tensor(cartesianImageTemp).unsqueeze(-1)),
                                    -1)  # [H, W, d]

    return cartesianImage, ptSettings


def polarToCartersian_given_setting(polarImage, ptSettings, order=0):
    """
    Args:
        polarImage: [H, W, d] input image tensor
        ptSettings: setting recorded from cartersianToPolar
        order : :class:`int` (0-5), optional
        The order of the spline interpolation, default is 3. The order has to be in the range 0-5.

        The following orders have special names:

            * 0 - nearest neighbor
            * 1 - bilinear
            * 3 - bicubic

    returns:
        cartesianImage: [H, W, d] cartesian image of input image

    """

    H, W, d = polarImage.shape

    # initialize first channel, [H, W]
    cartesianImage = ptSettings.convertToCartesianImage(polarImage[:, :, 0], order=0)
    cartesianImage = th.tensor(cartesianImage).unsqueeze(-1)  # [H, W, 1]

    for i in range(d):
        if i > 0:
            # [H, W]
            cartesianImageTemp = ptSettings.convertToCartesianImage(polarImage[:, :, i], order=order)
            cartesianImage = th.cat((cartesianImage, th.tensor(cartesianImageTemp).unsqueeze(-1)),
                                    -1)  # [H, W, d]

    return cartesianImage


def center_crop_np(img, bounding):
    """
    Args:
        img: 2d or 3d numpy
        bounding: input tuple show center_crop size (H, W)

    Returns:
        img: 2d or 3d numpy after center crop with center bounding dim (H, W)

    """
    start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, bounding))
    end = tuple(map(operator.add, start, bounding))
    slices = tuple(map(slice, start, end))
    return img[slices]
