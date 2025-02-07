import random
import torch
import pickle

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from utils.fastmri_loader import FastmriDataset, read_datafile
import fastmri
import torchvision.transforms as transforms
import fastmri.data
from utils.transform_util import center_crop_with_pad
from help_func import print_var_detail


def get_patch(*args, patch_size=96, scale=2, input_large=False):
    """
    randomly crop the given image given patch size and scale
    :param args: given image pair, processed image, target image.
    :param patch_size: desired patch size
    :param scale: given scale size during image processing
    :param input_large: if input scale is reversed, default false
    :return: a patched image pair with [processed image (C, patch_size//scale, patch_size//scale),
     target image (C, patch_size, patch_size)]
    """
    ih, iw = args[0].shape[1:3]

    tp = patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy
    patched_pair = [
        args[0][:, iy:iy + ip, ix:ix + ip],
        *[a[:, ty:ty + tp, tx:tx + tp] for a in args[1:]]
    ]

    return patched_pair


def create_fastmri_dataloader_multi(
        data_dir,
        data_info_list_path,
        batch_size,
        random_flip=False,
        is_distributed=False,
        is_train=False,
        mask_func=None,
        post_process=None,
        center_crop_size=320,
        image_size=(224, 224),
        patch_size=48,
        scales=[1],
        num_layer=4,
        step_scale=2,
        target_mode='single',
        output_mode='img',
        norm_mode='unit_norm',
        num_workers=0,
        func_list=[None],
        if_rand_param=True,
        fix_scale_idx=None,
):
    """
    Create a dataloader for MRIPT fastMRI dataset.
    Supports multi-scale undersamplings.

    :param data_dir: str, the directory saving data.
    :param data_info_list_path: str, the .pkl file containing data list info
    :param batch_size: int, batch size for dataloader
    :param random_flip: bool, whether to flip image, default False
    :param is_distributed: bool, whether to use distributed sampler
    :param is_train: bool, whether used for training
    :param mask_func: mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc
    :param post_process: function, used to post-process image, image_zf, kspace, kspace_zf and mask
    :param center_crop_size: int, determine the center crop size of image and reconvert to kspace if necessary, default 320
    :param patch_size: int, desired patch size
    :param scales: list of int, given scale size during image processing
    :param num_layer: int, number of layers for model to up/down sample, default 4
    :param step_scale: int, scale of each layer for model to up/down sample, default 2
    :param target_mode: str, target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil (reconstruction_rss). default 'single'
    :param output_mode: str, 'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data. default 'img'
    :param norm_mode: str, 'unit_norm' or None. Whether to normalize output. default 'unit_norm'
    :param num_workers: int, number of workers if using multiple GPU, default 0
    :param func_list: list of func, store functions used to preprocess image for each head-tail
    :return: fastmri loader for ipt trainer, output format determined by fastmri dataset
    """

    if not data_dir:
        raise ValueError("unspecified dta directory.")

    if patch_size is not None:
        if patch_size * max(scales) > center_crop_size:
            raise ValueError("scaled image size exceeds target images size.")

    # read data information which is saved in a list.
    with open(data_info_list_path, "rb") as f:
        data_info_list = pickle.load(f)

    # create dataset
    dataset = FastmriDataSetMulti(
        data_dir=data_dir,
        data_info_list=data_info_list,
        random_flip=random_flip,
        mask_func=mask_func,
        post_process=post_process,
        scales=scales,
        patch_size=patch_size,
        center_crop_size=center_crop_size,
        image_size=image_size,
        num_layer=num_layer,
        step_scale=step_scale,
        target_mode=target_mode,
        output_mode=output_mode,
        norm_mode=norm_mode,
        func_list=func_list,
        if_rand_param=if_rand_param,
        fix_scale_idx=fix_scale_idx,
    )

    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(dataset)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )

    # return loader, note all np in FastmriDataset will be converted to tensor via dataloader
    return loader


class FastmriDataSetMulti(Dataset):
    """
    A customized dataset class used to create and preprocess fastMRI data for MRIPT training.
    Supports multi-scale undersamplings.
    get_item outputs list of input and target image pairs, list of filename and list of slice index

    Args:
    ----------
    data_dir : str
        the directory saving data.
    data_info_list : list
        a list containing elements like (file_name, index). A .h5 file contains multi slices.
    random_flip : bool
        whether to random flip image.
    mask_func : function
        mask function applied on kspace, normally use fastmri.subsample.RandomMaskFunc.
    post_process : function
        used to post-process image, image_zf, kspace, kspace_zf and mask.
    patch_size: int
        desired patch size
    scales: list of int
        given scale size during image processing
    center_crop_size : int
        determine the center crop size of image and reconvert to kspace if necessary
    num_layer: int,
        number of layers for model to up/down sample
    step_scale: int,
        scale of each layer for model to up/down sample
    target_mode: str,
        target mri, 'single' for single-coil (reconstruction_esc), or 'multi' for multi-coil
        (reconstruction_rss).
    output_mode: str,
        'img' or 'kspace'. Whether input and target are recovered mri img or raw k-space data.
    norm_mode: str,
        'unit_norm' or None. Whether to normalize output.
    func_list: list of functions
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, data_info_list, random_flip, mask_func,
            post_process, scales, patch_size, center_crop_size, image_size,
            num_layer, step_scale, target_mode, output_mode,
            norm_mode, func_list, if_rand_param, fix_scale_idx
    ):
        super().__init__()
        self.data_dir = data_dir
        self.random_flip = random_flip
        self.mask_func = mask_func
        self.post_process = post_process
        self.data_info_list = data_info_list
        self.center_crop_size = center_crop_size
        self.target_mode = target_mode
        self.output_mode = output_mode
        self.num_layer = num_layer
        self.step_scale = step_scale
        self.norm_mode = norm_mode
        self.patch_size = patch_size
        self.scales = scales
        self.func_list = func_list
        self.if_rand_param = if_rand_param
        self.fix_scale_idx = fix_scale_idx
        self.image_size = image_size

        if not len(self.func_list) == len(self.scales):
            raise ValueError("unequal number of scales and function list.")
    def __len__(self):
        return len(self.data_info_list)

    def __getitem__(self, idx):
        # load image data
        file_name, index = self.data_info_list[idx]
        acquisition, kspace_raw, image_rss, image_esc \
            = read_datafile(self.data_dir, file_name, index, self.target_mode)

        if self.output_mode == 'img':
            # set target mri image
            if self.target_mode == 'multi':
                target = fastmri.data.transforms.to_tensor(image_rss)  # [H, W]

            else:
                target = fastmri.data.transforms.to_tensor(image_esc)  # [H, W]

            # re-crop target and mask
            target = target.unsqueeze(0)# [1,H,W]
            target = center_crop_with_pad(target, self.center_crop_size, self.center_crop_size)

            # normalization
            # max = torch.max(image_masked_abs)
            # use target max
            max = torch.max(target)

            if max > 1e-12:
                scale_coeff = 1. / max
            else:
                scale_coeff = 0.0

            target = target * scale_coeff
        image = target # [1,H,W]

        filename = file_name

        # resize image after center crop
        transform_resize = transforms.Resize(size=self.image_size)
        image = transform_resize(image)

        # random pick a type of downsample
        if self.fix_scale_idx:
            idx_sample_type = self.fix_scale_idx[0]
        else:
            idx_sample_type = torch.randint(0, len(self.scales), (1,)).squeeze(-1)

        # random pick a level of a downsample type
        if self.fix_scale_idx:
            idx_scale = self.fix_scale_idx[1]
        else:
            idx_scale = torch.randint(0, len(self.scales[idx_sample_type]), (1,)).squeeze(-1)
        scale = self.scales[idx_sample_type][idx_scale]

        # scale images
        if scale != 1:
            image_process = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=1 / scale,
                                                            mode='bicubic')
            image_process = image_process.squeeze(0)
        else:
            image_process = image

        mask = torch.zeros_like(image_process)
        # preprocess images
        if self.func_list[idx_sample_type][idx_scale]:
            func = self.func_list[idx_sample_type][idx_scale]
            image_process, _, mask = func(image_process)  # [1, H, W], complex value

        image_process = abs(image_process.squeeze(0)).unsqueeze(0)  # [1, H, W]

        image_process = image_process.to(torch.float32)
        image = image.to(torch.float32)
        pair = (image_process, image)

        return pair, idx_scale, filename, idx, mask, idx_sample_type

def create_combine_dataloader(
        datasets,
        batch_size,
        is_distributed=False,
        is_train=False,
        num_workers=0,
):
    """
    Create a combined dataloader for all patient datasets
    """
    combined_dataset = datasets[0]
    for i in range(len(datasets)):
        if i > 0:
            combined_dataset = torch.utils.data.ConcatDataset([combined_dataset, datasets[i]])
    data_sampler = None
    if is_distributed:
        data_sampler = DistributedSampler(combined_dataset)

    # create dataloader
    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(data_sampler is None) and is_train,
        sampler=data_sampler,
        num_workers=num_workers,
        drop_last=is_train,
        pin_memory=True,
    )

    # return loader, note all np in FastmriDataset will be converted to tensor via dataloader
    return loader