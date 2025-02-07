import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from imutils import paths
from help_func import print_var_detail
from torch.utils.data import Dataset
from utils.transform_util import *
# from fastmri.data import transforms
from PIL import Image


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
):
    """
    Create a dataloader for MRIPT RadImageNet dataset.
    Supports multi-scale undersamplings.

    :param data_dir: str, the directory saving data.
    :param random_seed: int, random seed to control any randomization in the dataset
    :param val_split: float, validation size of the dataset in percent, 0.1 gives 10% of the data
    :param image_size: int, unify image size for all input images to (image_size, image_size)
    :param batch_size: int, batch size for dataloader
    :param is_distributed: bool, whether to use distributed sampler
    :param is_train: bool, whether used for training
    :param scales: list of int, given scale size during image processing
    :param func_list: list of func, store functions used to preprocess image for each head-tail
    :param num_workers: int, number of workers if using multiple GPU, default 0
    :param if_rand_param: bool, use random parameters such as acceleration or center ratio, etc.
    :param fix_scale_idx: int, if it is not None, fix the scaling of the image and preprocess function to
            that particular index, meaning dataset will only use one downsampling regardless of the rest of functions in
            func_list.

    :return: fastmri loader for ipt trainer, output format determined by fastmri dataset
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

    # return loader
    return loader


class RadimgnetDataSetMulti(Dataset):
    """
    A customized dataset class used to create and preprocess RadImageNet data for MRIPT training.
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
    func_list: list of function lists
        store functions used to preprocess image for each head-tail
    """

    def __init__(
            self,
            data_dir, random_seed, val_split, image_size, scales,
            func_list, fix_scale_idx, is_train
    ):

        self.data_dir = data_dir
        self.random_seed = random_seed
        self.val_split = val_split
        self.image_size = image_size
        self.scales = scales
        self.func_list = func_list
        self.fix_scale_idx = fix_scale_idx
        self.is_train = is_train

        if not len(self.func_list) == len(self.scales):
            raise ValueError("unequal number of scales and function list.")

        # construct training and validation dataset
        imagePaths = list(paths.list_images(self.data_dir))
        np.random.seed(self.random_seed)
        np.random.shuffle(imagePaths)

        valPathsLen = int(len(imagePaths) * val_split)
        trainPathsLen = len(imagePaths) - valPathsLen
        self.trainPaths = imagePaths[:trainPathsLen]
        self.valPaths = imagePaths[trainPathsLen:]
        print('train size: ' + str(len(self.trainPaths)))
        print('validation size: ' + str(len(self.valPaths)))

    def __len__(self):
        if self.is_train:
            return len(self.trainPaths)
        else:
            return len(self.valPaths)

    def __getitem__(self, idx):
        # load image data
        if self.is_train:
            image = Image.open(self.trainPaths[idx])
            filename = self.trainPaths[idx]
        else:
            image = Image.open(self.valPaths[idx])
            filename = self.valPaths[idx]

        # Define a transform to convert PIL
        # image to a Torch tensor
        transform_toTensor = transforms.Compose([
            transforms.PILToTensor()
        ])

        # transform = transforms.PILToTensor()
        # Convert the PIL image to Torch tensor
        transform_resize = transforms.Resize(size=self.image_size)
        image = transform_resize(image)
        image = transform_toTensor(image)[0].unsqueeze(0)
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
