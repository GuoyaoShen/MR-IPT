# MR-IPT

import math

import torch
import torch.nn as nn
from .image_encoder import ImageEncoderViT
from .prompt_encoder import PromptEncoderMulti
from .image_decoder import ImageDecoderMulti
from help_func import print_var_detail


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class BasicBlock(nn.Sequential):
    def __init__(
            self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
            bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # check if scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class MRIPT(nn.Module):
    def __init__(self, n_feats, n_colors, scale, conv_kernel_size, res_kernel_size,
                 image_encoder: ImageEncoderViT,
                 prompt_encoder: PromptEncoderMulti,
                 image_decoder: ImageDecoderMulti,
                 conv=default_conv,
                 mode='combine'):
        '''
        MR-IPT model. It supports three modes:
        (1) type mode: heads and tails are grouped based on acceleration ratios, with each head-tail pair specializing in different sampling masks;
        (2) level mode: heads and tails are grouped based on sampling masks, allowing each pair to generalize across different acceleration ratios;
        (3) combine (split) mode: each unique combination of sampling mask and acceleration ratio is assigned a dedicated head-tail pair.
        Args:
            mode: 'type', 'level', or 'combine'
        '''
        super(MRIPT, self).__init__()

        self.scale_idx = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_feats = n_feats
        self.conv_kernel_size = conv_kernel_size
        self.res_kernel_size = res_kernel_size
        self.n_colors = n_colors
        self.scale = scale
        self.mode = mode
        act = nn.ReLU(True)

        if self.mode == 'combine':
            head_list = []
            tail_list = []
            for i in range(len(self.scale)):
                head_i = nn.ModuleList([nn.Sequential(
                    conv(self.n_colors, self.n_feats, self.conv_kernel_size),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act)
                ) for _ in self.scale[i]])
                tail_i = nn.ModuleList([
                    nn.Sequential(
                        Upsampler(conv, 1, self.n_feats, act=False),
                        conv(self.n_feats, self.n_colors, self.conv_kernel_size)
                    ) for _ in self.scale[i]
                ])
                head_list.append(head_i)
                tail_list.append(tail_i)
            self.head = nn.ModuleList(head_list)
            self.tail = nn.ModuleList(tail_list)
        elif self.mode == 'level':
            self.head = nn.ModuleList([
                nn.Sequential(
                    conv(self.n_colors, self.n_feats, self.conv_kernel_size),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act)
                ) for _ in self.scale[0]
            ])
            self.tail = nn.ModuleList([
                nn.Sequential(
                    Upsampler(conv, 1, self.n_feats, act=False),
                    conv(self.n_feats, self.n_colors, self.conv_kernel_size)
                ) for _ in self.scale[0]
            ])
        elif self.mode == 'type':
            self.head = nn.ModuleList([
                nn.Sequential(
                    conv(self.n_colors, self.n_feats, self.conv_kernel_size),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act),
                    ResBlock(conv, self.n_feats, self.res_kernel_size, act=act)
                ) for _ in self.scale
            ])
            self.tail = nn.ModuleList([
                nn.Sequential(
                    Upsampler(conv, 1, self.n_feats, act=False),
                    conv(self.n_feats, self.n_colors, self.conv_kernel_size)
                ) for _ in self.scale
            ])

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.image_decoder = image_decoder

    def forward(self, x, levels, types):
        x = x.to(self.device)
        levels = levels.to(self.device)
        types = types.to(self.device)

        head_out = torch.zeros((x.shape[0], self.n_feats, x.shape[-2], x.shape[-1]),
                               device=self.device)
        for i in range(x.shape[0]):
            if self.mode == 'combine':
                level_i = levels[i]
                type_i = types[i]
                head_out[i] = self.head[type_i][level_i](x[i].unsqueeze(0).to(self.device)).squeeze(0)
            elif self.mode == 'level':
                level_i = levels[i]
                head_out[i] = self.head[level_i](x[i].unsqueeze(0).to(self.device)).squeeze(0)
            elif self.mode == 'type':
                type_i = types[i]
                head_out[i] = self.head[type_i](x[i].unsqueeze(0).to(self.device)).squeeze(0)

        image_embeddings = self.image_encoder(head_out)
        sparse_embeddings = self.prompt_encoder(levels=levels, types=types)
        image_decodings = self.image_decoder(image_embeddings=image_embeddings,
                                                        image_pe=self.prompt_encoder.get_dense_pe(),
                                                        sparse_prompt_embeddings=sparse_embeddings
                                                        )
        res = image_decodings + head_out

        tail_out = torch.zeros((x.shape[0], self.n_colors, x.shape[-2], x.shape[-1]),
                               device=self.device)
        for i in range(res.shape[0]):
            if self.mode == 'combine':
                level_i = levels[i]
                type_i = types[i]
                tail_out[i] = self.tail[type_i][level_i](res[i].unsqueeze(0).to(self.device)).squeeze(0)
            elif self.mode == 'level':
                level_i = levels[i]
                tail_out[i] = self.tail[level_i](res[i].unsqueeze(0).to(self.device)).squeeze(0)
            elif self.mode == 'type':
                type_i = types[i]
                tail_out[i] = self.tail[type_i](res[i].unsqueeze(0).to(self.device)).squeeze(0)

        return tail_out