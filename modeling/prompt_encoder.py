# Code borrowed from: https://github.com/facebookresearch/segment-anything
# Code has been customized

import numpy as np
import torch
from torch import nn
from help_func import print_var_detail

from typing import Any, Optional, Tuple


class PromptEncoder(nn.Module):
    """
    Encodes prompts for each head

    Args:
    ----------
    num_head : int
        Number of downsample heads
    embed_dim : int
        The prompts' embedding dimension
    image_embedding_size : tuple(int, int)
        The spatial size of the image embedding, as (H, W).
    input_image_size : int
        The padded size of the image as input to the image encoder, as (H, W).
    """
    def __init__(
        self,
        num_head: int,
        num_param_per_head: int,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_param_per_head = num_param_per_head
        self.num_head_embeddings = num_head
        head_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_head_embeddings)]
        self.head_embeddings = nn.ModuleList(head_embeddings)
    def _embed_heads(
        self,
        labels: torch.Tensor, # [B, ]
    ) -> torch.Tensor:
        """Embeds head prompts."""
        head_embedding = torch.zeros((labels.shape[0], 1, self.embed_dim), device=labels.device) # [B, num_param_per_head,embed_dim]
        for i in range(labels.shape[0]):
            label = labels[i]
            head_embedding[i] = self.head_embeddings[label].weight[0]

        return head_embedding # [B, 1, embed_dim]

    def forward(
        self,
        params: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.
        """

        bs = self._get_batch_size(params, labels)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        head_embeddings = self._embed_heads(labels)
        sparse_embeddings = torch.cat([sparse_embeddings, head_embeddings], dim=1)
        return sparse_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _get_batch_size(
        self,
        params: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if params is not None:
            return params.shape[0]
        elif labels is not None:
            return labels.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.head_embeddings[0].weight.device

class PromptEncoderMulti(nn.Module):
    """
    Encodes prompts for each head

    Args:
    ----------
    num_head : int
        Number of downsample heads
    embed_dim : int
        The prompts' embedding dimension
    image_embedding_size : tuple(int, int)
        The spatial size of the image embedding, as (H, W).
    input_image_size : int
        The padded size of the image as input to the image encoder, as (H, W).
    """
    def __init__(
        self,
        num_level: int,
        num_type: int,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int]
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.num_level_embeddings = num_level
        level_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_level_embeddings)]
        self.level_embeddings = nn.ModuleList(level_embeddings)

        self.num_type_embeddings = num_type
        type_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_type_embeddings)]
        self.type_embeddings = nn.ModuleList(type_embeddings)

    def _embed_levels(
        self,
        levels: torch.Tensor, # [B, ]
    ) -> torch.Tensor:
        """Embeds head prompts."""
        level_embedding = torch.zeros((levels.shape[0], 1, self.embed_dim), device=levels.device) # [B, num_param_per_head,embed_dim]
        for i in range(levels.shape[0]):
            level = levels[i]
            level_embedding[i] = self.level_embeddings[level].weight[0]

        return level_embedding # [B, 1, embed_dim]

    def _embed_types(
        self,
        types: torch.Tensor, # [B, ]
    ) -> torch.Tensor:
        """Embeds head prompts."""
        type_embedding = torch.zeros((types.shape[0], 1, self.embed_dim), device=types.device) # [B, num_param_per_head,embed_dim]
        for i in range(types.shape[0]):
            type = types[i]
            type_embedding[i] = self.type_embeddings[type].weight[0]

        return type_embedding # [B, 1, embed_dim]

    def forward(
        self,
        levels: Optional[torch.Tensor],
        types: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.
        """

        bs = self._get_batch_size(levels, types)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        head_embeddings = self._embed_levels(levels)
        type_embeddings = self._embed_types(types)
        sparse_embeddings = torch.cat([sparse_embeddings, head_embeddings], dim=1)
        sparse_embeddings = torch.cat([sparse_embeddings, type_embeddings], dim=1)
        return sparse_embeddings

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _get_batch_size(
        self,
        params: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if params is not None:
            return params.shape[0]
        elif labels is not None:
            return labels.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.level_embeddings[0].weight.device
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def get_position_encoding_given_key(d, k, n=torch.tensor([10000.0])):
    P = torch.zeros((d))
    for i in range(int(d/2)):
        denominator = torch.pow(n, 2*i/d)
        P[2*i] = torch.sin((k)/denominator)
        P[2*i+1] = torch.cos((k)/denominator)
    return P
