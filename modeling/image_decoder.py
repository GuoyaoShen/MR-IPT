# Code borrowed from: https://github.com/facebookresearch/segment-anything

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d
from help_func import print_var_detail


class ImageDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_params: int = 4,
            activation: Type[nn.Module] = nn.GELU,
            output_dim_factor: int = 8,
            SSIM_head_depth: int = 3,
            SSIM_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer.
          transformer (nn.Module): the transformer used to predict masks
          num_params (int): the number of param prompts included in prompt encoder
          activation (nn.Module): the type of activation to use when
            upscaling masks
          SSIM_head_depth (int): the depth of the MLP used to predict
            mask quality
          SSIM_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_param_tokens = num_params
        self.SSIM_token = nn.Embedding(1, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // (output_dim_factor // 2), kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // (output_dim_factor // 2)),
            activation(),
            nn.ConvTranspose2d(transformer_dim // (output_dim_factor // 2), transformer_dim // output_dim_factor,
                               kernel_size=2, stride=2),
            activation(),
        )

        # legacy mlp to invoke param tokens
        self.output_hypernetworks_mlp = MLP(transformer_dim * self.num_param_tokens,
                                            transformer_dim * self.num_param_tokens,
                                            transformer_dim // output_dim_factor, 3)
        self.SSIM_prediction_head = MLP(
            transformer_dim, SSIM_head_hidden_dim, 1, SSIM_head_depth)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        images = self.predict_images(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return images  # [B, 256/4, H, W]

    def predict_images(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = self.SSIM_token.weight
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)


        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        SSIM_token_out = hs[:, 0, :]

        # Upscale embeddings and predict images using the combined tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Generate mask quality predictions
        SSIM_pred = self.SSIM_prediction_head(SSIM_token_out)

        return upscaled_embedding, SSIM_pred

    def _get_transformer_dim(self):
        return self.transformer_dim


class ImageDecoderMulti(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_params: int = 4,
            activation: Type[nn.Module] = nn.GELU,
            output_dim_factor: int = 8,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer.
          transformer (nn.Module): the transformer used to predict masks
          num_params (int): the number of param prompts included in prompt encoder
          activation (nn.Module): the type of activation to use when
            upscaling masks
          SSIM_head_depth (int): the depth of the MLP used to predict
            mask quality
          SSIM_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_param_tokens = num_params

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // (output_dim_factor // 2), kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // (output_dim_factor // 2)),
            activation(),
            nn.ConvTranspose2d(transformer_dim // (output_dim_factor // 2), transformer_dim // output_dim_factor,
                               kernel_size=2, stride=2),
            activation(),
        )

        # legacy mlp to invoke param tokens
        self.output_hypernetworks_mlp = MLP(transformer_dim * self.num_param_tokens,
                                            transformer_dim * self.num_param_tokens,
                                            transformer_dim // output_dim_factor, 3)

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        images = self.predict_images(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
        )

        # Prepare output
        return images  # [B, 256/4, H, W]

    def predict_images(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens

        # tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        tokens = sparse_prompt_embeddings
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Upscale embeddings and predict images using the combined tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        return upscaled_embedding

    def _get_transformer_dim(self):
        return self.transformer_dim

# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
