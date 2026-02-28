import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.embeddings import get_2d_sincos_pos_embed

from torch import Tensor
from torchmetrics import Metric
from torchmetrics.image.fid import _compute_fid
import os


def get_2d_circular_positional_embedding(embedding_dim: int, height: int, width: int) -> torch.Tensor:
    """
    Generate a 2D positional embedding for equirectangular (360-degree panoramic) images,
    where the horizontal axis is treated circularly to enforce wrap-around consistency.

    Args:
        embedding_dim (int): Dimensionality of the positional embedding.
                             Must be divisible by 4.
        height (int): Spatial height (vertical size) of the feature map.
        width (int): Spatial width (horizontal size) of the feature map (360° panorama).

    Returns:
        pos_embed (torch.Tensor): Positional embeddings with shape (height * width, embedding_dim)
                                  where each row corresponds to a position.
    """
    if embedding_dim % 4 != 0:
        raise ValueError("embedding_dim must be divisible by 4 for balanced sin/cos encoding.")

    # Number of features for each sin/cos pair per axis.
    num_pos_feats = embedding_dim // 4

    # Create coordinate grids.
    # For vertical axis: Normalize to [0, 1] using (height - 1)
    y = torch.arange(height, dtype=torch.float32)  # shape (height,)
    y = y / (height - 1)

    # For horizontal axis: Normalize to [0, 1] using width, because it is circular.
    # Notice: using width instead of (width - 1) so that 0 and 1 map to the same angle.
    x = torch.arange(width, dtype=torch.float32)  # shape (width,)
    x = x / width

    # Create meshgrid for positions.
    # grid_y and grid_x will each be of shape (height, width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    # Compute scaling factors using a temperature (e.g., 10000) like in standard sin/cos positional encodings.
    temperature = 10000
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    ## Vertical positional encoding (non-circular)
    # Scale the normalized y coordinate accordingly.
    pos_y = grid_y.unsqueeze(-1) / dim_t  # shape (height, width, num_pos_feats)

    # Compute sine and cosine components and flatten the last two dimensions.
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(2)  # shape (height, width, 2*num_pos_feats)

    ## Horizontal (circular) positional encoding
    # Convert normalized x coordinate to an angle in radians over [0, 2π)
    grid_x_angle = grid_x * 2 * math.pi
    pos_x = grid_x_angle.unsqueeze(-1) / dim_t  # shape (height, width, num_pos_feats)
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(2)  # shape (height, width, 2*num_pos_feats)

    # Concatenate vertical and horizontal encodings.
    pos_embed = torch.cat((pos_y, pos_x), dim=-1)  # shape (height, width, embedding_dim)

    # Flatten spatial dimensions if needed (e.g., for a transformer which expects sequence input).
    pos_embed = pos_embed.view(-1, embedding_dim)

    return pos_embed


class CircularPatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for SD3 cropping and panoramic circular consistency.

    Args:
        height (`int`, defaults to `224`): The height of the image.
        width (`int`, defaults to `224`): The width of the image.
        patch_size (`int`, defaults to `16`): The size of the patches.
        in_channels (`int`, defaults to `3`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        layer_norm (`bool`, defaults to `False`): Whether or not to use layer normalization.
        flatten (`bool`, defaults to `True`): Whether or not to flatten the output.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
        interpolation_scale (`float`, defaults to `1`): The scale of the interpolation.
        pos_embed_type (`str`, defaults to `"sincos"`): The type of positional embedding.
        pos_embed_max_size (`int`, defaults to `None`): The maximum size of the positional embedding.
    """

    def __init__(
        self,
        height=224,
        width=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size) * (width // patch_size)
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_circular_positional_embedding(
                embed_dim, self.height, self.width,
            ).float().unsqueeze(0)
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", pos_embed, persistent=persistent)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size, latent.shape[-1] // self.patch_size

        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_circular_positional_embedding(
                    self.pos_embed.shape[-1], height, width,
                ).float().unsqueeze(0).to(latent.device)
                # pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(latent.device)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class CircularPadding(nn.Module):

    def __init__(self, pad):
        super(CircularPadding, self).__init__()
        self.pad = pad

    def forward(self, x):
        if self.pad == 0:
            return x
        x = torch.nn.functional.pad(x,
                                    (self.pad, self.pad, self.pad, self.pad),
                                    'constant', 0)
        x[:, :, :, 0:self.pad] = x[:, :, :, -2 * self.pad:-self.pad]
        x[:, :, :, -self.pad:] = x[:, :, :, self.pad:2 * self.pad]
        return x


class Conv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super(Conv2d, self).__init__()
        self.pad = CircularPadding(padding)
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding=0)

    def forward(self, x):
        x = self.conv2d(self.pad(x))
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ResBlock, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels,
                            out_channels,
                            kernel_size,
                            stride=1,
                            padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        out = self.relu(self.batchnorm1(self.conv1(x)))
        out = self.batchnorm2(self.conv2(out))
        out += x

        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding):
        super(ConvBlock, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size, stride,
                            padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.relu(self.batchnorm1(self.conv1(x)))

        return x


class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.upsampling = nn.functional.interpolate

        self.upconv2_rgb = ConvBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv3_rgb = ConvBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv4_rgb = ConvBlock(in_channels=128,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.upconv5_rgb = ConvBlock(in_channels=64,
                                     out_channels=32,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.outconv_rgb = Conv2d(in_channels=32,
                                  out_channels=3,
                                  kernel_size=9,
                                  stride=1,
                                  padding=4)

        self.upres2_rgb = ResBlock(in_channels=128,
                                   out_channels=128,
                                   kernel_size=3,
                                   padding=1)
        self.upres3_rgb = ResBlock(in_channels=128,
                                   out_channels=128,
                                   kernel_size=5,
                                   padding=2)
        self.upres4_rgb = ResBlock(in_channels=64,
                                   out_channels=64,
                                   kernel_size=7,
                                   padding=3)
        self.upres5_rgb = ResBlock(in_channels=32,
                                   out_channels=32,
                                   kernel_size=9,
                                   padding=4)

    def forward(self, x):

        x = self.upsampling(x,
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
        rgb = x[:, :128]

        rgb = self.upconv2_rgb(rgb)
        rgb = self.upres2_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv3_rgb(rgb)
        rgb = self.upres3_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv4_rgb(rgb)
        rgb = self.upres4_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.upconv5_rgb(rgb)
        rgb = self.upres5_rgb(rgb)
        rgb = self.upsampling(rgb,
                              scale_factor=2,
                              mode='bilinear',
                              align_corners=False)
        rgb = self.outconv_rgb(rgb)
        rgb = torch.tanh(rgb)

        return rgb


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.downconv1_rgb = Conv2d(in_channels=3,
                                    out_channels=32,
                                    kernel_size=9,
                                    stride=1,
                                    padding=4)

        self.downconv2_rgb = ConvBlock(in_channels=32,
                                       out_channels=64,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv3_rgb = ConvBlock(in_channels=64,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv4_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv5_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downconv6_rgb = ConvBlock(in_channels=128,
                                       out_channels=128,
                                       kernel_size=4,
                                       stride=2,
                                       padding=1)

        self.downres1_rgb = ResBlock(in_channels=32,
                                     out_channels=32,
                                     kernel_size=9,
                                     padding=4)

        self.downres2_rgb = ResBlock(in_channels=64,
                                     out_channels=64,
                                     kernel_size=7,
                                     padding=3)

        self.downres3_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=5,
                                     padding=2)

        self.downres4_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1)

        self.downres5_rgb = ResBlock(in_channels=128,
                                     out_channels=128,
                                     kernel_size=3,
                                     padding=1)

        self.fuse = Conv2d(in_channels=128,
                           out_channels=128,
                           kernel_size=3,
                           stride=1,
                           padding=1)

    def forward(self, x):
        rgb = x[:, :3]
        rgb = self.downconv1_rgb(rgb)
        rgb = self.downres1_rgb(rgb)
        rgb = self.downconv2_rgb(rgb)
        rgb = self.downres2_rgb(rgb)
        rgb = self.downconv3_rgb(rgb)
        rgb = self.downres3_rgb(rgb)
        rgb = self.downconv4_rgb(rgb)
        rgb = self.downres4_rgb(rgb)
        rgb = self.downconv5_rgb(rgb)
        rgb = self.downres5_rgb(rgb)
        rgb = self.downconv6_rgb(rgb)

        x = self.fuse(rgb)

        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class FAED_wrapper(nn.Module):

    def __init__(self):
        super(FAED_wrapper, self).__init__()

        self.net = AutoEncoder()

    def forward(self, x):
        return self.net(x)

class FrechetAutoEncoderDistance(Metric):
    higher_is_better = False

    def __init__(self, pano_height: int, ckpt_path: str = None, **kwargs):
        super().__init__(**kwargs)
        ckpt_path = os.path.join('weights', 'faed.ckpt') if ckpt_path is None else ckpt_path
        faed = FAED_wrapper()
        faed.load_state_dict(torch.load(ckpt_path, weights_only=True)['state_dict'])
        self.encoder = faed.net.encoder

        num_features = pano_height * 4
        mx_num_feets = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feets).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def get_activation(self, imgs):
        imgs = (imgs.type(torch.float32) / 127.5) - 1
        features = self.encoder(imgs)
        mean_feature = torch.mean(features, dim=3)
        weight = torch.cos(
            torch.linspace(math.pi / 2, -math.pi / 2, mean_feature.shape[-1], device=mean_feature.device)
            ).unsqueeze(0).unsqueeze(0).expand_as(mean_feature)
        mean_feature = weight * mean_feature
        mean_vector = mean_feature.view(-1, (mean_feature.shape[-2] * mean_feature.shape[-1]))
        return mean_vector

    def update(self, imgs: Tensor, real: bool):
        features = self.get_activation(imgs)
        features = features.double()
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake)