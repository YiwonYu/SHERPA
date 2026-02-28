import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from .GridGenerator import GridGenerator
from diffusers.models.resnet import Upsample2D
from utils.pano import icosahedron_sample_camera
from external.Perspective_and_Equirectangular import e2p, mp2e_torch
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear



class SphereCompatibleConv2d(nn.Conv2d):
  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(SphereCompatibleConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None

    self.use_sphere_conv = True

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / (h - 1)) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / (w - 1)) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.Tensor(grid).to(self.weight.dtype)
      self.grid.requires_grad = False

  def _conv2d_forward(self, x):
    return F.conv2d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

  def _sphereconv2d_forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, groups=self.groups)

    return x  # (B, out_c, H/stride_h, W/stride_w)

  def forward(self, x):
    return self._sphereconv2d_forward(x) if self.use_sphere_conv else self._conv2d_forward(x)


class SphereAttention(nn.Module):

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, n_heads=8):
    super(SphereAttention, self).__init__()
    self.grid_shape = None
    self.grid = None
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = stride if isinstance(stride, tuple) else (stride, stride)
    self.n_heads = n_heads
    self.n_head_channels = in_channels // self.n_heads
    self.scale = self.n_head_channels ** -0.5
    assert self.n_heads * self.n_head_channels == in_channels, f"`in_channels` {in_channels} is not divisible by `n_heads` {n_heads}."
    self.nc = in_channels
    self.proj_q = nn.Conv2d(
        self.nc, self.nc,
        kernel_size=1, stride=1, padding=0
    )

    self.proj_k = nn.Conv2d(
        self.nc, self.nc,
        kernel_size=1, stride=1, padding=0
    )

    self.proj_v = nn.Conv2d(
        self.nc, self.nc,
        kernel_size=1, stride=1, padding=0
    )

    self.proj_out = nn.Conv2d(
        self.nc, out_channels,
        kernel_size=1, stride=1, padding=0
    )

  def genSamplingPattern(self, h, w, dtype):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / (h - 1)) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / (w - 1)) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.Tensor(grid).to(dtype)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    q = self.proj_q(x) # (B, C, H, W)

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W, x.dtype)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    Kh, Kw = self.kernel_size

    x_sampled = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    x_sampled = x_sampled.reshape(B, C, H, Kh, W, Kw).permute(0, 2, 4, 1, 3, 5) # (B, H, W, C, Kh, Kw)
    x_sampled = x_sampled.reshape(B * H * W, C, Kh, Kw) # (B*H*W, C, Kh, Kw)
    
    q = q.permute(0, 2, 3, 1).reshape(B * H * W, C, 1).reshape(B * self.n_heads * H * W, self.n_head_channels, 1) # (B*Nh*H*W, Ch, 1)
    k = self.proj_k(x_sampled).reshape(B * self.n_heads * H * W, self.n_head_channels, -1) # (B*Nh*H*W, Ch, Kh*Kw)
    v = self.proj_v(x_sampled).reshape(B * self.n_heads * H * W, self.n_head_channels, -1) # (B*Nh*H*W, Ch, Kh*Kw)

    attn = torch.einsum('b c m, b c n -> b m n', q, k) # (B*Nh*H*W, 1, Kh*Kw)
    attn = attn.mul(self.scale)

    attn = F.softmax(attn, dim=2)

    out = torch.einsum('b m n, b c n -> b c m', attn, v) # (B*Nh*H*W, Ch, 1)

    out = out.reshape(B * H * W, C, 1).reshape(B, H, W, C).permute(0, 3, 1, 2) # (B, C, H, W)

    y = self.proj_out(out)

    return y, None, None


class SphereConv2d(nn.Conv2d):
  """
  kernel_size: (H, W)
  """

  def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
               stride=1, padding=0, dilation=1,
               groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
    super(SphereConv2d, self).__init__(
      in_channels, out_channels, kernel_size,
      stride, padding, dilation, groups, bias, padding_mode)
    self.grid_shape = None
    self.grid = None

  def genSamplingPattern(self, h, w):
    gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
    LonLatSamplingPattern = gridGenerator.createSamplingPattern()

    # generate grid to use `F.grid_sample`
    lat_grid = (LonLatSamplingPattern[:, :, :, 0] / (h - 1)) * 2 - 1
    lon_grid = (LonLatSamplingPattern[:, :, :, 1] / (w - 1)) * 2 - 1

    grid = np.stack((lon_grid, lat_grid), axis=-1)
    with torch.no_grad():
      self.grid = torch.Tensor(grid).to(self.weight.dtype)
      self.grid.requires_grad = False

  def forward(self, x):
    # Generate Sampling Pattern
    B, C, H, W = x.shape

    if (self.grid_shape is None) or (self.grid_shape != (H, W)):
      self.grid_shape = (H, W)
      self.genSamplingPattern(H, W)

    with torch.no_grad():
      grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
      grid.requires_grad = False

    x = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

    # self.weight -> (out_c, in_c, Kh, Kw)
    x = F.conv2d(x, self.weight, self.bias, stride=self.kernel_size, groups=self.groups)

    return x  # (B, out_c, H/stride_h, W/stride_w)



class SphereUpsample2D(Upsample2D):
    def __init__(self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels=None,
        name: str = "conv",):
        super(SphereUpsample2D, self).__init__(channels, use_conv, use_conv_transpose, out_channels, name)
        self.use_sphere_conv = True

    def forward_pano(self, hidden_states):
        assert hidden_states.shape[1] == self.channels

        B, C, H, W = hidden_states.shape

        # if self.norm is not None:
        #     hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        theta, phi = icosahedron_sample_camera()
        theta, phi = np.rad2deg(theta), np.rad2deg(phi)

        fov = np.full_like(theta, 90, dtype=int)

        b, c, eh, ew = hidden_states.shape
        x = []
        for i in range(b):
            pers = e2p(hidden_states[i][None].repeat(len(theta), 1, 1, 1), fov, theta, phi, (eh * 2, eh * 2))

            equi = mp2e_torch(pers[:, None], fov, theta, phi, (eh * 2, ew * 2))
            x.append(equi[0].to(dtype))

        hidden_states = torch.stack(x).to(hidden_states.device)

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

    def forward_pers(
        self,
        hidden_states,
        output_size=None,
        scale=1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                if isinstance(self.conv, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.conv(hidden_states, scale)
                else:
                    hidden_states = self.conv(hidden_states)
            else:
                if isinstance(self.Conv2d_0, LoRACompatibleConv) and not USE_PEFT_BACKEND:
                    hidden_states = self.Conv2d_0(hidden_states, scale)
                else:
                    hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

    def forward(self, x, output_size=None, scale=1.0,):
        return self.forward_pano(x) if self.use_sphere_conv else self.forward_pers(x, output_size, scale)



class SphereCompatibleUpsample2D(Upsample2D):
    def __init__(self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels=None,
        name: str = "conv",
        scale: int = 2,):
        super(SphereCompatibleUpsample2D, self).__init__(channels, use_conv, use_conv_transpose, out_channels, name)
        self.scale = scale
        self.use_sphere_conv = True

        conv = None
        # if use_conv_transpose:
        #     conv = nn.ConvTranspose2d(channels, self.out_channels, 4, 2, 1)
        if use_conv:
            conv = SphereCompatibleConv2d(self.channels, self.out_channels, 3, padding=1)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def genSamplingPattern(self, h, w):
        gridGenerator = GridGenerator(h, w, (self.scale, self.scale), (1, 1))
        LonLatSamplingPattern = gridGenerator.createSamplingPattern()

        # generate grid to use `F.grid_sample`
        lat_grid = (LonLatSamplingPattern[:, :, :, 0] / (h - 1)) * 2 - 1
        lon_grid = (LonLatSamplingPattern[:, :, :, 1] / (w - 1)) * 2 - 1

        with torch.no_grad():
            self.grid = torch.Tensor(np.stack((lon_grid, lat_grid), axis=-1))

    def forward(self, hidden_states: torch.Tensor, output_size=None, *args, **kwargs) -> torch.Tensor:
        assert hidden_states.shape[1] == self.channels

        B, C, H, W = hidden_states.shape

        # if self.norm is not None:
        #     hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        # if self.interpolate:
        self.genSamplingPattern(H, W)
        with torch.no_grad():
            grid = self.grid.repeat((B, 1, 1, 1)).to(hidden_states.device)  # (B, H*Kh, W*Kw, 2)
            grid.requires_grad = False

        hidden_states = F.grid_sample(hidden_states, grid, align_corners=True, mode='nearest')
            # if output_size is None:
            #     hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
            # else:
            #     hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np

# class SphereConv2d(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3),
#                stride=1, padding=0, dilation=1,
#                groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
#         super(SphereConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = self._pair(kernel_size)
#         self.stride = self._pair(stride)
#         self.padding = self._pair(padding)
#         self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size))
#         self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / np.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)
    
#     def _pair(self, x):
#         if isinstance(x, tuple):
#             return x
#         return (x, x)
    
#     def forward(self, x):
#         batch_size, in_channels, height, width = x.size()
#         assert in_channels == self.in_channels

#         # Compute spherical coordinates
#         theta = torch.linspace(0, np.pi, height, device=x.device).view(1, height, 1).expand(batch_size, height, width)
#         phi = torch.linspace(0, 2 * np.pi, width, device=x.device).view(1, 1, width).expand(batch_size, height, width)

#         # Compute sampling grid
#         kernel_radius_h = self.kernel_size[0] // 2
#         kernel_radius_w = self.kernel_size[1] // 2
#         sampling_grid = []
#         for i in range(-kernel_radius_h, kernel_radius_h + 1):
#             for j in range(-kernel_radius_w, kernel_radius_w + 1):
#                 dtheta = i * np.pi / height
#                 dphi = j * 2 * np.pi / width
#                 theta_sample = torch.clamp(theta + dtheta, 0, np.pi)
#                 phi_sample = (phi + dphi) % (2 * np.pi)
#                 sampling_grid.append(torch.stack([phi_sample, theta_sample], dim=-1))
        
#         sampling_grid = torch.stack(sampling_grid, dim=-2).view(batch_size, height, width, self.kernel_size[0] * self.kernel_size[1], 2)
#         sampling_grid = sampling_grid.permute(0, 3, 1, 2, 4).contiguous().view(batch_size * self.kernel_size[0] * self.kernel_size[1], height, width, 2)
        
#         # Sample input
#         x_expanded = x.unsqueeze(1).expand(batch_size, self.kernel_size[0] * self.kernel_size[1], in_channels, height, width)
#         x_expanded = x_expanded.contiguous().view(batch_size * self.kernel_size[0] * self.kernel_size[1], in_channels, height, width)
#         sampled_input = F.grid_sample(x_expanded, sampling_grid, align_corners=True)
#         sampled_input = sampled_input.view(batch_size, in_channels, height * self.kernel_size[0], width * self.kernel_size[1])
        
#         # # Reshape for convolution
#         # sampled_input = sampled_input.permute(0, 2, 3, 4, 1).contiguous().view(batch_size, in_channels, height, width, -1)
#         # sampled_input = sampled_input.permute(0, 2, 3, 1, 4).contiguous().view(batch_size * height * width, in_channels, -1)
#         output = F.conv2d(sampled_input, self.weight, self.bias, stride=self.kernel_size)

#         # print(sampled_input.shape)
        
#         # weight = self.weight.view(self.out_channels * self.kernel_size[0] * self.kernel_size[1], self.in_channels)
#         # output = torch.bmm(sampled_input, weight.t().unsqueeze(0).expand(batch_size * height * width, -1, -1))
        
#         # Reshape to final output
#         # output = output.view(batch_size, height, width, self.out_channels)
#         # output = output.permute(0, 3, 1, 2).contiguous()
        
#         # if self.bias is not None:
#         #     output = output + self.bias.view(1, -1, 1, 1)
        
#         return output