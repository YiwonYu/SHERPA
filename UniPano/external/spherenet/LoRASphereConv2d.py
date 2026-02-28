import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from diffusers.models.lora import *

from .GridGenerator import GridGenerator
from .SphereConv2d import SphereConv2d


class LoRASphereConv2dLayer(nn.Module):
    r"""
    A convolutional layer that is used with LoRA.

    Parameters:
        in_features (`int`):
            Number of input features.
        out_features (`int`):
            Number of output features.
        rank (`int`, `optional`, defaults to 4):
            The rank of the LoRA layer.
        kernel_size (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The kernel size of the convolution.
        stride (`int` or `tuple` of two `int`, `optional`, defaults to 1):
            The stride of the convolution.
        padding (`int` or `tuple` of two `int` or `str`, `optional`, defaults to 0):
            The padding of the convolution.
        network_alpha (`float`, `optional`, defaults to `None`):
            The value of the network alpha used for stable learning and preventing underflow. This value has the same
            meaning as the `--network_alpha` option in the kohya-ss trainer script. See
            https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        kernel_size: Union[int, Tuple[int, int]] = (1, 1),
        stride: Union[int, Tuple[int, int]] = (1, 1),
        padding: Union[int, Tuple[int, int], str] = 0,
        network_alpha: Optional[float] = None,
    ):
        super().__init__()

        self.down = SphereConv2d(in_features, rank, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        # according to the official kohya_ss trainer kernel_size are always fixed for the up layer
        # # see: https://github.com/bmaltais/kohya_ss/blob/2accb1305979ba62f5077a23aabac23b4c37e935/networks/lora_diffusers.py#L129
        self.up = SphereConv2d(rank, out_features, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)


class LoRACompatibleSphereConv2d(nn.Conv2d):
    """
    A convolutional layer that can be used with LoRA.
    """

    def __init__(self, *args, lora_layer: Optional[LoRASphereConv2dLayer] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_layer = lora_layer

        self.grid_shape = None
        self.grid = None

        self.use_sphere_conv = True

    def set_lora_layer(self, lora_layer: Optional[LoRASphereConv2dLayer]):
        self.lora_layer = lora_layer

    def genSamplingPattern(self, h, w):
        gridGenerator = GridGenerator(h, w, self.kernel_size, self.stride)
        LonLatSamplingPattern = gridGenerator.createSamplingPattern()

        # generate grid to use `F.grid_sample`
        lat_grid = (LonLatSamplingPattern[:, :, :, 0] / (h - 1)) * 2 - 1
        lon_grid = (LonLatSamplingPattern[:, :, :, 1] / (w - 1)) * 2 - 1

        grid = np.stack((lon_grid, lat_grid), axis=-1)
        with torch.no_grad():
            self.grid = torch.Tensor(grid).to(self.weight.dtype) #torch.FloatTensor(grid)
            self.grid.requires_grad = False

    def _fuse_lora(self, lora_scale: float = 1.0, safe_fusing: bool = False):
        if self.lora_layer is None:
            return

        dtype, device = self.weight.data.dtype, self.weight.data.device

        w_orig = self.weight.data.float()
        w_up = self.lora_layer.up.weight.data.float()
        w_down = self.lora_layer.down.weight.data.float()

        if self.lora_layer.network_alpha is not None:
            w_up = w_up * self.lora_layer.network_alpha / self.lora_layer.rank

        fusion = torch.mm(w_up.flatten(start_dim=1), w_down.flatten(start_dim=1))
        fusion = fusion.reshape((w_orig.shape))
        fused_weight = w_orig + (lora_scale * fusion)

        if safe_fusing and torch.isnan(fused_weight).any().item():
            raise ValueError(
                "This LoRA weight seems to be broken. "
                f"Encountered NaN values when trying to fuse LoRA weights for {self}."
                "LoRA weights will not be fused."
            )

        self.weight.data = fused_weight.to(device=device, dtype=dtype)

        # we can drop the lora layer now
        self.lora_layer = None

        # offload the up and down matrices to CPU to not blow the memory
        self.w_up = w_up.cpu()
        self.w_down = w_down.cpu()
        self._lora_scale = lora_scale

    def _unfuse_lora(self):
        if not (getattr(self, "w_up", None) is not None and getattr(self, "w_down", None) is not None):
            return

        fused_weight = self.weight.data
        dtype, device = fused_weight.data.dtype, fused_weight.data.device

        self.w_up = self.w_up.to(device=device).float()
        self.w_down = self.w_down.to(device).float()

        fusion = torch.mm(self.w_up.flatten(start_dim=1), self.w_down.flatten(start_dim=1))
        fusion = fusion.reshape((fused_weight.shape))
        unfused_weight = fused_weight.float() - (self._lora_scale * fusion)
        self.weight.data = unfused_weight.to(device=device, dtype=dtype)

        self.w_up = None
        self.w_down = None

    def _sphereconv_forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        B, C, H, W = x.shape

        if (self.grid_shape is None) or (self.grid_shape != (H, W)):
            self.grid_shape = (H, W)
            self.genSamplingPattern(H, W)

        with torch.no_grad():
            grid = self.grid.repeat((B, 1, 1, 1)).to(x.device)  # (B, H*Kh, W*Kw, 2)
            grid.requires_grad = False

        hidden_states = F.grid_sample(x, grid, align_corners=True, mode='nearest')  # (B, in_c, H*Kh, W*Kw)

        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, stride=self.kernel_size
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, stride=self.kernel_size
            )
            return original_outputs + (scale * self.lora_layer(x))
    
    def _conv_forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        if self.lora_layer is None:
            # make sure to the functional Conv2D function as otherwise torch.compile's graph will break
            # see: https://github.com/huggingface/diffusers/pull/4315
            return F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        else:
            original_outputs = F.conv2d(
                hidden_states, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )
            return original_outputs + (scale * self.lora_layer(hidden_states))

    def forward(self, x: torch.Tensor, scale: float = 1.0):
        return self._sphereconv_forward(x, scale) if self.use_sphere_conv else self._conv_forward(x, scale)