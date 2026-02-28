import torch
import torch.nn as nn
from ..modules.transformer import BasicTransformerBlock, SphericalPE
from .utils import get_coords, get_masks
from einops import rearrange, repeat
from external.spherenet import SphereConv2d, SphereAttention
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from diffusers.utils import USE_PEFT_BACKEND, deprecate, logging
from typing import Callable, Optional, Union
import copy
# from DCNv4 import DCNv4
from ..modules.dat import LocalAttention, DAttentionBaseline
from ..modules.senet import SEBlock
from ..modules.moe import SparseMoeLoRA


class WarpAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(
            dim, dim//32, 32, context_dim=dim)
        self.pe = SphericalPE(dim//4)

    def forward(self, pers_x, equi_x, cameras):
        bm, c, pers_h, pers_w = pers_x.shape
        b, c, equi_h, equi_w = equi_x.shape
        m = bm // b
        pers_masks, equi_masks = get_masks(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)
        pers_coords, equi_coords = get_coords(
            pers_h, pers_w, equi_h, equi_w, cameras, pers_x.device, pers_x.dtype)

        # # cross attention from perspective to equirectangular
        # pers_xs = rearrange(pers_xs, '(b m) c h w -> b m c h w', m=m)
        # pers_masks = rearrange(pers_masks, '(b m) ... -> b m ...', m=m)
        # equi_masks = rearrange(equi_masks, '(b m) ... -> b m ...', m=m)
        # pers_xs_out, equi_xs_out = [], []
        # for pers_x, equi_x, pers_mask, equi_mask in zip(pers_xs, equi_xs, pers_masks, equi_masks):
        #     key_value = rearrange(pers_x, 'm c h w -> (m h w) c', m=m)
        #     query = rearrange(equi_x, 'c h w -> (h w) c')
        #     pers_mask = rearrange(pers_mask, 'm h w -> b (m h w)', m=m)
        #     out = self.transformer(query, key_value, pers_masks)

        # add positional encoding
        pers_pe = self.pe(pers_coords)
        pers_pe = rearrange(pers_pe, 'b h w c -> b c h w')
        pers_x_wpe = pers_x + pers_pe
        equi_pe = self.pe(equi_coords)
        equi_pe = repeat(equi_pe, 'h w c -> b c h w', b=b)
        equi_x_wpe = equi_x + equi_pe

        # cross attention from perspective to equirectangular
        query = rearrange(equi_x, 'b c h w -> b (h w) c')
        key_value = rearrange(pers_x_wpe, '(b m) c h w -> b (m h w) c', m=m)
        pers_masks = rearrange(pers_masks, '(b m) eh ew ph pw -> b (eh ew) (m ph pw)', m=m)
        equi_pe = rearrange(equi_pe, 'b c h w -> b (h w) c')
        equi_x_out = self.transformer(query, key_value, mask=pers_masks, query_pe=equi_pe)

        # cross attention from equirectangular to perspective
        query = rearrange(pers_x, '(b m) c h w -> b (m h w) c', m=m)
        key_value = rearrange(equi_x_wpe, 'b c h w -> b (h w) c')
        equi_masks = rearrange(equi_masks, '(b m) ph pw eh ew -> b (m ph pw) (eh ew)', m=m)
        pers_pe = rearrange(pers_pe, '(b m) c h w -> b (m h w) c', m=m)
        pers_x_out = self.transformer(query, key_value, mask=equi_masks, query_pe=pers_pe)

        pers_x_out = rearrange(pers_x_out, 'b (m h w) c -> (b m) c h w', m=m, h=pers_h, w=pers_w)
        equi_x_out = rearrange(equi_x_out, 'b (h w) c -> b c h w', h=equi_h, w=equi_w)
        return pers_x_out, equi_x_out


class AdaptableAttnProcessor():
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AdaptableLoRAAttnProcessor(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism with additional adaptable LoRA layer for `to_out`.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
        kwargs (`dict`):
            Additional keyword arguments to pass to the `LoRALinearLayer` layers.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        network_alpha: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = out_hidden_size if out_hidden_size is not None else hidden_size

        # self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        # self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        self.to_out_lora = SparseMoeLoRA(out_hidden_size, out_rank, network_alpha, num_experts=4)

    def __call__(self, attn: Attention, hidden_states: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        self_cls_name = self.__class__.__name__
        # deprecate(
        #     self_cls_name,
        #     "0.26.0",
        #     (
        #         f"Make sure use {self_cls_name[4:]} instead by setting"
        #         "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
        #         " `LoraLoaderMixin.load_lora_weights`"
        #     ),
        # )
        # attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        # attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        attn.processor = AdaptableAttnProcessor()
        return attn.processor(attn, hidden_states, *args, **kwargs)
