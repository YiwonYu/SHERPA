import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import xformers.ops as xops
import math

from typing import Callable, Optional, Union
from diffusers.models.lora import LoRACompatibleLinear, LoRALinearLayer
from diffusers.models.attention_processor import AttnProcessor, Attention


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        linear = nn.Linear(inner_dim, dim_out)
        linear.weight.data.fill_(0)
        linear.bias.data.fill_(0)
        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            linear
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim

        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, query_dim)
        self.to_out.weight.data.fill_(0)
        self.to_out.bias.data.fill_(0)

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)

        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        mask = repeat(mask, 'b i j -> (b h) i j', h=h)
        # with torch.backends.cuda.sdp_kernel(enable_math=False, enable_mem_efficient=False):
        #     out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = xops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        return self.to_out(out)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True)
                             for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


class BasicTransformerBlock(nn.Module):

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True,
                 disable_self_attn=False, use_checkpoint=True):
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                                    context_dim=context_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.checkpoint = checkpoint
        self.use_checkpoint=use_checkpoint

    def forward(self, x, context=None, mask=None, query_pe=None):
        if self. use_checkpoint:
            return checkpoint(self._forward, (x, context, mask, query_pe), self.parameters(), self.checkpoint)
        else:
            return self._forward(x, context, mask, query_pe)

    def _forward(self, x, context=None, mask=None, query_pe=None):
        if context is None:
            context = x
        query=x
        if query_pe is not None:
            query=query+query_pe
        query=self.norm1(query)
        context=self.norm1(context)
        x = self.attn1(query, context=context, mask=mask) + x
        x = self.ff(self.norm2(x)) + x

        return x


class SphericalPE(nn.Module):
    def __init__(self, N_freqs, logscale=True):
        """
        Defines a function that embeds theta and phi to a vector of size N_freqs*2
        """
        super(SphericalPE, self).__init__()
        self.N_freqs = N_freqs
        # self.funcs = [torch.sin, torch.cos]
        # self.out_channels = in_channels*(len(self.funcs)*N_freqs)
        if N_freqs <= 80:
            base = 2
        else:
            base = 5000**(1/(N_freqs/2.5))
        if logscale:
            freq_bands = base**torch.linspace(0,
                                              N_freqs-1, N_freqs)
        else:
            freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)
        self.register_buffer('freq_bands', freq_bands)

        self.fc = nn.Linear(N_freqs * 4, N_freqs * 4)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, coords):
        """
        Embeds coords = (theta, phi) to a vector of size N_freqs*2
        Inputs:
            coords: (B, 2)
        Outputs:
            out: (B, N_freqs * 2)
        """
        shape = coords.shape[:-1]
        coords = coords.reshape(-1, 2, 1)

        encodings = coords * self.freq_bands
        sin_encodings = torch.sin(encodings)  # (n, c, num_encoding_functions)
        cos_encodings = torch.cos(encodings)
        pe = torch.cat([sin_encodings, cos_encodings], dim=1)
        pe = pe.reshape(*shape, -1)
        return self.fc(pe)


class PositionalEncoding2D(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding2D, self).__init__()
        self.fc = nn.Linear(dim, dim)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.dim = dim

    def forward(self, h, w):
        """
        Embeds coords = (theta, phi) to a vector of size N_freqs*2
        Inputs:
            coords: (B, 2)
        Outputs:
            out: (B, N_freqs * 2)
        """
        pers_pe = positionalencoding2d(self.dim, h, w, self.fc.weight.device)
        pers_pe = rearrange(pers_pe, 'c h w -> h w c')
        return self.fc(pers_pe)


"""
credited to https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
"""
def positionalencoding2d(d_model, height, width, device):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width).to(device)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


class LoRAAttnProcessor_q_out(nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism applied to `to_q` and `to_out` only.

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

        self.to_q_lora = LoRALinearLayer(q_hidden_size, q_hidden_size, q_rank, network_alpha)
        # self.to_k_lora = LoRALinearLayer(cross_attention_dim or hidden_size, hidden_size, rank, network_alpha)
        # self.to_v_lora = LoRALinearLayer(cross_attention_dim or v_hidden_size, v_hidden_size, v_rank, network_alpha)
        # self.to_out_lora = LoRALinearLayer(out_hidden_size, out_hidden_size, out_rank, network_alpha)

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
        attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
        # attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
        # attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
        # attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)

        attn._modules.pop("processor")
        attn.processor = AttnProcessor()
        return attn.processor(attn, hidden_states, *args, **kwargs)
