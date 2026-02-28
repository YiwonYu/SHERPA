import torch
import torch.nn as nn
# from .modules import WarpAttn
from einops import rearrange, repeat
from utils.pano import pad_pano, unpad_pano

from .utils import check_sphere_conv, toggle_sphere_conv, get_equi_coords, get_pers_coords


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, pano_unet, pers_cn=None, pano_cn=None, pano_pad=True, use_pe=False, use_adapter=False):
        super().__init__()

        self.unet = None #unet
        self.pano_unet = pano_unet
        self.pers_cn = None #pers_cn
        self.pano_cn = pano_cn
        self.pano_pad = pano_pad


    def forward_pers(self, latents, timestep, prompt_embd, cameras=None, pers_layout_cond=None):
        # bs*m, 4, 64, 64
        b, m, c, h, w = latents.shape
        hidden_states = rearrange(latents, 'b m c h w -> (b m) c h w')
        if cameras is not None:
            cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        if prompt_embd is not None:
            prompt_embd = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps
        
        timestep = timestep.reshape(-1)
        t_emb = self.pano_unet.time_proj(timestep).to(self.pano_unet.dtype)  # (bs, 320)
        emb = self.pano_unet.time_embedding(t_emb)  # (bs, 1280)

        if self.pers_cn is None:
            pers_layout_cond = None
        if pers_layout_cond is not None:
            pers_layout_cond = rearrange(pers_layout_cond, 'b m ... -> (b m) ...')
            down_block_additional_residuals, mid_block_additional_residual = self.pers_cn(
                hidden_states,
                timestep,
                encoder_hidden_states=prompt_embd,
                controlnet_cond=pers_layout_cond,
                return_dict=False,
            )

        hidden_states = self.pano_unet.conv_in(hidden_states)  # bs*m, 320, 64, 64

        # unet
        # a. downsample
        down_block_res_samples = (hidden_states,)
        for i, downsample_block in enumerate(self.pano_unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for j in range(len(downsample_block.resnets)):
                    hidden_states = self.pano_unet.down_blocks[i].resnets[j](hidden_states, emb)

                    hidden_states = self.pano_unet.down_blocks[i].attentions[j](
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample

                    down_block_res_samples += (hidden_states,)
            else:
                for j in range(len(downsample_block.resnets)):
                    hidden_states = self.pano_unet.down_blocks[i].resnets[j](hidden_states, emb)
                    down_block_res_samples += (hidden_states,)

            if downsample_block.downsamplers is not None:
                for j in range(len(downsample_block.downsamplers)):
                    hidden_states = self.pano_unet.down_blocks[i].downsamplers[j](hidden_states)
                down_block_res_samples += (hidden_states,)

                # if self.unet is not None:
                #     hidden_states, pano_hidden_states = self.cp_blocks_encoder[i](
                #         hidden_states, pano_hidden_states, cameras)

        if pers_layout_cond is not None:
            new_down_block_res_samples = ()
            for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)
            down_block_res_samples = new_down_block_res_samples

        # b. mid
        hidden_states = self.pano_unet.mid_block.resnets[0](
            hidden_states, emb)

        for i in range(len(self.pano_unet.mid_block.attentions)):
            hidden_states = self.pano_unet.mid_block.attentions[i](
                hidden_states, encoder_hidden_states=prompt_embd
            ).sample
            hidden_states = self.pano_unet.mid_block.resnets[i+1](hidden_states, emb)

        if pers_layout_cond is not None:
            hidden_states = hidden_states + mid_block_additional_residual

        # if self.unet is not None:
        #     hidden_states, pano_hidden_states = self.cp_blocks_mid(
        #         hidden_states, pano_hidden_states, cameras)

        # c. upsample
        for i, upsample_block in enumerate(self.pano_unet.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for j in range(len(upsample_block.resnets)):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = self.pano_unet.up_blocks[i].resnets[j](hidden_states, emb)
                    
                    hidden_states = self.pano_unet.up_blocks[i].attentions[j](
                        hidden_states, encoder_hidden_states=prompt_embd
                    ).sample
            else:
                for j in range(len(upsample_block.resnets)):
                    res_hidden_states = res_samples[-1]
                    res_samples = res_samples[:-1]
                    hidden_states = torch.cat(
                        [hidden_states, res_hidden_states], dim=1)
                    hidden_states = self.pano_unet.up_blocks[i].resnets[j](hidden_states, emb)

            if upsample_block.upsamplers is not None:
                # if self.unet is not None:
                #     hidden_states, pano_hidden_states = self.cp_blocks_decoder[i](
                #         hidden_states, pano_hidden_states, cameras)
                for j in range(len(upsample_block.upsamplers)):
                    if latents is not None: #self.unet is not None:
                        hidden_states = self.pano_unet.up_blocks[i].upsamplers[j](hidden_states)

        # 4.post-process
        sample = self.pano_unet.conv_norm_out(hidden_states)
        sample = self.pano_unet.conv_act(sample)
        sample = self.pano_unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)

        return sample

    def forward_pano(self, pano_latent, timestep, pano_prompt_embd, pano_layout_cond=None):
        pano_latent = rearrange(pano_latent, 'b m c h w -> (b m) c h w')
        pano_prompt_embd = rearrange(pano_prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps
        pano_timestep = timestep
        pano_t_emb = self.pano_unet.time_proj(pano_timestep).to(self.pano_unet.dtype)  # (bs, 320)
        pano_emb = self.pano_unet.time_embedding(pano_t_emb)  # (bs, 1280)

        if self.pano_cn is None:
            pano_layout_cond = None
        if pano_layout_cond is not None:
            pano_layout_cond = rearrange(pano_layout_cond, 'b m ... -> (b m) ...')
            pano_down_block_additional_residuals, pano_mid_block_additional_residual = self.pano_cn(
                pano_latent,
                pano_timestep,
                encoder_hidden_states=pano_prompt_embd,
                controlnet_cond=pano_layout_cond,
                return_dict=False,
            )

        if self.pano_pad:
            pano_latent = pad_pano(pano_latent, 1)
        pano_hidden_states = self.pano_unet.conv_in(pano_latent)
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 1)

        # unet
        # a. downsample
        pano_down_block_res_samples = (pano_hidden_states,)
        for i, downsample_block in enumerate(self.pano_unet.down_blocks):
            if hasattr(downsample_block, 'has_cross_attention') and downsample_block.has_cross_attention:
                for j in range(len(downsample_block.resnets)):
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

                    pano_hidden_states = self.pano_unet.down_blocks[i].attentions[j](
                        pano_hidden_states, encoder_hidden_states=pano_prompt_embd
                    ).sample

                    pano_down_block_res_samples += (pano_hidden_states,)
            else:
                for j in range(len(downsample_block.resnets)):
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    pano_down_block_res_samples += (pano_hidden_states,)

            if downsample_block.downsamplers is not None:
                for j in range(len(downsample_block.downsamplers)):
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.down_blocks[i].downsamplers[j](
                        pano_hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 1)

                pano_down_block_res_samples += (pano_hidden_states,)

                # if self.unet is not None:
                #     hidden_states, pano_hidden_states = self.cp_blocks_encoder[i](
                #         hidden_states, pano_hidden_states, cameras)

        if pano_layout_cond is not None:
            new_pano_down_block_res_samples = ()
            for pano_down_block_res_sample, pano_down_block_additional_residual in zip(
                pano_down_block_res_samples, pano_down_block_additional_residuals
            ):
                pano_down_block_res_sample = pano_down_block_res_sample + pano_down_block_additional_residual
                new_pano_down_block_res_samples = new_pano_down_block_res_samples + (pano_down_block_res_sample,)
            pano_down_block_res_samples = new_pano_down_block_res_samples

        # b. mid
        if self.pano_pad:
            pano_hidden_states = pad_pano(pano_hidden_states, 2)
        pano_hidden_states = self.pano_unet.mid_block.resnets[0](
            pano_hidden_states, pano_emb)
        if self.pano_pad:
            pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        for i in range(len(self.pano_unet.mid_block.attentions)):
            pano_hidden_states = self.pano_unet.mid_block.attentions[i](
                pano_hidden_states, encoder_hidden_states=pano_prompt_embd
            ).sample
            if self.pano_pad:
                pano_hidden_states = pad_pano(pano_hidden_states, 2)
            pano_hidden_states = self.pano_unet.mid_block.resnets[i+1](
                pano_hidden_states, pano_emb)
            if self.pano_pad:
                pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        if pano_layout_cond is not None:
            pano_hidden_states = pano_hidden_states + pano_mid_block_additional_residual

        # if self.unet is not None:
        #     hidden_states, pano_hidden_states = self.cp_blocks_mid(
        #         hidden_states, pano_hidden_states, cameras)

        # c. upsample
        for i, upsample_block in enumerate(self.pano_unet.up_blocks):
            pano_res_samples = pano_down_block_res_samples[-len(upsample_block.resnets):]
            pano_down_block_res_samples = pano_down_block_res_samples[:-len(
                upsample_block.resnets)]

            if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
                for j in range(len(upsample_block.resnets)):
                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat(
                        [pano_hidden_states, pano_res_hidden_states], dim=1)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.up_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)
                    
                    pano_hidden_states = self.pano_unet.up_blocks[i].attentions[j](
                        pano_hidden_states, encoder_hidden_states=pano_prompt_embd
                    ).sample
            else:
                for j in range(len(upsample_block.resnets)):
                    pano_res_hidden_states = pano_res_samples[-1]
                    pano_res_samples = pano_res_samples[:-1]
                    pano_hidden_states = torch.cat(
                        [pano_hidden_states, pano_res_hidden_states], dim=1)
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 2)
                    pano_hidden_states = self.pano_unet.up_blocks[i].resnets[j](
                        pano_hidden_states, pano_emb)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

            if upsample_block.upsamplers is not None:
                # if self.unet is not None:
                #     hidden_states, pano_hidden_states = self.cp_blocks_decoder[i](
                #         hidden_states, pano_hidden_states, cameras)

                for j in range(len(upsample_block.upsamplers)):
                    if self.pano_pad:
                        pano_hidden_states = pad_pano(pano_hidden_states, 1)
                    pano_hidden_states = self.pano_unet.up_blocks[i].upsamplers[j](
                        pano_hidden_states)
                    if self.pano_pad:
                        pano_hidden_states = unpad_pano(pano_hidden_states, 2)

        pano_sample = self.pano_unet.conv_norm_out(pano_hidden_states)
        pano_sample = self.pano_unet.conv_act(pano_sample)
        if self.pano_pad:
            pano_sample = pad_pano(pano_sample, 1)
        pano_sample = self.pano_unet.conv_out(pano_sample)
        if self.pano_pad:
            pano_sample = unpad_pano(pano_sample, 1)
        pano_sample = rearrange(pano_sample, '(b m) c h w -> b m c h w', m=1)

        return pano_sample

    def forward(self, latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, cameras,
                pers_layout_cond=None, pano_layout_cond=None):
        sample = self.forward_pers(latents, timestep, prompt_embd, cameras, pers_layout_cond)
        pano_sample = self.forward_pano(pano_latent, timestep[:, 0], pano_prompt_embd, pano_layout_cond)

        return sample, pano_sample
