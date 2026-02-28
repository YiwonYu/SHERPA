from .PanoGenerator import PanoGenerator
import torch
import os
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
from einops import rearrange

# import sys
# sys.path.append('../..')
from utils.pano import CubemapNoise
import numpy as np
from external.Perspective_and_Equirectangular import mp2e
import lpips


class VAEOnly(PanoGenerator):
    def instantiate_model(self):
        self.lpips_loss_fn = lpips.LPIPS(net="alex").to(self.device)

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        b = batch['pano'].shape[0]

        # pano_pad = self.pad_pano(batch['pano'])
        # pano_latent_pad = self.encode_image(pano_pad, self.vae)
        # pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
        # pano_latent = self.encode_image(batch['pano'], self.vae)

        x_input = rearrange(batch['pano'].to(self.vae.dtype), 'b l c h w -> (b l) c h w')
        posterior = self.vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = posterior.mode()
        # z = rearrange(z, '(b l) c h w -> b l c h w', b=b)

        # use the scaling factor from the vae config
        # z = z * self.vae.config.scaling_factor
        # z = z.to(self.dtype)

        recon = self.vae.decode(z).sample

        # print(recon.shape, x_input.shape, z.shape)

        # kl_loss = posterior.kl().mean()
        mse_loss = torch.nn.functional.mse_loss(recon, x_input)
        lpips_loss = self.lpips_loss_fn(recon, x_input).mean()
        # loss_mask = torch.ones_like(denoise, device=pano_latent.device)
        # loss_mask[:, :, :, :(loss_mask.shape[-2] // 8)] = 0.
        # loss_mask[:, :, :, -(loss_mask.shape[-2] // 8):] = 0.
        # loss *= loss_mask
        # loss = loss.mean()
        loss = mse_loss + 1e-1 * lpips_loss #+ 1e-6 * kl_loss
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/mse_loss', mse_loss)
        # self.log('train/kl_loss', kl_loss)
        self.log('train/lpips_loss', lpips_loss)
        return loss


    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()
        device = self.device

        pano_latent = self.encode_image(batch['pano'], self.vae) #self.init_noise(bs, batch['height'].item()//8, batch['width'].item()//8, device=device, batch=batch)

        pano_pred = self.decode_latent(pano_latent, self.vae)
        pano_pred = tensor_to_image(pano_pred)

        return pano_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pano_pred = self.inference(batch)
        self.log_val_image(pano_pred, batch['pano'], batch['pano_prompt'],
                           batch.get('pano_layout_cond'))

    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, pano_pred, pano, pano_prompt,
                      pano_layout_cond=None):
        log_dict = {
            'val/pano_pred': self.temp_wandb_image(
                pano_pred[0, 0], pano_prompt[0] if pano_prompt else None),
            'val/pano_gt': self.temp_wandb_image(
                pano[0, 0], pano_prompt[0] if pano_prompt else None),
        }
        if pano_layout_cond is not None:
            log_dict['val/pano_layout_cond'] = self.temp_wandb_image(
                pano_layout_cond[0, 0], pano_prompt[0] if pano_prompt else None)
        self.logger.experiment.log(log_dict)
