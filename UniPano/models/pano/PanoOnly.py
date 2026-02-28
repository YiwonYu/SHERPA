from .PanoGenerator import PanoGenerator
import torch
import os
from PIL import Image
from lightning.pytorch.utilities import rank_zero_only
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel

# import sys
# sys.path.append('../..')
from utils.pano import CubemapNoise
import numpy as np
from external.Perspective_and_Equirectangular import mp2e
from external.py360convert import c2e


class PanoOnly(PanoGenerator):
    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        self.mv_base_model = MultiViewBaseModel(None, pano_unet, None, cn, self.hparams.unet_pad)

    def init_noise(self, bs, equi_h, equi_w, device, batch):
        pano_noise = []
        for idx in range(bs):
            # pers_noise = np.random.randn(8, equi_h*2, equi_h*2, 4)
            # # pers_noise = torch.randn(8, 1, 4, 512, 512)
            # normalized_noise = (255 * (pers_noise - pers_noise.min()) / (pers_noise.max() - pers_noise.min())).astype(np.uint8)
            # noise = mp2e(normalized_noise, [100] * 8, [0, 0, 90, 0, 90, 0, 180, 270], [0, 90, 0, 180, 180, 270, 0, 0], (equi_h, equi_w))
            # noise = (noise.astype(float) / 255) * (pers_noise.max() - pers_noise.min()) + pers_noise.min()
            # pano_noise.append(torch.tensor(noise, device=device).permute(2, 0, 1).unsqueeze(0).float())
            keys = ['F', 'R', 'B', 'L', 'U', 'D']
            cubemap = {k: np.random.randn(equi_h*2, equi_h*2, 4) for k in keys}
            equirect = c2e(cubemap, equi_h, equi_w, mode='nearest', cube_format='dict')
            pano_noise.append(torch.tensor(equirect, device=device).permute(2, 0, 1).unsqueeze(0).float())
        return torch.stack(pano_noise)
        # return torch.randn(bs, 1, 4, equi_h, equi_w, device=device)

    def embed_prompt(self, batch):
        pano_prompt = self.get_pano_prompt(batch)
        pano_prompt_embd = self.encode_text(pano_prompt)
        return pano_prompt_embd[:, None]

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        b = batch['pano'].shape[0]

        # pano_pad = self.pad_pano(batch['pano'])
        # pano_latent_pad = self.encode_image(pano_pad, self.vae)
        # pano_latent = self.unpad_pano(pano_latent_pad, latent=True)
        pano_latent = self.encode_image(batch['pano'], self.vae)

        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                          (b,), device=pano_latent.device).long()

        pano_prompt_embd = self.embed_prompt(batch)

        pano_noise = self.init_noise(b, *pano_latent.shape[-2:], device=device, batch=batch)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)

        denoise = self.mv_base_model.forward_pano(
            pano_noise_z, t, pano_prompt_embd, batch.get('pano_layout_cond'))

        loss = torch.nn.functional.mse_loss(denoise, pano_noise)
        # loss_mask = torch.ones_like(denoise, device=pano_latent.device)
        # loss_mask[:, :, :, :(loss_mask.shape[-2] // 8)] = 0.
        # loss_mask[:, :, :, -(loss_mask.shape[-2] // 8):] = 0.
        # loss *= loss_mask
        # loss = loss.mean()
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/loss_pano', loss)
        return loss

    @torch.no_grad()
    def forward_cls_free(self, pano_latent, timestep, pano_prompt_embd, layout_cond=None):
        pano_latent, timestep, layout_cond = self.gen_cls_free_guide_pair(pano_latent, timestep, layout_cond)

        # _, pano_noise_pred = self.mv_base_model(
        #     None, pano_latent, timestep, None, pano_prompt_embd, None,
        #     None, layout_cond)
        pano_noise_pred = self.mv_base_model.forward_pano(
            pano_latent, timestep, pano_prompt_embd, layout_cond)

        pano_noise_pred = self.combine_cls_free_guide_pred(pano_noise_pred)
        return pano_noise_pred


    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()
        device = self.device

        pano_latent = self.init_noise(bs, batch['height'].item()//8, batch['width'].item()//8, device=device, batch=batch)

        pano_prompt_embd = self.embed_prompt(batch)
        prompt_null = self.encode_text('')[:, None]
        pano_prompt_embd = torch.cat([prompt_null, pano_prompt_embd])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            pano_latent = self.rotate_latent(pano_latent)
            curr_rot += self.hparams.rot_diff

            if self.hparams.layout_cond:
                pano_layout_cond = self.rotate_latent(pano_layout_cond)
                pano_noise_pred = self.forward_cls_free(
                    pano_latent, t[None], pano_prompt_embd, pano_layout_cond)
            else:
                pano_noise_pred = self.forward_cls_free(
                    pano_latent, t[None], pano_prompt_embd)

            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent = self.rotate_latent(pano_latent, -curr_rot)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
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
