from .PanoGenerator import PanoGenerator
from ..modules.utils import tensor_to_image
from .MVGenModel import MultiViewBaseModel
import torch
import os
from PIL import Image
from external.Perspective_and_Equirectangular import e2p
from einops import rearrange
from lightning.pytorch.utilities import rank_zero_only

import sys
sys.path.append('../..')
from external.Perspective_and_Equirectangular import mp2e_torch
from utils.pano import icosahedron_sample_camera
from external.py360convert import c2e
import numpy as np


class UniPano(PanoGenerator):
    def __init__(
            self,
            use_pers_prompt: bool = True,
            use_pano_prompt: bool = True,
            copy_pano_prompt: bool = True,
            pano_ratio: float = None,
            use_pe: bool = False,
            use_adapter: bool = False,
            **kwargs
            ):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def instantiate_model(self):
        pano_unet, cn = self.load_pano()
        # unet, pers_cn = self.load_pers()
        self.mv_base_model = MultiViewBaseModel(None, pano_unet, None, cn, self.hparams.unet_pad, self.hparams.use_pe, self.hparams.use_adapter)
        # if not self.hparams.layout_cond:
        if hasattr(self.mv_base_model, "trainable_parameters"):
            self.trainable_params.extend(self.mv_base_model.trainable_parameters)

    def init_noise(self, bs, equi_h, equi_w, pers_h, pers_w, cameras, device):
        # cameras = {k: rearrange(v, 'b m ... -> (b m) ...') for k, v in cameras.items()}
        pano_noise = torch.randn(
            bs, 1, 4, equi_h, equi_w, device=device)
        # TODO and NOTE, current noise sampling in panorama space is very inefficient!
        # pano_noise = []
        # for idx in range(bs):
        #     # pers_noise = np.random.randn(8, equi_h*2, equi_h*2, 4)
        #     # # pers_noise = torch.randn(8, 1, 4, 512, 512)
        #     # theta, phi = icosahedron_sample_camera()
        #     # theta, phi = np.rad2deg(theta), np.rad2deg(phi)

        #     # fov = np.full_like(theta, 90, dtype=int)

        #     # equi = mp2e_torch(torch.randn(len(theta), 1, 4, equi_h, equi_h, device=device).float(), fov, theta, phi, (equi_h, equi_w), mode="nearest")
        #     # pano_noise.append(equi.to(device).float())

        #     keys = ['F', 'R', 'B', 'L', 'U', 'D']
        #     cubemap = {k: np.random.randn(equi_h, equi_h, 4) for k in keys}
        #     equirect = c2e(cubemap, equi_h, equi_w, mode='nearest', cube_format='dict')
        #     pano_noise.append(torch.tensor(equirect, device=device).permute(2, 0, 1).unsqueeze(0).float())
        # pano_noise = torch.stack(pano_noise)
        # pano_noises = pano_noise.expand(-1, 1, -1, -1, -1)
        # pano_noises = rearrange(pano_noises, 'b m c h w -> (b m) c h w')
        # noise = e2p(
        #     pano_noises,
        #     cameras['FoV'], cameras['theta'], cameras['phi'],
        #     (pers_h, pers_w), mode='nearest')
        # noise = rearrange(noise, '(b m) c h w -> b m c h w', b=bs, m=len(cameras['FoV']))
        noise = torch.randn(bs, 1, 4, pers_h, pers_w, device=device).float()
        return pano_noise, noise

    def embed_prompt(self, batch, num_cameras):
        if self.hparams.use_pers_prompt:
            pers_prompt = self.get_pers_prompt(batch)
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = rearrange(pers_prompt_embd, '(b m) l c -> b m l c', m=num_cameras)
        else:
            pers_prompt = ''
            pers_prompt_embd = self.encode_text(pers_prompt)
            pers_prompt_embd = pers_prompt_embd[:, None].repeat(1, num_cameras, 1, 1)

        if self.hparams.use_pano_prompt:
            pano_prompt = self.get_pano_prompt(batch)
        else:
            pano_prompt = ''
        pano_prompt_embd = self.encode_text(pano_prompt)
        pano_prompt_embd = pano_prompt_embd[:, None]

        return pers_prompt_embd, pano_prompt_embd

    def training_step(self, batch, batch_idx):
        device = batch['images'].device
        b = batch['pano'].shape[0]
        pano_latent = self.encode_image(batch['pano'], self.vae)
        t = torch.randint(0, self.scheduler.config.num_train_timesteps,
                    (b,), device=pano_latent.device).long()

        _, pano_prompt_embd = self.embed_prompt(batch, 1)

        pano_noise, _ = self.init_noise(
            b, *pano_latent.shape[-2:], 64, 64, None, device)
        pano_noise_z = self.scheduler.add_noise(pano_latent, pano_noise, t)

        denoise = self.mv_base_model.forward_pano(
            pano_noise_z, t, pano_prompt_embd, batch.get('pano_layout_cond'))

        loss = torch.nn.functional.mse_loss(denoise, pano_noise)
        self.log('train/loss', loss, prog_bar=True)
        return loss

    @torch.no_grad()
    def forward_cls_free(self, latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, batch, pano_layout_cond=None):
        latents, pano_latent, timestep, cameras, images_layout_cond, pano_layout_cond = self.gen_cls_free_guide_pair(
            latents, pano_latent, timestep, batch['cameras'],
            batch.get('images_layout_cond'), pano_layout_cond)

        noise_pred, pano_noise_pred = self.mv_base_model(
            latents, pano_latent, timestep, prompt_embd, pano_prompt_embd, cameras,
            images_layout_cond, pano_layout_cond)

        noise_pred, pano_noise_pred = self.combine_cls_free_guide_pred(noise_pred, pano_noise_pred)

        return noise_pred, pano_noise_pred

    def rotate_latent(self, pano_latent, cameras, degree=None):
        if degree is None:
            degree = self.hparams.rot_diff
        if degree % 360 == 0:
            return pano_latent, cameras

        pano_latent = super().rotate_latent(pano_latent, degree)
        cameras = cameras.copy()
        cameras['theta'] = (cameras['theta'] + degree) % 360
        return pano_latent, cameras

    @torch.no_grad()
    def inference(self, batch):
        bs, m = batch['cameras']['height'].shape[:2]
        m = 1
        h, w = batch['cameras']['height'][0, 0].item(), batch['cameras']['width'][0, 0].item()
        device = self.device

        pano_latent, latents = self.init_noise(
            bs, batch['height'].item()//8, batch['width'].item()//8, h//8, h//8, batch['cameras'], device)

        pers_prompt_embd, pano_prompt_embd = self.embed_prompt(batch, m)
        prompt_null = self.encode_text('')[:, None]
        pano_prompt_embd = torch.cat([prompt_null, pano_prompt_embd])
        prompt_null = prompt_null.repeat(1, m, 1, 1)
        pers_prompt_embd = torch.cat([prompt_null, pers_prompt_embd])

        self.scheduler.set_timesteps(self.hparams.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        pano_layout_cond = batch.get('pano_layout_cond')

        curr_rot = 0
        for i, t in enumerate(timesteps):
            timestep = torch.cat([t[None, None]]*m, dim=1)

            pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'])
            curr_rot += self.hparams.rot_diff

            if self.hparams.layout_cond:
                pano_layout_cond = super().rotate_latent(pano_layout_cond)
            else:
                pano_layout_cond = None
            noise_pred, pano_noise_pred = self.forward_cls_free(
                latents, pano_latent, timestep, pers_prompt_embd, pano_prompt_embd, batch, pano_layout_cond)

            latents = self.scheduler.step(
                noise_pred, t, latents).prev_sample
            pano_latent = self.scheduler.step(
                pano_noise_pred, t, pano_latent).prev_sample

        pano_latent, batch['cameras'] = self.rotate_latent(pano_latent, batch['cameras'], -curr_rot)

        images_pred = self.decode_latent(latents, self.vae)
        images_pred = tensor_to_image(images_pred)

        pano_latent_pad = self.pad_pano(pano_latent, latent=True)
        pano_pred_pad = self.decode_latent(pano_latent_pad, self.vae)
        pano_pred = self.unpad_pano(pano_pred_pad)
        pano_pred = tensor_to_image(pano_pred)

        # # test encoded pano latent
        # img1 = self.decode_latent(pano_latent, self.vae).squeeze()
        # img1 = np.roll(img1, img1.shape[0]//2, axis=0)
        # img1 = np.roll(img1, img1.shape[1]//2, axis=1)
        # img2 = pano_pred.squeeze()
        # img2 = np.roll(img2, img2.shape[0]//2, axis=0)
        # img2 = np.roll(img2, img2.shape[1]//2, axis=1)

        return images_pred, pano_pred

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images_pred, pano_pred = self.inference(batch)
        self.log_val_image(images_pred, batch['images'], pano_pred, batch.get('pano'), batch.get('pano_prompt'),
                           batch.get('images_layout_cond'), batch.get('pano_layout_cond'))

    def inference_and_save(self, batch, output_dir, ext='png'):
        prompt_path = os.path.join(output_dir, 'prompt.txt')
        if os.path.exists(prompt_path):
            return

        _, pano_pred = self.inference(batch)

        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"pano.{ext}")
        im = Image.fromarray(pano_pred[0, 0])
        im.save(path)

        with open(prompt_path, 'w') as f:
            f.write(batch['pano_prompt'][0]+'\n')

    @torch.no_grad()
    @rank_zero_only
    def log_val_image(self, images_pred, images, pano_pred, pano, pano_prompt,
                      images_layout_cond=None, pano_layout_cond=None):
        log_dict = {f"val/{k}_pred": v for k, v in self.temp_wandb_images(
            images_pred, pano_pred, None, pano_prompt).items()}
        log_dict.update({f"val/{k}_gt": v for k, v in self.temp_wandb_images(
            images, pano, None, pano_prompt).items()})
        if images_layout_cond is not None and pano_layout_cond is not None:
            log_dict.update({f"val/{k}_layout_cond": v for k, v in self.temp_wandb_images(
                images_layout_cond, pano_layout_cond, None, pano_prompt).items()})
        self.logger.experiment.log(log_dict)

    def temp_wandb_images(self, images, pano, prompt=None, pano_prompt=None):
        log_dict = {}
        pers = []
        for m_i in range(images.shape[1]):
            pers.append(self.temp_wandb_image(
                images[0, m_i], prompt[m_i][0] if prompt else None))
        log_dict['pers'] = pers
        
        if pano is not None:
            log_dict['pano'] = self.temp_wandb_image(
                pano[0, 0], pano_prompt[0] if pano_prompt else None)
        return log_dict
