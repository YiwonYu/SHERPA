import cv2
import numpy as np
from .p2e import p2e
import torch


def mp2e(p_imgs, fov_degs, u_degs, v_degs, out_hw, mode=None):
    merge_image = np.zeros((*out_hw, p_imgs[0].shape[2]))
    merge_mask = np.zeros((*out_hw, p_imgs[0].shape[2]))
    for p_img, fov_deg, u_deg, v_deg in zip(p_imgs, fov_degs, u_degs, v_degs):
        img, mask = p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode)
        mask = mask.astype(np.float32)
        img = img.astype(np.float32)

        weight_mask = np.zeros((p_img.shape[0], p_img.shape[1], p_img.shape[2]))
        w = p_img.shape[1]
        weight_mask[:, 0:w//2, :] = np.linspace(0, 1, w//2)[..., None]
        weight_mask[:, w//2:, :] = np.linspace(1, 0, w//2)[..., None]
        weight_mask, _ = p2e(weight_mask, fov_deg, u_deg, v_deg, out_hw, mode)
        blur = cv2.blur(mask, (5, 5))
        blur = blur * mask
        mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
        merge_image += img * weight_mask
        merge_mask += weight_mask

    merge_image[merge_mask == 0] = 255.
    merge_mask = np.where(merge_mask == 0, 1, merge_mask)
    merge_image = (np.divide(merge_image, merge_mask)).astype(np.uint8)
    return merge_image

def mp2e_torch(p_imgs, fov_degs, u_degs, v_degs, out_hw, mode=None):
    merge_image = torch.zeros((1, p_imgs.shape[2], *out_hw)).to(p_imgs.device).to(p_imgs.dtype)
    merge_mask = torch.zeros((1, p_imgs.shape[2], *out_hw)).to(p_imgs.device).to(p_imgs.dtype)
    for p_img, fov_deg, u_deg, v_deg in zip(p_imgs, fov_degs, u_degs, v_degs):
        img, mask = p2e(p_img, fov_deg, u_deg, v_deg, out_hw, mode)
        mask = mask.float()
        img = img.float()

        weight_mask = torch.zeros(*p_img.shape).to(p_imgs.device).to(p_imgs.dtype) #torch.zeros((p_img.shape[0], p_img.shape[1], p_img.shape[2]))
        w = p_img.shape[-1]
        weight_mask[:, :, :, 0:w//2] = torch.linspace(0, 1, w//2).to(p_imgs.device).to(p_imgs.dtype)
        weight_mask[:, :, :, w//2:] = torch.linspace(1, 0, w//2).to(p_imgs.device).to(p_imgs.dtype)
        weight_mask, _ = p2e(weight_mask, fov_deg, u_deg, v_deg, out_hw, mode)
        # blur = cv2.blur(mask.numpy(), (5, 5))
        # blur = torch.tensor(blur) * mask
        # mask = (blur == 1) * blur + (blur != 1) * blur * 0.05
        merge_image += img * weight_mask
        merge_mask += weight_mask

    merge_image[merge_mask == 0] = 0.
    merge_mask = torch.where(merge_mask == 0, 1, merge_mask)
    merge_image = torch.divide(merge_image, merge_mask)
    return merge_image
