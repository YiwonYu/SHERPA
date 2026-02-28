from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from glob import glob
from tqdm import tqdm
import os
import random


pers_dir = "/disk/work/jhni/generated_pers"
data_dir = "/home/jhni/PanFusion/data/Matterport3D/mp3d_skybox"
n_imgs = 20000

if __name__ == "__main__":
    if not os.path.exists(pers_dir):
        os.mkdir(pers_dir)
    # if len(os.listdir(pers_dir)) == 0:
        # print(f"{pers_dir} is empty, generating data...")
    prompts = glob(os.path.join(data_dir, '*', 'blip3', '*.txt'))
    random.shuffle(prompts)
    prompts = prompts[:n_imgs]
    model_id = 'stabilityai/stable-diffusion-2-base'

    # Use the Euler scheduler here instead
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    for idx, prompt_path in enumerate(tqdm(prompts)):
        if os.path.exists(os.path.join(pers_dir, f"{idx}.png")):
            continue
        with open(prompt_path, "r") as f:
            prompt = f.readlines()[0]
        image = pipe(prompt).images[0]
            
        image.save(os.path.join(pers_dir, f"{idx}.png"))
        with open(os.path.join(pers_dir, f"{idx}.txt"), "w") as f:
            f.write(prompt)