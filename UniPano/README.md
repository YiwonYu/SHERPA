# What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?

Jinhong Ni<sup>1</sup>, Chang-Bin Zhang<sup>2</sup>, Qiang Zhang<sup>3,4</sup>, Jing Zhang<sup>1</sup>

<sup>1</sup>Australian National University
<sup>2</sup>The University of Hong Kong
<sup>3</sup>Beijing Innovation Center of Humanoid Robotics
<sup>4</sup>Hong Kong University of Science and Technology (Guangzhou)

[![Paper](https://img.shields.io/badge/arXiv-2505.22129-b31b1b.svg)](https://arxiv.org/abs/2505.22129)
<a href="mailto: jinhong.ni@anu.edu.au">
        <img alt="emal" src="https://img.shields.io/badge/contact_me-email-yellow">
    </a>



## Updates
- [07/25] Our code is released.
- [06/25] Our paper is accepted to ICCV 2025.


## Key Findings

Our paper examines the key components that enable the adaptation of pre-trained Stable Diffusion for panorama generation. In particular, we summarize the two key findings of our paper:

<img width="800" alt="" src="assets/qkvo_isolation.png">

- The four attention matrices ($W_{\{q,k,v,o\}}$) behave differently when fine-tuned in isolation with LoRA. $W_q$ or $W_k$ fails to capture the spherical structure of the panoramas, whereas $W_v$ and $W_o$ succeed.

<img width="800" alt="" src="assets/qkvo_joint.png">

- Jointly fine-tuned LoRA weights associated with the four attention matrices have different functionalities. (a) All four LoRAs together generate panoramic images; (b) naturally, the four LoRAs trained on panoramas lose the ability to generate perspective images; (c) excluding $W_v$ and $W_o$ LoRAs recovers the ability to generate perspective images; (d) excluding $W_q$ and $W_k$ LoRAs preserves the fine-tuned model's ability to generate panorams.

For more details, please refer to [our paper](https://arxiv.org/abs/2505.22129).

## Environment Setup

We use Anaconda to manage the environment. You can create the environment by running the following command:

```bash
cd UniPano
bash setup_env.sh
```

We use [wandb](https://www.wandb.com/) to log and visualize the training process.

```bash
wandb login
```

## Data Preparation

We follow [PanFusion](https://github.com/chengzhag/PanFusion) and [MVDiffusion](https://github.com/Tangshitao/MVDiffusion) to download the [Matterport3D](https://niessner.github.io/Matterport/) skybox dataset. Please refer to their [Data Preparation Section](https://github.com/chengzhag/PanFusion?tab=readme-ov-file#data-preparation) to download and prepare the dataset.


## Training and Testing

For training UniPano with default settings, run the following command:
```bash
WANDB_NAME=unipano python main.py fit --data=Matterport3D --model=UniPano
```
Our training log can be found at [wandb](https://wandb.ai/mclean/unipano/runs/ud4lfpye).

Please follow [PanFusion](https://github.com/chengzhag/PanFusion?tab=readme-ov-file#faed) to download the FAED checkpoint. Replace `<WANDB_RUN_ID>` with the `wandb` run ID and run the following command for testing:
```bash
WANDB_RUN_ID=<WANDB_RUN_ID> python main.py test --data=Matterport3D --model=UniPano --ckpt_path=last
WANDB_RUN_ID=<WANDB_RUN_ID> python main.py test --data=Matterport3D --model=EvalPanoGen
```

## UniPano with Stable Diffusion 3
As mentioned in our paper, our uni-branch solution can be easily integrated into more advanced and memory-exhaustive diffusion models such as Stable Diffusion 3. We use a different codebase for Stable Diffusion 3. Please refer to `UniPano_SD3` folder for more details.

## Citation
```
@article{ni2025makes,
  title={What Makes for Text to 360-degree Panorama Generation with Stable Diffusion?},
  author={Ni, Jinhong and Zhang, Chang-Bin and Zhang, Qiang and Zhang, Jing},
  journal={arXiv preprint arXiv:2505.22129},
  year={2025}
}
```

## Acknowledgement

This repository is mainly developed based on [PanFusion](https://github.com/chengzhag/PanFusion). The codebase also benefits from [DiT-MoE](https://github.com/feizc/DiT-MoE) for MoE implementation.
