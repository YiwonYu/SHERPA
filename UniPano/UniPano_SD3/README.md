# UniPano with Stable Diffusion 3

## Environment Setup
Ideally, the codebase should work on any version later than `diffusers-0.32.0` while we only have tested it on its development version. In case there is a problem, please install `diffusers-0.32.0.dev0` from source by running the following command
```bash
pip install -e .
```

## Training and Testing
Replace `<DATA_DIR>` with the dataset directory and run the following for both training and testing. You may optionally disable testing by running without `--evaluate` flag.
```bash
accelerate launch train_pano_lora_sd3.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-3-medium-diffusers" \
    --data_dir=<DATA_DIR> \
    --output_dir="logs/pano" \
    --mixed_precision="no" \
    --resolution=1024 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --learning_rate=2e-4 \
    --report_to="wandb" \
    --lr_scheduler="cosine" \
    --lr_num_cycles="0.5" \
    --lr_warmup_steps=500 \
    --rank="4" \
    --lora_alpha="4" \
    --num_train_epochs=1 \
    --validation_epochs=1 \
    --checkpointing_steps=5000 \
    --seed="0" \
    --evaluate \
    --is_unipano=True
```

## Note and TODOs

Cicular padding cannot be trivially implemented for DiT. As a result, current implementation may result in notable artifacts around the left and right edges. Potential solution to this is applying latent rotation during training, but this is not expected to work as good as circular padding. This problem remains open to date and we would be grateful to engage the community to resolve this.

## Acknowledgement

This codebase is implemented using `diffusers` and is largely based on `examples/dreambooth/train_dreambooth_lora_sd3.py`