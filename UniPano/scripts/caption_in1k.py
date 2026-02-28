import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import os
import argparse
from tqdm import tqdm

from huggingface_hub import login
HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

model_id = "meta-llama/Llama-3.2-11B-Vision"

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

def parse_args():
    parser = argparse.ArgumentParser(description="Caption ImageNet1K Images")
    parser.add_argument("--data_path", default=".", type=str)
    parser.add_argument
    return parser.parse_args()


def main():
    args = parse_args()

    idx_list = [int(filename[:-4]) for filename in os.listdir(args.data_path) if filename.endswith('png')]
    for idx in tqdm(idx_list, desc="Captioning"):
        image = Image.open(os.path.join(args.data_path, f"{idx}.png"))

        prompt = "<|image|><|begin_of_text|>a photo of "
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, max_new_tokens=30)
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:]
        caption = processor.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        with open(os.path.join(args.data_path, f"{idx}_mllama.txt"), "w") as f:
            f.write(caption)


if __name__ == '__main__':
    main()
