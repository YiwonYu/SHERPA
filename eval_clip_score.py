#!/usr/bin/env python3
"""
CLIP-Score Evaluation for Generated Panoramas
Computes CLIP-Score between generated images and their text prompts.
"""

import os
import glob
from pathlib import Path
from PIL import Image
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
from tqdm import tqdm
import json


def load_prompt(prompt_path):
    """Load prompt from text file."""
    with open(prompt_path, "r") as f:
        return f.read().strip()


def get_samples(log_dir, max_seeds=None, max_samples_per_seed=None):
    """
    Gather all generated panorama samples from log directory.

    Structure expected:
        log_dir/
            predict_seed_{seed}/
                000000/
                    pano.jpg
                    prompt.txt
                000001/
                    ...
    """
    samples = []

    # Find all seed directories
    seed_dirs = sorted(glob.glob(os.path.join(log_dir, "predict_seed_*")))
    if max_seeds:
        seed_dirs = seed_dirs[:max_seeds]

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)

        # Find all sample directories
        sample_dirs = sorted(glob.glob(os.path.join(seed_dir, "*")))
        if max_samples_per_seed:
            sample_dirs = sample_dirs[:max_samples_per_seed]

        for sample_dir in sample_dirs:
            pano_path = os.path.join(sample_dir, "pano.jpg")
            prompt_path = os.path.join(sample_dir, "prompt.txt")

            if os.path.exists(pano_path) and os.path.exists(prompt_path):
                samples.append(
                    {
                        "pano_path": pano_path,
                        "prompt_path": prompt_path,
                        "seed": seed_name,
                        "sample_id": os.path.basename(sample_dir),
                    }
                )

    return samples


def compute_clip_score(
    samples, model_name="openai/clip-vit-base-patch16", device=None, batch_size=32
):
    """Compute CLIP-Score for all samples."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading CLIP model: {model_name}")

    clip_metric = CLIPScore(model_name=model_name)
    clip_metric = clip_metric.to(device)
    clip_metric.eval()

    all_scores = []

    print(f"Computing CLIP-Score for {len(samples)} samples...")

    with torch.no_grad():
        for i in tqdm(range(0, len(samples), batch_size)):
            batch = samples[i : i + batch_size]

            images = []
            prompts = []

            for sample in batch:
                img = Image.open(sample["pano_path"]).convert("RGB")
                images.append(img)
                prompts.append(load_prompt(sample["prompt_path"]))

            # Compute CLIP-Score
            images_tensor = torch.stack([torch.tensor(img) for img in images])

            # CLIPScore expects images in (B, C, H, W) format
            # and will handle normalization internally
            try:
                score = clip_metric(images_tensor, prompts)
                all_scores.append(score.item())
            except Exception as e:
                print(f"Error processing batch {i}: {e}")
                # Try one by one
                for img, prompt in zip(images, prompts):
                    try:
                        score = clip_metric(
                            img.unsqueeze(0)
                            if isinstance(img, torch.Tensor)
                            else torch.tensor(img),
                            [prompt],
                        )
                        all_scores.append(score.item())
                    except:
                        pass

    return all_scores


def evaluate_log_dir(
    log_dir,
    output_json=None,
    max_seeds=None,
    max_samples_per_seed=None,
    model_name="openai/clip-vit-base-patch16",
):
    """Evaluate a single log directory."""
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {log_dir}")
    print(f"{'=' * 60}")

    samples = get_samples(log_dir, max_seeds, max_samples_per_seed)
    print(f"Found {len(samples)} samples")

    if len(samples) == 0:
        print("No samples found!")
        return None

    scores = compute_clip_score(samples, model_name=model_name)

    if scores:
        import numpy as np

        mean_score = np.mean(scores)
        std_score = np.std(scores)

        print(f"\nResults:")
        print(f"  Mean CLIP-Score: {mean_score:.4f}")
        print(f"  Std CLIP-Score:  {std_score:.4f}")
        print(f"  Min: {min(scores):.4f}, Max: {max(scores):.4f}")

        result = {
            "log_dir": log_dir,
            "num_samples": len(scores),
            "mean_clip_score": mean_score,
            "std_clip_score": std_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "model": model_name,
        }

        if output_json:
            with open(output_json, "w") as f:
                json.dump(result, f, indent=2)
            print(f"Results saved to: {output_json}")

        return result

    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLIP-Score Evaluation for Panoramas")
    parser.add_argument(
        "--unipano",
        type=str,
        default="/workspace/UniPano/logs/ud4lfpye",
        help="UniPano log directory",
    )
    parser.add_argument(
        "--panfusion",
        type=str,
        default="/workspace/PanFusion/logs/4142dlo4",
        help="PanFusion log directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Maximum number of seeds to evaluate",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per seed to evaluate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP model name",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    results = []

    # Evaluate UniPano
    if os.path.exists(args.unipano):
        output_json = os.path.join(args.output_dir, "unipano_clip_score.json")
        result = evaluate_log_dir(
            args.unipano,
            output_json=output_json,
            max_seeds=args.max_seeds,
            max_samples_per_seed=args.max_samples,
            model_name=args.model,
        )
        if result:
            results.append(result)

    # Evaluate PanFusion
    if os.path.exists(args.panfusion):
        output_json = os.path.join(args.output_dir, "panfusion_clip_score.json")
        result = evaluate_log_dir(
            args.panfusion,
            output_json=output_json,
            max_seeds=args.max_seeds,
            max_samples_per_seed=args.max_samples,
            model_name=args.model,
        )
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        for r in results:
            print(
                f"{os.path.basename(r['log_dir'])}: {r['mean_clip_score']:.4f} ± {r['std_clip_score']:.4f}"
            )

    return results


if __name__ == "__main__":
    main()
