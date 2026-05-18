import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from .utils import ensure_dir, mask_to_color_image, rgb_mask_to_index
from .area_calculator import calculate_area_percentages
from .prompt_builder import generate_base_prompt, build_final_prompt


DEFAULT_HF_REPO = "Pranilllllll/segformer-satellite-segementation"


def load_segformer(repo: str, device: str):
    from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
    model = SegformerForSemanticSegmentation.from_pretrained(repo).to(device)
    processor = SegformerFeatureExtractor.from_pretrained(repo)
    model.eval()
    return model, processor


def segment_image(image_path: str, model, processor, device: str):
    import torch
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    return img, pred_mask


def process_dataset(dataset_dir: str, output_dir: str, hf_repo: str, device: str,
                    custom_prompt: str = "", strategy: str = "append",
                    use_dataset_masks: bool = True):
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "masks"))
    ensure_dir(os.path.join(output_dir, "prompts"))

    model = processor = None
    if not use_dataset_masks:
        model, processor = load_segformer(hf_repo, device)

    image_paths = sorted([str(p) for p in Path(images_dir).glob("**/*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not image_paths:
        print(f"No images found in {images_dir}")
        return

    for img_path in image_paths:
        name = Path(img_path).stem
        print(f"Processing {name}...")
        if use_dataset_masks:
            mask_path = os.path.join(masks_dir, f"{name}.png")
            if not os.path.exists(mask_path):
                print(f"Missing mask for {name}: {mask_path}")
                continue
            mask_rgb = Image.open(mask_path).convert("RGB")
            pred_mask = rgb_mask_to_index(np.array(mask_rgb))
            mask_out = mask_path
        else:
            _, pred_mask = segment_image(img_path, model, processor, device)
            color_mask = mask_to_color_image(pred_mask)
            mask_out = os.path.join(output_dir, "masks", f"{name}_mask.png")
            color_mask.save(mask_out)

        # area stats and prompt
        area_stats = calculate_area_percentages(pred_mask)
        base_prompt = generate_base_prompt(area_stats)
        final_prompt = build_final_prompt(base_prompt, custom_prompt, strategy)

        # save metadata
        meta = {
            "image": img_path,
            "mask": mask_out,
            "base_prompt": base_prompt,
            "final_prompt": final_prompt,
            "area_stats": area_stats,
        }
        with open(os.path.join(output_dir, "prompts", f"{name}.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print("Saved:")
        print(" -", mask_out)
        print(" -", os.path.join(output_dir, "prompts", f"{name}.json"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="dataset", help="Path to dataset folder (images/, masks/)")
    p.add_argument("--output_dir", default="outputs", help="Where to save masks and prompts")
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO, help="HF repo or local path for SegFormer checkpoint")
    p.add_argument("--device", default="cuda" if _torch_available_cuda() else "cpu")
    p.add_argument("--custom_prompt", default="", help="Optional custom prompt to append or override")
    p.add_argument("--strategy", choices=["append", "override"], default="append")
    p.add_argument("--use_segformer", action="store_true", help="Run SegFormer instead of using dataset/masks")
    args = p.parse_args()

    process_dataset(
        args.dataset_dir,
        args.output_dir,
        args.hf_repo,
        args.device,
        args.custom_prompt,
        args.strategy,
        not args.use_segformer,
    )


def _torch_available_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
