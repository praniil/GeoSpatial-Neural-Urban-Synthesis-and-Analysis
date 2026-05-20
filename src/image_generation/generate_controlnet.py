import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionPipeline,
    UniPCMultistepScheduler,
)

from .area_calculator import calculate_area_percentages
from .prompt_builder import generate_base_prompt, build_final_prompt
from .utils import mask_to_color_image, rgb_mask_to_index


DEFAULT_HF_REPO = "Pranilllllll/segformer-satellite-segementation"
DEFAULT_BASE_MODEL = "runwayml/stable-diffusion-v1-5"


def load_segformer(repo: str, device: str):
    from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
    model = SegformerForSemanticSegmentation.from_pretrained(repo).to(device)
    processor = SegformerFeatureExtractor.from_pretrained(repo)
    model.eval()
    return model, processor


def segment_image(image_path: str, model, processor, device: str):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["pixel_values"].to(device)).logits
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    return img, pred_mask


def load_mask(mask_path: str) -> np.ndarray:
    mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
    return rgb_mask_to_index(mask_rgb)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input satellite image path")
    p.add_argument("--mask", default="", help="Optional RGB mask path; if empty, SegFormer will run")
    p.add_argument("--controlnet_path", required=True, help="Path to ControlNet checkpoint")
    p.add_argument("--base_model", default=DEFAULT_BASE_MODEL, help="Base SD model when loading .safetensors")
    p.add_argument("--hf_repo", default=DEFAULT_HF_REPO, help="SegFormer HF repo or local path")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--custom_prompt", default="")
    p.add_argument("--strategy", choices=["append", "override"], default="append")
    p.add_argument("--negative_prompt", default="blurry, low quality, distorted, cartoon, painting")
    p.add_argument("--num_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)
    p.add_argument("--controlnet_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mask:
        pred_mask = load_mask(args.mask)
        original_image = Image.open(args.image).convert("RGB")
    else:
        model, processor = load_segformer(args.hf_repo, device)
        original_image, pred_mask = segment_image(args.image, model, processor, device)

    area_stats = calculate_area_percentages(pred_mask)
    base_prompt = generate_base_prompt(area_stats)
    final_prompt = build_final_prompt(base_prompt, args.custom_prompt, args.strategy)

    control_image = mask_to_color_image(pred_mask, size=(512, 512))

    controlnet_path = Path(args.controlnet_path)
    dtype = torch.float16 if device == "cuda" else torch.float32
    if controlnet_path.is_file() and controlnet_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        base_pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            safety_checker=None,
        )
        controlnet = ControlNetModel.from_unet(base_pipe.unet, load_weights_from_unet=False)
        controlnet.load_state_dict(load_file(str(controlnet_path)))
        controlnet = controlnet.to(dtype)
        del base_pipe
    else:
        controlnet = ControlNetModel.from_pretrained(
            args.controlnet_path,
            torch_dtype=dtype,
        )
        controlnet = controlnet.to(dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    output = pipe(
        prompt=final_prompt,
        negative_prompt=args.negative_prompt,
        image=control_image,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    )

    output_image = output.images[0]
    output_image.save(os.path.join(args.output_dir, "output_generated.png"))
    control_image.save(os.path.join(args.output_dir, "output_mask.png"))
    original_image.save(os.path.join(args.output_dir, "output_original.png"))

    print("Base prompt:", base_prompt)
    print("Final prompt:", final_prompt)
    print("Saved outputs to", args.output_dir)


if __name__ == "__main__":
    main()
