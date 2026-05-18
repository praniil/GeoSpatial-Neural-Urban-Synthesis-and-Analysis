import argparse
import json
import os
from pathlib import Path

from PIL import Image

from .area_calculator import CLASSES
from .prompt_builder import generate_base_prompt, build_final_prompt
from .utils import COLOR_TO_INDEX


def area_stats_from_mask(mask_img: Image.Image) -> dict:
    """Compute area stats directly from mask colors (fast, avoids per-pixel loops)."""
    w, h = mask_img.size
    total = w * h
    maxcolors = total
    colors = mask_img.getcolors(maxcolors=maxcolors)
    if colors is None:
        raise ValueError("Mask has too many colors; expected a small fixed palette.")

    counts_by_idx = {idx: 0 for idx in CLASSES.keys()}
    for count, color in colors:
        key = (int(color[0]), int(color[1]), int(color[2]))
        if key not in COLOR_TO_INDEX:
            raise ValueError(f"Unknown mask color: {key}")
        counts_by_idx[COLOR_TO_INDEX[key]] += int(count)

    stats = {}
    for idx, name in CLASSES.items():
        count = counts_by_idx.get(idx, 0)
        pct = round(float(count) / float(total) * 100.0, 2)
        stats[idx] = {"name": name, "percentage": pct, "pixel_count": count}

    stats = dict(sorted(stats.items(), key=lambda x: x[1]["percentage"], reverse=True))
    return stats


def build_metadata(dataset_dir: str, output_jsonl: str, custom_prompt: str = "", strategy: str = "append"):
    images_dir = os.path.join(dataset_dir, "images")
    masks_dir = os.path.join(dataset_dir, "masks")

    image_paths = sorted([p for p in Path(images_dir).glob("**/*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    entries = []
    for img_path in image_paths:
        name = img_path.stem
        mask_path = Path(masks_dir) / f"{name}.png"
        if not mask_path.exists():
            print(f"Skipping {name}: missing mask {mask_path}")
            continue

        mask_img = Image.open(mask_path).convert("RGB")
        area_stats = area_stats_from_mask(mask_img)
        base_prompt = generate_base_prompt(area_stats)
        final_prompt = build_final_prompt(base_prompt, custom_prompt, strategy)

        # Use relative paths so datasets/imagefolder can resolve files under --train_data_dir.
        rel_image = os.path.relpath(img_path, dataset_dir)
        rel_mask = os.path.relpath(mask_path, dataset_dir)
        entries.append({
            "file_name": rel_image,
            "conditioning_image_file_name": rel_mask,
            "text": final_prompt,
        })

    if not entries:
        raise RuntimeError("No image/mask pairs found to write JSONL.")

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(entries)} entries to {output_jsonl}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="dataset", help="Dataset dir with images/ and masks/")
    p.add_argument("--output_jsonl", default="dataset/metadata.jsonl", help="Output JSONL path")
    p.add_argument("--custom_prompt", default="", help="Optional prompt to append/override")
    p.add_argument("--strategy", choices=["append", "override"], default="append")
    args = p.parse_args()

    build_metadata(args.dataset_dir, args.output_jsonl, args.custom_prompt, args.strategy)


if __name__ == "__main__":
    main()
