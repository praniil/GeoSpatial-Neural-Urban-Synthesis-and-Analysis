from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
OUTPUT_EXTENSION = ".png"


def load_image_paths(source_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "_aug" not in path.stem
    )


def save_variant(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def add_variants(image: Image.Image, stem: str) -> Iterable[tuple[str, Image.Image]]:
    yield "hflip", ImageOps.mirror(image)
    yield "vflip", ImageOps.flip(image)
    yield "rot90", image.rotate(90, expand=True)
    yield "rot270", image.rotate(270, expand=True)


def augment_dataset(source_dir: Path, output_dir: Path) -> int:
    image_paths = load_image_paths(source_dir)
    created = 0

    for index, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            base_image = image.copy()

        for variant_name, variant_image in add_variants(base_image, image_path.stem):
            output_path = output_dir / f"{image_path.stem}_aug_{variant_name}{OUTPUT_EXTENSION}"
            if output_path.exists():
                continue
            save_variant(variant_image, output_path)
            created += 1

        print(f"Processed {index}/{len(image_paths)} source images; created {created} augmented files", flush=True)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment control-net dataset images in place and save variants alongside the originals."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("dataset/controlnet-images"),
        help="Directory containing the source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/controlnet-images"),
        help="Directory where augmented images will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = augment_dataset(args.source_dir, args.output_dir)
    print(f"Created {created} augmented images in {args.output_dir}")


if __name__ == "__main__":
    main()