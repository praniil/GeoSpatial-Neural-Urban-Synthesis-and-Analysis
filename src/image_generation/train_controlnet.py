import argparse
import os
import subprocess
import sys
from pathlib import Path


def build_command(args):
    train_script = Path(args.diffusers_dir) / "examples" / "controlnet" / "train_controlnet.py"
    if not train_script.exists():
        raise FileNotFoundError(
            f"train_controlnet.py not found at {train_script}. Clone diffusers and set --diffusers_dir."
        )

    cmd = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    if args.cpu:
        cmd.append("--cpu")
    cmd += [
        str(train_script),
        f"--pretrained_model_name_or_path={args.pretrained_model}",
        f"--output_dir={args.output_dir}",
        f"--train_data_dir={args.dataset_dir}",
        "--caption_column=text",
        "--image_column=image",
        "--conditioning_image_column=conditioning_image",
        f"--resolution={args.resolution}",
        f"--learning_rate={args.learning_rate}",
        f"--train_batch_size={args.train_batch_size}",
        f"--num_train_epochs={args.num_train_epochs}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
        f"--mixed_precision={args.mixed_precision}",
        f"--checkpointing_steps={args.checkpointing_steps}",
        f"--validation_steps={args.validation_steps}",
        f"--report_to={args.report_to}",
    ]

    if args.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if args.use_8bit_adam:
        cmd.append("--use_8bit_adam")
    if args.enable_xformers:
        cmd.append("--enable_xformers_memory_efficient_attention")
    if args.allow_tf32:
        cmd.append("--allow_tf32")
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", default="dataset", help="Dataset dir with images/, masks/, metadata.jsonl")
    p.add_argument("--output_dir", default="controlnet-satellite-output")
    p.add_argument("--pretrained_model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--diffusers_dir", default="diffusers", help="Path to diffusers repo")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=10)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--mixed_precision", default="fp16")
    p.add_argument("--checkpointing_steps", type=int, default=1000)
    p.add_argument("--validation_steps", type=int, default=500)
    p.add_argument("--report_to", default="tensorboard")
    p.add_argument("--cpu", action="store_true", help="Force CPU training (no CUDA)")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--enable_xformers", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--dry_run", action="store_true", help="Print command without executing")
    args = p.parse_args()

    cmd = build_command(args)
    print(" ".join(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
