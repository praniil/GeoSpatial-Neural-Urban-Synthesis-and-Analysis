# image_generation

Utilities and scripts to prepare ControlNet training data and run generation.

Build ControlNet training metadata from your dataset masks:

```bash
python -m src.image_generation.prepare_controlnet_dataset --dataset_dir dataset --output_jsonl dataset/metadata.jsonl
```

This writes `dataset/metadata.jsonl` with fields `file_name`, `conditioning_image_file_name`, and `text` (prompt).

Train ControlNet (wrapper around diffusers training script):

```bash
python -m src.image_generation.train_controlnet --dataset_dir dataset --output_dir controlnet-satellite-output --diffusers_dir diffusers
```

Generate with ControlNet:

```bash
python -m src.image_generation.generate_controlnet \
	--image dataset/images/example.png \
	--mask dataset/masks/example.png \
	--controlnet_path controlnet-satellite-output/checkpoint-XXXX \
	--custom_prompt "Redesigned as a modern smart city with green corridors"
```

SegFormer inference + prompt JSON output (optional):

```bash
python -m src.image_generation.inference_pipeline --dataset_dir dataset --output_dir outputs --use_segformer
```

Options:
- `--hf_repo`: HF repo id or local path for the SegFormer checkpoint (default uses your fine-tuned model name)
- `--custom_prompt`: optional prompt to append or override
- `--strategy`: `append` or `override`
