import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import os

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HF MODEL REPO
HF_REPO = "Pranilllllll/segformer-satellite-segementation"

# LOAD MODEL + PROCESSOR FROM HUGGING FACE
model = SegformerForSemanticSegmentation.from_pretrained(HF_REPO).to(device)
processor = SegformerFeatureExtractor.from_pretrained(HF_REPO)
model.eval()

# loop to load the images
controlnet_image_dir = "./dataset/controlnet-images"
controlnet_mask_dir = "./dataset/controlnet-masks"
os.makedirs(controlnet_mask_dir, exist_ok=True)


controlnet_images = os.listdir(controlnet_image_dir)
# print(type(controlnet_images))

for controlnet_image in controlnet_images:
    # load image
    image_path = os.path.join(controlnet_image_dir, controlnet_image)
    image = Image.open(image_path).convert("RGB")
    # print(image_path)

    # PREPROCESS
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    
    # INFERENCE
    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, 7, H, W]
    
    # PREDICT MASK
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    pred_mask = Image.fromarray(pred_mask.astype(np.uint8)).resize(image.size, Image.NEAREST)
    pred_mask = np.array(pred_mask)
    
    # COLOR MAP (authored in BGR, convert to RGB for matplotlib/PIL)
    colors = np.array([
        [0,   0,   0],     # Background -> black
        [128, 0,   0],     # Residential -> maroon (dark red)
        [0,   128, 0],     # Road -> green
        [0,   0,   128],   # River -> navy (dark blue)
        [0,   128, 128],   # Forest -> teal (blue-green)
        [128, 128, 0],     # Unused land -> olive
        [128, 0,   128],   # Agriculture -> purple (magenta-like)
    ], dtype=np.uint8)
    colors = colors[:, ::-1]
    
    # CREATE SEGMENTATION IMAGE
    seg_image = colors[pred_mask]
    
    # SAVE RESULT
    output_name = os.path.splitext(controlnet_image)[0] + ".png"
    output_path = os.path.join(controlnet_mask_dir, output_name)
    Image.fromarray(seg_image).save(output_path)
    
    print(f"Segmentation saved at {output_path}")
