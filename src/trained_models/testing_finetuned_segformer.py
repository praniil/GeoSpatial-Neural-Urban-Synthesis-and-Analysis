import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor

# DEVICE
device = "cuda" if torch.cuda.is_available() else "cpu"

# HF MODEL REPO
HF_REPO = "Pranilllllll/segformer-satellite-segementation"

# LOAD MODEL + PROCESSOR FROM HUGGING FACE
model = SegformerForSemanticSegmentation.from_pretrained(HF_REPO).to(device)
processor = SegformerFeatureExtractor.from_pretrained(HF_REPO)
model.eval()

# LOAD IMAGE
image_path = "../datasets/test_dataset/kat_lal_bhak_tiles/output_408.png"
image = Image.open(image_path).convert("RGB")

# PREPROCESS
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs["pixel_values"].to(device)

# INFERENCE
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits  # [1, 7, H, W]

# PREDICT MASK
pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

# COLOR MAP (authored in BGR, convert to RGB for matplotlib/PIL)
colors = np.array([
    [0,   0,   0],     # Background
    [128, 0,   0],     # Residential
    [0,   128, 0],     # Road
    [0,   0,   128],   # River
    [0,   128, 128],   # Forest
    [128, 128, 0],     # Unused land
    [128, 0,   128],   # Agriculture
], dtype=np.uint8)
colors = colors[:, ::-1]

# CREATE SEGMENTATION IMAGE
seg_image = colors[pred_mask]

# DISPLAY
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(seg_image)
plt.title("Segmentation Output")
plt.axis("off")

# SAVE RESULT
output_path = "hf_segformer_prediction.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Segmentation saved at {output_path}")
