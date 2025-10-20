import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SegformerForSemanticSegmentation, SegformerFeatureExtractor
import torch.nn.functional as F

# CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
batch_size = 2
learning_rate = 5e-5
num_epochs = 20
image_size = 512

# Folder structure
train_img_dir = "../../datasets/deepGlobe Land Cover Classification Dataset/train/split_image"
train_mask_dir = "../../datasets/deepGlobe Land Cover Classification Dataset/train/split_label-mask"
val_img_dir = "../../datasets/deepGlobe Land Cover Classification Dataset/validation/image"
val_mask_dir = "../../datasets/deepGlobe Land Cover Classification Dataset/validation/label-mask"

output_dir = "./segformer_logs"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# DATASET CLASS
class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, feature_extractor, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.feature_extractor = feature_extractor
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = np.array(Image.open(os.path.join(self.img_dir, self.images[idx])).convert("RGB"))
        mask = np.array(Image.open(os.path.join(self.mask_dir, self.masks[idx])))

        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        encoded = self.feature_extractor(images=img, return_tensors="pt")
        pixel_values = encoded["pixel_values"].squeeze()
        return pixel_values, mask.clone().long()  # fixed warning

# TRANSFORMS
train_tf = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.5),
    A.Resize(image_size, image_size),
    ToTensorV2()
])

val_tf = A.Compose([
    A.Resize(image_size, image_size),
    ToTensorV2()
])

# DATALOADERS
feature_extractor = SegformerFeatureExtractor.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

train_dataset = SegDataset(train_img_dir, train_mask_dir, feature_extractor, train_tf)
val_dataset = SegDataset(val_img_dir, val_mask_dir, feature_extractor, val_tf)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# MODEL SETUP
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

# METRICS
def mean_iou(pred, target, num_classes):
    pred = torch.argmax(pred, dim=1)  # B,H,W
    # Resize target to match pred size
    target_resized = F.interpolate(target.unsqueeze(1).float(), size=pred.shape[1:], mode="nearest").squeeze(1).long()

    ious = []
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target_resized == cls)
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        if union == 0:
            continue
        ious.append(intersection / union)
    return torch.mean(torch.tensor(ious)) if ious else torch.tensor(0.0)
# TRAINING LOOP
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    total_train_iou = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(pixel_values=imgs, labels=masks)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        total_train_iou += mean_iou(outputs.logits.detach(), masks, num_classes).item()

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_iou = total_train_iou / len(train_loader)

    # VALIDATION
    model.eval()
    total_val_loss = 0.0
    total_val_iou = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(pixel_values=imgs, labels=masks)
            loss = outputs.loss

            total_val_loss += loss.item()
            total_val_iou += mean_iou(outputs.logits, masks, num_classes).item()

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_iou = total_val_iou / len(val_loader)

  
    # LOGGING + CHECKPOINT
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f} | Val IoU:   {avg_val_iou:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), ckpt_path)

# SAVE FINAL MODEL
final_path = os.path.join(output_dir, "segformer_finetuned_final.pth")
torch.save(model.state_dict(), final_path)
print(f"\nTraining complete! Final model saved at: {final_path}")
