import os
import shutil
from sklearn.model_selection import train_test_split

# dataset
image_set = "../../datasets/deepGlobe Land Cover Classification Dataset/total_dataset/image"
labelmask_set = "../../datasets/deepGlobe Land Cover Classification Dataset/total_dataset/label-mask"

# train set
train_image_set = "../../datasets/deepGlobe Land Cover Classification Dataset/train/split_image"
train_labelmask_set = "../../datasets/deepGlobe Land Cover Classification Dataset/train/split_label-mask"

# validation set
validation_image_set = "../../datasets/deepGlobe Land Cover Classification Dataset/validation/image"
validation_labelmask_set = "../../datasets/deepGlobe Land Cover Classification Dataset/validation/label-mask"

os.makedirs(train_image_set, exist_ok=True)
os.makedirs(train_labelmask_set, exist_ok=True)
os.makedirs(validation_image_set, exist_ok=True)
os.makedirs(validation_labelmask_set, exist_ok=True)

images = sorted(os.listdir(image_set))
masks = sorted(os.listdir(labelmask_set))

# Split into train and validation
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    images, masks, test_size=0.2, random_state=42
)

# Move files
for img, mask in zip(train_imgs, train_masks):
    shutil.copy(os.path.join(image_set, img), os.path.join(train_image_set, img))
    shutil.copy(os.path.join(labelmask_set, mask), os.path.join(train_labelmask_set, mask))

for img, mask in zip(val_imgs, val_masks):
    shutil.copy(os.path.join(image_set, img), os.path.join(validation_image_set, img))
    shutil.copy(os.path.join(labelmask_set, mask), os.path.join(validation_labelmask_set, mask))