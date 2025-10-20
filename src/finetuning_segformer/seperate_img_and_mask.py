import os
import shutil

#seperate train
source_dir_train = "../../datasets/deepGlobe Land Cover Classification Dataset/train"
image_dir_train = f"{source_dir_train}/image"
mask_dir_train = f"{source_dir_train}/mask"


os.makedirs(image_dir_train, exist_ok=True)
os.makedirs(mask_dir_train, exist_ok=True)

for fname in os.listdir(source_dir_train):
    if fname.endswith("_sat.jpg"):
        shutil.move(os.path.join(source_dir_train,fname), os.path.join(image_dir_train, fname))
    elif fname.endswith("_mask.png"):
        shutil.move(os.path.join(source_dir_train, fname), os.path.join(mask_dir_train, fname))



