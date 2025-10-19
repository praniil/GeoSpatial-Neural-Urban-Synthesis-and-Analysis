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

#seperate_test
source_dir_test = "../../datasets/deepGlobe Land Cover Classification Dataset/test"
image_dir_test = f"{source_dir_test}/image"
mask_dir_test = f"{source_dir_test}/mask"

os.makedirs(image_dir_test, exist_ok=True)
os.makedirs(mask_dir_test, exist_ok=True)

for fname in os.listdir(source_dir_test):
    if fname.endswith("_sat.jpg"):
        shutil.move(os.path.join(source_dir_test,fname), os.path.join(image_dir_test, fname))
    elif fname.endswith("_mask.png"):
        shutil.move(os.path.join(source_dir_test, fname), os.path.join(mask_dir_test, fname))

#seperate_valid
source_dir_valid = "../../datasets/deepGlobe Land Cover Classification Dataset/valid"
image_dir_valid = f"{source_dir_valid}/image"
mask_dir_valid = f"{source_dir_valid}/mask"


os.makedirs(image_dir_valid, exist_ok=True)
os.makedirs(mask_dir_valid, exist_ok=True)

for fname in os.listdir(source_dir_valid):
    if fname.endswith("_sat.jpg"):
        shutil.move(os.path.join(source_dir_valid,fname), os.path.join(image_dir_valid, fname))
    elif fname.endswith("_mask.png"):
        shutil.move(os.path.join(source_dir_valid, fname), os.path.join(mask_dir_valid, fname))

