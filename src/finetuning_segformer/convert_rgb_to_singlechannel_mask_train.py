import numpy as np
from PIL import Image
import os

color_to_label = {
    (0, 255, 255): 0,    # Cyan → Urban Land
    (255, 255, 0): 1,    # Yellow → Agriculture Land
    (255, 0, 255): 2,    # Magenta → Rangeland
    (0, 255, 0): 3,      # Green → Forest Land
    (0, 0, 255): 4,      # Blue → Water
    (255, 255, 255): 5,  # White → Barren Land
    (0, 0, 0): 6         # Black → Unknown
}


mask_train_directory = "../../datasets/deepGlobe Land Cover Classification Dataset/train/mask"
output_mask_directory = "../../datasets/deepGlobe Land Cover Classification Dataset/train/label-mask"
os.makedirs(output_mask_directory, exist_ok=True)

for mask_name in os.listdir(mask_train_directory):
    mask_path = os.path.join(mask_train_directory, mask_name)
    mask = np.array(Image.open(mask_path))

    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for color, label in color_to_label.items():
        matches = np.all(mask == color, axis = -1)
        label_mask[matches] = label

    output_path = os.path.join(output_mask_directory, mask_name)
    Image.fromarray(label_mask).save(output_path)

print("All masks converted to single channel label mask")
print(label_mask)