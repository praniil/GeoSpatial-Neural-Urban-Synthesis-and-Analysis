from PIL import Image
import numpy as np

def crop_white_border(image_path, output_path, tolerance=250):
    img = Image.open(image_path).convert('RGB')
    np_img = np.array(img)

    # Create a mask of pixels that are NOT almost white
    mask = np.any(np_img < tolerance, axis=-1)

    if not np.any(mask):
        print("Image appears fully white or above tolerance.")
        return

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1  # add 1 to include last row/col

    cropped = img.crop((x0, y0, x1, y1))
    cropped.save(output_path)
    print(f"Cropped image saved to: {output_path}")

# Example usage
crop_white_border(
    '/home/pranil/workspace/python_projects/smartcity_model/datasets/kathmandu_satellite_images_tiles/output_1.png',
    '/home/pranil/workspace/python_projects/smartcity_model/src/output_1_cropped.png'
)
