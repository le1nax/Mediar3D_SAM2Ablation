import os
import numpy as np
import tifffile

def count_nonzero_images_in_file(filepath):
    """Count how many images (frames) in a TIFF file have at least one nonzero pixel."""
    count = 0
    with tifffile.TiffFile(filepath) as tif:
        for page in tif.pages:
            image = page.asarray()
            if np.any(image):
                count += 1
    return count

def main():
    total_nonzero_images = 0
    dir = "../../../Datasets/CTC/sim3d/01_2d_cpsam"
    for filename in os.listdir(dir):
        if filename.endswith('_label.tiff'):
            filepath = os.path.join(dir, filename)
            count = count_nonzero_images_in_file(filepath)
            print(f"{filename}: {count} images with nonzero pixels")
            total_nonzero_images += count

    print(f"\nTotal nonzero images across all files: {total_nonzero_images}")

if __name__ == "__main__":

    main()