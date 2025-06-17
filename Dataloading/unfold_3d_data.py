import os
import tifffile

def slice_and_save_image(input_path, output_dir, start_index, label):
    """
    Loads an image, determines if it's 3D based on number of channels.
    If 3D (more than 5 channels), slices along the z-axis and saves each
    slice as a separate TIFF image in the same output directory.

    Slice filenames: cell_XXXXX.tiff (global sequential numbering)

    Parameters:
    - input_path (str): Path to the input image.
    - output_dir (str): Directory to save the sliced images.
    - start_index (int): Starting index for naming the slices.

    Returns:
    - next_index (int): The next available index after saving slices.
    """
    os.makedirs(output_dir, exist_ok=True)

    image = tifffile.imread(input_path)
    print(f"Loaded image: {input_path} | shape: {image.shape}")

    next_index = start_index

    if image.ndim == 3:
        c, h, w = image.shape
        if c > 5:
            z_dim = c
            print(f"Classified as 3D with {z_dim} slices.")

            for z in range(z_dim):
                slice_img = image[:, :, z]
                if(label):
                    filename = os.path.join(output_dir, f"cell_{next_index:05d}_label.tiff")
                else:
                    filename = os.path.join(output_dir, f"cell_{next_index:05d}.tiff")
                tifffile.imwrite(filename, slice_img)
                print(f"Saved {filename}")
                next_index += 1
        else:
            print("Image classified as 2D RGB — no slicing performed.")
    else:
        print(f"Unsupported image shape: {image.shape}")

    return next_index

def process_directory(input_dir, output_dir, label):
    """
    Applies `slice_and_save_image` to all images in the specified directory,
    saving all slices into a single shared output directory with continuous
    numbering across all images.

    Parameters:
    - input_dir (str): Directory containing input images.
    - output_dir (str): Directory to save all slices.
    """
    supported_exts = ('.tif', '.tiff')
    index = 0

    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith(supported_exts):
            input_path = os.path.join(input_dir, filename)
            index = slice_and_save_image(input_path, output_dir, index, label)



process_directory("/work/scratch/geiger/Datasets/CTC/sim3d/01_test",
                    "/work/scratch/geiger/Datasets/CTC/sim3d/01_test_2d",
                     label=False)