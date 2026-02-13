import imageio.v3 as iio

# Read the image
img = iio.imread("/netshares/BiomedicalImageAnalysis/Resources/dataset_collection/DSBowl2018/data-science-bowl-2018/stage1_train/CTC_format/01/t0000.png")

# Print the shape
print("Image shape:", img.shape)