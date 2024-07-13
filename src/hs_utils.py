import re
import math
from functools import partial
import numpy as np
import pandas as pd
from PIL import Image


def read_calib(calib_path):
    calibration_df = pd.read_csv(calib_path)
    return calibration_df


def apply_calib(image, calib_data):
    calib_values = calib_data["reflectance"].values
    calibrated_image = image * calib_values.reshape(1, 1, -1)
    return calibrated_image


def find_closest_wavelength(wavs, target):
    return (np.abs(np.array(wavs) - np.array(target))).argmin()


perc_mapping = {
    "0": "0.0",
    "8": "8.0",
    "16": "16.7",
    "32": "33.33",
    "64": "66.7",
    "100": "100.0",
}
elt_mapping = {"N": "N", "P": "P", "K": "K", "CA": "Ca", "S": "S", "MG": "Mg"}
rep_mapping = {"A": 1, "B": 2, "C": 3, "D": 4}

# Normalize by dividing by each columns
norm_factors =  {
    "N": 5.96, 
    "P": 1.15, 
    "K": 10.3, 
    "Ca": 2.12, 
    "S": 1.75, 
    "Mg": 0.38
}

def get_label_from_fname(fname, gt_data):
    regex = r"([A-Z]+)([0-9]+)_([A-Z]+)*"
    # print("fname", fname)
    # Find element, percentage and rep number in fname
    try:
        elt, perc, rep = re.match(regex, fname, re.I).groups()
    except AttributeError:
        raise ValueError(f"fname {fname} BENJAMIN")

    # print("elt, perc, rep", (elt, perc, rep))

    elt, perc, rep = elt_mapping[elt], perc_mapping[perc], rep_mapping[rep]

    # Find corresponding row in csv file
    match1 = gt_data[gt_data["Element"] == elt]
    match2 = match1[match1["Percentage"].astype(str) == perc]
    match3 = match2[match2["Rep#"] == rep]

    cols = ["N", "P", "K", "Ca", "Mg", "S"]
    try:
        label = {}
        for c in cols:
            val = float(match3[c].iloc[0])
            if math.isnan(val):
                return None
            val /= norm_factors[c]
            label[c] = val
        return label
    except (ValueError, IndexError):
        return None


# def hs_crop(image, crop_size=(224, 224)):
#     """
#     Takes a random crop from a hypercube.

#     Parameters:
#         image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).
#         crop_size (tuple): The desired crop size (height, width).

#     Returns:
#         numpy.ndarray: The cropped image.
#     """
#     h, w = image.shape[:2]
#     crop_h, crop_w = crop_size

#     # Ensure the crop size is not greater than the image size
#     # print("(h, w)", (h, w))
#     if crop_h > h or crop_w > w:
#         raise ValueError("Crop size must be smaller than the image dimensions.")

#     # Choose the top-left corner of the crop randomly
#     start_h = np.random.randint(0, h - crop_h + 1)
#     start_w = np.random.randint(0, w - crop_w + 1)

#     # Perform the crop
#     return image[start_h : start_h + crop_h, start_w : start_w + crop_w, :]


def hs_crop(image, crop_size=(224, 224), crop_mode='random'):
    """
    Takes a random or center crop from an image.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).
        crop_size (tuple): The desired crop size (height, width).
        crop_mode (str): The mode of cropping - 'random' or 'center'.

    Returns:
        numpy.ndarray: The cropped image.
    """
    h, w = image.shape[:2]
    crop_h, crop_w = crop_size

    # Ensure the crop size is not greater than the image size
    if crop_h > h or crop_w > w:
        
        raise ValueError(f"Crop size must be smaller than the image dimensions., got {crop_h} > {h} or {crop_w} > {w}")

    if crop_mode == 'random':
        # Choose the top-left corner of the crop randomly
        start_h = np.random.randint(0, h - crop_h + 1)
        start_w = np.random.randint(0, w - crop_w + 1)
    elif crop_mode == 'center':
        # Choose the top-left corner of the crop to be the center of the image
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
    else:
        raise ValueError("crop_mode must be either 'random' or 'center'.")

    # Perform the crop
    return image[start_h : start_h + crop_h, start_w : start_w + crop_w, :]



def random_resized_crop(image, size, scale=(0.42, 1.0), ratio=(0.75, 1.33)):
    """
    Perform a random resized crop on the input image and resize it to the given size.

    Args:
        image (np.ndarray): Input image of shape (h, w, c).
        size (tuple): Output size (height, width).
        scale (tuple): Range for the area of the crop (default: (0.08, 1.0)).
        ratio (tuple): Range for the aspect ratio of the crop (default: (0.75, 1.3333333333333333)).

    Returns:
        np.ndarray: Cropped and resized image of shape (size[0], size[1], c).
    """
    h, w, c = image.shape
    area = h * w

    for _ in range(10):
        target_area = np.random.uniform(*scale) * area
        aspect_ratio = np.random.uniform(*ratio)

        crop_w = int(round(np.sqrt(target_area * aspect_ratio)))
        crop_h = int(round(np.sqrt(target_area / aspect_ratio)))

        if crop_w <= w and crop_h <= h:
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)

            crop = image[top:top+crop_h, left:left+crop_w, :]
            crop_image = Image.fromarray(crop)
            resized_crop = crop_image.resize(size, Image.ANTIALIAS)
            return np.array(resized_crop)
    
    # Fallback to central crop if no valid crop is found
    in_ratio = w / h
    if in_ratio < min(ratio):
        crop_w = w
        crop_h = int(w / min(ratio))
    elif in_ratio > max(ratio):
        crop_h = h
        crop_w = int(h * max(ratio))
    else:
        crop_w = w
        crop_h = h

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    crop = image[top:top+crop_h, left:left+crop_w, :]
    crop_image = Image.fromarray(crop)
    resized_crop = crop_image.resize(size, Image.ANTIALIAS)
    return np.array(resized_crop)

def hs_random_horizontal_flip(image, p=0.5):
    """
    Flips an image array horizontally (left to right).

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array of shape (h, w, c).

    Returns:
        numpy.ndarray: The horizontally flipped image.
    """
    if np.random.rand() < p:
        # Reverse the order of columns, using all rows and channels
        return image[:, ::-1, :]
    else:
        # Return the original image if no flip is performed
        return image


def center_square_crop_and_resize(image, new_hw=(224, 224)):
    """
    Takes a square center crop of the input image and resizes it to the desired dimensions.

    Args:
        image (np.ndarray): Input image of shape (h, w, c).
        new_hw (tuple): Desired height and width of the output image.

    Returns:
        np.ndarray: Cropped and resized image of shape (new_h, new_w, c).
    """
    new_h, new_w = new_hw
    h, w = image.shape[:2]

    # Determine the size of the square crop (smallest dimension of the input image)
    crop_size = min(h, w)

    # Calculate the top-left corner of the crop
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2

    # Perform the crop
    crop = image[top:top+crop_size, left:left+crop_size, :]

    # Convert the crop to a PIL image
    crop_image = Image.fromarray(crop)

    # Resize the crop to the desired dimensions
    resized_crop = crop_image.resize((new_w, new_h), Image.ANTIALIAS)

    # Convert the resized crop back to a NumPy array and return it
    return np.array(resized_crop)
   
def hs_train_transforms(
    crop_size=(224, 224),
    flip_lr_prob=0.5,
):
    """
    Returns list of transforms to be applied sequentially to the HS image.
    """
    return [
        partial(hs_random_horizontal_flip, p=flip_lr_prob),
        partial(hs_crop, crop_size=crop_size, crop_mode="random"),
        # partial(random_resized_crop, crop_size=crop_size),
    ]


def hs_val_transforms(
    crop_size=(224, 224),
):
    """
    Returns list of transforms to be applied sequentially to the HS image.
    """
    return [
        # partial(center_square_crop_and_resize, new_hw=crop_size),
        partial(hs_crop, crop_size=crop_size, crop_mode="center"),
    ]
