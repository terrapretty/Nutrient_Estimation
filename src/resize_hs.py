"""
This script reads hyperspectral (HS) image files (.bil and .hdr) from a specified folder,
applies calibration (if provided), resizes the images while preserving aspect ratio,
and saves the resized images as .npy arrays in an output folder. Optionally, it also
computes a pseudo-RGB image and saves it as a .png file.

Changes made by Abi:
- Modularized image resizing and RGB conversion into functions.
- Integrated argparse for command-line interface.
- Improved file handling and directory creation.
"""

import os
import argparse
from tqdm import tqdm
import cv2
import numpy as np
from spectral import open_image
from hs_utils import read_calib, apply_calib, find_closest_wavelength

def resize_hs_images(input_folder, output_folder, calib_path=None, new_width=224, save_rgb=True):
    # Retrieve filenames
    filenames = [os.path.splitext(x)[0] for x in os.listdir(input_folder) if x.endswith(".bil")]

    for fname in tqdm(filenames, desc="Resizing HS Images"):
        # Construct file paths
        bil_path = os.path.join(input_folder, fname + ".bil")
        hdr_path = os.path.join(input_folder, fname + ".hdr")

        # Read HS image and calib
        img = open_image(hdr_path).load()
        h, w = img.shape[:2]

        if calib_path:
            calib = read_calib(calib_path)
            img = apply_calib(img, calib)

        # Resize image
        new_h = round(h / (w / new_width))
        resized_img = cv2.resize(img, (new_width, new_h))

        # Save resized image as .npy
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, fname + ".npy")
        np.save(output_path, resized_img)

        if save_rgb:
            # Get pseudo-RGB
            rgb_targets = [630, 532, 465]
            rgb_wavs = [find_closest_wavelength(calib["wavelength"], wav) for wav in rgb_targets]
            rgb_image = resized_img[:, :, rgb_wavs]
            rgb_image_norm = (rgb_image / rgb_image.max() * 255).astype(np.uint8)

            # Save pseudo-RGB as .png
            output_rgb_path = os.path.join(output_folder, fname + "_rgb.png")
            cv2.imwrite(output_rgb_path, rgb_image_norm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to folder containing the .bil and .hdr images",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        type=str,
        help="Path where to store the resized file, as .npy arrays",
    )
    parser.add_argument(
        "--calib",
        "-c",
        required=False,
        type=str,
        help="Path to .csv calib file of the HS camera",
    )
    parser.add_argument(
        "--new-width",
        "-w",
        required=False,
        type=int,
        default=224,
        help="Width to which the HS images should be resized. Original aspect ratio is preserved",
    )
    parser.add_argument(
        "--save-rgb",
        required=False,
        action="store_true",
        help="If set, a pseudo-RGB image will also be computed and saved in the output folder",
    )
    args = parser.parse_args()

    # Call resize_hs_images function with parsed arguments
    resize_hs_images(args.input, args.output, args.calib, args.new_width, args.save_rgb)
