# preprocess_no_bg_removal.py

import os
import argparse
from tqdm import tqdm
import numpy as np
from spectral import open_image
from hs_utils import read_calib, apply_calib, find_closest_wavelength

def extract_spectral_data(image):
    """
    Extract the averaged spectrum for each plant.
    """
    h, w, c = image.shape
    return np.mean(image, axis=(0, 1))

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
        help="Path where to store the spectral data",
    )
    parser.add_argument(
        "--calib",
        "-c",
        required=False,
        type=str,
        help="Path to .csv calib file of the HS camera",
    )
    args = parser.parse_args()

    folder = args.input
    calib_path = args.calib
    output_folder = args.output

    filenames = [x.split(".")[0] for x in os.listdir(folder) if ".bil" in x]
    spectral_data = []

    for fname in tqdm(filenames):
        filepath = os.path.join(folder, fname)
        hdr_path = filepath + ".bil.hdr"

        img = open_image(hdr_path).load()
        calib = read_calib(calib_path)
        calib_image = apply_calib(img, calib)
        spectrum = extract_spectral_data(calib_image)
        spectral_data.append(spectrum)

    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, "spectral_data_no_bg_removal.npy"), spectral_data)
