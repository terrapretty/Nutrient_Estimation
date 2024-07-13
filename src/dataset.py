import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.hs_utils import get_label_from_fname


class SierraDataset(Dataset):
    """Dataset for Dr. Young's hypercubes and csv ground truths."""

    def __init__(self, csv_file, root_dir, transforms=[]):
        """
        Arguments:
            csv_file (string): Path to the csv file with ground truths concentrations.
            root_dir (string): Path to directory containing the hypercubes (i.e. the preprocessed .npy files).
            transform (list, optional): List of transform function to be applied to hypercube.
                All transforms should be applicable to (h,w,c) np array and should be callable with no other
                argument than the hypercube itself.
        """
        self.gt_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.fnames = [x for x in os.listdir(root_dir) if x.endswith(".npy")]
        self.transforms = transforms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        # Read image
        fname = self.fnames[idx]
        fpath = os.path.join(self.root_dir, fname)
        with open(fpath, "rb") as f:
            img = np.load(f) / 255.0

        # Read ground truth
        label = get_label_from_fname(fname, self.gt_data)

        # `label` can be None is csv contains missing/corrupted data
        # in this case, return None
        if label is None:
            return None

        # Label is a dict by default, get values
        label = torch.tensor(list(label.values()))

        # Apply data aug
        if self.transforms:
            for t in self.transforms:
                img = t(img)
        return img, label


def sierra_collate_fn(batch):
    """
    Custom collate function to handle the `None` labels
    when readind Dr. Young's CSV files.
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]

    # Stack batch
    images, ground_truths = zip(*batch)
    images = torch.stack(
        [torch.from_numpy(np.array(img)).permute(2, 0, 1).float() for img in images]
    )
    ground_truths = torch.stack(ground_truths)

    return images, ground_truths
