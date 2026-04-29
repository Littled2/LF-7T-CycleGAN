"""

This file was implemented by Ed Blewitt to enable cycleGAN to handle .npy files. The code in this file includes snippets from
the default dataset classes included as part of the cycleGAN library.

"""

import os
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
import torch
import random
from torchvision import transforms
import numpy as np


class MRIDataset(BaseDataset):
    """
    Dataset class for loading unpaired MRI images for LF-7T-CycleGAN training and testing.

    Data must be organised into testA, testB, trainA, and trainB folders under the specified dataroot.
    
    Each folder should contain .npy files representing MRI scans. All files should shore arrays of the same shape.

    Images are randomly mutated at train time to improve the robustness of the model.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + "A")  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + "B")  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(os.path.join(self.dir_A, f) for f in os.listdir(self.dir_A) if f.endswith('.npy'))  # load .npy files from '/path/to/data/trainA'
        self.B_paths = sorted(os.path.join(self.dir_B, f) for f in os.listdir(self.dir_B) if f.endswith('.npy'))  # load .npy files from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        btoA = self.opt.direction == "BtoA"

        # Same transform for datasets A and B
        self.transform = self._build_transform()

    def _build_transform(self):
        # Only apply at train time
        if self.opt.isTrain:

            # Apply random horizontal fliping, rotation and scaling.

            return transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15,
                    scale=(0.9, 1.1),
                    interpolation=transforms.InterpolationMode.BILINEAR
                )
            ])
        
        return None


    def __getitem__(self, index):
        """
        
        Return a data point and associated metadata.

        Parameters:
            index: An integer specifying the index of the datapoint

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        A = self._load(A_path)
        B = self._load(B_path)

        if self.transform:
            A = self.transform(A)
            B = self.transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def _load(self, path):
        """
        Helper function to load a single image

        Args:
            path (string)

        """


        arr = np.load(path).astype(np.float32)
        return torch.from_numpy(arr)

    def __len__(self):
        """
        Return the total number of images in the dataset.

        """

        return max(self.A_size, self.B_size)
