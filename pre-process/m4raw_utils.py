# Slices the 0.3T scans from the M4Raw dataset

# For each scan type, extracts max slices from axial, coronal and sagittal directions

# Saves files to specified directory in sub directory for direction file name format: <subjectID>_<scanType>_<sliceDirection>_<sliceNumber>

import numpy as np
from PIL import Image
import h5py
from general_preprocessor import GeneralMRIPreProcessor






class M4RawPreProcessor(GeneralMRIPreProcessor):
    """
    Provides an implementation of the GeneralMRIPreProcessor for the m4raw dataset.

    """


    def __init__(self, input_dir, output_dir, domain_letter, axial_slice_indexes, coronal_slice_indexes, sagittal_slice_indexes, search_pattern):
        super().__init__(input_dir, output_dir, search_pattern, domain_letter)

        self.axial_slice_indexes = axial_slice_indexes
        self.coronal_slice_indexes = coronal_slice_indexes
        self.sagittal_slice_indexes = sagittal_slice_indexes



    def get_scans_in_dir(self):
        """
        Returns a list of paths of 0.3T scans in the input directory

        Returns:
            List of paths to .h5 files in the input directory

        """

        return list(self.input_dir.glob(self.search_pattern))


    def slice_scan(self, scan_data):
        """
        Slices the 3D scan into 2D slices in all three directions

        Args:
            scan_data: (3D numpy array) 3D array of scan data

        Returns:
            slices: A dictionary of tuples, each containing the slice direction and a list of of slices.

        """

        slices = []

        # Axial
        for i in self.axial_slice_indexes:
            slices.append((scan_data[:, :, i], 'axial', i))
        # Coronal
        for i in self.coronal_slice_indexes:
            slices.append((scan_data[:, i, :], 'coronal', i))
        # Sagittal
        for i in self.sagittal_slice_indexes:
            slices.append((scan_data[i, :, :], 'sagittal', i))
        
        return slices


    def load_scan(self, path):
        """
        Loads an individual scan from the m4raw dataset and returns it as a numpy array

        Args:
            path: (string) The path to a specific scan to be loaded
        
        Returns:
            (numpy array) Array containing scan data

        """

        # Load h5 file
        scan_file = h5py.File(path, "r")

        # Load data into numpy array
        volume = np.array(scan_file['reconstruction_rss'])

        slice_axis = np.argmin(volume.shape)
        
        # Move slice axis to last position
        axes = list(range(3))
        axes.remove(slice_axis)
        axes.append(slice_axis)
        volume = np.transpose(volume, axes)

        return volume



    def preprocess_slice_array(self, slice_array):
        """
        Scales intensity, pads and resizes a 2D slice to (1, 256, 256).
        
        Args:
            slice_array: (numpy array) list of extracted slices

        Returns:
            List of pre-processed slices.
        """
        
        # Scale intensity to [-1, 1] using percentiles
        p_low, p_high = np.percentile(slice_array, [0.5, 99.5])
        slice_array = np.clip(slice_array, p_low, p_high)

        if p_high > p_low:
            slice_array = (slice_array - p_low) / (p_high - p_low)

        slice_array = slice_array * 2 - 1

        slice_array = slice_array.astype(np.float32)

        # Scale up so head occupies the same amount of space as the 7T scans
        scale_factor = 88 / 75  # ≈ 1.1733

        h, w = slice_array.shape
        new_h = int(round(h * scale_factor))
        new_w = int(round(w * scale_factor))
        img = Image.fromarray(slice_array, mode='F')
        img = img.resize((new_w, new_h), Image.BICUBIC)
        slice_array = np.array(img)

        # Pad with black to ensure a minimum size of 256x256
        h, w = slice_array.shape
        pad_h = max(0, 256 - h)
        pad_w = max(0, 256 - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        slice_array = np.pad(slice_array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=-1)

        # Centre-crop to 256x256
        h, w = slice_array.shape
        start_h = (h - 256) // 2
        start_w = (w - 256) // 2
        slice_array = slice_array[start_h:start_h + 256, start_w:start_w + 256]

        # Add a en extra dimension
        slice_array = slice_array[np.newaxis, :]

        return slice_array



