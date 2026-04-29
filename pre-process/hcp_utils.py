# Slices the 7T scans from the HCP dataset

# For each scan type, extracts max slices from axial, coronal and sagittal directions

# Saves files to specified directory in sub directory for direction file name format: <subjectID>_<scanType>_<sliceDirection>_<sliceNumber>

import numpy as np
from pathlib import Path
import nibabel as nib
from PIL import Image
import zipfile
import gzip
from io import BytesIO
from general_preprocessor import GeneralMRIPreProcessor
import os






class HCPPreProcessor(GeneralMRIPreProcessor):
    """
    Provides an implementation of the GeneralMRIPreProcessor for the HCP dataset.

    """
        

    def __init__(self, input_dir, output_dir, search_pattern, domain_letter, axial_slice_indexes, coronal_slice_indexes, sagittal_slice_indexes, subject_ids_7T):
        super().__init__(input_dir, output_dir, search_pattern, domain_letter)

        self.axial_slice_indexes = axial_slice_indexes
        self.coronal_slice_indexes = coronal_slice_indexes
        self.sagittal_slice_indexes = sagittal_slice_indexes
        self.subject_ids_7T = subject_ids_7T
        
    

    def get_scans_in_dir(self):
        """
        Returns a list of paths to the 7T scans in the input directory

        """


        # These are the scans that are 7T:
        paths = []

        for s in self.subject_ids_7T:

            s_zip_path = Path(self.input_dir) / f"{str(s)}{self.search_pattern}"
            
            if(os.path.exists(s_zip_path)):
                paths.append(s_zip_path)

        return paths


    def slice_scan(self, scan_data):
        """
        Slices the 3D scan into 2D slices in all three directions

        Args:
            scan_data: (3D array) The 3D array of scan data

        Returns:
            slices: A dictionary of tuples, each containing the slice direction and a list of of slices.

        """
        
        slices = []

        # Axial
        for i in self.axial_slice_indexes:
            if i >= scan_data.shape[2]:
                print(f"Skipping slice {i} — out of bounds for depth {scan_data.shape[2]}")
                continue
            this_slice = scan_data[:, :, i]
            # Rotate so top of head is top of image
            this_slice = np.rot90(this_slice)
            slices.append((this_slice, 'axial', i))
        # Coronal
        for i in self.coronal_slice_indexes:
            this_slice = scan_data[:, i, :]
            this_slice = np.rot90(this_slice, k=1)
            slices.append((this_slice, 'coronal', i))
        # Sagittal
        for i in self.sagittal_slice_indexes:
            this_slice = scan_data[i, :, :]
            this_slice = np.rot90(this_slice, k=1)
            slices.append((this_slice, 'sagittal', i))
        
        return slices


    def load_scan(self, path):
        """
        Loads a scan from the HCP dataset and returns it as a numpy array

        Args:
            path: (string) The path to a specific scan to be loaded
        
            Returns: (numpy array) Array containing scan data

        """

        parts = path.stem.split("_")
        subject_id = parts[0]
    
        internal_scan_path = f"{subject_id}/T1w/T1w_acpc_dc_restore.nii.gz"

        # Open the zip file
        with zipfile.ZipFile(path, 'r') as z:
            with z.open(internal_scan_path) as f:
                # Decompress .nii.gz into memory
                buf = BytesIO(gzip.decompress(f.read()))

        # Parse raw file data
        img = nib.FileHolder(fileobj=buf)
        img = nib.Nifti1Image.from_file_map({'header': img, 'image': img})

        # Re-orient
        img = nib.as_closest_canonical(img)

        return img.get_fdata()


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

        # Avoid division by zero
        if p_high > p_low:
            slice_array = (slice_array - p_low) / (p_high - p_low)

        slice_array = slice_array * 2 - 1

        slice_array = slice_array.astype(np.float32)

        # Pad with black to ensure a minimum size of 256x256
        h, w = slice_array.shape
        pad_h = max(0, 256 - h)
        pad_w = max(0, 256 - w)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        slice_array = np.pad(slice_array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=-1)

        # Resize to 256 x 256 using bicubic
        img = Image.fromarray(slice_array, mode='F')
        img = img.resize((256, 256), Image.BICUBIC)
        slice_array = np.array(img)

        # Add channel dimension
        slice_array = slice_array[np.newaxis, :]

        return slice_array