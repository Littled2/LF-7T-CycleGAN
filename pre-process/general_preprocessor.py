
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from PIL import Image


class GeneralMRIPreProcessor(ABC):
    """
    Provides a baseline class for MRI pre-processing. Individual dataset processors implement this class.

    """

    def __init__(self, input_dir, output_dir, search_pattern, domain_letter):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.search_pattern = search_pattern
        self.domain_letter = domain_letter

    @abstractmethod
    def get_scans_in_dir(self):
        pass
    
    @abstractmethod
    def load_scan(self):
        pass

    @abstractmethod
    def slice_scan(self):
        pass

    @abstractmethod
    def preprocess_slice_array(self):
        pass


    def save_slice(self, slice_array, direction, slice_number, scan_name, scan_type, dataset_type, domain_letter, save_png=False):
        """
        Saves an individual slice as a .npy file and optionally as a .png

        Args:
            slice_array: (list) List of processed slice arrays
            direction: (string) The directoin of this slice in the brain. E.g. "axial", "coronal" or "saggital"
            slice_number: (int) The index of the slice in the original scan
            scan_name: (string) The ID of the subject of which this slice represents.
            scan_type: (string) The contrast of this scan. E.g. "T1w" or "T2w"
            dataset_type: (string) The current dataset type E.g. "train" or "test"
            domain_letter: (string) A single letter stating if this slice is in translation domain "A" or "B"
            save_png: (boolean) A boolean flag to specify if .png files should be saved in addition to numpy files
            
        """

        # Create folder if it doesn't exist
        save_path = Path(self.output_dir) / direction / f"{dataset_type}{domain_letter}"
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{scan_name}_{scan_type}_{direction}_{slice_number:03d}.npy"

        # Save image
        np.save(save_path / filename, slice_array)

        # Save PNGs as well if specified to same directories
        if save_png:
            arr = slice_array.squeeze()
            arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            png_filename = f"{scan_name}_{scan_type}_{direction}_{slice_number:03d}.png"
            Image.fromarray(arr).save(save_path / png_filename)