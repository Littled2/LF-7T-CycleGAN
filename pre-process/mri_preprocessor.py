from hcp_utils import HCPPreProcessor
from m4raw_utils import M4RawPreProcessor
import numpy as np


class MRIPreProcessor:
    """
    Provides a generat MRI pre-processor class which implements the individual classes for each dataset

    """

    def __init__(self, config, scan_type_config, output_dir, max_scans=None, hcp_only=None, m4raw_only=None, save_pngs=None):
        
        self.output_dir = output_dir
        self.max_scans = max_scans if max_scans is not None else float('inf')
        self.hcp_only = hcp_only
        self.m4raw_only = m4raw_only
        self.save_pngs = save_pngs
        self.scan_type = scan_type_config["scan_type"]

        # Initialise the HCP preprocessor class
        self.HCP_preprocessor = HCPPreProcessor(
            input_dir=config["preprocess"]["general"]["hcp"]["input_dir"],
            output_dir=self.output_dir,
            search_pattern=scan_type_config["hcp"]["search_pattern"],
            domain_letter=config["preprocess"]["general"]["hcp"]["domain_letter"],
            axial_slice_indexes=scan_type_config["hcp"]["axial_slice_indexes"],
            coronal_slice_indexes=scan_type_config["hcp"]["coronal_slice_indexes"],
            sagittal_slice_indexes=scan_type_config["hcp"]["sagittal_slice_indexes"],
            subject_ids_7T=config["preprocess"]["general"]["hcp"]["7T_subject_ids"]
        ) if not self.m4raw_only else None

        # Initialise the M4Raw preprocessor class
        self.M4Raw_preprocessor = M4RawPreProcessor(
            input_dir=config["preprocess"]["general"]["m4raw"]["input_dir"],
            output_dir=self.output_dir,
            domain_letter=config["preprocess"]["general"]["m4raw"]["domain_letter"],
            search_pattern=scan_type_config["m4raw"]["search_pattern"],
            axial_slice_indexes=scan_type_config["m4raw"]["axial_slice_indexes"],
            coronal_slice_indexes=scan_type_config["m4raw"]["coronal_slice_indexes"],
            sagittal_slice_indexes=scan_type_config["m4raw"]["sagittal_slice_indexes"]
        ) if not self.hcp_only else None
    
    def train_test_split(self, scan_paths, train_ratio=0.8):
        """
        Split the data into setparate train/test sections

        Args:
            scan_paths: (list) List of scan paths
            train_ratio: (float) Decimal to represent portion of data to use for training

        Returns:
            Tuple containing sub-tuples that include the dataset section name and train/test data

        """
        
        # Shuffle the scan paths
        np.random.shuffle(scan_paths)

        # Calculate which index to split the data on
        split_index = int(len(scan_paths) * train_ratio)

        # Return the data split into training and test sets
        return (
            ( "train", scan_paths[:split_index] ),
            ( "test", scan_paths[split_index:] )
        )
        


    def run(self):
        """
        Perform slicing and pre-processing for HCP and M4Raw datasets

        """

        # Determine which pre-processors to run
        if self.hcp_only and not self.m4raw_only:
            preprocessors = [ self.HCP_preprocessor ]
        elif self.m4raw_only and not self.hcp_only:
            preprocessors = [ self.M4Raw_preprocessor ]
        else:
            preprocessors = [ self.HCP_preprocessor, self.M4Raw_preprocessor ]

        for preprocessor in preprocessors:

            scans_paths = preprocessor.get_scans_in_dir()

            # Make train/test split
            train_scans, test_scans = self.train_test_split(scans_paths)

            print(f"\033[95m{preprocessor.__class__.__name__}\033[0m - Processing {min(len(train_scans[1]) + len(test_scans[1]), self.max_scans)} scans")

            processed_counter = 0


            for dataset_type, dataset_scans_paths in [ train_scans, test_scans ]:

                print(f"Processing {len(dataset_scans_paths)} scans for dataset type: {dataset_type}")


                for scan_path in dataset_scans_paths:

                    if self.max_scans is not None and processed_counter >= self.max_scans:
                        break

                    # Get scan name
                    parts = scan_path.stem.split("_")
                    scan_name = parts[0]

                    print(f"Processing scan #{processed_counter + 1} : {scan_name}")

                    # Load scan
                    scan_data = preprocessor.load_scan(scan_path)

                    # Slice scan
                    slices = preprocessor.slice_scan(scan_data)

                    # Pre-process and save slices
                    for slice_tuple in slices:

                        slice_array, direction, slice_number = slice_tuple
                        
                        # Pre-process slice
                        slice_array = preprocessor.preprocess_slice_array(slice_array)

                        # Save slice
                        preprocessor.save_slice(slice_array, direction, slice_number, scan_name, self.scan_type, dataset_type, preprocessor.domain_letter, self.save_pngs)


                    processed_counter += 1

