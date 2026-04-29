from argparse import ArgumentParser
import json
from mri_preprocessor import MRIPreProcessor
from pathlib import Path




if __name__ == "__main__":
    

    parser = ArgumentParser(
        description="Program to slice HCP and M4Raw scans and build a training and test dataset for LF-7T-CycleGAN"
    )

    # Define command line args
    parser.add_argument("--config-path", help="Path to the config file", default="pre-process-config.json")
    parser.add_argument("--max-scans", help="The maximum number of scans to process for each dataset", default=None, type=int)
    parser.add_argument("--hcp-only", help="Only process HCP scans", default=None, action="store_true")
    parser.add_argument("--m4raw-only", help="Only process M4Raw scans", default=None, action="store_true")
    parser.add_argument("--save-pngs", help="Save PNGs of slices", default=None, action="store_true")

    args = parser.parse_args()

    # Load the config file
    with open(args.config_path) as f:
        config = json.load(f)

    # For each scan type, run the pre-processor
    for scan_type_config in config["preprocess"]["scan_types"]:

        print(f"\n{'='*30}  {scan_type_config['scan_type']}  {'='*30}")

        output_dir = Path(config["preprocess"]["general"]["output_dir"]) / scan_type_config["scan_type"]

        preprocessor = MRIPreProcessor(
            config=config,
            scan_type_config=scan_type_config,
            output_dir=output_dir,
            max_scans=args.max_scans,
            hcp_only=args.hcp_only,
            m4raw_only=args.m4raw_only,
            save_pngs=args.save_pngs
        )

        preprocessor.run()



    print("Done!")




