"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md


Changes by Ed Blewitt:

1. Implemented optional ensemble inference functionality, allowing inference using deep ensembles or a single model
2. Added uncertainty quantification via variance maps which are also overlayed on the generated mean predictions

"""

import os
from pathlib import Path
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


if __name__ == "__main__":

    opt = TestOptions().parse()  # get test options
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    
    ensemble_models = []

    # If no ensemble model names are provided, test_name is not required.
    if len(opt.ensemble_models) != 0 and not opt.test_name:
        print("Error: If testing using an ensemble, a test name must be provided using --test_name.")
        exit(1)

    # Check if test_name was not provided, a name was provided
    if not opt.test_name and not opt.name:
        print("Error: If not testing using an ensemble, a model name must be provided using --name.")
        exit(1)

    # Set a test name based on whether one was provided
    test_name = opt.test_name if opt.test_name else opt.name

    # Prepare models for testing. If no ensemble models are provided, test model with --name as per default behaviour.
    if opt.ensemble_models:
        ensemble_model_names = opt.ensemble_models
    else:
        ensemble_model_names = [opt.name]  # If no ensemble names are provided, provide functionality via --name


    for model_name in ensemble_model_names:

        opt.name = model_name  # temporarily set the name of the experiment to the current ensemble model name

        model = create_model(opt) # create a model given opt.model and other options
        model.setup(opt) # regular setup: load and print networks; create schedulers

        # test with eval mode. This only affects layers like batchnorm and dropout.
        # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
        # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
        if opt.eval:
            model.eval()
        ensemble_models.append(model)

    # Reset opt.name for saving to first ensemble model name
    opt.name = test_name

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    # create a website
    web_dir = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}"  # define the website directory

    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = Path(f"{web_dir}_iter{opt.load_iter}")

    print(f"creating web directory {web_dir}")
    webpage = html.HTML(web_dir, f"Experiment = {opt.name}, Phase = {opt.phase}, Epoch = {opt.epoch}")

    if opt.save_npy:
        npy_dir = web_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)


    
    # Run inference using each ensemble model

    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply the model to opt.num_test images.
            break

        model_outputs = []

        # Run inference on all models
        for model in ensemble_models:

            model.set_input(data)  # unpack data from data loader
            model.test()  # run inference
            model_outputs.append(model.get_current_visuals()["fake_B"])

        # Compute average outputs
        mean_output = torch.stack(model_outputs).mean(dim=0)

        # Remove white pixels from mean_output
        mean_output = torch.clamp(mean_output, min=-1.0, max=0.99)


        # Get variance of outputs but insert matrix of zeros when testing only 1 model
        variance_map = torch.stack(model_outputs).var(dim=0, correction=0) if len(model_outputs) > 1 else torch.zeros_like(model_outputs[0])

        # Normalize the variance map to [0, 1] for visualization
        u = variance_map.squeeze().cpu().float().numpy()
        u = (u - u.min()) / (u.max() - u.min() + 1e-8)

        # Convert the normalized variance map to a tensor
        u_tensor = torch.from_numpy(u).unsqueeze(0).unsqueeze(0).float().to(mean_output.device)

        # Define a scaling factor
        orange = torch.tensor([2.0, 0.8, 0.0]).view(1, 3, 1, 1).to(mean_output.device)

        # Get the visuals from one of the models (they should be the same/very similar)
        visuals = model.get_current_visuals()

        visuals["fake_B"] = mean_output
        visuals["variance_map"] = variance_map
        visuals["uncertainty"] = torch.clamp(mean_output + u_tensor * orange, -1.0, 0.99) # overlay the variance map on the mean prediction and remove white pixels

        # Remove white pixels
        visuals["rec_B"] = torch.clamp(visuals["rec_B"], min=-1.0, max=0.99)
        visuals["fake_A"] = torch.clamp(visuals["fake_A"], min=-1.0, max=0.99)
            
        img_path = model.get_image_paths()  # get image paths        

        if i % 5 == 0:
            print(f"processing ({i:04d})-th image... {img_path}")

        # Optionally, save as numpy files
        if opt.save_npy:
            for key, tensor in visuals.items():
                stem = Path(img_path[0]).stem
                arr = tensor.squeeze().cpu().float().numpy()
                np.save(npy_dir / f"{stem}_{key}.npy", arr)

        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
