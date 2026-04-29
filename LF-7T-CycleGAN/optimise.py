import copy
import random
import time
import numpy as np
import optuna
import torch
from skimage.metrics import structural_similarity as ssim
from data import create_dataset
from models import create_model
from options.train_options import TrainOptions
from util.util import cleanup_ddp, init_ddp
from util.visualizer import Visualizer



def set_rand_seeds(opt):
    """
    Sets the random seeds for each library.

    Args:
        opt: The options object containing a .seed integer attribute.

    """

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)



def normalise_tensor(x):
    """
    Normalises the input tensor to the range 0 to 1.

    """

    low, high = x.min(), x.max()
    return (x - low) / (high - low + 1e-8)



def evaluate_model(model, train_opt):
    """
    Evaluates model using the validation set

    Args:
        model: The model to evaluate
        train_opt: The training options object

     Returns:
        The average SSIM score
    
    """

    eval_opt = copy.deepcopy(train_opt)
    eval_opt.phase = "val"
    eval_opt.max_dataset_size = float("inf")
    eval_opt.serial_batches = True  # deterministic ordering

    dataset = create_dataset(eval_opt)
    scores = []

    # Process each datapoint in val dataset
    with torch.no_grad():
        for data in dataset:
            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            # Check if file does not exist just in case
            if "fake_B" not in visuals or "real_B" not in visuals:
                continue

            # Select the fake_B and real_B images as numpy arrays
            fake_B = visuals["fake_B"][0, 0].cpu().numpy()
            real_B = visuals["real_B"][0, 0].cpu().numpy()

            # Calculate SSIM
            score = ssim(normalise_tensor(real_B), normalise_tensor(fake_B), data_range=1.0)
            scores.append(score)

    if not scores:
        return 0.0

    return float(np.mean(scores))




def train_once(opt, trial):
    """
    Trains candidate model and evaluates it using the validation set.
    
    The main training loop used here is mostly copied from the originl train.py file.

    Args:
        opt: Options object with settings for this trial
        trial: This Optuna trial

    Returns:
        Score for this run

    """
    set_rand_seeds(opt)

    # Initialise dataset and model
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    
    total_iters = 0

    # Main training loop - mostly taken from the original train.py
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        visualizer.reset()

        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            t_data = iter_start_time - iter_data_time  # always defined before use

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters()

            if total_iters % opt.display_freq == 0:
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(
                    model.get_current_visuals(), epoch, total_iters, save_result
                )

            if total_iters % opt.print_freq == 0:
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()

        # Score model on the validation dataset
        val_score = evaluate_model(model, opt)
        
        # Reportvia Optuna
        trial.report(val_score, epoch)

        # End trial if pruned
        if trial.should_prune():
            print(f"Pruning trial early at epoch {epoch}")
            raise optuna.TrialPruned()

        # Compute time taken and output results
        elapsed_time = time.time() - epoch_start_time
        print(f"End of epoch {epoch} of {opt.n_epochs + opt.n_epochs_decay} - SSIM: {val_score:.4f} \t Time Taken: {elapsed_time} sec")

    return val_score



def objective(trial, base_opt):
    """
    Sets the hyperparameters for a trial, runs training, evaluation and returns the best SSIM score.

    Args:
        trial: This Optuna trial
        base_opt: The basic options passed in when the program was started
    
    Returns:
        The score for this trial

    """

    
    # Copy the options to avoid changing parameters for other trials 
    opt = copy.deepcopy(base_opt)

    # Define hyperparameter search space

    opt.lambda_A = trial.suggest_float("lambda_A", 0.5, 14.0)
    opt.lambda_B = trial.suggest_float("lambda_B", 0.5, 14.0)
    opt.lambda_identity = trial.suggest_float("lambda_identity", 0.0, 0.5)
    opt.lambda_per_c = trial.suggest_float("lambda_per_c", 0.1, 1.0)
    opt.lambda_per_s = trial.suggest_float("lambda_per_s", 0.2, 3.0)
    opt.lambda_content_adversarial = trial.suggest_float("lambda_content_adversarial", 0.2, 3.0)
    opt.lr = trial.suggest_float("lr", 5e-5,  3e-4, log=True)

    opt.n_epochs = 8 # Set the max epochs to 8
    opt.n_epochs_decay = 0
    opt.lr_policy = "step"
    opt.lr_decay_iters = 999_999_999
    opt.max_dataset_size = 400 # Restrict dataset size to increase speed

    # Disable any logging or saving
    opt.print_freq = 999_999_999
    opt.display_freq = 999_999_999
    opt.save_latest_freq = 999_999_999
    opt.save_epoch_freq = 999_999_999
    opt.no_html = True

    # Print the parameters for this trial
    print("This trial parameters:")
    for k in ["lambda_A", "lambda_B", "lambda_identity", "lambda_per_c", "lambda_per_s", "lambda_content_adversarial", "lr"]:
        print(f"  {k}: {getattr(opt, k)}")

    return train_once(opt, trial)





if __name__ == "__main__":

    # Parse the train options passed from the command line.    
    base_opt = TrainOptions().parse()

    base_opt.device = init_ddp()

    # Define the Optuna study
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=1,
            interval_steps=1,
        ) # Implement early pruning after 3 epochs
    )


    # Run the study
    try:
        study.optimize(
            lambda trial:  objective(trial, base_opt),
            n_trials=80,
        )
    finally:
        cleanup_ddp() # Cleanup even if study fails


    # Output the best hyperparameters and SSIM score
    print(f"\n{'='*20} BEST RESULT {'='*20}")
    print(study.best_params)
    print(f"Best SSIM: {study.best_value}")