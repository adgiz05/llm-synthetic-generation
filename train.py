import os
import math
import copy
import yaml
import argparse
import sys; sys.path.append('..')  # Add parent directory to system path to import modules from src

from src.modules import TextClassificationModule
from src.data_modules import TextDataModule

import pandas as pd
from tqdm import tqdm
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

# Constant for experiment folder naming
BASE_EXPERIMENT = 'different_data_sizes'


def parse_args():
    """
    Parses command-line arguments and loads additional configuration from a YAML file.

    The function requires a dataset name and a device number.
    It also allows setting a default seed, number of seeds, and validation batch size.
    If a corresponding YAML configuration file exists, its parameters are merged with the command-line arguments.

    Returns:
        argparse.Namespace: Namespace object containing all configuration parameters.
    """
    parser = argparse.ArgumentParser(description='Train with different data sizes.')

    parser.add_argument('-D', '--dataset', type=str, required=True, help="Dataset name (without file extension)")
    parser.add_argument('-d', '--device', type=int, required=True, help="GPU device number to use")
    parser.add_argument('-ds', '--default-seed', type=int, default=5555, help="Default random seed")
    parser.add_argument('-s', '--seeds', type=int, default=10, help="Number of different seeds to try")
    parser.add_argument('-vbs', '--val-batch-size', type=int, default=64, help="Batch size for validation")

    args = parser.parse_args()

    try:
        # Load additional configuration from a YAML file if it exists
        with open(f'configs/train_diff_sizes_{args.dataset}.yaml', 'r') as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        # Merge command-line arguments and YAML configuration
        return argparse.Namespace(**{**vars(args), **config})
    except:
        print(f'No config file found for {args.dataset}.')
        return args


def get_trainer(args, dirpath, version):
    """
    Creates and returns a PyTorch Lightning Trainer with model checkpointing and early stopping callbacks.

    Args:
        args (argparse.Namespace): Configuration parameters including validation check interval.
        dirpath (str): Directory path where checkpoint and log files will be saved.
        version (str): Version name for the checkpoint file.

    Returns:
        Trainer: Configured PyTorch Lightning Trainer.
    """
    # Callback to save the best checkpoint (minimizing validation loss)
    ckpt_callback = ModelCheckpoint(
        monitor='val_loss', mode='min',
        dirpath=dirpath, filename=version,
        save_weights_only=True, save_top_k=1, enable_version_counter=False
    )

    # Callback to stop training early if validation loss does not improve for a set number of epochs
    early_stopping_callback = EarlyStopping(
        monitor='val_loss', mode='min',
        min_delta=0.00, patience=10,
        verbose=True
    )

    return Trainer(
        max_epochs=50,
        callbacks=[ckpt_callback, early_stopping_callback],
        accelerator='gpu', devices=[args.device], precision='16-mixed',
        logger=CSVLogger(save_dir=dirpath),
        deterministic=True,
        val_check_interval=args.val_check_interval
    )


def load_dataset(args, train, method):
    """
    Loads and augments the training dataset based on the specified augmentation method.

    This function allows the use of several augmentation methods. Depending on the method, it:
      - Returns the original training set ('vanilla').
      - Reads synthetic data CSV files (for LLM-based synthetic generation, nlpaug, or other methods)
        and appends matching synthetic samples to the training set.

    Args:
        args (argparse.Namespace): Configuration parameters including the dataset name.
        train (pd.DataFrame): Original training DataFrame.
        method (str): Augmentation method name (e.g., 'vanilla', 'llm_synthetic_generation', 'nlpaug-<algorithm>').

    Returns:
        pd.DataFrame: The augmented training DataFrame.
    """
    # Use a deep copy of the dataset name from args
    dataset = copy.deepcopy(args.dataset)

    if method == 'vanilla':
        return train

    elif method == 'llm_synthetic_generation':
        # Load synthetic samples generated by LLM for both n2n and p2p modes
        n2n = pd.read_csv(f'data/generated/{dataset}/n2n.csv')
        p2p = pd.read_csv(f'data/generated/{dataset}/p2p.csv')
        to_add = pd.concat([n2n, p2p])

    elif method.startswith('nlpaug'):
        # For nlpaug based methods, extract the algorithm name and load corresponding CSV
        algorithm = method.split('-')[-1]  # e.g., 'nlpaug-swap' -> 'swap'
        to_add = pd.read_csv(f'data/generated/{dataset}/{algorithm}.csv')
    else:
        # Default: load the CSV file corresponding to the given method name
        to_add = pd.read_csv(f'data/generated/{dataset}/{method}.csv')
    
    # For each training sample, append matching synthetic samples (where from_sample_id equals the original sample_id)
    synthetics = []
    for _, row in train.iterrows():
        s = to_add[to_add['from_sample_id'] == row['sample_id']]
        for _, srow in s.iterrows():
            synthetics.append({
                'sample_id': srow['from_sample_id'],
                'text': srow['text'],
                'label': row['label'],
                'usage': 'train',
                'synthetic': 1.
            })
        
    synthetics = pd.DataFrame(synthetics)
    return pd.concat([train, synthetics])


def train(args):
    """
    Main training loop for experiments with different data sizes.

    This function:
      - Sets the random seed for reproducibility.
      - Loads the source dataset.
      - Iterates over various experiments, model IDs, data sizes, and random seeds.
      - Optionally reduces the training set size by sampling equal numbers of positive and negative samples.
      - Augments the training set based on the experiment (augmentation method).
      - Trains a text classification model using PyTorch Lightning.
      - Evaluates the model on the test set.
      - Saves the results and removes checkpoint files to save space.

    Args:
        args (argparse.Namespace): Configuration parameters including dataset, model IDs, experiments, sizes, seeds, etc.
    """
    # Set the random seed for reproducibility across experiments
    seed_everything(args.default_seed, workers=True)
    # Load the entire dataset from a CSV file
    df = pd.read_csv(f'data/source/{args.dataset}.csv')

    # Iterate over each experiment (augmentation method)
    for exp in args.experiments:
        results = []
        # Iterate over each model ID to evaluate
        for model_id in args.model_ids:
            # Iterate over different training data sizes and corresponding batch sizes
            for size, batch_size in zip(args.sizes, args.batch_sizes):
                # Iterate over a range of seeds to test variability
                for seed in tqdm(range(args.default_seed, args.default_seed + args.seeds)):
                    # Select the training split from the dataset
                    train = df[df['usage'] == 'train']

                    # If a specific data size is requested (not 'all'), sample equal numbers of positive and negative examples
                    if size != 'all':
                        train_positives = train[train['label'] == 1].sample(size // 2, random_state=seed)
                        train_negatives = train[train['label'] == 0].sample(size // 2, random_state=seed)
                        train = pd.concat([train_positives, train_negatives])

                    # Augment the training dataset using the specified experiment method
                    train = load_dataset(args, train, exp)
                    # Select validation and test splits
                    val = df[df['usage'] == 'val']
                    test = df[df['usage'] == 'test']

                    # Concatenate all splits into one dataset (may be used by the data module)
                    dataset = pd.concat([train, val, test])
                    
                    # Calculate the number of training steps per epoch and set the validation check interval accordingly
                    steps = len(train) // batch_size
                    args.val_check_interval = math.ceil(steps / 2)

                    # Initialize the data module with the model_id, dataset, and batch sizes
                    data_module = TextDataModule(model_id, dataset, batch_size=batch_size, val_batch_size=args.val_batch_size)
                    # Initialize the text classification model
                    model = TextClassificationModule(model_id, n_classes=2)

                    # Construct directory and version strings for saving results and checkpoints
                    dirpath = f'results/{args.dataset}/{BASE_EXPERIMENT}/{exp}/{model_id.split("/")[-1]}/{size}/split_{seed}'
                    version = f'{model_id.split("/")[-1]}_{size}_split_{seed}'

                    # Initialize the trainer with callbacks and logging
                    trainer = get_trainer(args, dirpath, version)
                    try:
                        # Train the model using the provided data module
                        trainer.fit(model, data_module)
                    except:
                        # Skip the current configuration if training fails
                        continue

                    # Load the best checkpointed model
                    model = TextClassificationModule.load_from_checkpoint(
                        f'{dirpath}/{version}.ckpt',
                        model_id=model_id,
                        n_classes=2
                    )   

                    # Evaluate the model on the test set and collect metrics
                    metrics = trainer.test(model, data_module)[0]

                    results.append({
                        'experiment': exp,
                        'model_id': model_id,
                        'size': size,
                        'seed': seed,
                        'test_loss': metrics['test_loss'],
                        'test_f1': metrics['test_f1'],
                        'test_recall': metrics['test_recall'],
                        'test_precision': metrics['test_precision']
                    })

                    # Remove the checkpoint file to conserve disk space
                    os.remove(f'{dirpath}/{version}.ckpt')
    
        # Save all experiment results into a CSV file for later analysis
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'results/{args.dataset}/{BASE_EXPERIMENT}/{exp}/results.csv', index=None)
