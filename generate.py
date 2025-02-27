import os
import argparse

from src.generators import SampleGenerator

import torch
import pandas as pd

def parse_args():
    """
    Parses command-line arguments for the sample generation script.
    
    The arguments include:
      - dataset: Name of the dataset to be used.
      - generator: The generation method to use (e.g., nlpaug, auggpt, backtranslation, llm_synthetic_generation).
      - device: (Optional) GPU device identifier.
      - n_samples: (Optional) Number of samples to generate per input.
      
    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Generate samples from a dataset.')

    parser.add_argument('-D', '--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('-g', '--generator', type=str, required=True, help="Generation method to use")
    parser.add_argument('-d', '--device', type=str, default='0', required=False, help="GPU device to use")
    parser.add_argument('-ns', '--n-samples', type=int, default=3, required=False, help="Number of samples to generate per input sample")

    return parser.parse_args()

def generate_samples(args):
    """
    Generates synthetic samples for the specified dataset using various augmentation methods.
    
    Depending on the specified generator argument, the function will:
      - Create a directory for generated samples if it doesn't exist.
      - Load the source dataset from a CSV file.
      - For the 'nlpaug' generator, generate character-level or word-level augmentations.
      - For the 'backtranslation' generator (or its alias 'bt'), perform back-translation.
      - For the 'llm_synthetic_generation' generator, generate synthetic samples using an LLM.
      
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Ensure the directory for generated data exists for the given dataset
    generated_dir = f'data/generated/{args.dataset}'
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    # Instantiate the sample generator
    generator = SampleGenerator()

    # Load the source dataset CSV file
    source = pd.read_csv(f'data/source/{args.dataset}.csv')

    if args.generator == 'nlpaug':
        generation_path = f'data/generated/{args.dataset}/'
        try:
            # Try to load existing augmented data for 'insert_char'
            augmented = pd.read_csv(f'{generated_dir}/insert_char.csv')
            # Exclude samples that already have an augmented version
            left = source[~source['sample_id'].isin(augmented['from_sample_id'])]
        except:
            # If no augmented data is found, use the entire source dataset
            left = source
        # Generate augmentations using nlpaug
        generator.nlpaug(left, generation_path)
    
    elif args.generator == 'auggpt':
        # Clear the cached GPU device count and set the visible CUDA device based on argument
        torch.cuda.device_count.cache_clear()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

        generation_path = f'data/generated/{args.dataset}/auggpt_single_turn.csv'
        try:
            # Load existing augmented data if available
            augmented = pd.read_csv(generation_path)
            left = source[~source['sample_id'].isin(augmented['from_sample_id'])]
        except:
            left = source
        # Generate samples using the AugGPT method with the specified mode
        generator.auggpt(left, generation_path, mode=args.mode)

    elif args.generator in ['backtranslation', 'bt']:
        # Define the generation path for backtranslation output
        generation_path = f'data/generated/{args.dataset}/backtranslation-cn-es-de.csv'
        try:
            # Load existing backtranslated samples if they exist
            augmented = pd.read_csv(generation_path)
            left = source[~source['sample_id'].isin(augmented['from_sample_id'])]
        except:
            left = source
        # Generate backtranslated samples
        generator.backtranslate(left, generation_path)
    
    elif args.generator == 'llm_synthetic_generation':
        # Clear cached GPU device count and set the device for LLM-based generation
        torch.cuda.device_count.cache_clear()
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device
        generation_path = f'data/generated/{args.dataset}/'
        try:
            # Load previously generated synthetic samples for negatives and positives
            augmented_negatives = pd.read_csv(f'{generated_dir}/n2n.csv')
            augmented_positives = pd.read_csv(f'{generated_dir}/p2p.csv')
            augmented = pd.concat([augmented_negatives, augmented_positives])
            # Exclude samples that already have synthetic counterparts
            left = source[~source['sample_id'].isin(augmented['from_sample_id'])]
        except:
            left = source
        # Generate synthetic samples using the LLM-based method
        generator.generate_based_on_samples(left, generation_path, dataset=args.dataset)

if __name__ == "__main__":
    # Parse the command-line arguments
    args = parse_args()
    # Generate samples based on the parsed arguments
    generate_samples(args)
