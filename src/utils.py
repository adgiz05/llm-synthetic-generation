import os
import logging
import shutil

from src.datasets import TextDataset  

import torch
import numpy as np
import pandas as pd
from glob import glob
import pytorch_lightning as pl
import plotly.graph_objects as go  
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import TrainerCallback
from plotly.subplots import make_subplots

def plot_training(dataset):
    """
    Plots training metrics (loss and evaluation metrics) for each experiment of a given dataset.
    
    The function searches for experiment directories under 'results/{dataset}/', reads the CSV files 
    containing experiment results, identifies the best ablation based on test_f1 score, and plots the training 
    and validation losses along with evaluation metrics (F1, recall, precision). The plots are saved as HTML files.
    
    Args:
        dataset (str): The name of the dataset for which to plot the training metrics.
    """
    # List experiments by extracting the folder names from the results directory.
    experiments = [exp.split('/')[-2] for exp in glob(f'results/{dataset}/*/')]
    for experiment in experiments:
        try:
            # Read the experiment's results CSV to determine the best ablation
            df = pd.read_csv(f'results/{dataset}/{experiment}/results.csv')
            best_ablation = df.sort_values(by='test_f1', ascending=False).iloc[0]
            best_seed = best_ablation['seed']
            best_f1 = best_ablation['test_f1']

            # Load training metrics for the best experiment run.
            metrics = pd.read_csv(f'results/{dataset}/{experiment}/{experiment}_{best_seed}/lightning_logs/version_0/metrics.csv')

            # Create a subplot with two rows: one for losses and one for metrics.
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

            # Plot train and validation losses on the first row.
            fig.add_trace(go.Scatter(x=metrics['step'], y=metrics['train_loss'],
                                       mode='lines+markers',
                                       name='Train Loss',
                                       connectgaps=True), row=1, col=1)
            fig.add_trace(go.Scatter(x=metrics['step'], y=metrics['val_loss'],
                                       mode='lines+markers',
                                       name='Validation Loss',
                                       connectgaps=True), row=1, col=1)

            # Plot validation F1, recall, and precision on the second row.
            fig.add_trace(go.Scatter(x=metrics['step'], y=metrics['val_f1'],
                                       mode='lines+markers',
                                       name='Validation F1',
                                       connectgaps=True), row=2, col=1)
            fig.add_trace(go.Scatter(x=metrics['step'], y=metrics['val_recall'],
                                       mode='lines+markers',
                                       name='Validation Recall',
                                       connectgaps=True), row=2, col=1)
            fig.add_trace(go.Scatter(x=metrics['step'], y=metrics['val_precision'],
                                       mode='lines+markers',
                                       name='Validation Precision',
                                       connectgaps=True), row=2, col=1)

            # Update axis labels for clarity.
            fig.update_xaxes(title_text="Training Steps", row=2, col=1)
            fig.update_yaxes(title_text="Loss", row=1, col=1)
            fig.update_yaxes(title_text="Metrics", row=2, col=1)

            # Set the overall layout with a title indicating the experiment, dataset, best seed, and test F1 score.
            fig.update_layout(title=f'{experiment} - {dataset} - Best Seed {best_seed} - Test F1 {best_f1:.4f}',
                              height=600)

            # Save the plot as an HTML file.
            fig.write_html(f"results/{dataset}/{experiment}/{experiment}_{best_seed}_training_plot.html")
        except:
            # If any error occurs (e.g., file missing or experiment not completed), print a message.
            print(f'{experiment} not completed')

def clean_ckpts(dataset):
    """
    Cleans up checkpoint files for experiments in a given dataset.
    
    For each experiment, the function moves the best checkpoint (based on test_f1) from its subdirectory 
    to the experiment's root folder and deletes all other checkpoint files.
    
    Args:
        dataset (str): The name of the dataset whose experiment checkpoints need cleaning.
    """
    experiments = [exp.split('/')[-2] for exp in glob(f'results/{dataset}/*/')]
    for experiment in experiments:
        try:
            # Read the experiment's results to find the best seed.
            df = pd.read_csv(f'results/{dataset}/{experiment}/results.csv')
            best_seed = df.sort_values(by='test_f1', ascending=False).iloc[0]['seed']

            # Move the best checkpoint file to the root of the experiment directory.
            shutil.move(f'results/{dataset}/{experiment}/{experiment}_{best_seed}/{experiment}_{best_seed}.ckpt',
                        f'results/{dataset}/{experiment}/{experiment}_{best_seed}.ckpt')

            # Remove all other checkpoint files within subdirectories.
            for model in glob(f'results/{dataset}/{experiment}/*/*.ckpt'):
                os.remove(model)
        except:
            print(f'Already cleaned: {experiment} or not completed')

def compute_baseline_deltas(dataset, clean_ckpts=False):
    """
    Computes performance deltas between experimental results and a baseline for a given dataset.
    
    The function reads the baseline results and calculates the mean and best performance metrics for the baseline.
    It then iterates over each experiment, calculates the differences (deltas) in metrics relative to the baseline,
    and concatenates the results into a single CSV file.
    
    Optionally, it cleans the checkpoint files after processing.
    
    Args:
        dataset (str): The dataset name for which to compute deltas.
        clean_ckpts (bool): Whether to clean checkpoint files after computation.
    
    Returns:
        pd.DataFrame: A DataFrame containing the computed deltas and metrics for each experiment.
    """
    path = f'results/{dataset}'
    baseline_path = os.path.join(path, 'baseline', 'results.csv')
    baseline_results = pd.read_csv(baseline_path)

    # Compute the average baseline metrics.
    baseline_means = baseline_results[['test_loss', 'test_f1', 'test_recall', 'test_precision']].mean()
    # Extract the best baseline results based on maximum test_f1.
    baseline_bests = baseline_results[baseline_results['test_f1'] == baseline_results['test_f1'].max()].squeeze()

    results = pd.DataFrame()

    # Iterate over each experiment results CSV in the dataset.
    for p in glob(path + '/*/*.csv'):
        experiment_results = pd.read_csv(p)
        experiment = experiment_results['experiment'].unique()[0]

        metrics = experiment_results[['test_loss', 'test_f1', 'test_recall', 'test_precision']]

        means = metrics.mean()
        stds = metrics.std()
        bests = metrics[metrics['test_f1'] == metrics['test_f1'].max()].squeeze()

        # Calculate the deltas between experiment metrics and baseline metrics.
        mean_deltas = means - baseline_means
        best_deltas = bests - baseline_bests

        # Rename indices for clarity.
        means.index = ['mean_' + idx for idx in means.index]
        stds.index = ['std_' + idx for idx in stds.index]
        bests.index = ['best(maxf1)_' + idx for idx in bests.index]
        mean_deltas.index = ['mean_delta_' + idx for idx in mean_deltas.index]
        best_deltas.index = ['best(maxf1)_delta_' + idx for idx in best_deltas.index]
        
        # Concatenate all computed metrics into one DataFrame row.
        results = pd.concat([results, pd.concat([pd.Series({'experiment': experiment, 'dataset': dataset}),
                                                 means, stds, bests, mean_deltas, best_deltas], axis=0)
                             .to_frame().T.reset_index(drop=True)])
        
    # Sort the results: baseline first, then experiments sorted by mean test_f1 in descending order.
    results = pd.concat([results[results['experiment'] == 'baseline'],
                         results[results['experiment'] != 'baseline'].sort_values(by='mean_test_f1', ascending=False)],
                        ignore_index=True)
    # Save the combined results as a CSV.
    results.to_csv(os.path.join(path, 'results.csv'), index=None)

    if clean_ckpts:
        clean_ckpts(dataset)
    
    return results

def classification_results(model, data_module, df):
    """
    Evaluates a text classification model on a given DataFrame of samples.
    
    The function sets the DataFrame index to sample IDs, creates a DataLoader using the data_module,
    performs model inference, and writes the predicted class, logits, loss, and misclassification flag
    back to the DataFrame.
    
    Args:
        model: The text classification model.
        data_module: The data module containing the auxiliary DataLoader.
        df (pd.DataFrame): DataFrame containing samples with at least 'sample_id', 'text', 'label', etc.
    
    Returns:
        pd.DataFrame: The updated DataFrame with additional columns for predictions, logits, loss, and misclassification.
    """
    df['sample_id'] = df['sample_id'].astype(int)
    df.set_index('sample_id', inplace=True, drop=False)
    dataloader = data_module.aux_dataloader(df)
    model = model.eval()

    with torch.no_grad():
        for batch in dataloader:
            ids, inputs, targets, _ = batch
            input_ids = inputs['input_ids'].to(model.device)
            attention_mask = inputs['attention_mask'].to(model.device)
            outputs, _ = model(input_ids, attention_mask)

            outputs = outputs.detach().cpu()
            preds = torch.max(outputs, 1)[1]
            loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none').numpy()
            misclassified = (preds != targets).tolist()
            preds, outputs = preds.numpy(), outputs.numpy()

            # Update each sample in the DataFrame with the evaluation results.
            for i, id_ in enumerate(ids):
                df.loc[id_, 'predicted'] = preds[i]
                logits = softmax(outputs[i].tolist())
                df.loc[id_, 'logit_0'] = logits[0]
                df.loc[id_, 'logit_1'] = logits[1]
                df.loc[id_, 'loss'] = loss[i]
                df.loc[id_, 'misclassified'] = misclassified[i]

    df.reset_index(inplace=True, drop=True)
    return df

def clean_mistral_output(df):
    """
    Cleans the output text generated by the Mistral language model.
    
    The function removes any NaN entries, replaces certain dashes with spaces, and keeps only the first line of text.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'text' column with generated outputs.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    df = df[df['text'].notna()]
    
    def clean(text):
        text = text.replace(' – ', ' ')
        text = text.replace(' - ', ' ')
        # Keep only the first line of the text.
        text = text.split('\n')[0]
        return text
    
    df['text'] = df['text'].apply(clean)
    return df

def proportions(df):
    """
    Prints the proportion of positive and negative samples for each usage split in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'usage' column and a 'label' column.
    """
    for usage in list(df['usage'].unique()):
        total = len(df[df['usage'] == usage])
        positives = int(df[df['usage'] == usage]['label'].sum())
        negatives = total - positives
        print(f'{usage} split: {total} samples (pos: {positives}, neg: {negatives}); {positives/total * 100}% positives.')

def softmax(x):
    """
    Computes the softmax of a list or numpy array.
    
    Args:
        x (list or np.array): Input vector.
    
    Returns:
        np.array: Softmax probabilities.
    """
    return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))

def gaussian(x, mu=.5, sigma=.1):
    """
    Computes the Gaussian (normal) function for a given x value.
    
    Args:
        x (float or np.array): Input value(s).
        mu (float): Mean of the Gaussian.
        sigma (float): Standard deviation of the Gaussian.
    
    Returns:
        np.array: Gaussian probability density value(s).
    """
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

class GpuMemoryTracker(pl.Callback):
    """
    A PyTorch Lightning Callback for tracking GPU memory usage.
    
    This callback captures GPU memory snapshots at the end of every training batch and saves them to a file.
    It helps monitor GPU memory consumption during training.
    
    Args:
        max_entries (int): Maximum number of memory snapshot entries to record.
        file_prefix (str): Prefix for the file where memory snapshots are saved.
    """
    def __init__(self, max_entries=100000, file_prefix="gpu_usage/gpu_memory_snapshot"):
        super().__init__()
        self.max_entries = max_entries
        self.file_prefix = file_prefix
        self.snapshot_counter = 0

    def on_train_start(self, trainer, pl_module):
        # Start recording GPU memory snapshot history.
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Capture and save a memory snapshot after each training batch.
        try:
            snapshot_file = f"{self.file_prefix}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_file)
            self.snapshot_counter += 1
        except Exception as e:
            print(f"Failed to capture memory snapshot: {e}")

    def on_train_end(self, trainer, pl_module):
        # Stop recording memory history.
        torch.cuda.memory._record_memory_history(enabled=None)

def text_dataloader(df, model_id='microsoft/deberta-v3-base', batch_size=64, num_workers=4, prefetch_factor=8, shuffle=False, pin_memory=True):
    """
    Creates a DataLoader for text data based on the provided DataFrame.
    
    The DataLoader uses a custom collate function that tokenizes the text data using a pretrained tokenizer.
    
    Args:
        df (pd.DataFrame): DataFrame containing text samples.
        model_id (str): Identifier for the pretrained tokenizer.
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        prefetch_factor (int): Number of samples to prefetch per worker.
        shuffle (bool): Whether to shuffle the data.
        pin_memory (bool): Whether to pin memory in DataLoader.
    
    Returns:
        DataLoader: Configured DataLoader for the text dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def collate_fn(data):
        # Extract individual elements from the batch.
        ids = [d[0] for d in data]
        texts = [d[1] for d in data]
        targets = torch.tensor([d[2] for d in data]).long()
        synthetics = torch.tensor([d[3] for d in data]).long()
        # Tokenize texts with padding and truncation.
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        inputs = {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
        return ids, inputs, targets, synthetics
    
    return DataLoader(
        TextDataset(df), 
        batch_size=batch_size, 
        collate_fn=collate_fn, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor, 
        pin_memory=pin_memory
    )

class GenerationAnalysisCallback(TrainerCallback):
    """
    A TrainerCallback for analyzing generated text during model training.
    
    This callback logs generated texts from a language model at specified intervals during training.
    It uses a set of evaluation prompts to generate text, appends the results to a CSV file, and logs the activity.
    
    Args:
        tokenizer: The tokenizer used to process prompts.
        model: The language model used for text generation.
        eval_texts (list): List of texts to be used as prompts for evaluation.
        model_path (str): Directory path where the generations CSV will be stored.
        log_interval (int): Interval (in training steps) at which generations are analyzed.
    """
    def __init__(self, tokenizer, model, eval_texts, model_path, log_interval=30):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_texts = eval_texts
        self.log_interval = log_interval
        self.model_path = model_path
        self.logger = logging.getLogger(__name__)
    
    def on_step_end(self, args, state, control, **kwargs):
        # Log generation analysis at specified intervals.
        if state.global_step % self.log_interval == 0:
            self.logger.info(f"Step {state.global_step}: Analyzing generations...")
            self.analyze_generations(state)

    def analyze_generations(self, state):
        """
        Generates text for each evaluation prompt and appends the results to a CSV file.
        
        If a CSV file already exists at the specified model_path, it is loaded and new entries are appended.
        Otherwise, a new DataFrame is created.
        
        Args:
            state: Training state object containing the current global_step.
        """
        # Check if a generations CSV exists; if not, create a new DataFrame.
        if os.path.exists(f'{self.model_path}/generations.csv'):
            df = pd.read_csv(f'{self.model_path}/generations.csv')
        else:
            df = pd.DataFrame(columns=['step', 'prompt', 'generations'])

        self.model.eval()
        # Process each evaluation prompt.
        for text in self.eval_texts:
            # Tokenize the prompt using the chat template.
            inputs = self.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                # Generate text with a maximum of 150 new tokens.
                generated_text = self.tokenizer.decode(self.model.generate(inputs, max_new_tokens=150)[0], skip_special_tokens=True)
            
            # Create a new row with the current step, prompt, and generated text.
            new_row = pd.DataFrame({'step': [state.global_step], 'prompt': [text], 'generations': [generated_text]})
            df = pd.concat([df, new_row], ignore_index=True)
        
        # Save the updated generations to CSV.
        df.to_csv(f'{self.model_path}/generations.csv', index=False)
        self.model.train()
