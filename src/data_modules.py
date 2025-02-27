from src.datasets import TextDataset, SRRTextDataset  

import torch                                      
import pandas as pd                               
import pytorch_lightning as L                     
from transformers import AutoTokenizer            
from torch.utils.data import DataLoader           

# Set a fixed random state for reproducibility
RANDOM_STATE = 5555

class TextDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for text data handling and preparation.
    
    This module is responsible for:
      - Loading and splitting a dataset into training, validation, and testing subsets.
      - Handling synthetic data augmentation through random replacement if configured.
      - Tokenizing text samples using a pretrained tokenizer.
      - Providing DataLoaders for different stages (train, validation, test, prediction).
    
    Parameters:
        model_id (str): Identifier for the pretrained model tokenizer.
        dataset (str or pandas.DataFrame): Either a filename (without extension) to load a CSV
                                           or a DataFrame containing the dataset.
        batch_size (int): Batch size for training. Default is 64.
        srr_config (dict, optional): Configuration for Synthetic Random Replace (SRR). If provided,
                                     it must contain keys 'synthetic' (DataFrame of synthetic data)
                                     and 'replace_prob' (probability of replacing a sample).
        num_workers (int): Number of worker processes for DataLoader. Default is 4.
        prefetch_factor (int): Number of batches to prefetch per worker. Default is 8.
        val_batch_size (int, optional): Batch size for validation/testing/prediction.
                                        Defaults to batch_size if not provided.
    """
    
    def __init__(self, model_id, dataset, batch_size=64, srr_config=None, num_workers=4, prefetch_factor=8, val_batch_size=None):
        # Initialize the base LightningDataModule
        super().__init__()
        # Load the tokenizer using a pretrained model identifier.
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.batch_size = batch_size
        # Use the provided validation batch size or default to the training batch size.
        self.val_batch_size = val_batch_size if val_batch_size is not None else batch_size

        # Load the dataset:
        # If a string is provided, read the corresponding CSV file from 'data/source/' directory.
        # If a DataFrame is provided, use it directly.
        if isinstance(dataset, str):
            self.df = pd.read_csv(f'data/source/{dataset}.csv')
        elif isinstance(dataset, pd.DataFrame):
            self.df = dataset
            
        # Save the synthetic random replacement configuration and DataLoader parameters.
        self.srr_config = srr_config
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        
    def split(self, usage):
        """
        Splits the dataset based on the 'usage' column.
        
        Args:
            usage (str): The type of usage to filter the dataset ('train', 'val', or 'test').
        
        Returns:
            pandas.DataFrame: A subset of the dataset matching the specified usage.
        """
        return self.df[self.df['usage'] == usage]
    
    def append_synthetic(self, synthetic):
        """
        Appends synthetic data to the current dataset.
        
        This method assigns new sample IDs to the synthetic data and marks them as 'train',
        then concatenates the synthetic data with the existing dataset.
        
        Args:
            synthetic (pandas.DataFrame): DataFrame containing synthetic samples.
        """
        # Calculate initial and final sample IDs for the synthetic data based on current maximum ID.
        init, end = self.last_id, self.last_id + len(synthetic)
        # Assign new sample IDs to each synthetic sample.
        synthetic['sample_id'] = [i for i in range(init, end, 1)]
        # Set the usage for all synthetic samples as 'train'.
        synthetic['usage'] = ['train'] * len(synthetic)
        # Concatenate the synthetic data with the original dataset and reset the index.
        self.df = pd.concat([self.df, synthetic]).reset_index(drop=True)

    def setup(self, stage: str):
        """
        Sets up the datasets for training, validation, and testing.
        
        Depending on the presence of an SRR configuration, this method initializes the training dataset
        as either SRRTextDataset (with synthetic replacement) or a standard TextDataset.
        The validation and test datasets are always set up as TextDataset instances.
        
        Args:
            stage (str): The stage to set up ('fit', 'test', etc.). Can be used to customize setup if needed.
        """
        # For the training dataset, use SRRTextDataset if synthetic random replacement is enabled.
        self.train_dataset = SRRTextDataset(
            source=self.split('train'), 
            synthetic=self.srr_config['synthetic'], 
            replace_prob=self.srr_config['replace_prob']
        ) if self.srr_config else TextDataset(self.split('train'))
        
        # Create validation and test datasets from the corresponding splits.
        self.val_dataset = TextDataset(self.split('val'))
        self.test_dataset = TextDataset(self.split('test'))

    def collate_fn(self, data):
        """
        Custom collate function to prepare batches of data.
        
        This function aggregates individual samples into a batch by:
          - Extracting sample IDs, texts, labels, and synthetic flags.
          - Tokenizing the texts with padding and truncation.
          - Converting the tokens and metadata into PyTorch tensors.
        
        Args:
            data (list): List of tuples, each containing (id, text, label, synthetic).
        
        Returns:
            tuple: A tuple containing:
                - ids (list): List of sample IDs.
                - inputs (dict): Dictionary with 'input_ids' and 'attention_mask' tensors.
                - targets (torch.Tensor): Tensor of labels.
                - synthetics (torch.Tensor): Tensor of synthetic flags.
        """
        # Extract sample IDs from the batch.
        ids = [d[0] for d in data]
        # Extract text data from the batch.
        texts = [d[1] for d in data]
        # Convert labels to a long tensor.
        targets = torch.tensor([d[2] for d in data]).long()
        # Convert synthetic flags to a long tensor.
        synthetics = torch.tensor([d[3] for d in data]).long()
        # Tokenize the batch of texts with padding and truncation to a maximum length of 512.
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
        # Prepare the dictionary of inputs required by the model.
        inputs = {'input_ids': tokens['input_ids'], 'attention_mask': tokens['attention_mask']}
        return ids, inputs, targets, synthetics

    def train_dataloader(self):
        """
        Creates the DataLoader for the training dataset.
        
        Returns:
            DataLoader: DataLoader for training with shuffling enabled.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def val_dataloader(self):
        """
        Creates the DataLoader for the validation dataset.
        
        Returns:
            DataLoader: DataLoader for validation.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def test_dataloader(self):
        """
        Creates the DataLoader for the testing dataset.
        
        Returns:
            DataLoader: DataLoader for testing.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def predict_dataloader(self):
        """
        Creates the DataLoader for the prediction phase.
        
        Returns:
            DataLoader: DataLoader for making predictions, typically using the test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )

    def aux_dataloader(self, df, shuffle=False):
        """
        Creates an auxiliary DataLoader from an external DataFrame.
        
        This is useful for evaluating or predicting on additional data that is not part
        of the main training/validation/test splits.
        
        Args:
            df (pandas.DataFrame): The DataFrame to create the DataLoader from.
            shuffle (bool): Whether to shuffle the data. Default is False.
        
        Returns:
            DataLoader: DataLoader for the auxiliary dataset.
        """
        return DataLoader(
            TextDataset(df),
            batch_size=self.val_batch_size,
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor
        )
        
    @property
    def last_id(self):
        """
        Retrieves the highest sample ID currently in the dataset.
        
        Returns:
            int: The maximum value from the 'sample_id' column.
        """
        return self.df['sample_id'].max()
