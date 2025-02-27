import random 

from torch.utils.data import Dataset

class TextDataset(Dataset):
    """
    A dataset class for text data that wraps a pandas DataFrame.
    
    This dataset is designed to provide access to text samples along with their
    associated metadata such as sample ID, label, and synthetic information.
    
    Attributes:
        df (pandas.DataFrame): DataFrame containing the dataset with columns like 
                               'sample_id', 'text', 'label', and 'synthetic'.
    """
    
    def __init__(self, df):
        """
        Initializes the TextDataset with a pandas DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame containing the dataset.
        """
        self.df = df
        
    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of rows in the DataFrame.
        """
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Retrieves the sample corresponding to the given index.
        
        Args:
            idx (int): Index of the sample to be fetched.
            
        Returns:
            tuple: A tuple containing:
                - sample_id: Unique identifier for the sample.
                - text: The text content.
                - label: Label or target associated with the text.
                - synthetic: Indicator or data about whether the sample is synthetic.
        """
        # Retrieve the row at index 'idx' from the DataFrame using iloc.
        item = self.df.iloc[idx]
        
        # Extract required fields from the row.
        id_ = item['sample_id']
        text = item['text']
        label = item['label']
        synthetic = item['synthetic']
        
        return id_, text, label, synthetic

class SRRTextDataset(Dataset):
    """
    SRRTextDataset (Synthetic Random Replace) conditionally replaces samples with synthetic ones.
    
    This dataset extends the base functionality by introducing a random chance to replace
    a given sample with its synthetic counterpart from a separate synthetic DataFrame.
    
    Attributes:
        source (pandas.DataFrame): Original dataset containing text samples and metadata.
        synthetic (pandas.DataFrame): DataFrame containing synthetic samples that can replace 
                                      the original samples.
        replace_prob (float): Probability with which an original sample is replaced by a synthetic sample.
                              Default is 0.1.
    """
    
    def __init__(self, source, synthetic, replace_prob=0.1):
        """
        Initializes the SRRTextDataset with source and synthetic datasets.
        
        Args:
            source (pandas.DataFrame): Original dataset with columns like 'sample_id', 'text', 'label', 'synthetic'.
            synthetic (pandas.DataFrame): Synthetic dataset where each synthetic sample corresponds to an original sample.
            replace_prob (float, optional): Probability to replace an original sample with its synthetic version.
                                            Defaults to 0.1.
        """
        self.source = source
        self.synthetic = synthetic
        self.replace_prob = replace_prob
        
    def __len__(self):
        """
        Returns the total number of samples in the source dataset.
        
        Returns:
            int: Number of rows in the source DataFrame.
        """
        return len(self.source)
    
    def __getitem__(self, idx):
        """
        Retrieves a sample at the given index, with a random chance to replace it with a synthetic sample.
        
        The method first generates a random number between 0 and 1. If this number is less than or equal to 
        the specified replace_prob, it looks up the corresponding synthetic sample by matching 'from_sample_id'
        with the original sample's 'sample_id' and randomly selects one synthetic sample.
        
        Args:
            idx (int): Index of the sample to be fetched.
            
        Returns:
            tuple: A tuple containing:
                - sample_id: The identifier for the sample.
                - text: The text content (either original or synthetic).
                - label: The label associated with the text.
                - synthetic: Indicator or data about the synthetic nature of the sample.
        """
        # Generate a random threshold between 0 and 1.
        threshold = random.random()
        
        # Retrieve the original sample from the source DataFrame.
        item = self.source.iloc[idx]
        id_ = item['sample_id']
        
        # With a probability defined by replace_prob, replace the sample with a synthetic one.
        if threshold <= self.replace_prob:
            # Filter synthetic DataFrame for rows where 'from_sample_id' matches the original sample's id,
            # then randomly select one synthetic sample.
            item = self.synthetic[self.synthetic['from_sample_id'] == id_].sample(1).iloc[0]
        
        # Extract the necessary fields from the final selected item.
        text = item['text']
        label = item['label']
        synthetic = item['synthetic']
        
        return id_, text, label, synthetic
