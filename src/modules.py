from src.utils import softmax 

import torch
import pytorch_lightning as L
from transformers import AutoModel
from torch.nn.functional import cross_entropy
from torchmetrics.functional.classification import binary_f1_score, binary_recall, binary_precision

# Define the pooling operation for each model based on its identifier.
POOLING_OPERATION = {
    'google-bert/bert-base-uncased': 'CLS',
    'microsoft/deberta-v3-base': 'MEAN_POOLING',
    'FacebookAI/roberta-base': 'MEAN_POOLING'
}

class TextClassificationModule(L.LightningModule):
    """
    A PyTorch Lightning Module for text classification tasks.
    
    This module uses a pretrained encoder from the Hugging Face Transformers library and
    a linear head for classification. Optionally, it supports synthetic training by employing
    an additional head for synthetic data.
    
    Args:
        model_id (str): Identifier of the pretrained model to use.
        n_classes (int): Number of target classes for classification.
        synthetic_training (bool): Flag to enable training on synthetic data.
        synthetic_weight (float): Weighting factor for the synthetic loss component.
        body_lr (float): Learning rate for the encoder (body) parameters.
        head_lr (float): Learning rate for the classification head parameters.
        label_smoothing (float): Label smoothing parameter for loss computation.
    """
    def __init__(self, model_id='microsoft/deberta-v3-base', n_classes=2, synthetic_training=False, synthetic_weight=1., body_lr=1e-5, head_lr=1e-3, label_smoothing=0.):
        super().__init__()
        # Save hyperparameters for later use and logging.
        self.save_hyperparameters()
        # Set pooling method based on the model identifier.
        self.pooling = POOLING_OPERATION[model_id]

        # Load the pretrained encoder.
        self.encoder = AutoModel.from_pretrained(model_id)
        # Define the classification head.
        self.head = torch.nn.Linear(self.encoder.config.hidden_size, n_classes)

        # If synthetic training is enabled, initialize an additional head.
        if self.hparams.synthetic_training:
            self.synthetic_head = torch.nn.Linear(self.encoder.config.hidden_size, n_classes)
            
    @staticmethod
    def mean_pooling(token_embeddings, attention_mask):
        """
        Performs mean pooling on token embeddings.
        
        Computes a weighted average of token embeddings using the attention mask,
        so that padded tokens do not affect the result.
        
        Args:
            token_embeddings (torch.Tensor): Tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (torch.Tensor): Tensor of shape (batch_size, sequence_length) with 1 for valid tokens.
        
        Returns:
            torch.Tensor: Pooled embeddings of shape (batch_size, hidden_size).
        """
        # Expand the attention mask to match the dimensions of token_embeddings.
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # Compute the weighted sum and normalize by the sum of valid tokens.
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def pool(self, logits, attention_mask=None):
        """
        Pools the encoder output using the specified pooling strategy.
        
        Args:
            logits (torch.Tensor): Encoder outputs.
            attention_mask (torch.Tensor, optional): Attention mask for mean pooling.
        
        Returns:
            torch.Tensor: Pooled representation.
        """
        if self.pooling == 'CLS':
            # Use the first token ([CLS]) representation.
            return logits[:, 0, :]
        elif self.pooling == 'MEAN_POOLING':
            # Compute the mean over valid tokens.
            return self.mean_pooling(logits, attention_mask)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the model.
        
        Args:
            input_ids (torch.Tensor): Tensor containing token ids.
            attention_mask (torch.Tensor): Tensor indicating valid tokens.
        
        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Logits from the classification head.
                - synthetic_output (torch.Tensor or None): Logits from the synthetic head if synthetic training is enabled.
        """
        # Get encoder outputs (logits has shape [batch, sequence_length, hidden_size]).
        logits = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        # Pool the outputs to obtain a fixed-size representation.
        pooled_output = self.pool(logits, attention_mask)
        # Pass the pooled output through the classification head.
        output = self.head(pooled_output)
        # If synthetic training is enabled, process through the synthetic head.
        synthetic_output = self.synthetic_head(pooled_output) if self.hparams.synthetic_training else None
        return output, synthetic_output

    def compute_metrics(self, output, targets, split):
        """
        Computes loss and evaluation metrics for a given batch.
        
        Args:
            output (torch.Tensor): Logits from the model.
            targets (torch.Tensor): Ground truth labels.
            split (str): Identifier for the dataset split (e.g., 'train', 'val', 'test').
        
        Returns:
            dict: Dictionary containing loss, F1 score, recall, and precision.
        """
        # Calculate cross-entropy loss with optional label smoothing.
        loss = cross_entropy(output, targets, label_smoothing=self.hparams.label_smoothing)
        # Calculate evaluation metrics using binary classification metrics.
        f1 = binary_f1_score(torch.argmax(output, dim=1), targets)
        recall = binary_recall(torch.argmax(output, dim=1), targets)
        precision = binary_precision(torch.argmax(output, dim=1), targets)
        return {
            f'{split}_loss': loss,
            f'{split}_f1': f1,
            f'{split}_recall': recall,
            f'{split}_precision': precision
        }
        
    def compute_batch(self, batch, split):
        """
        Processes a batch: computes predictions, metrics, and logs the results.
        
        Args:
            batch (tuple): A tuple containing (ids, inputs, targets, synthetics).
            split (str): Identifier for the current dataset split.
        
        Returns:
            torch.Tensor: Loss value for the batch, potentially combined with synthetic loss.
        """
        # Unpack the batch data.
        _, inputs, targets, synthetics = batch
        # Forward pass through the model.
        output, synthetic_output = self(**inputs)

        # Compute metrics for the main output.
        metrics = self.compute_metrics(output, targets, split)
        # If synthetic training is enabled, compute metrics for the synthetic output.
        synthetic_metrics = self.compute_metrics(synthetic_output, synthetics, f'synthetic_{split}') if self.hparams.synthetic_training else None

        # Combine metrics if synthetic training is enabled.
        metrics = {**metrics, **synthetic_metrics} if self.hparams.synthetic_training else metrics
        # Log the metrics for the current batch.
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)

        # Return the combined loss if synthetic training is enabled.
        return self.dual_loss(metrics[f'{split}_loss'], metrics[f'synthetic_{split}_loss']) if self.hparams.synthetic_training else metrics[f'{split}_loss']

    def dual_loss(self, loss, synthetic_loss):
        """
        Computes the combined loss from the original and synthetic outputs. Here, we tried to predict with the same model the label and
        if the sample was synthetic with the assumption that that double goal would make the model more robust. This did not work at all.
        
        Args:
            loss (torch.Tensor): Loss from the main classification head.
            synthetic_loss (torch.Tensor): Loss from the synthetic classification head.
        
        Returns:
            torch.Tensor: Combined loss value.
        """
        return loss + self.hparams.synthetic_weight * synthetic_loss

    def training_step(self, batch, batch_idx):
        """
        Executes one training step.
        
        Args:
            batch (tuple): Batch of training data.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Loss for the current training step.
        """
        try:
            loss = self.compute_batch(batch, 'train')
            # Clear cached GPU memory.
            torch.cuda.empty_cache()
            return loss
        except:
            pass

    def validation_step(self, batch, batch_idx):
        """
        Executes one validation step.
        
        Args:
            batch (tuple): Batch of validation data.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Loss for the current validation step.
        """
        loss = self.compute_batch(batch, 'val')
        torch.cuda.empty_cache()
        return loss

    def test_step(self, batch, batch_idx):
        """
        Executes one test step.
        
        Args:
            batch (tuple): Batch of test data.
            batch_idx (int): Index of the batch.
        
        Returns:
            torch.Tensor: Loss for the current test step.
        """
        return self.compute_batch(batch, 'test')

    def configure_optimizers(self):
        """
        Configures the optimizers for training.
        
        Returns:
            torch.optim.Optimizer: Optimizer configured with different learning rates for encoder and head.
        """
        # Define parameter groups with different learning rates.
        params = [
            {'params': self.encoder.parameters(), 'lr': self.hparams.body_lr},
            {'params': self.head.parameters(), 'lr': self.hparams.head_lr},
        ]

        # Include synthetic head parameters if synthetic training is enabled.
        if self.hparams.synthetic_training:
            params.append({'params': self.synthetic_head.parameters(), 'lr': self.hparams.head_lr})
        return torch.optim.AdamW(params)

    def embed(self, dataloader):
        """
        Computes embeddings for all samples in a dataloader.
        
        This method runs inference on the encoder and pools the outputs to produce embeddings.
        
        Args:
            dataloader (DataLoader): Dataloader containing the dataset to embed.
        
        Returns:
            list: A list of dictionaries, each containing a sample ID and its corresponding embedding.
        """
        embeddings = []
        with torch.no_grad():
            for ids, inputs, targets, _ in dataloader:
                # Move tensors to the current device.
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                # Get encoder outputs.
                encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                # Pool the outputs to get a fixed-size representation.
                pooled_output = self.pool(encoder_output, attention_mask).detach().cpu().numpy()
                # Associate each embedding with its sample ID.
                result = [{'sample_id': id_, 'embedding': emb} for id_, emb in zip(ids, pooled_output)]
                embeddings.extend(result)
        return embeddings
    
    def report(self, dataloader):
        """
        Generates a detailed report for samples in the dataloader.
        
        For each sample, it computes the predictions, logits, loss, and whether the sample was misclassified.
        
        Args:
            dataloader (DataLoader): Dataloader containing the dataset to report on.
        
        Returns:
            list: A list of dictionaries with detailed information for each sample.
        """
        report = []
        with torch.no_grad():
            for batch in dataloader:
                ids, inputs, targets, _ = batch
                # Move inputs to device.
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)
                # Get model outputs.
                outputs, _ = self(input_ids, attention_mask)
                outputs = outputs.detach().cpu()
                # Get predictions by selecting the class with the highest logit.
                preds = torch.max(outputs, 1)[1]
                # Compute softmax probabilities for each output.
                logits = [softmax(logit) for logit in outputs.tolist()]
                # Compute loss for each sample.
                losses = torch.nn.functional.cross_entropy(outputs, targets, reduction='none').numpy()
                # Determine which samples were misclassified.
                misclassified = (preds != targets).tolist()
                preds, outputs = preds.numpy(), outputs.numpy()

                # Create a detailed report entry for each sample.
                report.extend([{
                    'sample_id': id_,
                    'predicted': pred,
                    'logit_0': logit[0],
                    'logit_1': logit[1],
                    'loss': loss,
                    'misclassified': miss
                } for id_, pred, logit, loss, miss in zip(ids, preds, logits, losses, misclassified)])
        return report
