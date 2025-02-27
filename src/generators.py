import os
import re

from src.prompts import PROMPTS  

import torch
import pandas as pd
from tqdm.notebook import tqdm  
import nlpaug.augmenter.word as naw  
import nlpaug.augmenter.char as nac  
from BackTranslation import BackTranslation  
from dotenv import load_dotenv; load_dotenv()  
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

class SampleGenerator:
    """
    A class to generate synthetic samples using various augmentation techniques.
    
    This class supports multiple augmentation strategies:
      - Generation with a large language model (LLM).
      - Back-translation augmentation.
      - Augmentation using the nlpaug library (character, word, and encoder-based augmentations).
      - Augmentation with an LLM in a GPT-based approach (auggpt).
    
    It also provides utility methods to update CSV files with newly generated samples.
    """
    
    def _load_llm(self, model_id='mistralai/Mistral-7B-Instruct-v0.2'):
        """
        Loads a pretrained language model with quantization for efficient inference.
        
        This method configures the quantization settings to load the model in 4-bit precision,
        and initializes both the model and its tokenizer.
        
        Args:
            model_id (str): Identifier of the pretrained model to load.
            
        Sets:
            self.model: The loaded language model.
            self.tokenizer: The corresponding tokenizer.
        """
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=os.getenv('HF_TOKEN')
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model, self.tokenizer = model, tokenizer

    def _generate_samples_with_llm(self, row, prompt_idx, n_samples=3, prompt=None, temperature=1., typical_p=1., repetition_penalty=1.):
        """
        Generates synthetic samples using the loaded language model.
        
        If no prompt is provided, the method constructs one using the pre-defined PROMPTS.
        It applies a chat template, generates text using the model, and then cleans the output.
        
        Args:
            row (pandas.Series): A row from the input DataFrame containing a text sample.
            prompt_idx (str): Key to select the appropriate prompt template.
            n_samples (int): Number of samples to request from the model.
            prompt (list, optional): Custom prompt to use. Defaults to None.
            temperature (float): Sampling temperature.
            typical_p (float): Typical probability sampling parameter.
            repetition_penalty (float): Penalty to discourage repetition in generated text.
            
        Returns:
            str: Generated text sample after post-processing.
        """
        # Construct prompt if not provided
        if not prompt:
            prompt = [{
                "role": "user",
                "content": PROMPTS[prompt_idx].format(text=row.text, n_samples=n_samples)
            }]

        # Ensure the tokenizer has a pad token set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Apply the chat template and convert to tensor for model input
        tokens = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.model.device)

        # Generate output from the model
        output = self.tokenizer.batch_decode(
            self.model.generate(
                tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                max_new_tokens=1024,
                do_sample=True,
                temperature=temperature,
                typical_p=typical_p,
                repetition_penalty=repetition_penalty
            )
        )[0]
        # Process the output to remove the prompt artifacts and trailing tokens
        return output.split("[/INST]")[-1].replace("</s>", "").strip()

    def generate_based_on_samples(self, inputs, path, dataset='cmsb', n_samples=3):
        """
        Generates synthetic samples based on input samples using LLM generation.
        
        For each sample in the input DataFrame, it decides a mode ('n2n' or 'p2p') based on the label,
        generates new text using the language model, and appends the resulting samples to a CSV file.
        
        Args:
            inputs (pandas.DataFrame): DataFrame containing input samples.
            path (str): Directory or file path where augmented samples will be saved.
            dataset (str): Identifier used in prompt selection.
            n_samples (int): Number of samples to generate per input.
        """
        # Load the language model if not already loaded
        if not hasattr(self, 'model'):
            self._load_llm()
        
        # Iterate over each input sample with a progress bar
        for _, row in tqdm(inputs.iterrows()):
            # Determine augmentation mode based on label
            mode = 'n2n' if row.label == 0 else 'p2p'
            # Construct the path for saving augmented samples
            path_ = os.path.join(path, f'{mode}.csv')
            prompt_idx = f'{dataset}_{mode}'
            # Generate synthetic text using the LLM
            output = self._generate_samples_with_llm(
                row, 
                prompt_idx=prompt_idx, 
                n_samples=n_samples, 
                temperature=1, 
                typical_p=0.8, 
                repetition_penalty=1.2
            )
            # Parse the generated output into a list of texts
            texts = self.parse_numbered_list(output)

            # Helper lambda to create a list with repeated elements
            expand_ = lambda x: [x] * len(texts)

            # Create a DataFrame with the generated samples and corresponding metadata
            samples = pd.DataFrame({
                'sample_id' : expand_(-1),  # -1 indicates synthetic sample
                'text' : texts,
                'usage' : expand_(row.usage),
                'label' : expand_(row.label),
                'synthetic' : expand_(1.),
                'from_sample' : expand_(row.text),
                'from_sample_id' : expand_(row.sample_id),
            })
                
            self.df_update(samples, path_)

    def backtranslate(self, inputs, path):
        """
        Performs back-translation on the input samples to generate augmented text.
        
        Uses two Google Translate URLs to perform the translation process through multiple languages.
        For each sample, back-translates through 'zh-cn', 'es', and 'de' (Chinese, Spanish, German).
        
        Args:
            inputs (pandas.DataFrame): DataFrame containing input samples.
            path (str): File path where back-translated samples will be saved.
        """
        # Initialize the BackTranslation augmenter with specified URLs
        bt = BackTranslation(url=[
            'translate.google.com',
            'translate.google.co.kr',
        ])

        # Process each sample
        for _, row in tqdm(inputs.iterrows()):
            results = []    
            # Back-translate through several languages
            for lang in ['zh-cn', 'es', 'de']:
                mode = f'BT-{lang}'
                try:
                    # Translate from English to the target language and back to English
                    output = bt.translate(row.text, src='en', tmp=lang).result_text
                except:
                    # In case of an error, fallback to the original text
                    output = row.text
    
                results.append({
                    'sample_id' : -1,
                    'text' : output,
                    'usage' : 'train',
                    'label' : row.label,
                    'synthetic' : 1.,
                    'from_sample' : row.text,
                    'from_sample_id' : row.sample_id,
                    'mode' : mode
                })

            # Create a DataFrame from the back-translated results and update the CSV file
            samples = pd.DataFrame(results)
            self.df_update(samples, path)

    def nlpaug(self, inputs, path, n_samples=3):
        """
        Applies various augmentation techniques using the nlpaug library.
        
        Supports:
          - Character-level augmentations: insert, substitute, swap, delete.
          - Word-level augmentations: swap, delete, synonym replacement using WordNet.
          - Encoder-based augmentations: insert and substitute using BERT contextual embeddings.
        
        Generated augmented samples are saved to separate CSV files named after the augmentation method.
        
        Args:
            inputs (pandas.DataFrame): DataFrame with original text samples.
            path (str): Directory path to save the augmented CSV files.
            n_samples (int): Number of augmentations to generate per input sample.
        """
        # Character Augmentation methods
        insert_char_augmentation = nac.RandomCharAug(action="insert")
        substitute_char_augmentaion = nac.RandomCharAug(action="substitute")
        swap_char_augmentation = nac.RandomCharAug(action="swap")
        delete_char_augmentation = nac.RandomCharAug(action="delete")

        # Word Augmentation methods
        swap_word_augmentation = naw.RandomWordAug(action="swap")
        delete_word_augmentation = naw.RandomWordAug(action="delete")
        substitute_wordnet_augmentation = naw.SynonymAug(aug_src='wordnet')
        
        # Encoder-based Augmentation methods using BERT
        insert_bert_augmentation = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert", device='cuda')
        substitute_bert_augmentation = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", device='cuda')

        # Dictionary mapping method names to their corresponding augmentation function
        methods = {
            'insert_char': insert_char_augmentation,
            'substitute_char': substitute_char_augmentaion,
            'swap_char': swap_char_augmentation,
            'delete_char': delete_char_augmentation,
            'swap_word': swap_word_augmentation,
            'delete_word': delete_word_augmentation,
            'substitute_wordnet': substitute_wordnet_augmentation,
            'insert_bert': insert_bert_augmentation,
            'substitute_bert': substitute_bert_augmentation
        }

        # Iterate over each augmentation method
        for name, method in tqdm(methods.items()):
            results = []

            # Apply augmentation to each input sample
            for idx, row in tqdm(inputs.iterrows()):
                augmented = method.augment(row['text'], n=n_samples)

                # Store each augmented text along with its metadata
                for text in augmented:
                    results.append({
                        'sample_id': -1,
                        'text': text,
                        'usage': row.usage,
                        'label': row.label,
                        'synthetic': 1,
                        'from_sample': row.text,
                        'from_sample_id': row.sample_id
                    })
            
            # Update the CSV file for the current augmentation method
            self.df_update(pd.DataFrame(results), os.path.join(path, f'{name}.csv'))

    def auggpt(self, inputs, path, n_samples=3):
        """
        Generates augmented samples using an LLM in a GPT-style approach.
        
        For each input sample, generates multiple augmented texts by calling the LLM generation method.
        The resulting samples are then saved to a CSV file.
        
        Args:
            inputs (pandas.DataFrame): DataFrame containing original samples.
            path (str): File path to save the augmented samples.
            n_samples (int): Number of augmented samples to generate per input sample.
        """
        # Ensure the language model is loaded
        if not hasattr(self, 'model'):
            self._load_llm()

        # Iterate over each sample with a progress bar
        for _, row in tqdm(inputs.iterrows()):
            texts = []
            # Generate a number of augmented texts for the sample
            for _ in range(n_samples):
                texts.append(self._generate_samples_with_llm(row, prompt_idx='auggpt_single_turn', temperature=1.5))

            # Helper lambda to create lists with repeated values
            expand_ = lambda x: [x] * len(texts)

            # Construct a DataFrame with the generated augmented samples
            samples = pd.DataFrame({
                'sample_id' : expand_(-1),
                'text' : texts,
                'usage' : expand_(row.usage),
                'label' : expand_(row.label),
                'synthetic' : expand_(1.),
                'from_sample' : expand_(row.text),
                'from_sample_id' : expand_(row.sample_id),
            })

            self.df_update(samples, path)

    @staticmethod
    def parse_numbered_list(text):
        """
        Parses a numbered list from a text string.
        
        Uses regular expressions to extract the list items by matching numbered prefixes.
        
        Args:
            text (str): Text containing a numbered list.
            
        Returns:
            list: A list of strings, each representing an item from the numbered list.
        """
        return re.findall(r'\d+\.\s*(.*)', text)
    
    @staticmethod
    def df_update(df, path):
        """
        Updates a CSV file by appending new data.
        
        If the file does not exist, it creates an empty CSV with the same columns as the DataFrame.
        Then, it reads the existing data, appends the new DataFrame, and writes the combined data back to the CSV.
        
        Args:
            df (pandas.DataFrame): DataFrame containing new samples to append.
            path (str): File path of the CSV to update.
        """
        # If the file doesn't exist, create an empty CSV with the correct columns
        if not os.path.exists(path):
            pd.DataFrame(columns=df.columns).to_csv(path, index=None)
        
        # Read the existing CSV, concatenate with new data, and save the result
        pd.concat([pd.read_csv(path), df]).to_csv(path, index=None)


class SampleAnnotator:
    """
    A class for annotating samples using multiple language models.
    
    This class loads multiple models and their tokenizers, and provides methods to generate
    annotations for text samples using various prompt templates.
    """
    
    def __init__(self, model_ids, max_new_tokens=150):
        """
        Initializes the annotator with a list of model IDs and configuration for token generation.
        
        Args:
            model_ids (list): List of model identifier strings.
            max_new_tokens (int): Maximum number of tokens to generate for each annotation.
        """
        self.model_ids = model_ids
        # Load each model and tokenizer pair for the provided model IDs
        self.models, self.tokenizers = zip(*[self.load_llm(model_id) for model_id in model_ids])
        self.max_new_tokens = max_new_tokens
        
        ###REMOVE
        # self.model, self.tokenizer = self.load_llm(model_ids[0])

    def load_llm(self, model_id):
        """
        Loads a pretrained language model and its tokenizer with quantization settings.
        
        Args:
            model_id (str): Identifier for the pretrained model.
            
        Returns:
            tuple: A tuple (model, tokenizer) where model is the loaded language model and tokenizer is its tokenizer.
        """
        print(f'Loading {model_id}...')
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            trust_remote_code=True,
            token=os.getenv('HF_TOKEN')
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    
    def annotate_samples(self, inputs, path, dataset='cmsb'):
        """
        Annotates a set of input samples using the loaded language models.
        
        For each sample, a prompt is generated based on the text and a dataset-specific template.
        Each model then generates an annotation which is printed out.
        
        Args:
            inputs (pandas.DataFrame): DataFrame containing samples to annotate.
            path (str): File path where annotations might be saved.
            dataset (str): Dataset identifier to select the appropriate prompt.
        """
        for _, row in tqdm(inputs.iterrows()):
            # Create a prompt using the pre-defined templates and the sample text
            prompt = PROMPTS[f'{dataset}_label'].format(text=row.text)
            print(row.text)
            print()
            # Annotate the sample with each loaded model
            for model_id, model, tokenizer in zip(self.model_ids, self.models, self.tokenizers):
                annotation = self.annotate(prompt, model, tokenizer, model_id)
                print(annotation)
                print('-----------------------------------------------')
                # Uncomment the following line to update annotations to a CSV file
                # self.df_update(annotations, path)
           
    def annotate(self, prompt_content, model, tokenizer, model_id):
        """
        Generates an annotation for a given prompt using a specific model.
        
        Depending on the model, the method adjusts the prompt format and output processing.
        
        Args:
            prompt_content (str): The content of the prompt.
            model: The language model to use.
            tokenizer: The tokenizer corresponding to the model.
            model_id (str): Identifier of the model to apply model-specific processing.
            
        Returns:
            str: The generated annotation after processing.
        """
        def _generate(prompt, model, tokenizer):
            """
            Generates output tokens from the model given a prompt.
            
            This helper function ensures that a pad token is set and then tokenizes the prompt,
            generates model output, and decodes the tokens.
            
            Args:
                prompt (list): The prompt formatted for the model.
                model: The language model.
                tokenizer: The corresponding tokenizer.
                
            Returns:
                str: Decoded model output.
            """
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            tokens = tokenizer.apply_chat_template(prompt, return_tensors="pt").to(model.device)
            output = tokenizer.batch_decode(
                model.generate(
                    tokens,
                    pad_token_id=tokenizer.pad_token_id,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True
                )
            )[0]
            return output
        
        # Process based on model identifier
        if model_id in ['mistralai/Mistral-7B-Instruct-v0.2', 'mistralai/Mistral-7B-Instruct-v0.3']:
            # Format prompt as a simple user message
            prompt = [{"role": "user", "content": prompt_content}]
            output = _generate(prompt, model, tokenizer)
            # Remove prompt-specific tokens from output
            output = output.split("[/INST]")[-1].replace("</s>", "").strip()
        
        elif model_id == 'google/gemma-7b-it':
            prompt = [{"role": "user", "content": prompt_content}]
            output = _generate(prompt, model, tokenizer)
            # Process output specific to the gemma model
            output = output.split("<end_of_turn>")[-1].replace("<eos>", "").strip()

        elif model_id == 'meta-llama/Meta-Llama-3-8B-Instruct':
            # For this model, split the prompt: all but the last line as system, last line as user
            prompt = [
                {"role": "system", "content": '\n'.join(prompt_content.split('\n')[:-1])},
                {"role": "user", "content": prompt_content.split('\n')[-1]},
            ]
            output = _generate(prompt, model, tokenizer)
            # Remove model-specific header and end tokens
            output = output.split("<|end_header_id|>")[-1].replace("<|eot_id|>", "").strip()
        
        return output
