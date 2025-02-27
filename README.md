# Text Classification with LLM-based Synthetic Data Augmentation

An end-to-end project for training text classification models enhanced with various synthetic data augmentation techniques. This repository leverages [PyTorch Lightning](https://www.pytorchlightning.ai/), [Hugging Face Transformers](https://huggingface.co/transformers/), and state-of-the-art augmentation methods (e.g., NLPaug, backtranslation, LLM-based generation) to improve model performance, especially in low-resource settings.

---

## Table of Contents

- [Abstract](#abstract)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Datasets and CSV Format](#datasets-and-csv-format)
- [Prompt Templates](#prompt-templates)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Generating Synthetic Samples](#generating-synthetic-samples)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Abstract

As social media and digital platforms become more influential, the spread of hate speech has accelerated, inflicting serious harm on individuals, communities, and society as a whole. Addressing this challenge is further complicated by the limitations of traditional text data augmentation techniques—such as adding noise to the original text, substituting words based on context, or rephrasing sentences—which often fail to introduce new perspectives that could help classifiers generalize better to new instances. This limitation is especially critical in fields like hate speech detection, where creating datasets is highly labor-intensive, involving the collection of positive samples and expert labeling. In this work, we present an automatic text data augmentation method based on Large Language Models (LLMs) using demonstration-based generation. Our approach generates diverse samples that maintain the original writing style, effectively bridging semantic gaps in the data. While our method focuses on low-resource hate speech datasets, we have empirically demonstrated its potential and consistency even in larger and more robust dataset scenarios. The careful design of the prompt, combined with token sampling strategies, positions our method as the most consistent alternative for LLM-based synthetic data generation compared to existing state-of-the-art methods and prompts. Our benchmarking includes (1) _CMSB_, a dataset centered on sexism; (2) _ETHOS_, a diverse dataset encompassing multiple forms of hate speech; (3) _Stormfront_, which contains white supremacist discourse; and (4) _Antiasian_, an anti-Asian hate speech dataset. We outperform other traditional augmentation methods such as _NLPAug_ or _BackTranslation_ and benchmark with promising results against another LLM-based techniques.

---

## Overview

This project aims to boost text classification performance by augmenting training data with synthetic samples. It provides:
- **Data Modules:** Easily load, preprocess, and split datasets using a custom PyTorch Lightning DataModule.
- **Synthetic Data Generation:** Multiple augmentation techniques including character/word level transformations, backtranslation, and LLM-driven generation.
- **Model Training:** A flexible text classification module built with pretrained encoders and a custom classification head.
- **Experiment Management:** Tools for running experiments over different data sizes, logging, checkpoint management, and result analysis.

---

## Features

- **Flexible Data Loading:** Supports both CSV file inputs and Pandas DataFrames.
- **Multiple Augmentation Strategies:** 
  - **NLPaug:** Character, word, and contextual embedding-based augmentations.
  - **Backtranslation:** Multi-language backtranslation (e.g., Chinese, Spanish, German).
  - **LLM-based Generation:** Generate synthetic samples using state-of-the-art language models.
- **Robust Training Pipeline:** Built on PyTorch Lightning for ease of experimentation with callbacks such as early stopping and model checkpointing.
- **Comprehensive Analysis:** Plot training curves, clean checkpoints, compute baseline deltas, and generate detailed classification reports.
- **Reproducibility:** Seed management and configurable parameters ensure experiments can be reliably reproduced.

---

## Project Structure

```plaintext
├── configs/                  # YAML configuration files for experiments
├── data/
│   ├── source/               # Raw dataset CSV files
│   └── generated/            # Generated/augmented samples will be stored here
├── results/                  # Training outputs, logs, checkpoints, and plots
├── src/
│   ├── datasets/             # Custom Dataset classes (e.g., TextDataset)
│   ├── data_modules/         # PyTorch Lightning DataModules
│   ├── generators/           # Synthetic sample generators and annotators
│   ├── modules/              # Model definition (e.g., TextClassificationModule)
│   ├── prompts/              # Predefined prompts for LLM-based generation
│   └── utils/                # Utility functions (e.g., plotting, checkpoint cleanup)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## Datasets and CSV Format

The project is designed to work with text classification datasets provided in CSV format. Below is an outline of the expected structure for the source CSV files and the additional columns introduced during augmentation:

### Source CSV Files (e.g., `data/source/your_dataset.csv`)

Each source CSV file should include the following columns:

- **sample_id**:  
  A unique integer identifier for each sample.

- **text**:  
  The text content to be classified.

- **label**:  
  The target label for the text sample. In our experiments, labels are typically binary (e.g., `0` or `1`).

- **usage**:  
  Indicates the intended usage of the sample. Typical values include:  
  - `train`: Samples used for training.  
  - `val`: Samples used for validation.  
  - `test`: Samples used for testing.

- **synthetic**:  
  A flag indicating whether the sample is synthetic or original. For source data, this is set to `0`. Augmented samples generated by the system will have this flag set to `1`.

### Augmented CSV Files

When synthetic samples are generated (via methods such as LLM-based generation, nlpaug, or backtranslation), additional columns are added:

- **from_sample**:  
  The original text from which the synthetic sample was derived.

- **from_sample_id**:  
  The `sample_id` of the original sample that generated the synthetic sample.

### Example

A minimal source CSV might look like:

| sample_id | text                         | label | usage | synthetic |
|-----------|------------------------------|-------|-------|-----------|
| 1         | "This is a sample text."     | 0     | train | 0         |
| 2         | "Another example text here." | 1     | val   | 0         |
| 3         | "More data for testing."     | 0     | test  | 0         |

Following augmentation, synthetic samples will include the additional columns `from_sample` and `from_sample_id` to trace back to the original data.

Ensure that your CSV files follow this structure to enable proper processing by the data modules and generators.

---

## Prompt Templates

This project utilizes a collection of prompt templates to guide the generation of synthetic samples and the labeling process. These templates are defined in the `src/prompts.py` file and are organized by dataset type and augmentation strategy. Each key in the dictionary corresponds to a specific generation or labeling method, and they are carefully designed to instruct language models in producing text that adheres to the desired tone, style, and content criteria.

### Overview of Prompts

- **Generation Prompts:**  
  These prompts instruct the model to generate new text samples based on the provided input. For instance:
  - **`cmsb_p2p` prompt:**  
    Instructs the model (an expert sociologist) to generate new sexist sentences that preserve the informal slang and writing style of the input while ensuring the content remains sexist.
  - **`ethos_binary_p2p` prompt:**  
    Guides the model in generating hate speech sentences by mimicking the style and tone of the input.
  - **`antiasian_*` and `stormfront_*` prompts:**  
    Tailored for generating or modifying hate speech content specific to anti-Asian or white supremacy discourse.

- **Negative Generation Prompts:**  
  Prompts like **`cmsb_n2n`** or **`ethos_binary_n2n`** direct the model to create samples on different topics while maintaining the style and tone of the original sentence, thereby serving as negative examples for training.

> Note: *p2p* stands for positive-to-positive while *n2n* stands for neutral-to-neutral.

- **Labeling Prompts:**  
  The **`cmsb_label`** prompt is used for annotating samples by providing a short reasoning followed by a label (SEXIST or NEUTRAL).

- **Augmentation Prompts for Rephrasing:**  
  The **`auggpt_single_turn`** prompt is designed for generating augmented sentences by rephrasing the input text.

### Customization

These prompt templates are central to the synthetic generation and annotation pipeline. They can be customized or extended to suit different datasets or experimental setups. When modifying these prompts, ensure that the changes preserve the intended tone, style, and content requirements to maintain consistency across generated samples.

Below is a snippet from the `src/prompts.py` file for reference:

```python
PROMPTS = {
    'cmsb_p2p': """You are an expert sociologist in sexism content detection. This sentence: "{text}" is considered sexist. Your goal is to propose new sexist sentences preserving the writing style and the informal slang. You should provide {n_samples} different and diverse options. Try not to repeat same hashtags and/or names. Make sure they are sexist. Present the sentences in a numbered list format (Each line like 1. <sentence>) only outputing the sentences without explaining why you create them.""",
    'cmsb_n2n': """You are an expert in generate negatives samples for a classification dataset. Take this sentence: "{text}" as an example for the generation. Your goal is to propose {n_samples} new samples. ...""",
    'auggpt_single_turn': "Please rephrase the following sentence: {text}. ",
    # ... other prompt definitions
}
```

Feel free to try yours!

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/adgiz05/llm-synthetic-generation.git
   cd llm-synthetic-generation
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**

   Create a `.env` file in the root directory to store your Hugging Face token and any other necessary environment variables (this will be necessary for some LLMs):

   ```env
   HF_TOKEN=your_huggingface_token
   ```

---

## Usage

### Training the Model

Run the training script with the desired arguments. For example:

```bash
python train.py --dataset your_dataset_name --device 0 --default-seed 5555 --seeds 10 --val-batch-size 64
```

- **--dataset:** Name of your dataset (expects a CSV file in `data/source/`).
- **--device:** GPU device number.
- **--default-seed:** Default seed for reproducibility.
- **--seeds:** Number of seed experiments to run.
- **--val-batch-size:** Batch size for validation.

Additional parameters can be configured via YAML files in the `configs/` folder.

### Generating Synthetic Samples

Generate augmented samples using various methods (e.g., `nlpaug`, `auggpt`, `backtranslation`, `llm_synthetic_generation`):

```bash
python generate_samples.py --dataset your_dataset_name --generator nlpaug
```

For AugGPT, you must also provide a mode:

```bash
python generate_samples.py --dataset your_dataset_name --generator auggpt --mode single-turn
```

---

## Configuration

Experiment configurations can be stored and loaded from YAML files within the `configs/` directory. This allows for flexible adjustment of hyperparameters, augmentation methods, and training settings without modifying the code.

Example `configs/train_diff_sizes_your_dataset.yaml`:

```yaml
experiments: ["vanilla", "nlpaug-swap", "llm_synthetic_generation"]
model_ids: ["microsoft/deberta-v3-base", "facebook/roberta-base"]
sizes: [1000, 5000, "all"]
batch_sizes: [32, 64, 64]
val_check_interval: 100
```

---