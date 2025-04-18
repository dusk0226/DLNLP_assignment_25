""" This file contains classes or functions to do NLP. """
# Step 1: Environment Setup
import torch
import numpy as np
import random
import pandas as pd
from transformers import DistilBertTokenizerFast
from torch.utils.data import Dataset, random_split

# Function to set random seeds for reproducibility
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across torch, numpy and python's random module.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def combine_text(entity, content, special_marks=True):
    """
    Combine the entity and message while wrapping the entity in special markers.
    Special markers <e> and </e> are used to emphasize the entity.
    """
    if special_marks:
        return f"<e> {entity} </e> [SEP] {content}"
    else:
        return f"{entity} [SEP] {content}"

def data_preprocess(dataset: pd.DataFrame):
    """ 
    Add column names to the dataset and combine texts of entity and content.
    """
    dataset.copy()
    dataset.columns = ['index','entity','sentiment','content']
    dataset['input_text'] = dataset.apply(
        lambda row: combine_text(row['entity'], row['content'], 
                                 special_marks=True), axis=1)
    return dataset

def tokenize(special_token_list:list[str]):
    # Initialize the DistilBERT tokenizer.
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # Add the special tokens to the tokenizer's vocabulary.

    special_tokens = {'additional_special_tokens': special_token_list}
    num_added_tokens = tokenizer.add_tokens(special_tokens['additional_special_tokens'])
    print(f"Added {num_added_tokens} special tokens to the tokenizer.")
    return tokenizer

class CreatDataset(Dataset):
    """ Creat torch dataset for the data. """
    def __init__(self, dataframe, tokenizer, max_len=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Retrieve the input text and the corresponding sentiment
        row = self.dataframe.iloc[idx]
        text = row['input_text']
        sentiment = row['sentiment']
        
        # Map sentiment to integer values as labels.
        label_map = {"Positive": 0, "Negative": 1, "Neutral": 2}
        label = label_map.get(sentiment, 2)
        
        # Tokenize the input text with specified truncation and padding.
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        # Return the tokenized input as a dictionary with the label.
        return {
            'input_ids': encoding['input_ids'].squeeze(),    # Remove extra dimensions
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def split_data(dataset, val_ratio=0.1, test_ratio=0.1):
    """
    Randomly split dataset to train, validation, and test datasets based on ratio. 
    """
    len_total = len(dataset)
    len_val = int(len_total*val_ratio)
    len_test = int(len_total*test_ratio)
    len_train = len_total - len_val - len_test
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [len_train, len_val, len_test],
        generator=torch.Generator().manual_seed(42))  # for reproducibility
   
    return train_dataset, val_dataset, test_dataset