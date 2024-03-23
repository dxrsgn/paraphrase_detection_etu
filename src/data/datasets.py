import os
import torch
import transformers
import pandas as pd

class ParaphraseDataset(torch.utils.data.Dataset):
    def __init__(self, table: pd.DataFrame, tokenizer: transformers.PreTrainedTokenizer, maxlen: int):
        """_summary_

        Args:
            table (pd.DataFrame): Dataframe with paraphraze pairs
            tokenizer (transformers.PreTrainedTokenizer): Autotokenizer
            maxlen (int): Max len for token list
        """
        super().__init__()
        
        self.q1 = table["question1"]
        self.q2 = table["question2"]
        
        self.label = table["is_duplicate"]
        
        self.tokenizer = tokenizer
        self.maxlen = maxlen
    
    def __len__(self):
        return len(self.q1)
    
    def __getitem__(self, index: int):
        q1 = self.q1[index]
        q2 = self.q2[index]
        label = self.label[index]
        
        tokenizer_out = self.tokenizer(
            q1, # first string
            q2, # second string
            return_tensors="pt", # return torch tensor
            padding="max_length", # Pad seqeunces
            max_length=self.maxlen, # Max len for padded seq
            truncation=True, # Truncate string
            return_token_type_ids=True, # Return mask for q1 and q2
        )
        
        return {"labels": torch.LongTensor([label]), **tokenizer_out}
    

        
