from torch.utils.data import Dataset
import torch

class MyDataSet(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.text = data['text'].tolist()
        self.label = data['label'].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.text[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt' 
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        return {
            "input_ids" : input_ids,
            "labels": torch.tensor(self.label[idx], dtype=torch.long),
            "attention_mask": attention_mask
        }