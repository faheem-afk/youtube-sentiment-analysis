from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import AdamW
from dataset_ import MyDataSet
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import *



def load_params(params_path: str) -> dict:
    try: 
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info("params loaded safely")
            return params
    except Exception as e:
        raise CustomException(e)
    

from torch.utils.data import Dataset

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
            return_tensors='pt' # return pytorch tensor
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        return {
            "input_ids" : input_ids,
            "labels": self.label[idx],
            "attention_mask": attention_mask
        }
        
    
def bert_modelling():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")    

    current_dir_name = os.path.dirname(os.path.abspath(__file__))
    train_data = load_data('data/processed/train_data.csv').dropna()

    train_ds = MyDataSet(train_data, tokenizer, max_len=128)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=32, drop_last=True)

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to('mps')

    optimizer = AdamW(
        model.parameters(),       
        lr=2e-5,                   
        betas=(0.9, 0.999),        
        eps=1e-8,                  
        weight_decay=0.01          
    )

    total_loss = 0.0
    for epoch in range(3):
        model.train()
        train_epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            batch = {k: v.to('mps')  for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_epoch_loss += loss.item() * batch['input_ids'].size(0)
        total_loss += train_epoch_loss / len(train_ds)
            
        print(f"""Avg train_loss per sample for epoch {epoch+1}:{train_epoch_loss / len(train_ds)}""")
            
    print(f"Avg Train_loss per Sample per epoch: {total_loss / 10}")
    
    torch.save(model.state_dict(), "model/model_mps.pth")
    

if __name__ == "__main__":
    bert_modelling()
    
    
    
    