import json
import os
import pandas as pd
import yaml 
from logger_ import logger
from exp import CustomException
import joblib
from transformers import AutoTokenizer


def load_data(path: str):
    try:
        data = pd.read_csv(path)
        logger.info(f"Data Loaded from the given url: {path}")
        return data
    
    except Exception as e:
        raise CustomException(e)


def load_params(params_path: str) -> dict:
    try: 
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            logger.info("params loaded safely")
            return params
    except Exception as e:
        raise CustomException(e)
    

def save_data(data, path: str) -> None:
    try: 
        dir = os.path.dirname(path)
        file_name = path.split('/')[-1]
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, file_name)
        if isinstance(data, pd.DataFrame):
            data.to_csv(file_path, index=False)
        else:
            joblib.dump(data, file_path)
        logger.info(f"Saving data to {file_path}")
    except Exception as e:
        raise CustomException(e)
    

def save_model(model, path) -> None:
    try:
        dir = os.path.dirname(path)
        file_name = path.split('/')[-1]
        os.makedirs(dir, exist_ok=True)
        file_path = os.path.join(dir, file_name)
        joblib.dump(model, file_path)
        logger.info(f"Saving model to {file_path}")
    except Exception as e:
        raise CustomException(e)
    
    

def save_artifact_info(artifact_uri, run_id, file):
    with open(file, 'w') as f:
        json.dump({
            'run_id': run_id,
            'artifact_uri': artifact_uri
        }, f, indent=2)
        


def get_encoding(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    inputs = tokenizer(text, 
                       return_tensors="pt", 
                       truncation=True, 
                       padding='max_length')

    return inputs