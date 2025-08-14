import pandas as pd
import sys
import os
from utils import *

def data_ing():
    
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
  
    df = load_data("https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv")
    df = df.dropna().drop_duplicates()
    
    df['category']= df['category'].map({-1:0, 0:1, 1:2})
    df.rename(columns = {"clean_comment": "text", "category": "label"}, inplace=True)

    save_data(df, os.path.join(current_dir_path, "../../data/raw/data.csv"))
    
    
if __name__ == "__main__":
    data_ing()


