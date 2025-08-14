import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from utils import *
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import mlflow
from dataset_ import MyDataSet
from torch.utils.data import DataLoader


def eval_():
    
    current_dir_name = os.path.dirname(os.path.abspath(__file__))
    param_path = os.path.join(current_dir_name, '../../params.yaml')
    params = load_params(param_path)
    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to('cpu')  
    
    model = mlflow.pytorch.load_model("s3://ysa-bucketv1/659050820026623362/models/m-b0fd8c06eeef45fa8c1cb1d19f67329e/artifacts").to('cpu')
    # model.load_state_dict(torch.load(os.path.join(current_dir_name, "../../model/model_cpu.pth")))
    model.eval() 
    data = load_data(os.path.join(current_dir_name, '../../data/processed/test_data.csv')).dropna()
    y_test = data.iloc[:, -1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    
    test_ds = MyDataSet(data, tokenizer, max_len=128)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=16)
    

    preds = []
    with torch.no_grad():
        for batch in test_loader:
            outputs = model(**batch)
            probs = outputs.logits.softmax(dim=1)
            preds.extend(probs.argmax(dim=1).numpy())
    
    report = classification_report(y_test, preds, output_dict=True)   

    
    mlflow.set_tracking_uri("http://35.175.240.84:5000")
    mlflow.set_experiment('dvc pipeline')
    with  mlflow.start_run() as run:
        for label, score_dic in report.items():
            if isinstance(score_dic, dict):
                mlflow.log_metrics(
                    {
                        f"test_{label}_precision": score_dic['precision'],
                        f"test_{label}_recall": score_dic['recall'],
                        f"test_{label}_f1-score": score_dic['f1-score'],
                     
                     }    
                )
        for key, value in model.config.to_dict().items():
            mlflow.log_param(key, value)
        
        for _, attr in params.items():
            for key, value in attr.items():
                mlflow.log_param(key, value)  
        
        mlflow.pytorch.log_model(
        pytorch_model=model,
        name='model',                    
        input_example=None,
        signature = None,
        )
        
        artifact_uri = mlflow.get_artifact_uri()
        save_artifact_info(artifact_uri, run.info.run_id, 'experiment_info.json')
    
    
if __name__ == "__main__":
    eval_()
    