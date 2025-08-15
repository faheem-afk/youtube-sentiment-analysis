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
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient


def eval_():
    
    # current_dir_name = os.path.dirname(os.path.abspath(__file__))
    param_path = 'params.yaml'
    params = load_params(param_path)
    
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to('cpu')
    
    model.load_state_dict(torch.load("model/model_mps.pth", map_location="cpu"))
    model.eval()
    
    data = load_data('data/processed/test_data.csv').dropna()
    y_test = data.iloc[:, -1]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    
    test_ds = MyDataSet(data, tokenizer, max_len=128)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=32)
    
    preds = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
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


        input_examples = data['text'][: 2].values.tolist()
        encoded_inputs = get_encoding(input_examples)
        signature = infer_signature(input_examples, model(**encoded_inputs).logits.detach().numpy().tolist())
        
        mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path='model',                    
        signature = signature,
        )
        
        client = MlflowClient()

        local_path = client.download_artifacts(run.info.run_id, "model/MLmodel")
        with open(local_path, "r") as f:
            mlmodel_data = yaml.safe_load(f)

        artifact_path = mlmodel_data.get("artifact_path")
        save_artifact_info(artifact_path, run.info.run_id, 'experiment_info.json')
    
    
if __name__ == "__main__":
    eval_()
    