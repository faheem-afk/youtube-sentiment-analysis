from utils import *
from sklearn.metrics import classification_report
import json
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification


def eval_():
    
    current_dir_name = os.path.dirname(os.path.abspath(__file__))
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).to('cpu')  
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    
    model.load_state_dict(torch.load(os.path.join(current_dir_name, "../../model/model_cpu.pth")))
    model.eval() 
    
    data = load_data(os.path.join(current_dir_name, '../../data/processed/test_data.csv')).dropna()
    X_test = data['text'].tolist()
    y_test = data['label'].tolist()
    
    preds = []
    for sent in X_test:
        encodings = tokenizer(
                            sent,
                            padding='max_length',
                            truncation=True,
                            max_length=128,
                            return_tensors='pt' 
                            )
        output = model(**encodings)
        probs = output.logits.softmax(dim=1)
        predicted_class = probs.argmax().item()
        # prob = probs.detach().numpy()

        # label_map = {0: "negative", 1: "neutral", 2: "positive"}
        # print(f"Prediction: {label_map[predicted_class]} — {prob.tolist()[0][predicted_class]}")
        preds.append(predicted_class)
    
    report = classification_report(y_test, preds, output_dict=True)    
    
    with open(os.path.join(current_dir_name,'../../reports/scores.json'), 'a') as file:
        json.dump(report, file, indent=2)    
    
    
if __name__ == "__main__":
    eval_()
    