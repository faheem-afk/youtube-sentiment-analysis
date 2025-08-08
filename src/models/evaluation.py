import joblib
from utils import *
from sklearn.metrics import classification_report
import json

def eval_():
    
    current_dir_name = os.path.dirname(os.path.abspath(__file__))
    best_model = joblib.load(os.path.join(current_dir_name, '../../model/model.joblib'))
    X_test = joblib.load(os.path.join(current_dir_name,'../../data/vectorized/test_vec.joblib'))
    test_x_y = load_data(os.path.join(current_dir_name,'../../data/processed/test_data.csv')).dropna()
    y_test = test_x_y['label']
    y_pred = best_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)    
    
    
    with open(os.path.join(current_dir_name,'../../reports/scores.json'), 'a') as file:
        json.dump(report, file, indent=2)    
    
    
if __name__ == "__main__":
    eval_()
    