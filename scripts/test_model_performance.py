import mlflow
from mlflow.tracking import MlflowClient
import pytest
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import *


mlflow.set_tracking_uri("http://35.175.240.84:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("sentiment-bert", "staging"),])
def test_model_for_performance(model_name, stage):
    client = MlflowClient()
    
    try:
        file = open('experiment_info.json', 'r')
        run_id = json.load(file)['run_id']
        client = MlflowClient()
        run = client.get_run(run_id)
        accuracy_threshold = run.data.metrics.get('test_0_recall')
        print(accuracy_threshold)
        assert accuracy_threshold < 1    
        file.close()
        
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
    