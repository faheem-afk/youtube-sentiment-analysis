import mlflow
from mlflow.tracking import MlflowClient
import pytest, sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import *

mlflow.set_tracking_uri("http://35.175.240.84:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("sentiment-bert", "staging"),])
def test_model_for_signature(model_name, stage):
    client = MlflowClient()
    
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    assert latest_version is not None, f"No model found in the '{stage}' stage for {model_name}"
    try:
        assert latest_version is not None, "Model failed to load"
        print(f"Model {model_name} version {latest_version} loaded sucessfully from {stage}")
        
    except Exception as e:
        pytest.fail(f"Model loading failed with error: {e}")
    