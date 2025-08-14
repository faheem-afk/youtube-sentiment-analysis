import mlflow
from mlflow.tracking import MlflowClient
import pytest
import json
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import *
import requests

BASE_URL = "http://localhost:5002"


def test_model_endpoint():
    
    data = {"comments":
    [
        {
        "text":    "this is weird",
        "timestamp": "2025-08-13T14:27:45Z"
        }
    ]
}
    
    response = requests.post(f"{BASE_URL}/predict", json=data)
    
    print(response.json())
    assert response.status_code == 200
    assert isinstance(response.json(), dict)
    
