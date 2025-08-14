import mlflow
import json


def model_stages(model_name):
    mlflow.set_tracking_uri("http://35.175.240.84:5000")
    mlflow.set_experiment('dvc pipeline')
    model_info = json.load(open('experiment_info.json', 'r'))
    
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_info["artifact_uri"], model_name)
    client.transition_model_version_stage(
        name=model_name,
        version = mv.version,
        stage='Staging'
    )
    
if __name__ == "__main__":
    model_stages('sentiment-bert')