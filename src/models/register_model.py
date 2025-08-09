import mlflow
import json


def model_register(model_name):
    mlflow.set_tracking_uri("http://54.162.237.251:5000")
    mlflow.set_experiment('dvc pipeline')
    model_info = json.load(open('experiment_info.json', 'r'))
    model_uri = f"{model_info['artifact_uri']}"
    
    registered_model = mlflow.register_model(model_uri, model_name)
    
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version = registered_model.version,
        stage='Staging'
    )
    
if __name__ == "__main__":
    model_register('yt_chrome_plugin_model')