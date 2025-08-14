import os 
import mlflow

def promote_model():
    
    mlflow.set
    
    client = mlflow.MlflowClient()
    
    model_name = 'sentiment-bert'
    latest_version_staging = client.get_latest_versions(model_name, stages=['Staging'])[0].version

    prob_versions = client.get_latest_versions(model_name, stages=['Production'])
    for version in prob_versions:
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Archived"
                
        )
    client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
                
        )
    
if __name__ "__main__":
    promote_model()
    
