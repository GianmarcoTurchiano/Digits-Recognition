"""
Code for loading model from MLFlow registry.
"""
import mlflow
import torch
import yaml
import dagshub


def mlflow_model_setup():
    """
    Loads in the latest version of a model in the MLFlow registry.
    The name of the model has to be specified in the environment variables.
    """
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)

    model_name = params['mlflow']['model_name']
    repo_name = params['dagshub']['repo_name']
    repo_owner = params['dagshub']['repo_owner']

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version)).version

    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    print(f'Model loaded from: "{model_uri}"')

    return model, device
