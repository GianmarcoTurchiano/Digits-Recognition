"""
MLFlow Setup module
"""
import os

import dagshub
import mlflow
from dotenv import load_dotenv
import torch


def _dagshub_setup():
    load_dotenv('.env', override=True)

    repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)


def mlflow_experiment_setup():
    """
    Prepares all that it is necessary for logging experiments with MLFlow
    """
    _dagshub_setup()

    experiment_name = os.getenv('CURRENT_EXPERIMENT_NAME')

    mlflow.set_experiment(experiment_name)


def mlflow_model_setup():
    """
    Loads in the latest version of a model in the MLFlow registry.
    The name of the model has to be specified in the environment variables.
    """
    _dagshub_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = os.getenv('MODEL_NAME')
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version)).version

    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    return model, device
