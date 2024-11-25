"""
Code for loading model from MLFlow registry.
"""
import os

import mlflow
import torch

from digits_recognition.dagshub_setup import dagshub_setup


def mlflow_model_setup():
    """
    Loads in the latest version of a model in the MLFlow registry.
    The name of the model has to be specified in the environment variables.
    """
    dagshub_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = os.getenv('MODEL_NAME')
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version)).version

    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.pytorch.load_model(model_uri, map_location=device)

    print(f'Model loaded from: "{model_uri}"')

    return model, device
