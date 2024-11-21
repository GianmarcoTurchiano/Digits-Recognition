"""
MLFlow Setup module
"""
import os

import dagshub
import mlflow
from dotenv import load_dotenv


def mlflow_setup():
    """
    Prepares all that it is necessary for logging experiments with MLFlow
    """
    load_dotenv('.env', override=True)

    repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    experiment_name = os.getenv('CURRENT_EXPERIMENT_NAME')

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.set_experiment(experiment_name)
