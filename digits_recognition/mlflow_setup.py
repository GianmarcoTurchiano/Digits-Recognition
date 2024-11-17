import os
import mlflow
import dagshub
from dotenv import load_dotenv

def mlflow_setup():
    load_dotenv('.env')

    repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    experiment_name = os.getenv('CURRENT_EXPERIMENT_NAME')

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    mlflow.set_experiment(experiment_name)