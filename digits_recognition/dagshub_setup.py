"""
Dagshub setup module
"""
import os

import dagshub
from dotenv import load_dotenv


def dagshub_setup():
    """
    Initializes dagshub with variables read from .env file
    """
    load_dotenv('.env', override=True)

    repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
