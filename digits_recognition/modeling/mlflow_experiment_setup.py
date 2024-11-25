"""
Code for setting up an experiment on MLFlow.
"""
import os
import mlflow

from digits_recognition.dagshub_setup import dagshub_setup


def mlflow_experiment_setup():
    """
    Prepares all that it is necessary for logging experiments with MLFlow
    """
    dagshub_setup()

    experiment_name = os.getenv('CURRENT_EXPERIMENT_NAME')

    mlflow.set_experiment(experiment_name)
