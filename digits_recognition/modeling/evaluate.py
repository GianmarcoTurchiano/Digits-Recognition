"""
Script for evaluating the trained classifier.
"""
import argparse

import mlflow
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

from digits_recognition.load_dataset import load_dataset
from digits_recognition.mlflow_setup import mlflow_setup
from digits_recognition.modeling.classifier import DigitClassifier


def inference_step(model, device, loader):
    """
    Inference step, executed once per epoch.
    """
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []

        for images, labels in tqdm(loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def evaluation_step(all_labels, all_preds):
    """
    Evaluation step, executed once per epoch.
    """
    accuracy = accuracy_score(all_labels, all_preds)

    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(classification_report(all_labels, all_preds))

    return (
        accuracy,
        weighted_precision,
        weighted_recall,
        weighted_f1,
        macro_precision,
        macro_recall,
        macro_f1
    )


def setup_components(test_set_path, batch_size, model_path):
    """
    Initializes and returns components that are required for evaluation.
    """
    mlflow_setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = load_dataset(
        test_set_path,
        batch_size=batch_size,
        device=device
    )

    model = DigitClassifier()

    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)

    return model, device, loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-t', '--test_set_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int)

    args = parser.parse_args()

    model, device, test_loader = setup_components(
        args.test_set_path,
        args.batch_size,
        args.model_path
    )

    with mlflow.start_run():
        all_labels, all_preds = inference_step(model, device, test_loader)

        (
            accuracy,
            weighted_precision,
            weighted_recall,
            weighted_f1,
            macro_precision,
            macro_recall,
            macro_f1
        ) = evaluation_step(all_labels, all_preds)

        mlflow.log_param("Batch size", args.batch_size)

        mlflow.log_metric("Accuracy", accuracy)

        mlflow.log_metric("Precision macro", macro_precision)
        mlflow.log_metric("Recall macro", macro_recall)
        mlflow.log_metric("F1 macro", macro_f1)

        mlflow.log_metric("Precision weighted", weighted_precision)
        mlflow.log_metric("Recall weighted", weighted_recall)
        mlflow.log_metric("F1 weighted", weighted_f1)
