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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-t', '--test_set_path', type=str)
    parser.add_argument('-b', '--batch_size', type=int)

    args = parser.parse_args()

    mlflow_setup()

    test_loader = load_dataset(
        args.test_set_path,
        shuffle=False,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitClassifier()
    model.load_state_dict(torch.load(args.model_path, weights_only=True))
    model.to(device)

    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []

        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            _, predicted = torch.max(logits, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    weighted_precision = precision_score(all_labels, all_preds, average='weighted')
    weighted_recall = recall_score(all_labels, all_preds, average='weighted')
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted')

    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    print(classification_report(all_labels, all_preds))

    with mlflow.start_run(run_name='Evaluation'):
        mlflow.log_param("batch_size", args.batch_size)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("precision_macro", macro_precision)
        mlflow.log_metric("recall_macro", macro_recall)
        mlflow.log_metric("f1_macro", macro_f1)

        mlflow.log_metric("precision_weighted", weighted_precision)
        mlflow.log_metric("recall_weighted", weighted_recall)
        mlflow.log_metric("f1_weighted", weighted_f1)
