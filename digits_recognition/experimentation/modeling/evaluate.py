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
import dagshub
from codecarbon import EmissionsTracker

from digits_recognition.experimentation.modeling.classifier import init_model
from digits_recognition.experimentation.modeling.dataset import get_data_loader
from digits_recognition.inference import infer_labels


def load_model(
    model_path,
    input_height,
    input_width,
    input_channels,
    class_count,
    random_seed=None
):
    """
    Instantiates the model and when a path is provided loads in the parameters.
    """
    model, device = init_model(input_height, input_width, input_channels, class_count, random_seed)

    if model_path:
        model_data = torch.load(model_path, weights_only=True)
        run_id = model_data['run_id']
        model.load_state_dict(model_data['weights'])
    else:
        run_id = None

    model.to(device)

    return model, device, run_id


def inference_step(model, device, loader):
    """
    Inference step, executed once per epoch.
    """
    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []

        for images, labels in tqdm(loader):
            labels = labels.to(device)

            predicted = infer_labels(model, device, images)

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


def setup_components(
    test_set_path,
    batch_size,
    model_path,
    input_height,
    input_width,
    input_channels,
    class_count,
    random_seed=None
):
    """
    Initializes and returns components that are required for evaluation.
    """
    model, device, run_id = load_model(
        model_path,
        input_height,
        input_width,
        input_channels,
        class_count,
        random_seed
    )

    loader = get_data_loader(
        test_set_path,
        batch_size=batch_size,
        device=device
    )

    return model, device, loader, run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--test_set_path', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--image_channels', type=int)
    parser.add_argument('--image_width', type=int)
    parser.add_argument('--image_height', type=int)
    parser.add_argument('--classes', type=int)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--emissions_path', type=str)

    args = parser.parse_args()

    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    model, device, test_loader, run_id = setup_components(
        args.test_set_path,
        args.batch_size,
        args.model_path,
        args.image_height, args.image_width, args.image_channels, args.classes
    )

    with mlflow.start_run(run_id=run_id):
        tracker = EmissionsTracker(
            project_name='Inference',
            save_to_file=True,
            output_file=args.emissions_path,
            log_level='critical'
        )

        tracker.start()

        all_labels, all_preds = inference_step(model, device, test_loader)

        tracker.stop()

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
