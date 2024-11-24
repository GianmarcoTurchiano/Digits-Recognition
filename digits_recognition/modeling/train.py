"""
Script for training the classifier.
"""
import argparse
import os

import mlflow
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm
from dotenv import set_key

from digits_recognition.load_dataset import load_dataset
from digits_recognition.mlflow_setup import mlflow_setup
from digits_recognition.training_inference import infer_logits, setup_model


def training_step(model, loader, device, optimizer, criterion):
    """
    Training step, called once each epoch.
    """
    model.train()

    train_loss = 0

    for images, labels in tqdm(
        loader,
        desc="Training",
        leave=False
    ):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()

        train_loss += loss.item()

    return train_loss / len(loader)


def validation_step(model, loader, device, criterion):
    """
    Validation step, called once each epoch.
    """
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            labels = labels.to(device)

            logits = infer_logits(model, device, images)

            loss = criterion(logits, labels)

            val_loss += loss.item()

    return val_loss / len(loader)


def setup_training_components(
    train_set_path,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    polynomial_scheduler_power,
    data_augmentation=True,
    random_seed=None,
):
    """
    Initializes and returns components that are required for training.
    """
    model, device = setup_model(random_seed)

    loader = load_dataset(
        train_set_path,
        shuffle=True,
        batch_size=batch_size,
        augment=data_augmentation,
        device=device,
        num_workers=8,
        persistent_workers=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = PolynomialLR(
        optimizer,
        total_iters=epochs,
        power=polynomial_scheduler_power
    )

    return model, loader, device, optimizer, criterion, scheduler


def setup_components(
    train_set_path,
    val_set_path,
    batch_size,
    learning_rate,
    weight_decay,
    epochs,
    polynomial_scheduler_power,
    train_data_augmentation=False,
    random_seed=None,
):
    """
    Initializes and returns components that are required for training and for validation.
    """
    model, train_loader, device, optimizer, criterion, scheduler = setup_training_components(
        train_set_path,
        batch_size,
        learning_rate,
        weight_decay,
        epochs,
        polynomial_scheduler_power,
        train_data_augmentation,
        random_seed,
    )

    val_loader = load_dataset(
        val_set_path,
        batch_size=batch_size,
        device=device,
    )

    return model, train_loader, val_loader, device, optimizer, criterion, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train_set_path', type=str)
    parser.add_argument('-v', '--val_set_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-pat', '--patience', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-w', '--weight_decay', type=float)
    parser.add_argument('-s', '--random_seed', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-pow', '--polynomial_scheduler_power', type=float)
    parser.add_argument('-a', '--train_data_augmentation', action='store_true')

    args = parser.parse_args()

    mlflow_setup()
    os.environ.pop('MLFLOW_RUN_ID', None)

    run = mlflow.start_run()
    set_key('.env', 'MLFLOW_RUN_ID', run.info.run_id)

    mlflow.log_param("Epochs", args.epochs)
    mlflow.log_param("Initial learning rate", args.learning_rate)
    mlflow.log_param("Batch size", args.batch_size)
    mlflow.log_param("Patience", args.patience)
    mlflow.log_param("Weight decay", args.weight_decay)
    mlflow.log_param("Random seed", args.random_seed)
    mlflow.log_param("Polynomial scheduler power", args.polynomial_scheduler_power)
    mlflow.log_param("Train data augmentation", args.train_data_augmentation)

    model, train_loader, val_loader, device, optimizer, criterion, scheduler = setup_components(
        args.train_set_path,
        args.val_set_path,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.epochs,
        args.polynomial_scheduler_power,
        args.train_data_augmentation,
        args.random_seed
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        [last_lr] = scheduler.get_last_lr()

        mlflow.log_metric("Learning rate", last_lr, step=epoch)
        tqdm.write(f"Learning rate: {last_lr}")

        avg_train_loss = training_step(
            model,
            train_loader,
            device,
            optimizer,
            criterion
        )

        mlflow.log_metric("Training loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

        avg_val_loss = validation_step(
            model,
            val_loader,
            device,
            criterion
        )

        mlflow.log_metric("Validation loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_path)
            mlflow.log_metric("Output validation loss", avg_val_loss, step=epoch)
            mlflow.log_metric("Output training loss", avg_train_loss, step=epoch)
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

        scheduler.step()

    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()
