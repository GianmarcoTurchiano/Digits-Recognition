"""
Script for training the classifier.
"""
import argparse

import mlflow
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm

from digits_recognition.load_dataset import load_dataset
from digits_recognition.mlflow_setup import mlflow_setup
from digits_recognition.modeling.classifier import DigitClassifier


def train_step(model, loader, device, optimizer, criterion):
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
    Training step, called once each epoch.
    """
    model.eval()

    val_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(
            loader,
            desc="Validation",
            leave=False
        ):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)

            loss = criterion(logits, labels)

            val_loss += loss.item()

    return val_loss / len(loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train_set_path', type=str)
    parser.add_argument('-v', '--val_set_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-pat', '--patience', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-w', '--weight_decay', type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--random_seed', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-pow', '--polynomial_scheduler_power', type=float)

    args = parser.parse_args()

    torch.manual_seed(args.random_seed)

    train_loader = load_dataset(
        args.train_set_path,
        normalize=args.normalize,
        shuffle=True,
        batch_size=args.batch_size
    )
    val_loader = load_dataset(
        args.val_set_path,
        normalize=args.normalize,
        shuffle=False,
        batch_size=args.batch_size
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    digit_classifier = DigitClassifier().to(device)

    mlflow_setup()

    mlflow.start_run(run_name="Training")

    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("weight_decay", args.weight_decay)
    mlflow.log_param("random_seed", args.random_seed)
    mlflow.log_param("normalize", args.normalize)
    mlflow.log_param("polynomial_scheduler_power", args.polynomial_scheduler_power)

    # Loss and optimizer
    cross_entropy_criterion = nn.CrossEntropyLoss()
    adam_optimizer = optim.Adam(
        digit_classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    polynomial_scheduler = PolynomialLR(
        adam_optimizer,
        total_iters=args.epochs,
        power=args.polynomial_scheduler_power
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        [last_lr] = polynomial_scheduler.get_last_lr()

        mlflow.log_metric("Learning rate", last_lr, step=epoch)
        tqdm.write(f"Learning rate: {last_lr}")

        avg_train_loss = train_step(
            digit_classifier,
            train_loader,
            device,
            adam_optimizer,
            cross_entropy_criterion
        )

        mlflow.log_metric("Train loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

        avg_val_loss = validation_step(
            digit_classifier,
            val_loader,
            device,
            cross_entropy_criterion
        )

        mlflow.log_metric("Validation loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(digit_classifier.state_dict(), args.model_path)
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

        polynomial_scheduler.step()

    mlflow.log_artifact(args.model_path)
    mlflow.end_run()
