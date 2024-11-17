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

    model = DigitClassifier().to(device)

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = PolynomialLR(
        optimizer,
        total_iters=args.epochs,
        power=args.polynomial_scheduler_power
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        # Training loop

        model.train()

        [last_lr] = scheduler.get_last_lr()

        mlflow.log_metric("Learning rate", last_lr, step=epoch)
        tqdm.write(f"Learning rate: {last_lr}")
        train_loss = 0

        for images, labels in tqdm(
            train_loader,
            desc=f"Epoch [{epoch}/{args.epochs}], Training",
            leave=False
        ):
            X = images.to(device)
            y = labels.to(device)

            optimizer.zero_grad()

            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        mlflow.log_metric("Train loss", avg_train_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Train Loss: {avg_train_loss}")

        # Validation loop

        val_loss = 0

        model.eval()

        with torch.no_grad():
            for images, labels in tqdm(
                val_loader,
                desc=f"Epoch [{epoch}/{args.epochs}], Validation",
                leave=False
            ):
                X = images.to(device)
                y = labels.to(device)

                logits = model(X)

                loss = criterion(logits, y)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        mlflow.log_metric("Validation loss", avg_val_loss, step=epoch)
        tqdm.write(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_path)
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

        scheduler.step()

    mlflow.log_artifact(args.model_path)

    mlflow.end_run()
