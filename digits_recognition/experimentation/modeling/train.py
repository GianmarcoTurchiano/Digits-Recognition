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
import dagshub
from codecarbon import EmissionsTracker

from digits_recognition.experimentation.modeling.dataset import get_data_loader
from digits_recognition.inference import infer_logits
from digits_recognition.experimentation.modeling.classifier import init_model


def training_step(model, loader, device, optimizer, criterion):
    """
    Training step, called once per epoch.
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
    Validation step, called once per epoch.
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
    input_height, input_width, input_channels, class_count,
    data_augmentation=True,
    random_seed=None,
):
    """
    Initializes and returns components that are required for training.
    """
    model, device = init_model(
        input_height,
        input_width,
        input_channels,
        class_count,
        random_seed
    )

    loader = get_data_loader(
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
    input_height, input_width, input_channels, class_count,
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
        input_height, input_width, input_channels, class_count,
        train_data_augmentation,
        random_seed,
    )

    val_loader = get_data_loader(
        val_set_path,
        batch_size=batch_size,
        device=device,
    )

    return model, train_loader, val_loader, device, optimizer, criterion, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_channels', type=int)
    parser.add_argument('--image_width', type=int)
    parser.add_argument('--image_height', type=int)
    parser.add_argument('--classes', type=int)
    parser.add_argument('--repo_owner', type=str)
    parser.add_argument('--repo_name', type=str)
    parser.add_argument('--experiment_name', type=str)

    parser.add_argument('--train_set_path', type=str)
    parser.add_argument('--val_set_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--random_seed', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--polynomial_scheduler_power', type=float)
    parser.add_argument('--train_data_augmentation', action='store_true')
    parser.add_argument('--emissions_path', type=str)

    args = parser.parse_args()

    dagshub.init(repo_owner=args.repo_owner, repo_name=args.repo_name, mlflow=True)
    mlflow.set_experiment(args.experiment_name)

    run = mlflow.start_run()

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
        args.image_height, args.image_width, args.image_channels, args.classes,
        args.train_data_augmentation,
        args.random_seed
    )

    tracker = EmissionsTracker(
        project_name='Training',
        save_to_file=True,
        output_file=args.emissions_path,
        log_level='critical'
    )

    tracker.start()

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
            torch.save({
                'weights': model.state_dict(),
                'run_id': run.info.run_id
            }, args.model_path)
            mlflow.log_metric("Output validation loss", avg_val_loss, step=epoch)
            mlflow.log_metric("Output training loss", avg_train_loss, step=epoch)
            tqdm.write("Best model weights have been saved.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve == args.patience:
            tqdm.write(f"Early stopping triggered after {epoch} epochs.")
            break

        scheduler.step()

    tracker.stop()

    model_data = torch.load(args.model_path, weights_only=True)

    model.load_state_dict(model_data['weights'])
    mlflow.pytorch.log_model(model, "model")

    mlflow.end_run()
