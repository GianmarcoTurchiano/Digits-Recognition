import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import torch.optim as optim
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm
from digits_recognition.dataset import load_dataset
import dagshub
import mlflow
import hashlib

class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-p', '--patience', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-w', '--weight_decay', type=float)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--random_seed', type=int)
    parser.add_argument('-b', '--batch_size', type=int)

    args = parser.parse_args()

    data = load_dataset(args.dataset_path)

    torch.manual_seed(args.random_seed)

    X_train = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train = torch.tensor(data['y_train'], dtype=torch.long)
    X_val = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val = torch.tensor(data['y_val'], dtype=torch.long)

    if args.normalize:
        X_train = X_train / 255.0
        X_val = X_val / 255.0

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitClassifier().to(device)

    # Create a unique identifier based on model and args
    model_id = hashlib.sha256(str(model).encode()).hexdigest()[:8]
    args_id = hashlib.sha256(str(args).encode()).hexdigest()[:8]
    experiment_name = f"{model._get_name()} ({model_id}, {args_id})"

    dagshub.init(repo_owner="GianmarcoTurchiano", repo_name="Digits-Recognition", mlflow=True)
    mlflow.set_experiment(experiment_name)

    mlflow.start_run(run_name="training")
    
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("weight_decay", args.weight_decay)
    mlflow.log_param("random_seed", args.random_seed)
    mlflow.log_param("normalize", args.normalize)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = PolynomialLR(optimizer, total_iters=args.epochs, power=2.0)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in tqdm(range(1, args.epochs+1), desc="Epochs"):
        # Training loop

        model.train()

        tqdm.write(f"Learning rate: {scheduler.get_last_lr()}")
        train_loss = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}], Training", leave=False):
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

        model.eval()

        val_loss = 0

        for images, labels in tqdm(val_loader, desc=f"Epoch [{epoch}/{args.epochs}], Validation", leave=False):
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
    
    mlflow.end_run()