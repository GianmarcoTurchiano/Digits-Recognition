from digits_recognition.modeling.classifier import DigitClassifier
import torch
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, classification_report
from digits_recognition.load_dataset import load_dataset
import argparse
from tqdm import tqdm
import os
import dagshub
import mlflow
from dotenv import load_dotenv

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-t', '--test_set_path', type=str)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-s', '--random_seed', type=int)
    parser.add_argument('-b', '--batch_size', type=int)

    args = parser.parse_args()

    load_dotenv('.env')

    repo_owner = os.getenv('DAGSHUB_REPO_OWNER')
    repo_name = os.getenv('DAGSHUB_REPO_NAME')
    experiment_name = os.getenv('CURRENT_EXPERIMENT_NAME')

    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)

    torch.manual_seed(args.random_seed)

    test_loader = load_dataset(args.test_set_path, normalize=args.normalize, shuffle=False, batch_size=args.batch_size)

    model = DigitClassifier()
    model.load_state_dict(torch.load(args.model_path))

    model.eval()

    with torch.no_grad():
        all_preds = []
        all_labels = []
        
        for data, labels in tqdm(test_loader):
            logits = model(data)
            
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

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name='Evaluation'):
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("random_seed", args.random_seed)
        mlflow.log_param("normalize", args.normalize)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.log_metric("precision_macro", macro_precision)
        mlflow.log_metric("recall_macro", macro_recall)
        mlflow.log_metric("f1_macro", macro_f1)

        mlflow.log_metric("precision_weighted", weighted_precision)
        mlflow.log_metric("recall_weighted", weighted_recall)
        mlflow.log_metric("f1_weighted", weighted_f1)