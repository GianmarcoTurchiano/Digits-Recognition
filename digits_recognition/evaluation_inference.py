"""
Code for loading and querying the model at test time.
"""
import torch
from digits_recognition.training_inference import infer_logits, setup_model


def load_model(model_path, random_seed=None):
    """
    Instantiates the model and when a path is provided loads in the parameters.
    """
    model, device = setup_model(random_seed)

    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)

    return model, device


def infer_labels(model, device, image):
    """
    Queries the model and outputs a label for each input.
    """
    logits = infer_logits(model, device, image)
    _, predicted = torch.max(logits, 1)

    return predicted
