"""
Code for loading the model from file.
"""
import torch
from digits_recognition.modeling.init_model import init_model


def load_model(model_path, random_seed=None):
    """
    Instantiates the model and when a path is provided loads in the parameters.
    """
    model, device = init_model(random_seed)

    if model_path:
        model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)

    return model, device
