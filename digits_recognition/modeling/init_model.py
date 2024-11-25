"""
Code for instantiating the model.
"""
import torch
from digits_recognition.modeling.classifier import DigitClassifier


def init_model(random_seed=None):
    """
    Instantiates the model and loads it onto the available device.
    """
    if random_seed:
        torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitClassifier()

    model.to(device)

    return model, device
