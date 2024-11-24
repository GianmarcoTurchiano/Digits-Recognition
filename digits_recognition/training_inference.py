"""
Code for instantiating and querying the model at train time.
"""
import torch
from digits_recognition.modeling.classifier import DigitClassifier


def setup_model(random_seed=None):
    """
    Instantiates the model and loads it onto the available device.
    """
    if random_seed:
        torch.manual_seed(random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DigitClassifier()

    model.to(device)

    return model, device


def infer_logits(model, device, image):
    """
    Queries the model and outputs a probability distribution for each input.
    """
    image = image.to(device)
    logits = model(image)

    return logits
