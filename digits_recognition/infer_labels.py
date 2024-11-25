"""
Code for querying labels to the model.
"""
import torch
from digits_recognition.infer_logits import infer_logits


def infer_labels(model, device, image):
    """
    Queries the model and outputs a label for each input.
    """
    logits = infer_logits(model, device, image)
    _, predicted = torch.max(logits, 1)

    return predicted
