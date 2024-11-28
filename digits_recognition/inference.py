"""
Code for querying the model.
"""
import torch


def infer_logits(model, device, image):
    """
    Queries the model and outputs logits for each input.
    """
    image = image.to(device)
    logits = model(image)

    return logits


def infer_labels(model, device, image):
    """
    Queries the model and outputs a label for each input.
    """
    logits = infer_logits(model, device, image)
    _, predicted = torch.max(logits, 1)

    return predicted
