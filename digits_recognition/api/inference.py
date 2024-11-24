"""
Code for processing the input, querying the model and then processing the outputs.
"""
import torch
import cv2

from digits_recognition.evaluation_inference import infer_labels
from digits_recognition.training_inference import infer_logits
from digits_recognition.api.preprocess_input import find_digits


def compute_predictions(model, device, image):
    """
    Queries the model for labels in an image.
    Returns a list of pairs [coordinates, labels]
    """
    coords, digits = find_digits(image)
    predictions = infer_labels(model, device, digits)

    return list(zip(coords, predictions.cpu().numpy().tolist()))


def annotate_image_with_predictions(model, device, image):
    """
    Queries the model for labels in an image and then annotates them in the image itself.
    """
    coords, predictions = compute_predictions(model, device, image)
    annotated_image = image.copy()

    for idx, (x, y) in enumerate(coords):
        label = f"{predictions[idx]}"
        cv2.putText(
            annotated_image,
            label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    return annotated_image


def compute_probabilities(model, device, image):
    """
    Queries the model for labels in an image.
    Returns a list of pairs [coordinates, probabilities]
    """
    coords, digits = find_digits(image)
    logits = infer_logits(model, device, digits)
    probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)

    return list(zip(coords, probabilities.cpu().detach().numpy().tolist()))
