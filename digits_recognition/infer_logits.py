"""
Code for querying logits to the model at train time.
"""


def infer_logits(model, device, image):
    """
    Queries the model and outputs logits for each input.
    """
    image = image.to(device)
    logits = model(image)

    return logits
