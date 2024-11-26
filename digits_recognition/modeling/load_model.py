"""
Code for loading the model from file.
"""
import torch
from digits_recognition.modeling.init_model import init_model


def load_model(
    model_path,
    input_height,
    input_width,
    input_channels,
    class_count,
    random_seed=None
):
    """
    Instantiates the model and when a path is provided loads in the parameters.
    """
    model, device = init_model(input_height, input_width, input_channels, class_count, random_seed)

    if model_path:
        model_data = torch.load(model_path, weights_only=True)
        run_id = model_data['run_id']
        model.load_state_dict(model_data['weights'])
    else:
        run_id = None

    model.to(device)

    return model, device, run_id
