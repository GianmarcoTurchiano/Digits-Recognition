import pytest
import torch
from torchvision.transforms import functional as F
import yaml

from digits_recognition.modeling.dataset import (
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
    data_augmentation
)
from digits_recognition.modeling.evaluate import setup_components


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

MODEL_PATH = params['model']
TEST_SET_PATH = params['data']['processed']['test_set']
BATCH_SIZE = params['evaluation']['batch_size']
RANDOM_SEED = params['evaluation']['random_seed']


@pytest.fixture
def components():
    return setup_components(
        TEST_SET_PATH,
        BATCH_SIZE,
        MODEL_PATH,
        RANDOM_SEED
    )


@pytest.mark.parametrize(
    'transformation',
    [
        rotation_transform,
        gaussian_blur_transform,
        resize_crop_transform,
        data_augmentation
    ],
    ids=[
        'Rotation Transform',
        'Gaussian Blur Transform',
        'Resize and Crop Transform',
        'Data Augmentation'
    ]
)
def test_invariance(components, transformation):
    model, device, loader = components

    total_cases = 0
    invariant_cases = 0

    for images, _ in loader:
        pil_images = [F.to_pil_image(image, mode='L') for image in images]

        # Apply the transformation to each image
        transformed_images = [transformation(pil_image) for pil_image in pil_images]

        # Convert the list of transformed images back to tensors
        transformed_images = torch.stack([F.to_tensor(transformed_image) for transformed_image in transformed_images]).to(device)

        images = images.to(device)

        model_prediction_original = model(images)
        predicted_label_original = torch.argmax(model_prediction_original, dim=1)

        model_prediction_transformed = model(transformed_images)
        predicted_label_transformed = torch.argmax(model_prediction_transformed, dim=1)

        total_cases += predicted_label_original.size(0)
        invariant_cases += torch.sum(predicted_label_original == predicted_label_transformed).item()

    threshold = 95

    invariance_percentage = (invariant_cases / total_cases) * 100

    assert invariance_percentage >= threshold, \
        f"Invariance below threshold ({threshold}%): {invariance_percentage:.2f}%"
