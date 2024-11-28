import pytest
import torch
from torchvision.transforms import functional as F
import yaml

from digits_recognition.experimentation.modeling.dataset import (
    rotation_transform,
    gaussian_blur_transform,
    data_augmentation
)
from digits_recognition.experimentation.modeling.evaluate import setup_components
from digits_recognition.infer_labels import infer_labels


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

data_params = params['data']
evaluation_params = params['evaluation']
data_meta_params = data_params['meta']
data_meta_images_params = data_meta_params['images']

MODEL_PATH = params['model']
TEST_SET_PATH = data_params['processed']['test_set']
BATCH_SIZE = evaluation_params['batch_size']
RANDOM_SEED = evaluation_params['random_seed']
CLASS_COUNT = data_meta_params['classes']['count']
IMAGE_WIDTH = data_meta_images_params['width']
IMAGE_HEIGHT = data_meta_images_params['height']
IMAGE_CHANNELS = data_meta_images_params['channels']


@pytest.fixture
def components():
    return setup_components(
        TEST_SET_PATH,
        BATCH_SIZE,
        MODEL_PATH,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CLASS_COUNT,
        RANDOM_SEED
    )


@pytest.mark.parametrize(
    'transformation',
    [
        rotation_transform,
        gaussian_blur_transform,
        data_augmentation
    ],
    ids=[
        'Rotation Transform',
        'Gaussian Blur Transform',
        'Data Augmentation'
    ]
)
def test_invariance(components, transformation):
    model, device, loader, _ = components

    total_cases = 0
    invariant_cases = 0

    for images, _ in loader:
        pil_images = [F.to_pil_image(image, mode='L') for image in images]

        # Apply the transformation to each image
        transformed_images = [transformation(pil_image) for pil_image in pil_images]

        # Convert the list of transformed images back to tensors
        transformed_images = torch.stack([F.to_tensor(transformed_image) for transformed_image in transformed_images])

        preds_original = infer_labels(model, device, images)
        preds_transformed = infer_labels(model, device, transformed_images)

        total_cases += preds_original.size(0)
        invariant_cases += torch.sum(preds_original == preds_transformed).item()

    threshold = 95

    invariance_percentage = (invariant_cases / total_cases) * 100

    assert invariance_percentage >= threshold, \
        f"Invariance below threshold ({threshold}%): {invariance_percentage:.2f}%"
