import pytest
import torch
from torchvision.transforms import functional as F

from digits_recognition.experimentation.modeling.dataset import (
    rotation_transform,
    gaussian_blur_transform,
    data_augmentation
)
from digits_recognition.experimentation.modeling.evaluate import setup_components
from digits_recognition.inference import infer_labels
from digits_recognition.experimentation.modeling.tests.params import params_test


(
    MODEL_PATH,
    TEST_SET_PATH,
    BATCH_SIZE,
    RANDOM_SEED,
    CLASS_COUNT,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
) = params_test()


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
