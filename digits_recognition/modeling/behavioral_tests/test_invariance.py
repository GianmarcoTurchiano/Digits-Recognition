import pytest
import torch
from torchvision.transforms import functional as F

from digits_recognition.load_dataset import (
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
    data_augmentation
)
from digits_recognition.modeling.evaluate import setup_components


MODEL_PATH = r'./models/digit_classifier.pth'
TEST_SET_PATH = r'./data/processed/test_set.pkl'
BATCH_SIZE = 64


@pytest.fixture
def components():
    return setup_components(
        TEST_SET_PATH,
        BATCH_SIZE,
        MODEL_PATH
    )


@pytest.mark.parametrize('transformation', [
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
    data_augmentation
])
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

        # assert (torch.equal(predicted_label_original, predicted_label_transformed)), \
        #     f"Prediction differs for original and transformed image ({transformation})"

        total_cases += predicted_label_original.size(0)
        invariant_cases += torch.sum(predicted_label_original == predicted_label_transformed).item()

    threshold = 100

    invariance_percentage = (invariant_cases / total_cases) * 100

    assert invariance_percentage >= threshold, \
        f"Invariance below threshold ({threshold}%): {invariance_percentage:.2f}%"
