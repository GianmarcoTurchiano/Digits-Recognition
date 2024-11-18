import pytest
import torch
from digits_recognition.modeling.classifier import DigitClassifier
from digits_recognition.load_dataset import (
    load_dataset,
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
    data_augmentation
)

MODEL_PATH = r'./models/digit_classifier.pth'
TEST_SET_PATH = r'./data/processed/test_set.pkl'
BATCH_SIZE = 64


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def model(device):
    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    model.to(device)

    return model


@pytest.fixture
def data_loader():
    loader = load_dataset(TEST_SET_PATH, shuffle=False, batch_size=BATCH_SIZE)

    return loader


@pytest.mark.parametrize('transformation', [
    rotation_transform,
    gaussian_blur_transform,
    resize_crop_transform,
    data_augmentation
])
def test_invariance(model, device, data_loader, transformation):
    for images, _ in data_loader:
        images = images.to(device)
        transformed_images = transformation(images).to(device)

        model_prediction_original = model(images)
        predicted_label_original = torch.argmax(model_prediction_original, dim=1).item()

        model_prediction_rotated = model(transformed_images)
        predicted_label_rotated = torch.argmax(model_prediction_rotated, dim=1).item()

        assert (predicted_label_original == predicted_label_rotated), \
            f"Prediction differs for original and transformed image ({transformation})"
