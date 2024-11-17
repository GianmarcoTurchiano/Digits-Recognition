import pytest
import torch
from digits_recognition.modeling.classifier import DigitClassifier
from digits_recognition.load_dataset import load_data_tensors


MODEL_PATH = r'./models/digit_classifier.pth'
TEST_SET_PATH = r'./data/processed/test_set.pkl'
NORMALIZE = False


@pytest.fixture
def model():
    model = DigitClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()

    return model


@pytest.fixture
def sample_data():
    images, _ = load_data_tensors(TEST_SET_PATH, NORMALIZE)

    return images


def add_noise(inputs, noise_level=0.1):
    noise = torch.randn_like(inputs) * noise_level
    return torch.clamp(inputs + noise, 0, 1)  # Keep values in [0, 1]


def flip_horizontal(inputs):
    return torch.flip(inputs, dims=[-1])


def rotate_90(inputs):
    return torch.rot90(inputs, k=1, dims=[-2, -1])


def test_noise_invariance(model, sample_data):
    outputs_original = model(sample_data)
    noisy_data = add_noise(sample_data, noise_level=0.2)
    outputs_noisy = model(noisy_data)
    assert torch.allclose(
        outputs_original, outputs_noisy, atol=1e-1
    ), "Model is not invariant to noise!"


def test_horizontal_flip_invariance(model, sample_data):
    outputs_original = model(sample_data)
    flipped_data = flip_horizontal(sample_data)
    outputs_flipped = model(flipped_data)
    assert not torch.allclose(
        outputs_original, outputs_flipped
    ), "Model should not be invariant to horizontal flips!"


def test_rotation_invariance(model, sample_data):
    outputs_original = model(sample_data)
    rotated_data = rotate_90(sample_data)
    outputs_rotated = model(rotated_data)
    assert not torch.allclose(
        outputs_original, outputs_rotated
    ), "Model should not be invariant to rotations!"
