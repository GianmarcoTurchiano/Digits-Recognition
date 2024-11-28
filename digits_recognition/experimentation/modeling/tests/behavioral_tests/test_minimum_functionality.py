import pytest
import torch

from digits_recognition.experimentation.modeling.evaluate import setup_components
from digits_recognition.infer_labels import infer_labels
from digits_recognition.experimentation.modeling.tests.params import params_test


(
    MODEL_PATH,
    _,
    BATCH_SIZE,
    RANDOM_SEED,
    CLASS_COUNT,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS
) = params_test()

TEST_SET_PATH = 'digits_recognition/experimentation/modeling/tests/behavioral_tests/data_minimum_functionality.pkl'


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


def test_minimum_functionality(components):
    model, device, loader, _ = components

    total_cases = 0
    correct_cases = 0

    for images, labels in loader:
        labels = labels.to(device)

        preds = infer_labels(model, device, images)

        total_cases += preds.size(0)
        correct_cases += torch.sum(preds == labels).item()

    threshold = 85

    correctness_percentage = (correct_cases / total_cases) * 100

    assert correctness_percentage >= threshold, \
        f"Minimum functionality below threshold ({threshold}%): {correctness_percentage:.2f}%"
