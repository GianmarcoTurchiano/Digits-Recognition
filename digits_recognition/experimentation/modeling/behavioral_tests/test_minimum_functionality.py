import pytest
import torch
import yaml

from digits_recognition.experimentation.modeling.evaluate import setup_components
from digits_recognition.infer_labels import infer_labels


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

data_params = params['data']
evaluation_params = params['evaluation']
data_meta_params = data_params['meta']
data_meta_images_params = data_meta_params['images']

MODEL_PATH = params['model']
TEST_SET_PATH = 'digits_recognition/experimentation/modeling/behavioral_tests/data_minimum_functionality.pkl'
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
