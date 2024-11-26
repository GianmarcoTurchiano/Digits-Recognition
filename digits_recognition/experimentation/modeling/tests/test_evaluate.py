import pytest
import yaml

from digits_recognition.experimentation.modeling.evaluate import (
    inference_step,
    evaluation_step,
    setup_components
)


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

TEST_SET_PATH = params['data']['processed']['test_set']
BATCH_SIZE = params['evaluation']['batch_size']
IMAGE_WIDTH = params['data']['meta']['images']['width']
IMAGE_HEIGHT = params['data']['meta']['images']['height']
IMAGE_CHANNELS = params['data']['meta']['images']['channels']
CLASS_COUNT = params['data']['meta']['classes']['count']


@pytest.fixture
def components():
    return setup_components(
        TEST_SET_PATH,
        BATCH_SIZE,
        None,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CLASS_COUNT
    )


def test_evaluation_step(components):
    model, device, loader, _ = components

    try:
        all_labels, all_preds = inference_step(model, device, loader)
        evaluation_step(all_labels, all_preds)
    except Exception as e:
        pytest.fail(f"evaluation_step raised an exception: {e}")