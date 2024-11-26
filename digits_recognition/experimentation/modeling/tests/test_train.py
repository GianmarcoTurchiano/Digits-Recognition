import pytest
import yaml

from digits_recognition.experimentation.modeling.train import (
    training_step,
    validation_step,
    setup_training_components,
    setup_components
)


with open('params.yaml', 'r') as file:
    params = yaml.safe_load(file)

TRAIN_SET_PATH = params['data']['processed']['train_set']
VAL_SET_PATH = params['data']['processed']['val_set']
BATCH_SIZE = params['training']['batch_size']
LEARNING_RATE = params['training']['learning_rate']
WEIGHT_DECAY = params['training']['weight_decay']
EPOCHS = params['training']['epochs']
POLYNOMIAL_SCHEDULER_POWER = params['training']['polynomial_scheduler_power']
IMAGE_WIDTH = params['data']['meta']['images']['width']
IMAGE_HEIGHT = params['data']['meta']['images']['height']
IMAGE_CHANNELS = params['data']['meta']['images']['channels']
CLASS_COUNT = params['data']['meta']['classes']['count']


@pytest.fixture
def training_components():
    return setup_training_components(
        TRAIN_SET_PATH,
        BATCH_SIZE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        EPOCHS,
        POLYNOMIAL_SCHEDULER_POWER,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CLASS_COUNT
    )


@pytest.fixture
def validation_components():
    return setup_components(
        TRAIN_SET_PATH,
        VAL_SET_PATH,
        BATCH_SIZE,
        LEARNING_RATE,
        WEIGHT_DECAY,
        EPOCHS,
        POLYNOMIAL_SCHEDULER_POWER,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS,
        CLASS_COUNT
    )


def test_training_step(training_components):
    model, loader, device, optimizer, criterion, _ = training_components

    try:
        training_step(model, loader, device, optimizer, criterion)
    except Exception as e:
        pytest.fail(f"training_step raised an exception: {e}")


def test_validation_step(validation_components):
    model, _, loader, device, _, criterion, _ = validation_components

    try:
        validation_step(model, loader, device, criterion)
    except Exception as e:
        pytest.fail(f"validation_step raised an exception: {e}")
