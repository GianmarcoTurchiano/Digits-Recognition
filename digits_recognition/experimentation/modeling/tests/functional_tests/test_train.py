import pytest

from digits_recognition.experimentation.modeling.train import (
    training_step,
    validation_step,
    setup_training_components,
    setup_components
)
from digits_recognition.experimentation.modeling.tests.params import params_train


(
    TRAIN_SET_PATH,
    VAL_SET_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    POLYNOMIAL_SCHEDULER_POWER,
    CLASS_COUNT,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    IMAGE_CHANNELS,
) = params_train()


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
