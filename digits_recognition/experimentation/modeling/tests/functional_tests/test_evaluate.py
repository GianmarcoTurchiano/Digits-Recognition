import pytest

from digits_recognition.experimentation.modeling.evaluate import (
    inference_step,
    evaluation_step,
    setup_components
)
from digits_recognition.experimentation.modeling.tests.params import params_test


(
    _,
    TEST_SET_PATH,
    BATCH_SIZE,
    _,
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
