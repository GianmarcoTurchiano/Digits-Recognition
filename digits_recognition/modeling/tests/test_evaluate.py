import pytest
from digits_recognition.modeling.evaluate import (
    inference_step,
    evaluation_step,
    setup_components
)


TEST_SET_PATH = r'./data/processed/train_set.pkl'
MODEL_PATH = r'./models/digit_classifier.pth'
BATCH_SIZE = 64


@pytest.fixture
def components():
    return setup_components(
        TEST_SET_PATH,
        BATCH_SIZE,
        MODEL_PATH
    )


def test_evaluation_step(components):
    model, device, loader = components

    try:
        all_labels, all_preds = inference_step(model, device, loader)
        evaluation_step(all_labels, all_preds)
    except Exception as e:
        pytest.fail(f"evaluation_step raised an exception: {e}")
