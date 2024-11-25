import pytest

from fastapi.testclient import TestClient

from digits_recognition.api.endpoints import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize(
    'endpoint',
    [
        'predictions',
        'probabilities',
        'annotations',
    ],
    ids=[
        '/predictions',
        '/probabilities',
        '/annotations',
    ]
)
def test_endpoint(client, endpoint):
    url = f'http://localhost:8000/{endpoint}'

    file_path = 'digits_recognition/api/tests/test.png'

    with open(file_path, "rb") as image_file:
        response = client.post(
            url,
            files={"file": image_file}
        )

    assert response.status_code == 200
