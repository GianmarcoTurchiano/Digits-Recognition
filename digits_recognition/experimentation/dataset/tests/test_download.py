from digits_recognition.experimentation.dataset.download import download, DATASET_URL


def test_download():
    response = download(DATASET_URL)

    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/zip'
