from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_get_transcription(mocker):
    mock_get = mocker.patch(
        "api.routes.retrieval_endpoints.transcription_repo.get"
    )

    mock_get.return_value = mocker.Mock(
        model_dump=lambda: {"id": "1"}
    )

    response = client.get("/transcriptions/1")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_get_preprocessing(mocker):
    mock_get = mocker.patch(
        "api.routes.retrieval_endpoints.preprocessing_repo.get"
    )

    mock_get.return_value = mocker.Mock(
        model_dump=lambda: {"id": "1"}
    )

    response = client.get("/preprocessings/1")

    assert response.status_code == 200
    assert response.json()["status"] == "success"