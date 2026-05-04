from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_process_endpoint(mocker):
    mocker.patch(
        "api.routes.preprocess_endpoints.preprocessor.preprocess",
        return_value={
            "id": "1",
            "name": "test",
            "preprocessed_transcription": "hello world",
            "timestamp": "2026-01-01T00:00:00"
        }
    )

    payload = {
        "id": "1",
        "name": "test",
        "transcription": "hello"
    }

    response = client.post("/process", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "success"