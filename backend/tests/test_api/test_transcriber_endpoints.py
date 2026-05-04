from fastapi.testclient import TestClient
from api.main import app
from pydantic_schemas import Transcription

client = TestClient(app)


def test_transcribe_audio_endpoint(mocker):
    mocker.patch(
        "api.routes.transcriber_endpoints.transcriber.transcribe",
        return_value=(
            Transcription(
                id="1",
                name="audio",
                transcription="hello world",
                timestamp="2026-01-01T00:00:00"
            ),
            mocker.Mock()
        )
    )

    mocker.patch(
        "api.routes.transcriber_endpoints.cache.get",
        return_value=None
    )

    mocker.patch(
        "api.routes.transcriber_endpoints.cache.set"
    )

    files = {"file": ("audio.mp3", b"fake audio", "audio/mpeg")}

    response = client.post("/transcribe/audio", files=files)

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_transcribe_text_endpoint(mocker):
    from pydantic_schemas import Transcription

    mocker.patch(
        "api.routes.transcriber_endpoints.transcription_repo.save",
        return_value=Transcription(
            id="123",
            name="test",
            transcription="hello",
            timestamp="2026-01-01T00:00:00"
        )
    )

    payload = {
        "name": "test",
        "transcription": "hello"
    }

    response = client.post("/transcribe/text", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "success"