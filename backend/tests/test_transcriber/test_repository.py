from core.transcriber.transcription_repository import TranscriptionRepository


def test_save_calls_db(mocker):
    repo = TranscriptionRepository()

    mock_session = mocker.Mock()
    mock_db = mocker.Mock()
    repo.db_repository = mock_db

    mocker.patch(
        "core.transcriber.transcription_repository.get_session",
        return_value=[mock_session]
    )

    mock_db.save.return_value = mocker.Mock(id=1)

    result = repo.save("audio.mp3", "hello", "session1")

    mock_db.save.assert_called_once()
    assert result.id == 1