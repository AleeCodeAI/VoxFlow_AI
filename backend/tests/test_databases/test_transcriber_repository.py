import pytest
from datetime import datetime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo():
    from databases.transcriber_repository import TranscriptionRepository
    return TranscriptionRepository()


@pytest.fixture
def mock_transcription_model(mocker):
    """
    Patch TranscriptionModel and configure the instance it returns so its
    attributes are real strings — Pydantic rejects MagicMock attributes.
    """
    mock_cls = mocker.patch("databases.transcriber_repository.TranscriptionModel")
    instance = mock_cls.return_value
    instance.id = "session-new"
    instance.name = "new.mp3"
    instance.transcription = "raw transcription"
    instance.timestamp = datetime(2024, 1, 1, 12, 0, 0)
    return mock_cls


@pytest.fixture
def base_session(mock_session):
    """Session where no existing record is found by default."""
    mock_session.get.return_value = None
    return mock_session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTranscriptionRepositorySave:
    def test_save_creates_new_record_when_not_exists(
        self, repo, base_session, mock_transcription_model
    ):
        """
        When session.get returns None a new TranscriptionModel must be
        instantiated, added to the session, and flushed.
        """
        repo.save(
            session=base_session,
            session_id="session-new",
            audio_name="new.mp3",
            transcription_text="raw transcription",
        )

        mock_transcription_model.assert_called_once()
        base_session.add.assert_called_once()
        base_session.flush.assert_called_once()

    def test_save_updates_existing_record_without_add(self, repo, mock_session):
        """
        When session.get returns an existing object, fields must be updated
        in-place and session.add must NOT be called.
        """
        existing = mock_session.get.return_value
        # Give the existing mock real string values so Pydantic is satisfied
        existing.id = "session-existing"
        existing.name = "updated.mp3"
        existing.transcription = "updated transcription"
        existing.timestamp = datetime(2024, 1, 1, 12, 0, 0)

        repo.save(
            session=mock_session,
            session_id="session-existing",
            audio_name="updated.mp3",
            transcription_text="updated transcription",
        )

        assert existing.name == "updated.mp3"
        assert existing.transcription == "updated transcription"
        mock_session.add.assert_not_called()
        mock_session.flush.assert_called_once()

    def test_save_returns_transcription_object(self, repo, mock_session):
        """save() must return a Transcription pydantic object with correct fields."""
        from pydantic_schemas import Transcription

        existing = mock_session.get.return_value
        existing.id = "session-123"
        existing.name = "audio.mp3"
        existing.transcription = "raw text"
        existing.timestamp = datetime(2024, 3, 10, 8, 0, 0)

        result = repo.save(
            session=mock_session,
            session_id="session-123",
            audio_name="audio.mp3",
            transcription_text="raw text",
        )

        assert isinstance(result, Transcription)
        assert result.id == "session-123"
        assert result.transcription == "raw text"


class TestTranscriptionRepositoryGet:
    def test_get_returns_none_when_not_found(self, repo, mock_session):
        """get() must return None when session.get finds no matching record."""
        mock_session.get.return_value = None

        result = repo.get(session=mock_session, session_id="missing-id")

        assert result is None

    def test_get_returns_transcription_when_found(self, repo, mock_session):
        """get() must return a Transcription mapped from the DB object."""
        from pydantic_schemas import Transcription

        db_obj = mock_session.get.return_value
        db_obj.id = "session-xyz"
        db_obj.name = "found.mp3"
        db_obj.transcription = "hello world"
        db_obj.timestamp = datetime(2024, 9, 1, 14, 0, 0)

        result = repo.get(session=mock_session, session_id="session-xyz")

        assert isinstance(result, Transcription)
        assert result.id == "session-xyz"
        assert result.transcription == "hello world"