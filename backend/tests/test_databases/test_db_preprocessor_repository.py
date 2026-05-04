import pytest
from datetime import datetime
from pydantic_schemas import DatabaseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def repo():
    from databases.preprocessor_repository import PreprocessingRepository
    return PreprocessingRepository()


@pytest.fixture
def mock_preprocessing_model(mocker):
    """
    Patch PreprocessingModel and configure the instance it returns so its
    attributes are real strings — Pydantic rejects MagicMock attributes.
    """
    mock_cls = mocker.patch("databases.preprocessor_repository.PreprocessingModel")
    instance = mock_cls.return_value
    instance.id = "session-new"
    instance.name = "new.mp3"
    instance.preprocessed_transcription = "clean transcription"
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

class TestPreprocessingRepositorySave:
    def test_save_creates_new_record_when_not_exists(
        self, repo, base_session, mock_preprocessing_model
    ):
        """
        When session.get returns None a new PreprocessingModel must be
        instantiated, added to the session, and flushed.
        """
        repo.save(
            session=base_session,
            session_id="session-new",
            audio_name="new.mp3",
            clean_text="clean transcription",
        )

        mock_preprocessing_model.assert_called_once()
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
        existing.preprocessed_transcription = "updated text"
        existing.timestamp = datetime(2024, 1, 1, 12, 0, 0)

        repo.save(
            session=mock_session,
            session_id="session-existing",
            audio_name="updated.mp3",
            clean_text="updated text",
        )

        assert existing.name == "updated.mp3"
        assert existing.preprocessed_transcription == "updated text"
        mock_session.add.assert_not_called()
        mock_session.flush.assert_called_once()

    def test_save_raises_database_error_on_exception(self, repo, base_session, mock_preprocessing_model):
        """Any exception from the session must be wrapped into DatabaseError."""
        base_session.flush.side_effect = Exception("disk full")

        with pytest.raises(DatabaseError, match="new.mp3"):
            repo.save(
                session=base_session,
                session_id="session-new",
                audio_name="new.mp3",
                clean_text="some text",
            )

    def test_save_returns_preprocessed_result(self, repo, mock_session):
        """save() must return a PreprocessedResult with the correct field values."""
        from pydantic_schemas import PreprocessedResult

        existing = mock_session.get.return_value
        existing.id = "session-123"
        existing.name = "audio.mp3"
        existing.preprocessed_transcription = "cleaned"
        existing.timestamp = datetime(2024, 1, 1, 12, 0, 0)

        result = repo.save(
            session=mock_session,
            session_id="session-123",
            audio_name="audio.mp3",
            clean_text="cleaned",
        )

        assert isinstance(result, PreprocessedResult)
        assert result.id == "session-123"
        assert result.preprocessed_transcription == "cleaned"


class TestPreprocessingRepositoryGet:
    def test_get_returns_none_when_not_found(self, repo, mock_session):
        """get() must return None when session.get finds no matching record."""
        mock_session.get.return_value = None

        result = repo.get(session=mock_session, session_id="missing-id")

        assert result is None

    def test_get_returns_preprocessed_result_when_found(self, repo, mock_session):
        """get() must return a PreprocessedResult mapped from the DB object."""
        from pydantic_schemas import PreprocessedResult

        db_obj = mock_session.get.return_value
        db_obj.id = "session-abc"
        db_obj.name = "found.mp3"
        db_obj.preprocessed_transcription = "some clean text"
        db_obj.timestamp = datetime(2024, 6, 15, 9, 30, 0)

        result = repo.get(session=mock_session, session_id="session-abc")

        assert isinstance(result, PreprocessedResult)
        assert result.id == "session-abc"
        assert result.name == "found.mp3"