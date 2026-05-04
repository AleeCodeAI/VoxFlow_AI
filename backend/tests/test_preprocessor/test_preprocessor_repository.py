import pytest
from pydantic_schemas import DatabaseError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_db_repository(mocker):
    """Mock the underlying DB repository that hits the real database."""
    return mocker.patch(
        "core.preprocessor.preprocessor_repository.DBPreprocessingRepository",
        autospec=True,
    )


@pytest.fixture
def mock_get_session(mocker):
    """
    Mock get_session as a generator that yields a single fake session object.
    This mirrors the `for session in get_session()` pattern in the repository.
    """
    fake_session = mocker.MagicMock()

    def _generator():
        yield fake_session

    mocker.patch(
        "core.preprocessor.preprocessor_repository.get_session",
        side_effect=_generator,
    )
    return fake_session


@pytest.fixture
def repository(mock_db_repository, mock_get_session):
    """Construct PreprocessedRepository with all external deps mocked."""
    from core.preprocessor.preprocessor_repository import PreprocessedRepository
    return PreprocessedRepository()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_success_returns_db_result(self, repository, mock_db_repository, mock_get_session):
        """
        Happy path: save() must call db_repository.save with the right args
        and return whatever the DB layer returns.
        """
        fake_result = mock_db_repository.return_value.save.return_value

        result = repository.save(
            session_id="session-123",
            audio_name="interview.mp3",
            clean_text="This is the cleaned transcription.",
        )

        mock_db_repository.return_value.save.assert_called_once_with(
            session=mock_get_session,
            session_id="session-123",
            audio_name="interview.mp3",
            clean_text="This is the cleaned transcription.",
        )
        assert result is fake_result

    def test_save_raises_database_error_on_exception(
        self, repository, mock_db_repository, mock_get_session
    ):
        """
        If the DB layer raises any exception, save() must wrap it in
        DatabaseError (not let the raw exception propagate).
        """
        mock_db_repository.return_value.save.side_effect = RuntimeError("connection lost")

        with pytest.raises(DatabaseError, match="interview.mp3"):
            repository.save(
                session_id="session-123",
                audio_name="interview.mp3",
                clean_text="Some text.",
            )

    def test_save_logs_audio_name_and_length_on_success(
        self, repository, mock_db_repository, mock_get_session, mocker
    ):
        """
        After a successful save, the repository must log both the audio name
        and the character length of the cleaned text.
        """
        spy_log = mocker.patch.object(repository, "log")

        repository.save(
            session_id="session-123",
            audio_name="standup.mp3",
            clean_text="Short clean text.",
        )

        log_calls = " ".join(str(c) for c in spy_log.call_args_list)
        assert "standup.mp3" in log_calls
        assert str(len("Short clean text.")) in log_calls