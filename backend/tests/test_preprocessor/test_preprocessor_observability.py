import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_langfuse_client(mocker):
    """Mock the Langfuse client so no real HTTP calls are made."""
    return mocker.patch(
        "core.preprocessor.preprocessor_observability.Langfuse",
        autospec=True,
    )


@pytest.fixture
def mock_langfuse_context(mocker):
    """Mock langfuse_context so decorator side-effects are fully isolated."""
    return mocker.patch(
        "core.preprocessor.preprocessor_observability.langfuse_context",
        autospec=True,
    )


@pytest.fixture
def mock_main_settings(mocker, mock_settings):
    return mocker.patch(
        "core.preprocessor.preprocessor_observability.MainSettings",
        return_value=mock_settings,
    )


@pytest.fixture
def observability(mock_langfuse_client, mock_langfuse_context, mock_main_settings):
    """Construct a PreprocessorObservability with all external deps mocked."""
    from core.preprocessor.preprocessor_observability import PreprocessorObservability
    return PreprocessorObservability()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUpdateTrace:
    def test_update_trace_calls_langfuse_context_with_correct_args(
        self, observability, mock_langfuse_context
    ):
        """update_trace must forward all metadata to langfuse_context."""
        observability.update_trace(
            session_id="session-123",
            audio_name="meeting.mp3",
            transcription_length=4200,
            chunk_size=2000,
        )

        mock_langfuse_context.update_current_trace.assert_called_once_with(
            session_id="session-123",
            tags=["preprocessing", "audio"],
            metadata={
                "audio_name": "meeting.mp3",
                "transcription_length": 4200,
                "chunk_size": 2000,
            },
        )


class TestScoreSuccess:
    def test_score_success_calls_langfuse_score_when_trace_exists(
        self, observability, mock_langfuse_context, mock_langfuse_client
    ):
        """score_success must call langfuse.score with value=1 when a trace ID is present."""
        mock_langfuse_context.get_current_trace_id.return_value = "trace-abc"

        observability.score_success()

        observability.langfuse.score.assert_called_once_with(
            trace_id="trace-abc",
            name="preprocessing-success",
            value=1,
            comment="Successfully completed preprocessing",
        )

    def test_score_success_does_nothing_when_no_trace_id(
        self, observability, mock_langfuse_context
    ):
        """score_success must be a no-op when get_current_trace_id returns None."""
        mock_langfuse_context.get_current_trace_id.return_value = None

        observability.score_success()

        observability.langfuse.score.assert_not_called()


class TestScoreFailure:
    def test_score_failure_calls_langfuse_score_with_value_zero(
        self, observability, mock_langfuse_context
    ):
        """score_failure must record value=0 and pass the reason as the comment."""
        mock_langfuse_context.get_current_trace_id.return_value = "trace-xyz"

        observability.score_failure(reason="LLM timeout")

        observability.langfuse.score.assert_called_once_with(
            trace_id="trace-xyz",
            name="preprocessing-failure",
            value=0,
            comment="LLM timeout",
        )