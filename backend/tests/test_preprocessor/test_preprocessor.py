import pytest
from pydantic_schemas import LLMCallError, DatabaseError, PreprocessorError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_deps(mocker, mock_settings):
    """
    Patch all three collaborators (LLM, repository, observability) and
    MainSettings so Preprocessor can be instantiated without any real I/O.
    """
    mocker.patch(
        "core.preprocessor.preprocessor.MainSettings",
        return_value=mock_settings,
    )
    mock_llm = mocker.patch(
        "core.preprocessor.preprocessor.PreprocessorLLM", autospec=True
    )
    mock_repo = mocker.patch(
        "core.preprocessor.preprocessor.PreprocessedRepository", autospec=True
    )
    mock_obs = mocker.patch(
        "core.preprocessor.preprocessor.PreprocessorObservability", autospec=True
    )
    mock_prompt = mocker.patch(
        "core.preprocessor.preprocessor.PREPROCESSOR_PROMPT", autospec=True
    )

    # Sensible default return values
    mock_llm.return_value.call.return_value = "cleaned text"
    mock_llm.return_value.retries = 0
    mock_repo.return_value.save.return_value = mocker.MagicMock()
    mock_obs.return_value.score_success.return_value = None
    mock_prompt.system_prompt = "You are a helpful assistant."
    mock_prompt.render_user.return_value = "user prompt text"

    return {
        "llm": mock_llm,
        "repo": mock_repo,
        "obs": mock_obs,
        "prompt": mock_prompt,
    }


@pytest.fixture
def preprocessor(mock_deps):
    from core.preprocessor.preprocessor import Preprocessor
    return Preprocessor()


SHORT_TEXT = "Hello world. This is a short transcription."

LONG_TEXT = (
    "This is the first sentence of a long chunk. " * 30          # ~1 320 chars
    + "This ends the first chunk! "
    + "Second chunk starts here and keeps going. " * 30           # another ~1 260 chars
    + "End of second chunk."
)

BASE_INPUT = {
    "id": "session-abc",
    "name": "audio.mp3",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPreprocessSingleChunk:
    def test_short_text_uses_single_llm_call(self, preprocessor, mock_deps):
        """
        Text shorter than chunk_size must go through exactly one LLM call
        without chunking.
        """
        preprocessor.preprocess({**BASE_INPUT, "transcription": SHORT_TEXT}, chunk_size=2000)

        mock_deps["llm"].return_value.call.assert_called_once()

    def test_single_chunk_result_is_saved(self, preprocessor, mock_deps):
        """The cleaned text from a single-pass call must be persisted."""
        mock_deps["llm"].return_value.call.return_value = "cleaned output"

        preprocessor.preprocess({**BASE_INPUT, "transcription": SHORT_TEXT}, chunk_size=2000)

        mock_deps["repo"].return_value.save.assert_called_once_with(
            "session-abc", "audio.mp3", "cleaned output"
        )


class TestPreprocessMultiChunk:
    def test_long_text_triggers_multiple_llm_calls(self, preprocessor, mock_deps):
        """
        Text longer than chunk_size must produce more than one LLM call,
        one per chunk.
        """
        preprocessor.preprocess({**BASE_INPUT, "transcription": LONG_TEXT}, chunk_size=500)

        call_count = mock_deps["llm"].return_value.call.call_count
        assert call_count > 1

    def test_chunks_are_joined_before_saving(self, preprocessor, mock_deps):
        """
        Multi-chunk output must be joined with a space and saved as a
        single string, not a list.
        """
        mock_deps["llm"].return_value.call.return_value = "chunk"

        preprocessor.preprocess({**BASE_INPUT, "transcription": LONG_TEXT}, chunk_size=500)

        saved_text = mock_deps["repo"].return_value.save.call_args[0][2]
        assert isinstance(saved_text, str)
        assert "chunk chunk" in saved_text  # multiple chunks joined


class TestPreprocessMissingFields:
    @pytest.mark.parametrize("bad_input", [
        {"id": "session-abc", "name": "audio.mp3", "transcription": ""},   # empty transcription
        {"id": "", "name": "audio.mp3", "transcription": "Some text."},    # missing id
        {"name": "audio.mp3", "transcription": "Some text."},              # no id key at all
    ])
    def test_missing_required_fields_raises_preprocessor_error(
        self, preprocessor, mock_deps, bad_input
    ):
        """Missing or empty transcription/id must raise PreprocessorError."""
        with pytest.raises(PreprocessorError):
            preprocessor.preprocess(bad_input)


class TestPreprocessErrorHandling:
    def test_llm_call_error_propagates_and_scores_failure(self, preprocessor, mock_deps):
        """
        LLMCallError from the LLM layer must propagate out of preprocess()
        and trigger observability.score_failure.
        """
        mock_deps["llm"].return_value.call.side_effect = LLMCallError("LLM timed out")

        with pytest.raises(LLMCallError):
            preprocessor.preprocess({**BASE_INPUT, "transcription": SHORT_TEXT})

        mock_deps["obs"].return_value.score_failure.assert_called_once()

    def test_database_error_propagates_and_scores_failure(self, preprocessor, mock_deps):
        """
        DatabaseError from the repository must propagate out of preprocess()
        and trigger observability.score_failure.
        """
        mock_deps["repo"].return_value.save.side_effect = DatabaseError("DB unavailable")

        with pytest.raises(DatabaseError):
            preprocessor.preprocess({**BASE_INPUT, "transcription": SHORT_TEXT})

        mock_deps["obs"].return_value.score_failure.assert_called_once()


class TestPreprocessReport:
    def test_report_is_populated_after_multi_chunk_processing(self, preprocessor, mock_deps):
        """
        After processing multiple chunks, preprocessor.report must reflect
        the correct chunk_count and chunks_processed values.
        """
        preprocessor.preprocess({**BASE_INPUT, "transcription": LONG_TEXT}, chunk_size=500)

        report = preprocessor.report
        assert report.chunk_count > 1
        assert report.chunks_processed == report.chunk_count