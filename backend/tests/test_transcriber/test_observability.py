from core.transcriber.observability import TranscriberObservability


def test_update_trace(mocker):
    obs = TranscriberObservability()

    mocker.patch("core.transcriber.observability.langfuse_context.update_current_trace")

    obs.update_trace("session1", "audio.mp3", "whisper", 1000)


def test_score_success(mocker):
    obs = TranscriberObservability()

    mock_score = mocker.patch.object(obs.langfuse, "score")
    mocker.patch(
        "core.transcriber.observability.langfuse_context.get_current_trace_id",
        return_value="trace123"
    )

    obs.score_success()

    mock_score.assert_called_once()


def test_flush():
    obs = TranscriberObservability()
    obs.langfuse.flush = lambda: None
    obs.flush()