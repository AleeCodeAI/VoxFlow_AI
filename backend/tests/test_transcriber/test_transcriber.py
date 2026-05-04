import pytest
from core.transcriber.transcriber import Transcriber
from pydantic_schemas import TranscriptionError

@pytest.fixture
def transcriber(mocker):
    mocker.patch("core.transcriber.transcriber.whisper.load_model")
    mocker.patch("core.transcriber.transcriber.AudioProcessor")
    mocker.patch("core.transcriber.transcriber.TranscriptionRepository")
    mocker.patch("core.transcriber.transcriber.TranscriberObservability")

    return Transcriber()

def test_transcribe_raises_when_file_missing(transcriber):
    with pytest.raises(TranscriptionError):
        transcriber.transcribe("fake/path/audio.mp3")


def test_transcribe_raises_on_invalid_extension(transcriber, mocker):
    mocker.patch(
        "core.transcriber.transcriber.os.path.exists",
        return_value=True
    )

    with pytest.raises(TranscriptionError):
        transcriber.transcribe("audio.txt")


def test_transcribe_happy_path(transcriber, mocker):
    fake_audio = mocker.Mock()

    mocker.patch(
        "core.transcriber.transcriber.os.path.exists",
        return_value=True
    )
    mocker.patch(
        "core.transcriber.transcriber.AudioSegment.from_file",
        return_value=fake_audio
    )

    transcriber.audio_processor.split_audio_chunks.return_value = ["c1", "c2"]

    transcriber.audio_processor.process_chunks.return_value = (
        {0: "hello", 1: "world"},
        mocker.Mock()
    )

    transcriber.repository.save.return_value = mocker.Mock(id=123)

    result, report = transcriber.transcribe("audio.mp3")

    transcriber.repository.save.assert_called_once()
    assert result.id == 123


def test_transcribe_joins_chunks_in_order(transcriber, mocker):
    fake_audio = mocker.Mock()

    mocker.patch(
        "core.transcriber.transcriber.os.path.exists",
        return_value=True
    )
    mocker.patch(
        "core.transcriber.transcriber.AudioSegment.from_file",
        return_value=fake_audio
    )

    transcriber.audio_processor.split_audio_chunks.return_value = ["c1", "c2", "c3"]

    transcriber.audio_processor.process_chunks.return_value = (
        {0: "A", 1: "B", 2: "C"},
        mocker.Mock()
    )

    transcriber.repository.save.return_value = mocker.Mock(id=1)

    transcriber.transcribe("audio.mp3")

    args, _ = transcriber.repository.save.call_args
    saved_text = args[1]

    assert saved_text == "A B C"

