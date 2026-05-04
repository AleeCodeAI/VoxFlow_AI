import pytest
from core.transcriber.audio_processor import AudioProcessor
from pydantic_schemas import TranscriptionError


@pytest.fixture
def processor(mocker):
    return AudioProcessor(
        whisper_model=mocker.Mock(),
        chunk_length_ms=1000
    )


def test_transcribe_chunk_success(processor, mocker):
    processor.whisper.transcribe.return_value = {"text": "hello"}

    idx, text, retries = processor.transcribe_chunk((0, "file.mp3"))

    assert text == "hello"
    assert idx == 0
    assert retries.success is True


def test_transcribe_chunk_retry_success(processor, mocker):
    processor.whisper.transcribe.side_effect = [
        Exception("fail"),
        {"text": "done"}
    ]

    idx, text, retries = processor.transcribe_chunk((0, "file.mp3"))

    assert text == "done"
    assert idx == 0


def test_transcribe_chunk_fails(processor, mocker):
    processor.whisper.transcribe.side_effect = Exception("fail")

    with pytest.raises(TranscriptionError):
        processor.transcribe_chunk((0, "file.mp3"))


def test_split_audio_chunks_merges(processor, mocker):
    audio = mocker.MagicMock()
    audio.__len__.return_value = 2100

    chunks = processor.split_audio_chunks(audio)

    assert isinstance(chunks, list)