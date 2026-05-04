import pytest


@pytest.fixture
def mock_session(mocker):
    """A mock SQLAlchemy session with all common methods available."""
    session = mocker.MagicMock()
    return session