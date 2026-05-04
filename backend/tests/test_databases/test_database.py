import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_session(mocker):
    """Patch SessionLocal so get_session() receives our mock session."""
    session = mocker.MagicMock()
    mocker.patch("databases.database.SessionLocal", return_value=session)
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetSession:
    def test_commits_on_success(self, mock_session):
        """Session must be committed when the body completes normally."""
        from databases.database import get_session

        for session in get_session():
            pass

        session.commit.assert_called_once()
        session.rollback.assert_not_called()

    def test_rolls_back_and_reraises_on_exception(self, mock_session):
        """
        Session must roll back and re-raise on exception.

        A plain for-loop does NOT throw exceptions into the generator —
        it just stops iterating. We must drive the generator manually
        with .throw() so get_session()'s except block actually fires.
        """
        from databases.database import get_session

        gen = get_session()
        session = next(gen)  # advance to the yield, get the session

        with pytest.raises(RuntimeError, match="something went wrong"):
            gen.throw(RuntimeError("something went wrong"))

        session.rollback.assert_called_once()
        session.commit.assert_not_called()

    def test_always_closes_session(self, mock_session):
        """session.close() must be called regardless of success or failure."""
        from databases.database import get_session

        # success path
        for session in get_session():
            success_session = session
        success_session.close.assert_called()

        # failure path — use .throw() so the except/finally in get_session fires
        mock_session.reset_mock()
        gen = get_session()
        failure_session = next(gen)
        with pytest.raises(ValueError):
            gen.throw(ValueError("boom"))
        failure_session.close.assert_called()