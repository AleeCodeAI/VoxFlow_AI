from configs import MainSettings
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from databases.models import Base

settings = MainSettings()

DATABASE_URL = settings.POSTGRESQL_URL

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set in environment variables")

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)


def create_tables():
    """Create all tables in the database if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Get a database session. Always use as a context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

if __name__ == "__main__":
    create_tables()
    print("Database tables created successfully.")