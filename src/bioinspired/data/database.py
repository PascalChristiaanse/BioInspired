"""
Database connection and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from .models import Base

# Load environment variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://bioinspired_user:bioinspired_pass@localhost:5432/bioinspired")

# Create engine
engine = create_engine(DATABASE_URL, echo=os.getenv("DEBUG", "False").lower() == "true")

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)


def get_session():
    """Get a database session."""
    session = SessionLocal()
    try:
        return session
    except Exception:
        session.close()
        raise


def get_session_context():
    """Get a database session as a context manager."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
