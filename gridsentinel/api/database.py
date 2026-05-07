"""SQLAlchemy database setup for GridSentinel — users and audit log tables."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    Column, DateTime, ForeignKey, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ── DB path: always at the project root (D:/BESCOM/gridsentinel/gridsentinel.db)
_db_env = os.environ.get("GS_DB_PATH", "")
_DB_PATH = Path(_db_env) if _db_env else (
    Path(__file__).parent.parent.parent / "gridsentinel.db"
)
DATABASE_URL = f"sqlite:///{_DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id             = Column(Integer, primary_key=True, index=True)
    username       = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)
    role           = Column(String(32), default="viewer", nullable=False)
    created_at     = Column(DateTime, default=datetime.utcnow, nullable=False)


class AuditLog(Base):
    __tablename__ = "audit_log"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=True)
    action      = Column(String(128), nullable=False)
    target_id   = Column(String(128), nullable=True)
    target_type = Column(String(32), nullable=True)
    details     = Column(Text, nullable=True)
    timestamp   = Column(DateTime, default=datetime.utcnow, nullable=False)


# Create tables immediately on import
Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a SQLAlchemy session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
