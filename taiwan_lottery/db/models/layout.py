"""Layout corpus, immutable inference traces and separate scored outcomes."""
from datetime import datetime
from sqlalchemy import Integer, String, DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from taiwan_lottery.db.base import Base


class LayoutHistory(Base):
    __tablename__ = "layout_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(200), unique=True)
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True)
    cells: Mapped[list] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LayoutRun(Base):
    __tablename__ = "layout_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    rows: Mapped[int] = mapped_column(Integer)
    cols: Mapped[int] = mapped_column(Integer)
    request: Mapped[dict] = mapped_column(JSONB)
    result: Mapped[dict] = mapped_column(JSONB)
    outcome: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
