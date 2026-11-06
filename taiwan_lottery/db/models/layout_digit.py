"""Human-confirmed OCR cells with source provenance and immutable raw evidence."""
from datetime import datetime
from sqlalchemy import Integer, String, DateTime, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from taiwan_lottery.db.base import Base


class LayoutDigitSample(Base):
    __tablename__="layout_digit_dataset"
    __table_args__=(UniqueConstraint("capture_id","cell_index",name="uq_layout_digit_capture_cell"),)
    id: Mapped[int]=mapped_column(Integer,primary_key=True)
    capture_id: Mapped[str]=mapped_column(String(36),nullable=False)
    cell_index: Mapped[int]=mapped_column(Integer,nullable=False)
    source_board: Mapped[str]=mapped_column(String(200),nullable=False,index=True)
    image_hash: Mapped[str]=mapped_column(String(64),nullable=False,index=True)
    crop_path: Mapped[str]=mapped_column(Text,nullable=False)
    rows: Mapped[int]=mapped_column(Integer,nullable=False)
    cols: Mapped[int]=mapped_column(Integer,nullable=False)
    state: Mapped[str]=mapped_column(String(12),nullable=False)
    label: Mapped[int | None]=mapped_column(Integer,nullable=True)
    raw_outputs: Mapped[dict]=mapped_column(JSONB,nullable=False)
    quality: Mapped[dict]=mapped_column(JSONB,nullable=False)
    created_at: Mapped[datetime]=mapped_column(DateTime,server_default=func.now())
