"""Confirmed layout corpus and reproducible inference/evaluation traces."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "b921a0_layout"
down_revision = "ac41ae7f9661"
branch_labels = depends_on = None


def upgrade():
    op.create_table("layout_history",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("cols", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(200), nullable=False, unique=True),
        sa.Column("fingerprint", sa.String(64), nullable=False, unique=True),
        sa.Column("cells", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False))
    op.create_table("layout_runs",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("cols", sa.Integer(), nullable=False),
        sa.Column("request", JSONB(), nullable=False),
        sa.Column("result", JSONB(), nullable=False),
        sa.Column("outcome", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False))
    op.create_index("ix_layout_runs_kind", "layout_runs", ["kind"])


def downgrade():
    op.drop_table("layout_runs")
    op.drop_table("layout_history")
