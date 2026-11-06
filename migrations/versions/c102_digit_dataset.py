"""Human-confirmed OCR dataset grouped by source board/image."""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
revision="c102_digit_dataset"
down_revision="b921a0_layout"
branch_labels=depends_on=None


def upgrade():
    op.create_table("layout_digit_dataset",
        sa.Column("id",sa.Integer(),primary_key=True),
        sa.Column("capture_id",sa.String(36),nullable=False),
        sa.Column("cell_index",sa.Integer(),nullable=False),
        sa.Column("source_board",sa.String(200),nullable=False),
        sa.Column("image_hash",sa.String(64),nullable=False),
        sa.Column("crop_path",sa.Text(),nullable=False),
        sa.Column("rows",sa.Integer(),nullable=False),
        sa.Column("cols",sa.Integer(),nullable=False),
        sa.Column("state",sa.String(12),nullable=False),
        sa.Column("label",sa.Integer(),nullable=True),
        sa.Column("raw_outputs",JSONB(),nullable=False),
        sa.Column("quality",JSONB(),nullable=False),
        sa.Column("created_at",sa.DateTime(),server_default=sa.func.now(),nullable=False),
        sa.UniqueConstraint("capture_id","cell_index",name="uq_layout_digit_capture_cell"))
    op.create_index("ix_layout_digit_dataset_source_board","layout_digit_dataset",["source_board"])
    op.create_index("ix_layout_digit_dataset_image_hash","layout_digit_dataset",["image_hash"])


def downgrade():
    op.drop_table("layout_digit_dataset")
