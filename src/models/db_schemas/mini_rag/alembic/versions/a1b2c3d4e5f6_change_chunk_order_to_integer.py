"""change chunk_order from String to Integer

Revision ID: a1b2c3d4e5f6
Revises: 6843b52b0642
Create Date: 2026-07-30 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '6843b52b0642'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column(
        'chunks',
        'chunk_order',
        type_=sa.Integer(),
        postgresql_using='chunk_order::integer',
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        'chunks',
        'chunk_order',
        type_=sa.String(),
        existing_nullable=False,
    )
