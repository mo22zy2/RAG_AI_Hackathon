"""add composite index on assets(asset_project_id, asset_name)

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-08-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # AssetModel.get_asset_record filters on this exact (project_id, name)
    # pair; only asset_project_id was indexed before.
    op.create_index(
        'ix_asset_project_id_asset_name',
        'assets',
        ['asset_project_id', 'asset_name'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_asset_project_id_asset_name', table_name='assets')
