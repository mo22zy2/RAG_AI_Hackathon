import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text as sql_text
from helpers.config import get_settings


async def reset():
    settings = get_settings()
    conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    engine = create_async_engine(conn)
    session_factory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with session_factory() as session:
        async with session.begin():
            await session.execute(sql_text("DROP TABLE IF EXISTS chunks CASCADE"))
            await session.execute(sql_text("DROP TABLE IF EXISTS assets CASCADE"))
            await session.execute(sql_text("DROP TABLE IF EXISTS projects CASCADE"))
            await session.execute(sql_text("DROP TABLE IF EXISTS alembic_version"))
            await session.execute(sql_text(
                "DO $$ DECLARE tbl TEXT; BEGIN "
                "FOR tbl IN SELECT tablename FROM pg_tables WHERE tablename LIKE 'collection_%' "
                "LOOP EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(tbl) || ' CASCADE'; END LOOP; "
                "END $$;"
            ))
            await session.execute(sql_text("DROP EXTENSION IF EXISTS vector"))

    await engine.dispose()
    print("All data dropped. Now run: alembic upgrade head")


if __name__ == "__main__":
    asyncio.run(reset())
