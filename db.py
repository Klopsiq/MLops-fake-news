import asyncio
import os

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def main():
    engine = create_async_engine(os.environ["DATABASE_URL"])

    
    async with engine.connect() as connection:
        result = await connection.execute(text("SELECT version()"))
        print(result.scalar_one())
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
