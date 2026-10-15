import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine


@asynccontextmanager
async def lifespan(app: FastAPI):

	engine = create_async_engine(os.environ["DATABASE_URL"])
	app.state.engine = engine

	try:
		yield  # приостановка во время работы приоложения
	finally:  # если завершение, искючение - все что в блоке finally все равно выполнится
		await engine.dispose()  # закрытие пула соединений
