import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine

# создали логгери с именем модуля
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

	engine = create_async_engine(os.environ["DATABASE_URL"])
	app.state.engine = engine

	try:
		logger.info("Приложение начало работу с версией %s", app.version)
		yield  # приостановка во время работы приоложения
	finally:  # если завершение, искючение - все что в блоке finally все равно выполнится
		await engine.dispose()  # закрытие пула соединений
		logger.info("Приложение освободило ресурсы БД")
