import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.model import load_champion

# создали логгери с именем модуля
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

	engine = create_async_engine(os.environ["DATABASE_URL"])
	app.state.engine = engine

	try:
		# До yield: одна загрузка на процесс. Ошибка Registry прерывает startup.
		model, name, version = await run_in_threadpool(load_champion)
		app.state.model = model
		app.state.model_name = name
		app.state.model_version = version
		logger.info("Загружена модель %s версии %s", name, version)
		logger.info("Приложение начало работу с версией %s", app.version)
		yield  # приостановка во время работы приоложения
	finally:  # если завершение, искючение - все что в блоке finally все равно выполнится
		app.state.model = None
		await engine.dispose()  # закрытие пула соединений
		logger.info("Приложение освободило ресурсы БД")
