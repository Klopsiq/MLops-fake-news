import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import create_async_engine
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.model import load_champion
from app.models import Base
from app.task_service import TaskPublisher

# создали логгери с именем модуля
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

	engine = create_async_engine(os.environ["DATABASE_URL"])
	app.state.engine = engine
	app.state.settings = get_settings()
	app.state.task_publisher = TaskPublisher(app.state.settings)
	try:
		# До yield: одна загрузка на процесс. Ошибка Registry прерывает startup.
		model, name, version = await run_in_threadpool(load_champion)
		async with engine.begin() as connection:
			await connection.run_sync(Base.metadata.create_all)
		app.state.model = model
		app.state.model_name = name
		app.state.model_version = version
		logger.info("Загружена модель %s версии %s", name, version)
		logger.info("Приложение начало работу с версией %s", app.version)
		yield  # приостановка во время работы приоложения
	finally:  # если завершение, искючение - все что в блоке finally все равно выполнится
		await app.state.task_publisher.close()
		app.state.model = None
		await engine.dispose()  # закрытие пула соединений
		logger.info("Приложение освободило ресурсы БД")
