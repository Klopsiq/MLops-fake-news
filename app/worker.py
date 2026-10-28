import asyncio
from contextlib import asynccontextmanager

from faststream import FastStream
from faststream.rabbit import RabbitBroker
from faststream.rabbit.schemas import Channel
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.messaging import task_queue
from app.model import load_champion
from app.models import Base
from app.processing import TaskProcessor, register_handler
from app.task_service import TaskPublisher


def build_worker(settings, model_loader=load_champion):
	broker = RabbitBroker(
		settings.rabbitmq_url,
		default_channel=Channel(prefetch_count=settings.worker_concurrency),
		graceful_timeout=settings.graceful_timeout_seconds,
	)
	state = {}

	async def health(reader, writer):
		try:
			await asyncio.wait_for(reader.read(4096), 2)
			writer.write(
				b'HTTP/1.1 200 OK\r\nContent-Length: 15\r\nConnection: close\r\n\r\n{"status":"ok"}'
			)
			await writer.drain()
		finally:
			writer.close()
			await writer.wait_closed()

	@asynccontextmanager
	async def lifespan():
		model, name, version = await asyncio.to_thread(model_loader)
		engine = create_async_engine(settings.database_url)
		async with engine.begin() as connection:
			await connection.run_sync(Base.metadata.create_all)
		processor = TaskProcessor(engine, model, settings, name, version)
		state["processor"] = processor
		publisher = TaskPublisher(settings)
		await publisher.start()
		await publisher.close()
		server = await asyncio.start_server(
			health, "0.0.0.0", settings.worker_health_port
		)
		try:
			yield
		finally:
			server.close()
			await server.wait_closed()
			await asyncio.to_thread(processor.pool.shutdown, wait=True)
			await engine.dispose()

	handler = register_handler(broker, task_queue(settings), lambda: state["processor"])
	return FastStream(broker, lifespan=lifespan), broker, handler


app, broker, process_task = build_worker(get_settings())
