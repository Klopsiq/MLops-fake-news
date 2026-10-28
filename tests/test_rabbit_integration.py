import asyncio
import os
from dataclasses import replace
from uuid import uuid4

import pytest
from aio_pika.exceptions import DeliveryError
from testcontainers.community.rabbitmq import RabbitMqContainer

from app.config import get_settings
from app.schemas import TaskMessage
from app.task_service import TaskPublisher


@pytest.mark.skipif(
	os.getenv("RUN_TESTCONTAINERS") != "1",
	reason="Requires Docker; set RUN_TESTCONTAINERS=1",
)
def test_real_rabbit_rejects_overflow_and_accepts_after_drain():
	with RabbitMqContainer("rabbitmq:4-management") as rabbit:
		params = rabbit.get_connection_params()
		asyncio.run(
			check_overflow(
				f"amqp://{rabbit.username}:{rabbit.password}@{params.host}:{params.port}/"
			)
		)


async def check_overflow(url):
	settings = replace(
		get_settings(),
		rabbitmq_url=url,
		queue_name=f"integration.{uuid4().hex}",
		queue_max_length=2,
	)
	publisher = TaskPublisher(settings)
	await publisher.start()
	try:
		await publisher.publish(TaskMessage(text="first"))
		await publisher.publish(TaskMessage(text="second"))
		await asyncio.sleep(0.2)
		with pytest.raises(DeliveryError):
			await publisher.publish(TaskMessage(text="overflow"))
		from fastapi import FastAPI
		from httpx import ASGITransport, AsyncClient
		from sqlalchemy.ext.asyncio import create_async_engine

		from app.routes import api_router
		from app.task_service import get_task

		app = FastAPI()
		app.include_router(api_router)
		app.state.settings = settings
		app.state.engine = create_async_engine(settings.database_url)
		app.state.task_publisher = publisher
		rejected_id = uuid4()
		async with AsyncClient(
			transport=ASGITransport(app=app), base_url="http://test"
		) as http:
			response = await http.post(
				"/api/v1/tasks", json={"task_id": str(rejected_id), "text": "full"}
			)
		assert response.status_code == 503
		assert int(response.headers["Retry-After"]) > 0
		assert await get_task(app.state.engine, rejected_id) is None
		await app.state.engine.dispose()
		from app.messaging import task_queue

		queue = await publisher.broker.declare_queue(task_queue(settings))
		message = await queue.get(timeout=2)
		await message.ack()
		await asyncio.sleep(0.2)
		await publisher.publish(TaskMessage(text="after drain"))
		await check_failed_delivery(settings)
	finally:
		await publisher.close()


async def check_failed_delivery(settings):
	from unittest.mock import Mock

	from faststream.rabbit import RabbitBroker
	from faststream.rabbit.schemas import Channel
	from sqlalchemy.ext.asyncio import create_async_engine

	from app.messaging import dead_letter_queue, task_queue
	from app.models import Base, TaskStatus
	from app.processing import TaskProcessor, register_handler
	from app.task_service import get_task

	settings = replace(
		settings, queue_name=f"poison.{uuid4().hex}", inference_delay_seconds=0
	)
	engine = create_async_engine(settings.database_url)
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	model = Mock()
	model.predict.side_effect = ValueError("permanent inference failure")
	processor = TaskProcessor(engine, model, settings, "test", "1")
	publisher = TaskPublisher(settings)
	broker = RabbitBroker(
		settings.rabbitmq_url, default_channel=Channel(prefetch_count=1)
	)
	register_handler(broker, task_queue(settings), lambda: processor)
	try:
		await publisher.start()
		await broker.start()
		task = TaskMessage(text="always fails")
		await publisher.publish(task)
		dlq = await publisher.broker.declare_queue(dead_letter_queue())
		for _ in range(100):
			message = await dlq.get(fail=False)
			if message:
				assert str(task.task_id).encode() in message.body
				await message.ack()
				break
			await asyncio.sleep(0.1)
		else:
			pytest.fail("Failed inference never reached DLQ")
		assert model.predict.call_count == 4
		assert (await get_task(engine, task.task_id)).status == TaskStatus.FAILED
		await publisher.broker.publish(["invalid schema"], queue=settings.queue_name)
		for _ in range(50):
			message = await dlq.get(fail=False)
			if message:
				await message.ack()
				break
			await asyncio.sleep(0.1)
		else:
			pytest.fail("Invalid schema never reached DLQ")
	finally:
		await broker.stop()
		await publisher.close()
		processor.pool.shutdown()
		await engine.dispose()
