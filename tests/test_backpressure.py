from uuid import uuid4

import pytest
from faststream.rabbit import RabbitBroker
from faststream.rabbit.testing import TestRabbitBroker

from app.config import get_settings
from app.messaging import task_queue
from app.schemas import TaskMessage


def test_queue_has_a_hard_limit_and_rejects_publish():
	settings = get_settings()
	queue = task_queue(settings)
	assert queue.arguments["x-max-length"] == settings.queue_max_length
	assert queue.arguments["x-overflow"] == "reject-publish"


@pytest.mark.asyncio
async def test_task_message_handler_with_test_rabbit_broker():
	settings = get_settings()
	broker = RabbitBroker(settings.rabbitmq_url)
	received: list[TaskMessage] = []

	@broker.subscriber(task_queue(settings))
	async def handler(message: TaskMessage):
		received.append(message)

	message = TaskMessage(task_id=uuid4(), text="news")
	async with TestRabbitBroker(broker):
		await broker.publish(message, queue=settings.queue_name)

	assert received == [message]


def test_overload_returns_503_with_retry_after_and_no_pending_row(monkeypatch):
	from fastapi.testclient import TestClient
	from sqlalchemy import func, select
	from sqlalchemy.ext.asyncio import async_sessionmaker

	from app.main import app
	from app.models import InferenceTask

	task_id = uuid4()

	async def fail_publish(*args, **kwargs):
		raise TimeoutError("publisher confirm timed out")

	async def count_tasks():
		from sqlalchemy.ext.asyncio import create_async_engine

		engine = create_async_engine(app.state.settings.database_url)
		factory = async_sessionmaker(engine)
		async with factory() as session:
			count = await session.scalar(
				select(func.count())
				.select_from(InferenceTask)
				.where(InferenceTask.id == task_id)
			)
		await engine.dispose()
		return count

	with TestClient(app) as client:
		import asyncio

		before = asyncio.run(count_tasks())
		monkeypatch.setattr(app.state.task_publisher, "publish", fail_publish)
		response = client.post(
			"/api/v1/tasks", json={"task_id": str(task_id), "text": "news"}
		)
		after = asyncio.run(count_tasks())

	assert response.status_code == 503
	assert response.headers["Retry-After"] == str(
		app.state.settings.retry_after_seconds
	)
	assert after == before
