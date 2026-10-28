import asyncio
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from faststream.exceptions import RejectMessage
from faststream.rabbit import RabbitBroker
from faststream.rabbit.testing import TestRabbitBroker
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.messaging import task_queue
from app.models import Base, TaskStatus
from app.processing import TaskProcessor, register_handler
from app.schemas import TaskMessage
from app.task_service import get_task


@pytest.mark.asyncio
async def test_real_handler_saves_once_and_keeps_loop_responsive():
	settings = replace(get_settings(), inference_delay_seconds=0.15)
	engine = create_async_engine(settings.database_url)
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	model = Mock()
	model.predict.return_value = [1]
	processor = TaskProcessor(engine, model, settings, "test-model", "9")
	broker = RabbitBroker()
	register_handler(broker, task_queue(settings), lambda: processor)
	task = TaskMessage(task_id=uuid4(), text="worker test")
	try:
		async with TestRabbitBroker(broker):
			job = asyncio.create_task(broker.publish(task, queue=settings.queue_name))
			await asyncio.sleep(0.03)
			assert not job.done()
			await job
			await broker.publish(task, queue=settings.queue_name)
		row = await get_task(engine, task.task_id)
		assert row.status == TaskStatus.COMPLETED
		assert row.result["model_version"] == "9"
		assert row.result["label"] == "fake"
		assert model.predict.call_count == 1
	finally:
		processor.pool.shutdown()
		await engine.dispose()


@pytest.mark.asyncio
async def test_failed_inference_rolls_back_then_commits_failed_before_dlq():
	settings = replace(get_settings(), inference_delay_seconds=0)
	engine = create_async_engine(settings.database_url)
	async with engine.begin() as conn:
		await conn.run_sync(Base.metadata.create_all)
	model = Mock()
	model.predict.side_effect = ValueError("poison")
	processor = TaskProcessor(engine, model, settings, "test", "1")
	task = TaskMessage(task_id=uuid4(), text="poison test")
	try:
		for _ in range(3):
			with pytest.raises(ValueError):
				await processor.process(task)
		assert await get_task(engine, task.task_id) is None
		with pytest.raises(RejectMessage):
			await processor.process(task)
		assert (await get_task(engine, task.task_id)).status == TaskStatus.FAILED
	finally:
		processor.pool.shutdown()
		await engine.dispose()
