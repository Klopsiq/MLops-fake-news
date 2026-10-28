import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from typing import Any

from faststream import AckPolicy
from faststream.exceptions import RejectMessage
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import InferenceTask, TaskAttempt, TaskStatus
from app.schemas import TaskMessage


class TaskProcessor:
	def __init__(self, engine, model, settings, name, version):
		self.sessions = async_sessionmaker(engine, expire_on_commit=False)
		self.model = model
		self.settings = settings
		self.name, self.version = name, version
		self.pool = ThreadPoolExecutor(max_workers=settings.worker_concurrency)

	def infer(self, text):
		time.sleep(self.settings.inference_delay_seconds)
		label = int(self.model.predict([text])[0])
		return {
			"label": "fake" if label == 1 else "real",
			"model_name": self.name,
			"model_version": self.version,
		}

	async def process(self, task):
		terminal = False
		async with self.sessions() as session, session.begin():
			attempt = await session.scalar(
				insert(TaskAttempt)
				.values(id=task.task_id, count=1)
				.on_conflict_do_update(
					index_elements=[TaskAttempt.id],
					set_={"count": TaskAttempt.count + 1},
				)
				.returning(TaskAttempt.count)
			)
		async with self.sessions() as session, session.begin():
			await session.execute(
				insert(InferenceTask)
				.values(
					id=task.task_id,
					text=task.text,
					status=TaskStatus.PENDING,
					created_at=task.submitted_at,
				)
				.on_conflict_do_nothing(index_elements=[InferenceTask.id])
			)
			row = await session.scalar(
				select(InferenceTask)
				.where(InferenceTask.id == task.task_id)
				.with_for_update()
			)
			if row.text != task.text:
				raise RejectMessage()
			if row.status == TaskStatus.COMPLETED:
				return
			if row.status == TaskStatus.FAILED:
				raise RejectMessage()
			try:
				if attempt > 4:
					raise RuntimeError("Retry budget exhausted")
				result = await asyncio.get_running_loop().run_in_executor(
					self.pool, partial(self.infer, task.text)
				)
			except Exception as exc:
				if attempt < 4:
					await asyncio.sleep(0.2)
					raise
				row.status = TaskStatus.FAILED
				row.error = type(exc).__name__
				terminal = True
			else:
				result["task_seconds"] = (
					datetime.now(UTC) - task.submitted_at
				).total_seconds()
				row.result = result
				row.status = TaskStatus.COMPLETED
				row.error = None
		if terminal:
			raise RejectMessage()


def register_handler(broker, queue, get_processor):
	@broker.subscriber(queue, ack_policy=AckPolicy.NACK_ON_ERROR)
	async def handle(body: Any):
		try:
			task = TaskMessage.model_validate(body)
		except ValidationError as exc:
			raise RejectMessage() from exc
		await get_processor().process(task)

	return handle
