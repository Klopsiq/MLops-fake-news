import asyncio
from uuid import UUID

from faststream.rabbit import RabbitBroker
from faststream.rabbit.schemas import Channel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from app.config import Settings
from app.messaging import dead_letter_exchange, dead_letter_queue, task_queue
from app.models import InferenceTask
from app.schemas import TaskMessage


class TaskPublisher:
	def __init__(self, settings: Settings):
		self.settings = settings
		self.broker = RabbitBroker(
			settings.rabbitmq_url,
			default_channel=Channel(publisher_confirms=True, on_return_raises=True),
		)
		self._started = False
		self._lock = asyncio.Lock()

	async def start(self) -> None:
		if self._started:
			return
		async with self._lock:
			if self._started:
				return
			await self.broker.connect()
			dlx = await self.broker.declare_exchange(dead_letter_exchange())
			dlq = await self.broker.declare_queue(dead_letter_queue())
			await dlq.bind(dlx, routing_key="failed")
			await self.broker.declare_queue(task_queue(self.settings))
			self._started = True

	async def close(self) -> None:
		if self._started:
			await self.broker.stop()
			self._started = False

	async def publish(self, message: TaskMessage) -> None:
		async with asyncio.timeout(self.settings.publish_timeout_seconds):
			await self.start()
			await self.broker.publish(
				message,
				queue=self.settings.queue_name,
				mandatory=True,
				persist=True,
				timeout=self.settings.publish_timeout_seconds,
				message_id=str(message.task_id),
			)


async def get_task(engine: AsyncEngine, task_id: UUID) -> InferenceTask | None:
	session_factory = async_sessionmaker(engine, expire_on_commit=False)
	async with session_factory() as session:
		return await session.scalar(
			select(InferenceTask).where(InferenceTask.id == task_id)
		)


async def create_and_publish_task(
	engine: AsyncEngine, publisher: TaskPublisher, task: InferenceTask
) -> None:
	await publisher.publish(TaskMessage(task_id=task.id, text=task.text))
