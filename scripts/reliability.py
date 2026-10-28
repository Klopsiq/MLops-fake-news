import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import docker
import httpx
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import get_settings
from app.messaging import dead_letter_queue
from app.schemas import TaskMessage
from app.task_service import TaskPublisher, get_task


async def main():
	client = docker.from_env()
	settings = replace(
		get_settings(), queue_name=f"hw4.reliability.{uuid4().hex}", queue_max_length=20
	)
	publisher = TaskPublisher(settings)
	await publisher.start()
	engine = create_async_engine(settings.database_url)
	worker = client.containers.run(
		"mlops-fake-news-worker:latest",
		command=["faststream", "run", "app.worker:app"],
		detach=True,
		network="mlops-fake-news_default",
		name=f"hw4-reliability-{uuid4().hex[:8]}",
		environment={
			"DATABASE_URL": "postgresql+asyncpg://app:local_dev_password@db:5432/fake_news",
			"RABBITMQ_URL": "amqp://app:local_rabbit_password@rabbitmq:5672/",
			"MLFLOW_TRACKING_URI": "http://mlflow:5000",
			"MODEL_NAME": "fake-news-classifier",
			"MODEL_ALIAS": "champion",
			"TASK_QUEUE": settings.queue_name,
			"QUEUE_MAX_LENGTH": "20",
			"WORKER_CONCURRENCY": "2",
			"INFERENCE_DELAY_SECONDS": "8",
		},
	)
	result = {"queue": settings.queue_name}
	try:
		async with httpx.AsyncClient(auth=("app", "local_rabbit_password")) as http:
			url = f"http://host.docker.internal:15672/api/queues/%2F/{settings.queue_name}"

			async def wait_queue(predicate):
				for _ in range(180):
					q = (await http.get(url)).json()
					if predicate(q):
						return q
					await asyncio.sleep(0.5)
				raise RuntimeError("Queue deadline: " + worker.logs(tail=15).decode())

			async def wait_results(tasks):
				for _ in range(180):
					rows = [await get_task(engine, t.task_id) for t in tasks]
					if all(r and r.result for r in rows):
						return rows
					await asyncio.sleep(0.5)
				raise RuntimeError("Result deadline")

			await wait_queue(lambda q: q.get("consumers") == 1)
			tasks = [
				TaskMessage(text="The city council announced a new transport policy.")
				for _ in range(4)
			]
			for task in tasks:
				await publisher.publish(task)
			await wait_queue(lambda q: q.get("messages_unacknowledged", 0) > 0)
			tick = time.monotonic()
			health = await asyncio.to_thread(
				worker.exec_run,
				[
					"python",
					"-c",
					"import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2).status)",
				],
			)
			assert health.exit_code == 0 and b"200" in health.output
			result["health_check_seconds_including_exec"] = time.monotonic() - tick
			tick = time.monotonic()
			await asyncio.to_thread(worker.stop, timeout=35)
			worker.reload()
			assert worker.attrs["State"]["ExitCode"] == 0
			result["sigterm_seconds"] = time.monotonic() - tick
			committed = [await get_task(engine, t.task_id) for t in tasks]
			result["committed_before_restart"] = sum(
				row is not None for row in committed
			)
			await asyncio.to_thread(worker.start)
			await wait_results(tasks)
			result["sigterm_completed"] = len(tasks)
			await wait_queue(lambda q: q.get("messages", 0) == 0)
			crash_tasks = [
				TaskMessage(text="The parliament voted on the budget yesterday.")
				for _ in range(2)
			]
			for task in crash_tasks:
				await publisher.publish(task)
			await wait_queue(lambda q: q.get("messages_unacknowledged", 0) > 0)
			await asyncio.to_thread(worker.kill, signal="SIGKILL")
			await asyncio.to_thread(worker.start)
			await wait_results(crash_tasks)
			result["sigkill_recovered"] = len(crash_tasks)
			before = (await get_task(engine, tasks[0].task_id)).result
			await publisher.publish(tasks[0])
			await asyncio.sleep(1)
			assert (await get_task(engine, tasks[0].task_id)).result == before
			result["duplicate_result_unchanged"] = True
			poison = {"task_id": str(uuid4()), "text": None}
			await publisher.broker.publish(
				poison, queue=settings.queue_name, persist=True
			)
			dlq = await publisher.broker.declare_queue(dead_letter_queue())
			held = []
			try:
				for _ in range(100):
					message = await dlq.get(fail=False)
					if message:
						held.append(message)
						body = json.loads(message.body)
						if body.get("task_id") == poison["task_id"]:
							result["malformed_in_dlq"] = True
							break
					await asyncio.sleep(0.2)
			finally:
				for message in held:
					await message.nack(requeue=True)
			assert result.get("malformed_in_dlq")
		Path("docs/experiments/reliability.json").write_text(
			json.dumps(result, indent=2)
		)
		print(json.dumps(result), flush=True)
	finally:
		await asyncio.to_thread(worker.stop, timeout=35)
		await asyncio.to_thread(worker.remove)
		await publisher.close()
		await engine.dispose()


if __name__ == "__main__":
	asyncio.run(main())
