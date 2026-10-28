import asyncio
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4, uuid5

import docker
import httpx
import psutil
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import get_settings
from app.messaging import task_queue
from app.models import InferenceTask
from app.source import send_file
from app.task_service import TaskPublisher


async def scenario(label, limit, replicas, client, input_path, output):
	run_id = uuid4()
	settings = replace(
		get_settings(), queue_name=f"hw4.{label}.{run_id.hex}", queue_max_length=limit
	)
	publisher = TaskPublisher(settings)
	await publisher.start()
	observed_queue = await publisher.broker.declare_queue(task_queue(settings))
	containers = []
	engine = create_async_engine(settings.database_url)
	sessions = async_sessionmaker(engine)
	count = sum(1 for _ in input_path.open())
	ids = [uuid5(run_id, str(i)) for i in range(count)]
	progress = {"accepted": 0, "rejected": 0}
	try:
		for i in range(replicas):
			container = client.containers.run(
				"mlops-fake-news-worker:latest",
				command=["faststream", "run", "app.worker:app"],
				detach=True,
				name=f"hw4-exp-{run_id.hex[:10]}-{i}",
				network="mlops-fake-news_default",
				environment={
					"DATABASE_URL": "postgresql+asyncpg://app:local_dev_password@db:5432/fake_news",
					"RABBITMQ_URL": "amqp://app:local_rabbit_password@rabbitmq:5672/",
					"MLFLOW_TRACKING_URI": "http://mlflow:5000",
					"MODEL_NAME": "fake-news-classifier",
					"MODEL_ALIAS": "champion",
					"TASK_QUEUE": settings.queue_name,
					"QUEUE_MAX_LENGTH": str(limit),
					"WORKER_CONCURRENCY": "2",
					"INFERENCE_DELAY_SECONDS": "0.2",
				},
			)
			containers.append(container)
		async with httpx.AsyncClient(
			auth=("app", "local_rabbit_password"), timeout=10
		) as http:
			queue_url = f"http://host.docker.internal:15672/api/queues/%2F/{settings.queue_name}"
			for _ in range(180):
				q = (await http.get(queue_url)).json()
				if q.get("consumers") == replicas:
					break
				await asyncio.sleep(1)
			else:
				raise RuntimeError(
					"Workers not ready: " + containers[0].logs(tail=20).decode()
				)
			started = time.monotonic()
			producer = asyncio.create_task(
				send_file(publisher, input_path, run_id, progress)
			)
			with (output / f"{label}.csv").open("w", newline="") as stream:
				writer = csv.DictWriter(
					stream,
					fieldnames=[
						"seconds",
						"ready",
						"unacked",
						"source_rss_mb",
						"worker_rss_mb",
						"accepted",
						"rejected",
						"rate",
						"completed",
						"mean_task_seconds",
					],
				)
				writer.writeheader()
				last_accepted, last_time = 0, started
				while time.monotonic() - started < 180:
					q = (await http.get(queue_url)).json()
					await observed_queue.declare()
					async with sessions() as session:
						rows = (
							await session.scalars(
								select(InferenceTask).where(InferenceTask.id.in_(ids))
							)
						).all()
					stats = await asyncio.gather(
						*[asyncio.to_thread(c.stats, stream=False) for c in containers]
					)
					now = time.monotonic()
					latencies = [r.result["task_seconds"] for r in rows if r.result]
					writer.writerow(
						{
							"seconds": round(now - started, 3),
							"ready": observed_queue.declaration_result.message_count,
							"unacked": q.get("messages_unacknowledged", 0),
							"source_rss_mb": psutil.Process().memory_info().rss / 2**20,
							"worker_rss_mb": sum(
								s["memory_stats"].get("usage", 0) for s in stats
							)
							/ 2**20,
							**progress,
							"rate": (progress["accepted"] - last_accepted)
							/ (now - last_time),
							"completed": len(rows),
							"mean_task_seconds": sum(latencies) / len(latencies)
							if latencies
							else 0,
						}
					)
					stream.flush()
					last_accepted, last_time = progress["accepted"], now
					if producer.done() and len(rows) == count:
						break
					await asyncio.sleep(0.5)
				else:
					raise RuntimeError("Experiment did not drain")
			await producer
			latencies = [r.result["task_seconds"] for r in rows if r.result]
			result = {
				"run_id": str(run_id),
				"queue": settings.queue_name,
				"workers": replicas,
				"limit": limit,
				"accepted": progress["accepted"],
				"completed": len(latencies),
				"rejected_attempts": progress["rejected"],
				"duration_seconds": time.monotonic() - started,
				"mean_task_seconds": sum(latencies) / len(latencies),
				"max_task_seconds": max(latencies),
				"model_versions": sorted(
					{r.result["model_version"] for r in rows if r.result}
				),
			}
			(output / f"{label}.json").write_text(json.dumps(result, indent=2))
			print(label, result, flush=True)
	finally:
		for container in containers:
			await asyncio.to_thread(container.stop, timeout=35)
			await asyncio.to_thread(container.remove)
		await publisher.close()
		await engine.dispose()


async def main():
	output = Path("docs/experiments")
	output.mkdir(exist_ok=True)
	input_path = output / "input.jsonl"
	with (
		Path("data/processed/validation.csv").open(encoding="utf-8") as source,
		input_path.open("w", encoding="utf-8") as target,
	):
		count = 0
		for row in csv.DictReader(source):
			if len(row["text"]) <= 50000:
				target.write(json.dumps({"text": row["text"]}) + "\n")
				count += 1
				if count == 200:
					break
	client = docker.from_env()
	for label, limit, replicas in [
		("unbounded-1", 0, 1),
		("bounded-1", 20, 1),
		("bounded-2", 20, 2),
	]:
		await scenario(label, limit, replicas, client, input_path, output)


if __name__ == "__main__":
	asyncio.run(main())
