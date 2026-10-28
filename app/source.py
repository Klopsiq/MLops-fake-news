import argparse
import asyncio
import json
from pathlib import Path
from uuid import UUID, uuid5

from aio_pika.exceptions import AMQPException

from app.config import get_settings
from app.schemas import TaskMessage
from app.task_service import TaskPublisher


async def send_file(publisher, path, run_id, progress=None):
	accepted = rejected = 0
	with Path(path).open(encoding="utf-8") as stream:
		for index, line in enumerate(stream):
			task = TaskMessage(
				task_id=uuid5(run_id, str(index)), text=json.loads(line)["text"]
			)
			delay = 0.02
			while True:
				try:
					await publisher.publish(task)
				except (AMQPException, OSError, TimeoutError):
					rejected += 1
					if progress is not None:
						progress["rejected"] = rejected
					await asyncio.sleep(delay)
					delay = min(delay * 2, 0.2)
				else:
					accepted += 1
					if progress is not None:
						progress["accepted"] = accepted
					break
	return accepted, rejected


async def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("input", help="JSONL file: one {text: ...} object per line")
	parser.add_argument(
		"--run-id",
		type=UUID,
		required=True,
		help="Reuse on restart for stable task IDs",
	)
	args = parser.parse_args()
	publisher = TaskPublisher(get_settings())
	try:
		await publisher.start()
		print(await send_file(publisher, args.input, args.run_id))
	finally:
		await publisher.close()


if __name__ == "__main__":
	asyncio.run(main())
