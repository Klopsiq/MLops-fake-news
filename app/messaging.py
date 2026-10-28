from faststream.rabbit import ExchangeType, QueueType, RabbitExchange, RabbitQueue

from app.config import Settings

DLX_NAME = "fake-news.tasks.dlx"
DLQ_NAME = "fake-news.tasks.dlq"
DLQ_ROUTING_KEY = "failed"


def task_queue(settings: Settings) -> RabbitQueue:
	arguments = {
		"x-overflow": "reject-publish",
		"x-dead-letter-exchange": DLX_NAME,
		"x-dead-letter-routing-key": DLQ_ROUTING_KEY,
	}
	if settings.queue_max_length > 0:
		arguments["x-max-length"] = settings.queue_max_length
	return RabbitQueue(
		settings.queue_name,
		durable=True,
		queue_type=QueueType.CLASSIC,
		arguments=arguments,
	)


def dead_letter_exchange() -> RabbitExchange:
	return RabbitExchange(DLX_NAME, type=ExchangeType.DIRECT, durable=True)


def dead_letter_queue() -> RabbitQueue:
	return RabbitQueue(DLQ_NAME, durable=True, routing_key=DLQ_ROUTING_KEY)
