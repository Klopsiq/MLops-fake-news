import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
	database_url: str
	rabbitmq_url: str
	queue_name: str
	queue_max_length: int
	worker_concurrency: int
	inference_delay_seconds: float
	mlflow_tracking_uri: str
	mlflow_model_name: str
	mlflow_model_alias: str
	publish_timeout_seconds: float
	retry_after_seconds: int
	graceful_timeout_seconds: float
	worker_health_port: int

	@property
	def model_uri(self) -> str:
		return f"models:/{self.mlflow_model_name}@{self.mlflow_model_alias}"


def get_settings() -> Settings:
	return Settings(
		database_url=os.environ["DATABASE_URL"],
		rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"),
		queue_name=os.getenv("TASK_QUEUE", "fake-news.tasks.v2"),
		queue_max_length=int(os.getenv("QUEUE_MAX_LENGTH", "100")),
		worker_concurrency=int(os.getenv("WORKER_CONCURRENCY", "4")),
		inference_delay_seconds=float(os.getenv("INFERENCE_DELAY_SECONDS", "1")),
		mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
		mlflow_model_name=os.getenv("MODEL_NAME", "fake-news-classifier"),
		mlflow_model_alias=os.getenv("MODEL_ALIAS", "champion"),
		publish_timeout_seconds=float(os.getenv("PUBLISH_TIMEOUT_SECONDS", "2")),
		retry_after_seconds=int(os.getenv("RETRY_AFTER_SECONDS", "2")),
		graceful_timeout_seconds=float(os.getenv("GRACEFUL_TIMEOUT_SECONDS", "30")),
		worker_health_port=int(os.getenv("WORKER_HEALTH_PORT", "8000")),
	)
