import logging
import tomllib
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request

from app.db import lifespan
from app.logging_config import configure_logging
from app.routes import api_router, router

configure_logging()
logger = logging.getLogger(__name__)

project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"

with project_file.open("rb") as file:
	project = tomllib.load(file)


app = FastAPI(
	title="Fake News API",
	version=project["project"]["version"],
	lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
	started = perf_counter()
	fields = {"method": request.method, "path": request.url.path}
	try:
		response = await call_next(request)
	except Exception as exc:
		logger.error(
			"http_request_failed",
			extra={
				**fields,
				"status_code": 500,
				"duration_ms": round((perf_counter() - started) * 1000, 2),
				"error_type": type(exc).__name__,
			},
		)
		raise
	logger.info(
		"http_request_completed",
		extra={
			**fields,
			"status_code": response.status_code,
			"duration_ms": round((perf_counter() - started) * 1000, 2),
		},
	)
	return response


app.include_router(router)
app.include_router(api_router)
