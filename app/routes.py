import asyncio
import logging
from time import perf_counter
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from starlette.concurrency import run_in_threadpool

from app.models import InferenceTask, TaskStatus
from app.schemas import TaskAccepted, TaskCreate, TaskView
from app.task_service import create_and_publish_task, get_task

router = APIRouter()

api_router = APIRouter(prefix="/api/v1", tags=["api"])

logger = logging.getLogger(__name__)


@api_router.post("/tasks", response_model=TaskAccepted, status_code=202)
@router.post("/process", response_model=TaskAccepted, status_code=202, deprecated=True)
async def create_task(payload: TaskCreate, request: Request):
	task = InferenceTask(
		id=payload.task_id, text=payload.text, status=TaskStatus.PENDING
	)
	try:
		await create_and_publish_task(
			request.app.state.engine, request.app.state.task_publisher, task
		)
	except Exception as exc:
		logger.warning("task publication failed: %s", type(exc).__name__)
		return JSONResponse(
			status_code=503,
			headers={
				"Retry-After": str(request.app.state.settings.retry_after_seconds)
			},
			content={"detail": "Task queue is full or unavailable; retry later"},
		)
	return TaskAccepted(task_id=task.id, status=task.status)


@api_router.get("/tasks/{task_id}", response_model=TaskView)
async def read_task(task_id: UUID, request: Request):
	task = await get_task(request.app.state.engine, task_id)
	if task is None:
		return JSONResponse(
			status_code=404,
			content={
				"detail": "No persisted result yet (queued, processing, or unknown task ID)"
			},
		)
	return TaskView(
		task_id=task.id, status=task.status, result=task.result, error=task.error
	)


class ProcessInput(BaseModel):
	text: Annotated[
		str,
		StringConstraints(
			strict=True, strip_whitespace=True, min_length=1, max_length=50000
		),
	]


class ProcessOutput(BaseModel):
	label: Literal["real", "fake"]
	model_name: str
	model_version: str


@api_router.post("/predict", response_model=ProcessOutput, tags=["hw3"])
async def process(payload: ProcessInput, request: Request):
	# sklearn.predict синхронный: не выполняем его прямо в event loop.
	predictions = await run_in_threadpool(
		request.app.state.model.predict, [payload.text]
	)
	labels = {0: "real", 1: "fake"}
	return ProcessOutput(
		label=labels[int(predictions[0])],
		model_name=request.app.state.model_name,
		model_version=request.app.state.model_version,
	)


@router.get("/healthz")
async def get_health():
	return {"status": "ok"}


@api_router.get("/version")
async def get_version(request: Request):
	return {"version": request.app.version}


@api_router.get("/health")
async def health(request: Request):
	started = perf_counter()
	postgres = {"status": "ok", "version": None}

	try:
		async with asyncio.timeout(3):
			async with request.app.state.engine.connect() as connection:
				result = await connection.execute(text("SELECT version()"))
				postgres["version"] = result.scalar_one()
	except (SQLAlchemyError, OSError) as exc:
		postgres["status"] = "unavailable"
		logger.warning("PostgreSQL checkfaild with exception %s", type(exc).__name__)

	postgres["response_time_ms"] = round((perf_counter() - started) * 1000, 2)

	healthy = postgres["status"] == "ok"

	return JSONResponse(
		status_code=200 if healthy else 503,
		content={
			"status": "ok" if healthy else "error",
			"postgres": postgres,
		},
	)
