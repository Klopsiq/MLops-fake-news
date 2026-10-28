from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, StringConstraints

from app.models import TaskStatus


class TaskCreate(BaseModel):
	task_id: UUID = Field(default_factory=uuid4)
	text: Annotated[
		str,
		StringConstraints(
			strict=True, strip_whitespace=True, min_length=1, max_length=50000
		),
	]


class TaskMessage(TaskCreate):
	submitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskAccepted(BaseModel):
	task_id: UUID
	status: TaskStatus


class TaskView(TaskAccepted):
	result: dict[str, Any] | None = None
	error: str | None = None
