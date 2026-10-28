import enum
import uuid
from datetime import datetime

from sqlalchemy import DateTime, Enum, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
	pass


class TaskStatus(str, enum.Enum):
	PENDING = "PENDING"
	PROCESSING = "PROCESSING"
	COMPLETED = "COMPLETED"
	FAILED = "FAILED"


class TaskAttempt(Base):
	__tablename__ = "task_attempts"
	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
	count: Mapped[int] = mapped_column(Integer, nullable=False)


class InferenceTask(Base):
	__tablename__ = "inference_tasks"

	id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
	text: Mapped[str] = mapped_column(Text)
	status: Mapped[TaskStatus] = mapped_column(
		Enum(TaskStatus, name="task_status"), default=TaskStatus.PENDING, index=True
	)
	result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
	error: Mapped[str | None] = mapped_column(String(200), nullable=True)
	created_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), server_default=func.now()
	)
	updated_at: Mapped[datetime] = mapped_column(
		DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
	)
