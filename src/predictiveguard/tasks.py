from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from predictiveguard.schemas import ProcessRequest


class TaskMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: UUID = Field(default_factory=uuid4)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    features: ProcessRequest

    def wire(self) -> dict:
        return self.model_dump(mode="json")
