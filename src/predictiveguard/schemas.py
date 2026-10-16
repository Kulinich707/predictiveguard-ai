from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LivenessResponse(BaseModel):
    status: Literal["ok"]


class VersionResponse(BaseModel):
    version: str


class ComponentHealth(BaseModel):
    name: str
    status: Literal["healthy", "unavailable"]
    version: str | None
    response_time_ms: float
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    components: list[ComponentHealth]


class ProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    type: Literal["L", "M", "H"]
    air_temperature_k: float = Field(ge=250, le=350, examples=[298.1])
    process_temperature_k: float = Field(ge=250, le=400, examples=[308.6])
    rotational_speed_rpm: float = Field(ge=500, le=3500, examples=[1551])
    torque_nm: float = Field(ge=0, le=100, examples=[42.8])
    tool_wear_min: float = Field(ge=0, le=500, examples=[108])


class ModelMetadata(BaseModel):
    name: str
    alias: str
    version: str
    run_id: str


class ProcessResponse(BaseModel):
    prediction: Literal["no_failure", "failure"]
    failure_probability: float = Field(ge=0, le=1)
    model: ModelMetadata
