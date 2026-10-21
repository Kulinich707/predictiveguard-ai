from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "predictiveguard"
    postgres_user: str = "predictiveguard"
    postgres_password: str = "predictiveguard"
    postgres_connect_timeout_seconds: float = 3.0
    postgres_command_timeout_seconds: float = 3.0
    postgres_health_timeout_seconds: float = 5.0

    mlflow_tracking_uri: str = "http://127.0.0.1:5050"
    model_name: str = "PredictiveGuardFailureModel"
    model_alias: str = "champion"

    rabbit_url: str = "amqp://guest:guest@localhost:5672/"
    task_queue: str = "predictiveguard.tasks"
    queue_max_length: int = Field(default=100, ge=0)
    prefetch_count: int = Field(default=1, ge=1)
    worker_concurrency: int = Field(default=1, ge=1)
    inference_delay_seconds: float = Field(default=0.5, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    publish_timeout_seconds: float = Field(default=1.0, gt=0)
    retry_delay_seconds: float = Field(default=0.1, gt=0)
    shutdown_timeout_seconds: float = Field(default=30, gt=0)
    worker_health_port: int = 8001

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def model_uri(self) -> str:
        return f"models:/{self.model_name}@{self.model_alias}"


settings = Settings()
