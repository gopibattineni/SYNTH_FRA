from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    id: str
    name: str
    task: Literal["classification", "regression"]
    target_col: str
    row_count: int
    column_count: int
    description: str = ""


class GeneratorSummary(BaseModel):
    id: str
    name: str
    family: str
    description: str
    available: bool = True
    reason: str | None = None
    auto_setup: bool = False


class GenerateRequest(BaseModel):
    dataset_id: str
    generator_ids: list[str] = Field(min_length=1)
    n_samples: int = Field(default=1000, ge=10, le=10000)
    seed: int = Field(default=42, ge=0)


class JobStatus(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    dataset_id: str
    generator_ids: list[str]
    n_samples: int
    seed: int
    progress: float = 0.0
    message: str = ""
    results: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int
    quality_score: float | None = None


class BenchmarkOverview(BaseModel):
    source: str
    n_generator_seeds: int
    generator_seeds: list[int]
    split_seed: int
    aggregation: str
    tasks: list[str]
    n_datasets: int
    datasets: list[str]
    n_generators: int
    generators: list[str]
    categories: list[str]
    n_metric_rows: int
    n_runs: int | None = None
    files: dict[str, str | None] = Field(default_factory=dict)


class MetricInfo(BaseModel):
    id: str
    label: str
    category: str
    better: str


class MetricRow(BaseModel):
    task: str
    dataset: str
    generator: str
    category: str
    metric: str
    mean: float | None = None
    sd: float | None = None
    mean_sd: str | None = None
    n_seeds: int = 0
