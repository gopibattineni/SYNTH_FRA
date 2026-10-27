from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.config import STATIC_DIR
from app.data_loader import summarize_dataset
from app.jobs import job_manager
from app.models import (
    BenchmarkOverview,
    DatasetSummary,
    GenerateRequest,
    GeneratorSummary,
    JobStatus,
    MetricInfo,
    MetricRow,
    PreviewResponse,
)
from app.registry import (
    check_generator_availability,
    get_datasets,
    get_generators,
)
from app import results_loader

app = FastAPI(
    title="SYNTH Benchmark Webapp",
    description=(
        "Explore 10-seed multi-generator results (utility, fidelity, privacy) "
        "and generate synthetic tabular data across 15 datasets."
    ),
    version="1.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
  index_path = STATIC_DIR / "index.html"
  return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/benchmark/overview", response_model=BenchmarkOverview)
def benchmark_overview() -> BenchmarkOverview:
    try:
        return BenchmarkOverview(**results_loader.get_benchmark_overview())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/metrics", response_model=list[MetricInfo])
def benchmark_metric_catalog(
    task: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
) -> list[MetricInfo]:
    try:
        return [MetricInfo(**m) for m in results_loader.list_metrics(task=task, category=category)]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/rows", response_model=list[MetricRow])
def benchmark_rows(
    task: Optional[str] = Query(default=None),
    category: Optional[str] = Query(default=None),
    dataset: Optional[str] = Query(default=None),
    generator: Optional[str] = Query(default=None),
    metric: Optional[str] = Query(default=None),
) -> list[MetricRow]:
    try:
        rows = results_loader.query_metrics(
            task=task,
            category=category,
            dataset=dataset,
            generator=generator,
            metric=metric,
        )
        return [MetricRow(**r) for r in rows]
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/matrix")
def benchmark_matrix(
    task: str = Query(..., pattern="^(classification|regression)$"),
    metric: str = Query(...),
) -> dict[str, Any]:
    try:
        return results_loader.metric_matrix(task=task, metric=metric)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/ranking")
def benchmark_ranking(
    task: str = Query(..., pattern="^(classification|regression)$"),
    metric: str = Query(...),
) -> list[dict[str, Any]]:
    try:
        return results_loader.generator_ranking(task=task, metric=metric)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmark/table")
def benchmark_wide_table(
    task: str = Query(..., pattern="^(classification|regression)$"),
    category: str = Query(..., pattern="^(Utility|Fidelity|Privacy|Compute)$"),
    dataset: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    try:
        return results_loader.wide_category_table(task=task, category=category, dataset=dataset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets", response_model=list[DatasetSummary])
def list_datasets() -> list[DatasetSummary]:
    summaries: list[DatasetSummary] = []
    for ds in get_datasets():
        try:
            summaries.append(summarize_dataset(ds.id))
        except Exception:
            summaries.append(
                DatasetSummary(
                    id=ds.id,
                    name=ds.name,
                    task=ds.task,
                    target_col=ds.target_col,
                    row_count=0,
                    column_count=0,
                    description=ds.description,
                )
            )
    return summaries


@app.get("/api/generators", response_model=list[GeneratorSummary])
def list_generators() -> list[GeneratorSummary]:
    availability = check_generator_availability()
    out: list[GeneratorSummary] = []
    for gen in get_generators():
        avail = availability.get(gen.id, {})
        out.append(
            GeneratorSummary(
                id=gen.id,
                name=gen.name,
                family=gen.family,
                description=gen.description,
                available=bool(avail.get("available", True)),
                reason=avail.get("reason"),
                auto_setup=bool(avail.get("auto_setup", False)),
            )
        )
    return out


@app.post("/api/generate", response_model=JobStatus)
def start_generation(request: GenerateRequest) -> JobStatus:
    dataset_ids = {ds.id for ds in get_datasets()}
    if request.dataset_id not in dataset_ids:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: {request.dataset_id}")

    generator_ids = {g.id for g in get_generators()}
    for gid in request.generator_ids:
        if gid not in generator_ids:
            raise HTTPException(status_code=400, detail=f"Unknown generator: {gid}")

    availability = check_generator_availability()
    for gid in request.generator_ids:
        avail = availability.get(gid, {})
        if not avail.get("available", True):
            raise HTTPException(
                status_code=400,
                detail=avail.get("reason") or f"Generator {gid} is not available",
            )

    return job_manager.create_job(
        dataset_id=request.dataset_id,
        generator_ids=request.generator_ids,
        n_samples=request.n_samples,
        seed=request.seed,
    )


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/preview/{generator_id}", response_model=PreviewResponse)
def preview_result(job_id: str, generator_id: str, limit: int = 20) -> PreviewResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job is not complete")

    try:
        df = job_manager.load_synthetic_csv(job_id, generator_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result_meta = job.results.get(generator_id, {})
    preview = df.head(limit)
    return PreviewResponse(
        columns=list(df.columns),
        rows=preview.where(preview.notna(), None).to_dict(orient="records"),
        total_rows=len(df),
        quality_score=result_meta.get("quality_score"),
    )


@app.get("/api/jobs/{job_id}/download/{generator_id}")
def download_result(job_id: str, generator_id: str) -> FileResponse:
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed":
        raise HTTPException(status_code=400, detail="Job is not complete")

    path = Path(job.results.get(generator_id, {}).get("file", ""))
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Output file not found")

    filename = f"synth_{job.dataset_id}_{generator_id}.csv"
    return FileResponse(path, media_type="text/csv", filename=filename)
