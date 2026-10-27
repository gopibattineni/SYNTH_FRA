from __future__ import annotations

import json
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

from app.config import SESSIONS_DIR
from app.data_loader import load_training_data
from app.generators.runner import generate_synthetic
from app.models import JobStatus
from app.registry import check_generator_availability, get_generator


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        dataset_id: str,
        generator_ids: list[str],
        n_samples: int,
        seed: int,
    ) -> JobStatus:
        job_id = uuid.uuid4().hex
        job = JobStatus(
            job_id=job_id,
            status="queued",
            dataset_id=dataset_id,
            generator_ids=generator_ids,
            n_samples=n_samples,
            seed=seed,
            message="Queued",
        )
        with self._lock:
            self._jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _update(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            self._jobs[job_id] = job.model_copy(update=kwargs)

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if job is None:
            return

        self._update(job_id, status="running", message="Loading training data", progress=0.05)
        job_dir = SESSIONS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        try:
            info, train_df = load_training_data(job.dataset_id)
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(train_df)
            availability = check_generator_availability()
            results: dict[str, Any] = {}
            total = len(job.generator_ids)

            for idx, generator_id in enumerate(job.generator_ids):
                gen = get_generator(generator_id)
                avail = availability.get(generator_id, {})
                if not avail.get("available", True):
                    raise RuntimeError(avail.get("reason") or f"{gen.name} is not available")

                progress = 0.1 + (0.8 * idx / max(total, 1))
                self._update(
                    job_id,
                    message=f"Training {gen.name} ({idx + 1}/{total})",
                    progress=progress,
                )

                synthetic = generate_synthetic(
                    generator_id,
                    train_df,
                    info,
                    job.n_samples,
                    job.seed,
                )
                synthetic = synthetic.reindex(columns=train_df.columns)
                out_path = job_dir / f"{generator_id}.csv"
                synthetic.to_csv(out_path, index=False)

                quality_score = None
                try:
                    quality = evaluate_quality(
                        real_data=train_df,
                        synthetic_data=synthetic,
                        metadata=metadata,
                    )
                    quality_score = float(quality.get_score())
                except Exception:
                    quality_score = None

                results[generator_id] = {
                    "generator_name": gen.name,
                    "rows": len(synthetic),
                    "columns": list(synthetic.columns),
                    "quality_score": quality_score,
                    "file": str(out_path),
                }

            meta = {
                "dataset_id": job.dataset_id,
                "dataset_name": info.name,
                "target_col": info.target_col,
                "task": info.task,
                "n_samples": job.n_samples,
                "seed": job.seed,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "results": results,
            }
            (job_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

            self._update(
                job_id,
                status="completed",
                progress=1.0,
                message="Generation complete",
                results=results,
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                progress=1.0,
                message="Generation failed",
                error=str(exc),
            )
            (job_dir / "error.txt").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )

    def load_synthetic_csv(self, job_id: str, generator_id: str) -> pd.DataFrame:
        path = SESSIONS_DIR / job_id / f"{generator_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"No output for generator {generator_id} in job {job_id}")
        return pd.read_csv(path)


job_manager = JobManager()
