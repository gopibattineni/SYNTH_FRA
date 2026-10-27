#!/usr/bin/env python3
"""Recompute missing One-to-One Greedy Mahalanobis means (Wine + Air Quality).

Fills the 6 blank heatmap cells that lack usable notebook greedy outputs:
  Wine × {CTGAN, CopulaGAN, TVAE, GaussianCopula}
  AirQuality × {CTGAN, CopulaGAN}

Multi-GPU: each (dataset, generator) job is assigned to one CUDA device and
Mahalanobis pairwise distances are computed on that GPU (chunked).

Expected synthetic CSVs (place when available)
---------------------------------------------
  recompute_inputs/Wine/CTGAN.csv
  recompute_inputs/Wine/CopulaGAN.csv
  recompute_inputs/Wine/TVAE.csv
  recompute_inputs/Wine/GaussianCopula.csv
  recompute_inputs/AirQuality/CTGAN.csv
  recompute_inputs/AirQuality/CopulaGAN.csv

Each CSV must contain the same numeric feature columns used in the SDV
notebooks (target excluded). Column names may use spaces or underscores.

Usage
-----
  # Status / dry-run (default): report which inputs are present
  python recompute_missing_mahalanobis_greedy.py

  # Run all available jobs on all visible GPUs
  python recompute_missing_mahalanobis_greedy.py --run

  # Restrict GPUs / force full-size matching against existing Hungarian means
  CUDA_VISIBLE_DEVICES=0,1,3,5 python recompute_missing_mahalanobis_greedy.py \\
      --run --mode full --gpus 0,1,2,3 --patch-summary --rebuild-heatmap

  # Sampled greedy (notebook-style MAX_ROWS=2000) for a quick check
  python recompute_missing_mahalanobis_greedy.py --run --mode sampled --max-rows 2000
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
INPUT_DIR = BASE / "recompute_inputs"
OUT_DIR = BASE / "recompute_outputs"
SUMMARY_CSV = BASE / "extracted_results" / "hungarian_vs_greedy_summary.csv"
LONG_MAHA = BASE / "extracted_results" / "mahalanobis_distance_results.csv"

JOBS = [
    # dataset_key, generator, target_col, real_loader_key, hungarian_pref
    ("Wine", "CTGAN", "quality", "wine", 14.60013448951508),
    ("Wine", "CopulaGAN", "quality", "wine", 14.97249458978774),
    ("Wine", "TVAE", "quality", "wine", 13.65101514825886),
    ("Wine", "GaussianCopula", "quality", "wine", 15.50733446018729),
    ("AirQuality", "CTGAN", "CO(GT)", "air", 145.4258331068809),
    ("AirQuality", "CopulaGAN", "CO(GT)", "air", 42.25175791284352),
]


def _norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(c).strip().lower())


def align_numeric_features(
    real: pd.DataFrame, synth: pd.DataFrame, target: str | None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Align feature columns between real and synthetic (drop target)."""
    real = real.copy()
    synth = synth.copy()
    real.columns = [str(c) for c in real.columns]
    synth.columns = [str(c) for c in synth.columns]

    # Prefer numeric columns from real, excluding target.
    real_num = real.select_dtypes(include=[np.number]).copy()
    if target is not None:
        drop = [c for c in real_num.columns if _norm_col(c) == _norm_col(target)]
        real_num = real_num.drop(columns=drop, errors="ignore")

    # Map synth columns by normalized name.
    synth_map = {_norm_col(c): c for c in synth.columns}
    cols: list[str] = []
    synth_cols: list[str] = []
    for c in real_num.columns:
        key = _norm_col(c)
        if key in synth_map:
            cols.append(c)
            synth_cols.append(synth_map[key])

    if not cols:
        raise ValueError(
            f"No overlapping numeric features. Real={list(real_num.columns)[:12]} "
            f"Synth={list(synth.columns)[:12]}"
        )

    Xr = real_num[cols].to_numpy(dtype=np.float64)
    Xs = synth[synth_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(Xr).any() or np.isnan(Xs).any():
        # Median-impute remaining NaNs column-wise using real medians.
        med = np.nanmedian(Xr, axis=0)
        for j in range(Xr.shape[1]):
            Xr[np.isnan(Xr[:, j]), j] = med[j]
            Xs[np.isnan(Xs[:, j]), j] = med[j]
    return Xr, Xs, cols


def load_real_wine() -> pd.DataFrame:
    """Load combined red+white Wine Quality (6497 rows) like the SDV notebook."""
    try:
        from ucimlrepo import fetch_ucirepo

        wine = fetch_ucirepo(id=186)
        data = pd.concat([wine.data.features, wine.data.targets], axis=1)
        return data.reset_index(drop=True)
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError(
            "Could not load Wine via ucimlrepo (id=186). "
            "Install ucimlrepo or provide recompute_inputs/Wine/_real.csv"
        ) from exc


def load_real_air(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Load Air Quality like the SDV notebook (1000-row sample, seed=42)."""
    candidates = [
        INPUT_DIR / "AirQuality" / "_real.csv",
        ROOT / "Generators" / "Other GANS" / "air_quality_data_1000.csv",
        ROOT / "Generators" / "Experiment with utility data leak" / "12. Air Quality" / "air_quality_train.csv",
    ]
    for path in candidates:
        if path.exists():
            data = pd.read_csv(path)
            # If this is already the 1000-row notebook sample, use as-is.
            if len(data) == n_samples and path.name == "air_quality_data_1000.csv":
                return data.reset_index(drop=True)
            data = data.drop(columns=["Date", "Time", "date_time"], errors="ignore")
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            data = data.replace(-200, np.nan)
            for col in data.columns:
                data[col] = data[col].fillna(data[col].median())
            data = data.dropna().reset_index(drop=True)
            if len(data) > n_samples:
                data = data.sample(n=n_samples, random_state=seed).reset_index(drop=True)
            return data

    try:
        from ucimlrepo import fetch_ucirepo

        air = fetch_ucirepo(id=360)
        data = air.data.features.copy()
        data = data.drop(columns=["Date", "Time"], errors="ignore")
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
        data = data.replace(-200, np.nan)
        for col in data.columns:
            data[col] = data[col].fillna(data[col].median())
        data = data.dropna().reset_index(drop=True)
        return data.sample(n=min(n_samples, len(data)), random_state=seed).reset_index(drop=True)
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError(
            "Could not load Air Quality real data. Place CSV at "
            f"{INPUT_DIR / 'AirQuality' / '_real.csv'}"
        ) from exc


def fit_inv_cov(real: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    cov = np.cov(real, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    p = cov.shape[0]
    try:
        return np.linalg.inv(cov + reg * np.eye(p))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov + reg * np.eye(p))


def greedy_one_to_one_fast(D: np.ndarray) -> tuple[float, float, int]:
    """Notebook-compatible greedy matching via sorted edge list."""
    n_rows, n_cols = D.shape
    n_match = min(n_rows, n_cols)
    used_r = np.zeros(n_rows, dtype=bool)
    used_c = np.zeros(n_cols, dtype=bool)
    order = np.argsort(D, axis=None, kind="mergesort")
    dists: list[float] = []
    for idx in order:
        i = int(idx // n_cols)
        j = int(idx % n_cols)
        if used_r[i] or used_c[j]:
            continue
        used_r[i] = True
        used_c[j] = True
        dists.append(float(D[i, j]))
        if len(dists) == n_match:
            break
    arr = np.asarray(dists, dtype=np.float64)
    return float(arr.sum()), float(arr.mean()), int(len(arr))


def pairwise_mahalanobis_gpu(
    real: np.ndarray,
    synth: np.ndarray,
    inv_cov: np.ndarray,
    device: str,
    row_chunk: int = 512,
) -> np.ndarray:
    """Chunked Mahalanobis distance matrix on one CUDA device."""
    import torch

    torch.cuda.set_device(device)
    VI = torch.as_tensor(inv_cov, dtype=torch.float32, device=device)
    S = torch.as_tensor(synth, dtype=torch.float32, device=device)
    n_r, n_s = real.shape[0], synth.shape[0]
    out = np.empty((n_r, n_s), dtype=np.float32)

    # Precompute whitened synth: S @ L where VI = L L^T via cholesky of VI if SPD,
    # else use VI^{1/2} via eigh. Safer: d^2 = (x-y)^T VI (x-y).
    # Expand: x^T VI x + y^T VI y - 2 x^T VI y
    Sy = S @ VI  # (n_s, p)
    y_term = (Sy * S).sum(dim=1)  # (n_s,)

    for start in range(0, n_r, row_chunk):
        end = min(start + row_chunk, n_r)
        R = torch.as_tensor(real[start:end], dtype=torch.float32, device=device)
        Rx = R @ VI
        x_term = (Rx * R).sum(dim=1, keepdim=True)  # (chunk, 1)
        cross = R @ Sy.T  # (chunk, n_s)
        d2 = x_term + y_term.unsqueeze(0) - 2.0 * cross
        d2 = torch.clamp(d2, min=0.0)
        D = torch.sqrt(d2)
        D = torch.nan_to_num(D, nan=1e10, posinf=1e10, neginf=1e10)
        out[start:end] = D.detach().cpu().numpy()
        del R, Rx, x_term, cross, d2, D
        torch.cuda.empty_cache()
    return out


def pairwise_mahalanobis_cpu(real: np.ndarray, synth: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    from scipy.spatial.distance import cdist

    D = cdist(real, synth, metric="mahalanobis", VI=inv_cov)
    return np.nan_to_num(np.asarray(D, dtype=np.float64), nan=1e10, posinf=1e10, neginf=1e10)


@dataclass
class JobResult:
    dataset: str
    generator: str
    status: str
    greedy_mean: float | None = None
    greedy_total: float | None = None
    n_matches: int | None = None
    n_real: int | None = None
    n_synth: int | None = None
    n_features: int | None = None
    hungarian_reference: float | None = None
    mode: str | None = None
    device: str | None = None
    seconds: float | None = None
    note: str | None = None
    synth_path: str | None = None


def _worker(payload: dict) -> dict:
    """Process-pool worker: one (dataset, generator) on one GPU."""
    t0 = time.time()
    dataset = payload["dataset"]
    generator = payload["generator"]
    target = payload["target"]
    mode = payload["mode"]
    max_rows = payload["max_rows"]
    seed = payload["seed"]
    device_id = payload["device_id"]
    synth_path = Path(payload["synth_path"])
    hung_ref = payload["hungarian_reference"]
    row_chunk = payload["row_chunk"]

    try:
        if dataset == "Wine":
            real_override = INPUT_DIR / "Wine" / "_real.csv"
            real_df = pd.read_csv(real_override) if real_override.exists() else load_real_wine()
        else:
            real_df = load_real_air()
        synth_df = pd.read_csv(synth_path)
        Xr, Xs, cols = align_numeric_features(real_df, synth_df, target)
        inv = fit_inv_cov(Xr)

        rng = np.random.default_rng(seed)
        if mode == "sampled":
            n = min(len(Xr), len(Xs), max_rows)
            ri = rng.choice(len(Xr), size=n, replace=False) if len(Xr) > n else np.arange(len(Xr))
            si = rng.choice(len(Xs), size=n, replace=False) if len(Xs) > n else np.arange(len(Xs))
            Xr_m, Xs_m = Xr[ri], Xs[si]
        else:
            # Full one-to-one on min(n_real, n_synth) using all rows available
            n = min(len(Xr), len(Xs))
            Xr_m, Xs_m = Xr[:n], Xs[:n]

        use_gpu = device_id is not None and device_id >= 0
        if use_gpu:
            # Pin this child to a single physical GPU before importing torch.
            os.environ["CUDA_VISIBLE_DEVICES"] = str(int(device_id))
            D = pairwise_mahalanobis_gpu(Xr_m, Xs_m, inv, device="cuda:0", row_chunk=row_chunk)
            D = D.astype(np.float64, copy=False)
        else:
            D = pairwise_mahalanobis_cpu(Xr_m, Xs_m, inv)

        total, mean, n_match = greedy_one_to_one_fast(D)
        note = None
        if hung_ref is not None and mean + 1e-9 < hung_ref * 0.5:
            note = (
                "WARNING: greedy mean << known Hungarian mean; "
                "check feature alignment / covariance / synth table"
            )
        return asdict(
            JobResult(
                dataset=dataset,
                generator=generator,
                status="ok",
                greedy_mean=mean,
                greedy_total=total,
                n_matches=n_match,
                n_real=int(len(Xr_m)),
                n_synth=int(len(Xs_m)),
                n_features=int(len(cols)),
                hungarian_reference=hung_ref,
                mode=mode,
                device=f"cuda:{device_id}" if use_gpu else "cpu",
                seconds=time.time() - t0,
                note=note,
                synth_path=str(synth_path),
            )
        )
    except Exception as exc:
        return asdict(
            JobResult(
                dataset=dataset,
                generator=generator,
                status="error",
                hungarian_reference=hung_ref,
                mode=mode,
                device=f"cuda:{device_id}" if device_id is not None else "cpu",
                seconds=time.time() - t0,
                note=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                synth_path=str(synth_path),
            )
        )


def list_gpus(requested: str | None) -> list[int]:
    """Return physical GPU ids to use.

    Honours CUDA_VISIBLE_DEVICES if set (recommended: 0,1,3,5 on this host).
    ``--gpus`` selects among the *visible* devices (0-based within that list),
    then maps back to physical ids for child processes.
    """
    import torch

    n_visible = torch.cuda.device_count()
    if n_visible <= 0:
        return []

    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        physical = [int(x) for x in env.split(",") if x.strip() != ""]
    else:
        physical = list(range(n_visible))

    if requested is None or requested.strip().lower() in {"all", ""}:
        local = list(range(min(n_visible, len(physical))))
    else:
        local = [int(x) for x in requested.split(",") if x.strip() != ""]
        local = [i for i in local if 0 <= i < n_visible]

    # Map visible ordinal → physical id for child CUDA_VISIBLE_DEVICES pinning
    return [physical[i] if i < len(physical) else i for i in local]


def discover_jobs() -> list[dict]:
    rows = []
    for dataset, generator, target, _key, hung in JOBS:
        path = INPUT_DIR / dataset / f"{generator}.csv"
        rows.append(
            {
                "dataset": dataset,
                "generator": generator,
                "target": target,
                "synth_path": path,
                "present": path.exists(),
                "hungarian_reference": hung,
            }
        )
    return rows


def patch_summary(results: pd.DataFrame) -> Path:
    """Write greedy means into hungarian_vs_greedy_summary.csv for ok jobs."""
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(SUMMARY_CSV)
    summary = pd.read_csv(SUMMARY_CSV)
    patched = 0
    for r in results.itertuples():
        if r.status != "ok" or pd.isna(r.greedy_mean):
            continue
        mask = (summary["Dataset"] == r.dataset) & (summary["Generator"] == r.generator)
        if not mask.any():
            continue
        summary.loc[mask, "Greedy_Mahalanobis"] = float(r.greedy_mean)
        if "Hungarian_Mahalanobis" in summary.columns:
            hung = summary.loc[mask, "Hungarian_Mahalanobis"]
            summary.loc[mask, "Delta_Mahalanobis"] = (
                summary.loc[mask, "Greedy_Mahalanobis"] - hung
            )
        summary.loc[mask, "nb_src"] = f"recompute:{Path(str(r.synth_path)).name}"
        summary.loc[mask, "hung_src"] = summary.loc[mask, "hung_src"].fillna("recompute_keep")
        patched += 1
    out = OUT_DIR / "hungarian_vs_greedy_summary_patched.csv"
    summary.to_csv(out, index=False)
    # Also overwrite primary summary used by figure scripts
    summary.to_csv(SUMMARY_CSV, index=False)

    # Append long-form greedy rows for extract_and_plot consumers
    long_rows = []
    for r in results.itertuples():
        if r.status != "ok":
            continue
        long_rows.append(
            {
                "Dataset": {"Wine": "6. Wine dataset", "AirQuality": "12. Air Quality"}[r.dataset],
                "Dataset_Short": r.dataset,
                "Generator": r.generator,
                "Num_Matches": r.n_matches,
                "Source_File": r.synth_path,
                "Mapping_Method": "One-to-One Greedy",
                "Metric": "Mahalanobis Distance",
                "Mean": r.greedy_mean,
                "Source_Note": "recompute_missing_mahalanobis_greedy",
                "Median": np.nan,
                "Min": np.nan,
                "Max": np.nan,
                "Std": np.nan,
            }
        )
    if long_rows and LONG_MAHA.exists():
        long = pd.read_csv(LONG_MAHA)
        add = pd.DataFrame(long_rows)
        key = ["Dataset_Short", "Generator", "Mapping_Method", "Metric"]
        long = pd.concat([long, add], ignore_index=True)
        long = long.drop_duplicates(subset=key, keep="last")
        long.to_csv(LONG_MAHA, index=False)

    print(f"Patched {patched} summary rows → {SUMMARY_CSV}")
    return SUMMARY_CSV


def rebuild_heatmap() -> None:
    import runpy

    print("Rebuilding Mahalanobis greedy-vs-Hungarian figures…")
    runpy.run_path(str(BASE / "create_mahalanobis_greedy_vs_hungarian.py"), run_name="__main__")


def print_status(jobs: list[dict], gpus: list[int]) -> None:
    print("=" * 72)
    print("Missing Mahalanobis greedy recompute — status")
    print("=" * 72)
    print(f"Input dir : {INPUT_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print(f"GPUs      : {gpus if gpus else 'none (CPU fallback)'}")
    print()
    present = 0
    for j in jobs:
        mark = "READY" if j["present"] else "MISSING"
        if j["present"]:
            present += 1
        print(
            f"  [{mark:7}] {j['dataset']:11} / {j['generator']:14}  "
            f"→ {j['synth_path']}"
        )
    print()
    print(f"{present}/{len(jobs)} synthetic tables found.")
    if present < len(jobs):
        print(
            "\nPlace missing CSVs under recompute_inputs/<Dataset>/<Generator>.csv\n"
            "Then run:\n"
            "  CUDA_VISIBLE_DEVICES=0,1,3,5 python recompute_missing_mahalanobis_greedy.py "
            "--run --mode full --patch-summary --rebuild-heatmap"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="store_true", help="Execute recompute for available inputs")
    parser.add_argument(
        "--mode",
        choices=["full", "sampled"],
        default="full",
        help="full = greedy on all rows (preferred for heatmap); sampled = notebook MAX_ROWS style",
    )
    parser.add_argument("--max-rows", type=int, default=2000, help="Only for --mode sampled")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpus",
        type=str,
        default="all",
        help="Comma-separated CUDA device indices within CUDA_VISIBLE_DEVICES, or 'all'",
    )
    parser.add_argument("--row-chunk", type=int, default=256, help="GPU Mahalanobis row chunk size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if GPUs are visible")
    parser.add_argument("--patch-summary", action="store_true", help="Write greedy means into extracted summary")
    parser.add_argument("--rebuild-heatmap", action="store_true", help="Regenerate Fig04 after patching")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "Wine").mkdir(parents=True, exist_ok=True)
    (INPUT_DIR / "AirQuality").mkdir(parents=True, exist_ok=True)

    jobs = discover_jobs()
    gpus = [] if args.cpu else list_gpus(args.gpus)
    print_status(jobs, gpus)

    if not args.run:
        return

    ready = [j for j in jobs if j["present"]]
    if not ready:
        print("\nNothing to run — no synthetic CSVs found.")
        return

    payloads = []
    for i, j in enumerate(ready):
        device_id = None if not gpus else gpus[i % len(gpus)]
        payloads.append(
            {
                "dataset": j["dataset"],
                "generator": j["generator"],
                "target": j["target"],
                "synth_path": str(j["synth_path"]),
                "hungarian_reference": j["hungarian_reference"],
                "mode": args.mode,
                "max_rows": args.max_rows,
                "seed": args.seed,
                "device_id": device_id if device_id is not None else -1,
                "row_chunk": args.row_chunk,
            }
        )

    print(f"\nLaunching {len(payloads)} job(s) "
          f"({'multi-GPU' if gpus else 'CPU'}, mode={args.mode})…")
    results: list[dict] = []
    # One process per job so each can pin CUDA_VISIBLE_DEVICES independently.
    max_workers = max(1, len(gpus) if gpus else min(2, len(payloads)))
    # Spawn start method required for CUDA in child processes
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
        futs = {ex.submit(_worker, p): p for p in payloads}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            status = res["status"]
            msg = (
                f"  [{status}] {res['dataset']}/{res['generator']} "
                f"device={res.get('device')} "
            )
            if status == "ok":
                msg += f"greedy_mean={res['greedy_mean']:.6f}  n={res['n_matches']}  {res['seconds']:.1f}s"
                if res.get("note"):
                    msg += f"  ({res['note']})"
            else:
                msg += str(res.get("note", "")).splitlines()[0]
            print(msg)

    out_csv = OUT_DIR / f"greedy_recompute_{args.mode}.csv"
    rdf = pd.DataFrame(results)
    rdf.to_csv(out_csv, index=False)
    (OUT_DIR / f"greedy_recompute_{args.mode}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {out_csv}")

    ok = rdf[rdf["status"] == "ok"]
    if len(ok) and args.patch_summary:
        patch_summary(ok)
        if args.rebuild_heatmap:
            rebuild_heatmap()
    elif len(ok):
        print(
            "Results ready. Re-run with --patch-summary --rebuild-heatmap "
            "to update the heatmap."
        )


if __name__ == "__main__":
    main()
