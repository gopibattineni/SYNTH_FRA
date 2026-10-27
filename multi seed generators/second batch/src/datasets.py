"""Dataset loading for Batch 1 (reads SYNTH/Datasets + UCI; does not modify originals)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .paths import DATASETS_DIR, SYNTH_ROOT, ensure_task_dirs
from .preprocessing import mark_missing, prepare_frames
from .splitting import make_split


def _resolve_local(rel: Optional[str]) -> Optional[Path]:
    if not rel:
        return None
    p = SYNTH_ROOT / rel
    return p if p.exists() else None


def _normalize_target(df: pd.DataFrame, target: str) -> str:
    if target in df.columns:
        return target
    lower = {c.lower(): c for c in df.columns}
    if target.lower() in lower:
        return lower[target.lower()]
    raise KeyError(f"Target {target!r} not in columns {list(df.columns)}")


def load_raw(cfg: dict[str, Any]) -> pd.DataFrame:
    """Load raw table using local cache when present, else UCI."""
    local = _resolve_local(cfg.get("local_csv")) or _resolve_local(cfg.get("local_xlsx"))
    sep = cfg.get("csv_sep", ",")

    if local is not None:
        if local.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(local)
        else:
            df = pd.read_csv(local, sep=sep)
    elif cfg.get("uci_id") is not None:
        from ucimlrepo import fetch_ucirepo

        repo = fetch_ucirepo(id=int(cfg["uci_id"]))
        X = repo.data.features
        y = repo.data.targets
        df = pd.concat([X, y], axis=1)
        # Align target name if UCI uses a different header
        if cfg["target"] not in df.columns:
            # last column is typically the target
            df = df.rename(columns={df.columns[-1]: cfg["target"]})
    else:
        raise FileNotFoundError(f"No local file or uci_id for dataset {cfg['id']}")

    # Drop configured columns
    for c in cfg.get("drop_cols") or []:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Also drop any remaining date/time columns (name or dtype).
    target_name = cfg.get("target")
    dt_drop: list[str] = []
    for c in list(df.columns):
        if target_name and c == target_name:
            continue
        cl = str(c).lower()
        if any(k in cl for k in ("date", "time", "timestamp", "datetime")):
            dt_drop.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            dt_drop.append(c)
    if dt_drop:
        df = df.drop(columns=dt_drop)

    # Resolve target name
    target = _normalize_target(df, cfg["target"])
    cfg = dict(cfg)
    cfg["target"] = target
    if dt_drop:
        cfg["dropped_datetime_cols"] = dt_drop

    # Alzheimer label map
    if cfg.get("label_map") and target in df.columns:
        df[target] = df[target].map(lambda x: cfg["label_map"].get(x, x))

    # Adult / similar: normalize binary income labels from mixed UCI train+test exports
    if target in df.columns and cfg.get("id") == "adult":
        s = df[target].astype(str).str.strip().str.replace(r"\.$", "", regex=True)
        df[target] = s

    # Drop all-null columns
    df = df.dropna(axis=1, how="all")
    return df, cfg


def load_and_split(
    cfg: dict[str, Any],
    test_size: float = 0.2,
    split_seed: int = 42,
    use_cache: bool = True,
) -> dict[str, Any]:
    """
    Leakage-safe Stage 1 order (matches corrected SYNTH fig1 Stage 1):

      raw → mark missing ('?'→NaN, no stats)
         → fixed train/test split (entity-aware when needed)
         → fit imputer on Real-Train ONLY
         → transform Real-Train + Real-Test

    Split is FIXED across generator seeds (cached under <task>/cache/splits/).
    """
    paths = ensure_task_dirs(cfg["task"])
    SPLITS_DIR = paths["splits"]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    # bump cache key when label/impute logic changes
    cache_key = f"{cfg['id']}__split{split_seed}__ts{test_size}__v5_uniform1000.pkl"
    cache_path = SPLITS_DIR / cache_key
    meta_path = SPLITS_DIR / cache_key.replace(".pkl", ".json")

    if use_cache and cache_path.exists() and meta_path.exists():
        try:
            obj = pd.read_pickle(cache_path)
            with meta_path.open() as f:
                meta = json.load(f)
            return {
                "train": obj["train"],
                "test": obj["test"],
                "cfg": obj["cfg"],
                "meta": meta,
                "from_cache": True,
            }
        except Exception:
            # Stale pickle from another numpy/pandas — rebuild from source.
            pass

    raw, cfg = load_raw(cfg)
    n_raw = len(raw)

    # Entity counts before dropping group col
    group_col = cfg.get("group_col")
    n_entities = int(raw[group_col].nunique()) if group_col and group_col in raw.columns else None

    # Mark missing only — NEVER compute mean/median/mode on full data
    work = mark_missing(raw)

    # Optional subsample BEFORE split (fixed seed; no imputation stats)
    subsample = cfg.get("subsample")
    if subsample is not None and len(work) > int(subsample):
        work = work.sample(n=int(subsample), random_state=split_seed).reset_index(drop=True)

    # SPLIT FIRST
    train, test, split_info = make_split(work, cfg, test_size=test_size, split_seed=split_seed)

    # THEN impute: fit on Real-Train only
    train, test, prep_meta = prepare_frames(train, test)

    meta = {
        "dataset_id": cfg["id"],
        "dataset_name": cfg["name"],
        "n_raw_rows": n_raw,
        "n_unique_entities": n_entities,
        "group_col": group_col,
        **split_info,
        **prep_meta,
        "target": cfg["target"],
        "task": cfg["task"],
        "n_samples_synthetic": cfg.get("n_samples", 1000),
        "train_columns": list(train.columns),
        "stage1_order": "split_then_train_only_impute",
    }

    # OASIS validation
    if group_col == "Subject ID" or cfg["id"] == "alzhimers":
        if split_info.get("entity_overlap", 0) != 0:
            raise AssertionError(
                f"OASIS participant leakage detected: overlap={split_info.get('entity_overlap')}"
            )
        meta["oasis_participant_level_split"] = True
        meta["oasis_train_participants"] = split_info.get("n_train_entities")
        meta["oasis_test_participants"] = split_info.get("n_test_entities")
        meta["oasis_participant_overlap"] = 0

    pd.to_pickle({"train": train, "test": test, "cfg": cfg}, cache_path)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "train": train,
        "test": test,
        "cfg": cfg,
        "meta": meta,
        "from_cache": False,
    }
