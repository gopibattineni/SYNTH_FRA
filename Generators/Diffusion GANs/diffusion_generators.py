"""
Shared diffusion-model training helpers for the Diffusion GANs benchmark notebooks.

Each function fits on a pandas DataFrame and returns synthetic data with the same
columns / dtypes as the input training frame. Only the generator logic lives here;
preprocessing in the notebooks remains unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import types
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler

LOGGER = logging.getLogger(__name__)

MODEL_ORDER = ["TabDDPM", "ForestDiffusion"]

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "_vendor"
TAB_DDPM_ROOT = VENDOR / "tab-ddpm"
TAB_DDPM_SCRIPTS = TAB_DDPM_ROOT / "scripts"
GOGGLE_ROOT = VENDOR / "goggle" / "src"
CODI_ROOT = VENDOR / "CoDi"


def _ensure_tabddpm_paths() -> None:
    for p in (TAB_DDPM_ROOT, TAB_DDPM_SCRIPTS):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected vendor checkout at {p}. "
                "Clone: git clone https://github.com/yandex-research/tab-ddpm _vendor/tab-ddpm"
            )
    for p in (TAB_DDPM_ROOT, TAB_DDPM_SCRIPTS):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def _ensure_paths() -> None:
    for p in (TAB_DDPM_ROOT, TAB_DDPM_SCRIPTS, GOGGLE_ROOT, CODI_ROOT):
        if not p.exists():
            raise FileNotFoundError(
                f"Expected vendor checkout at {p}. "
                "Clone official repos into _vendor/ before running notebooks."
            )
    for p in (TAB_DDPM_ROOT, TAB_DDPM_SCRIPTS, GOGGLE_ROOT, CODI_ROOT):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)


def infer_column_types(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    if isinstance(categorical_columns, pd.DataFrame):
        raise TypeError(
            "categorical_columns must be a list of column names, not a DataFrame. "
            "Use e.g. _categorical_columns = [target_col] or []."
        )
    cat_cols = list(categorical_columns or [])
    for col in df.columns:
        if col in cat_cols:
            continue
        if df[col].dtype == object or str(df[col].dtype) == "category":
            cat_cols.append(col)
        elif col == target_col and df[col].nunique() <= 30 and not _is_regression_target(df[col]):
            cat_cols.append(col)
    num_cols = [c for c in df.columns if c not in cat_cols]
    return cat_cols, num_cols


def _is_regression_target(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return False
    vals = np.sort(pd.to_numeric(series, errors="coerce").dropna().unique())
    n = len(vals)
    if n == 0:
        return False
    if n > 30:
        return True
    if n <= 2:
        return False
    # Small sets of consecutive integers (0..k or 1..k) are class labels.
    if np.array_equal(vals, np.arange(n)) or np.array_equal(vals, np.arange(1, n + 1)):
        return False
    # Otherwise treat wide-spread numeric targets as regression (e.g. price levels).
    return (vals[-1] - vals[0]) > n


def _resolve_regression_target(series: pd.Series, is_regression: Optional[bool] = None) -> bool:
    if is_regression is not None:
        return is_regression
    return _is_regression_target(series)


def _xgboost_gpu_available() -> bool:
    """True when XGBoost can train on CUDA (patten-server A100, etc.)."""
    try:
        import xgboost as xgb

        info = xgb.build_info()
        if isinstance(info, dict):
            return info.get("USE_CUDA", "0") in ("1", 1, True)
        return "USE_CUDA" in str(info) and "USE_CUDA=1" in str(info)
    except Exception:
        return False


def _forestdiffusion_params(
    fast_mode: bool,
    n_t: Optional[int],
    duplicate_K: Optional[int],
    n_jobs: Optional[int],
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    subsample: Optional[float] = None,
    gpu_hist: Optional[bool] = None,
) -> Dict[str, object]:
    """Tune ForestDiffusion speed vs quality."""
    cpu_count = os.cpu_count() or 8
    xgb_gpu = _xgboost_gpu_available()

    if fast_mode:
        return {
            "n_t": n_t if n_t is not None else 8,
            "duplicate_K": duplicate_K if duplicate_K is not None else 10,
            "n_jobs": n_jobs if n_jobs is not None else min(4, max(1, cpu_count // 8)),
            "n_estimators": n_estimators if n_estimators is not None else 30,
            "max_depth": max_depth if max_depth is not None else 5,
            "subsample": subsample if subsample is not None else 0.8,
            "gpu_hist": gpu_hist if gpu_hist is not None else xgb_gpu,
        }
    return {
        "n_t": n_t if n_t is not None else 50,
        "duplicate_K": duplicate_K if duplicate_K is not None else 100,
        "n_jobs": n_jobs if n_jobs is not None else min(8, cpu_count),
        "n_estimators": n_estimators if n_estimators is not None else 100,
        "max_depth": max_depth if max_depth is not None else 7,
        "subsample": subsample if subsample is not None else 1.0,
        "gpu_hist": gpu_hist if gpu_hist is not None else xgb_gpu,
    }


def _safe_stratify(y_arr: np.ndarray) -> Optional[np.ndarray]:
    if y_arr is None:
        return None
    _, counts = np.unique(y_arr, return_counts=True)
    if len(counts) <= 1 or counts.min() < 2:
        return None
    return y_arr


def _label_encoder_inverse(le: LabelEncoder, values) -> np.ndarray:
    """Decode encoded categoricals; clip to known classes (TabDDPM may emit out-of-range indices)."""
    arr = np.asarray(values).astype(int)
    if arr.size == 0:
        return arr
    arr = np.clip(arr, 0, len(le.classes_) - 1)
    return le.inverse_transform(arr)


def _dataframe_to_tabddpm_dir(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    out_dir: Path,
    seed: int = 42,
    is_regression: Optional[bool] = None,
) -> Dict[str, LabelEncoder]:
    from sklearn.model_selection import train_test_split

    out_dir.mkdir(parents=True, exist_ok=True)
    encoders: Dict[str, LabelEncoder] = {}
    work = df.copy()

    x_cat_parts = []
    for col in cat_cols:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le
        x_cat_parts.append(work[col].to_numpy().reshape(-1, 1))
    x_cat = np.hstack(x_cat_parts) if x_cat_parts else None

    x_num = work[num_cols].astype(float).to_numpy() if num_cols else None
    y = work[target_col]
    if _resolve_regression_target(y, is_regression):
        task_type = "regression"
        y_arr = y.astype(float).to_numpy()
    else:
        if target_col not in encoders:
            le = LabelEncoder()
            y_arr = le.fit_transform(y.astype(str))
            encoders[target_col] = le
        else:
            y_arr = work[target_col].to_numpy()
        task_type = "binclass" if len(np.unique(y_arr)) == 2 else "multiclass"

    n = len(y_arr)
    indices = np.arange(n)
    strat = (
        _safe_stratify(y_arr)
        if task_type != "regression" and len(np.unique(y_arr)) > 1
        else None
    )
    if n < 10:
        train_idx, val_idx, test_idx = indices, indices[:1], indices[:1]
    else:
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=strat
        )
        strat_temp = _safe_stratify(y_arr[temp_idx]) if strat is not None else None
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=seed, stratify=strat_temp
        )

    for split, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if x_num is not None:
            np.save(out_dir / f"X_num_{split}.npy", x_num[idx])
        if x_cat is not None:
            np.save(out_dir / f"X_cat_{split}.npy", x_cat[idx])
        np.save(out_dir / f"y_{split}.npy", y_arr[idx])

    info = {
        "name": "custom",
        "task_type": task_type,
        "n_num_features": 0 if x_num is None else x_num.shape[1],
        "n_cat_features": 0 if x_cat is None else x_cat.shape[1],
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "test_size": len(test_idx),
    }
    if task_type != "regression":
        info["n_classes"] = int(len(np.unique(y_arr)))
    with open(out_dir / "info.json", "w", encoding="utf-8") as fh:
        json.dump(info, fh)
    return encoders


def _tabddpm_arrays_to_dataframe(
    df_template: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    num_cols: List[str],
    encoders: Dict[str, LabelEncoder],
    x_num: Optional[np.ndarray],
    x_cat: Optional[np.ndarray],
    y: np.ndarray,
) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(y)))
    col_idx_num = 0
    col_idx_cat = 0
    feature_cat_cols = [c for c in cat_cols if c != target_col]
    for col in df_template.columns:
        if col in num_cols:
            out[col] = x_num[:, col_idx_num]
            col_idx_num += 1
        elif col in feature_cat_cols:
            raw = x_cat[:, col_idx_cat].astype(int)
            if col in encoders:
                raw = _label_encoder_inverse(encoders[col], raw)
            out[col] = raw
            col_idx_cat += 1
        elif col == target_col:
            if _is_regression_target(df_template[target_col]):
                out[col] = y.astype(float)
            elif target_col in encoders:
                out[col] = _label_encoder_inverse(encoders[target_col], y)
            else:
                out[col] = y
    return out[df_template.columns]


def train_tabddpm(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    steps: int = 1000,
    device: Optional[str] = None,
    is_regression: Optional[bool] = None,
) -> pd.DataFrame:
    _ensure_tabddpm_paths()
    from scripts.sample import sample as tabddpm_sample
    from scripts.train import train as tabddpm_train

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cat_cols, num_cols = infer_column_types(df, target_col, categorical_columns)
    regression = _resolve_regression_target(df[target_col], is_regression)
    # TabDDPM stores y separately and prepends it to X_num (regression) or X_cat (classification).
    if target_col in num_cols:
        num_cols.remove(target_col)
    if not regression and target_col not in cat_cols:
        cat_cols.append(target_col)
    feature_cat_cols = [c for c in cat_cols if c != target_col]
    num_feature_count = len(num_cols)

    with tempfile.TemporaryDirectory(prefix="tabddpm_") as tmp:
        data_dir = Path(tmp) / "data"
        model_dir = Path(tmp) / "model"
        data_dir.mkdir()
        model_dir.mkdir()
        encoders = _dataframe_to_tabddpm_dir(
            df, target_col, feature_cat_cols, num_cols, data_dir, seed=seed,
            is_regression=regression,
        )

        model_params = {
            "num_classes": 0 if regression else int(df[target_col].nunique()),
            "is_y_cond": False,
            "rtdl_params": {"d_layers": [256, 256, 256], "dropout": 0.0},
        }
        t_dict = {
            "seed": seed,
            "normalization": "quantile",
            "num_nan_policy": None,
            "cat_nan_policy": None,
            "cat_min_frequency": None,
            "cat_encoding": "one-hot",
            "y_policy": "default",
        }

        tabddpm_train(
            parent_dir=str(model_dir),
            real_data_path=str(data_dir),
            steps=steps,
            lr=0.002,
            weight_decay=1e-4,
            batch_size=min(1024, max(64, len(df) // 10)),
            model_type="mlp",
            model_params=model_params,
            num_timesteps=1000,
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=t_dict,
            device=torch.device(device),
            seed=seed,
        )

        tabddpm_sample(
            parent_dir=str(model_dir),
            real_data_path=str(data_dir),
            batch_size=min(2000, n_samples),
            num_samples=n_samples,
            model_type="mlp",
            model_params=model_params,
            model_path=str(model_dir / "model.pt"),
            num_timesteps=1000,
            gaussian_loss_type="mse",
            scheduler="cosine",
            T_dict=t_dict,
            num_numerical_features=num_feature_count,
            device=torch.device(device),
            seed=seed,
        )

        x_num = (
            np.load(model_dir / "X_num_train.npy", allow_pickle=True)
            if (model_dir / "X_num_train.npy").exists()
            else None
        )
        x_cat = (
            np.load(model_dir / "X_cat_train.npy", allow_pickle=True)
            if (model_dir / "X_cat_train.npy").exists()
            else None
        )
        y = np.load(model_dir / "y_train.npy", allow_pickle=True)
        synth = _tabddpm_arrays_to_dataframe(
            df, target_col, feature_cat_cols, num_cols, encoders, x_num, x_cat, y
        )
        return synth.head(n_samples).reset_index(drop=True)


def train_goggle(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    epochs: int = 300,
    device: Optional[str] = None,
) -> pd.DataFrame:
    _ensure_paths()
    from goggle.data_utils import get_dataloader
    from goggle.model.Goggle import Goggle
    from goggle.model.GoggleLoss import GoggleLoss

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cat_cols, num_cols = infer_column_types(df, target_col, categorical_columns)
    work = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    scaler = StandardScaler()
    if num_cols:
        work[num_cols] = scaler.fit_transform(work[num_cols].astype(float))

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = get_dataloader(work, batch_size=64, seed=seed)["train"]
    input_dim = work.shape[1]
    model = Goggle(
        input_dim=input_dim,
        encoder_dim=64,
        encoder_l=2,
        het_encoding=True,
        decoder_dim=64,
        decoder_l=2,
        threshold=0.1,
        decoder_arch="gcn",
        graph_prior=None,
        prior_mask=None,
        device=device,
    ).to(device)
    loss_fn = GoggleLoss(alpha=0.1, beta=0.1, graph_prior=None, device=device)
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-3)

    model.train()
    for _epoch in range(epochs):
        for batch in train_loader:
            x = batch[0].to(device)
            optimiser.zero_grad()
            x_hat, adj, mu_z, logvar_z = model(x, _epoch)
            loss, _, _, _ = loss_fn(x_hat, x, mu_z, logvar_z, adj)
            loss.backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        synth_arr = model.sample(n_samples).cpu().numpy()

    synth = pd.DataFrame(synth_arr, columns=work.columns)
    for col in num_cols:
        synth[col] = scaler.inverse_transform(synth[[col]])
    for col in cat_cols:
        vals = np.clip(np.round(synth[col]), 0, len(encoders[col].classes_) - 1).astype(int)
        synth[col] = _label_encoder_inverse(encoders[col], vals)
    return synth[df.columns].reset_index(drop=True)


def train_forestdiffusion(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    fast_mode: bool = True,
    n_t: Optional[int] = None,
    duplicate_K: Optional[int] = None,
    n_jobs: Optional[int] = None,
    n_estimators: Optional[int] = None,
    max_depth: Optional[int] = None,
    subsample: Optional[float] = None,
    gpu_hist: Optional[bool] = None,
    is_regression: Optional[bool] = None,
) -> pd.DataFrame:
    from ForestDiffusion import ForestDiffusionModel

    fd_params = _forestdiffusion_params(
        fast_mode,
        n_t,
        duplicate_K,
        n_jobs,
        n_estimators,
        max_depth,
        subsample,
        gpu_hist,
    )
    if fast_mode:
        print(
            "ForestDiffusion fast_mode: "
            f"n_t={fd_params['n_t']}, duplicate_K={fd_params['duplicate_K']}, "
            f"n_estimators={fd_params['n_estimators']}, max_depth={fd_params['max_depth']}, "
            f"n_jobs={fd_params['n_jobs']}, gpu_hist={fd_params['gpu_hist']}"
        )

    cat_cols, num_cols = infer_column_types(df, target_col, categorical_columns)
    regression = _resolve_regression_target(df[target_col], is_regression)
    classification = not regression
    work = df.copy()
    encoders: Dict[str, LabelEncoder] = {}
    col_order = list(work.columns)

    for col in cat_cols:
        le = LabelEncoder()
        work[col] = le.fit_transform(work[col].astype(str))
        encoders[col] = le

    # Binary 0/1 features -> bin_indexes; multi-category features -> cat_indexes.
    feature_cat_cols = [c for c in cat_cols if c != target_col]
    bin_cols: List[str] = []
    multi_cat_cols: List[str] = []
    for col in feature_cat_cols:
        vals = pd.to_numeric(work[col], errors="coerce").dropna().unique()
        if len(vals) <= 2:
            bin_cols.append(col)
        else:
            multi_cat_cols.append(col)
    for col in num_cols:
        vals = pd.to_numeric(work[col], errors="coerce").dropna().unique()
        if len(vals) <= 2 and set(np.round(vals).astype(int)).issubset({0, 1}):
            bin_cols.append(col)
    bin_cols = list(dict.fromkeys(bin_cols))

    label_y = None
    feature_cols = [c for c in col_order if c != target_col]
    if regression:
        x_arr = work.to_numpy()
        bin_indexes = [col_order.index(c) for c in bin_cols]
        cat_indexes = [col_order.index(c) for c in multi_cat_cols]
    elif classification:
        label_y = work[target_col].to_numpy()
        x_arr = work[feature_cols].to_numpy()
        bin_indexes = [feature_cols.index(c) for c in bin_cols if c in feature_cols]
        cat_indexes = [feature_cols.index(c) for c in multi_cat_cols if c in feature_cols]
    else:
        x_arr = work.to_numpy()
        bin_indexes = [col_order.index(c) for c in bin_cols]
        cat_indexes = [col_order.index(c) for c in multi_cat_cols]

    p_in_one = not np.isnan(x_arr).any()

    model = ForestDiffusionModel(
        x_arr,
        label_y=label_y,
        n_t=int(fd_params["n_t"]),
        duplicate_K=int(fd_params["duplicate_K"]),
        n_estimators=int(fd_params["n_estimators"]),
        max_depth=int(fd_params["max_depth"]),
        subsample=float(fd_params["subsample"]),
        bin_indexes=bin_indexes,
        cat_indexes=cat_indexes,
        int_indexes=[],
        diffusion_type="flow",
        n_jobs=int(fd_params["n_jobs"]),
        gpu_hist=bool(fd_params["gpu_hist"]),
        p_in_one=p_in_one,
        remove_miss=False,
        seed=seed,
    )
    generated = model.generate(batch_size=n_samples)
    if classification and label_y is not None:
        full = np.zeros((generated.shape[0], len(col_order)))
        for i, col in enumerate(feature_cols):
            full[:, col_order.index(col)] = generated[:, i]
        full[:, col_order.index(target_col)] = generated[:, -1]
        generated = full

    synth = pd.DataFrame(generated, columns=col_order)
    for col in cat_cols:
        vals = np.clip(np.round(synth[col]), 0, len(encoders[col].classes_) - 1).astype(int)
        synth[col] = _label_encoder_inverse(encoders[col], vals)
    return synth[df.columns].reset_index(drop=True)


def _dataframe_to_codi_bundle(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    dataset_name: str,
) -> Tuple[np.ndarray, dict, Dict[str, LabelEncoder]]:
    encoders: Dict[str, LabelEncoder] = {}
    work = df.copy()
    col_meta = []
    dis_idx: List[int] = []

    for i, col in enumerate(work.columns):
        if col in cat_cols:
            le = LabelEncoder()
            work[col] = le.fit_transform(work[col].astype(str))
            encoders[col] = le
            dis_idx.append(i)
            col_meta.append(
                {
                    "name": col,
                    "type": "categorical",
                    "size": len(le.classes_),
                    "i2s": [str(x) for x in le.classes_],
                }
            )
        else:
            vals = work[col].astype(float).to_numpy()
            col_meta.append(
                {
                    "name": col,
                    "type": "continuous",
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
            )

    train = work.to_numpy().astype(float)
    if _is_regression_target(df[target_col]):
        problem_type = "regression"
    elif df[target_col].nunique() == 2:
        problem_type = "binary_classification"
    else:
        problem_type = "multiclass_classification"

    meta = {"columns": col_meta, "problem_type": problem_type}
    codi_data_dir = CODI_ROOT / "tabular_datasets"
    codi_data_dir.mkdir(parents=True, exist_ok=True)
    np.savez(codi_data_dir / f"{dataset_name}.npz", train=train, test=train[: max(1, len(train) // 5)])
    with open(codi_data_dir / f"{dataset_name}.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return train, meta, encoders


def _codi_sample_to_dataframe(
    sample: np.ndarray,
    columns: List[str],
    cat_cols: List[str],
    encoders: Dict[str, LabelEncoder],
) -> pd.DataFrame:
    out = pd.DataFrame(sample, columns=columns)
    for col in cat_cols:
        vals = np.clip(np.round(out[col]), 0, len(encoders[col].classes_) - 1).astype(int)
        out[col] = _label_encoder_inverse(encoders[col], vals)
    return out


def train_codi(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
    total_epochs: int = 200,
    device: Optional[str] = None,
) -> pd.DataFrame:
    _ensure_paths()

    import tabular_dataload as codi_load
    from diffusion_continuous import GaussianDiffusionSampler, GaussianDiffusionTrainer
    from diffusion_discrete import MultinomialDiffusion
    from models.tabular_unet import tabularUnet
    from utils import apply_activate, infiniteloop, log_sample_categorical, make_negative_condition, sampling_with, training_with, warmup_lr

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    cat_cols, _con_cols = infer_column_types(df, target_col, categorical_columns)
    dataset_name = f"benchmark_{abs(hash(tuple(df.columns))) % 10**8}"
    train_np, _meta, encoders = _dataframe_to_codi_bundle(df, target_col, cat_cols, dataset_name)

    batch_size = min(512, max(64, len(train_np) // 4))
    flags = types.SimpleNamespace(
        data=dataset_name,
        logdir=tempfile.mkdtemp(prefix="codi_"),
        train=True,
        eval=False,
        encoder_dim_con="64,128,256",
        encoder_dim_dis="64,128,256",
        nf_con=16,
        nf_dis=64,
        activation="relu",
        training_batch_size=batch_size,
        eval_batch_size=batch_size,
        T=50,
        beta_1=1e-5,
        beta_T=0.02,
        lr_con=2e-3,
        lr_dis=2e-3,
        total_epochs_both=total_epochs,
        grad_clip=1.0,
        parallel=False,
        sample_step=max(10, total_epochs // 5),
        mean_type="epsilon",
        var_type="fixedsmall",
        ns_method=0,
        lambda_con=0.2,
        lambda_dis=0.2,
        input_size=None,
        cond_size=None,
        output_size=None,
        encoder_dim=None,
        nf=None,
    )

    train_np, train_con_data, train_dis_data, _, transformers, con_idx, dis_idx = codi_load.get_dataset(flags)
    transformer_con, transformer_dis, _ = transformers
    train_iter_con = torch.utils.data.DataLoader(train_con_data, batch_size=batch_size)
    train_iter_dis = torch.utils.data.DataLoader(train_dis_data, batch_size=batch_size)
    datalooper_train_con = infiniteloop(train_iter_con)
    datalooper_train_dis = infiniteloop(train_iter_dis)

    num_class = np.array([item[0] for item in transformer_dis.output_info])

    flags.input_size = train_con_data.shape[1]
    flags.cond_size = train_dis_data.shape[1]
    flags.output_size = train_con_data.shape[1]
    flags.encoder_dim = list(map(int, flags.encoder_dim_con.split(",")))
    flags.nf = flags.nf_con
    model_con = tabularUnet(flags).to(device)
    optim_con = torch.optim.Adam(model_con.parameters(), lr=flags.lr_con)
    sched_con = torch.optim.lr_scheduler.LambdaLR(optim_con, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(model_con, flags.beta_1, flags.beta_T, flags.T).to(device)
    net_sampler = GaussianDiffusionSampler(
        model_con, flags.beta_1, flags.beta_T, flags.T, flags.mean_type, flags.var_type
    ).to(device)

    flags.input_size = train_dis_data.shape[1]
    flags.cond_size = train_con_data.shape[1]
    flags.output_size = train_dis_data.shape[1]
    flags.encoder_dim = list(map(int, flags.encoder_dim_dis.split(",")))
    flags.nf = flags.nf_dis
    model_dis = tabularUnet(flags).to(device)
    optim_dis = torch.optim.Adam(model_dis.parameters(), lr=flags.lr_dis)
    sched_dis = torch.optim.lr_scheduler.LambdaLR(optim_dis, lr_lambda=warmup_lr)
    trainer_dis = MultinomialDiffusion(
        num_class, train_dis_data.shape, model_dis, flags, timesteps=flags.T, loss_type="vb_stochastic"
    ).to(device)

    total_steps = flags.total_epochs_both * int(train_np.shape[0] / batch_size + 1)
    for step in range(total_steps):
        model_con.train()
        model_dis.train()
        x_0_con = next(datalooper_train_con).to(device).float()
        x_0_dis = next(datalooper_train_dis).to(device)
        ns_con, ns_dis = make_negative_condition(x_0_con, x_0_dis)
        con_loss, con_loss_ns, dis_loss, dis_loss_ns = training_with(
            x_0_con, x_0_dis, trainer, trainer_dis, ns_con, ns_dis, transformer_dis, flags
        )
        loss_con = con_loss + flags.lambda_con * con_loss_ns
        loss_dis = dis_loss + flags.lambda_dis * dis_loss_ns
        optim_con.zero_grad()
        loss_con.backward()
        torch.nn.utils.clip_grad_norm_(model_con.parameters(), flags.grad_clip)
        optim_con.step()
        sched_con.step()
        optim_dis.zero_grad()
        loss_dis.backward()
        torch.nn.utils.clip_grad_norm_(trainer_dis.parameters(), flags.grad_clip)
        optim_dis.step()
        sched_dis.step()

    model_con.eval()
    model_dis.eval()
    rows = []
    with torch.no_grad():
        while len(rows) < n_samples:
            bs = min(n_samples - len(rows), train_con_data.shape[0])
            x_T_con = torch.randn(bs, train_con_data.shape[1]).to(device)
            log_x_T_dis = log_sample_categorical(
                torch.zeros(bs, train_dis_data.shape[1], device=device), num_class
            ).to(device)
            x_con, x_dis = sampling_with(x_T_con, log_x_T_dis, net_sampler, trainer_dis, transformer_con, flags)
            x_dis = apply_activate(x_dis, transformer_dis.output_info)
            sample_con = transformer_con.inverse_transform(x_con.detach().cpu().numpy())
            sample_dis = transformer_dis.inverse_transform(x_dis.detach().cpu().numpy())
            sample = np.zeros((bs, train_np.shape[1]))
            for i in range(len(con_idx)):
                sample[:, con_idx[i]] = sample_con[:, i]
            for i in range(len(dis_idx)):
                sample[:, dis_idx[i]] = sample_dis[:, i]
            rows.append(sample)
    sample_all = np.vstack(rows)[:n_samples]
    return _codi_sample_to_dataframe(sample_all, list(df.columns), cat_cols, encoders)


def train_all_diffusion_models(
    df: pd.DataFrame,
    target_col: str,
    categorical_columns: Optional[Sequence[str]] = None,
    n_samples: int = 1000,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    return {
        "TabDDPM": train_tabddpm(df, target_col, categorical_columns, n_samples, seed),
        "CoDi": train_codi(df, target_col, categorical_columns, n_samples, seed),
        "GOGGLE": train_goggle(df, target_col, categorical_columns, n_samples, seed),
        "ForestDiffusion": train_forestdiffusion(
            df, target_col, categorical_columns, n_samples, seed
        ),
    }
