"""Export multi-seed Mean±SD Excel results into docs/data JSON for GitHub Pages."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MERGED_ROOT = REPO_ROOT / "multi seed generators" / "merged 10 seeds"
CLS_XLSX = MERGED_ROOT / "classification" / "results" / "multi_seed_results.xlsx"
REG_XLSX = MERGED_ROOT / "regression" / "results" / "multi_seed_results.xlsx"
DOCS_DATA = REPO_ROOT / "docs" / "data"

GENERATORS = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
    "TabDDPM",
    "ForestDiffusion",
]

# Match existing dashboard numbered labels (problem-type filter uses prefix ≥10 = regression).
DATASET_DISPLAY = {
    "Cancer": "1. Cancer",
    "Alzheimers": "2. Alzhimers",
    "Adult": "3. Adult",
    "ForestCover": "4. Forest cover dataset",
    "Bank": "5. Bank Markting",
    "Wine": "6. Wine dataset",
    "CDC": "7. CDC diabetes dataset",
    "Mushroom": "8. Mushroom dataset",
    "MAGIC": "9. MAGIC Gamma Telescope",
    "Metro": "10. Metro interstate",
    "OnlineShop": "11. online shopping",
    "AirQuality": "12. Air Quality",
    "Concrete": "13. Concrete Compressive Strength",
    "Energy": "14. Energy Efficiency",
    "RealEstate": "15. Real Estate Valuation",
}

SEEDS = [42, 123, 2024, 68, 91, 2025, 55, 155, 255, 355]


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


def _display_dataset(name: str) -> str:
    return DATASET_DISPLAY.get(str(name), str(name))


def _load_all_metrics(path: Path, task: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing merged workbook: {path}")
    df = pd.read_excel(path, sheet_name="All metrics")
    df = df.copy()
    df["task"] = task
    df["Dataset"] = df["dataset"].map(_display_dataset)
    df["Generator"] = df["generator"].astype(str)
    df["Mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["Std"] = pd.to_numeric(df["sd"], errors="coerce")
    df["n_seeds"] = pd.to_numeric(df["n_seeds"], errors="coerce").fillna(0).astype(int)
    return df


def _split_utility_name(metric_name: str) -> tuple[str, str] | None:
    """Return (base_or_gap_metric, EvaluationType) or None."""
    name = str(metric_name)
    if name.endswith("_TRTR"):
        return name[: -len("_TRTR")], "TRTR"
    if name.endswith("_TSTR"):
        return name[: -len("_TSTR")], "TSTR"
    if name.endswith("_Gap"):
        return name, "Gap"
    if name.endswith("_Increase"):
        return name, "Gap"
    return None


def _build_utility(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    util = df[df["metric_category"] == "Utility"].copy()
    rows = []
    gap_rows = []
    for _, r in util.iterrows():
        parsed = _split_utility_name(r["metric_name"])
        if not parsed:
            continue
        metric, etype = parsed
        base = {
            "Dataset": r["Dataset"],
            "Generator": r["Generator"],
            "TaskType": r["task"],
            "Metric": metric,
            "EvaluationType": etype,
            "Mean": None if pd.isna(r["Mean"]) else float(r["Mean"]),
            "Std": None if pd.isna(r["Std"]) else float(r["Std"]),
            "Count": int(r["n_seeds"]),
        }
        rows.append(base)
        if etype == "Gap":
            gap_rows.append(
                {
                    "Dataset": r["Dataset"],
                    "Generator": r["Generator"],
                    "TaskType": r["task"],
                    "Metric": metric,
                    "Mean": base["Mean"],
                }
            )
            # Aliases used by older dashboard filters
            if metric.endswith("_Gap"):
                drop = metric.replace("_Gap", "_Drop")
                gap_rows.append({**gap_rows[-1], "Metric": drop})
                rows.append({**base, "Metric": drop})
    return pd.DataFrame(rows), pd.DataFrame(gap_rows)


def _build_pillar(df: pd.DataFrame, category: str) -> pd.DataFrame:
    sub = df[df["metric_category"] == category].copy()
    out = sub.rename(columns={"metric_name": "Metric"})[
        ["Dataset", "Generator", "Metric", "Mean", "Std"]
    ]
    return out.dropna(subset=["Mean"], how="any") if not out.empty else out


def _fidelity_catalog(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    higher = {"Quality_Score", "KS_Complement"}
    catalog = []
    for metric, grp in df.groupby("Metric", dropna=False):
        catalog.append(
            {
                "id": metric,
                "label": str(metric).replace("_", " "),
                "count": int(len(grp)),
                "higher_is_better": metric in higher,
                "is_unit_interval": metric in {"Quality_Score", "KS_Complement"},
            }
        )
    catalog.sort(key=lambda x: x["count"], reverse=True)
    return catalog


def _privacy_catalog(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    catalog = []
    for metric, grp in df.groupby("Metric", dropna=False):
        catalog.append(
            {
                "id": metric,
                "label": str(metric).replace("_", " "),
                "count": int(len(grp)),
                "lower_is_better": metric in {"MIA_AUC"},
                "is_similarity": metric == "MIA_AUC",
                "is_sample_size": False,
            }
        )
    catalog.sort(key=lambda x: x["count"], reverse=True)
    return catalog


def _minmax(series: pd.Series, higher_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or lo == hi:
        return pd.Series(np.nan, index=s.index)
    norm = (s - lo) / (hi - lo)
    return norm if higher_better else 1.0 - norm


def _build_rankings_and_tradeoff(
    util_agg: pd.DataFrame, fidelity: pd.DataFrame, privacy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Utility score: prefer classification Accuracy_Gap + regression R2_Gap (lower better)
    gap = util_agg[
        (util_agg["EvaluationType"] == "Gap")
        & (util_agg["Metric"].isin(["Accuracy_Gap", "R2_Gap"]))
    ].copy()
    util_by_gen = gap.groupby("Generator")["Mean"].mean()

    fid = fidelity[fidelity["Metric"] == "Quality_Score"].groupby("Generator")["Mean"].mean()
    mia = privacy[privacy["Metric"] == "MIA_AUC"].groupby("Generator")["Mean"].mean()

    gens = [g for g in GENERATORS if g in set(util_by_gen.index) | set(fid.index) | set(mia.index)]
    frame = pd.DataFrame(index=gens)
    frame["UtilityRaw"] = util_by_gen
    frame["FidelityRaw"] = fid
    frame["PrivacyRaw"] = mia
    frame["Utility"] = _minmax(frame["UtilityRaw"], higher_better=False)  # lower gap better
    frame["Fidelity"] = _minmax(frame["FidelityRaw"], higher_better=True)
    frame["Privacy"] = _minmax(frame["PrivacyRaw"], higher_better=False)  # lower MIA better

    # Fill missing pillar scores with column mean so ranking still works
    for col in ("Utility", "Fidelity", "Privacy"):
        frame[col] = frame[col].fillna(frame[col].mean())

    frame["OverallScore"] = 0.4 * frame["Utility"] + 0.3 * frame["Fidelity"] + 0.3 * frame["Privacy"]
    frame = frame.sort_values("OverallScore", ascending=False)
    frame["OverallRank"] = np.arange(1, len(frame) + 1, dtype=float)
    weighted = frame.reset_index().rename(columns={"index": "Generator"})
    weighted.insert(0, "index", np.arange(len(weighted)))

    # Borda: sum of ranks across pillars (lower rank sum better)
    borda = frame[["Utility", "Fidelity", "Privacy"]].copy()
    for col in borda.columns:
        borda[f"{col}_rank"] = borda[col].rank(ascending=False, method="average")
    borda["BordaScore"] = borda[[c for c in borda.columns if c.endswith("_rank")]].sum(axis=1)
    borda = borda.sort_values("BordaScore")
    borda["BordaRank"] = np.arange(1, len(borda) + 1, dtype=float)
    borda_out = borda.reset_index().rename(columns={"index": "Generator"})
    borda_out.insert(0, "index", np.arange(len(borda_out)))

    trade = weighted[["Generator", "Utility", "Privacy", "Fidelity"]].copy()
    # Dominated if another generator is ≥ on all and > on one
    dominated = []
    vals = trade.set_index("Generator")[["Utility", "Fidelity", "Privacy"]]
    for gen in vals.index:
        others = vals.drop(index=gen)
        dom = False
        for _, o in others.iterrows():
            if (o >= vals.loc[gen]).all() and (o > vals.loc[gen]).any():
                dom = True
                break
        dominated.append(dom)
    trade["Dominated"] = dominated

    return weighted, borda_out, trade


def export_multiseed_dashboard_data(output_dir: Path | None = None) -> Path:
    out = Path(output_dir) if output_dir else DOCS_DATA
    out.mkdir(parents=True, exist_ok=True)

    frames = []
    if CLS_XLSX.is_file():
        frames.append(_load_all_metrics(CLS_XLSX, "classification"))
    if REG_XLSX.is_file():
        frames.append(_load_all_metrics(REG_XLSX, "regression"))
    if not frames:
        raise FileNotFoundError(
            "No merged 10-seed Excel files found under multi seed generators/merged 10 seeds/"
        )
    all_df = pd.concat(frames, ignore_index=True)

    util_agg, util_gaps = _build_utility(all_df)
    fidelity = _build_pillar(all_df, "Fidelity")
    privacy = _build_pillar(all_df, "Privacy")

    (out / "utility_agg.json").write_text(json.dumps(_records(util_agg), indent=2), encoding="utf-8")
    (out / "utility_gaps.json").write_text(json.dumps(_records(util_gaps), indent=2), encoding="utf-8")
    # Per-classifier / per-regressor detail is already averaged in the Excel Mean±SD.
    (out / "utility_classifier.json").write_text("[]\n", encoding="utf-8")
    (out / "utility_regressor.json").write_text("[]\n", encoding="utf-8")

    (out / "fidelity.json").write_text(json.dumps(_records(fidelity), indent=2), encoding="utf-8")
    (out / "fidelity_metrics.json").write_text(
        json.dumps(_fidelity_catalog(fidelity), indent=2), encoding="utf-8"
    )
    (out / "privacy.json").write_text(json.dumps(_records(privacy), indent=2), encoding="utf-8")
    (out / "privacy_metrics.json").write_text(
        json.dumps(_privacy_catalog(privacy), indent=2), encoding="utf-8"
    )

    weighted, borda, tradeoff = _build_rankings_and_tradeoff(util_agg, fidelity, privacy)
    (out / "rankings_weighted.json").write_text(
        json.dumps(_records(weighted), indent=2), encoding="utf-8"
    )
    (out / "rankings_borda.json").write_text(json.dumps(_records(borda), indent=2), encoding="utf-8")
    (out / "tradeoff.json").write_text(json.dumps(_records(tradeoff), indent=2), encoding="utf-8")

    coverage = (
        util_agg.groupby(["Dataset", "Generator"], dropna=False)
        .size()
        .reset_index(name="Rows")
    )
    coverage["Available"] = (coverage["Rows"] > 0).astype(int)
    (out / "coverage.json").write_text(json.dumps(_records(coverage), indent=2), encoding="utf-8")

    n_datasets = int(all_df["Dataset"].nunique())
    meta = {
        "source": "multi seed generators/merged 10 seeds",
        "aggregation": "Mean ± SD across 10 generator-training seeds (ddof=1)",
        "generator_seeds": SEEDS,
        "split_seed": 42,
        "n_generator_seeds": len(SEEDS),
        "generators": GENERATORS,
        "n_datasets": n_datasets,
        "n_utility_rows": int(len(util_agg)),
        "n_fidelity_rows": int(len(fidelity)),
        "n_privacy_rows": int(len(privacy)),
        "n_files": int(CLS_XLSX.is_file()) + int(REG_XLSX.is_file()),
        "n_excel_files": int(CLS_XLSX.is_file()) + int(REG_XLSX.is_file()),
        "n_runs": 15 * 8 * 10,
        "weights": {"utility": 0.4, "privacy": 0.3, "fidelity": 0.3},
        "pillars": ["Utility", "Fidelity", "Privacy"],
        "privacy_metrics": sorted(privacy["Metric"].dropna().unique().tolist()) if not privacy.empty else [],
        "fidelity_metrics": sorted(fidelity["Metric"].dropna().unique().tolist()) if not fidelity.empty else [],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Keep statistics / correlation payloads present for the UI (empty-safe).
    if not (out / "statistics.json").exists():
        (out / "statistics.json").write_text(
            json.dumps(
                {
                    "pca_errors": [],
                    "wilcoxon": [],
                    "effect_sizes": [],
                    "classification_stats": [],
                    "regression_stats": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if not (out / "correlation_tradeoff.json").exists():
        (out / "correlation_tradeoff.json").write_text(
            json.dumps({"analyses": [], "source": "multi-seed"}, indent=2), encoding="utf-8"
        )
    if not (out / "notebook_error_stats.json").exists():
        (out / "notebook_error_stats.json").write_text("[]\n", encoding="utf-8")

    return out


if __name__ == "__main__":
    dest = export_multiseed_dashboard_data()
    print(f"Exported multi-seed dashboard data → {dest}")
