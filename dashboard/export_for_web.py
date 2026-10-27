"""Export compact JSON datasets for the static GitHub Pages dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Results"
DOCS_DATA = REPO_ROOT / "docs" / "data"


def _read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _is_benchmark_dataset(name: object) -> bool:
    """Keep only numbered benchmark datasets (1–15), drop workbook sheet aliases."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return False
    text = str(name).strip()
    return bool(re.match(r"^(?:[1-9]|1[0-5])\.", text))


def _harmonize_num_matches(df: pd.DataFrame) -> pd.DataFrame:
    """
    Num_Matches in Hungarian summaries is len(matched pairs) = matching
    subsample size, not a privacy-risk count.

    Older SDV Excel runs often matched 5000 / 10000 / full-N rows, while Other
    GAN and Diffusion notebooks used the 1000-sample protocol. When a dataset
    mixes those sizes, snap inflated values down to the protocol size (1000
    when present).
    """
    if df.empty or "Metric" not in df.columns or "Mean" not in df.columns:
        return df
    out = df.copy()
    mask = out["Metric"] == "Num_Matches"
    if not mask.any():
        return out

    for _, idx in out.loc[mask].groupby("Dataset").groups.items():
        vals = pd.to_numeric(out.loc[idx, "Mean"], errors="coerce")
        if (vals == 1000).any():
            protocol = 1000.0
        else:
            small = vals[vals <= 1000]
            if small.empty or not (vals > small.min()).any():
                continue
            protocol = float(small.min())
        inflated = vals > protocol
        if not inflated.any():
            continue
        targets = idx[inflated.to_numpy()]
        out.loc[targets, "Mean"] = protocol
        for col in ("MetricValue", "Value"):
            if col in out.columns:
                out.loc[targets, col] = protocol
    return out


def _extract_pca_error_stats() -> pd.DataFrame:
    """
    Average error by model (Mean / Median / Std Error %) for all 8 generators
    × 15 datasets, from paper-results fidelity_privacy workbooks (notebook PCA tables).
    """
    try:
        from analysis.data_loader import normalize_generator
    except ImportError:
        def normalize_generator(value):  # type: ignore[misc]
            return None if value is None else str(value).strip()

    paper = REPO_ROOT / "paper results"
    if not paper.exists():
        return pd.DataFrame()

    rows: list[dict] = []
    for ds_dir in sorted(p for p in paper.iterdir() if p.is_dir()):
        fp = ds_dir / "fidelity_privacy_metrics.xlsx"
        if not fp.exists():
            continue
        try:
            df = pd.read_excel(fp, sheet_name="PCA_Mean_Errors")
        except Exception:
            continue
        if df.empty:
            continue

        model_col = "Generator" if "Generator" in df.columns else ("Model" if "Model" in df.columns else None)
        if model_col is None or "Mean Error %" not in df.columns:
            continue

        chunk = df.copy()
        chunk["Generator"] = chunk[model_col].map(normalize_generator)
        chunk = chunk.dropna(subset=["Generator"])
        for gen, g in chunk.groupby("Generator", dropna=False):
            # Prefer percentage-scale rows over rank-normalized 0–1 duplicates.
            mean_abs = pd.to_numeric(g["Mean Error %"], errors="coerce").abs()
            idx = mean_abs.idxmax()
            r = g.loc[idx]
            rows.append(
                {
                    "Dataset": ds_dir.name,
                    "Generator": gen,
                    "Mean_Error_Pct": float(pd.to_numeric(r["Mean Error %"], errors="coerce")),
                    "Median_Error_Pct": float(pd.to_numeric(r.get("Median Error %"), errors="coerce")),
                    "Std_Error_Pct": float(pd.to_numeric(r.get("Std Error %"), errors="coerce")),
                    "Source_Group": None if pd.isna(r.get("Source_Group")) else str(r.get("Source_Group")),
                }
            )

    return pd.DataFrame(rows)


def _fill_mahalanobis_from_mean_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adult SDV (and similar) notebooks omit Synth_Mean_MD_2D tables, but Excel
    Hungarian-Mahalanobis summaries store Mean_Distance for the same models.
    Use Mean_Distance only to fill missing Mahalanobis_Distance cells.
    """
    if df.empty or "Metric" not in df.columns:
        return df
    out = df.copy()
    mean_d = out[out["Metric"] == "Mean_Distance"][["Dataset", "Generator", "Mean"]].dropna()
    if mean_d.empty:
        return out
    existing = {
        (r.Dataset, r.Generator)
        for r in out[out["Metric"] == "Mahalanobis_Distance"][["Dataset", "Generator"]].itertuples(index=False)
    }
    extras = []
    for row in mean_d.itertuples(index=False):
        key = (row.Dataset, row.Generator)
        if key in existing:
            continue
        extras.append(
            {
                "Dataset": row.Dataset,
                "Generator": row.Generator,
                "Metric": "Mahalanobis_Distance",
                "Mean": row.Mean,
                "Std": None,
            }
        )
        existing.add(key)
    if extras:
        out = pd.concat([out, pd.DataFrame(extras)], ignore_index=True)
    return out


def _load_privacy_long() -> pd.DataFrame:
    """Prefer unified Master_Data privacy rows (includes NNDR, Mahalanobis, cosine)."""
    for rel in ("Master_Data/privacy_long.csv", "MasterData/privacy_long.csv"):
        df = _read_csv(rel)
        if df.empty:
            continue
        out = df.copy()
        if "MetricValue" in out.columns:
            mv = pd.to_numeric(out["MetricValue"], errors="coerce")
            if "Mean" in out.columns:
                mean = pd.to_numeric(out["Mean"], errors="coerce")
                out["Mean"] = mean.fillna(mv)
            else:
                out["Mean"] = mv
        if "Mean" not in out.columns and "Value" in out.columns:
            out["Mean"] = out["Value"]
        out["Mean"] = pd.to_numeric(out["Mean"], errors="coerce")
        if "Dataset" in out.columns:
            out = out[out["Dataset"].map(_is_benchmark_dataset)]
        keep = [c for c in ["Dataset", "Generator", "Metric", "Mean", "Std"] if c in out.columns]
        out = out[keep].dropna(subset=["Mean"])
        out = _harmonize_num_matches(out)
        return _fill_mahalanobis_from_mean_distance(out)
    return pd.DataFrame()


def _load_fidelity_long() -> pd.DataFrame:
    """Prefer unified Master_Data fidelity rows (all feature-level metrics)."""
    for rel in ("Master_Data/fidelity_long.csv", "MasterData/fidelity_long.csv"):
        df = _read_csv(rel)
        if df.empty:
            continue
        out = df.copy()
        if "MetricValue" in out.columns:
            mv = pd.to_numeric(out["MetricValue"], errors="coerce")
            if "Mean" in out.columns:
                mean = pd.to_numeric(out["Mean"], errors="coerce")
                out["Mean"] = mean.fillna(mv)
            else:
                out["Mean"] = mv
        if "Mean" not in out.columns and "Value" in out.columns:
            out["Mean"] = out["Value"]
        out["Mean"] = pd.to_numeric(out["Mean"], errors="coerce")
        if "Dataset" in out.columns:
            out = out[out["Dataset"].map(_is_benchmark_dataset)]
        if "Generator" in out.columns:
            out = out[out["Generator"].notna() & (out["Generator"].astype(str).str.strip() != "")]
        # Adult-only Mean_Error_Pct rows are a mislabeled PCA duplicate; the
        # Fidelity tab uses PCA_Mean_Error_Pct (full 15×8 coverage). Statistics
        # still exposes Mean Error % via notebook_error_stats / PCA sheets.
        if "Metric" in out.columns:
            out = out[out["Metric"] != "Mean_Error_Pct"]
        keep = [c for c in ["Dataset", "Generator", "Metric", "Mean", "Std", "NormalizedScore"] if c in out.columns]
        return out[keep].dropna(subset=["Mean"])
    return pd.DataFrame()


def _fidelity_metric_catalog(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    higher_better = {
        "Quality_Score",
        "Quality",
        "KS_Complement",
        "Cosine_Similarity",
        "Gower_Similarity",
    }
    unit_interval = {
        "Quality_Score",
        "Quality",
        "KS_Complement",
        "JS_Divergence",
        "Gower_Distance",
    }
    labels = {
        "PCA_Mean_Error": "PCA Mean Error",
        "PCA_Mean_Error_Pct": "PCA Mean Error Pct",
        "Outlier_Count_Diff": "Outlier Count Diff",
        "Gower_Similarity": "Gower Similarity",
    }
    catalog = []
    for metric, grp in df.groupby("Metric", dropna=False):
        catalog.append({
            "id": metric,
            "label": labels.get(metric, str(metric).replace("_", " ")),
            "count": int(len(grp)),
            "higher_is_better": metric in higher_better,
            "is_unit_interval": metric in unit_interval,
        })
    catalog.sort(key=lambda x: x["count"], reverse=True)
    return catalog


def _privacy_metric_catalog(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    catalog = []
    for metric, grp in df.groupby("Metric", dropna=False):
        catalog.append({
            "id": metric,
            "label": str(metric).replace("_", " "),
            "count": int(len(grp)),
            "lower_is_better": metric not in {
                "Hungarian_Cosine_Similarity",
                "Cosine_Similarity",
                "Num_Matches",  # subsample size used for matching, not a risk score
            },
            "is_similarity": "Cosine" in str(metric) or metric == "MIA_AUC",
            "is_sample_size": metric == "Num_Matches",
        })
    catalog.sort(key=lambda x: x["count"], reverse=True)
    return catalog


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df.empty:
        return []
    chunk = df.head(limit) if limit else df
    return json.loads(chunk.to_json(orient="records"))


def export_dashboard_data(output_dir: Path | None = None) -> Path:
    out = output_dir or DOCS_DATA
    out.mkdir(parents=True, exist_ok=True)

    utility_long = _read_csv("MasterData/utility_long.csv")
    fidelity = _read_csv("MasterData/fidelity_long.csv")
    privacy = _read_csv("MasterData/privacy_long.csv")
    tradeoff = _read_csv("Supplementary/tradeoff_points.csv")
    inventory = _read_csv("MasterData/file_inventory.csv")

    clf_stats = _read_csv("Supplementary/utility_classification_stats.csv")
    reg_stats = _read_csv("Supplementary/utility_regression_stats.csv")
    fid_stats = _read_csv("Supplementary/fidelity_stats.csv")
    priv_stats = _read_csv("Supplementary/privacy_stats.csv")
    wilcoxon = _read_csv("Supplementary/pairwise_wilcoxon.csv")
    effects = _read_csv("Supplementary/effect_sizes.csv")

    # --- Meta / overview ---
    meta = {
        "generators": [
            "CTGAN", "CopulaGAN", "TVAE", "GaussianCopula",
            "WGAN_GP", "CTABGAN", "TabDDPM", "ForestDiffusion",
        ],
        "n_datasets": int(utility_long["Dataset"].nunique()) if not utility_long.empty else 0,
        "n_utility_rows": len(utility_long),
        "n_fidelity_rows": len(fidelity),
        "n_privacy_rows": len(privacy),
        "n_files": len(inventory),
    }
    summary_path = RESULTS_DIR / "Supplementary" / "pipeline_summary.json"
    if summary_path.exists():
        meta.update(json.loads(summary_path.read_text(encoding="utf-8")))

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- Utility: TRTR/TSTR aggregated ---
    if not utility_long.empty:
        util_agg = (
            utility_long[utility_long["EvaluationType"].isin(["TRTR", "TSTR", "Gap"])]
            .groupby(
                ["Dataset", "Generator", "TaskType", "Metric", "EvaluationType"],
                dropna=False,
            )
            .agg(Mean=("Mean", "mean"), Std=("Std", "mean"), Count=("Value", "count"))
            .reset_index()
        )
        (out / "utility_agg.json").write_text(
            json.dumps(_records(util_agg), indent=2), encoding="utf-8"
        )

        # Per classifier TRTR/TSTR for classification
        clf_detail = utility_long[
            (utility_long["TaskType"] == "classification")
            & (utility_long["EvaluationType"].isin(["TRTR", "TSTR"]))
            & (utility_long["Metric"].isin(["Accuracy", "F1", "Precision", "Recall"]))
        ][["Dataset", "Generator", "Classifier", "Metric", "EvaluationType", "Mean", "Std"]]
        (out / "utility_classifier.json").write_text(
            json.dumps(_records(clf_detail), indent=2), encoding="utf-8"
        )

        # Per regressor TRTR/TSTR for regression
        reg_model_col = next(
            (c for c in ("RegressionModel", "Regressor") if c in utility_long.columns),
            None,
        )
        if reg_model_col:
            reg_detail = utility_long[
                (utility_long["TaskType"] == "regression")
                & (utility_long["EvaluationType"].isin(["TRTR", "TSTR"]))
                & (utility_long["Metric"].isin(["R2", "RMSE", "MAE", "MSE"]))
                & (utility_long[reg_model_col].notna())
            ][
                ["Dataset", "Generator", reg_model_col, "Metric", "EvaluationType", "Mean", "Std"]
            ].rename(columns={reg_model_col: "Regressor"})
            (out / "utility_regressor.json").write_text(
                json.dumps(_records(reg_detail), indent=2), encoding="utf-8"
            )

        # Summary gaps per generator per dataset
        gaps = utility_long[
            utility_long["Metric"].str.contains("Drop|Increase|Gap", na=False, regex=True)
        ].groupby(["Dataset", "Generator", "TaskType", "Metric"], dropna=False)["Mean"].mean().reset_index()
        (out / "utility_gaps.json").write_text(
            json.dumps(_records(gaps), indent=2), encoding="utf-8"
        )

    # --- Fidelity & privacy ---
    fid_src = _load_fidelity_long()
    if fid_src.empty:
        fid_src = fid_stats if not fid_stats.empty else fidelity
    priv_src = _load_privacy_long()
    if priv_src.empty:
        priv_src = priv_stats if not priv_stats.empty else privacy
    (out / "fidelity.json").write_text(json.dumps(_records(fid_src), indent=2), encoding="utf-8")
    (out / "fidelity_metrics.json").write_text(
        json.dumps(_fidelity_metric_catalog(fid_src), indent=2), encoding="utf-8"
    )
    (out / "privacy.json").write_text(json.dumps(_records(priv_src), indent=2), encoding="utf-8")
    (out / "privacy_metrics.json").write_text(
        json.dumps(_privacy_metric_catalog(priv_src), indent=2), encoding="utf-8"
    )

    # --- Trade-off & rankings ---
    if not tradeoff.empty:
        (out / "tradeoff.json").write_text(json.dumps(_records(tradeoff), indent=2), encoding="utf-8")

    weighted = _read_csv("Tables/weighted_rankings.csv")
    borda = _read_csv("Tables/borda_rankings.csv")
    if not weighted.empty:
        weighted.index.name = "index"
        (out / "rankings_weighted.json").write_text(
            json.dumps(_records(weighted.reset_index()), indent=2), encoding="utf-8"
        )
    if not borda.empty:
        borda.index.name = "index"
        (out / "rankings_borda.json").write_text(
            json.dumps(_records(borda.reset_index()), indent=2), encoding="utf-8"
        )

    # --- Statistics ---
    pca_errors = _extract_pca_error_stats()
    stats_payload = {
        "pca_errors": _records(pca_errors),
        "wilcoxon": _records(wilcoxon),
        "effect_sizes": _records(effects),
        "classification_stats": _records(clf_stats, limit=2000),
        "regression_stats": _records(reg_stats, limit=2000),
    }
    (out / "statistics.json").write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    if not pca_errors.empty:
        (out / "notebook_error_stats.json").write_text(
            json.dumps(_records(pca_errors), indent=2), encoding="utf-8"
        )

    # --- Correlation trade-off (Cancer + Mushroom primary means) ---
    corr_payload = _export_correlation_tradeoff()
    (out / "correlation_tradeoff.json").write_text(
        json.dumps(corr_payload, indent=2), encoding="utf-8"
    )

    # --- Coverage matrix ---
    if not utility_long.empty:
        coverage = (
            utility_long.groupby(["Dataset", "Generator"], dropna=False)
            .size()
            .reset_index(name="Rows")
        )
        coverage["Available"] = (coverage["Rows"] > 0).astype(int)
        (out / "coverage.json").write_text(
            json.dumps(_records(coverage), indent=2), encoding="utf-8"
        )

    return out


def _export_correlation_tradeoff() -> dict:
    """
    Primary generator-mean panels from Correlation Trade-off Analysis
    (Cancer + Mushroom), matching Results/.../Correlation_Tradeoff_Analysis.
    """
    roots = [
        REPO_ROOT / "Results" / "Two_Datasets_Assessment" / "Correlation_Tradeoff_Analysis",
        REPO_ROOT / "Results" / "Correlation_Tradeoff_Analysis",
    ]
    root = next((p for p in roots if p.exists()), None)
    analyses: list[dict] = []
    if root is None:
        return {"analyses": [], "source": None}

    # Prefer Processed_Data flat CSVs; fall back to nested analysis folders
    candidates = sorted((root / "Processed_Data").glob("generator_means_*.csv"))
    if not candidates:
        candidates = sorted(root.rglob("generator_means_*.csv"))

    seen: set[str] = set()
    for path in candidates:
        key = path.stem.replace("generator_means_", "")  # e.g. Accuracy_MIA
        if key in seen:
            continue
        seen.add(key)
        parts = key.split("_", 1)
        util_metric = parts[0] if parts else key
        priv_key = parts[1] if len(parts) > 1 else "MIA"
        df = pd.read_csv(path)
        if df.empty or "Generator" not in df.columns:
            continue
        rows = []
        for i, row in df.iterrows():
            rows.append({
                "PointId": int(i) + 1 if "PointId" not in df.columns else int(row.get("PointId", i) + 1),
                "Generator": str(row["Generator"]),
                "Utility": float(row["Utility"]) if pd.notna(row.get("Utility")) else None,
                "UtilityStd": float(row["UtilityStd"]) if pd.notna(row.get("UtilityStd")) else None,
                "Fidelity": float(row["Fidelity"]) if pd.notna(row.get("Fidelity")) else None,
                "FidelityStd": float(row["FidelityStd"]) if pd.notna(row.get("FidelityStd")) else None,
                "Privacy": float(row["Privacy"]) if pd.notna(row.get("Privacy")) else None,
                "PrivacyStd": float(row["PrivacyStd"]) if pd.notna(row.get("PrivacyStd")) else None,
                "UtilityMetric": str(row.get("UtilityMetric", util_metric)),
                "FidelityMetric": str(row.get("FidelityMetric", "Quality_Score")),
                "PrivacyMetric": str(row.get("PrivacyMetric", priv_key)),
                "PrivacyLabel": str(row.get("PrivacyLabel", priv_key)),
                "HigherIsPrivate": bool(row.get("HigherIsPrivate", True))
                if not pd.isna(row.get("HigherIsPrivate", True))
                else True,
            })
        # Stable generator order
        order = {
            "CTGAN": 0, "CopulaGAN": 1, "TVAE": 2, "GaussianCopula": 3,
            "WGAN_GP": 4, "CTABGAN": 5, "TabDDPM": 6, "ForestDiffusion": 7,
        }
        rows.sort(key=lambda r: order.get(r["Generator"], 99))
        for i, r in enumerate(rows, start=1):
            r["PointId"] = i
        analyses.append({
            "id": key,
            "label": f"{util_metric} · {priv_key}",
            "utility_metric": util_metric,
            "privacy_metric": priv_key,
            "privacy_label": rows[0]["PrivacyLabel"] if rows else priv_key,
            "higher_is_private": rows[0]["HigherIsPrivate"] if rows else True,
            "generators": rows,
        })

    return {
        "source": str(root.relative_to(REPO_ROOT)) if root else None,
        "analyses": analyses,
    }


if __name__ == "__main__":
    path = export_dashboard_data()
    print(f"Exported dashboard data to {path}")
