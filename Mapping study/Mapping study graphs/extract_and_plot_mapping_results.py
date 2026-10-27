#!/usr/bin/env python3
"""Extract existing mapping results for ALL 15 datasets and create publication figures.

Does NOT recompute mappings or modify notebooks.
Primary curated source: Agreed analysis/mapping_cost_comparison.csv (15 datasets).
Enriched with excel sheets/2. Privacy/* and notebook One-to-One Greedy outputs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
SYNTH = ROOT.parent
GEN = SYNTH / "Generators"
PRIV = SYNTH / "excel sheets" / "2. Privacy"
MAP_CSV = SYNTH / "Agreed analysis" / "mapping_cost_comparison.csv"
OUT_EXT = ROOT / "extracted_results"
OUT_GRAPH = ROOT / "graphs"
OUT_SUM = ROOT / "summary_tables"

GENERATOR_ORDER = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN",
    "TabDDPM",
    "ForestDiffusion",
]
METHOD_ORDER = ["Greedy", "One-to-One Greedy", "Hungarian"]
DPI = 300

DATASET_ORDER = [
    "1. Cancer",
    "2. Alzhimers",
    "3. Adult",
    "4. Forest cover dataset",
    "5. Bank Markting",
    "6. Wine dataset",
    "7. CDC diabetes dataset",
    "8. Mushroom dataset",
    "9. MAGIC Gamma Telescope",
    "10. Metro interstate",
    "11. online shopping",
    "12. Air Quality",
    "13. Concrete Compressive Strength",
    "14. Energy Efficiency",
    "15. Real Estate Valuation",
]

SHORT = {
    "1. Cancer": "Cancer",
    "2. Alzhimers": "Alzheimers",
    "3. Adult": "Adult",
    "4. Forest cover dataset": "ForestCover",
    "5. Bank Markting": "Bank",
    "6. Wine dataset": "Wine",
    "7. CDC diabetes dataset": "CDC_Diabetes",
    "8. Mushroom dataset": "Mushroom",
    "9. MAGIC Gamma Telescope": "MAGIC",
    "10. Metro interstate": "Metro",
    "11. online shopping": "OnlineShop",
    "12. Air Quality": "AirQuality",
    "13. Concrete Compressive Strength": "Concrete",
    "14. Energy Efficiency": "Energy",
    "15. Real Estate Valuation": "RealEstate",
}

# Latest Adult/Alzheimer generator Excel (prefer over older curated sheets when present)
LATEST_HUNG_COSINE = {
    "2. Alzhimers": [
        GEN / "SDV models" / "Hungarian_Matchings_All_Models_AD.xlsx",
        GEN / "Other GANS" / "Hungarian_Matchings_All_Models.xlsx",
        GEN / "Diffusion GANs" / "Hungarian_Matchings_All_Models.xlsx",
    ],
    "3. Adult": [
        GEN / "SDV models" / "Matchings_All_Models_Adult_Census.xlsx",
        GEN / "Other GANS" / "Adult_Hungarian_Matchings_All_Models.xlsx",
        GEN / "Diffusion GANs" / "Adult_Hungarian_Matchings_All_Models.xlsx",
    ],
}
LATEST_HUNG_MAHA = {
    "2. Alzhimers": [
        GEN / "SDV models" / "Hungarian_Mahalanobis_Four_Models_AD.xlsx",
        GEN / "Other GANS" / "Hungarian_Mahalanobis_WGAN_CTABGAN.xlsx",
        GEN / "Diffusion GANs" / "Hungarian_Mahalanobis_WGAN_TabDDPM.xlsx",
    ],
    "3. Adult": [
        GEN / "SDV models" / "Hungarian_Mahalanobis_Four_Models_Adult_Census.xlsx",
        GEN / "Other GANS" / "Adult_Hungarian_Mahalanobis.xlsx",
        GEN / "Diffusion GANs" / "Adult_Hungarian_Mahalanobis.xlsx",
    ],
}

_fonts_dir = Path(__file__).resolve().parent / "fonts"
if _fonts_dir.is_dir():
    from matplotlib import font_manager as fm

    for _ttf in _fonts_dir.glob("*.ttf"):
        try:
            fm.fontManager.addfont(str(_ttf))
        except (OSError, RuntimeError, ValueError):
            pass

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman",
            "Liberation Serif",
            "TeX Gyre Termes",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "svg.fonttype": "none",  # keep text as text in SVG
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    }
)


def _norm_gen(name: str) -> str:
    return str(name).strip().replace("WGAN-GP", "WGAN_GP").replace("WGAN GP", "WGAN_GP")


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def load_mapping_cost_csv() -> pd.DataFrame:
    """All 15 datasets — Hungarian cosine / Mahalanobis + pairwise cosine reference."""
    df = pd.read_csv(MAP_CSV)
    df["Generator"] = df["Generator"].map(_norm_gen)
    rows = []
    for _, r in df.iterrows():
        base = {
            "Dataset": r["Dataset"],
            "Dataset_Short": r.get("Dataset_Short", SHORT.get(r["Dataset"], r["Dataset"])),
            "Generator": r["Generator"],
            "Num_Matches": r.get("Num_Matches", np.nan),
            "Source_File": "Agreed analysis/mapping_cost_comparison.csv",
        }
        if pd.notna(r.get("Hungarian_Cosine")):
            rows.append(
                {
                    **base,
                    "Mapping_Method": "Hungarian",
                    "Metric": "Cosine Similarity",
                    "Mean": float(r["Hungarian_Cosine"]),
                    "Source_Note": "curated_mapping_cost_comparison",
                }
            )
        if pd.notna(r.get("Hungarian_Mahalanobis")):
            rows.append(
                {
                    **base,
                    "Mapping_Method": "Hungarian",
                    "Metric": "Mahalanobis Distance",
                    "Mean": float(r["Hungarian_Mahalanobis"]),
                    "Source_Note": "curated_mapping_cost_comparison",
                }
            )
        # Pairwise cosine is NOT a mapping method; keep separately as reference
        if pd.notna(r.get("Pairwise_Cosine")):
            rows.append(
                {
                    **base,
                    "Mapping_Method": "Pairwise (no assignment)",
                    "Metric": "Cosine Similarity",
                    "Mean": float(r["Pairwise_Cosine"]),
                    "Source_Note": "pairwise_mean_all_pairs_NOT_a_mapping_method",
                }
            )
    return pd.DataFrame(rows)


def enrich_from_privacy_excel(summary: pd.DataFrame) -> pd.DataFrame:
    """Add Median/Min/Max/Std from excel sheets/2. Privacy when available."""
    extra_rows = []
    for ds_dir in sorted(PRIV.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset = ds_dir.name
        # Hungarian cosine
        hm = ds_dir / "Hungarian_Matching.xlsx"
        if hm.exists():
            df = pd.read_excel(hm)
            gen_col = "Generator" if "Generator" in df.columns else "Model"
            for _, r in df.iterrows():
                gen = _norm_gen(r[gen_col])
                extra_rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset_Short": SHORT.get(dataset, dataset),
                        "Generator": gen,
                        "Mapping_Method": "Hungarian",
                        "Metric": "Cosine Similarity",
                        "Mean": float(r["Avg_Cosine_Similarity"]) if pd.notna(r.get("Avg_Cosine_Similarity")) else np.nan,
                        "Median": float(r["Median_Cosine_Similarity"]) if pd.notna(r.get("Median_Cosine_Similarity")) else np.nan,
                        "Min": float(r["Min_Cosine_Similarity"]) if pd.notna(r.get("Min_Cosine_Similarity")) else np.nan,
                        "Max": float(r["Max_Cosine_Similarity"]) if pd.notna(r.get("Max_Cosine_Similarity")) else np.nan,
                        "Std": np.nan,
                        "Num_Matches": r.get("Num_Matches", np.nan),
                        "Source_File": str(hm.relative_to(SYNTH)),
                        "Source_Note": "excel_sheets_privacy_hungarian_matching",
                    }
                )
        # Hungarian Mahalanobis summary
        ms = ds_dir / "Mahalanobis_Summary.xlsx"
        if ms.exists():
            df = pd.read_excel(ms)
            if "Mean_Distance" in df.columns and "Model" in df.columns:
                # Deduplicate models if sheet mixes cosine cols
                for gen, gdf in df.groupby(df["Model"].map(_norm_gen)):
                    # Prefer rows with Mean_Distance present
                    gdf = gdf.dropna(subset=["Mean_Distance"])
                    if gdf.empty:
                        continue
                    r = gdf.iloc[0]
                    extra_rows.append(
                        {
                            "Dataset": dataset,
                            "Dataset_Short": SHORT.get(dataset, dataset),
                            "Generator": gen,
                            "Mapping_Method": "Hungarian",
                            "Metric": "Mahalanobis Distance",
                            "Mean": float(r["Mean_Distance"]),
                            "Median": float(r["Median_Distance"]) if pd.notna(r.get("Median_Distance")) else np.nan,
                            "Min": float(r["Min_Distance"]) if pd.notna(r.get("Min_Distance")) else np.nan,
                            "Max": float(r["Max_Distance"]) if pd.notna(r.get("Max_Distance")) else np.nan,
                            "Std": float(r["Std_Distance"]) if pd.notna(r.get("Std_Distance")) else np.nan,
                            "Num_Matches": r.get("Num_Matches", np.nan),
                            "Source_File": str(ms.relative_to(SYNTH)),
                            "Source_Note": "excel_sheets_privacy_mahalanobis_summary",
                        }
                    )
    return pd.DataFrame(extra_rows)


def load_latest_adult_alzheimer_excel() -> pd.DataFrame:
    rows = []
    for dataset, paths in LATEST_HUNG_COSINE.items():
        for path in paths:
            if not path.exists():
                continue
            xl = pd.ExcelFile(path)
            sheet = "Summary" if "Summary" in xl.sheet_names else next(
                (s for s in xl.sheet_names if "summ" in s.lower()), None
            )
            if sheet is None:
                continue
            df = pd.read_excel(path, sheet_name=sheet)
            mean_col = "Avg_Cosine_Similarity" if "Avg_Cosine_Similarity" in df.columns else "Avg_cosine_similarity"
            for _, r in df.iterrows():
                rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset_Short": SHORT[dataset],
                        "Generator": _norm_gen(r["Model"]),
                        "Mapping_Method": "Hungarian",
                        "Metric": "Cosine Similarity",
                        "Mean": float(r[mean_col]),
                        "Median": float(r.get("Median_Cosine_Similarity", r.get("Median_cosine_similarity", np.nan))),
                        "Min": float(r.get("Min_Cosine_Similarity", r.get("Min_cosine_similarity", np.nan))),
                        "Max": float(r.get("Max_Cosine_Similarity", r.get("Max_cosine_similarity", np.nan))),
                        "Std": np.nan,
                        "Num_Matches": r.get("Num_Matches", np.nan),
                        "Source_File": str(path.relative_to(SYNTH)),
                        "Source_Note": "latest_generator_excel_preferred",
                    }
                )
    for dataset, paths in LATEST_HUNG_MAHA.items():
        for path in paths:
            if not path.exists():
                continue
            xl = pd.ExcelFile(path)
            sheet = "Summary" if "Summary" in xl.sheet_names else next(
                (s for s in xl.sheet_names if "summ" in s.lower()), None
            )
            if sheet is None:
                continue
            df = pd.read_excel(path, sheet_name=sheet)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset_Short": SHORT[dataset],
                        "Generator": _norm_gen(r["Model"]),
                        "Mapping_Method": "Hungarian",
                        "Metric": "Mahalanobis Distance",
                        "Mean": float(r["Mean_Distance"]),
                        "Median": float(r["Median_Distance"]) if pd.notna(r.get("Median_Distance")) else np.nan,
                        "Min": float(r["Min_Distance"]) if pd.notna(r.get("Min_Distance")) else np.nan,
                        "Max": float(r["Max_Distance"]) if pd.notna(r.get("Max_Distance")) else np.nan,
                        "Std": float(r["Std_Distance"]) if pd.notna(r.get("Std_Distance")) else np.nan,
                        "Num_Matches": r.get("Num_Matches", np.nan),
                        "Source_File": str(path.relative_to(SYNTH)),
                        "Source_Note": "latest_generator_excel_preferred",
                    }
                )
    return pd.DataFrame(rows)


def _cell_text(cell: dict) -> str:
    texts = []
    for o in cell.get("outputs") or []:
        if o.get("output_type") == "stream":
            texts.append("".join(o.get("text", [])))
        elif o.get("output_type") in ("execute_result", "display_data"):
            data = o.get("data") or {}
            if "text/plain" in data:
                t = data["text/plain"]
                texts.append("".join(t) if isinstance(t, list) else t)
    return "\n".join(texts)


def _dataset_from_notebook_path(path: Path) -> str | None:
    name = path.name.lower()
    mapping = [
        ("cancer", "1. Cancer"),
        ("alzhim", "2. Alzhimers"),
        ("adult", "3. Adult"),
        ("forest", "4. Forest cover dataset"),
        ("bank", "5. Bank Markting"),
        ("wine", "6. Wine dataset"),
        ("cdc", "7. CDC diabetes dataset"),
        ("heart", "7. CDC diabetes dataset"),  # legacy heart naming — verify
        ("mushroom", "8. Mushroom dataset"),
        ("magic", "9. MAGIC Gamma Telescope"),
        ("metro", "10. Metro interstate"),
        ("shopping", "11. online shopping"),
        ("online", "11. online shopping"),
        ("air", "12. Air Quality"),
        ("concrete", "13. Concrete Compressive Strength"),
        ("energy", "14. Energy Efficiency"),
        ("real estate", "15. Real Estate Valuation"),
        ("real_estate", "15. Real Estate Valuation"),
    ]
    # Heart is NOT CDC — fix
    if "heart" in name and "cdc" not in name:
        return None  # Heart not in the 15 canonical list used by paper results
    for key, ds in mapping:
        if key in name:
            if key == "heart":
                continue
            return ds
    return None


def parse_o2o_greedy_all_notebooks() -> pd.DataFrame:
    rows = []
    for path in GEN.rglob("*.ipynb"):
        if "Experiment" in str(path) or "CTAB-GAN" in str(path):
            continue
        dataset = _dataset_from_notebook_path(path)
        if dataset is None:
            continue
        try:
            nb = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))
            if "greedy_one_to_one_matching" not in src:
                continue
            text = _cell_text(cell)
            if not text.strip():
                continue
            for m in re.finditer(
                r"✓ Processing ([A-Za-z0-9_]+):.*?\n\s*Greedy mean:\s*([0-9.eE+\-]+)\s*vs Hungarian:\s*([0-9.eE+\-]+)",
                text,
            ):
                rows.append(
                    {
                        "Dataset": dataset,
                        "Dataset_Short": SHORT.get(dataset, dataset),
                        "Generator": _norm_gen(m.group(1)),
                        "Mapping_Method": "One-to-One Greedy",
                        "Metric": "Mahalanobis Distance",
                        "Mean": float(m.group(2)),
                        "Median": np.nan,
                        "Min": np.nan,
                        "Max": np.nan,
                        "Std": np.nan,
                        "Num_Matches": np.nan,
                        "Source_File": str(path.relative_to(SYNTH)),
                        "Source_Note": "notebook_output_greedy_mean",
                    }
                )
            table = re.search(
                r"Greedy one-to-one matching \(Mahalanobis distance\):\s*\n\s*(.*?)\n\s*Comparison:",
                text,
                re.S,
            )
            if table:
                for line in table.group(1).splitlines():
                    mm = re.match(
                        r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)\s*$",
                        line,
                    )
                    if not mm:
                        continue
                    gen = _norm_gen(mm.group(1))
                    if any(
                        r["Dataset"] == dataset
                        and r["Generator"] == gen
                        and r["Mapping_Method"] == "One-to-One Greedy"
                        and Path(r["Source_File"]).name == path.name
                        for r in rows
                    ):
                        continue
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Dataset_Short": SHORT.get(dataset, dataset),
                            "Generator": gen,
                            "Mapping_Method": "One-to-One Greedy",
                            "Metric": "Mahalanobis Distance",
                            "Mean": float(mm.group(3)),
                            "Median": np.nan,
                            "Min": np.nan,
                            "Max": np.nan,
                            "Std": np.nan,
                            "Num_Matches": np.nan,
                            "Source_File": str(path.relative_to(SYNTH)),
                            "Source_Note": "notebook_output_summary_table",
                        }
                    )
    return pd.DataFrame(rows)


def merge_prefer_latest(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge sources; prefer latest_generator_excel > excel_sheets > curated CSV for same key."""
    priority = {
        "latest_generator_excel_preferred": 0,
        "excel_sheets_privacy_hungarian_matching": 1,
        "excel_sheets_privacy_mahalanobis_summary": 1,
        "notebook_output_greedy_mean": 0,
        "notebook_output_summary_table": 1,
        "curated_mapping_cost_comparison": 2,
        "pairwise_mean_all_pairs_NOT_a_mapping_method": 3,
    }
    df = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True, sort=False)
    for c in ["Median", "Min", "Max", "Std", "Num_Matches", "Dataset_Short", "Source_File", "Source_Note"]:
        if c not in df.columns:
            df[c] = np.nan
    df["Generator"] = df["Generator"].map(_norm_gen)
    df["_prio"] = df["Source_Note"].map(lambda s: priority.get(str(s), 9))
    df = df.sort_values("_prio")
    df = df.drop_duplicates(subset=["Dataset", "Generator", "Mapping_Method", "Metric"], keep="first")
    return df.drop(columns=["_prio"])


def build_coverage(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mapping_only = summary[summary["Mapping_Method"].isin(METHOD_ORDER)]
    for dataset in DATASET_ORDER:
        for gen in GENERATOR_ORDER:
            for method in METHOD_ORDER:
                for metric in ["Cosine Similarity", "Mahalanobis Distance"]:
                    hit = mapping_only[
                        (mapping_only["Dataset"] == dataset)
                        & (mapping_only["Generator"] == gen)
                        & (mapping_only["Mapping_Method"] == method)
                        & (mapping_only["Metric"] == metric)
                    ]
                    rows.append(
                        {
                            "Dataset": dataset,
                            "Dataset_Short": SHORT[dataset],
                            "Generator": gen,
                            "Mapping_Method": method,
                            "Metric": metric,
                            "Available": bool(len(hit)),
                            "Mean": float(hit["Mean"].iloc[0]) if len(hit) else np.nan,
                        }
                    )
    return pd.DataFrame(rows)


def plot_overview_heatmap(summary: pd.DataFrame, metric: str, method: str, outstem: Path, higher_better: bool) -> None:
    sub = summary[
        (summary["Metric"] == metric)
        & (summary["Mapping_Method"] == method)
        & (summary["Dataset"].isin(DATASET_ORDER))
    ].copy()
    if sub.empty:
        return
    mat = sub.pivot_table(index="Dataset", columns="Generator", values="Mean", aggfunc="first")
    mat = mat.reindex(index=DATASET_ORDER, columns=GENERATOR_ORDER)
    mat.index = [SHORT[i] for i in mat.index]
    fig, ax = plt.subplots(figsize=(11, 7.5))
    cmap = "YlGn" if higher_better else "YlOrRd_r"
    sns.heatmap(mat, annot=True, fmt=".3f", cmap=cmap, ax=ax, linewidths=0.3)
    direction = "higher = better" if higher_better else "lower = better"
    ax.set_title(f"All 15 datasets — {method} {metric}\n({direction})")
    ax.set_xlabel("Generator")
    ax.set_ylabel("Dataset")
    fig.tight_layout()
    save_fig(fig, outstem)


def plot_overview_bars_by_dataset(summary: pd.DataFrame, metric: str, method: str, outstem: Path, ylabel: str) -> None:
    sub = summary[(summary["Metric"] == metric) & (summary["Mapping_Method"] == method)].copy()
    if sub.empty:
        return
    # mean across generators per dataset
    g = sub.groupby("Dataset", as_index=False)["Mean"].mean()
    g["Dataset"] = pd.Categorical(g["Dataset"], categories=DATASET_ORDER, ordered=True)
    g = g.sort_values("Dataset")
    g["Short"] = g["Dataset"].map(SHORT)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.bar(range(len(g)), g["Mean"], color="#54A24B" if "Cosine" in metric else "#E45756")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g["Short"], rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"All 15 datasets — mean {method} {metric} (averaged over available generators)")
    fig.tight_layout()
    save_fig(fig, outstem)


def plot_per_dataset_method_bars(summary: pd.DataFrame, dataset: str, metric: str, outstem: Path, ylabel: str) -> None:
    sub = summary[
        (summary["Dataset"] == dataset)
        & (summary["Metric"] == metric)
        & (summary["Mapping_Method"].isin(METHOD_ORDER))
    ].copy()
    if sub.empty:
        return
    gens = [g for g in GENERATOR_ORDER if g in set(sub["Generator"])]
    methods = [m for m in METHOD_ORDER if m in set(sub["Mapping_Method"])]
    if not gens:
        return
    x = np.arange(len(gens))
    width = 0.8 / max(len(methods), 1)
    colors = {"Greedy": "#4C78A8", "One-to-One Greedy": "#F58518", "Hungarian": "#54A24B"}
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    for i, method in enumerate(methods):
        vals = []
        for g in gens:
            hit = sub[(sub["Generator"] == g) & (sub["Mapping_Method"] == method)]
            vals.append(float(hit["Mean"].iloc[0]) if len(hit) else np.nan)
        ax.bar(x + i * width - 0.4 + width / 2, vals, width, label=method, color=colors.get(method))
    ax.set_xticks(x)
    ax.set_xticklabels(gens, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{SHORT.get(dataset, dataset)}: {metric}")
    ax.legend(frameon=True)
    fig.tight_layout()
    save_fig(fig, outstem)


def plot_coverage_all(coverage: pd.DataFrame, outstem: Path) -> None:
    cov = coverage.copy()
    cov["ok"] = cov["Available"].astype(int)
    # fraction of 48 conditions per dataset
    frac = cov.groupby("Dataset", as_index=False)["ok"].mean()
    frac["Dataset"] = pd.Categorical(frac["Dataset"], categories=DATASET_ORDER, ordered=True)
    frac = frac.sort_values("Dataset")
    frac["Short"] = frac["Dataset"].map(SHORT)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.bar(range(len(frac)), frac["ok"] * 48, color="#4C78A8")
    ax.axhline(48, color="gray", ls="--", lw=0.8, label="48 = full grid")
    ax.set_xticks(range(len(frac)))
    ax.set_xticklabels(frac["Short"], rotation=35, ha="right")
    ax.set_ylabel("# conditions extracted (max 48)")
    ax.set_title("Coverage of 8 generators × 3 methods × 2 metrics across 15 datasets")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, outstem)


def main() -> None:
    curated = load_mapping_cost_csv()
    privacy = enrich_from_privacy_excel(curated)
    latest = load_latest_adult_alzheimer_excel()
    o2o = parse_o2o_greedy_all_notebooks()

    summary = merge_prefer_latest([curated, privacy, latest, o2o])
    summary["Dataset_Short"] = summary["Dataset"].map(lambda d: SHORT.get(d, d))

    OUT_EXT.mkdir(parents=True, exist_ok=True)
    OUT_SUM.mkdir(parents=True, exist_ok=True)

    summary.to_csv(OUT_EXT / "all_mapping_summary_15datasets.csv", index=False)
    summary[summary["Metric"] == "Cosine Similarity"].to_csv(OUT_EXT / "cosine_similarity_results.csv", index=False)
    summary[summary["Metric"] == "Mahalanobis Distance"].to_csv(
        OUT_EXT / "mahalanobis_distance_results.csv", index=False
    )
    # Keep previous Adult/Alzheimer-only file name as alias to full summary for compatibility
    summary.to_csv(OUT_EXT / "all_mapping_summary.csv", index=False)

    coverage = build_coverage(summary)
    coverage.to_csv(OUT_SUM / "condition_coverage_48x15datasets.csv", index=False)
    summary.to_csv(OUT_SUM / "mapping_summary_table.csv", index=False)

    # Per-dataset availability counts
    avail = (
        coverage.groupby(["Dataset", "Dataset_Short"], as_index=False)["Available"]
        .sum()
        .rename(columns={"Available": "N_available_of_48"})
    )
    avail.to_csv(OUT_SUM / "coverage_by_dataset.csv", index=False)

    # --- Overview figures (all 15) ---
    plot_overview_heatmap(
        summary,
        "Cosine Similarity",
        "Hungarian",
        OUT_GRAPH / "comparison" / "ALL15_Hungarian_cosine_heatmap",
        True,
    )
    plot_overview_heatmap(
        summary,
        "Mahalanobis Distance",
        "Hungarian",
        OUT_GRAPH / "comparison" / "ALL15_Hungarian_mahalanobis_heatmap",
        False,
    )
    plot_overview_bars_by_dataset(
        summary,
        "Cosine Similarity",
        "Hungarian",
        OUT_GRAPH / "cosine" / "ALL15_Hungarian_cosine_by_dataset",
        "Mean Hungarian cosine (higher = better)",
    )
    plot_overview_bars_by_dataset(
        summary,
        "Mahalanobis Distance",
        "Hungarian",
        OUT_GRAPH / "mahalanobis" / "ALL15_Hungarian_mahalanobis_by_dataset",
        "Mean Hungarian Mahalanobis (lower = better)",
    )
    plot_coverage_all(coverage, OUT_GRAPH / "comparison" / "ALL15_condition_coverage")

    # One-to-One Greedy overview if enough data
    o2o_sum = summary[summary["Mapping_Method"] == "One-to-One Greedy"]
    if len(o2o_sum):
        plot_overview_heatmap(
            summary,
            "Mahalanobis Distance",
            "One-to-One Greedy",
            OUT_GRAPH / "comparison" / "ALL15_OneToOneGreedy_mahalanobis_heatmap",
            False,
        )

    # Pairwise vs Hungarian cosine (all 15) — from curated comparison
    pair = summary[
        (summary["Metric"] == "Cosine Similarity")
        & (summary["Mapping_Method"].isin(["Pairwise (no assignment)", "Hungarian"]))
    ]
    if len(pair):
        wide = pair.pivot_table(index=["Dataset", "Generator"], columns="Mapping_Method", values="Mean")
        if {"Pairwise (no assignment)", "Hungarian"}.issubset(wide.columns):
            fig, ax = plt.subplots(figsize=(6.5, 6))
            ax.scatter(
                wide["Pairwise (no assignment)"],
                wide["Hungarian"],
                alpha=0.7,
                edgecolor="k",
                linewidth=0.3,
            )
            ax.plot([-0.1, 1.05], [-0.1, 1.05], "k--", lw=0.8)
            ax.set_xlabel("Pairwise cosine (no assignment)")
            ax.set_ylabel("Hungarian cosine")
            ax.set_title("All 15 datasets × generators:\nPairwise vs Hungarian cosine")
            fig.tight_layout()
            save_fig(fig, OUT_GRAPH / "cosine" / "ALL15_pairwise_vs_hungarian_cosine")

    # Per-dataset method comparison bars
    for dataset in DATASET_ORDER:
        short = SHORT[dataset]
        plot_per_dataset_method_bars(
            summary,
            dataset,
            "Cosine Similarity",
            OUT_GRAPH / "cosine" / f"by_dataset/{short}_cosine_method_comparison",
            "Cosine similarity (higher = better)",
        )
        plot_per_dataset_method_bars(
            summary,
            dataset,
            "Mahalanobis Distance",
            OUT_GRAPH / "mahalanobis" / f"by_dataset/{short}_mahalanobis_method_comparison",
            "Mahalanobis distance (lower = better)",
        )

    # Console report
    print("=== 15-dataset extraction complete ===")
    print("Datasets in summary:", sorted(summary["Dataset"].unique().tolist()))
    print("N rows:", len(summary))
    print(avail.to_string(index=False))
    print("\nO2O greedy rows:", len(o2o))
    mapping_only = summary[summary["Mapping_Method"].isin(METHOD_ORDER)]
    print(
        "Mapping-method rows by method:\n",
        mapping_only.groupby(["Mapping_Method", "Metric"]).size().to_string(),
    )


if __name__ == "__main__":
    main()
