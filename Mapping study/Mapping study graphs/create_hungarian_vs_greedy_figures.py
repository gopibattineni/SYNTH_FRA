#!/usr/bin/env python3
"""Hungarian vs Greedy mapping improvement figures — extract-only (no remapping).

Data sources (existing only):
- mapping study graphs/supplementary/detailed_tables/all_mapping_units_long.csv
- Agreed analysis/mapping_cost_comparison.csv
- excel sheets/2. Privacy/*/Mahalanobis_Summary.xlsx
- excel sheets/2. Privacy/*/Hungarian_Matching.xlsx (Hungarian cosine check)

Cosine note:
  Many-to-one Greedy cosine means were not persisted in notebooks (top-10 only).
  Curated Pairwise_Cosine is used as the non-Hungarian cosine baseline and is
  labeled "Pairwise" (not Greedy) in figures and tables.

Mahalanobis:
  Notebook "Greedy" Mahalanobis is One-to-One Greedy; labeled "Greedy" with that
  clarification in README.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "mapping study graphs"
OUT_EXTRACT = BASE / "extracted_results"
OUT_MAIN = BASE / "main_paper"
OUT_SUPP = BASE / "supplementary"

GEN_ORDER = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "CTABGAN",
    "WGAN_GP",
    "TabDDPM",
    "ForestDiffusion",
]
GEN_LABEL = {
    "CTGAN": "CTGAN",
    "CopulaGAN": "Copula GAN",
    "TVAE": "TVAE",
    "GaussianCopula": "Gaussian Copula",
    "CTABGAN": "CTAB-GAN+",
    "WGAN_GP": "WGAN-GP",
    "TabDDPM": "TabDDPM",
    "ForestDiffusion": "Forest Diffusion",
}
DS_ORDER = [
    "Cancer",
    "Alzheimers",
    "Adult",
    "ForestCover",
    "Bank",
    "Wine",
    "CDC",
    "Mushroom",
    "MAGIC",
    "Metro",
    "OnlineShop",
    "AirQuality",
    "Concrete",
    "Energy",
    "RealEstate",
]
DS_LABEL = {
    "Cancer": "Cancer",
    "Alzheimers": "Alzh.",
    "Adult": "Adult",
    "ForestCover": "Forest",
    "Bank": "Bank",
    "Wine": "Wine",
    "CDC": "CDC",
    "Mushroom": "Mush.",
    "MAGIC": "MAGIC",
    "Metro": "Metro",
    "OnlineShop": "Shop",
    "AirQuality": "AirQ.",
    "Concrete": "Concrete",
    "Energy": "Energy",
    "RealEstate": "RE",
}
DS_ALIAS = {
    "Cancer": "Cancer",
    "1. Cancer": "Cancer",
    "Alzheimers": "Alzheimers",
    "Alzhimers": "Alzheimers",
    "2. Alzhimers": "Alzheimers",
    "AD": "Alzheimers",
    "Alzheimer": "Alzheimers",
    "Adult": "Adult",
    "3. Adult": "Adult",
    "Adult_Census": "Adult",
    "ForestCover": "ForestCover",
    "Forest cover dataset": "ForestCover",
    "4. Forest cover dataset": "ForestCover",
    "Bank": "Bank",
    "Bank Markting": "Bank",
    "5. Bank Markting": "Bank",
    "Wine": "Wine",
    "Wine dataset": "Wine",
    "WineQuality": "Wine",
    "6. Wine dataset": "Wine",
    "CDC": "CDC",
    "CDC diabetes dataset": "CDC",
    "CDC_Diabetes": "CDC",
    "Heart": "CDC",
    "7. CDC diabetes dataset": "CDC",
    "Mushroom": "Mushroom",
    "Mushroom dataset": "Mushroom",
    "Secondary_Mushroom": "Mushroom",
    "8. Mushroom dataset": "Mushroom",
    "MAGIC": "MAGIC",
    "MAGIC Gamma Telescope": "MAGIC",
    "9. MAGIC Gamma Telescope": "MAGIC",
    "Metro": "Metro",
    "Metro interstate": "Metro",
    "10. Metro interstate": "Metro",
    "OnlineShop": "OnlineShop",
    "online shopping": "OnlineShop",
    "11. online shopping": "OnlineShop",
    "AirQuality": "AirQuality",
    "Air Quality": "AirQuality",
    "12. Air Quality": "AirQuality",
    "Concrete": "Concrete",
    "Concrete Compressive Strength": "Concrete",
    "13. Concrete Compressive Strength": "Concrete",
    "Energy": "Energy",
    "Energy Efficiency": "Energy",
    "EnergyEfficiency": "Energy",
    "14. Energy Efficiency": "Energy",
    "RealEstate": "RealEstate",
    "Real Estate Valuation": "RealEstate",
    "15. Real Estate Valuation": "RealEstate",
}

VALID_GENS = set(GEN_ORDER)

_NB_DS_KEYS = [
    ("adult", "Adult"),
    ("alzh", "Alzheimers"),
    ("cancer", "Cancer"),
    ("forest", "ForestCover"),
    ("bank", "Bank"),
    ("wine", "Wine"),
    ("cdc", "CDC"),
    ("heart", "CDC"),
    ("mushroom", "Mushroom"),
    ("magic", "MAGIC"),
    ("metro", "Metro"),
    ("online", "OnlineShop"),
    ("shop", "OnlineShop"),
    ("air", "AirQuality"),
    ("concrete", "Concrete"),
    ("energy", "Energy"),
    ("realestate", "RealEstate"),
    ("estate", "RealEstate"),
]


def _guess_ds_from_name(path: Path) -> str | None:
    low = path.name.lower().replace(" ", "").replace("_", "")
    for key, ds in _NB_DS_KEYS:
        if key in low:
            return ds
    return None


def _plausible_maha_pair(greedy: float, hung: float) -> bool:
    """Reject overflow / mis-parsed notebook table cells."""
    if greedy is None or hung is None:
        return False
    if np.isnan(greedy) or np.isnan(hung):
        return False
    if greedy <= 0 or hung <= 0:
        return False
    if greedy > 1e5 or hung > 1e5:
        return False
    if hung > 0 and greedy / hung > 250:
        return False
    return True


def _clean_html_cell(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").strip()
    if text.lower() in {"", "nan", "none"}:
        return ""
    return text


def _parse_html_table(table_html: str) -> tuple[list[str], list[list[str]]]:
    """Return (headers, body_rows) from one pandas-style HTML table."""
    headers: list[str] = []
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S | re.I):
        cells = [
            _clean_html_cell(c)
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S | re.I)
        ]
        if not cells:
            continue
        joined = " ".join(cells).lower()
        if any(k in joined for k in ("greedy mean", "hungarian mean", "greedy total")):
            headers = cells
            continue
        if "model" in joined and "mean" in joined:
            headers = cells
            continue
        if headers and joined.replace("model", "").strip() == "":
            # pandas often adds a second header row containing only "Model"
            continue
        if headers:
            rows.append(cells)
    return headers, rows


def _col_idx(headers: list[str], *names: str) -> int | None:
    lowered = [h.lower() for h in headers]
    for name in names:
        name_l = name.lower()
        for i, h in enumerate(lowered):
            if name_l in h:
                return i
    return None


def _extract_sdv_comparison_from_html(
    html: str, ds: str, nb_name: str
) -> list[dict]:
    """Parse SDV notebooks' comparison_df HTML (sampled greedy vs full Hungarian)."""
    out: list[dict] = []
    for table_html in re.findall(r"<table[^>]*>.*?</table>", html, flags=re.S | re.I):
        headers, body = _parse_html_table(table_html)
        if not headers or not body:
            continue
        g_idx = _col_idx(headers, "greedy mean (sampled)", "greedy mean")
        h_idx = _col_idx(headers, "hungarian mean (full)", "hungarian mean")
        if g_idx is None or h_idx is None:
            continue
        model_idx = 0
        for row in body:
            if len(row) <= max(g_idx, h_idx):
                continue
            gen = norm_gen(row[model_idx])
            if gen not in VALID_GENS:
                continue
            try:
                gv = float(row[g_idx])
                hv = float(row[h_idx])
            except ValueError:
                continue
            if not _plausible_maha_pair(gv, hv):
                continue
            out.append(
                {
                    "Dataset": ds,
                    "Generator": gen,
                    "Greedy_Mahalanobis": gv,
                    "Hungarian_Mahalanobis": hv,
                    "nb_src": nb_name,
                    "nb_kind": "sdv_html_table",
                }
            )
    return out


def _extract_split_greedy_hungarian_html(
    html_blocks: list[str], ds: str, nb_name: str
) -> list[dict]:
    """Parse notebooks that print Greedy Mean and Hungarian Mean in separate tables."""
    greedy_map: dict[str, float] = {}
    hung_map: dict[str, float] = {}
    for html in html_blocks:
        for table_html in re.findall(r"<table[^>]*>.*?</table>", html, flags=re.S | re.I):
            headers, body = _parse_html_table(table_html)
            if not headers or not body:
                continue
            model_idx = _col_idx(headers, "model")
            if model_idx is None:
                model_idx = 0
            header_joined = " ".join(headers).lower()
            g_idx = _col_idx(headers, "greedy mean")
            h_idx = _col_idx(headers, "mean")
            if g_idx is not None and "greedy mean" in header_joined:
                for row in body:
                    if len(row) <= max(model_idx, g_idx):
                        continue
                    gen = norm_gen(row[model_idx])
                    if gen not in VALID_GENS:
                        continue
                    try:
                        greedy_map[gen] = float(row[g_idx])
                    except ValueError:
                        continue
            elif (
                h_idx is not None
                and "greedy" not in header_joined
                and "median" in header_joined
                and "std" in header_joined
            ):
                for row in body:
                    if len(row) <= max(model_idx, h_idx):
                        continue
                    gen = norm_gen(row[model_idx])
                    if gen not in VALID_GENS:
                        continue
                    try:
                        hung_map[gen] = float(row[h_idx])
                    except ValueError:
                        continue
    out: list[dict] = []
    for gen in sorted(set(greedy_map) & set(hung_map)):
        gv, hv = greedy_map[gen], hung_map[gen]
        if not _plausible_maha_pair(gv, hv):
            continue
        out.append(
            {
                "Dataset": ds,
                "Generator": gen,
                "Greedy_Mahalanobis": gv,
                "Hungarian_Mahalanobis": hv,
                "nb_src": nb_name,
                "nb_kind": "split_html_table",
            }
        )
    return out


def extract_notebook_maha_pairs() -> pd.DataFrame:
    """Extract existing Greedy vs Hungarian Mahalanobis means from notebook outputs."""
    pair_re = re.compile(
        r"Processing\s+([A-Za-z0-9_+]+)\s*:[^\n]*\n[ \t]*Greedy mean:\s*([0-9.eE+-]+)\s*vs Hungarian:\s*([0-9.eE+-]+)",
        re.M,
    )
    # Standard comparison table rows (Greedy mean / Hungarian mean columns)
    table_re = re.compile(
        r"^(GaussianCopula|CTGAN|CopulaGAN|TVAE|CTABGAN|WGAN_GP|TabDDPM|ForestDiffusion)"
        r"\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)",
        re.M,
    )
    # Sampled OnlineShop-style: Hungarian mean (sampled) block
    sampled_hung_re = re.compile(
        r"^(CTABGAN|WGAN_GP|TabDDPM|ForestDiffusion|CTGAN|CopulaGAN|TVAE|GaussianCopula)"
        r"\s+([0-9.eE+-]+)\s+[0-9.eE+-]+\s+[0-9.eE+-]+\s*$",
        re.M,
    )
    sampled_greedy_re = re.compile(
        r"^(CTABGAN|WGAN_GP|TabDDPM|ForestDiffusion|CTGAN|CopulaGAN|TVAE|GaussianCopula)"
        r"\s+[0-9.eE+-]+\s+([0-9.eE+-]+)\s*$",
        re.M,
    )

    rows: list[dict] = []
    for nb in (ROOT / "Generators").rglob("*.ipynb"):
        sp = str(nb)
        if "Experiment" in sp or "utility" in sp.lower() or "dataleak" in sp.lower():
            continue
        ds = _guess_ds_from_name(nb)
        if ds is None:
            continue
        try:
            data = json.loads(nb.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        texts: list[str] = []
        html_blocks: list[str] = []
        for cell in data.get("cells", []):
            for out in cell.get("outputs") or []:
                if out.get("output_type") == "stream":
                    texts.append("".join(out.get("text", [])))
                elif out.get("output_type") in ("execute_result", "display_data"):
                    payload = out.get("data", {}) or {}
                    t = payload.get("text/plain")
                    if t:
                        texts.append("".join(t) if isinstance(t, list) else t)
                    h = payload.get("text/html")
                    if h:
                        html_blocks.append("".join(h) if isinstance(h, list) else h)
        rows.extend(_extract_sdv_comparison_from_html("\n".join(html_blocks), ds, nb.name))
        rows.extend(_extract_split_greedy_hungarian_html(html_blocks, ds, nb.name))
        blob = "\n".join(texts)
        for m in pair_re.finditer(blob):
            g = norm_gen(m.group(1))
            if g not in VALID_GENS:
                continue
            rows.append(
                {
                    "Dataset": ds,
                    "Generator": g,
                    "Greedy_Mahalanobis": float(m.group(2)),
                    "Hungarian_Mahalanobis": float(m.group(3)),
                    "nb_src": nb.name,
                    "nb_kind": "print",
                }
            )
        # Standard comparison tables (avoid matching inside "(sampled)" OnlineShop layout)
        if "Greedy mean" in blob and "Hungarian mean" in blob:
            # Restrict to non-sampled comparison text when possible
            scan = blob
            if "Greedy mean (sampled)" in blob:
                # Keep text before first sampled header for standard tables
                scan = blob.split("Greedy mean (sampled)", 1)[0]
            if "Greedy mean" in scan and "Hungarian mean" in scan:
                for m in table_re.finditer(scan):
                    g = norm_gen(m.group(1))
                    if g not in VALID_GENS:
                        continue
                    rows.append(
                        {
                            "Dataset": ds,
                            "Generator": g,
                            "Greedy_Mahalanobis": float(m.group(2)),
                            "Hungarian_Mahalanobis": float(m.group(3)),
                            "nb_src": nb.name,
                            "nb_kind": "table",
                        }
                    )
        if "Greedy mean (sampled)" in blob:
            after = blob.split("Greedy mean (sampled)", 1)[1]
            hung_key = None
            for key in ("Hungarian mean (full)", "Hungarian mean (sampled)"):
                if key in after:
                    hung_key = key
                    break
            if hung_key is not None:
                g_block, h_rest = after.split(hung_key, 1)
                h_block = h_rest[:1200]
                greedy_map = {
                    norm_gen(a): float(b) for a, b in sampled_greedy_re.findall(g_block)
                }
                hung_map = {
                    norm_gen(a): float(b) for a, b in sampled_hung_re.findall(h_block)
                }
                for g in set(greedy_map) & set(hung_map):
                    if g not in VALID_GENS:
                        continue
                    gv, hv = greedy_map[g], hung_map[g]
                    if not _plausible_maha_pair(gv, hv):
                        continue
                    rows.append(
                        {
                            "Dataset": ds,
                            "Generator": g,
                            "Greedy_Mahalanobis": gv,
                            "Hungarian_Mahalanobis": hv,
                            "nb_src": nb.name,
                            "nb_kind": "sampled_table",
                        }
                    )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Dataset",
                "Generator",
                "Greedy_Mahalanobis",
                "Hungarian_Mahalanobis",
                "nb_src",
                "nb_kind",
            ]
        )
    df = pd.DataFrame(rows)
    # Prefer explicit print pairs, then HTML tables, then legacy text parsers
    kind_pref = {
        "print": 0,
        "sdv_html_table": 1,
        "split_html_table": 2,
        "table": 3,
        "sampled_table": 4,
    }
    df["_p"] = df["nb_kind"].map(kind_pref).fillna(9)
    df = df.sort_values("_p").drop_duplicates(["Dataset", "Generator"], keep="first")
    return df.drop(columns=["_p"])


def setup_style() -> None:
    fonts_dir = Path(__file__).resolve().parent / "fonts"
    if fonts_dir.is_dir():
        from matplotlib import font_manager as fm

        for ttf in fonts_dir.glob("*.ttf"):
            try:
                fm.fontManager.addfont(str(ttf))
            except (OSError, RuntimeError, ValueError):
                pass

    mpl.rcParams.update(
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
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def norm_gen(x) -> str:
    x = str(x).strip().replace("-", "_").replace(" ", "_")
    if x in {"WGAN_GP", "WGANGP", "WGAN"}:
        return "WGAN_GP"
    if x.upper().startswith("CTAB"):
        return "CTABGAN"
    if x in {"Forest_Diffusion"}:
        return "ForestDiffusion"
    return x


def norm_ds(x) -> str:
    x = str(x).strip()
    if x in DS_ALIAS:
        return DS_ALIAS[x]
    if len(x) > 3 and x[0].isdigit() and ". " in x:
        x = x.split(". ", 1)[1]
    return DS_ALIAS.get(x, x)


def pct_improve(numer: float, denom: float, eps: float = 1e-12) -> float:
    if denom is None or (isinstance(denom, float) and (np.isnan(denom) or abs(denom) < eps)):
        return np.nan
    return 100.0 * numer / denom


def save_fig(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    long = pd.read_csv(BASE / "supplementary/detailed_tables/all_mapping_units_long.csv")
    mc = pd.read_csv(ROOT / "Agreed analysis/mapping_cost_comparison.csv")
    meta = {
        "n_datasets_expected": 15,
        "n_generators_expected": 8,
        "cosine_greedy_available": False,
        "cosine_baseline": "Pairwise_Cosine (no assignment; curated)",
        "mahalanobis_greedy_definition": "One-to-One Greedy (notebook greedy Mahalanobis)",
    }

    # ---- Cosine: Pairwise vs Hungarian from curated mapping_cost ----
    cos = mc.copy()
    cos["Dataset"] = cos["Dataset_Short"].map(norm_ds)
    cos["Generator"] = cos["Generator"].map(norm_gen)
    cos = cos.rename(
        columns={
            "Pairwise_Cosine": "Baseline_Cosine",
            "Hungarian_Cosine": "Hungarian_Cosine",
        }
    )
    cos["Baseline_Cosine_Method"] = "Pairwise (no assignment)"
    cos = cos[
        ["Dataset", "Generator", "Baseline_Cosine", "Hungarian_Cosine", "Baseline_Cosine_Method", "Num_Matches"]
    ].drop_duplicates(["Dataset", "Generator"])

    # ---- Mahalanobis: One-to-One Greedy vs Hungarian ----
    # Primary: paired means already printed in generator notebook outputs
    nb_pairs = extract_notebook_maha_pairs()

    g = long[
        (long["Metric"] == "Mahalanobis Distance")
        & (long["Mapping_Method"] == "One-to-One Greedy")
    ].copy()
    g["Dataset"] = g["Dataset_Short"].map(norm_ds)
    g["Generator"] = g["Generator"].map(norm_gen)
    g = g.rename(columns={"Mean": "Greedy_Mahalanobis"})[
        ["Dataset", "Generator", "Greedy_Mahalanobis"]
    ]

    h_parts = []
    h1 = long[
        (long["Metric"] == "Mahalanobis Distance") & (long["Mapping_Method"] == "Hungarian")
    ].copy()
    h1["Dataset"] = h1["Dataset_Short"].map(norm_ds)
    h1["Generator"] = h1["Generator"].map(norm_gen)
    h1 = h1.rename(columns={"Mean": "Hungarian_Mahalanobis"})[
        ["Dataset", "Generator", "Hungarian_Mahalanobis"]
    ]
    h1["hung_src"] = "long_csv"
    h_parts.append(h1)

    h2 = mc.dropna(subset=["Hungarian_Mahalanobis"]).copy()
    h2["Dataset"] = h2["Dataset_Short"].map(norm_ds)
    h2["Generator"] = h2["Generator"].map(norm_gen)
    h2 = h2[["Dataset", "Generator", "Hungarian_Mahalanobis"]].copy()
    h2["hung_src"] = "mapping_cost"
    h_parts.append(h2)

    priv = ROOT / "excel sheets" / "2. Privacy"
    rows = []
    if priv.exists():
        for d in priv.iterdir():
            f = d / "Mahalanobis_Summary.xlsx"
            if not f.exists():
                continue
            df = pd.read_excel(f)
            dist_col = next(
                (c for c in df.columns if str(c).lower().replace(" ", "_") in {"mean_distance"}),
                None,
            )
            if dist_col is None:
                continue
            gen_col = "Generator" if "Generator" in df.columns else "Model"
            for _, r in df.iterrows():
                val = r[dist_col]
                if pd.isna(val):
                    continue
                ds_val = norm_ds(d.name)
                if "Dataset" in df.columns and pd.notna(r.get("Dataset")):
                    ds_val = norm_ds(r["Dataset"])
                rows.append(
                    {
                        "Dataset": ds_val,
                        "Generator": norm_gen(r[gen_col]),
                        "Hungarian_Mahalanobis": float(val),
                        "hung_src": "privacy_excel",
                    }
                )
    if rows:
        h_parts.append(pd.DataFrame(rows))

    # Generator-folder Hungarian Mahalanobis Summary workbooks
    FILE_DS_KEYS = [
        ("adult_census", "Adult"),
        ("adult", "Adult"),
        ("realestate", "RealEstate"),
        ("concrete", "Concrete"),
        ("magic", "MAGIC"),
        ("energyefficiency", "Energy"),
        ("energy_efficiency", "Energy"),
        ("airquality", "AirQuality"),
        ("winequality", "Wine"),
        ("wine", "Wine"),
        ("metro", "Metro"),
        ("heart", "CDC"),
        ("secondary_mushroom", "Mushroom"),
        ("mushroom", "Mushroom"),
        ("forestcover", "ForestCover"),
        ("forest_cover", "ForestCover"),
        ("online", "OnlineShop"),
        ("eshop", "OnlineShop"),
        ("four_models_ad.xlsx", "Alzheimers"),
        ("models_ad.xlsx", "Alzheimers"),
        ("_ad.xlsx", "Alzheimers"),
    ]
    rows_xlsx = []
    for f in (ROOT / "Generators").rglob("*Hungarian_Mahalanobis*.xlsx"):
        if ".bak_" in f.name:
            continue
        try:
            xl = pd.ExcelFile(f)
        except Exception:
            continue
        low = f.name.lower().replace(" ", "")
        ds_guess = None
        for key, ds in FILE_DS_KEYS:
            if key in low:
                ds_guess = ds
                break
        for s in xl.sheet_names:
            if "summ" not in s.lower():
                continue
            df = pd.read_excel(f, sheet_name=s)
            if "Mean_Distance" not in df.columns:
                continue
            model_col = (
                "Generator"
                if "Generator" in df.columns
                else ("Model" if "Model" in df.columns else None)
            )
            if model_col is None:
                continue
            for _, r in df.iterrows():
                if pd.isna(r["Mean_Distance"]):
                    continue
                ds = ds_guess
                if "Dataset" in df.columns and pd.notna(r.get("Dataset")):
                    ds = norm_ds(r["Dataset"])
                if ds is None:
                    continue
                rows_xlsx.append(
                    {
                        "Dataset": ds,
                        "Generator": norm_gen(r[model_col]),
                        "Hungarian_Mahalanobis": float(r["Mean_Distance"]),
                        "hung_src": "generator_excel",
                    }
                )
    if rows_xlsx:
        h_parts.append(pd.DataFrame(rows_xlsx))

    hung = pd.concat(h_parts, ignore_index=True) if h_parts else pd.DataFrame(
        columns=["Dataset", "Generator", "Hungarian_Mahalanobis", "hung_src"]
    )
    pref = {"long_csv": 0, "mapping_cost": 1, "generator_excel": 2, "privacy_excel": 3}
    hung["_p"] = hung["hung_src"].map(pref).fillna(9)
    hung = hung.sort_values("_p").drop_duplicates(["Dataset", "Generator"], keep="first")
    hung = hung.drop(columns=["_p"])

    # Full grid skeleton 15x8
    grid = pd.MultiIndex.from_product([DS_ORDER, GEN_ORDER], names=["Dataset", "Generator"]).to_frame(
        index=False
    )

    # Prefer notebook paired values (same run) for both Greedy and Hungarian
    maha = grid.merge(
        nb_pairs[
            [
                "Dataset",
                "Generator",
                "Greedy_Mahalanobis",
                "Hungarian_Mahalanobis",
                "nb_src",
                "nb_kind",
            ]
        ],
        on=["Dataset", "Generator"],
        how="left",
    )
    maha = maha.merge(g, on=["Dataset", "Generator"], how="left", suffixes=("", "_long"))
    maha = maha.merge(hung, on=["Dataset", "Generator"], how="left", suffixes=("", "_excel"))
    # Fill gaps from long/excel without overwriting notebook pairs
    maha["Greedy_Mahalanobis"] = maha["Greedy_Mahalanobis"].fillna(maha.get("Greedy_Mahalanobis_long"))
    maha["Hungarian_Mahalanobis"] = maha["Hungarian_Mahalanobis"].fillna(
        maha.get("Hungarian_Mahalanobis_excel")
    )
    maha["hung_src"] = np.where(
        maha["nb_src"].notna(),
        "notebook_pair:" + maha["nb_kind"].astype(str),
        maha["hung_src"],
    )
    drop_cols = [c for c in maha.columns if c.endswith("_long") or c.endswith("_excel")]
    maha = maha.drop(columns=drop_cols, errors="ignore")

    summary = grid.merge(cos, on=["Dataset", "Generator"], how="left").merge(
        maha[
            [
                "Dataset",
                "Generator",
                "Greedy_Mahalanobis",
                "Hungarian_Mahalanobis",
                "hung_src",
                "nb_src",
            ]
        ],
        on=["Dataset", "Generator"],
        how="left",
    )

    # Improvements (positive = Hungarian better)
    summary["Delta_Cosine"] = summary["Hungarian_Cosine"] - summary["Baseline_Cosine"]
    summary["Cosine_Improvement_Pct"] = [
        pct_improve(d, b) if pd.notna(d) and pd.notna(b) else np.nan
        for d, b in zip(summary["Delta_Cosine"], summary["Baseline_Cosine"])
    ]
    # Flag unstable % when |baseline| < 1e-3 (pairwise near zero)
    summary["Cosine_Pct_Unstable"] = summary["Baseline_Cosine"].abs() < 1e-3

    summary["Delta_Mahalanobis"] = summary["Greedy_Mahalanobis"] - summary["Hungarian_Mahalanobis"]
    summary["Mahalanobis_Improvement_Pct"] = [
        pct_improve(d, g) if pd.notna(d) and pd.notna(g) else np.nan
        for d, g in zip(summary["Delta_Mahalanobis"], summary["Greedy_Mahalanobis"])
    ]
    summary["Maha_Pct_Unstable"] = summary["Greedy_Mahalanobis"].abs() < 1e-12

    export = summary.copy()
    export["Greedy_Cosine"] = np.nan  # not available from notebooks
    export["Greedy_Cosine_Note"] = (
        "Not persisted in notebooks (many-to-one greedy cosine aggregates missing)"
    )
    export = export.rename(
        columns={
            "Baseline_Cosine": "Greedy_Cosine_Proxy_Pairwise",
            "Baseline_Cosine_Method": "Cosine_Baseline_Method",
        }
    )

    meta["n_cosine_pairs_pairwise_hungarian"] = int(
        summary.dropna(subset=["Baseline_Cosine", "Hungarian_Cosine"]).shape[0]
    )
    meta["n_maha_pairs_greedy_hungarian"] = int(
        summary.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"]).shape[0]
    )
    meta["datasets_present"] = sorted(summary["Dataset"].dropna().unique().tolist())
    meta["generators_present"] = sorted(summary["Generator"].dropna().unique().tolist())

    return summary, export, meta


def aggregate_stats(series: pd.Series, name: str) -> dict:
    s = series.dropna()
    n = len(s)
    pos = int((s > 0).sum())
    neg = int((s < 0).sum())
    tie = int((s == 0).sum())
    return {
        "Metric": name,
        "N": n,
        "Mean": float(s.mean()) if n else np.nan,
        "Median": float(s.median()) if n else np.nan,
        "Std": float(s.std(ddof=1)) if n > 1 else np.nan,
        "Min": float(s.min()) if n else np.nan,
        "Max": float(s.max()) if n else np.nan,
        "N_Positive": pos,
        "N_Negative": neg,
        "N_Ties": tie,
        "Pct_Hungarian_Improved": 100.0 * pos / n if n else np.nan,
        "Pct_Hungarian_Worse": 100.0 * neg / n if n else np.nan,
        "Pct_Ties": 100.0 * tie / n if n else np.nan,
    }


def paired_tests(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Cosine: pairwise vs hungarian
    sub = summary.dropna(subset=["Baseline_Cosine", "Hungarian_Cosine"])
    if len(sub) >= 5:
        w = stats.wilcoxon(sub["Hungarian_Cosine"], sub["Baseline_Cosine"], alternative="greater")
        # effect size: rank-biserial via matched pairs
        d = sub["Delta_Cosine"]
        r_rb = (d > 0).mean() - (d < 0).mean()
        rows.append(
            {
                "Comparison": "Hungarian Cosine vs Pairwise Cosine",
                "Test": "Wilcoxon signed-rank (paired, alternative greater)",
                "N": len(sub),
                "statistic": float(w.statistic),
                "p_value": float(w.pvalue),
                "Effect_RankBiserial": float(r_rb),
                "Note": "Pairwise is not Greedy mapping; test answers whether Hungarian cosine exceeds pairwise cosine.",
            }
        )
    sub = summary.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"])
    if len(sub) >= 5:
        # lower is better → Hungarian better if Hungarian < Greedy → Delta > 0
        w = stats.wilcoxon(
            sub["Greedy_Mahalanobis"], sub["Hungarian_Mahalanobis"], alternative="greater"
        )
        d = sub["Delta_Mahalanobis"]
        r_rb = (d > 0).mean() - (d < 0).mean()
        rows.append(
            {
                "Comparison": "Greedy (one-to-one) Mahalanobis vs Hungarian Mahalanobis",
                "Test": "Wilcoxon signed-rank (paired, alternative Greedy > Hungarian)",
                "N": len(sub),
                "statistic": float(w.statistic),
                "p_value": float(w.pvalue),
                "Effect_RankBiserial": float(r_rb),
                "Note": "Positive Delta_Mahalanobis = Hungarian improved (lower distance).",
            }
        )
    return pd.DataFrame(rows)


def fig1_cosine_improvement(summary: pd.DataFrame) -> None:
    """ΔCosine = Hungarian - Pairwise (labeled as such)."""
    sub = summary.dropna(subset=["Delta_Cosine"]).copy()
    sub["Dataset_Label"] = sub["Dataset"].map(DS_LABEL)
    order = [DS_LABEL[d] for d in DS_ORDER if d in set(sub["Dataset"])]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=sub,
        x="Dataset_Label",
        y="Delta_Cosine",
        order=order,
        color="#D9EAF7",
        fliersize=0,
        width=0.55,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="Dataset_Label",
        y="Delta_Cosine",
        order=order,
        color="#1F4E79",
        size=3.5,
        alpha=0.65,
        jitter=0.2,
        ax=ax,
    )
    ax.axhline(0, color="black", lw=1.0, ls="--")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Δ Cosine = Hungarian − Pairwise")
    ax.set_title("Figure 1. Hungarian improvement in cosine similarity\n(baseline: pairwise cosine; greedy cosine aggregates not persisted)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_1_cosine_improvement" / "Figure_1_Cosine_Improvement")


def fig2_maha_improvement(summary: pd.DataFrame) -> None:
    sub = summary.dropna(subset=["Delta_Mahalanobis"]).copy()
    sub["Dataset_Label"] = sub["Dataset"].map(DS_LABEL)
    order = [DS_LABEL[d] for d in DS_ORDER if d in set(sub["Dataset"])]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    sns.boxplot(
        data=sub,
        x="Dataset_Label",
        y="Delta_Mahalanobis",
        order=order,
        color="#E8F5E9",
        fliersize=0,
        width=0.55,
        ax=ax,
    )
    sns.stripplot(
        data=sub,
        x="Dataset_Label",
        y="Delta_Mahalanobis",
        order=order,
        color="#1B5E20",
        size=3.5,
        alpha=0.65,
        jitter=0.2,
        ax=ax,
    )
    ax.axhline(0, color="black", lw=1.0, ls="--")
    if sub["Delta_Mahalanobis"].abs().max() > 50 * max(sub["Delta_Mahalanobis"].abs().median(), 1e-6):
        ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Δ Mahalanobis = Greedy − Hungarian")
    ax.set_title(
        "Figure 2. Hungarian improvement in Mahalanobis distance\n(Greedy = one-to-one greedy matching from notebooks)"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_2_mahalanobis_improvement" / "Figure_2_Mahalanobis_Improvement")


def fig3_cosine_scatter(summary: pd.DataFrame) -> None:
    sub = summary.dropna(subset=["Baseline_Cosine", "Hungarian_Cosine"]).copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.scatter(
        sub["Baseline_Cosine"],
        sub["Hungarian_Cosine"],
        s=28,
        c="#1F4E79",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.plot([-0.05, 1.05], [-0.05, 1.05], "k--", lw=1, label="Y = X")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Pairwise Cosine Similarity")
    ax.set_ylabel("Hungarian Cosine Similarity")
    ax.set_title("Figure 3. Pairwise vs Hungarian cosine\n(above Y=X ⇒ Hungarian higher)")
    ax.legend(frameon=False, loc="lower right")
    n_above = int((sub["Hungarian_Cosine"] > sub["Baseline_Cosine"]).sum())
    ax.text(
        0.02,
        0.02,
        f"N = {len(sub)}; Hungarian > Pairwise in {n_above}/{len(sub)}",
        transform=ax.transAxes,
        va="bottom",
        fontsize=8,
    )
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_3_cosine_scatter" / "Figure_3_Cosine_Scatter")


def fig4_maha_scatter(summary: pd.DataFrame) -> None:
    sub = summary.dropna(subset=["Greedy_Mahalanobis", "Hungarian_Mahalanobis"]).copy()
    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.scatter(
        sub["Greedy_Mahalanobis"],
        sub["Hungarian_Mahalanobis"],
        s=28,
        c="#1B5E20",
        alpha=0.7,
        edgecolors="white",
        linewidths=0.3,
    )
    mx = max(sub["Greedy_Mahalanobis"].max(), sub["Hungarian_Mahalanobis"].max())
    # Use log scales if dynamic range large
    if mx > 100:
        ax.set_xscale("log")
        ax.set_yscale("log")
        lo = max(1e-3, min(sub["Greedy_Mahalanobis"].min(), sub["Hungarian_Mahalanobis"].min()) * 0.5)
        ax.plot([lo, mx * 1.2], [lo, mx * 1.2], "k--", lw=1, label="Y = X")
        ax.set_xlim(lo, mx * 1.2)
        ax.set_ylim(lo, mx * 1.2)
    else:
        lo = 0
        ax.plot([lo, mx * 1.05], [lo, mx * 1.05], "k--", lw=1, label="Y = X")
        ax.set_xlim(lo, mx * 1.05)
        ax.set_ylim(lo, mx * 1.05)
    n_below = int((sub["Hungarian_Mahalanobis"] < sub["Greedy_Mahalanobis"]).sum())
    ax.set_xlabel("Greedy Mahalanobis Distance (one-to-one)")
    ax.set_ylabel("Hungarian Mahalanobis Distance")
    ax.set_title("Figure 4. Greedy vs Hungarian Mahalanobis\n(below Y=X ⇒ Hungarian better)")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.02,
        f"N = {len(sub)}; Hungarian < Greedy in {n_below}/{len(sub)}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8,
    )
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_4_mahalanobis_scatter" / "Figure_4_Mahalanobis_Scatter")


def fig5_heatmaps(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2))

    # Cosine % — mark unstable near-zero denominators as hatched later; still show Δ instead of % if unstable
    # Use Delta_Cosine for panel a (more stable than % with near-zero pairwise)
    cos = summary.pivot(index="Dataset", columns="Generator", values="Delta_Cosine")
    cos = cos.reindex(index=DS_ORDER, columns=GEN_ORDER)
    cos.index = [DS_LABEL[i] for i in cos.index]
    cos.columns = [GEN_LABEL[c] for c in cos.columns]
    sns.heatmap(
        cos,
        ax=axes[0],
        cmap="RdBu",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Δ Cosine (Hung − Pairwise)"},
    )
    axes[0].set_title("(a) Cosine improvement (Δ)\nbaseline = Pairwise")
    axes[0].set_xlabel("Generator")
    axes[0].set_ylabel("Dataset")

    maha = summary.pivot(index="Dataset", columns="Generator", values="Mahalanobis_Improvement_Pct")
    maha = maha.reindex(index=DS_ORDER, columns=GEN_ORDER)
    maha.index = [DS_LABEL[i] for i in maha.index]
    maha.columns = [GEN_LABEL[c] for c in maha.columns]
    # robust color limits
    vals = maha.to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals):
        vmax = np.nanpercentile(np.abs(vals), 90)
        vmax = max(vmax, 5)
    else:
        vmax = 50
    sns.heatmap(
        maha,
        ax=axes[1],
        cmap="RdBu",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        annot=True,
        fmt=".1f",
        annot_kws={"size": 6},
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Mahalanobis improvement (%)"},
    )
    axes[1].set_title("(b) Mahalanobis improvement (%)\nGreedy (one-to-one) → Hungarian")
    axes[1].set_xlabel("Generator")
    axes[1].set_ylabel("Dataset")

    fig.suptitle("Figure 5. Hungarian improvement heatmaps", y=1.02, fontsize=12)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_5_improvement_heatmaps" / "Figure_5_Improvement_Heatmaps")


def fig6_distributions(summary: pd.DataFrame) -> None:
    cos = summary["Delta_Cosine"].dropna()
    maha = summary["Delta_Mahalanobis"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    for ax, data, title, ylabel, color in [
        (axes[0], cos, "(a) Δ Cosine\n(Hung − Pairwise)", "Δ Cosine", "#1F4E79"),
        (axes[1], maha, "(b) Δ Mahalanobis\n(Greedy − Hung)", "Δ Mahalanobis", "#1B5E20"),
    ]:
        sns.violinplot(y=data, ax=ax, color=color, alpha=0.25, inner=None, cut=0)
        sns.boxplot(y=data, ax=ax, width=0.25, color="white", showfliers=False)
        sns.stripplot(y=data, ax=ax, color=color, size=3, alpha=0.55, jitter=0.15)
        ax.axhline(0, color="black", ls="--", lw=1)
        if data.abs().max() > 50 * max(data.abs().median(), 1e-6):
            ax.set_yscale("symlog", linthresh=0.1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("")
        ax.set_xticks([])
        ax.text(
            0.98,
            0.98,
            f"N={len(data)}\nmean={data.mean():.3g}\nmedian={data.median():.3g}\n>{0}:{(data>0).mean()*100:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.9),
        )
    fig.suptitle("Figure 6. Overall distribution of Hungarian improvement", fontsize=12)
    fig.tight_layout()
    save_fig(fig, OUT_MAIN / "figure_6_improvement_distribution" / "Figure_6_Improvement_Distribution")


def supplementary(summary: pd.DataFrame) -> None:
    # dataset-level Δ maha bars
    ds_dir = OUT_SUPP / "dataset_level_improvement"
    ds_dir.mkdir(parents=True, exist_ok=True)
    for ds in DS_ORDER:
        sub = summary[summary["Dataset"] == ds].dropna(subset=["Delta_Mahalanobis"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 3.8))
        plot_df = sub.copy()
        plot_df["GenLabel"] = plot_df["Generator"].map(GEN_LABEL)
        plot_df = plot_df.set_index("Generator").reindex(GEN_ORDER).dropna(subset=["Delta_Mahalanobis"])
        colors = ["#2E7D32" if v >= 0 else "#C62828" for v in plot_df["Delta_Mahalanobis"]]
        ax.bar([GEN_LABEL[g] for g in plot_df.index], plot_df["Delta_Mahalanobis"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{DS_LABEL[ds]}: Δ Mahalanobis (Greedy − Hungarian)")
        ax.set_ylabel("Δ Mahalanobis")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        save_fig(fig, ds_dir / f"{ds}_delta_mahalanobis")

    # generator-level
    gen_dir = OUT_SUPP / "generator_level_improvement"
    gen_dir.mkdir(parents=True, exist_ok=True)
    for gen in GEN_ORDER:
        sub = summary[summary["Generator"] == gen].dropna(subset=["Delta_Mahalanobis"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 3.8))
        plot_df = sub.set_index("Dataset").reindex(DS_ORDER).dropna(subset=["Delta_Mahalanobis"])
        colors = ["#2E7D32" if v >= 0 else "#C62828" for v in plot_df["Delta_Mahalanobis"]]
        ax.bar([DS_LABEL[d] for d in plot_df.index], plot_df["Delta_Mahalanobis"], color=colors)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{GEN_LABEL[gen]}: Δ Mahalanobis across datasets")
        ax.set_ylabel("Δ Mahalanobis")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        save_fig(fig, gen_dir / f"{gen}_delta_mahalanobis")

    # three-method maha where available (supp)
    long = pd.read_csv(BASE / "supplementary/detailed_tables/all_mapping_units_long.csv")
    maha = long[long["Metric"] == "Mahalanobis Distance"].copy()
    maha["Dataset"] = maha["Dataset_Short"].map(norm_ds)
    maha["Generator"] = maha["Generator"].map(norm_gen)
    pivot = maha.pivot_table(index=["Dataset", "Generator"], columns="Mapping_Method", values="Mean")
    pivot.to_csv(OUT_SUPP / "detailed_tables" / "mahalanobis_methods_side_by_side.csv")


def write_readme(summary: pd.DataFrame, meta: dict, agg: pd.DataFrame, tests: pd.DataFrame) -> None:
    cos_n = meta["n_cosine_pairs_pairwise_hungarian"]
    maha_n = meta["n_maha_pairs_greedy_hungarian"]
    cos_stats = agg[agg["Metric"] == "Delta_Cosine_(Hung-Pairwise)"].iloc[0].to_dict() if len(agg) else {}
    maha_stats = agg[agg["Metric"] == "Delta_Mahalanobis_(Greedy-Hung)"].iloc[0].to_dict() if len(agg) else {}

    text = f"""# Mapping study graphs — Hungarian improvement over Greedy

**Extract-only analysis.** Mapping experiments were not rerun. Generators were not retrained.

## Research question

How much does **Hungarian matching** improve sample-to-sample matching compared with **Greedy matching**?

## Critical data availability

| Comparison | Available? | N (dataset×generator) | Source |
|---|---|---:|---|
| Greedy (many-to-one) Cosine vs Hungarian Cosine | **No** | 0 | Notebooks printed top-10 only; means not persisted |
| Pairwise Cosine vs Hungarian Cosine | Yes | {cos_n} | `Agreed analysis/mapping_cost_comparison.csv` |
| Greedy (one-to-one) Mahalanobis vs Hungarian | Yes | {maha_n} | Notebook outputs + Hungarian Excel / curated CSV |

**Cosine figures therefore use Pairwise Cosine as the non-Hungarian baseline** and are labeled accordingly (not “Greedy”).
**Mahalanobis figures use One-to-One Greedy**, which is the greedy Mahalanobis procedure implemented in the notebooks.

## Improvement definitions

```text
ΔCosine = Hungarian_Cosine − Pairwise_Cosine     # positive ⇒ Hungarian higher similarity
ΔMahalanobis = Greedy_Maha − Hungarian_Maha      # positive ⇒ Hungarian lower distance
```

Percentage improvements:

```text
Cosine % = (ΔCosine / Pairwise) × 100     # unstable when |Pairwise| ≈ 0 (flagged)
Mahalanobis % = (ΔMahalanobis / Greedy) × 100
```

Near-zero pairwise denominators are flagged in `hungarian_vs_greedy_summary.csv` (`Cosine_Pct_Unstable`).

## Overall results (from extracted pairs)

### Cosine (Pairwise → Hungarian)
- N = {cos_stats.get('N', '—')}
- Mean ΔCosine = {cos_stats.get('Mean', float('nan')):.4f}
- Median ΔCosine = {cos_stats.get('Median', float('nan')):.4f}
- % cases Hungarian higher = {cos_stats.get('Pct_Hungarian_Improved', float('nan')):.1f}%

### Mahalanobis (Greedy one-to-one → Hungarian)
- N = {maha_stats.get('N', '—')}
- Mean ΔMahalanobis = {maha_stats.get('Mean', float('nan')):.4f}
- Median ΔMahalanobis = {maha_stats.get('Median', float('nan')):.4f}
- % cases Hungarian improved (lower distance) = {maha_stats.get('Pct_Hungarian_Improved', float('nan')):.1f}%

## Main paper figures

| Figure | Path |
|---|---|
| Figure 1 | `main_paper/figure_1_cosine_improvement/` |
| Figure 2 | `main_paper/figure_2_mahalanobis_improvement/` |
| Figure 3 | `main_paper/figure_3_cosine_scatter/` |
| Figure 4 | `main_paper/figure_4_mahalanobis_scatter/` |
| Figure 5 | `main_paper/figure_5_improvement_heatmaps/` |
| Figure 6 | `main_paper/figure_6_improvement_distribution/` |

Each folder: PNG (300 DPI) + SVG.

## Summary tables

- `extracted_results/hungarian_vs_greedy_summary.csv` — unit-level (up to 15×8)
- `extracted_results/hungarian_vs_greedy_aggregate_stats.csv`
- `extracted_results/hungarian_vs_greedy_paired_tests.csv`
- `extracted_results/validation_checklist.json`

## Reproduce (figures only)

```bash
cd "SYNTH/mapping study graphs"
python create_hungarian_vs_greedy_figures.py
```
"""
    (BASE / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    for p in [
        OUT_EXTRACT,
        OUT_MAIN / "figure_1_cosine_improvement",
        OUT_MAIN / "figure_2_mahalanobis_improvement",
        OUT_MAIN / "figure_3_cosine_scatter",
        OUT_MAIN / "figure_4_mahalanobis_scatter",
        OUT_MAIN / "figure_5_improvement_heatmaps",
        OUT_MAIN / "figure_6_improvement_distribution",
        OUT_SUPP / "detailed_tables",
        OUT_SUPP / "dataset_level_improvement",
        OUT_SUPP / "generator_level_improvement",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    summary, export, meta = load_tables()

    out_cols = [
        "Dataset",
        "Generator",
        "Greedy_Cosine",
        "Greedy_Cosine_Note",
        "Greedy_Cosine_Proxy_Pairwise",
        "Cosine_Baseline_Method",
        "Hungarian_Cosine",
        "Delta_Cosine",
        "Cosine_Improvement_Pct",
        "Cosine_Pct_Unstable",
        "Greedy_Mahalanobis",
        "Hungarian_Mahalanobis",
        "Delta_Mahalanobis",
        "Mahalanobis_Improvement_Pct",
        "Maha_Pct_Unstable",
        "hung_src",
        "nb_src",
        "Num_Matches",
    ]
    for c in out_cols:
        if c not in export.columns:
            export[c] = np.nan if c != "Greedy_Cosine_Note" else ""
    export[out_cols].to_csv(OUT_EXTRACT / "hungarian_vs_greedy_summary.csv", index=False)

    agg = pd.DataFrame(
        [
            aggregate_stats(summary["Delta_Cosine"], "Delta_Cosine_(Hung-Pairwise)"),
            aggregate_stats(summary["Delta_Mahalanobis"], "Delta_Mahalanobis_(Greedy-Hung)"),
        ]
    )
    agg.to_csv(OUT_EXTRACT / "hungarian_vs_greedy_aggregate_stats.csv", index=False)

    tests = paired_tests(summary)
    tests.to_csv(OUT_EXTRACT / "hungarian_vs_greedy_paired_tests.csv", index=False)

    validation = {
        "datasets_identified": len(meta["datasets_present"]),
        "generators_identified": len(meta["generators_present"]),
        "expected_120": 15 * 8,
        "cosine_pairwise_hungarian_pairs": meta["n_cosine_pairs_pairwise_hungarian"],
        "mahalanobis_greedy_hungarian_pairs": meta["n_maha_pairs_greedy_hungarian"],
        "greedy_cosine_extracted": False,
        "hungarian_cosine_extracted": bool(summary["Hungarian_Cosine"].notna().any()),
        "greedy_mahalanobis_extracted": bool(summary["Greedy_Mahalanobis"].notna().any()),
        "hungarian_mahalanobis_extracted": bool(summary["Hungarian_Mahalanobis"].notna().any()),
        "delta_cosine_definition": "Hungarian - Pairwise (Pairwise used because Greedy cosine missing)",
        "delta_mahalanobis_definition": "Greedy - Hungarian (positive = Hungarian improvement)",
        "mapping_rerun": False,
        "datasets": meta["datasets_present"],
        "generators": meta["generators_present"],
    }
    (OUT_EXTRACT / "validation_checklist.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    fig1_cosine_improvement(summary)
    fig2_maha_improvement(summary)
    fig3_cosine_scatter(summary)
    fig4_maha_scatter(summary)
    fig5_heatmaps(summary)
    fig6_distributions(summary)
    supplementary(summary)
    write_readme(summary, meta, agg, tests)

    print("=== VALIDATION ===")
    print(json.dumps(validation, indent=2))
    print("\n=== AGGREGATE ===")
    print(agg.to_string(index=False))
    print("\n=== TESTS ===")
    print(tests.to_string(index=False) if len(tests) else "none")
    print(f"\nWrote figures under {OUT_MAIN}")
    print(f"Summary: {OUT_EXTRACT / 'hungarian_vs_greedy_summary.csv'}")


if __name__ == "__main__":
    main()
