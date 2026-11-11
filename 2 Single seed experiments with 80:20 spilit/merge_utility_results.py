"""Merge TRTR/TSTR Excel results from main and diffusion folders into utility results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "utility results"
MAIN_GENS = ["CTGAN", "CopulaGAN", "TVAE", "GaussianCopula", "WGAN_GP", "CTABGAN"]
DIFF_GENS = ["TabDDPM", "ForestDiffusion"]
ALL_GENS = MAIN_GENS + DIFF_GENS
COMMON_SHEETS = ["TRTR_Results", "All_Comparisons", "Summary"]

RESULT_PATTERNS = [
    "TRTR_TSTR_results.xlsx",
    "TRTR_TSTR_results_regression.xlsx",
    "TRTR_TSTR_results_air_quality.xlsx",
    "TRTR_TSTR_results_energy_efficiency.xlsx",
    "TRTR_TSTR_results_real_estate.xlsx",
    "TRTR_TSTR_results_concrete.xlsx",
]


def find_result_file(folder: Path) -> Path | None:
    if not folder.exists():
        return None
    for name in RESULT_PATTERNS:
        path = folder / name
        if path.exists():
            return path
    matches = sorted(folder.glob("TRTR_TSTR*.xlsx"))
    return matches[0] if matches else None


def dataset_folders(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        [d for d in root.iterdir() if d.is_dir() and d.name and d.name[0].isdigit()],
        key=lambda x: int(x.name.split(".")[0].strip()),
    )


def read_sheet(path: Path | None, sheet: str) -> pd.DataFrame | None:
    if path is None:
        return None
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return None


def concat_frames(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    valid = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    return pd.concat(valid, ignore_index=True)


def merge_dataset(
    dataset_name: str,
    main_path: Path | None,
    diff_path: Path | None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    sheets: dict[str, pd.DataFrame] = {}
    log = {
        "dataset": dataset_name,
        "main_file": str(main_path) if main_path else None,
        "diffusion_file": str(diff_path) if diff_path else None,
        "generators_included": [],
        "notes": [],
    }

    for sheet in COMMON_SHEETS:
        main_df = read_sheet(main_path, sheet)
        diff_df = read_sheet(diff_path, sheet)

        if sheet == "TRTR_Results":
            sheets[sheet] = main_df if main_df is not None else diff_df
        elif sheet in ("All_Comparisons", "Summary"):
            parts = []
            if main_df is not None:
                if "Synthetic_Model" in main_df.columns:
                    parts.append(main_df[main_df["Synthetic_Model"].isin(MAIN_GENS)])
                else:
                    parts.append(main_df)
            if diff_df is not None:
                if "Synthetic_Model" in diff_df.columns:
                    parts.append(diff_df[diff_df["Synthetic_Model"].isin(DIFF_GENS)])
                else:
                    parts.append(diff_df)
            sheets[sheet] = concat_frames(parts)
        else:
            sheets[sheet] = main_df if main_df is not None else diff_df

    quality_parts = []
    for path in (main_path, diff_path):
        quality_df = read_sheet(path, "Quality_Metrics")
        if quality_df is not None and not quality_df.empty:
            if "Generator" in quality_df.columns:
                allowed = set(ALL_GENS)
                quality_df = quality_df[quality_df["Generator"].isin(allowed)]
            quality_parts.append(quality_df)

    if quality_parts:
        quality_merged = concat_frames(quality_parts)
        if "Generator" in quality_merged.columns:
            quality_merged = quality_merged.drop_duplicates(
                subset=["Generator"], keep="first"
            ).reset_index(drop=True)
        sheets["Quality_Metrics"] = quality_merged

    for generator in ALL_GENS:
        source = main_path if generator in MAIN_GENS else diff_path
        if source is None:
            log["notes"].append(f"Missing source for {generator}")
            continue

        generator_df = read_sheet(source, generator)
        if generator_df is not None and not generator_df.empty:
            sheets[generator] = generator_df
            log["generators_included"].append(generator)
        else:
            log["notes"].append(f"Sheet {generator} not found in {source.name}")

    return sheets, log


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    logs: list[dict] = []

    for folder in dataset_folders(BASE):
        dataset_name = folder.name
        main_path = find_result_file(folder)
        diff_path = find_result_file(BASE / "diffusion_dataleak" / dataset_name)

        if main_path is None and diff_path is None:
            logs.append({"dataset": dataset_name, "status": "SKIPPED - no result files"})
            continue

        sheets, log = merge_dataset(dataset_name, main_path, diff_path)

        out_dir = OUT / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "TRTR_TSTR_results_merged.xlsx"

        write_order: list[str] = []
        if (
            "Quality_Metrics" in sheets
            and sheets["Quality_Metrics"] is not None
            and not sheets["Quality_Metrics"].empty
        ):
            write_order.append("Quality_Metrics")
        write_order.extend(COMMON_SHEETS)
        write_order.extend(generator for generator in ALL_GENS if generator in sheets)

        with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
            for sheet_name in write_order:
                frame = sheets.get(sheet_name)
                if frame is not None and not frame.empty:
                    frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)

        log["status"] = "OK"
        log["output"] = str(out_file)
        log["generator_count"] = len(log["generators_included"])
        logs.append(log)
        print(
            f"{dataset_name}: {log['generator_count']}/8 generators -> {out_file.name}"
        )

    log_path = OUT / "merge_log.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"\nSaved merge log to {log_path}")
    print(f"Total datasets processed: {len(logs)}")


if __name__ == "__main__":
    main()
