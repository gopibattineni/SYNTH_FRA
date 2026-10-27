#!/usr/bin/env python3
"""Populate figures/main and figures/supplementary with curated manuscript packs."""
from __future__ import annotations

import shutil
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
FIG = ANALYSIS / "figures"
MAIN = FIG / "main"
SUPP = FIG / "supplementary"

# (src relative to figures/, dest name under main/)
MAIN_PACK = [
    ("workflow/experimental_workflow.png", "Fig_workflow_experimental_pipeline.png"),
    ("workflow/experimental_design.png", "Fig_workflow_experimental_design.png"),
    ("classification/accuracy_gap_critical_difference_diagram.png", "Fig_cls_CD_accuracy_gap.png"),
    ("classification/precision_gap_critical_difference_diagram.png", "Fig_cls_CD_precision_gap.png"),
    ("classification/recall_gap_critical_difference_diagram.png", "Fig_cls_CD_recall_gap.png"),
    ("classification/f1_gap_critical_difference_diagram.png", "Fig_cls_CD_f1_gap.png"),
    ("classification/accuracy_gap_rank_table.png", "Fig_cls_rank_table_accuracy_gap.png"),
    ("classification/generator_robustness_average_rank.png", "Fig_cls_generator_robustness.png"),
    ("regression/r2_gap_critical_difference_diagram.png", "Fig_reg_CD_r2_gap.png"),
    ("regression/rmse_increase_critical_difference_diagram.png", "Fig_reg_CD_rmse_increase.png"),
    ("regression/mae_increase_critical_difference_diagram.png", "Fig_reg_CD_mae_increase.png"),
    ("regression/r2_gap_rank_table.png", "Fig_reg_rank_table_r2_gap.png"),
    ("regression/generator_robustness_average_rank.png", "Fig_reg_generator_robustness.png"),
    ("tradeoffs/classification_Fig1_Fidelity_vs_Utility.png", "Fig_cls_tradeoff_Fidelity_vs_Utility.png"),
    ("tradeoffs/classification_Fig2_Utility_vs_Privacy.png", "Fig_cls_tradeoff_Utility_vs_Privacy.png"),
    ("tradeoffs/classification_Fig3_Fidelity_vs_Privacy.png", "Fig_cls_tradeoff_Fidelity_vs_Privacy.png"),
    ("tradeoffs/regression_Fig1_Fidelity_vs_Utility.png", "Fig_reg_tradeoff_Fidelity_vs_Utility.png"),
    ("tradeoffs/regression_Fig2_Utility_vs_Privacy.png", "Fig_reg_tradeoff_Utility_vs_Privacy.png"),
    ("tradeoffs/regression_Fig3_Fidelity_vs_Privacy.png", "Fig_reg_tradeoff_Fidelity_vs_Privacy.png"),
    ("seed_stability/classification_accuracy_gap_seed_violin.png", "Fig_seed_stability_cls_accuracy_gap.png"),
    ("seed_stability/regression_r2_gap_seed_violin.png", "Fig_seed_stability_reg_r2_gap.png"),
]

SUPP_PACK = [
    ("classification/accuracy_gap_by_dataset_heatmap.png", "Supp_cls_heatmap_accuracy_gap.png"),
    ("classification/quality_score_by_dataset_heatmap.png", "Supp_cls_heatmap_quality_score.png"),
    ("classification/mia_auc_by_dataset_heatmap.png", "Supp_cls_heatmap_mia_auc.png"),
    ("regression/r2_gap_by_dataset_heatmap.png", "Supp_reg_heatmap_r2_gap.png"),
    ("regression/quality_score_by_dataset_heatmap.png", "Supp_reg_heatmap_quality_score.png"),
    ("regression/mia_auc_by_dataset_heatmap.png", "Supp_reg_heatmap_mia_auc.png"),
    ("seed_stability/classification_precision_gap_seed_violin.png", "Supp_seed_cls_precision_gap.png"),
    ("seed_stability/classification_recall_gap_seed_violin.png", "Supp_seed_cls_recall_gap.png"),
    ("seed_stability/classification_f1_gap_seed_violin.png", "Supp_seed_cls_f1_gap.png"),
    ("seed_stability/classification_quality_score_seed_violin.png", "Supp_seed_cls_quality_score.png"),
    ("seed_stability/classification_mia_auc_seed_violin.png", "Supp_seed_cls_mia_auc.png"),
    ("seed_stability/regression_rmse_increase_seed_violin.png", "Supp_seed_reg_rmse_increase.png"),
    ("seed_stability/regression_mae_increase_seed_violin.png", "Supp_seed_reg_mae_increase.png"),
    ("seed_stability/regression_quality_score_seed_violin.png", "Supp_seed_reg_quality_score.png"),
    ("seed_stability/regression_mia_auc_seed_violin.png", "Supp_seed_reg_mia_auc.png"),
]


def _copy_pack(items: list[tuple[str, str]], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for src_rel, dest_name in items:
        src = FIG / src_rel
        if not src.exists():
            print(f"MISSING: {src}")
            continue
        shutil.copy2(src, dest_dir / dest_name)
        # also copy pdf/svg siblings when present
        for ext in (".pdf", ".svg"):
            sib = src.with_suffix(ext)
            if sib.exists():
                shutil.copy2(sib, dest_dir / Path(dest_name).with_suffix(ext).name)
        n += 1
    return n


def main() -> None:
    n_main = _copy_pack(MAIN_PACK, MAIN)
    n_supp = _copy_pack(SUPP_PACK, SUPP)
    # index files
    (MAIN / "MANIFEST.md").write_text(
        "# Main manuscript figures\n\n"
        + "\n".join(f"- `{d}` ← `figures/{s}`" for s, d in MAIN_PACK)
        + "\n",
        encoding="utf-8",
    )
    (SUPP / "MANIFEST.md").write_text(
        "# Supplementary figures\n\n"
        + "\n".join(f"- `{d}` ← `figures/{s}`" for s, d in SUPP_PACK)
        + "\n",
        encoding="utf-8",
    )
    print(f"Packed main={n_main} supplementary={n_supp}")


if __name__ == "__main__":
    main()
