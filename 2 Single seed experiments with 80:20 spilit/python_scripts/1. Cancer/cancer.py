#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Converted from cancer.ipynb — notebook code cells below, order preserved.
# Runnable as a script from any working directory.

# ========================================================================
# RUNTIME BOOTSTRAP (not in notebook; enables plain-Python execution)
# ========================================================================
import argparse
import os
import subprocess
import sys
from pathlib import Path


def _parse_bootstrap_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-run synthetic data quality pipeline (cancer.ipynb)."
    )
    parser.add_argument(
        "--uci-id",
        type=int,
        default=17,
        help="UCI ML Repository dataset ID (default: 17 = Breast Cancer).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic rows to generate per generator.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction for train/test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the single-run synthesis block.",
    )
    parser.add_argument(
        "--train-csv",
        default="breast_cancer_train.csv",
        help="Filename for the CTAB-GAN training CSV written in the notebook dir.",
    )
    parser.add_argument(
        "--output-file",
        default="TRTR_TSTR_results.xlsx",
        help="Excel file for TRTR/TSTR evaluation results.",
    )
    parser.add_argument(
        "--notebook-dir",
        type=Path,
        default=None,
        help="Dataset working directory (must contain or clone CTAB-GAN-Plus).",
    )
    return parser.parse_args()


_BOOT = _parse_bootstrap_args()
UCI_ID = _BOOT.uci_id
N_SAMPLES = _BOOT.n_samples
TEST_SIZE = _BOOT.test_size
SEED = _BOOT.seed
TRAIN_CSV = _BOOT.train_csv
OUTPUT_FILE = _BOOT.output_file

SCRIPT_DIR = Path(__file__).resolve().parent
NOTEBOOK_DIR = (
    _BOOT.notebook_dir if _BOOT.notebook_dir is not None
    else SCRIPT_DIR.parent.parent / "1. Cancer"
).resolve()
os.chdir(NOTEBOOK_DIR)

CTAB_REPO = NOTEBOOK_DIR / "CTAB-GAN-Plus"
if not CTAB_REPO.is_dir():
    subprocess.run(
        ["git", "clone", "https://github.com/Team-TUD/CTAB-GAN-Plus"],
        cwd=NOTEBOOK_DIR,
        check=True,
    )

def display(obj):
    """Jupyter-compatible display shim for terminal execution."""
    try:
        from IPython.display import display as _ipython_display
        _ipython_display(obj)
    except ImportError:
        if hasattr(obj, "to_string"):
            print(obj.to_string())
        else:
            print(obj)

# END RUNTIME BOOTSTRAP


# ========================================================================
# Cell 0
# ========================================================================
# Notebook shell command (git clone) handled in RUNTIME BOOTSTRAP above
import sys
sys.path.append('./CTAB-GAN-Plus')
from model.ctabgan import CTABGAN


# ========================================================================
# Cell 1
# ========================================================================
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sdv.metadata import SingleTableMetadata
from sdv.single_table import (
    CTGANSynthesizer,
    CopulaGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer
)

from sdv.evaluation.single_table import evaluate_quality

from model.ctabgan import CTABGAN

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------
breast_cancer = fetch_ucirepo(id=UCI_ID)

X = breast_cancer.data.features
y = breast_cancer.data.targets

cancer_data = pd.concat([X, y], axis=1)

target_col = cancer_data.columns[-1]

# ----------------------------------------------------
# Metadata
# ----------------------------------------------------
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(cancer_data)

# ----------------------------------------------------
# Experiment Settings (N_SAMPLES, TEST_SIZE, SEED from bootstrap CLI)
# ----------------------------------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------
# Storage Containers
# ----------------------------------------------------
scores = {}

synthetic_datasets = {}

quality_results = []


# ========================================================================
# Cell 2
# ========================================================================
# ---------------------------------------------------
# SINGLE RUN
# ---------------------------------------------------

seed = SEED

print("\n================ SINGLE RUN ================")

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------
# TRAIN / TEST SPLIT (NO LEAKAGE)
# ---------------------------------------------------

train_real, test_real = train_test_split(
    cancer_data,
    test_size=TEST_SIZE,
    stratify=cancer_data[target_col],
    random_state=seed
)

train_metadata = SingleTableMetadata()
train_metadata.detect_from_dataframe(train_real)

# ---------------------------------------------------
# CTABGAN
# ---------------------------------------------------

try:

    data_path = TRAIN_CSV
    train_real.to_csv(data_path, index=False)

    ctabgan = CTABGAN(
        raw_csv_path=data_path,
        categorical_columns=[target_col],
        log_columns=[],
        mixed_columns={},
        integer_columns=[],
        problem_type={"Classification": target_col}
    )

    ctabgan.fit()

    synthetic_ctabgan = ctabgan.data_prep.inverse_prep(
        ctabgan.synthesizer.sample(N_SAMPLES)
    )

    synthetic_datasets["CTABGAN"] = synthetic_ctabgan.copy()

    quality = evaluate_quality(
        real_data=train_real,
        synthetic_data=synthetic_ctabgan,
        metadata=train_metadata
    )

    score = quality.get_score()

    scores["CTABGAN"] = score

    print("CTABGAN:", round(score, 4))

except Exception as e:
    print("CTABGAN Failed:", e)


# ========================================================================
# Cell 3
# ========================================================================
# WGAN-GP

try:

    import traceback

    data_wgan = train_real.copy()

    encoder = LabelEncoder()
    data_wgan[target_col] = encoder.fit_transform(data_wgan[target_col])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_wgan)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    real_tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32
    )

    batch_size = 64
    latent_dim = 64
    data_dim = real_tensor.shape[1]

    loader = torch.utils.data.DataLoader(
        real_tensor,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2),

                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, data_dim)
            )

        def forward(self, z):
            return self.model(z)

    class Critic(nn.Module):
        def __init__(self):
            super().__init__()

            self.model = nn.Sequential(
                nn.Linear(data_dim, 256),
                nn.LeakyReLU(0.2),

                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),

                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.model(x)

    generator = Generator().to(device)
    critic = Critic().to(device)

    optimizer_G = optim.Adam(
        generator.parameters(),
        lr=0.0001,
        betas=(0.5, 0.9)
    )

    optimizer_C = optim.Adam(
        critic.parameters(),
        lr=0.0001,
        betas=(0.5, 0.9)
    )

    def gradient_penalty(critic, real_samples, fake_samples):

        alpha = torch.rand(real_samples.size(0), 1, device=device)
        alpha = alpha.expand_as(real_samples)

        interpolates = (
            alpha * real_samples +
            (1 - alpha) * fake_samples
        ).requires_grad_(True)

        critic_interpolates = critic(interpolates)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    for epoch in range(100):

        for real_batch in loader:

            real_batch = real_batch.to(device)

            for _ in range(5):

                z = torch.randn(
                    real_batch.size(0),
                    latent_dim,
                    device=device
                )

                fake_batch = generator(z).detach()

                critic_real = critic(real_batch).mean()
                critic_fake = critic(fake_batch).mean()

                gp = gradient_penalty(
                    critic,
                    real_batch,
                    fake_batch
                )

                critic_loss = (
                    critic_fake
                    - critic_real
                    + 10 * gp
                )

                optimizer_C.zero_grad()
                critic_loss.backward()
                optimizer_C.step()

            z = torch.randn(
                real_batch.size(0),
                latent_dim,
                device=device
            )

            fake = generator(z)

            generator_loss = -critic(fake).mean()

            optimizer_G.zero_grad()
            generator_loss.backward()
            optimizer_G.step()

    generator.eval()

    with torch.no_grad():

        z = torch.randn(
            N_SAMPLES,
            latent_dim,
            device=device
        )

        synthetic_scaled = generator(z).cpu().numpy()

    synthetic = scaler.inverse_transform(synthetic_scaled)

    synthetic_wgan = pd.DataFrame(
        synthetic,
        columns=data_wgan.columns
    )

    synthetic_wgan[target_col] = (
        synthetic_wgan[target_col]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    synthetic_wgan[target_col] = encoder.inverse_transform(
        synthetic_wgan[target_col]
    )

    synthetic_datasets["WGAN_GP"] = synthetic_wgan.copy()

    quality = evaluate_quality(
        real_data=train_real,
        synthetic_data=synthetic_wgan,
        metadata=train_metadata
    )

    scores["WGAN_GP"] = quality.get_score()

    print("WGAN_GP:", round(scores["WGAN_GP"], 4))

    del generator
    del critic
    del real_tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

except Exception as e:

    print("WGAN_GP Failed:")
    traceback.print_exc()


# ========================================================================
# Cell 4
# ========================================================================
# SDV MODELS

sdv_models = {
    "CTGAN": CTGANSynthesizer(metadata=train_metadata),
    "CopulaGAN": CopulaGANSynthesizer(metadata=train_metadata),
    "TVAE": TVAESynthesizer(metadata=train_metadata),
    "GaussianCopula": GaussianCopulaSynthesizer(metadata=train_metadata)
}

for model_name, model in sdv_models.items():

    try:

        model.fit(train_real)

        synthetic_data = model.sample(N_SAMPLES)

        synthetic_datasets[model_name] = synthetic_data.copy()

        quality = evaluate_quality(
            real_data=train_real,
            synthetic_data=synthetic_data,
            metadata=train_metadata
        )

        scores[model_name] = quality.get_score()

        print(
            f"{model_name}: {round(scores[model_name], 4)}"
        )

    except Exception as e:

        print(
            f"{model_name} Failed: {e}"
        )


# ========================================================================
# Cell 5
# ========================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


models = {

    'LogReg': LogisticRegression(max_iter=5000, solver='liblinear', random_state=42),
    'SVM-RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN': KNeighborsClassifier(),
    'NaiveBayes': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'ExtraTrees':  ExtraTreesClassifier(random_state=42),
    'GradientBoost': GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "MLP": MLPClassifier(max_iter=2000, random_state=42),
}


# ========================================================================
# Cell 6
# ========================================================================
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import pandas as pd


# ========================================================================
# Cell 7
# ========================================================================
# TRTR (Train Real, Test Real)

print("--- Starting TRTR Evaluation (Train Real, Test Real) ---")

trtr_results = []

SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

for model_name, model in models.items():

    accuracy_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []

    print(f"Running {model_name}...")

    for seed in SEEDS:

        X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=seed
        )

        clf = clone(model)

        if hasattr(clf, "random_state"):
            clf.set_params(random_state=seed)

        clf.fit(X_train_real, y_train_real)

        y_pred = clf.predict(X_test_real)

        accuracy_scores.append(
            accuracy_score(y_test_real, y_pred)
        )

        f1_scores.append(
            f1_score(
                y_test_real,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )

        precision_scores.append(
            precision_score(
                y_test_real,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )

        recall_scores.append(
            recall_score(
                y_test_real,
                y_pred,
                average="weighted",
                zero_division=0
            )
        )

    acc_mean = np.mean(accuracy_scores)
    acc_std = np.std(accuracy_scores)

    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)

    prec_mean = np.mean(precision_scores)
    prec_std = np.std(precision_scores)

    rec_mean = np.mean(recall_scores)
    rec_std = np.std(recall_scores)

    trtr_results.append({
        "Model": model_name,
        "Accuracy (Mean±Std)_TRTR": f"{acc_mean:.4f} ± {acc_std:.4f}",
        "F1 (Mean±Std)_TRTR": f"{f1_mean:.4f} ± {f1_std:.4f}",
        "Precision (Mean±Std)_TRTR": f"{prec_mean:.4f} ± {prec_std:.4f}",
        "Recall (Mean±Std)_TRTR": f"{rec_mean:.4f} ± {rec_std:.4f}"
    })

trtr_results_df = pd.DataFrame(trtr_results)

display(
    trtr_results_df[
        [
            "Model",
            "Accuracy (Mean±Std)_TRTR",
            "F1 (Mean±Std)_TRTR",
            "Precision (Mean±Std)_TRTR",
            "Recall (Mean±Std)_TRTR"
        ]
    ]
)


# ========================================================================
# Cell 8
# ========================================================================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import clone
import pandas as pd
import numpy as np

def evaluate_models(
    train_df,
    test_df,
    label_col,
    models,
    test_size=0.2,
    seeds=[42,43,44,45,46,47,48,49,50,51]
):

    results = []

    for name, model in models.items():

        accuracy_scores = []
        f1_scores = []
        precision_scores = []
        recall_scores = []

        for seed in seeds:

            X_train = train_df.drop(columns=[label_col])
            y_train = train_df[label_col]

            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                test_size=test_size,
                random_state=seed,
                stratify=y_train
            )

            X_test = test_df.drop(columns=[label_col])
            y_test = test_df[label_col]

            _, X_test, _, y_test = train_test_split(
                X_test,
                y_test,
                test_size=test_size,
                random_state=seed,
                stratify=y_test
            )

            scaler = StandardScaler().fit(X_train)

            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            clf = clone(model)

            if hasattr(clf, "random_state"):
                clf.set_params(random_state=seed)

            clf.fit(X_train_s, y_train)

            y_pred = clf.predict(X_test_s)

            accuracy_scores.append(
                accuracy_score(y_test, y_pred)
            )

            f1_scores.append(
                f1_score(
                    y_test,
                    y_pred,
                    pos_label="M",
                    average="binary",
                    zero_division=0
                )
            )

            precision_scores.append(
                precision_score(
                    y_test,
                    y_pred,
                    pos_label="M",
                    average="binary",
                    zero_division=0
                )
            )

            recall_scores.append(
                recall_score(
                    y_test,
                    y_pred,
                    pos_label="M",
                    average="binary",
                    zero_division=0
                )
            )

        results.append({
            "Model": name,

            "Accuracy Mean": np.mean(accuracy_scores),
            "Accuracy Std": np.std(accuracy_scores),

            "F1 Mean": np.mean(f1_scores),
            "F1 Std": np.std(f1_scores),

            "Precision Mean": np.mean(precision_scores),
            "Precision Std": np.std(precision_scores),

            "Recall Mean": np.mean(recall_scores),
            "Recall Std": np.std(recall_scores),

            "Accuracy (Mean±Std)": f"{np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}",
            "F1 (Mean±Std)": f"{np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}",
            "Precision (Mean±Std)": f"{np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}",
            "Recall (Mean±Std)": f"{np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}"
        })

    return pd.DataFrame(results).sort_values(
        by="Accuracy Mean",
        ascending=False
    )


# ========================================================================
# Cell 9
# ========================================================================
import pandas as pd

label_col = "Diagnosis"

model_order = [
    "CTGAN",
    "CopulaGAN",
    "TVAE",
    "GaussianCopula",
    "WGAN_GP",
    "CTABGAN"
]

seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

print("TRTR (Train Real, Test Real)")

trtr_results = evaluate_models(
    train_df=cancer_data,
    test_df=cancer_data,
    label_col=label_col,
    models=models,
    test_size=TEST_SIZE,
    seeds=seeds
)

display(
    trtr_results[
        [
            "Model",
            "Accuracy (Mean±Std)",
            "F1 (Mean±Std)",
            "Precision (Mean±Std)",
            "Recall (Mean±Std)"
        ]
    ]
)

print("=" * 70)

all_comparisons = []

for synth_name in model_order:

    print(f"{synth_name} - TSTR")

    synthetic_train_df = synthetic_datasets[synth_name]

    tstr_results = evaluate_models(
        train_df=synthetic_train_df,
        test_df=cancer_data,
        label_col=label_col,
        models=models,
        test_size=TEST_SIZE,
        seeds=seeds
    )

    display(
        tstr_results[
            [
                "Model",
                "Accuracy (Mean±Std)",
                "F1 (Mean±Std)",
                "Precision (Mean±Std)",
                "Recall (Mean±Std)"
            ]
        ]
    )

    comparison = trtr_results.merge(
        tstr_results,
        on="Model",
        suffixes=("_TRTR", "_TSTR")
    )

    comparison["Accuracy_Drop"] = (
        comparison["Accuracy Mean_TRTR"]
        - comparison["Accuracy Mean_TSTR"]
    )

    comparison["F1_Drop"] = (
        comparison["F1 Mean_TRTR"]
        - comparison["F1 Mean_TSTR"]
    )

    comparison["Precision_Drop"] = (
        comparison["Precision Mean_TRTR"]
        - comparison["Precision Mean_TSTR"]
    )

    comparison["Recall_Drop"] = (
        comparison["Recall Mean_TRTR"]
        - comparison["Recall Mean_TSTR"]
    )

    comparison["Synthetic_Model"] = synth_name

    print(f"{synth_name} - TRTR vs TSTR")

    display(
        comparison[
            [
                "Synthetic_Model",
                "Model",
                "Accuracy_Drop",
                "F1_Drop",
                "Precision_Drop",
                "Recall_Drop",
                "Accuracy (Mean±Std)_TRTR",
                "Accuracy (Mean±Std)_TSTR"
            ]
        ]
    )

    all_comparisons.append(comparison)

combined_comparison = pd.concat(
    all_comparisons,
    ignore_index=True
)

summary = (
    combined_comparison
    .groupby("Synthetic_Model", as_index=False)
    [["Accuracy_Drop", "F1_Drop", "Precision_Drop", "Recall_Drop"]]
    .mean()
    .sort_values("Accuracy_Drop")
)

print("Average metric drop by synthetic generator (lower is better)")

display(summary)


# ========================================================================
# Cell 10
# ========================================================================
output_file = OUTPUT_FILE

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    
    trtr_results.to_excel(
        writer,
        sheet_name="TRTR_Results",
        index=False
    )

    combined_comparison.to_excel(
        writer,
        sheet_name="All_Comparisons",
        index=False
    )

    summary.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )

    for synth_name in model_order:
        synth_results = combined_comparison[
            combined_comparison["Synthetic_Model"] == synth_name
        ]

        synth_results.to_excel(
            writer,
            sheet_name=synth_name[:31],
            index=False
        )

print(f"Results saved to: {output_file}")

