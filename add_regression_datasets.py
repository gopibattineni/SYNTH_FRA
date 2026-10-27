"""
Add four UCI regression datasets (#12-#15) across all benchmark folders:
  - SDV models
  - Other GANS
  - Single run_Data_leak_Synth_Quality
  - diffusion_dataleak (via build_diffusion_dataleak_notebooks.py)
  - Diffusion GANs (via build_diffusion_notebooks.py)
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parent

DATASETS = [
    {
        "num": 12,
        "id": "air_quality",
        "name": "Air Quality",
        "folder": "12. Air Quality",
        "notebook": "air_quality.ipynb",
        "other_gan": "12. Air Quality_other GAN.ipynb",
        "sdv": "12. Air Quality.ipynb",
        "uci_id": 360,
        "target_col": "CO(GT)",
        "repo_var": "air_quality",
        "output_file": "TRTR_TSTR_results_air_quality.xlsx",
        "load_block": '''
air_quality = fetch_ucirepo(id=360)
X = air_quality.data.features.copy()
print(air_quality.metadata)
print(air_quality.variables)

target_col = "CO(GT)"
X = X.drop(columns=["Date", "Time"], errors="ignore")
data = X.copy()
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.replace(-200, np.nan)
for col in data.columns:
    data[col] = data[col].fillna(data[col].median())
data = data.dropna().reset_index(drop=True)
''',
    },
    {
        "num": 13,
        "id": "concrete",
        "name": "Concrete Compressive Strength",
        "folder": "13. Concrete Compressive Strength",
        "notebook": "concrete.ipynb",
        "other_gan": "13. Concrete Compressive Strength_other GAN.ipynb",
        "sdv": "13. Concrete Compressive Strength.ipynb",
        "uci_id": 165,
        "target_col": "Concrete compressive strength",
        "repo_var": "concrete",
        "output_file": "TRTR_TSTR_results_concrete.xlsx",
        "load_block": '''
concrete = fetch_ucirepo(id=165)
X = concrete.data.features
y = concrete.data.targets
print(concrete.metadata)
print(concrete.variables)

data = pd.concat([X, y], axis=1)
target_col = "Concrete compressive strength"
''',
    },
    {
        "num": 14,
        "id": "energy_efficiency",
        "name": "Energy Efficiency",
        "folder": "14. Energy Efficiency",
        "notebook": "energy_efficiency.ipynb",
        "other_gan": "14. Energy Efficiency_other GAN.ipynb",
        "sdv": "14. Energy Efficiency.ipynb",
        "uci_id": 242,
        "target_col": "Y1",
        "repo_var": "energy_efficiency",
        "output_file": "TRTR_TSTR_results_energy_efficiency.xlsx",
        "load_block": '''
energy_efficiency = fetch_ucirepo(id=242)
X = energy_efficiency.data.features
y = energy_efficiency.data.targets
print(energy_efficiency.metadata)
print(energy_efficiency.variables)

data = pd.concat([X, y[["Y1"]]], axis=1)
target_col = "Y1"
''',
    },
    {
        "num": 15,
        "id": "real_estate",
        "name": "Real Estate Valuation",
        "folder": "15. Real Estate Valuation",
        "notebook": "real_estate.ipynb",
        "other_gan": "15. Real Estate Valuation_other GAN.ipynb",
        "sdv": "15. Real Estate Valuation.ipynb",
        "uci_id": 477,
        "target_col": "Y house price of unit area",
        "repo_var": "real_estate",
        "output_file": "TRTR_TSTR_results_real_estate.xlsx",
        "load_block": '''
real_estate = fetch_ucirepo(id=477)
X = real_estate.data.features.copy()
y = real_estate.data.targets
print(real_estate.metadata)
print(real_estate.variables)

X = X.drop(columns=["X1 transaction date"], errors="ignore")
if "X4 number of convenience stores" in X.columns:
    X["X4 number of convenience stores"] = pd.to_numeric(
        X["X4 number of convenience stores"], errors="coerce"
    )

data = pd.concat([X, y], axis=1)
target_col = "Y house price of unit area"
''',
    },
]


DROP_FEATURE_BLOCK = """
# Drop date/time/session/ID features before generator training (high cardinality).
_drop_feature_cols = [
    "Date", "Time", "date_time",
    "session_id", "Session ID", "Session_ID", "session",
    "X1 transaction date",
]
data = data.drop(columns=[c for c in _drop_feature_cols if c in data.columns], errors="ignore")
"""


def _single_run_data_cell(cfg: dict) -> str:
    return f'''from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
import torch
import random

{cfg["load_block"].strip()}
{DROP_FEATURE_BLOCK}

data = data.sample(n=1000, random_state=42).reset_index(drop=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

processed_data = data.copy()

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(processed_data)

N_SAMPLES = 1000
TEST_SIZE = 0.2
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

scores = {{}}
synthetic_datasets = {{}}
quality_results = []
'''


def _other_gans_data_cell(cfg: dict) -> str:
    tc = cfg["target_col"]
    return f'''from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from model.ctabgan import CTABGAN


# -----------------------------
# Load {cfg["name"]} dataset
# -----------------------------
{cfg["load_block"].strip()}
{DROP_FEATURE_BLOCK}

data = data.sample(n=1000, random_state=42).reset_index(drop=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())

target_col = "{tc}"
X = data.drop(columns=[target_col])

# -----------------------------
# CTABGAN
# -----------------------------
data_path = "{cfg["id"]}_data_1000.csv"
data.to_csv(data_path, index=False)

categorical_columns = data.select_dtypes(include=["object", "category"]).columns.tolist()
integer_columns = X.select_dtypes(include=["int64", "int32"]).columns.tolist()

ctabgan = CTABGAN(
    raw_csv_path=data_path,
    categorical_columns=categorical_columns,
    log_columns=[],
    mixed_columns={{}},
    integer_columns=integer_columns,
    problem_type={{"Regression": target_col}}
)

ctabgan.fit()

synthetic_ctabgan = ctabgan.data_prep.inverse_prep(
    ctabgan.synthesizer.sample(1000)
)

# -----------------------------
# WGAN-GP
# -----------------------------
data_wgan = data.copy()

for col in data_wgan.columns:
    if data_wgan[col].dtype == "object":
        data_wgan[col] = data_wgan[col].fillna("Missing")
    else:
        data_wgan[col] = data_wgan[col].fillna(data_wgan[col].median())

label_encoders = {{}}
categorical_cols_wgan = data_wgan.select_dtypes(include=["object", "category"]).columns.tolist()

for col in categorical_cols_wgan:
    le = LabelEncoder()
    data_wgan[col] = le.fit_transform(data_wgan[col].astype(str))
    label_encoders[col] = le

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_wgan)
real_columns = data_wgan.columns

device = "cuda" if torch.cuda.is_available() else "cpu"
real_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)

batch_size = 64
latent_dim = 32
data_dim = real_tensor.shape[1]
epochs = 30
critic_iterations = 2
lambda_gp = 10

train_loader = torch.utils.data.DataLoader(
    real_tensor, batch_size=batch_size, shuffle=True, drop_last=True
)


class Generator(nn.Module):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, data_dim),
        )

    def forward(self, z):
        return self.model(z)


class Critic(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.model(x)


generator = Generator(latent_dim, data_dim).to(device)
critic = Critic(data_dim).to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.9))
optimizer_C = optim.Adam(critic.parameters(), lr=0.0001, betas=(0.5, 0.9))


def gradient_penalty(critic, real_samples, fake_samples):
    bs = real_samples.size(0)
    alpha = torch.rand(bs, 1, device=device).expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    critic_interpolates = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(bs, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


for epoch in range(epochs):
    for i, real_batch in enumerate(train_loader):
        real_batch = real_batch.to(device)
        for _ in range(critic_iterations):
            z = torch.randn(real_batch.size(0), latent_dim, device=device)
            fake_batch = generator(z)
            critic_real = critic(real_batch).mean()
            critic_fake = critic(fake_batch.detach()).mean()
            gp = gradient_penalty(critic, real_batch, fake_batch.detach())
            critic_loss = -(critic_real - critic_fake) + lambda_gp * gp
            optimizer_C.zero_grad()
            critic_loss.backward()
            optimizer_C.step()

        z = torch.randn(real_batch.size(0), latent_dim, device=device)
        fake_batch = generator(z)
        g_loss = -critic(fake_batch).mean()
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

z = torch.randn(1000, latent_dim, device=device)
fake_tensor = generator(z).detach().cpu().numpy()
synthetic_wgan = pd.DataFrame(scaler.inverse_transform(fake_tensor), columns=real_columns)

for col, le in label_encoders.items():
    synthetic_wgan[col] = le.inverse_transform(
        synthetic_wgan[col].round().astype(int).clip(0, len(le.classes_) - 1)
    )

synthetic_outputs = {{
    "CTABGAN": synthetic_ctabgan,
    "WGAN_GP": synthetic_wgan,
}}
'''


def _sdv_load_cell(cfg: dict) -> str:
    return f'''from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
{cfg["repo_var"]} = fetch_ucirepo(id={cfg["uci_id"]}) 
  
# data (as pandas dataframes) 
X = {cfg["repo_var"]}.data.features 
y = {cfg["repo_var"]}.data.targets 
  
# metadata 
print({cfg["repo_var"]}.metadata) 
  
# variable information 
print({cfg["repo_var"]}.variables) '''


def _sdv_preprocess_cell(cfg: dict) -> str:
    tc = cfg["target_col"]
    extra = ""
    if cfg["id"] == "air_quality":
        extra = '''
X = X.drop(columns=["Date", "Time"], errors="ignore")
data = X.copy()
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
data = data.replace(-200, np.nan)
for col in data.columns:
    data[col] = data[col].fillna(data[col].median())
data = data.dropna().reset_index(drop=True)
target_col = "CO(GT)"
'''
    elif cfg["id"] == "energy_efficiency":
        extra = f'''
data = pd.concat([X, y[["Y1"]]], axis=1)
target_col = "{tc}"
'''
    elif cfg["id"] == "real_estate":
        extra = f'''
X = X.drop(columns=["X1 transaction date"], errors="ignore")
if "X4 number of convenience stores" in X.columns:
    X["X4 number of convenience stores"] = pd.to_numeric(
        X["X4 number of convenience stores"], errors="coerce"
    )
data = pd.concat([X, y], axis=1)
target_col = "{tc}"
'''
    else:
        extra = f'''
data = pd.concat([X, y], axis=1)
target_col = "{tc}"
'''

    return f'''import pandas as pd
import numpy as np

{extra.strip()}
{DROP_FEATURE_BLOCK}

data = data.sample(n=1000, random_state=42).reset_index(drop=True)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    data[col] = data[col].fillna(data[col].median())
'''


def _replace_in_notebook(nb: dict, mapping: dict[str, str]) -> None:
    for cell in nb["cells"]:
        if not cell.get("source"):
            continue
        src = "".join(cell["source"])
        for old, new in mapping.items():
            src = src.replace(old, new)
        cell["source"] = [src]


def _set_cell_with_marker(nb: dict, marker: str, new_src: str) -> bool:
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if marker in src:
            cell["source"] = [new_src]
            cell["outputs"] = []
            cell["execution_count"] = None
            return True
    return False


def _patch_single_run(nb: dict, cfg: dict) -> None:
    tc = cfg["target_col"]
    mapping = {
        "wine_quality": cfg["repo_var"],
        "Wine": cfg["name"],
        "wine_quality_train.csv": f"{cfg['id']}_train.csv",
        'target_col = "quality"': f'target_col = "{tc}"',
    }
    _replace_in_notebook(nb, mapping)

    nb["cells"][2]["source"] = [_single_run_data_cell(cfg)]

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "train_test_split" in src and "stratify=processed_data" in src:
            src = re.sub(
                r"stratify=processed_data\[target_col\],\s*\n",
                "",
                src,
            )
            cell["source"] = [src]
        if "problem_type={\"Classification\"" in src:
            src = src.replace(
                'problem_type={"Classification": target_col}',
                'problem_type={"Regression": target_col}',
            )
            cell["source"] = [src]
        if "Wine quality target" in src:
            src = re.sub(
                r"\n\s*# Wine quality target.*?\.astype\(int\)\s*\)",
                "",
                src,
                flags=re.S,
            )
            cell["source"] = [src]
        if "encoder = LabelEncoder()" in src and "data_wgan[target_col]" in src:
            src = src.replace(
                "encoder = LabelEncoder()\n    data_wgan[target_col] = encoder.fit_transform(data_wgan[target_col])\n\n    scaler",
                "scaler",
            )
            src = re.sub(
                r"synthetic_wgan\[target_col\] = \(\s*synthetic_wgan\[target_col\].*?\.astype\(int\)\s*\)\s*\n\s*synthetic_wgan\[target_col\] = encoder\.inverse_transform\(\s*synthetic_wgan\[target_col\]\s*\)\s*\n",
                "",
                src,
                flags=re.S,
            )
            cell["source"] = [src]


def _patch_other_gans(nb: dict, cfg: dict) -> None:
    tc = cfg["target_col"]
    mapping = {
        "Wine Quality": cfg["name"],
        "wine_quality": cfg["repo_var"],
        "wine_quality_data_1000.csv": f"{cfg['id']}_data_1000.csv",
        'target_col = "quality"': f'target_col = "{tc}"',
        "target_col = y.columns[0]": f'target_col = "{tc}"',
        "Matchings_CTABGAN_WGAN_GP_Wine": f"Matchings_CTABGAN_WGAN_GP_{cfg['id']}",
    }
    _replace_in_notebook(nb, mapping)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "fetch_ucirepo" in src and "CTABGAN" in src:
            nb["cells"][i]["source"] = [_other_gans_data_cell(cfg)]
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            break


def _patch_sdv(nb: dict, cfg: dict) -> None:
    tc = cfg["target_col"]
    mapping = {
        "Wine Quality": cfg["name"],
        "Wine quality": cfg["name"],
        "wine_quality": cfg["repo_var"],
        "Metro Interstate Traffic Volume": cfg["name"],
        "metro_interstate_traffic_volume": cfg["repo_var"],
        'target_col = "quality"': f'target_col = "{tc}"',
        'target_col = "traffic_volume"': f'target_col = "{tc}"',
        "target_col = y.columns[0]": f'target_col = "{tc}"',
    }
    _replace_in_notebook(nb, mapping)

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if "fetch_ucirepo" in src and "print(" in src and i < 6:
            nb["cells"][i]["source"] = [_sdv_load_cell(cfg)]
            nb["cells"][i]["outputs"] = []
            break

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if i < 8 and (
            "pd.concat([X, y]" in src
            or "metro_data" in src
            or "data = data.sample" in src
            or "data.replace" in src and "?" in src
        ):
            nb["cells"][i]["source"] = [_sdv_preprocess_cell(cfg)]
            nb["cells"][i]["outputs"] = []
            break


def _write_notebook(path: Path, nb: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO)}")


def create_single_run(cfg: dict) -> None:
    src = REPO / "Single run_Data_leak_Synth_Quality" / "6. Wine dataset" / "Wine.ipynb"
    nb = json.loads(src.read_text(encoding="utf-8"))
    _patch_single_run(nb, cfg)
    out = REPO / "Single run_Data_leak_Synth_Quality" / cfg["folder"] / cfg["notebook"]
    _write_notebook(out, nb)


def create_other_gans(cfg: dict) -> None:
    src = REPO / "Other GANS" / "6. Winequality_other GAN.ipynb"
    nb = json.loads(src.read_text(encoding="utf-8"))
    _patch_other_gans(nb, cfg)
    out = REPO / "Other GANS" / cfg["other_gan"]
    _write_notebook(out, nb)


def create_sdv(cfg: dict) -> None:
    src = REPO / "SDV models" / "6. Wine quality.ipynb"
    nb = json.loads(src.read_text(encoding="utf-8"))
    _patch_sdv(nb, cfg)
    out = REPO / "SDV models" / cfg["sdv"]
    _write_notebook(out, nb)


def update_datasets_json() -> None:
    path = REPO / "Single run_Data_leak_Synth_Quality" / "python_scripts" / "hive" / "datasets.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    existing = {e["id"] for e in entries}
    for cfg in DATASETS:
        if cfg["id"] in existing:
            continue
        entries.append(
            {
                "id": cfg["id"],
                "name": cfg["name"],
                "uci_id": cfg["uci_id"],
                "notebook_dir": cfg["folder"],
                "notebook": cfg["notebook"],
                "train_csv": f"{cfg['id']}_train.csv",
                "n_samples": 1000,
                "test_size": 0.2,
                "seed": 42,
                "output_file": cfg["output_file"],
                "runner": "nbconvert",
            }
        )
    path.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    print(f"  updated {path.relative_to(REPO)}")


def main() -> None:
    print("Adding regression datasets #12-#15...")
    for cfg in DATASETS:
        print(f"\n[{cfg['num']}] {cfg['name']}")
        create_single_run(cfg)
        create_other_gans(cfg)
        create_sdv(cfg)

    update_datasets_json()

    print("\nBuilding Diffusion GANs notebooks...")
    subprocess.run([sys.executable, str(REPO / "build_diffusion_notebooks.py")], check=True)

    print("\nBuilding diffusion_dataleak notebooks...")
    subprocess.run([sys.executable, str(REPO / "build_diffusion_dataleak_notebooks.py")], check=True)

    print("\nDone. Added 4 regression datasets across all benchmark folders.")


if __name__ == "__main__":
    main()
