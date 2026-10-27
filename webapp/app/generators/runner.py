from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.config import CTABGAN_DIR, DIFFUSION_MODULE, TABDDPM_STEPS, WGAN_EPOCHS
from app.data_loader import infer_categorical_columns
from app.registry import DatasetInfo

GeneratorFn = Callable[[pd.DataFrame, DatasetInfo, int, int], pd.DataFrame]


def _set_seeds(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _ensure_ctabgan() -> Path:
    if not CTABGAN_DIR.is_dir():
        CTABGAN_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/Team-TUD/CTAB-GAN-Plus", str(CTABGAN_DIR)],
            check=True,
        )
    repo_path = str(CTABGAN_DIR)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    return CTABGAN_DIR


def _load_diffusion_module():
    spec = importlib.util.spec_from_file_location("diffusion_generators", DIFFUSION_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load diffusion module from {DIFFUSION_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_ctabgan(df: pd.DataFrame, info: DatasetInfo, n_samples: int, seed: int) -> pd.DataFrame:
    _ensure_ctabgan()
    from model.ctabgan import CTABGAN

    _set_seeds(seed)
    cats = infer_categorical_columns(df, info.target_col, info.task)
    work = df.copy()
    for col in cats:
        work[col] = work[col].astype(str)

    csv_path = CTABGAN_DIR.parent / f"_train_{info.id}.csv"
    work.to_csv(csv_path, index=False)

    if info.task == "regression":
        problem_type = {"Regression": info.target_col}
        integer_cols = [c for c in work.columns if pd.api.types.is_integer_dtype(work[c])]
    else:
        problem_type = {"Classification": info.target_col}
        integer_cols = []

    synthesizer = CTABGAN(
        raw_csv_path=str(csv_path),
        categorical_columns=cats,
        log_columns=[],
        mixed_columns={},
        integer_columns=integer_cols,
        problem_type=problem_type,
    )
    synthesizer.fit()
    return synthesizer.data_prep.inverse_prep(synthesizer.synthesizer.sample(n_samples))


def generate_wgan_gp(df: pd.DataFrame, info: DatasetInfo, n_samples: int, seed: int) -> pd.DataFrame:
    _set_seeds(seed)
    data = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    cat_cols = infer_categorical_columns(data, info.target_col, info.task)

    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    for col in data.columns:
        if col not in cat_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.fillna(0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real_tensor = torch.tensor(scaled, dtype=torch.float32)
    batch_size = min(64, max(8, len(data) // 4))
    latent_dim = 64
    data_dim = real_tensor.shape[1]

    loader = torch.utils.data.DataLoader(real_tensor, batch_size=batch_size, shuffle=True, drop_last=False)

    class Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.LayerNorm(128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 256),
                nn.LayerNorm(256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, data_dim),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.model(z)

    class Critic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(data_dim, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.2),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    generator = Generator().to(device)
    critic = Critic().to(device)
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer_c = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

    def gradient_penalty(critic_net: Critic, real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
        alpha = torch.rand(real_samples.size(0), 1, device=device).expand_as(real_samples)
        interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
        critic_interpolates = critic_net(interpolates)
        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    for _ in range(WGAN_EPOCHS):
        for real_batch in loader:
            real_batch = real_batch.to(device)
            for _ in range(5):
                z = torch.randn(real_batch.size(0), latent_dim, device=device)
                fake_batch = generator(z).detach()
                critic_real = critic(real_batch).mean()
                critic_fake = critic(fake_batch).mean()
                gp = gradient_penalty(critic, real_batch, fake_batch)
                critic_loss = critic_fake - critic_real + 10 * gp
                optimizer_c.zero_grad()
                critic_loss.backward()
                optimizer_c.step()

            z = torch.randn(real_batch.size(0), latent_dim, device=device)
            generator_loss = -critic(generator(z)).mean()
            optimizer_g.zero_grad()
            generator_loss.backward()
            optimizer_g.step()

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, device=device)
        synthetic_scaled = generator(z).cpu().numpy()

    synthetic = scaler.inverse_transform(synthetic_scaled)
    result = pd.DataFrame(synthetic, columns=data.columns)

    for col in cat_cols:
        le = encoders[col]
        rounded = np.clip(np.round(result[col]), 0, len(le.classes_) - 1).astype(int)
        result[col] = le.inverse_transform(rounded)

    if info.task == "regression":
        result[info.target_col] = pd.to_numeric(result[info.target_col], errors="coerce")

    return result[df.columns]


def generate_sdv(df: pd.DataFrame, info: DatasetInfo, n_samples: int, seed: int, model_name: str) -> pd.DataFrame:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import (
        CTGANSynthesizer,
        CopulaGANSynthesizer,
        GaussianCopulaSynthesizer,
        TVAESynthesizer,
    )

    _set_seeds(seed)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    model_map = {
        "ctgan": CTGANSynthesizer,
        "copulagan": CopulaGANSynthesizer,
        "tvae": TVAESynthesizer,
        "gaussian_copula": GaussianCopulaSynthesizer,
    }
    synthesizer = model_map[model_name](metadata=metadata)
    synthesizer.fit(df)
    return synthesizer.sample(n_samples)


def generate_tabddpm(df: pd.DataFrame, info: DatasetInfo, n_samples: int, seed: int) -> pd.DataFrame:
    diffusion = _load_diffusion_module()
    cats = infer_categorical_columns(df, info.target_col, info.task)
    is_regression = info.task == "regression"
    return diffusion.train_tabddpm(
        df,
        info.target_col,
        categorical_columns=cats,
        n_samples=n_samples,
        seed=seed,
        steps=TABDDPM_STEPS,
        is_regression=is_regression,
    )


def generate_forestdiffusion(df: pd.DataFrame, info: DatasetInfo, n_samples: int, seed: int) -> pd.DataFrame:
    diffusion = _load_diffusion_module()
    cats = infer_categorical_columns(df, info.target_col, info.task)
    return diffusion.train_forestdiffusion(
        df,
        info.target_col,
        categorical_columns=cats,
        n_samples=n_samples,
        seed=seed,
        is_regression=info.task == "regression",
    )


GENERATOR_FUNCTIONS: dict[str, GeneratorFn] = {
    "ctabgan": generate_ctabgan,
    "wgan_gp": generate_wgan_gp,
    "tabddpm": generate_tabddpm,
    "forestdiffusion": generate_forestdiffusion,
    "ctgan": lambda df, info, n, s: generate_sdv(df, info, n, s, "ctgan"),
    "copulagan": lambda df, info, n, s: generate_sdv(df, info, n, s, "copulagan"),
    "tvae": lambda df, info, n, s: generate_sdv(df, info, n, s, "tvae"),
    "gaussian_copula": lambda df, info, n, s: generate_sdv(df, info, n, s, "gaussian_copula"),
}


def generate_synthetic(
    generator_id: str,
    df: pd.DataFrame,
    info: DatasetInfo,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    fn = GENERATOR_FUNCTIONS.get(generator_id)
    if fn is None:
        raise ValueError(f"Unsupported generator: {generator_id}")
    return fn(df, info, n_samples, seed)
