import torch
from pathlib import Path
import pandas as pd
from typing import Any

from Helpers.configRunLoader import ConfigRun
from EnviromentSetup.corrupt.maliciousContributions.maliciousContributionsOnData import (
    MaliciousContributionsGeneratorOnData,
)
from EnviromentSetup.corrupt.maliciousContributions.maliciousContributionsOnGradient import (
    MaliciousContributionsGeneratorOnGradient,
)
from EnviromentSetup.corrupt.participationAttack.freeRiderAttack import (
    FreeRiderAttack,
    FreeRiderDataAttack,
)
from EnviromentSetup.corrupt.participationAttack.sybilAmplificationAttack import (
    SybilAmplificationAttack,
)

def safe_param_subtract(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape == b.shape:
        return a - b
    if a.numel() == b.numel():
        return a.reshape(b.shape) - b
    if a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[1]:
        max_rows = max(a.shape[0], b.shape[0])

        def pad(t, rows):
            if t.shape[0] == rows:
                return t
            pad = torch.zeros((rows - t.shape[0], t.shape[1]), device=t.device, dtype=t.dtype)
            return torch.cat([t, pad], dim=0)

        return pad(a, max_rows) - pad(b, max_rows)

    try:
        return torch.zeros_like(b)
    except:
        return torch.zeros(b.shape, dtype=b.dtype, device=b.device)

VALID_EXPERIMENTS = {
    "MC_DATA",
    "MC_GRAD",
    "MC_BOTH",
    "FR_DATA",
    "FR_GRAD",
    "FR_BOTH",
    "SYBIL_ONLY",
}


class ExperimentConfig:
    def __init__(self, run_cfg: Any):
        self.run_cfg = run_cfg
        self.experiment_case = getattr(run_cfg, "experiment_case", None)
        if self.experiment_case is None:
            atk_cat = getattr(run_cfg, "attack_category", None)
            atk_mode = getattr(run_cfg, "attack_mode", None)
            if atk_cat == "maliciousContributions":
                if atk_mode == "data":
                    self.experiment_case = "MC_DATA"
                elif atk_mode == "grad":
                    self.experiment_case = "MC_GRAD"
                elif atk_mode == "both":
                    self.experiment_case = "MC_BOTH"
            elif atk_cat == "participationAttack":
                if atk_mode == "data":
                    self.experiment_case = "FR_DATA"
                elif atk_mode == "grad":
                    self.experiment_case = "FR_GRAD"
                elif atk_mode == "both":
                    self.experiment_case = "FR_BOTH"
            else:
                # default
                self.experiment_case = "MC_DATA"

        if self.experiment_case not in VALID_EXPERIMENTS:
            raise ValueError(f"Unknown experiment_case='{self.experiment_case}'. Choose one of {VALID_EXPERIMENTS}")

        # Derived booleans
        self.is_mc = self.experiment_case.startswith("MC")
        self.is_fr = self.experiment_case.startswith("FR")
        self.is_sybil_only = self.experiment_case == "SYBIL_ONLY"
        self.data_mode = "_DATA" in self.experiment_case or "_BOTH" in self.experiment_case
        self.grad_mode = "_GRAD" in self.experiment_case or "_BOTH" in self.experiment_case or self.is_sybil_only

        self.mc_attack_type = getattr(run_cfg, "mc_attack_type", "random_label_flip")
        self.mc_grad_attack_type = getattr(run_cfg, "mc_grad_attack_type", "sign_flip")

    def summary(self) -> str:
        return (f"experiment_case={self.experiment_case} is_mc={self.is_mc} is_fr={self.is_fr} "
                f"is_sybil_only={self.is_sybil_only} data_mode={self.data_mode} grad_mode={self.grad_mode}")


class AttackEngines:
    def __init__(self, run_cfg):
        # load configs
        BASE_DIR = Path(__file__).resolve().parents[3]
        run_cfg = ConfigRun.load(BASE_DIR / "config" / "config_run.yaml")
        self.run_cfg = run_cfg if run_cfg is not None else run_cfg

        # data engines
        self.free_rider_data_engine = FreeRiderDataAttack(
            mode=run_cfg.free_rider_mode,
            tiny_fraction=run_cfg.tiny_fraction,
            duplicate_factor=run_cfg.duplicate_factor,
            fake_data_size=run_cfg.fake_data_size_data,
            seed=run_cfg.seed
        )

        self.mc_data_template = MaliciousContributionsGeneratorOnData(
            df=pd.DataFrame(),
            text_col=run_cfg.text_col,
            label_col=run_cfg.label_col,
            client_col=run_cfg.client_col,
            seed=run_cfg.seed,
            suspect_fraction=run_cfg.suspect_fraction
        )

        # gradient engines
        self.mc_grad_engine = MaliciousContributionsGeneratorOnGradient(
            attack_type=run_cfg.mc_grad_attack_type,
            scale_factor=run_cfg.mc_grad_scale_factor,
            seed=run_cfg.seed,
            history_window=run_cfg.mc_grad_window
        )

        self.free_rider_grad_engine = FreeRiderAttack(
            mode=run_cfg.free_rider_grad_mode,
            fake_data_size=run_cfg.fake_data_size_grad,
            noise_scale=run_cfg.noise_scale,
            seed=run_cfg.seed
        )

        sybil_shared_vector = getattr(run_cfg, "sybil_shared_vector", None)
        self.sybil_engine = SybilAmplificationAttack(
            amplification_factor=run_cfg.sybil_amplification_factor,
            sybil_mode=run_cfg.sybil_mode,
            shared_vector=None,                       # updated later each round
            fake_data_size=run_cfg.sybil_fake_data_size,
            alpha=run_cfg.alpha,
            collusion=run_cfg.sybil_collusion
        )