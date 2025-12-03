from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class ConfigRun:
    experiment_case: str = "MC_GRAD"   # MC_DATA, MC_GRAD, MC_BOTH, FR_DATA, FR_GRAD, FR_BOTH, SYBIL_ONLY

    attacker_ids: List[str] = field(default_factory=list)

    num_rounds: int = 5
    local_epochs: int = 5
    seed: int = 2709

    # Malicious Contribution — Data Attacks
    mc_data_attack_type: str = "random_label_flip"
    text_col: str = "Information"
    label_col: str = "Group"
    client_col: str = "Group"
    src_label: str = None
    tgt_label: str = None
    suspect_fraction: float = 0.10

    # Malicious Contribution — Gradient Attacks
    mc_grad_attack_type: str = "sign_flip"
    mc_grad_scale_factor: float = 10.0
    mc_grad_window: int = 5
    mc_grad_train: bool = False     # apply inside training loop
    mc_grad_delta: bool = True     # apply on model delta 

    # Free-Rider Data Attacks
    free_rider_mode: str = "tiny"       # empty | tiny | duplicate | random
    tiny_fraction: float = 0.02
    duplicate_factor: int = 20
    fake_data_size_data: int = 4

    # Free-Rider Gradient Attacks
    free_rider_grad_mode: str = "weak"   # weak | strong
    fake_data_size_grad: int = 2
    noise_scale: float = 0.001

    # Sybil
    sybil_mode: str = "static"          # static | leader | coordinated
    alpha: float = 0.8
    sybil_collusion: bool = True
    sybil_fake_data_size: int = 400
    sybil_history_window: int = 5
    sybil_amplification_factor: float = 1.0

    @staticmethod
    def load(path: str) -> "ConfigRun":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # BACKWARD COMPATIBILITY
        experiment_case = raw.get("experiment_case")
        if experiment_case is None:
            old_cat = raw.get("attack_category", "none")
            old_mode = raw.get("attack_mode", "none")

            if old_cat == "maliciousContributions":
                if old_mode == "data": experiment_case = "MC_DATA"
                elif old_mode == "grad": experiment_case = "MC_GRAD"
                elif old_mode == "both": experiment_case = "MC_BOTH"
                else: experiment_case = "MC_DATA"

            elif old_cat == "participationAttack":
                if old_mode == "data": experiment_case = "FR_DATA"
                elif old_mode == "grad": experiment_case = "FR_GRAD"
                elif old_mode == "both": experiment_case = "FR_BOTH"
                else: experiment_case = "FR_DATA"

            else:
                experiment_case = "SYBIL_ONLY"

        return ConfigRun(
            experiment_case = experiment_case,
            attacker_ids    = raw.get("attacker_ids", []),

            num_rounds      = raw.get("num_rounds", 5),
            local_epochs    = raw.get("local_epochs", 5),
            seed            = raw.get("seed", 2709),

            mc_data_attack_type = raw.get("mc_data_attack_type", "random_label_flip"),
            text_col           = raw.get("text_col", "Information"),
            label_col          = raw.get("label_col", "Group"),
            client_col         = raw.get("client_col", "Group"),
            src_label          = raw.get("src_label", None),
            tgt_label          = raw.get("tgt_label", None),
            suspect_fraction   = raw.get("suspect_fraction", 0.10),

            mc_grad_attack_type = raw.get("mc_grad_attack_type", "sign_flip"),
            mc_grad_scale_factor= raw.get("mc_grad_scale_factor", 10.0),
            mc_grad_window      = raw.get("mc_grad_window", 5),
            mc_grad_delta       = raw.get("mc_grad_delta", True),
            mc_grad_train       = raw.get("mc_grad_train", False),

            free_rider_mode     = raw.get("free_rider_mode", "tiny"),
            tiny_fraction       = raw.get("tiny_fraction", 0.02),
            duplicate_factor    = raw.get("duplicate_factor", 20),
            fake_data_size_data = raw.get("fake_data_size", 4),

            free_rider_grad_mode = raw.get("free_rider_grad_mode", "weak"),
            fake_data_size_grad  = raw.get("fake_data_size", 2),
            noise_scale          = raw.get("noise_scale", 0.001),  

            sybil_mode                 = raw.get("sybil_mode", "static"),
            alpha                      = raw.get("alpha", 0.8),
            sybil_collusion            = raw.get("sybil_collusion", True),
            sybil_fake_data_size       = raw.get("sybil_fake_data_size", 400),
            sybil_history_window       = raw.get("sybil_history_window", 5),
            sybil_amplification_factor = raw.get("sybil_amplification_factor", 1.0),
        )
