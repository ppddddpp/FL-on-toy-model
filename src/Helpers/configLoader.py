from dataclasses import dataclass, field, fields, asdict
import yaml
import os
from typing import Any, Dict

@dataclass
class Config:
    # Training parameters
    epochs: int = 10
    patience: int = 2
    batch_size: int = 2
    lr: float = 2e-5

    # Model parameters
    max_seq_len: int = 128
    ffn_dim: int = 128
    num_heads: int = 4
    model_dim: int = 64
    dropout: float = 0.1
    num_layers: int = 2

    # Knowledge graph parameters
    kg_model: str = "TransE"          # options: TransE | TransH | RotatE | CompGCN
    kg_method: str = "cosine"
    kg_emb_dim: int = 300
    kg_epochs: int = 30
    kg_mode: str = "hybrid"           # options: dataset | ontology | hybrid
    kg_lr : float = 5e-4
    kg_neg_samples: int = 8
    kg_margin: float = 5.0
    adv_temp: float = 0.5

    # CompGCN-specific (only used if kg_model=CompGCN)
    kg_num_layers: int = 2
    kg_dropout: float = 0.3
    kg_opn: str = "corr"              # options: sub | mult | corr

    # Loss parameters
    cls_weight: float = 3.0
    kg_weight: float = 0.1

    # Calibration / trust tuning parameters
    calib_interval: 10 

    # Wandb parameters
    project_name: str = "fl-toy-experiments"

    # Auto-generated
    run_name: str = field(init=False, default="")

    def set_run_name(self, name_method: str):
    # build dict skipping run_name and project_name
        cfg_dict = {
            fld.name: getattr(self, fld.name)
            for fld in fields(self)
            if fld.init and fld.name not in ("run_name", "project_name")
        }

        parts = [f"method={name_method}", f"kg_model={self.kg_model}"]

        # If CompGCN, include operator explicitly
        if self.kg_model == "CompGCN":
            parts.append(f"kg_opn={self.kg_opn}")

        for k, v in cfg_dict.items():
            if k in ("kg_model", "kg_opn"):  # already added above
                continue
            if isinstance(v, float):
                if "lr" in k:
                    parts.append(f"{k}={v:.0e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

        self.run_name = "_".join(parts)

    def validate(self) -> None:
        if not (isinstance(self.lr, (float, int)) and self.lr >= 0.0):
            raise ValueError(f"lr must be non-negative. Got {self.lr!r}")
        if not (isinstance(self.batch_size, int) and self.batch_size > 0):
            raise ValueError(f"batch_size must be >0. Got {self.batch_size!r}")
        if not (isinstance(self.epochs, int) and self.epochs > 0):
            raise ValueError(f"epochs must be >0. Got {self.epochs!r}")
        if self.kg_model not in ("TransE", "TransH", "RotatE", "CompGCN"):
            raise ValueError(f"Invalid kg_model: {self.kg_model}")

    @property
    def kg_model_kwargs(self) -> dict:
        """Returns model-specific kwargs to pass into KGTrainer."""
        if self.kg_model == "CompGCN":
            return {
                "num_layers": self.kg_num_layers,
                "dropout": self.kg_dropout,
                "opn": self.kg_opn,
            }
        return {}

    @staticmethod
    def _coerce_value(raw: Any, target_type: Any) -> Any:
        if raw is None:
            return None
        try:
            if target_type is float:
                return float(raw)
            if target_type is int:
                return int(raw)
            if target_type is bool:
                if isinstance(raw, bool):
                    return raw
                if isinstance(raw, str):
                    return raw.strip().lower() in ("true", "1", "yes", "y")
                return bool(raw)
            if target_type is str:
                return str(raw)
        except Exception:
            return raw
        return raw

    @staticmethod
    def load(path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

        if not isinstance(yaml_data, dict):
            raise ValueError("Config file must contain a mapping.")

        field_map: Dict[str, Any] = {fld.name: fld for fld in fields(Config) if fld.init}

        extra_keys = set(yaml_data.keys()) - set(field_map.keys())
        if extra_keys:
            print(f"[WARN] Unknown keys in config file (ignored): {sorted(extra_keys)}")

        used_defaults = []
        data: Dict[str, Any] = {}
        for name, fld in field_map.items():
            if name in yaml_data:
                raw = yaml_data[name]
                coerced = Config._coerce_value(raw, fld.type)
                data[name] = coerced
            else:
                data[name] = fld.default
                used_defaults.append(name)

        cfg = Config(**data)
        if used_defaults:
            print("Using default values for:", ", ".join(used_defaults))
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


if __name__ == "__main__":
    example_path = os.path.join("config", "config.yaml")
    if os.path.exists(example_path):
        cfg = Config.load(example_path)
        print("Loaded config:", cfg)
        print("run_name:", cfg.run_name)
        print("LR type:", type(cfg.lr), "value:", cfg.lr)
    else:
        cfg = Config()
        print("No config.yaml found — using defaults:")
        print(cfg)
        print("run_name:", cfg.run_name)
