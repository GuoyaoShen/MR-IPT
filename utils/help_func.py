import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml


DEFAULT_PATH_MODEL = str(
    (Path(__file__).resolve().parents[1] / "saved_models").resolve()
)


DEFAULT_TRAIN_CONFIG = {
    "dataset_path": "/bigdata/RadImageNet/rin2d/radiology_ai/MR/brain/normal",
    "experiment_name": "radimagenet_pretrain",
    "device": "cuda:0",
    "input_height": 128,
    "input_width": 128,
    "val_split": 0.1,
    "batch_size": 6,
    "num_epochs": 5,
    "learning_rate": 1e-5,
    "num_workers": 8,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "use_precomputed_mask": False,
    "seed": None,
    "mode": "type",
    "path_model": DEFAULT_PATH_MODEL,
    "loss": "l1",
    "resume_from": None,
}


def print_var_detail(var, name=""):
    """
    Args: Print basic detail of a variable

    :param var: input variable
    :param name: variable name, default is empty
    :return: string with basic information
    """
    print(
        name,
        "is a ",
        type(var),
        "with shape",
        var.shape if torch.is_tensor(var) or isinstance(var, np.ndarray) else None,
        "max: ",
        var.max()
        if torch.is_tensor(var)
        and not torch.is_complex(var)
        or isinstance(var, np.ndarray)
        else None,
        "min: ",
        var.min()
        if torch.is_tensor(var)
        and not torch.is_complex(var)
        or isinstance(var, np.ndarray)
        else None,
    )


def create_path(path):
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")
    else:
        print("Path already exists.")


def _normalize_optional_none(value):
    """Convert common string sentinels to ``None``."""
    if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
        return None
    return value


def _to_bool(value):
    """Convert common bool-like values to Python bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in {"true", "1", "yes", "y", "on"}:
            return True
        if val in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Cannot interpret as bool: {value!r}")


def _coerce_train_config_types(cfg):
    """Coerce known config keys to expected scalar types."""
    int_keys = [
        "input_height",
        "input_width",
        "batch_size",
        "num_epochs",
        "num_workers",
        "prefetch_factor",
    ]
    float_keys = ["val_split", "learning_rate"]
    bool_keys = ["persistent_workers", "use_precomputed_mask"]

    for key in int_keys:
        cfg[key] = int(cfg[key])
    for key in float_keys:
        cfg[key] = float(cfg[key])
    for key in bool_keys:
        cfg[key] = _to_bool(cfg[key])

    return cfg


def load_train_config(config_path):
    """Load training YAML and backfill omitted keys with defaults.

    Available config keys:
        dataset_path (str): Root image dataset directory.
        experiment_name (str): Experiment group subfolder name.
        device (str): Torch device string, e.g. "cpu", "cuda:0".
        input_height (int): Input image height.
        input_width (int): Input image width.
        val_split (float): Validation split ratio in [0, 1].
        batch_size (int): Training batch size.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Optimizer learning rate.
        num_workers (int): Dataloader worker process count.
        prefetch_factor (int): Dataloader prefetch factor.
        persistent_workers (bool): Keep workers alive between epochs.
        use_precomputed_mask (bool): Reuse precomputed masks.
        seed (int | None): Random seed, supports "none"/"null".
        mode (str): One of "type", "level", or "combine".
        path_model (str): Checkpoint root directory. Converted to
            absolute path during loading.
        loss (str): One of "l1", "l2", or "mixed_l1_grad".
        resume_from (str | None): Optional checkpoint path to load model
            weights from. If null/none, training starts from scratch.

    Args:
        config_path: YAML file path.

    Returns:
        A config dictionary that includes all required training fields.
    """
    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a mapping/object: {cfg_path}")

    cfg = dict(DEFAULT_TRAIN_CONFIG)
    cfg.update(loaded)

    if "num_epoch" in cfg and "num_epochs" not in loaded:
        cfg["num_epochs"] = cfg["num_epoch"]

    cfg["seed"] = _normalize_optional_none(cfg.get("seed"))
    cfg["resume_from"] = _normalize_optional_none(cfg.get("resume_from"))
    cfg["loss"] = str(cfg.get("loss", "l1")).lower()
    if cfg["loss"] not in {"l1", "l2", "mixed_l1_grad"}:
        raise ValueError("loss must be one of: l1, l2, mixed_l1_grad")
    cfg["mode"] = str(cfg.get("mode", "type")).lower()
    if cfg["mode"] not in {"type", "level", "combine"}:
        raise ValueError("mode must be one of: type, level, combine")
    cfg = _coerce_train_config_types(cfg)
    cfg["path_model"] = str(Path(cfg["path_model"]).expanduser().resolve())
    if cfg["resume_from"] is not None:
        cfg["resume_from"] = str(Path(cfg["resume_from"]).expanduser().resolve())

    cfg["config_path"] = str(cfg_path)
    return cfg


def _build_timestamp() -> str:
    """Return run timestamp in `YYYYMMDD-HHMMSS` format."""
    now = datetime.now()
    return now.strftime("%Y%m%d-%H%M%S")


def setup_save_dir(config) -> str:
    """Create a timestamped run directory and persist the resolved config.

    The directory layout is:
        <path_model>/<experiment_name>/YYYYMMDD-HHMMSS/

    Side effects:
        1. Creates missing directories for ``path_model``, experiment folder,
           and run folder.
        2. Saves the provided config snapshot to
           ``<run_dir>/train_config.yaml``.

    Args:
        config: Resolved training config dictionary. Must include:
            - ``path_model`` (str): Absolute or relative checkpoint root.
            - ``experiment_name`` (str): Experiment subfolder name.

    Returns:
        Absolute run directory path with a trailing slash. The trailing slash
        keeps compatibility with downstream checkpoint save code that appends
        filenames directly.
    """
    create_path(config["path_model"])
    experiment_dir = Path(config["path_model"]) / config["experiment_name"]
    create_path(str(experiment_dir))

    run_dir = experiment_dir / _build_timestamp()
    create_path(str(run_dir))

    config_path = run_dir / "train_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)

    return f"{run_dir}/"
