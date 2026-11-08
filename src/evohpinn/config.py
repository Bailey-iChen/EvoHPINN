import copy
import json
import math
from pathlib import Path

from .model import Architecture
from .problems import Problem


DEFAULT = {
    "status": "small-scale validation configuration; paper-scale benchmarks pending",
    "problem": "klein_gordon", "case": 1, "seed": 0, "device": "cpu", "threads": 1,
    "log_steps": True,
    "counts": {"residual": 32, "boundary": 16, "initial": 16, "reference": 16,
               "validation": 32, "test": 128, "observations": 20},
    "model": {"alpha": [1.0, 0.1], "gate": "residual", "normalization_epsilon": 1e-8, "norm_epsilon": 1e-12},
    "penalty": {"lambda": 1e-4, "depth": 1.0, "width": 0.1},
    "evaluation": "gated",
    "training": {"adam_steps": 0, "adam_lr": 1e-3, "lbfgs_steps": 2, "lbfgs_max_iter": 2,
                 "lbfgs_history": 10,
                 "weights": {"pde": 1.0, "boundary": 100.0, "initial": 100.0, "data": 100.0, "gauge": 100.0}},
    "search": {"population": 3, "generations": 2, "elite_ratio": 0.25,
               "crossover_mutation_split": 0.5, "crossover_probability": 1.0,
               "mutation": {"depth": 0.3, "width": 0.3, "activation": 0.7},
               "space": {"depths": [1, 2], "widths": [8, 12], "activations": ["tanh", "sin", "softplus"]}},
}


def merge(base, update, prefix=""):
    for key, value in update.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {prefix}{key}")
            merge(base[key], value, prefix + key + ".")
        else:
            base[key] = value
    return base


def load(path=None, **overrides):
    config = copy.deepcopy(DEFAULT)
    if path:
        merge(config, json.loads(Path(path).read_text()))
    merge(config, {k: v for k, v in overrides.items() if v is not None})
    validate(config)
    return config


def validate(config):
    Problem(config["problem"], config["case"])
    if config["device"] != "cpu" and not config["device"].startswith("cuda"):
        raise ValueError("Use cpu or cuda[:index]; float64 is required and MPS is unsupported")
    for key, value in config["counts"].items():
        if not isinstance(value, int) or isinstance(value, bool) or value < (2 if key == "boundary" else 1):
            raise ValueError(f"Invalid sample count: {key}")
    if config["problem"] == "hemodynamics" and config["counts"]["boundary"] % 4:
        raise ValueError("Hemodynamics boundary count must be divisible by four")
    if config["threads"] < 1:
        raise ValueError("threads must be positive")
    settings = config["search"]
    if settings["population"] < 1 or settings["generations"] < 1:
        raise ValueError("population and generations must be positive")
    for key in ("elite_ratio", "crossover_mutation_split", "crossover_probability"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"Invalid probability: {key}")
    if any(not 0 <= v <= 1 for v in settings["mutation"].values()):
        raise ValueError("Mutation probabilities must lie in [0, 1]")
    space = settings["space"]
    if not all(space.values()):
        raise ValueError("Search space must be nonempty")
    if space["depths"] != sorted(set(space["depths"])) or space["widths"] != sorted(set(space["widths"])):
        raise ValueError("Depths and widths must be sorted and unique")
    for depth in space["depths"]:
        for width in space["widths"]:
            for activation in space["activations"]:
                Architecture(depth, width, activation)
    if not config["model"]["alpha"] or any(a <= 0 or not math.isfinite(a) for a in config["model"]["alpha"]):
        raise ValueError("Capacity coefficients must be finite and positive")
    if config["model"]["gate"] not in {"uniform", "residual"} or config["evaluation"] not in {"gated", "uniform"}:
        raise ValueError("Invalid gating or evaluation rule")
    for key in ("normalization_epsilon", "norm_epsilon"):
        if config["model"][key] <= 0:
            raise ValueError(f"{key} must be positive")
    if any(v < 0 or not math.isfinite(v) for v in config["penalty"].values()):
        raise ValueError("Penalty coefficients must be finite and nonnegative")
    training = config["training"]
    if training["adam_steps"] < 0 or training["lbfgs_steps"] < 0 or training["adam_steps"] + training["lbfgs_steps"] == 0:
        raise ValueError("Specify a positive inner training budget")
    if training["adam_lr"] <= 0 or training["lbfgs_max_iter"] < 1 or training["lbfgs_history"] < 1:
        raise ValueError("Invalid optimizer settings")
    if any(v <= 0 or not math.isfinite(v) for v in training["weights"].values()):
        raise ValueError("Loss weights must be finite and positive")
