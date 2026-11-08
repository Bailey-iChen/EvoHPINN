"""Export checkpoint predictions on application grids for inspection and reuse."""

import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from .autodiff import coordinates
from .model import relative_errors


def write_csv(path, columns, values):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(values)


def export_fields(model, checkpoint, destination, grid_x=121, grid_y=61,
                  time=0.5, wall_points=121, batch_size=256):
    """No training or parameter fitting; every prediction comes from checkpoint."""
    name = model.problem.name
    if name not in {"inverse_burgers", "hemodynamics"}:
        raise ValueError("Application exports support inverse_burgers and hemodynamics")
    if min(grid_x, grid_y, wall_points) < 2 or batch_size < 1:
        raise ValueError("Grid and wall sizes must be at least 2; batch size must be positive")
    if not math.isfinite(time) or not 0 <= time <= 1:
        raise ValueError("Inverse time must lie in [0, 1]")
    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"Export directory already exists: {target}")
    device, dtype = model.reference.device, model.reference.dtype
    if name == "hemodynamics":
        points = model.problem.hemodynamics_grid(grid_x, grid_y).to(device=device, dtype=dtype)
    else:
        xx, yy = torch.meshgrid(torch.linspace(0, 1, grid_x, dtype=dtype, device=device),
                                torch.linspace(0, 1, grid_y, dtype=dtype, device=device), indexing="ij")
        points = torch.stack((xx.reshape(-1), yy.reshape(-1),
                              torch.full((grid_x * grid_y,), time, dtype=dtype, device=device)), dim=1)
    checkpoint = Path(checkpoint)
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    predictions, references, gates = [], [], []
    # Prediction uses residual derivatives, even though parameter fitting is off.
    with torch.enable_grad():
        normalizers = model.normalizers()
        if normalizers is not None:
            normalizers = normalizers.detach()
        for batch in points.split(batch_size):
            x = coordinates(batch)
            prediction, weights, _ = model.components(x, normalizers=normalizers)
            predictions.append(prediction.detach().cpu())
            references.append(model.problem.exact(x).detach().cpu())
            gates.append(weights.detach().cpu())
        metrics = relative_errors(model, points, batch_size=batch_size)
        wall = None
        if name == "hemodynamics":
            from .metrics import hemodynamics_wall_fields
            wall = hemodynamics_wall_fields(model, wall_points=wall_points, batch_size=batch_size)
            # relative_errors uses the standard 121-point WSS grid; honor an explicit export override.
            if wall_points != 121:
                from .metrics import hemodynamics_metrics
                metrics = hemodynamics_metrics(model, points, wall_points=wall_points, batch_size=batch_size)
    prediction, reference, weights = map(torch.cat, (predictions, references, gates))
    columns = ["x", "y", "t"] if name == "inverse_burgers" else ["x", "y"]
    variables = ["u"] if name == "inverse_burgers" else ["u", "v", "p"]
    for suffix in ("prediction", "reference"):
        columns.extend(f"{var}_{suffix}" for var in variables)
    columns.extend(f"{var}_absolute_error" for var in variables)
    columns.extend(f"gate_{i + 1}" for i in range(weights.shape[1]))
    rows = torch.cat((points.cpu(), prediction, reference, (prediction - reference).abs(), weights), dim=1)
    if not torch.isfinite(rows).all() or any(not math.isfinite(v) for v in metrics.values() if isinstance(v, (int, float))):
        raise FloatingPointError("Non-finite checkpoint output; no export was written")
    target.mkdir(parents=True, exist_ok=False)
    write_csv(target / "fields.csv", columns, rows.tolist())
    if wall is not None:
        values = torch.stack([wall[key].detach().cpu().reshape(-1)
                              for key in ("x", "wss_prediction", "wss_reference")], dim=1)
        write_csv(target / "wall_shear_stress.csv", ["x", "wss_prediction", "wss_reference"], values.tolist())
    metadata = {
        "source": "Predictions evaluated from the named checkpoint; not manuscript table values",
        "problem": name, "checkpoint": checkpoint.name, "checkpoint_sha256": checkpoint_hash,
        "grid": {"nx": grid_x, "ny": grid_y}, "metrics": metrics,
    }
    if name == "inverse_burgers":
        metadata["grid"]["t"] = time
        metadata["true_viscosity"] = model.problem.viscosity
        metadata["initial_viscosity"] = 0.05
        metadata["metric_scope"] = "Field relative L2 on this time slice; full-space-time evaluation uses evaluate --samples"
        samples_path = checkpoint.parent / "samples.pt"
        if samples_path.is_file():
            samples = torch.load(samples_path, map_location="cpu", weights_only=True)
            observed, noisy = samples["observations"], samples["values"]
            clean = model.problem.exact(observed).detach()
            write_csv(target / "observations.csv", ["x", "y", "t", "u_observed", "u_reference"],
                      torch.cat((observed, noisy, clean), dim=1).tolist())
            metadata["observations"] = {
                "source": samples_path.name,
                "source_sha256": hashlib.sha256(samples_path.read_bytes()).hexdigest(),
                "count": len(observed),
            }
    else:
        metadata["wall_points"] = wall_points
        metadata["metric_scope"] = "Pressure mean shifted in field errors; exported pressure columns retain raw values"
    (target / "metrics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
