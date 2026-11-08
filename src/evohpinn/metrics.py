"""Application diagnostics for the manufactured hemodynamics benchmark.

These quantities describe a trained model, not an architecture-search score.
Automatic differentiation remains enabled for residual gating and wall shear.
"""

import torch

from .autodiff import coordinates


def _normalizers(model):
    value = model.normalizers()
    return None if value is None else value.detach()


def _check(model, batch_size):
    if model.problem.name != "hemodynamics":
        raise ValueError("These metrics apply to the hemodynamics benchmark")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


@torch.enable_grad()
def hemodynamics_wall_fields(model, wall_points=121, batch_size=256):
    """Return detached x, predicted WSS and reference WSS tensors.

    Values use the model reference tensor's device/dtype. ``wall_points``
    includes both endpoints of the straight lower boundary, y = 0.
    """
    _check(model, batch_size)
    if wall_points < 2:
        raise ValueError("At least two wall points are required")
    position = torch.linspace(0, 4, wall_points, device=model.reference.device,
                              dtype=model.reference.dtype)
    points = torch.stack((position, torch.zeros_like(position)), dim=1)
    normalizers = _normalizers(model)
    predicted, exact = [], []
    for batch in points.split(batch_size):
        x = coordinates(batch)
        value = model(x, normalizers=normalizers)
        predicted.append(model.problem.wall_shear_stress(value, x).detach())
        exact.append(model.problem.wall_shear_stress(model.problem.exact(x), x).detach())
    return {"x": position.detach(), "wss_prediction": torch.cat(predicted).flatten(),
            "wss_reference": torch.cat(exact).flatten()}


@torch.enable_grad()
def hemodynamics_metrics(model, points, wall_points=121, batch_size=256):
    """Evaluate field, lower-wall shear and endpoint pressure-drop errors.

    Pressure is centered separately in the prediction and reference over
    *all* supplied evaluation points. ``relative_l2`` retains the usual
    aggregate-vector norm, while ``combined_field_relative_l2`` is the mean
    of the three component relative errors used by the manuscript.

    The caller supplies the evaluation points. Use
    ``Problem.hemodynamics_grid(121, 61)`` for the manuscript's grid.
    """
    _check(model, batch_size)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        raise ValueError("points must be a nonempty (N, 2) coordinate tensor")
    normalizers = _normalizers(model)
    predictions, exact_values = [], []
    for batch in points.split(batch_size):
        x = coordinates(batch)
        predictions.append(model(x, normalizers=normalizers).detach())
        exact_values.append(model.problem.exact(x).detach())
    prediction, exact = torch.cat(predictions), torch.cat(exact_values)
    prediction[:, 2] -= prediction[:, 2].mean()
    exact[:, 2] -= exact[:, 2].mean()
    numerator = (prediction - exact).square().sum(dim=0)
    denominator = exact.square().sum(dim=0)
    tiny = torch.finfo(points.dtype).tiny
    channels = (numerator / denominator.clamp_min(tiny)).sqrt()
    wall = hemodynamics_wall_fields(model, wall_points=wall_points, batch_size=batch_size)
    wss_error = torch.linalg.vector_norm(wall["wss_prediction"] - wall["wss_reference"])
    wss_error /= torch.linalg.vector_norm(wall["wss_reference"]).clamp_min(tiny)
    endpoints = coordinates(points.new_tensor([[0.0, 0.5], [4.0, 0.5]]))
    predicted_pressure = model(endpoints, normalizers=normalizers)[:, 2].detach()
    exact_pressure = model.problem.exact(endpoints)[:, 2].detach()
    pressure_drop = predicted_pressure[0] - predicted_pressure[1]
    reference_drop = exact_pressure[0] - exact_pressure[1]
    return {
        "relative_l2": float((numerator.sum() / denominator.sum().clamp_min(tiny)).sqrt()),
        "relative_l2_channels": channels.cpu().tolist(),
        "combined_field_relative_l2": float(channels.mean()),
        "wss_relative_l2": float(wss_error),
        "pressure_drop": float(pressure_drop),
        "pressure_drop_reference": float(reference_drop),
        "pressure_drop_relative_error": float((pressure_drop - reference_drop).abs() / reference_drop.abs()),
    }
