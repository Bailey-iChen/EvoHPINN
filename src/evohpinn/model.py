from dataclasses import dataclass, asdict
import math
import torch
from torch import nn

from .autodiff import coordinates


@dataclass(frozen=True)
class Architecture:
    depth: int
    width: int
    activation: str = "tanh"

    def __post_init__(self):
        if self.depth < 1 or self.width < 1 or self.activation not in {"tanh", "sin", "softplus"}:
            raise ValueError(f"Invalid architecture: {self}")

    def to_dict(self):
        return asdict(self)


class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class SmoothSoftplus(nn.Module):
    def forward(self, x):
        # logaddexp avoids Softplus's piecewise linear threshold for high-order AD.
        return torch.logaddexp(x, torch.zeros_like(x))


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, architecture):
        super().__init__()
        activation = {"tanh": nn.Tanh, "sin": Sin, "softplus": SmoothSoftplus}[architecture.activation]
        layers = []
        width_in = input_dim
        for _ in range(architecture.depth):
            layers.extend((nn.Linear(width_in, architecture.width), activation()))
            width_in = architecture.width
        layers.append(nn.Linear(width_in, output_dim))
        self.layers = nn.Sequential(*layers).double()
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Explicit Kaiming choice; the manuscript gives no gain/mode.
                nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="linear")
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.layers(x)


class EvoHPINN(nn.Module):
    def __init__(self, problem, architectures, reference, alpha=(1.0, 0.1),
                 gate="residual", normalization_epsilon=1e-8, norm_epsilon=1e-12):
        super().__init__()
        if len(architectures) != len(alpha) or not all(a > 0 for a in alpha):
            raise ValueError("Each subnetwork needs a positive capacity coefficient")
        if gate not in {"residual", "uniform"}:
            raise ValueError("gate must be residual or uniform")
        if normalization_epsilon <= 0 or norm_epsilon <= 0:
            raise ValueError("Residual regularizers must be positive")
        self.problem = problem
        self.architectures = list(architectures)
        self.gate = gate
        self.normalization_epsilon = normalization_epsilon
        self.norm_epsilon = norm_epsilon
        self.networks = nn.ModuleList([MLP(problem.input_dim, problem.output_dim, a) for a in architectures])
        self.register_buffer("reference", reference.detach().clone().double())
        self.register_buffer("alpha", reference.new_tensor(alpha).double())
        if problem.name == "inverse_burgers":
            self.log_viscosity = nn.Parameter(reference.new_tensor(math.log(0.05)).double())
        else:
            self.register_parameter("log_viscosity", None)

    @property
    def viscosity(self):
        return None if self.log_viscosity is None else self.log_viscosity.exp()

    def magnitudes(self, outputs, x):
        return torch.cat([
            torch.sqrt(self.problem.residual(y, x, self.viscosity).square().sum(dim=1, keepdim=True) + self.norm_epsilon**2)
            for y in outputs
        ], dim=1)

    def normalizers(self):
        if self.gate == "uniform":
            return None
        z = coordinates(self.reference)
        values = [network(z) for network in self.networks]
        # Independent z makes the denominator constant with respect to query x,
        # while retaining gradients with respect to weights and viscosity.
        return self.magnitudes(values, z).mean(dim=0, keepdim=True) + self.normalization_epsilon

    def components(self, x, normalizers=None, need_residuals=False):
        if not x.requires_grad and (self.gate == "residual" or need_residuals):
            raise ValueError("Residual gating requires coordinates with requires_grad=True")
        values = [network(x) for network in self.networks]
        residuals = self.magnitudes(values, x) if self.gate == "residual" or need_residuals else None
        if self.gate == "uniform":
            weights = x.new_full((len(x), len(values)), 1 / len(values))
        else:
            if normalizers is None:
                normalizers = self.normalizers()
            # Positive sign follows the manuscript's gate definition.
            weights = torch.softmax(self.alpha * residuals / normalizers, dim=1)
        fused = (torch.stack(values, dim=1) * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights, residuals

    def forward(self, x, normalizers=None):
        return self.components(x, normalizers)[0]

    def score(self, validation, penalty, evaluation="gated"):
        if evaluation not in {"gated", "uniform"}:
            raise ValueError("evaluation must be gated or uniform")
        x = coordinates(validation)
        _, weights, residuals = self.components(x, need_residuals=True)
        if evaluation == "uniform":
            weights = torch.full_like(weights, 1 / len(self.networks))
        residual_score = (weights * residuals).sum(dim=1).mean()
        cost = sum(penalty["depth"] * a.depth + penalty["width"] * a.width for a in self.architectures)
        total = residual_score + penalty["lambda"] * cost
        return {"score": float(total.detach()), "weighted_residual": float(residual_score.detach()),
                "complexity_penalty": penalty["lambda"] * cost}


def relative_errors(model, points, batch_size=256):
    """Diagnostic only. AD remains enabled because prediction uses PDE residuals."""
    if model.problem.name == "hemodynamics":
        from .metrics import hemodynamics_metrics
        return hemodynamics_metrics(model, points, batch_size=batch_size)
    normalizers = model.normalizers()
    if normalizers is not None:
        normalizers = normalizers.detach()
    numerator = points.new_zeros(model.problem.output_dim)
    denominator = points.new_zeros(model.problem.output_dim)
    for batch in points.split(batch_size):
        x = coordinates(batch)
        prediction = model(x, normalizers=normalizers).detach()
        exact = model.problem.exact(x).detach()
        numerator += (prediction - exact).square().sum(dim=0)
        denominator += exact.square().sum(dim=0)
    errors = (numerator / denominator.clamp_min(torch.finfo(points.dtype).tiny)).sqrt()
    result = {"relative_l2": float((numerator.sum() / denominator.sum()).sqrt()),
              "relative_l2_channels": errors.cpu().tolist()}
    if model.viscosity is not None:
        nu = float(model.viscosity.detach())
        result.update(viscosity=nu, viscosity_relative_error=abs(nu - model.problem.viscosity) / model.problem.viscosity)
    return result
