"""PDEs, analytical references and seeded sampling from the manuscript.

Exact solutions generate forcing/constraint data and diagnostic errors only.
They are never used to rank candidate architectures.
"""

from dataclasses import dataclass
import math
import torch

from .autodiff import coordinates, partial, second


@dataclass(frozen=True)
class Problem:
    name: str
    case: int = 1

    def __post_init__(self):
        if self.name not in {"klein_gordon", "burgers", "lame", "navier_stokes", "inverse_burgers", "hemodynamics"}:
            raise ValueError(f"Unknown problem: {self.name}")
        if self.case not in {1, 2, 3} or (self.name not in {"klein_gordon", "burgers"} and self.case != 1):
            raise ValueError("Cases 2 and 3 are defined only for Klein-Gordon and forward Burgers")

    @property
    def input_dim(self):
        return 3 if self.name in {"burgers", "inverse_burgers"} else 2

    @property
    def output_dim(self):
        return {"lame": 2, "navier_stokes": 3, "hemodynamics": 3}.get(self.name, 1)

    @property
    def time_dependent(self):
        return self.name in {"klein_gordon", "burgers", "inverse_burgers"}

    @property
    def viscosity(self):
        if self.name in {"inverse_burgers", "hemodynamics"}:
            return 0.01
        if self.name == "navier_stokes":
            return 1 / 40
        return {1: 0.1, 2: 0.15, 3: 0.05}[self.case]

    def exact(self, x):
        a, b = x[:, 0:1], x[:, 1:2]
        if self.name == "klein_gordon":
            power, amplitude, omega = self.kg_parameters()
            return a**power * torch.cos(omega * b) + amplitude * (a * b)**3
        if self.name == "burgers":
            t = x[:, 2:3]
            return torch.sigmoid(-0.5 * (a + b - t) / self.viscosity)
        if self.name == "inverse_burgers":
            t = x[:, 2:3]
            bump = torch.exp(-((a - 0.7)**2 + (b - 0.7)**2) / 0.02)
            return torch.sin(math.pi * a) * torch.sin(math.pi * b) * torch.cos(math.pi * t) + 0.3 * bump * torch.sin(math.pi * t)
        if self.name == "hemodynamics":
            modulation = torch.exp(-((a - 2.0) / 0.3)**2)
            stream_y = 6 * b * (1 - b) * (1 + 0.6 * modulation)
            stream_x = (3 * b**2 - 2 * b**3) * 0.6 * (-2 * (a - 2.0) / 0.3**2) * modulation
            return torch.cat((stream_y, -stream_x, -0.2 * a + 0.18 * modulation), dim=1)
        if self.name == "lame":
            e, mu, q1, q2, inner, outer = 2.1, 0.25, 23.0, 3.0, 1.0, 2.0
            aa = q1 * (1 - mu**2) * outer**2 / (e * (outer**2 * (1 + mu) + inner**2 * (1 - mu)))
            bb = q2 * (1 + mu) * outer**2 / (e * inner**2)
            ratio = inner**2 / (a**2 + b**2)
            return torch.cat((aa * (ratio - 1) * a + bb * (1 - ratio) * b,
                              aa * (ratio - 1) * b - bb * (1 - ratio) * a), dim=1)
        lam = 20.0 - math.sqrt(400.0 + 4 * math.pi**2)
        exp = torch.exp(lam * a)
        return torch.cat((1 - exp * torch.cos(2 * math.pi * b),
                          lam / (2 * math.pi) * exp * torch.sin(2 * math.pi * b),
                          0.5 * (1 - exp**2)), dim=1)

    def kg_parameters(self):
        return (2 if self.case == 3 else 1, 1.2 if self.case == 3 else 1.0,
                (4 if self.case == 2 else 5) * math.pi)

    def forcing(self, x):
        if self.name == "klein_gordon":
            a, t = x[:, 0:1], x[:, 1:2]
            power, c, omega = self.kg_parameters()
            utt = -omega**2 * a**power * torch.cos(omega * t) + 6 * c * a**3 * t
            uxx = (2 * torch.cos(omega * t) if power == 2 else 0.0) + 6 * c * a * t**3
            return utt - uxx + self.exact(x)**3
        if self.name == "inverse_burgers":
            # Keep the coordinate graph: the gate's derivatives include f_x.
            return self.burgers_operator(self.exact(x), x, self.viscosity)
        if self.name == "hemodynamics":
            # Forcing retains its coordinate graph because residual gates are
            # differentiated when evaluating the fused prediction's PDE loss.
            return self.navier_stokes_operator(self.exact(x), x, self.viscosity)[:, :2]
        return x[:, :1] * 0.0

    @staticmethod
    def burgers_operator(u, x, viscosity):
        return partial(u, x, 2) + u * (partial(u, x, 0) + partial(u, x, 1)) - viscosity * (second(u, x, 0) + second(u, x, 1))

    @staticmethod
    def navier_stokes_operator(prediction, x, viscosity):
        u, v, p = prediction[:, :1], prediction[:, 1:2], prediction[:, 2:3]
        return torch.cat((
            -viscosity * (second(u, x, 0) + second(u, x, 1)) + u * partial(u, x, 0) + v * partial(u, x, 1) + partial(p, x, 0),
            -viscosity * (second(v, x, 0) + second(v, x, 1)) + u * partial(v, x, 0) + v * partial(v, x, 1) + partial(p, x, 1),
            partial(u, x, 0) + partial(v, x, 1),
        ), dim=1)

    @staticmethod
    def hemodynamics_streamfunction(x):
        a, b = x[:, :1], x[:, 1:2]
        return (3 * b**2 - 2 * b**3) * (1 + 0.6 * torch.exp(-((a - 2.0) / 0.3)**2))

    @staticmethod
    def hemodynamics_grid(nx=121, ny=61):
        """Uniform evaluation grid; defaults follow the manuscript protocol."""
        if nx < 2 or ny < 2:
            raise ValueError("Each grid axis needs at least two points")
        xx, yy = torch.meshgrid(torch.linspace(0, 4, nx, dtype=torch.float64),
                                torch.linspace(0, 1, ny, dtype=torch.float64), indexing="ij")
        return torch.stack((xx.flatten(), yy.flatten()), dim=1)

    def wall_shear_stress(self, prediction, x):
        """Signed lower-wall quantity nu * du/dy defined in the manuscript."""
        if self.name != "hemodynamics":
            raise ValueError("Wall-shear-stress diagnostic is defined for hemodynamics")
        return self.viscosity * partial(prediction[:, :1], x, 1)

    def residual(self, prediction, x, viscosity=None):
        u = prediction[:, :1]
        if self.name == "klein_gordon":
            return second(u, x, 1) - second(u, x, 0) + u**3 - self.forcing(x)
        if self.name in {"burgers", "inverse_burgers"}:
            nu = self.viscosity if viscosity is None else viscosity
            return self.burgers_operator(u, x, nu) - self.forcing(x)
        v = prediction[:, 1:2]
        if self.name == "lame":
            mu, scale = 0.25, 2.1 / (1 - 0.25**2)
            return scale * torch.cat((
                second(u, x, 0) + (1 - mu) / 2 * second(u, x, 1) + (1 + mu) / 2 * partial(partial(v, x, 0), x, 1),
                second(v, x, 1) + (1 - mu) / 2 * second(v, x, 0) + (1 + mu) / 2 * partial(partial(u, x, 0), x, 1),
            ), dim=1)
        result = self.navier_stokes_operator(prediction, x, self.viscosity)
        if self.name == "hemodynamics":
            result = torch.cat((result[:, :2] - self.forcing(x), result[:, 2:3]), dim=1)
        return result

    def traction(self, prediction, x):
        u, v = prediction[:, :1], prediction[:, 1:2]
        mu, scale = 0.25, 2.1 / (1 - 0.25**2)
        ux, uy = partial(u, x, 0), partial(u, x, 1)
        vx, vy = partial(v, x, 0), partial(v, x, 1)
        xx, yy, xy = scale * (ux + mu * vy), scale * (vy + mu * ux), scale * (1 - mu) / 2 * (uy + vx)
        normal = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        nx, ny = normal[:, :1], normal[:, 1:2]
        return torch.cat((xx * nx + xy * ny, yy * ny + xy * nx), dim=1)

    @staticmethod
    def traction_target(x):
        normal = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        nx, ny = normal[:, :1], normal[:, 1:2]
        return torch.cat((-23 * nx + 3 * ny, -23 * ny - 3 * nx), dim=1)

    @staticmethod
    def rand(n, dim, generator):
        return torch.rand((n, dim), generator=generator, dtype=torch.float64)

    def interior(self, n, generator, near_fraction=0.0):
        x = self.rand(n, self.input_dim, generator)
        if self.name == "lame":
            radius, angle = torch.sqrt(1 + 3 * x[:, :1]), 2 * math.pi * x[:, 1:2]
            return torch.cat((radius * torch.cos(angle), radius * torch.sin(angle)), dim=1)
        if self.name == "navier_stokes":
            return x * x.new_tensor([1.5, 2.0]) + x.new_tensor([-0.5, -0.5])
        if self.name == "hemodynamics":
            x[:, 0] *= 4.0
            return x
        if self.name == "burgers":
            # Three equal-area squares give uniform measure on the L domain.
            cell = torch.randint(3, (n,), generator=generator)
            x[:, :2] *= 0.5
            x[:, 0] += (cell > 0).to(x.dtype) * 0.5
            x[:, 1] += (cell == 2).to(x.dtype) * 0.5
            x[:, 2] *= 2.0
        elif self.name == "inverse_burgers" and near_fraction:
            count = round(n * near_fraction)
            x[:count, :2] = 0.4 + 0.6 * x[:count, :2]
            x = x[torch.randperm(n, generator=generator)]
        return x

    def hemodynamics_collocation(self, n, generator):
        """50% uniform, 35% central strip and 15% near-wall mixture.

        Rounding affects only small counts. At the manuscript's 8000 points,
        the allocation is exactly 4000/2800/1200, with 600 at either wall.
        """
        x = self.interior(n, generator)
        uniform_count, central_count = round(n * 0.5), round(n * 0.35)
        wall_start = uniform_count + central_count
        x[uniform_count:wall_start, 0] = 1.7 + 0.6 * self.rand(central_count, 1, generator).flatten()
        wall_count = n - wall_start
        lower_count = wall_count // 2
        wall_y = 0.1 * self.rand(wall_count, 1, generator).flatten()
        wall_y[lower_count:] += 0.9
        x[wall_start:, 1] = wall_y
        return x[torch.randperm(n, generator=generator)]

    def boundary(self, n, generator):
        if self.name == "lame":
            theta = 2 * math.pi * self.rand(n, 1, generator)
            points = torch.cat((theta.cos(), theta.sin()), dim=1)
            return {"inner": points[:n // 2], "outer": 2 * points[n // 2:]}
        x = self.interior(n, generator)
        if self.name == "klein_gordon":
            x[:, 0] = torch.randint(2, (n,), generator=generator).to(x.dtype)
            return {"boundary": x}
        if self.name == "hemodynamics":
            if n < 4 or n % 4:
                raise ValueError("Hemodynamics boundary count must be divisible by four")
            # Equal counts on each side, rather than length-weighted sampling.
            sides = torch.arange(4).repeat_interleave(n // 4)
            x[:, 0] = torch.where(sides == 0, 0.0, x[:, 0])
            x[:, 0] = torch.where(sides == 1, 4.0, x[:, 0])
            x[:, 1] = torch.where(sides == 2, 0.0, x[:, 1])
            x[:, 1] = torch.where(sides == 3, 1.0, x[:, 1])
            return {"boundary": x[torch.randperm(n, generator=generator)]}
        if self.name == "burgers":
            # Polygon perimeter: bottom, right, top-right, reentrant vertical,
            # reentrant horizontal, left-bottom. Length-weighted sampling.
            starts = x.new_tensor([[0, 0], [1, 0], [1, 1], [.5, 1], [.5, .5], [0, .5]])
            ends = x.new_tensor([[1, 0], [1, 1], [.5, 1], [.5, .5], [0, .5], [0, 0]])
        elif self.name == "navier_stokes":
            starts = x.new_tensor([[-.5, -.5], [1, -.5], [1, 1.5], [-.5, 1.5]])
            ends = x.new_tensor([[1, -.5], [1, 1.5], [-.5, 1.5], [-.5, -.5]])
        else:
            starts = x.new_tensor([[0, 0], [1, 0], [1, 1], [0, 1]])
            ends = x.new_tensor([[1, 0], [1, 1], [0, 1], [0, 0]])
        lengths = torch.linalg.vector_norm(ends - starts, dim=1)
        segment = torch.multinomial(lengths, n, replacement=True, generator=generator)
        x[:, :2] = starts[segment] + self.rand(n, 1, generator) * (ends - starts)[segment]
        return {"boundary": x}

    def sample(self, counts, seed):
        generator = torch.Generator().manual_seed(seed)
        data = {}
        for key in ("residual", "validation", "reference", "test"):
            near = 0.4 if self.name == "inverse_burgers" and key == "residual" else 0.0
            if self.name == "hemodynamics" and key == "residual":
                data[key] = self.hemodynamics_collocation(counts[key], generator)
            else:
                data[key] = self.interior(counts[key], generator, near)
        data.update(self.boundary(counts["boundary"], generator))
        if self.time_dependent:
            data["initial"] = self.interior(counts["initial"], generator)
            data["initial"][:, -1] = 0.0
        if self.name == "navier_stokes":
            # One pressure reference removes the additive pressure nullspace.
            data["gauge"] = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
        if self.name == "inverse_burgers":
            data["observations"] = self.interior(counts["observations"], generator, 0.3)
            noise = torch.randn((counts["observations"], 1), generator=generator, dtype=torch.float64)
            data["values"] = self.exact(data["observations"]).detach() + 0.01 * noise
        return data


def loss_terms(model, data):
    """The physics loss of the fused prediction, not the architecture score."""
    problem = model.problem
    normalizers = model.normalizers()

    def predict(key):
        x = coordinates(data[key])
        return x, model(x, normalizers=normalizers)

    def mse(error):
        # Vector residuals use squared Euclidean norm per point, then mean.
        return error.square().sum(dim=1).mean()

    x, y = predict("residual")
    terms = {"pde": mse(problem.residual(y, x, model.viscosity))}
    if problem.name == "lame":
        _, yi = predict("inner")
        xo, yo = predict("outer")
        ni, no = len(yi), len(yo)
        terms["boundary"] = (ni * mse(yi) + no * mse(problem.traction(yo, xo) - problem.traction_target(xo))) / (ni + no)
    else:
        xb, yb = predict("boundary")
        channels = {"navier_stokes": 2, "hemodynamics": 3}.get(problem.name, 1)
        terms["boundary"] = mse(yb[:, :channels] - problem.exact(xb)[:, :channels].detach())
    if problem.time_dependent:
        xi, yi = predict("initial")
        terms["initial"] = mse(yi - problem.exact(xi).detach())
        if problem.name == "klein_gordon":
            terms["initial"] = terms["initial"] + mse(partial(yi, xi, 1))
    if "gauge" in data:
        xg, yg = predict("gauge")
        terms["gauge"] = mse(yg[:, 2:3] - problem.exact(xg)[:, 2:3].detach())
    if "observations" in data:
        _, yd = predict("observations")
        terms["data"] = mse(yd - data["values"])
    return terms
