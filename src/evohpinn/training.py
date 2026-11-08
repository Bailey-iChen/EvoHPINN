import math
import time
import torch

from .problems import loss_terms


def objective(model, data, weights):
    terms = loss_terms(model, data)
    loss = sum(weights[name] * value for name, value in terms.items())
    return loss, terms


def train(model, data, config, log=None):
    """Each call is one candidate's inner training budget in one generation.

    L-BFGS history is recreated each generation; inherited network parameters
    are retained. Closure evaluations are recorded separately from updates.
    """
    evaluations = 0
    updates = 0
    started = time.monotonic()
    model.train()
    parameters = list(model.parameters())
    stages = [
        ("adam", config["adam_steps"], lambda: torch.optim.Adam(parameters, lr=config["adam_lr"])),
        ("lbfgs", config["lbfgs_steps"], lambda: torch.optim.LBFGS(
            parameters, lr=1.0, max_iter=config["lbfgs_max_iter"],
            history_size=config["lbfgs_history"], line_search_fn="strong_wolfe",
            tolerance_grad=1e-9, tolerance_change=1e-12)),
    ]
    for stage, steps, factory in stages:
        if steps == 0:
            continue
        optimizer = factory()
        for step in range(steps):
            def closure():
                nonlocal evaluations
                optimizer.zero_grad(set_to_none=True)
                value, _ = objective(model, data, config["weights"])
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite training loss; no candidate score was accepted")
                value.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError("Non-finite gradients; check architecture, scale and regularization")
                evaluations += 1
                return value

            if stage == "adam":
                closure()
                optimizer.step()
            else:
                optimizer.step(closure)
            updates += 1
            if log is not None:
                value, terms = objective(model, data, config["weights"])
                log({"stage": stage, "stage_step": step + 1, "updates": updates,
                     "closure_evaluations": evaluations, "loss": float(value.detach()),
                     "terms": {key: float(v.detach()) for key, v in terms.items()}})
    value, terms = objective(model, data, config["weights"])
    result = {"loss": float(value.detach()), "terms": {key: float(v.detach()) for key, v in terms.items()},
              "updates": updates, "closure_evaluations": evaluations, "seconds": time.monotonic() - started}
    if not math.isfinite(result["loss"]):
        raise FloatingPointError("Non-finite final loss")
    return result
