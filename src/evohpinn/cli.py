import argparse
import hashlib
import json
import platform
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from . import __version__
from . import config as configuration
from .evolution import make_model, search
from .model import Architecture, relative_errors
from .problems import Problem


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def configure(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(config["threads"])
    if config["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.use_deterministic_algorithms(True)


def run(args):
    config = configuration.load(args.config, problem=args.problem, case=args.case,
                                seed=args.seed, device=args.device)
    configure(config)
    output = Path(args.output)
    # Never mix a new experiment with old output or overwrite a checkpoint.
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "config.json", config)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    dump(output / "environment.json", {"python": platform.python_version(), "torch": str(torch.__version__),
          "numpy": np.__version__, "package": __version__, "platform": platform.system(),
          "machine": platform.machine(), "device": config["device"], "dtype": "float64",
          "source_sha256": source_hashes})
    problem = Problem(config["problem"], config["case"])
    data = {k: v.to(config["device"]) for k, v in problem.sample(config["counts"], config["seed"]).items()}
    torch.save({k: v.cpu() for k, v in data.items()}, output / "samples.pt")
    started = time.monotonic()
    with (output / "events.jsonl").open("w") as stream:
        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            if record["event"] == "generation":
                print(f"generation={record['generation']} J={record['score']:.6g} relative_L2={record['relative_l2']:.6g}", flush=True)

        def save(model, generation):
            torch.save({"format_version": 1, "config": config, "generation": generation,
                        "architectures": [a.to_dict() for a in model.architectures],
                        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}},
                       output / f"generation_{generation:03d}.pt")

        best, history = search(problem, data, config, emit=emit, save_generation=save)
    summary = {"status": config["status"], "seconds": time.monotonic() - started,
               "selected": history[-1], "history": history}
    dump(output / "summary.json", summary)
    print(f"Saved run to {output}", flush=True)


def read_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["format_version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    config = checkpoint["config"]
    config["device"] = device
    configuration.validate(config)
    configure(config)
    problem = Problem(config["problem"], config["case"])
    model = make_model(problem, [Architecture(**a) for a in checkpoint["architectures"]],
                       {"reference": checkpoint["state_dict"]["reference"]}, config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config


def evaluate(args):
    model, config = read_checkpoint(args.checkpoint, args.device)
    if args.samples:
        data = torch.load(args.samples, map_location=args.device, weights_only=True)
        points = data["test"]
    else:
        generator = torch.Generator().manual_seed(args.seed)
        points = model.problem.interior(args.points, generator).to(args.device)
    print(json.dumps(relative_errors(model, points), indent=2, allow_nan=False))


def export_application(args):
    from .export import export_fields
    model, _ = read_checkpoint(args.checkpoint, args.device)
    result = export_fields(model, args.checkpoint, args.output, grid_x=args.grid_x,
                           grid_y=args.grid_y, time=args.time, wall_points=args.wall_points,
                           batch_size=args.batch_size)
    print(json.dumps(result, indent=2, allow_nan=False))


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(args.run)
    rows = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
    generations = [r for r in rows if r["event"] == "generation"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for axis, key, label in zip(axes, ("score", "relative_l2"), ("Selected architecture score J", "Selected relative L2 error")):
        axis.plot([r["candidate_evaluations"] for r in generations], [r[key] for r in generations], "o-")
        axis.set(xlabel="Evaluated candidates", ylabel=label, yscale="log")
        axis.grid(alpha=0.25)
    fig.suptitle("EvoHPINN search diagnostics: current run")
    fig.tight_layout()
    destination = root / "search_progress.png"
    fig.savefig(destination, dpi=180)
    plt.close(fig)
    print(destination)


def main(argv=None):
    parser = argparse.ArgumentParser(description="EvoHPINN training, architecture search and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("run", help="Train and search; output directory must not already exist")
    train.add_argument("--config")
    train.add_argument("--problem", choices=["klein_gordon", "burgers", "lame", "navier_stokes", "inverse_burgers", "hemodynamics"])
    train.add_argument("--case", type=int, choices=[1, 2, 3])
    train.add_argument("--seed", type=int)
    train.add_argument("--device")
    train.add_argument("--output", required=True)
    train.set_defaults(function=run)
    test = sub.add_parser("evaluate", help="Evaluate a saved generation without changing its parameters")
    test.add_argument("checkpoint")
    test.add_argument("--samples", help="Use test points from the run's samples.pt")
    test.add_argument("--points", type=int, default=2048)
    test.add_argument("--seed", type=int, default=10000)
    test.add_argument("--device", default="cpu")
    test.set_defaults(function=evaluate)
    export = sub.add_parser("export", help="Export inverse or hemodynamic checkpoint fields, gates and metrics as CSV/JSON")
    export.add_argument("checkpoint")
    export.add_argument("--output", required=True, help="New directory for the exported files")
    export.add_argument("--device", default="cpu")
    export.add_argument("--grid-x", type=int, default=121)
    export.add_argument("--grid-y", type=int, default=61)
    export.add_argument("--time", type=float, default=0.5, help="Time slice for inverse Burgers")
    export.add_argument("--wall-points", type=int, default=121)
    export.add_argument("--batch-size", type=int, default=256)
    export.set_defaults(function=export_application)
    drawing = sub.add_parser("plot", help="Plot actual generation diagnostics from a run")
    drawing.add_argument("run")
    drawing.set_defaults(function=plot)
    args = parser.parse_args(argv)
    args.function(args)


if __name__ == "__main__":
    main()
