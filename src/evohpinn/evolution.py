"""Seeded architecture search; selection never reads reference L2 errors."""

import copy
import math
import random
import torch

from .model import Architecture, EvoHPINN, relative_errors
from .training import train


def random_architecture(space, rng):
    depths, widths = space["depths"], space["widths"]
    return Architecture(rng.choice(depths), rng.choice(widths), rng.choice(space["activations"]))


def mutate(architecture, space, rng, rates):
    def change(value, choices, probability):
        alternatives = [x for x in choices if x != value]
        return rng.choice(alternatives) if alternatives and rng.random() < probability else value
    return Architecture(change(architecture.depth, space["depths"], rates["depth"]),
                        change(architecture.width, space["widths"], rates["width"]),
                        change(architecture.activation, space["activations"], rates["activation"]))


def make_model(problem, architectures, data, config):
    return EvoHPINN(problem, architectures, data["reference"], **config["model"]).to(data["reference"].device)


def child_model(parent, other, problem, data, config, rng):
    settings = config["search"]
    architectures = list(parent.architectures)
    donors = [parent] * len(architectures)
    if rng.random() < settings["crossover_mutation_split"]:
        if rng.random() < settings["crossover_probability"]:
            for k in range(len(architectures)):
                if rng.random() < 0.5:
                    architectures[k], donors[k] = other.architectures[k], other
    else:
        k = rng.randrange(len(architectures))
        architectures[k] = mutate(architectures[k], settings["space"], rng, settings["mutation"])
    child = make_model(problem, architectures, data, config)
    # Only complete, architecture-identical networks are inherited. No arbitrary
    # overlapping tensor slicing when width/depth changes.
    for k, (architecture, donor) in enumerate(zip(architectures, donors)):
        if architecture == donor.architectures[k]:
            child.networks[k].load_state_dict(donor.networks[k].state_dict())
    if child.log_viscosity is not None:
        with torch.no_grad():
            child.log_viscosity.copy_(parent.log_viscosity)
    return child


def search(problem, data, config, emit=None, save_generation=None):
    rng = random.Random(config["seed"])
    settings = config["search"]
    population = []
    for _ in range(settings["population"]):
        architectures = [random_architecture(settings["space"], rng)
                         for _ in config["model"]["alpha"]]
        population.append(make_model(problem, architectures, data, config))
    history = []
    for generation in range(1, settings["generations"] + 1):
        rows = []
        for index, model in enumerate(population):
            def report(record):
                if emit:
                    emit({"event": "training", "generation": generation, "candidate": index, **record})
            trained = train(model, data, config["training"], log=report if config["log_steps"] else None)
            score = model.score(data["validation"], config["penalty"], config["evaluation"])
            if not math.isfinite(score["score"]):
                raise FloatingPointError("Non-finite architecture score")
            row = {"event": "candidate", "generation": generation, "candidate": index,
                   "architectures": [a.to_dict() for a in model.architectures], **score, "training": trained}
            rows.append(row)
            if emit:
                emit(row)
        ranking = sorted(range(len(population)), key=lambda i: rows[i]["score"])
        best = population[ranking[0]]
        # Test error is measured only AFTER ranking by J.
        summary = {"event": "generation", "generation": generation,
                   "candidate_evaluations": generation * len(population),
                   "selected_candidate": ranking[0], "score": rows[ranking[0]]["score"],
                   "architectures": [a.to_dict() for a in best.architectures],
                   **relative_errors(best, data["test"])}
        history.append(summary)
        if emit:
            emit(summary)
        if save_generation:
            save_generation(best, generation)
        if generation == settings["generations"]:
            return best, history
        elite_count = max(1, math.ceil(settings["elite_ratio"] * len(population)))
        following = [copy.deepcopy(population[i]) for i in ranking[:elite_count]]

        def select():
            contestants = rng.sample(range(len(population)), min(3, len(population)))
            return population[min(contestants, key=lambda i: rows[i]["score"])]

        while len(following) < len(population):
            following.append(child_model(select(), select(), problem, data, config, rng))
        population = following
    raise ValueError("At least one generation is required")
