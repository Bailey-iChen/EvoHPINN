# EvoHPINN

Evolutionary architecture search with residual-guided gating for heterogeneous physics-informed neural networks.

## Method

EvoHPINN couples evolutionary subnetwork architecture search with residual-guided prediction fusion. The outer search selects each subnetwork's depth, width and activation, while inner optimization jointly trains the subnetworks using the physics-informed loss of their combined prediction. The same spatial gating weights combine predictions and weight subnetwork residual magnitudes for candidate evaluation, together with a structural complexity penalty. This links architecture selection to each subnetwork's spatial contribution without predefined domain partitions or an additional trainable gating network.

## Source files

| File | Purpose |
| --- | --- |
| [problems.py](src/evohpinn/problems.py) | PDEs, reference solutions, sampling and boundary/initial/observation losses |
| [model.py](src/evohpinn/model.py) | Subnetworks, residual gates, prediction fusion and candidate scoring |
| [evolution.py](src/evohpinn/evolution.py) | Random initialization, selection, crossover, mutation and elitism |
| [training.py](src/evohpinn/training.py) | Joint training with Adam and L-BFGS |
| [autodiff.py](src/evohpinn/autodiff.py) | Coordinate derivatives |
| [metrics.py](src/evohpinn/metrics.py) | Flow-field, wall shear stress and pressure-drop errors |
| [export.py](src/evohpinn/export.py) | Export checkpoint predictions, observations and gating weights |
| [config.py](src/evohpinn/config.py) | Configuration defaults and validation |
| [cli.py](src/evohpinn/cli.py) | Training, evaluation, export and plotting commands |
| [__init__.py](src/evohpinn/__init__.py) | Package version |

## Paper results

`results/paper/` contains manuscript-reported results.

| File | Contents |
| --- | --- |
| [forward_benchmarks.csv](results/paper/forward_benchmarks.csv) | Four forward benchmark comparisons |
| [inverse_burgers.csv](results/paper/inverse_burgers.csv) | Reconstruction and viscosity-identification results |
| [hemodynamic_metrics.csv](results/paper/hemodynamic_metrics.csv) | Combined-field, WSS and pressure-drop results |
| [hemodynamic_components.csv](results/paper/hemodynamic_components.csv) | Velocity-component and pressure errors |
| [inverse_fields.pdf](results/paper/figures/inverse_fields.pdf) | Inverse reference field, reconstruction error and gating maps |
| [flow_velocity.pdf](results/paper/figures/flow_velocity.pdf) | Flow velocity magnitude and streamlines |
| [wss_curve.pdf](results/paper/figures/wss_curve.pdf) | Wall shear stress profiles |
| [pressure_curve.pdf](results/paper/figures/pressure_curve.pdf) | Centerline pressure profiles |
| [results/paper/README.md](results/paper/README.md) | Result-file descriptions and units |
| [provenance.json](results/paper/provenance.json) | Table labels, source values and file hashes |
| [export_from_manuscript.py](results/paper/export_from_manuscript.py) | Extract and verify the reported result files |

## Package files

| File | Purpose |
| --- | --- |
| [pyproject.toml](pyproject.toml) | Package installation and dependencies |
| [requirements.txt](requirements.txt) | Dependency versions |
| [.gitignore](.gitignore) | Exclude local runs and temporary files |
| [LICENSE](LICENSE) | MIT license for the code and software documentation |

## Installation

Requires Python 3.10+ and float64 support.

```bash
python -m pip install -e '.[plot]'
evohpinn --help
```

## License

The source code is released under the MIT License. See [LICENSE](LICENSE).
