# Paper results

This directory contains results reported in the manuscript and its original figure files.

| File | Purpose |
| --- | --- |
| [forward_benchmarks.csv](forward_benchmarks.csv) | Errors and training times for four forward PDE benchmarks. |
| [inverse_burgers.csv](inverse_burgers.csv) | Field reconstruction errors, viscosity estimates and training times for inverse Burgers. |
| [hemodynamic_metrics.csv](hemodynamic_metrics.csv) | Combined-field, wall-shear-stress and pressure-drop errors, with training times. |
| [hemodynamic_components.csv](hemodynamic_components.csv) | Errors for velocity components `u`, `v` and pressure `p`. |
| [figures/inverse_fields.pdf](figures/inverse_fields.pdf) | Inverse Burgers reference, reconstruction error and gating weights at `t = 0.5`. |
| [figures/flow_velocity.pdf](figures/flow_velocity.pdf) | Predicted velocity magnitude and streamlines for the hemodynamic benchmark. |
| [figures/wss_curve.pdf](figures/wss_curve.pdf) | Predicted and reference wall shear stress. |
| [figures/pressure_curve.pdf](figures/pressure_curve.pdf) | Predicted and reference centerline pressure. |
| [provenance.json](provenance.json) | Source table labels, original TeX cells and file hashes. |
| [export_from_manuscript.py](export_from_manuscript.py) | Exports these tables and figures from the manuscript source and verifies file hashes. |

Statistics cover 10 runs: `_mean` and `_std` denote mean and standard deviation, while hemodynamic tables report means only; timings are in seconds, and `viscosity_relative_error_percent_reported = 1.23` means `1.23%`.

Verify the files from the repository root:

```bash
python results/paper/export_from_manuscript.py --verify
```
