#!/usr/bin/env python3
"""Export selected reported tables and figures from the author's manuscript.

This reads TeX tables; it does not run models or recover individual-run data.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


TABLES = {
    "forward_benchmarks.csv": "tab:main_benchmark_results",
    "inverse_burgers.csv": "tab:inverse_burgers",
    "hemodynamic_metrics.csv": "tab:stenosis_results",
    "hemodynamic_components.csv": "tab:stenosis_components",
}
FIGURES = {
    "inverse_fields.pdf": ("fig:inverse_heatmap", "all panels"),
    "flow_velocity.pdf": ("fig:stenosis_setup_solution", "panel b"),
    "wss_curve.pdf": ("fig:stenosis_solution", "panel a"),
    "pressure_curve.pdf": ("fig:stenosis_solution", "panel b"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(tex: str) -> str:
    text = tex.strip().replace(r"\!", "")
    text = re.sub(r"\\times\s*10\^\{([+-]?\d+)\}", r"e\1", text)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\\(?:textbf|mathbf|text)\{([^{}]*)\}", r"\1", text)
    return text.replace("$", "").replace("{", "").replace("}", "").strip()


def numeric_values(tex: str) -> list[str]:
    return re.findall(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", plain(tex))


def comparison_symbol(tex: str) -> str:
    match = re.search(r"\((\+|--|\\approx)\)", tex)
    return match.group(1) if match else ""


def table_block(source: str, label: str) -> str:
    matches = [
        match.group(0)
        for match in re.finditer(r"\\begin\{(table\*?)\}.*?\\end\{\1\}", source, re.S)
        if rf"\label{{{label}}}" in match.group(0)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one table with label {label}; found {len(matches)}")
    return matches[0]


def body_rows(block: str) -> list[list[str]]:
    body = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    body = body.replace(r"\midrule", "")
    rows = [[cell.strip() for cell in row.split("&")] for row in body.split(r"\\") if row.strip()]
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export(source_path: Path, out: Path) -> None:
    source = source_path.read_text(encoding="utf-8")
    # Do not silently assume a run count if the manuscript protocol changes.
    if "mean $\\pm$ standard deviation over 10 runs" not in source:
        raise ValueError("Expected global 10-run reporting statement was not found")
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    provenance = {
        "description": "Derived from manuscript tables and copied manuscript figure assets; not rerun results.",
        "source_file": source_path.name,
        "source_sha256": sha256(source_path),
        "reported_run_count": 10,
        "run_count_source": "Global experiment reporting statement: mean +/- standard deviation over 10 runs unless stated otherwise; hemodynamic tables report means.",
        "tables": {},
        "figures": {},
    }
    exports = {}
    for name, label in TABLES.items():
        block = table_block(source, label)
        rows = body_rows(block)
        provenance["tables"][name] = {
            "tex_label": label,
            "table_block_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
            "source_start_line": source[:source.index(block)].count("\n") + 1,
            "source_rows": rows,
        }
        exports[name] = []
        if name == "forward_benchmarks.csv":
            if len(rows) != 8 or any(len(row) != 5 for row in rows):
                raise ValueError("Unexpected forward table dimensions")
            names = ["Klein-Gordon", "Burgers", "Lame", "Navier-Stokes"]
            methods = ["NASPINN", "XPINN", "PINN", "EvoHPINN"]
            for index, benchmark in enumerate(names):
                errors, times = rows[2 * index:2 * index + 2]
                for column, method in enumerate(methods, start=1):
                    mean, std = numeric_values(errors[column])
                    time, = numeric_values(times[column])
                    exports[name].append({
                        "benchmark": benchmark,
                        "method": method,
                        "relative_l2_mean": mean,
                        "relative_l2_std": std,
                        "reported_runs": 10,
                        "wall_clock_s_mean": time,
                        "comparison_symbol": comparison_symbol(errors[column]),
                        "source_tex_label": label,
                        "source_error_row_1based": 2 * index + 1,
                        "source_time_row_1based": 2 * index + 2,
                        "source_method_column_1based": column + 1,
                    })
        elif name == "inverse_burgers.csv":
            if len(rows) != 6 or any(len(row) != 5 for row in rows):
                raise ValueError("Unexpected inverse table dimensions")
            for index, cells in enumerate(rows, start=1):
                field_mean, field_std = numeric_values(cells[1])
                viscosity_mean, viscosity_std = numeric_values(cells[2])
                relative_error, = numeric_values(cells[3])
                time, = numeric_values(cells[4])
                exports[name].append({
                    "method": plain(cells[0]),
                    "field_relative_l2_mean": field_mean,
                    "field_relative_l2_std": field_std,
                    "viscosity_mean": viscosity_mean,
                    "viscosity_std": viscosity_std,
                    "viscosity_relative_error_percent_reported": relative_error,
                    "wall_clock_s_reported": time,
                    "reported_runs": 10,
                    "comparison_symbol": comparison_symbol(cells[1]),
                    "source_tex_label": label,
                    "source_row_1based": index,
                })
        elif name == "hemodynamic_metrics.csv":
            if len(rows) != 6 or any(len(row) != 5 for row in rows):
                raise ValueError("Unexpected hemodynamic main table dimensions")
            for index, cells in enumerate(rows, start=1):
                values = [numeric_values(cell) for cell in cells[1:]]
                if any(len(value) != 1 for value in values):
                    raise ValueError("Expected single reported values in hemodynamic table")
                combined, wss, pressure, time = [value[0] for value in values]
                exports[name].append({
                    "method": plain(cells[0]),
                    "combined_field_error_mean": combined,
                    "wss_relative_l2_mean": wss,
                    "pressure_drop_relative_absolute_error_mean": pressure,
                    "wall_clock_s_reported": time,
                    "reported_runs": 10,
                    "source_tex_label": label,
                    "source_row_1based": index,
                })
        else:
            if len(rows) != 6 or any(len(row) != 4 for row in rows):
                raise ValueError("Unexpected hemodynamic component table dimensions")
            for index, cells in enumerate(rows, start=1):
                values = [numeric_values(cell) for cell in cells[1:]]
                if any(len(value) != 1 for value in values):
                    raise ValueError("Expected single component means")
                u, v, p = [value[0] for value in values]
                exports[name].append({
                    "method": plain(cells[0]),
                    "u_relative_l2_mean": u,
                    "v_relative_l2_mean": v,
                    "p_relative_l2_mean": p,
                    "reported_runs": 10,
                    "source_tex_label": label,
                    "source_row_1based": index,
                })
        write_csv(out / name, exports[name])
        provenance["tables"][name]["csv_sha256"] = sha256(out / name)
        provenance["tables"][name]["row_count"] = len(exports[name])
    for name, (label, panel) in FIGURES.items():
        original = source_path.parent / name
        if rf"\label{{{label}}}" not in source or f"{{{name}}}" not in source:
            raise ValueError(f"Missing manuscript figure reference: {name}")
        destination = out / "figures" / name
        shutil.copyfile(original, destination)
        provenance["figures"][f"figures/{name}"] = {
            "source_file": name,
            "tex_label": label,
            "panel": panel,
            "sha256": sha256(original),
            "bytes": original.stat().st_size,
        }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"csv_rows": {name: len(rows) for name, rows in exports.items()}, "figures": len(FIGURES)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manuscript", type=Path, nargs="?", help="Path to the source manuscript TeX; figures must be beside it")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify", action="store_true", help="Check the shared CSV and figure files against provenance.json; no manuscript is needed")
    args = parser.parse_args()
    if args.verify:
        provenance = json.loads((args.output / "provenance.json").read_text(encoding="utf-8"))
        for name, entry in provenance["tables"].items():
            if sha256(args.output / name) != entry["csv_sha256"]:
                raise ValueError(f"CSV hash mismatch: {name}")
            with (args.output / name).open(encoding="utf-8", newline="") as handle:
                if len(list(csv.DictReader(handle))) != entry["row_count"]:
                    raise ValueError(f"CSV row count mismatch: {name}")
        for name, entry in provenance["figures"].items():
            if sha256(args.output / name) != entry["sha256"]:
                raise ValueError(f"Figure hash mismatch: {name}")
        print("Verified 4 CSV files (34 rows) and 4 manuscript figures against the provenance manifest.")
    elif args.manuscript is None:
        parser.error("Supply the manuscript path for export, or use --verify")
    else:
        export(args.manuscript, args.output)


if __name__ == "__main__":
    main()
