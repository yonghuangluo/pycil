from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ExperimentRun:
    run_dir: Path
    run_name: str
    meta: dict[str, Any]
    summary: dict[str, Any]
    metrics: list[dict[str, Any]]


INDEX_FIELDS = [
    "run_name",
    "started_at",
    "model_name",
    "dataset",
    "seed",
    "init_cls",
    "increment",
    "memory_size",
    "num_tasks",
    "final_cnn_top1",
    "final_nme_top1",
    "avg_cnn_top1",
    "avg_nme_top1",
    "cnn_forgetting",
    "nme_forgetting",
    "output_dir",
]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_value(value: str) -> Any:
    if value == "":
        return value
    try:
        as_int = int(value)
        if str(as_int) == value:
            return as_int
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _read_metrics_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: _coerce_value(value) for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def load_run(run_dir: str | Path) -> ExperimentRun:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"

    missing = [str(path) for path in (config_path, summary_path, metrics_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required result files: {', '.join(missing)}")

    meta = _read_json(config_path)
    summary = _read_json(summary_path)
    metrics = _read_metrics_csv(metrics_path)
    run_name = str(meta.get("run_name") or summary.get("run_name") or run_dir.name)
    return ExperimentRun(run_dir=run_dir, run_name=run_name, meta=meta, summary=summary, metrics=metrics)


def _index_row(run: ExperimentRun) -> dict[str, Any]:
    cnn = run.summary.get("cnn", {})
    nme = run.summary.get("nme", {})
    return {
        "run_name": run.run_dir.name,
        "started_at": run.meta.get("started_at", run.summary.get("started_at", "")),
        "model_name": run.meta.get("model_name", ""),
        "dataset": run.meta.get("dataset", ""),
        "seed": run.meta.get("seed", ""),
        "init_cls": run.meta.get("init_cls", ""),
        "increment": run.meta.get("increment", ""),
        "memory_size": run.meta.get("memory_size", ""),
        "num_tasks": run.summary.get("num_tasks", len(run.metrics)),
        "final_cnn_top1": cnn.get("final_top1", ""),
        "final_nme_top1": nme.get("final_top1", ""),
        "avg_cnn_top1": cnn.get("average_top1", ""),
        "avg_nme_top1": nme.get("average_top1", ""),
        "cnn_forgetting": cnn.get("forgetting", ""),
        "nme_forgetting": nme.get("forgetting", ""),
        "output_dir": str(run.run_dir),
    }


def build_experiment_index(outputs_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(outputs_dir)
    rows = []
    for summary_path in sorted(outputs_dir.glob("*/summary.json")):
        run_dir = summary_path.parent
        try:
            rows.append(_index_row(load_run(run_dir)))
        except FileNotFoundError:
            continue
    return rows


def write_experiment_index(outputs_dir: str | Path, index_path: str | Path | None = None) -> Path:
    outputs_dir = Path(outputs_dir)
    index_path = Path(index_path) if index_path else outputs_dir / "experiment_index.csv"
    rows = build_experiment_index(outputs_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return index_path


def _figure_path(report_dir: Path, name: str) -> Path:
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir / name


def _tasks(run: ExperimentRun) -> list[int]:
    return [int(row["task"]) for row in run.metrics]


def _plot_curve(run: ExperimentRun, fields: list[tuple[str, str]], title: str, ylabel: str, path: Path) -> Path:
    tasks = _tasks(run)
    plt.figure(figsize=(8, 4.8))
    for field, label in fields:
        if all(field in row and row[field] != "" for row in run.metrics):
            plt.plot(tasks, [float(row[field]) for row in run.metrics], marker="o", linewidth=2, label=label)
    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_old_new_gap(run: ExperimentRun, path: Path) -> Path:
    tasks = _tasks(run)
    metrics_json_path = run.run_dir / "metrics.json"
    if metrics_json_path.exists():
        raw_metrics = json.loads(metrics_json_path.read_text(encoding="utf-8"))
        plt.figure(figsize=(8, 4.8))
        for head in ("cnn", "nme"):
            gaps = []
            for item in raw_metrics:
                grouped = item.get(head, {}).get("grouped", {})
                old = float(grouped.get("old", 0))
                new = float(grouped.get("new", 0))
                gaps.append(new - old)
            plt.plot(tasks, gaps, marker="o", linewidth=2, label=f"{head.upper()} new-old")
    else:
        plt.figure(figsize=(8, 4.8))
        plt.plot(tasks, [0 for _ in tasks], marker="o", linewidth=2, label="gap unavailable")
    plt.axhline(0, color="black", linewidth=1, alpha=0.4)
    plt.title("New-vs-Old Accuracy Gap")
    plt.xlabel("Task")
    plt.ylabel("New accuracy - old accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_accuracy_matrix(run: ExperimentRun, head: str, path: Path) -> Path:
    matrix = run.summary.get(head, {}).get("accuracy_matrix", [])
    plt.figure(figsize=(7.2, 6.0))
    image = plt.imshow(matrix, cmap="viridis", vmin=0, vmax=100, aspect="auto")
    plt.colorbar(image, label="Top-1 accuracy")
    plt.title(f"{head.upper()} Accuracy Matrix")
    plt.xlabel("Evaluation task")
    plt.ylabel("Class block")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_forgetting(run: ExperimentRun, path: Path) -> Path:
    labels = []
    values = []
    for head in ("cnn", "nme"):
        value = run.summary.get(head, {}).get("forgetting")
        if value != "" and value is not None:
            labels.append(head.upper())
            values.append(float(value))
    plt.figure(figsize=(5.5, 4.2))
    bars = plt.bar(labels, values, color=["#4C78A8", "#F58518"][: len(labels)])
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom")
    plt.title("Forgetting")
    plt.ylabel("Average forgetting")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _plot_final_group_accuracy(run: ExperimentRun, path: Path) -> Path:
    metrics_json_path = run.run_dir / "metrics.json"
    plt.figure(figsize=(9, 4.8))
    plotted = False
    if metrics_json_path.exists():
        last = json.loads(metrics_json_path.read_text(encoding="utf-8"))[-1]
        for head in ("cnn", "nme"):
            grouped = last.get(head, {}).get("grouped", {})
            labels = [key for key in grouped if "-" in key]
            values = [float(grouped[key]) for key in labels]
            offset = -0.18 if head == "cnn" else 0.18
            x_values = [idx + offset for idx in range(len(labels))]
            plt.bar(x_values, values, width=0.36, label=head.upper())
            plotted = True
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.title("Final Task Accuracy by Class Block")
    plt.xlabel("Class block")
    plt.ylabel("Top-1 accuracy")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.25)
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def _format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _write_report(run: ExperimentRun, report_dir: Path, figures: list[Path]) -> Path:
    cnn = run.summary.get("cnn", {})
    nme = run.summary.get("nme", {})
    rel_figures = [path.relative_to(report_dir).as_posix() for path in figures]
    lines = [
        f"# Experiment Report: {run.run_name}",
        "",
        "## Configuration",
        "",
        f"- Model: `{run.meta.get('model_name', 'n/a')}`",
        f"- Dataset: `{run.meta.get('dataset', 'n/a')}`",
        f"- Seed: `{run.meta.get('seed', 'n/a')}`",
        f"- Init/increment: `{run.meta.get('init_cls', 'n/a')}` / `{run.meta.get('increment', 'n/a')}`",
        f"- Memory size: `{run.meta.get('memory_size', 'n/a')}`",
        f"- Started at: `{run.meta.get('started_at', 'n/a')}`",
        "",
        "## Summary",
        "",
        "| Head | Final Top-1 | Average Top-1 | Forgetting |",
        "| --- | ---: | ---: | ---: |",
        f"| CNN | {_format_float(cnn.get('final_top1'))} | {_format_float(cnn.get('average_top1'))} | {_format_float(cnn.get('forgetting'))} |",
        f"| NME | {_format_float(nme.get('final_top1'))} | {_format_float(nme.get('average_top1'))} | {_format_float(nme.get('forgetting'))} |",
        "",
        "## Figures",
        "",
    ]
    for rel_path in rel_figures:
        title = Path(rel_path).stem.replace("_", " ").title()
        lines.extend([f"### {title}", "", f"![{title}]({rel_path})", ""])
    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_run_report(run_dir: str | Path, report_dir: str | Path) -> dict[str, Any]:
    run = load_run(run_dir)
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = [
        _plot_curve(run, [("cnn_top1", "CNN Top-1"), ("nme_top1", "NME Top-1")], "Top-1 Accuracy", "Top-1 accuracy", _figure_path(report_dir, "top1_curve.png")),
        _plot_curve(run, [("cnn_top5", "CNN Top-5"), ("nme_top5", "NME Top-5")], "Top-5 Accuracy", "Top-5 accuracy", _figure_path(report_dir, "top5_curve.png")),
        _plot_curve(run, [("avg_cnn_top1", "CNN Average Top-1"), ("avg_nme_top1", "NME Average Top-1")], "Average Top-1 Accuracy", "Average Top-1 accuracy", _figure_path(report_dir, "avg_accuracy_curve.png")),
        _plot_old_new_gap(run, _figure_path(report_dir, "old_new_gap.png")),
        _plot_accuracy_matrix(run, "cnn", _figure_path(report_dir, "accuracy_matrix_cnn.png")),
        _plot_accuracy_matrix(run, "nme", _figure_path(report_dir, "accuracy_matrix_nme.png")),
        _plot_forgetting(run, _figure_path(report_dir, "forgetting_bar.png")),
        _plot_final_group_accuracy(run, _figure_path(report_dir, "final_group_accuracy.png")),
    ]
    report_path = _write_report(run, report_dir, figures)
    return {"report": report_path, "figures": figures}


def generate_all_reports(outputs_dir: str | Path, reports_root: str | Path | None = None) -> list[dict[str, Any]]:
    outputs_dir = Path(outputs_dir)
    reports_root = Path(reports_root) if reports_root else outputs_dir / "reports"
    generated = []
    for row in build_experiment_index(outputs_dir):
        run_dir = Path(row["output_dir"])
        generated.append(generate_run_report(run_dir, reports_root / run_dir.name))
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build PyCIL experiment indexes and visual reports.")
    parser.add_argument("--outputs", default="outputs", help="Directory containing experiment run folders.")
    parser.add_argument("--run", help="Single run directory to visualize.")
    parser.add_argument("--all", action="store_true", help="Generate reports for all runs under --outputs.")
    parser.add_argument("--index", action="store_true", help="Write experiment_index.csv under --outputs.")
    parser.add_argument("--reports-root", help="Directory for generated reports.")
    args = parser.parse_args(argv)

    if args.index:
        index_path = write_experiment_index(args.outputs)
        print(f"Wrote index: {index_path}")
    if args.run:
        run_dir = Path(args.run)
        reports_root = Path(args.reports_root) if args.reports_root else Path(args.outputs) / "reports"
        result = generate_run_report(run_dir, reports_root / run_dir.name)
        print(f"Wrote report: {result['report']}")
    if args.all:
        reports = generate_all_reports(args.outputs, args.reports_root)
        print(f"Wrote {len(reports)} report(s)")
    if not (args.index or args.run or args.all):
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
