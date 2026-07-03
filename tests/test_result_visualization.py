import json
from pathlib import Path

from tools.result_visualization import (
    build_experiment_index,
    generate_run_report,
    load_run,
)


def _write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_run(root):
    run_dir = root / "demo_icarl_cifar100_seed1"
    run_dir.mkdir()
    _write_json(
        run_dir / "config.json",
        {
            "run_name": "demo_icarl_cifar100_seed1",
            "model_name": "icarl",
            "dataset": "cifar100",
            "seed": 1,
            "init_cls": 10,
            "increment": 10,
            "memory_size": 2000,
            "started_at": "2026-06-24T14:47:31",
        },
    )
    _write_json(
        run_dir / "summary.json",
        {
            "run_name": "demo_icarl_cifar100_seed1",
            "num_tasks": 2,
            "cnn": {
                "final_top1": 70.0,
                "average_top1": 80.0,
                "forgetting": 15.0,
                "accuracy_matrix": [[90.0, 70.0], [0.0, 85.0]],
            },
            "nme": {
                "final_top1": 74.0,
                "average_top1": 82.0,
                "forgetting": 10.0,
                "accuracy_matrix": [[91.0, 74.0], [0.0, 86.0]],
            },
        },
    )
    (run_dir / "metrics.csv").write_text(
        "\n".join(
            [
                "task,known_classes,total_classes,exemplar_size,cnn_top1,cnn_top5,nme_top1,nme_top5,avg_cnn_top1,avg_nme_top1",
                "0,10,10,2000,90.0,99.0,91.0,99.1,90.0,91.0",
                "1,20,20,2000,70.0,95.0,74.0,96.0,80.0,82.5",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir


def test_load_run_returns_metadata_summary_and_metrics(tmp_path):
    run_dir = _make_run(tmp_path)

    run = load_run(run_dir)

    assert run.run_name == "demo_icarl_cifar100_seed1"
    assert run.meta["model_name"] == "icarl"
    assert run.summary["nme"]["forgetting"] == 10.0
    assert [row["task"] for row in run.metrics] == [0, 1]
    assert run.metrics[1]["nme_top1"] == 74.0


def test_build_experiment_index_flattens_comparable_fields(tmp_path):
    run_dir = _make_run(tmp_path)

    rows = build_experiment_index(tmp_path)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_name"] == run_dir.name
    assert row["model_name"] == "icarl"
    assert row["dataset"] == "cifar100"
    assert row["final_nme_top1"] == 74.0
    assert row["nme_forgetting"] == 10.0


def test_generate_run_report_writes_markdown_and_expected_figures(tmp_path):
    run_dir = _make_run(tmp_path)
    report_dir = tmp_path / "reports" / run_dir.name

    generated = generate_run_report(run_dir, report_dir)

    assert (report_dir / "report.md").exists()
    report = (report_dir / "report.md").read_text(encoding="utf-8")
    assert "demo_icarl_cifar100_seed1" in report
    assert "Final Top-1" in report
    figure_names = {path.name for path in generated["figures"]}
    assert {
        "top1_curve.png",
        "top5_curve.png",
        "avg_accuracy_curve.png",
        "old_new_gap.png",
        "accuracy_matrix_cnn.png",
        "accuracy_matrix_nme.png",
        "forgetting_bar.png",
        "final_group_accuracy.png",
    }.issubset(figure_names)
