# Experiment Result Visualization Schema

This project treats each directory under `outputs/` as one experiment run when
the directory contains these files:

- `config.json`: run metadata and training configuration.
- `metrics.csv`: per-task scalar metrics.
- `metrics.json`: per-task grouped metrics, including old/new and class-block accuracy.
- `summary.json`: final aggregate metrics and accuracy matrices.

The visualization tools are intentionally read-only with respect to run
directories. Generated artifacts are written outside each run, under
`outputs/reports/` and `outputs/experiment_index.csv`.

## Commands

Generate or refresh the experiment index:

```powershell
python tools\visualize_results.py --outputs outputs --index
```

Generate a report for one run:

```powershell
python tools\visualize_results.py --outputs outputs --run outputs\local_cuda12_smoke_icarl_cifar100_init0_inc10_seed1993_20260624_144731
```

Generate reports for every indexed run:

```powershell
python tools\visualize_results.py --outputs outputs --index --all
```

## Index Fields

`outputs/experiment_index.csv` contains one row per valid run.

| Field | Meaning |
| --- | --- |
| `run_name` | Run directory name. |
| `started_at` | Timestamp from `config.json` or `summary.json`. |
| `model_name` | Incremental learning method, such as `icarl`. |
| `dataset` | Dataset name, such as `cifar100`. |
| `seed` | Random seed. |
| `init_cls` | Number of classes in the initial task. |
| `increment` | Number of classes added per task. |
| `memory_size` | Exemplar memory budget. |
| `num_tasks` | Number of incremental tasks. |
| `final_cnn_top1` | Final CNN Top-1 accuracy after the last task. |
| `final_nme_top1` | Final NME Top-1 accuracy after the last task. |
| `avg_cnn_top1` | Mean CNN Top-1 across tasks. |
| `avg_nme_top1` | Mean NME Top-1 across tasks. |
| `cnn_forgetting` | Average CNN forgetting from `summary.json`. |
| `nme_forgetting` | Average NME forgetting from `summary.json`. |
| `output_dir` | Path to the run directory. |

## Generated Figures

Each report contains:

- `top1_curve.png`: CNN and NME Top-1 across tasks.
- `top5_curve.png`: CNN and NME Top-5 across tasks.
- `avg_accuracy_curve.png`: running average Top-1 across tasks.
- `old_new_gap.png`: new-class accuracy minus old-class accuracy.
- `accuracy_matrix_cnn.png`: CNN class-block accuracy matrix.
- `accuracy_matrix_nme.png`: NME class-block accuracy matrix.
- `forgetting_bar.png`: CNN vs NME forgetting.
- `final_group_accuracy.png`: final-task accuracy by class block.

## Extension Rules

Future experiments should keep writing `config.json`, `metrics.csv`,
`metrics.json`, and `summary.json` with the existing field names. New metrics
can be added freely, but existing fields should stay stable so old and new runs
remain comparable.

If a method does not produce NME metrics, keep the CNN fields and omit the NME
section. The index and plots can then be extended to treat missing NME values as
optional.
