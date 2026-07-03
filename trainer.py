import sys
import logging
import copy
import csv
import json
import shutil
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from datetime import datetime


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    _set_random()
    _set_device(args)
    _setup_experiment(args, init_cls)

    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.FileHandler(filename=os.path.join(args["log_dir"], "train.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args["aug"] if "aug" in args else 1,
        args.get("data_root", "./data"),
        args.get("download", True),
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []
    metric_records = []

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        model.incremental_train(data_manager)
        cnn_accy, nme_accy = model.eval_task()
        model.after_task()
        task_record = {
            "task": task,
            "known_classes": model._known_classes,
            "total_classes": model._total_classes,
            "exemplar_size": model.exemplar_size,
            "cnn": _to_serializable(cnn_accy),
            "nme": _to_serializable(nme_accy),
        }

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)


            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))
            task_record["average_accuracy"] = {
                "cnn_top1": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                "nme_top1": sum(nme_curve["top1"]) / len(nme_curve["top1"]),
            }
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            task_record["average_accuracy"] = {
                "cnn_top1": sum(cnn_curve["top1"]) / len(cnn_curve["top1"]),
                "nme_top1": None,
            }

        task_record["curves"] = {
            "cnn_top1": copy.deepcopy(cnn_curve["top1"]),
            "cnn_top5": copy.deepcopy(cnn_curve["top5"]),
            "nme_top1": copy.deepcopy(nme_curve["top1"]),
            "nme_top5": copy.deepcopy(nme_curve["top5"]),
        }
        metric_records.append(_to_serializable(task_record))
        _save_metrics(args, metric_records)
        if args.get("save_checkpoint", True):
            ckpt_path = os.path.join(
                args["checkpoint_dir"], "task_{:02d}.pth".format(task)
            )
            model.save_task_checkpoint(ckpt_path, task_record)
            shutil.copyfile(ckpt_path, os.path.join(args["checkpoint_dir"], "latest.pth"))
            logging.info("Saved checkpoint: %s", ckpt_path)


    if len(cnn_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        print('Forgetting (CNN):', forgetting)
        logging.info('Forgetting (CNN): {}'.format(forgetting))
        _save_final_summary(args, metric_records, "cnn", np_acctable, forgetting)
    if len(nme_matrix)>0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (NME):')
        print(np_acctable)
        print('Forgetting (NME):', forgetting)
        logging.info('Forgetting (NME): {}'.format(forgetting))
        _save_final_summary(args, metric_records, "nme", np_acctable, forgetting)


def _set_device(args):
    device_type = args.get("device", ["0"])
    if not isinstance(device_type, (list, tuple)):
        device_type = [device_type]
    gpus = []

    for device in device_type:
        if str(device).lower() == "cpu" or int(device) == -1:
            device = torch.device("cpu")
        else:
            gpu_id = int(device)
            if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                device = torch.device("cuda:{}".format(gpu_id))
            else:
                logging.warning(
                    "Requested cuda:%s is unavailable. Falling back to %s.",
                    gpu_id,
                    "cuda:0" if torch.cuda.is_available() else "cpu",
                )
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if device not in gpus:
            gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)
        torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _setup_experiment(args, init_cls):
    started_at = datetime.now()
    run_name = args.get(
        "run_name",
        "{}_{}_{}_init{}_inc{}_seed{}_{}".format(
            args["prefix"],
            args["model_name"],
            args["dataset"],
            init_cls,
            args["increment"],
            args["seed"],
            started_at.strftime("%Y%m%d_%H%M%S"),
        ),
    )
    output_root = args.get("output_root", "outputs")
    output_dir = os.path.join(output_root, run_name)
    args["run_name"] = run_name
    args["output_dir"] = output_dir
    args["checkpoint_dir"] = os.path.join(output_dir, "checkpoints")
    args["log_dir"] = os.path.join(output_dir, "logs")
    args["metrics_path"] = os.path.join(output_dir, "metrics.json")
    args["metrics_csv_path"] = os.path.join(output_dir, "metrics.csv")
    args["summary_path"] = os.path.join(output_dir, "summary.json")
    args["started_at"] = started_at.isoformat(timespec="seconds")

    os.makedirs(args["checkpoint_dir"], exist_ok=True)
    os.makedirs(args["log_dir"], exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(_to_serializable(args), f, indent=2)


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.device):
        return str(obj)
    return obj


def _save_metrics(args, records):
    records = _to_serializable(records)
    with open(args["metrics_path"], "w") as f:
        json.dump(records, f, indent=2)

    fieldnames = [
        "task",
        "known_classes",
        "total_classes",
        "exemplar_size",
        "cnn_top1",
        "cnn_top5",
        "nme_top1",
        "nme_top5",
        "avg_cnn_top1",
        "avg_nme_top1",
    ]
    with open(args["metrics_csv_path"], "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "task": record["task"],
                    "known_classes": record["known_classes"],
                    "total_classes": record["total_classes"],
                    "exemplar_size": record["exemplar_size"],
                    "cnn_top1": record["cnn"].get("top1"),
                    "cnn_top5": record["cnn"].get("top5"),
                    "nme_top1": record["nme"].get("top1") if record["nme"] else None,
                    "nme_top5": record["nme"].get("top5") if record["nme"] else None,
                    "avg_cnn_top1": record["average_accuracy"].get("cnn_top1"),
                    "avg_nme_top1": record["average_accuracy"].get("nme_top1"),
                }
            )


def _save_final_summary(args, records, metric_name, accuracy_matrix, forgetting):
    summary = {}
    if os.path.exists(args["summary_path"]):
        with open(args["summary_path"]) as f:
            summary = json.load(f)

    summary["run_name"] = args["run_name"]
    summary["output_dir"] = args["output_dir"]
    summary["started_at"] = args["started_at"]
    summary["num_tasks"] = len(records)
    summary[metric_name] = {
        "final_top1": records[-1][metric_name]["top1"] if records else None,
        "average_top1": records[-1]["average_accuracy"].get(
            "{}_top1".format(metric_name)
        )
        if records
        else None,
        "forgetting": _to_serializable(forgetting),
        "accuracy_matrix": _to_serializable(accuracy_matrix),
    }
    with open(args["summary_path"], "w") as f:
        json.dump(_to_serializable(summary), f, indent=2)


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
