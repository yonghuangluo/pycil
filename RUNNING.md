# PyCIL-master 启动与配置说明

本文档用于在当前 `PyCIL-master` 仓库中快速启动不同增量学习模型，并说明如何通过 JSON 配置指定模型、数据集、GPU、训练参数和输出目录。

## 1. 基本启动方式

项目入口是 `main.py`，启动时只需要指定一个配置文件：

```bash
python main.py --config ./exps/icarl.json
```

`main.py` 会读取 JSON 配置，然后把配置传入 `trainer.py` 执行训练。当前代码只直接暴露了 `--config` 这个命令行参数；模型名、数据集、GPU、batch size、epoch 等参数都在 JSON 文件里修改。

建议不要直接覆盖原始配置，可以复制一份再改：

```bash
cp exps/icarl.json exps/my_icarl.json
python main.py --config ./exps/my_icarl.json
```

## 2. 快速示例

### 运行当前本地 iCaRL 配置

当前 `exps/icarl.json` 已经适配了本地数据路径和输出目录：

```bash
python main.py --config ./exps/icarl.json
```

### 切换到 FineTune

```bash
python main.py --config ./exps/finetune.json
```

### 切换到 FOSTER

```bash
python main.py --config ./exps/foster.json
```

### 指定单卡 GPU

在配置文件中修改：

```json
"device": ["0"]
```

### 指定 CPU

在配置文件中修改：

```json
"device": ["cpu"]
```

也可以使用：

```json
"device": ["-1"]
```

## 3. 配置文件工作方式

配置文件位于 `exps/` 目录。每个 JSON 文件通常对应一个算法，例如：

- `exps/icarl.json`: iCaRL
- `exps/finetune.json`: FineTune
- `exps/foster.json`: FOSTER
- `exps/memo.json`: MEMO
- `exps/tagfex.json`: TagFex

训练时真正生效的是 JSON 里的 `model_name` 字段，而不只是文件名。比如 `exps/beef.json` 中的模型名是：

```json
"model_name": "beefiso"
```

因此，如果复制配置文件后想切换模型，需要同时确认 `model_name` 和该模型需要的专属参数是否匹配。

## 4. 可用模型名

当前 `utils/factory.py` 支持以下 `model_name`：

| model_name | 常用配置文件 |
| --- | --- |
| `finetune` | `exps/finetune.json` |
| `replay` | `exps/replay.json` |
| `icarl` | `exps/icarl.json` |
| `bic` | `exps/bic.json` |
| `podnet` | `exps/podnet.json` |
| `lwf` | `exps/lwf.json` |
| `ewc` | `exps/ewc.json` |
| `wa` | `exps/wa.json` |
| `der` | `exps/der.json` |
| `gem` | `exps/gem.json` |
| `coil` | `exps/coil.json` |
| `foster` | `exps/foster.json` |
| `rmm-icarl` | `exps/rmm-icarl.json` |
| `rmm-foster` | `exps/rmm-foster.json` |
| `fetril` | `exps/fetril.json` |
| `pass` | `exps/pass.json` |
| `il2a` | `exps/il2a.json` |
| `ssre` | `exps/ssre.json` |
| `memo` | `exps/memo.json` |
| `beefiso` | `exps/beef.json` |
| `simplecil` | `exps/simplecil.json` |
| `acil` | `exps/acil.json` |
| `ds-al` | `exps/ds-al.json` |
| `aper_finetune` | `exps/aper_finetune.json` |
| `tagfex` | `exps/tagfex.json` |

## 5. 常用全局参数

以下字段多数模型都会使用，可以优先调整这些参数：

| 参数 | 作用 | 示例 |
| --- | --- | --- |
| `prefix` | 实验名前缀，影响日志和输出目录命名 | `"my_run"` |
| `dataset` | 数据集名称 | `"cifar100"` |
| `data_root` | 数据集根目录，未写时默认 `./data` | `"/root/autodl-tmp/datasets"` |
| `download` | CIFAR 数据是否自动下载 | `false` |
| `model_name` | 要运行的算法模型 | `"icarl"` |
| `convnet_type` | 主干网络类型 | `"resnet32"` |
| `device` | 使用的设备 | `["0"]` |
| `seed` | 随机种子列表，会逐个运行 | `[1993]` |
| `shuffle` | 是否打乱类别顺序 | `true` |
| `init_cls` | 第一个任务的类别数 | `10` |
| `increment` | 后续每个任务新增类别数 | `10` |
| `memory_size` | 样本记忆库总容量 | `2000` |
| `memory_per_class` | 每类样本记忆数量，部分模型会用 | `20` |
| `fixed_memory` | 是否固定总记忆库大小 | `false` |
| `batch_size` | 训练 batch size，模型支持时生效 | `128` |
| `num_workers` | DataLoader 进程数 | `4` |
| `output_root` | 新版输出目录根路径 | `"outputs"` |
| `save_checkpoint` | 是否保存每个任务 checkpoint | `true` |

注意：不同模型读取的训练超参数名称不完全相同，例如有的使用 `init_epoch`，有的使用 `init_epochs`，有的使用 `lr`，有的使用 `lrate`。修改模型专属参数时，优先参考对应的 `exps/[模型].json` 和 `models/[模型].py`。

## 6. 当前本地推荐模板

如果主要在本机跑 CIFAR-100，可以从下面模板开始复制修改：

```json
{
  "prefix": "my_exp",
  "dataset": "cifar100",
  "data_root": "/root/autodl-tmp/datasets",
  "download": false,
  "output_root": "outputs",
  "save_checkpoint": true,
  "memory_size": 2000,
  "memory_per_class": 20,
  "fixed_memory": false,
  "shuffle": true,
  "init_cls": 10,
  "increment": 10,
  "model_name": "icarl",
  "convnet_type": "resnet32",
  "device": ["0"],
  "seed": [1993],
  "init_epoch": 200,
  "epochs": 170,
  "batch_size": 512,
  "num_workers": 4,
  "pin_memory": true,
  "persistent_workers": false
}
```

保存为 `exps/my_icarl.json` 后启动：

```bash
python main.py --config ./exps/my_icarl.json
```

## 7. 切换模型的正确做法

推荐做法是复制目标模型已有配置，而不是只改 `model_name`：

```bash
cp exps/foster.json exps/my_foster.json
```

然后编辑 `exps/my_foster.json` 中的：

```json
"prefix": "my_foster",
"dataset": "cifar100",
"data_root": "/root/autodl-tmp/datasets",
"download": false,
"device": ["0"],
"seed": [1993]
```

最后启动：

```bash
python main.py --config ./exps/my_foster.json
```

这样能保留 FOSTER 需要的 `beta1`、`beta2`、`boosting_epochs`、`compression_epochs` 等专属参数。

## 8. 数据集说明

当前 `utils/data_manager.py` 支持：

- `cifar10`
- `cifar100`
- `cifar10_aa`
- `cifar100_aa`
- `imagenet100`
- `imagenet1000`

### CIFAR-100

如果本地已有数据，建议设置：

```json
"data_root": "/root/autodl-tmp/datasets",
"download": false
```

如果希望 torchvision 自动下载 CIFAR-100，可以设置：

```json
"data_root": "./data",
"download": true
```

### ImageNet-100 / ImageNet-1000

`utils/data.py` 中的 ImageNet 数据路径仍是占位符：

```python
train_dir = "[DATA-PATH]/train/"
test_dir = "[DATA-PATH]/val/"
```

如果要运行 ImageNet，需要先在 `utils/data.py` 里改成真实路径，或扩展代码从配置中读取路径。

## 9. 输出文件

训练开始后会生成两类输出。

### 传统日志目录

路径格式：

```text
logs/{model_name}/{dataset}/{init_cls}/{increment}/{prefix}_{seed}_{convnet_type}.log
```

例如：

```text
logs/icarl/cifar100/0/10/local_cuda12_smoke_1993_resnet32.log
```

### 新版实验输出目录

如果配置中包含 `output_root`，默认会写入：

```text
outputs/{prefix}_{model_name}_{dataset}_init{init_cls}_inc{increment}_seed{seed}_{timestamp}/
```

目录内包括：

- `config.json`: 本次实际运行配置
- `logs/train.log`: 本次运行日志
- `metrics.json`: 每个任务的详细指标
- `metrics.csv`: 便于表格查看的指标
- `summary.json`: 最终 top1、平均精度、forgetting、accuracy matrix
- `checkpoints/task_XX.pth`: 每个任务 checkpoint
- `checkpoints/latest.pth`: 最新 checkpoint

如果不想保存 checkpoint，可以设置：

```json
"save_checkpoint": false
```

## 10. 多随机种子运行

`seed` 是列表。写多个 seed 时，程序会按顺序多次训练：

```json
"seed": [1993, 1994, 1995]
```

每个 seed 会生成独立日志和输出目录。

## 11. 常见修改场景

### 只做快速冒烟测试

把 epoch 调小，先确认环境和数据没问题：

```json
"init_epoch": 1,
"epochs": 1,
"batch_size": 128,
"save_checkpoint": false
```

不同模型字段可能叫 `init_epochs`、`boosting_epochs`、`compression_epochs`、`expansion_epochs`、`fusion_epochs` 等，要按对应配置文件修改。

### 减少显存占用

优先调整：

```json
"batch_size": 64,
"device": ["0"],
"num_workers": 2
```

某些模型还有 `init_batch_size`、`IL_batch_size`、`reduce_batch_size` 等字段。

### 修改任务划分

例如 CIFAR-100 首次 50 类，后续每次 10 类：

```json
"init_cls": 50,
"increment": 10
```

例如 CIFAR-100 首次 10 类，后续每次 10 类：

```json
"init_cls": 10,
"increment": 10
```

## 12. 常见问题

### `assert 0` 或 `Unknown dataset`

检查 `dataset` 是否属于支持列表。ImageNet 默认路径是占位符，需要先修改 `utils/data.py`。

### `Unknown model` 或直接断言失败

检查 `model_name` 是否在第 4 节列表中。模型名区分连字符和下划线，例如 `ds-al`、`rmm-icarl`、`aper_finetune`。

### CUDA 不可用或 GPU 编号错误

`trainer.py` 会尝试回退到可用设备。建议先用单卡：

```json
"device": ["0"]
```

### 配置参数改了但没生效

检查对应模型代码是否读取该字段。全局字段在 `trainer.py` 和 `utils/data_manager.py` 中使用；模型训练超参数通常在 `models/[模型].py` 中读取。

## 13. 推荐实验流程

1. 从 `exps/` 复制目标模型配置到自己的文件，例如 `exps/my_icarl.json`。
2. 修改 `prefix`，避免覆盖或混淆实验日志。
3. 修改 `dataset`、`data_root`、`download`、`device`。
4. 根据显存修改 `batch_size`、`num_workers`。
5. 先用较小 epoch 冒烟测试。
6. 确认能跑通后恢复正式 epoch。
7. 在 `outputs/` 中查看 `summary.json`、`metrics.csv` 和 checkpoint。

