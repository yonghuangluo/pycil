import logging
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from models.podnet import pod_spatial_loss
from utils.inc_net import IncrementalNet
from utils.toolkit import target2onehot, tensor2numpy

EPSILON = 1e-8

init_epoch = 200
init_lr = 0.1
init_milestones = [60, 120, 170]
init_lr_decay = 0.1
init_weight_decay = 0.0005


epochs = 180
lrate = 0.1
milestones = [70, 120, 150]
lrate_decay = 0.1
batch_size = 128
weight_decay = 2e-4
num_workers = 4
T = 2
# EWC 正则项的权重，越大表示越重视保护旧任务参数，越小表示越偏向适应新任务。
lamda = 1000
# Fisher 重要性上限，防止某些参数的 Fisher 过大导致正则项主导训练。
fishermax = 0.0001


class EWC(BaseLearner):
    """EWC 增量学习方法。

    训练思路：
    - 第一个任务正常训练分类器，并在任务结束后保存参数快照和 Fisher；
    - 后续任务只用新类别数据训练，同时用 EWC loss 约束旧任务重要参数；
    - 每个任务结束后重新估计 Fisher，并更新参数快照，供下一个任务使用。
    """

    def __init__(self, args):
        super().__init__(args)
        # fisher 会在每个任务结束后估计，保存每个可训练参数对旧任务的重要性。
        # 初始为 None，表示还没有旧任务信息可保护。
        self.fisher = None
        # IncrementalNet 支持在增量任务到来时扩展分类头输出维度。
        self._network = IncrementalNet(args, False)

    def after_task(self):
        """任务收尾：更新已知类别数。"""
        # 当前任务训练完成后，把“已知类别数”推进到本轮训练后的总类别数。
        # 下一次 incremental_train 会从这个位置开始构造新任务类别范围。
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        """训练一个增量任务，并在任务结束后刷新 EWC 所需的旧任务信息。"""
        # 一个增量任务的完整流程：
        # 1. 更新当前任务编号和总类别数；
        # 2. 扩展分类头，使网络能输出到目前为止的所有类别；
        # 3. 构造当前任务训练集和累计类别测试集；
        # 4. 训练网络：首任务只用分类损失，后续任务用分类损失 + EWC 正则；
        # 5. 任务结束后估计 Fisher，并保存当前参数快照供后续任务使用。
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        # 分类头从旧类别数扩展到总类别数，例如 10 类扩展到 20 类。
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        # 训练集只包含当前增量任务的新类别。
        # 例如已知 10 类、本任务新增 10 类时，这里取 [10, 20)。
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        # 测试集包含从第 0 类到当前总类别的所有类别，用来评估累计分类能力。
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        # 多卡训练时临时包一层 DataParallel；训练后再还原成原始 module，
        # 方便后续保存参数快照和 Fisher 时参数名保持一致。
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # 每个任务训练结束后，EWC 需要保存两类旧任务信息：
        # 1. fisher：参数对已经学过任务的重要性，数值越大，后续越不希望该参数变化；
        # 2. mean：当前任务结束时的参数快照，后续训练新任务时会把它当作旧参数参考点。
        if self.fisher is None:
            # 第一个任务结束后，直接用当前任务数据估计 Fisher 对角项。
            self.fisher = self.getFisherDiagonal(self.train_loader)
        else:
            # 后续任务结束后，把旧 Fisher 和新任务 Fisher 按类别数量比例融合。
            # 这样 fisher 仍然表示“到目前为止所有已学任务”的参数重要性估计。
            alpha = self._known_classes / self._total_classes
            new_finsher = self.getFisherDiagonal(self.train_loader)
            for n, p in new_finsher.items():
                # 分类头会随新类别扩展，旧 fisher 只覆盖旧类别对应的前一段参数。
                # 因此这里只融合旧参数范围，新类别新增参数保留本轮估计。
                new_finsher[n][: len(self.fisher[n])] = (
                    alpha * self.fisher[n]
                    + (1 - alpha) * new_finsher[n][: len(self.fisher[n])]
                )
            self.fisher = new_finsher
        # 注意：这里的 mean 不是“多个数的平均值”，而是 EWC 公式里的旧参数中心点。
        # 它保存任务训练完成时每个可训练参数的副本，作为以后计算 EWC 惩罚的参照。
        self.mean = {
            n: p.clone().detach()
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }

    def _train(self, train_loader, test_loader):
        """根据当前任务编号，选择首任务训练或后续增量训练。"""
        # 根据任务编号选择训练策略：
        # 第 0 个任务没有旧知识需要保护，直接进行普通监督训练；
        # 后续任务已经有旧参数快照和 Fisher，需要在新任务分类损失外加入 EWC 正则。
        self._network.to(self._device)
        if self._cur_task == 0:
            # 初始任务训练轮数和学习率策略通常更充分，因为它奠定初始特征表示。
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=init_lr,
                weight_decay=init_weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=init_milestones, gamma=init_lr_decay
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            # 增量任务使用另一组训练轮数、学习率和 weight decay 配置。
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=lrate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=milestones, gamma=lrate_decay
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        """首任务训练流程：只使用交叉熵分类损失。"""
        # 首任务训练：此时还没有旧任务，因此只优化普通交叉熵分类损失。
        prog_bar = tqdm(range(init_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                # logits 覆盖首任务全部类别，targets 也是原始类别标签，因此直接做 CE。
                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # 统计当前 epoch 的训练准确率，仅用于日志显示，不参与反向传播。
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    init_epoch,
                    losses / len(train_loader),
                    train_acc,
                )

            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        """后续任务训练流程：当前任务分类损失 + EWC 参数约束。"""
        # 增量任务训练：当前 train_loader 只含新类别，但网络输出包含旧类别 + 新类别。
        # 这里一边学习新类别，一边用 EWC 约束旧任务重要参数。
        prog_bar = tqdm(range(epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                # 只取新类别对应的 logits 做当前任务分类。
                # 例如旧类数为 10，当前 targets 是 10~19；
                # logits[:, 10:] 的类别索引是 0~9，所以 targets 也要减去 self._known_classes。
                loss_clf = F.cross_entropy(
                    logits[:, self._known_classes :], targets - self._known_classes
                )
                # 新任务训练目标由两部分组成：
                # loss_clf 学当前任务的新类别；
                # loss_ewc 约束重要旧参数不要偏离旧任务结束时的参数快照太远。
                loss_ewc = self.compute_ewc()
                loss = loss_clf + lamda * loss_ewc

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                # 这里用完整 logits 统计在“所有已见类别”上的预测准确率。
                # 注意训练损失只对新类别切片做 CE，但预测时仍然比较全类别 argmax。
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        logging.info(info)

    def compute_ewc(self):
        """计算 EWC 正则损失，惩罚当前参数偏离旧任务参数快照。"""
        # EWC 正则项：
        #   1/2 * sum_i Fisher_i * (theta_i - theta_old_i)^2
        # 其中 theta_i 是当前参数，theta_old_i 是 self.mean 中保存的旧参数快照。
        # Fisher_i 越大，说明该参数对旧任务越重要，偏离旧值时惩罚越强。
        loss = 0
        if len(self._multiple_gpus) > 1:
            for n, p in self._network.module.named_parameters():
                if n in self.fisher.keys():
                    # p 可能比 self.mean[n] 更长，常见于分类头随新类别扩展。
                    # p[: len(self.mean[n])] 只取旧类别/旧结构已有的那部分参数参与惩罚。
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        else:
            for n, p in self._network.named_parameters():
                if n in self.fisher.keys():
                    # 对未扩展的 backbone 参数，这里的切片等价于取整个参数；
                    # 对扩展过的分类头，则只约束旧任务已有的前一段参数。
                    loss += (
                        torch.sum(
                            (self.fisher[n])
                            * (p[: len(self.mean[n])] - self.mean[n]).pow(2)
                        )
                        / 2
                    )
        return loss

    def getFisherDiagonal(self, train_loader):
        """估计 Fisher 信息矩阵的对角线，作为每个参数的重要性权重。"""
        # 用梯度平方的平均值近似 Fisher 信息矩阵的对角线。
        # 这里不保存完整 Fisher 矩阵，只为每个参数保存一个同形状的重要性张量。
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            if p.requires_grad
        }
        self._network.train()
        optimizer = optim.SGD(self._network.parameters(), lr=lrate)
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            logits = self._network(inputs)["logits"]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    # 梯度平方越大，表示旧任务 loss 对该参数越敏感，
                    # 后续训练新任务时就越应该保护这个参数。
                    fisher[n] += p.grad.pow(2).clone()
        for n, p in fisher.items():
            # 对所有 batch 取平均，得到每个参数的重要性估计；
            # 再用 fishermax 截断，避免少数极大值让 EWC 惩罚过强。
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher
