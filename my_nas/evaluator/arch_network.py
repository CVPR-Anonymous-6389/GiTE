"""
Networks that take architectures as inputs.
"""

import abc
import logging

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from my_nas import utils
from my_nas.utils.exception import expect, ConfigException
from my_nas.base import Component

__all__ = ["PointwiseComparator"]

class ArchNetwork(Component):
    REGISTRY = "arch_network"

    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass


class ArchEmbedder(Component, nn.Module):
    REGISTRY = "arch_embedder"

    def __init__(self, schedule_cfg):
        Component.__init__(self, schedule_cfg)
        nn.Module.__init__(self)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))



class PointwiseComparator(ArchNetwork, nn.Module):
    """
    Compatible to NN regression-based predictor of architecture performance.
    """
    NAME = "pointwise_comparator"

    def __init__(self, search_space,
                 arch_embedder_type="lstm", arch_embedder_cfg=None,
                 mlp_hiddens=(200, 200, 200), mlp_dropout=0.1,
                 optimizer={
                     "type": "Adam",
                     "lr": 0.001
                 }, scheduler=None,
                 compare_loss_type="margin_linear",
                 compare_margin=0.01,
                 margin_l2=False,
                 use_incorrect_list_only=False,
                 tanh_score=None,
                 max_grad_norm=None,
                 schedule_cfg=None):
        # [optional] arch reconstruction loss (arch_decoder_type/cfg)
        super(PointwiseComparator, self).__init__(schedule_cfg)
        nn.Module.__init__(self)

        # configs
        expect(compare_loss_type in {"binary_cross_entropy", "margin_linear"},
               "comparing loss type {} not supported".format(compare_loss_type),
               ConfigException)
        self.compare_loss_type = compare_loss_type
        self.compare_margin = compare_margin
        self.margin_l2 = margin_l2
        self.max_grad_norm = max_grad_norm
        # for update_argsort listwise only
        self.use_incorrect_list_only = use_incorrect_list_only
        self.tanh_score = tanh_score
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler

        self.search_space = search_space
        ae_cls = ArchEmbedder.get_class_(arch_embedder_type)
        self.arch_embedder = ae_cls(self.search_space, **(arch_embedder_cfg or {}))

        dim = self.embedding_dim = self.arch_embedder.out_dim
        # construct MLP from embedding to score
        self.mlp = []
        for hidden_size in mlp_hiddens:
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, hidden_size),
                nn.ReLU(inplace=False),
                nn.Dropout(p=mlp_dropout)))
            dim = hidden_size
        self.mlp.append(nn.Linear(dim, 1))
        self.mlp = nn.Sequential(*self.mlp)

        # init optimizer and scheduler
        self.optimizer = utils.init_optimizer(self.parameters(), optimizer)
        self.scheduler = utils.init_scheduler(self.optimizer, scheduler)

    def reset_optimizer(self):
        self.optimizer = utils.init_optimizer(self.parameters(), self.optimizer_cfg)
        self.scheduler = utils.init_scheduler(self.optimizer, self.scheduler_cfg)

    def predict_rollouts(self, rollouts, **kwargs):
        archs = [r.arch for r in rollouts]
        return self.predict(archs, **kwargs)

    def predict(self, arch, sigmoid=True, tanh=False):
        score = self.mlp(self.arch_embedder(arch)).squeeze(-1)
        if sigmoid:
            score = torch.sigmoid(score)
        elif tanh:
            score = torch.tanh(score)
        return score

    def update_predict_rollouts(self, rollouts, labels):
        archs = [r.arch for r in rollouts]
        return self.update_predict(archs, labels)

    def update_predict_list(self, predict_lst):
        # use MSE regression loss to step
        archs = [item[0] for item in predict_lst]
        labels = [item[1] for item in predict_lst]
        return self.update_predict(archs, labels)

    def update_predict(self, archs, labels):
        scores = torch.sigmoid(self.mlp(self.arch_embedder(archs)))
        mse_loss = F.mse_loss(
            scores.squeeze(), scores.new(labels))
        self.optimizer.zero_grad()
        mse_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        return mse_loss.item()

    def compare(self, arch_1, arch_2):
        # pointwise score and comparen
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        return torch.sigmoid(s_2 - s_1)

    def update_compare_rollouts(self, compare_rollouts, better_labels):
        arch_1, arch_2 = zip(*[(r.rollout_1.arch, r.rollout_2.arch) for r in compare_rollouts])
        return self.update_compare(arch_1, arch_2, better_labels)

    def update_compare_list(self, compare_lst):
        # use binary classification loss to step
        arch_1, arch_2, better_labels = zip(*compare_lst)
        return self.update_compare(arch_1, arch_2, better_labels)

    def update_compare_eq(self, arch_1, arch_2, better_eq_labels, margin=None):
        assert self.compare_loss_type == "margin_linear"
        # in range (0, 1) to make the `compare_margin` meaningful
        # s_1 = self.predict(arch_1)
        # s_2 = self.predict(arch_2)
        s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
        s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
        better_pm = s_1.new(np.array(better_eq_labels, dtype=np.float32))
        zero_ = s_1.new([0.])
        margin = [self.compare_margin] if margin is None else margin
        margin = s_1.new(margin)
        pair_loss = torch.mean(
            torch.where(
                better_pm == 0,
                torch.max(zero_, (s_2 - s_1).abs() - margin / 2),
                torch.max(zero_, margin - better_pm * (s_2 - s_1))
            ))
        self.optimizer.zero_grad()
        pair_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        # return pair_loss.item(), s_1, s_2
        return pair_loss.item()

    def update_compare(self, arch_1, arch_2, better_labels, margin=None):
        if self.compare_loss_type == "binary_cross_entropy":
            # compare_score = self.compare(arch_1, arch_2)
            s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
            compare_score = torch.sigmoid(s_2 - s_1)
            pair_loss = F.binary_cross_entropy(
                compare_score, compare_score.new(better_labels))
        elif self.compare_loss_type == "margin_linear":
            # in range (0, 1) to make the `compare_margin` meaningful
            # s_1 = self.predict(arch_1)
            # s_2 = self.predict(arch_2)
            s_1 = self.mlp(self.arch_embedder(arch_1)).squeeze()
            s_2 = self.mlp(self.arch_embedder(arch_2)).squeeze()
            better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
            zero_ = s_1.new([0.])
            margin = [self.compare_margin] if margin is None else margin
            margin = s_1.new(margin)
            if not self.margin_l2:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
            else:
                pair_loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)) ** 2 / np.maximum(1., margin))
        self.optimizer.zero_grad()
        pair_loss.backward()
        self._clip_grads()
        self.optimizer.step()
        # return pair_loss.item(), s_1, s_2
        return pair_loss.item()

    def argsort(self, archs, batch_size=None):
        pass

    def update_argsort(self, archs, idxes, first_n=None, accumulate_only=False, is_sorted=False):
        archs = np.array(archs)
        bs, len_ = archs.shape[:2]
        if idxes is not None:
            idxes = np.array(idxes)
            assert idxes.ndim == 2 and idxes.shape[0] == bs and idxes.shape[1] == len_
            # if idxes.ndim == 1:
            #     idxes = idxes[None, :]
        else:
            assert is_sorted
        flat_archs = archs.reshape([-1] + list(archs.shape[2:]))
        if self.tanh_score is not None:
            scores = self.tanh_score * self.predict(flat_archs, sigmoid=False, tanh=True)
        else:
            scores = self.predict(flat_archs, sigmoid=False)

        scores = scores.reshape((bs, len_))
        exp_score = (scores - scores.max(dim=-1, keepdim=True)[0].detach()).exp()
        if not is_sorted:
            exp_score_rank = exp_score[np.arange(0, bs)[:, None], idxes]
        else:
            exp_score_rank = exp_score
        EPS = 1e-12
        exp_score_rank = torch.max(exp_score_rank, torch.tensor(EPS).to(exp_score_rank.device))

        if self.use_incorrect_list_only:
            correct_idxes = torch.all(torch.argsort(
                exp_score_rank, dim=-1, descending=True) \
                                      == exp_score_rank.new(np.arange(len_)).to(torch.long),
                                      dim=-1)
            do_not_train_idxes = (exp_score_rank[:, -1] / exp_score_rank[:, 0] < 1e-9) & correct_idxes
            keep_list_idxes = 1 - do_not_train_idxes
            exp_score_rank = exp_score_rank[keep_list_idxes]
            actual_bs = torch.sum(keep_list_idxes).item()
            logging.debug("actual bs: {}".format(actual_bs))
        else:
            actual_bs = bs

        inds = (np.tile(np.arange(actual_bs)[:, None], [1, len_]),
                np.tile(np.arange(len_)[::-1][None, :], [actual_bs, 1]))
        normalize = torch.cumsum(exp_score_rank[inds], dim=1)[inds]

        if first_n is not None:
            exp_score_rank = exp_score_rank[:, :first_n]
            normalize = normalize[:, :first_n]

        normalize = torch.clamp(normalize, min=1.e-10)
        exp_score_rank = torch.clamp(exp_score_rank, min=1.e-11)
        loss = torch.mean(torch.mean(torch.log(normalize + EPS) - torch.log(exp_score_rank + EPS),
                                     dim=1))

        logging.debug("exp score maxmin: {} {}".format(exp_score_rank.min(), exp_score_rank.max()))
        logging.debug("normalize maxmin: {} {}".format(normalize.min(), normalize.max()))
        logging.debug("loss: {}".format(loss))
        if not accumulate_only:
            self.optimizer.zero_grad()
        loss.backward()
        if not accumulate_only:
            self._clip_grads()
            self.optimizer.step()
        return loss.item()

    # def argsort_list(self, archs, batch_size=None):
    #     # TODO
    #     pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    def on_epoch_start(self, epoch):
        super(PointwiseComparator, self).on_epoch_start(epoch)
        if self.scheduler is not None:
            self.scheduler.step(epoch - 1)
            self.logger.info("Epoch %3d: lr: %.5f", epoch, self.scheduler.get_lr()[0])

    def _clip_grads(self):
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

