"""
GiTE: A Generic Vision Transformer Encoding Scheme
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
from my_nas.evaluator.arch_network import ArchEmbedder


def construct_MLP(in_dim, out_dim, hidden_dims, mlp_dropout=0.1):
    mlp = []
    for dim in hidden_dims:
        mlp.append(nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.ReLU(inplace=False),
            nn.Dropout(p=mlp_dropout)))
        in_dim = dim
    mlp.append(nn.Linear(in_dim, out_dim))
    return nn.Sequential(*mlp)


class ViT_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "vit-seq"

    def __init__(self, search_space, dim, schedule_cfg=None):
        super(ViT_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.out_dim = dim
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        return self._placeholder_tensor.new(archs)


class ViT_LSTMSeqEmbedder(ArchEmbedder):
    NAME = "vit-lstm"

    def __init__(self, search_space,
                 info_dim=32,
                 op_dim=32,
                 num_layers=1,
                 num_op_choices=1,
                 use_mean=False,
                 use_info=False,
                 schedule_cfg=None
    ):
        super(ViT_LSTMSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.info_dim = info_dim
        self.op_dim = op_dim
        self.num_op_choices = num_op_choices
        self.num_layers = num_layers
        self.use_mean = use_mean
        self.use_info = use_info

        self.op_emb = nn.Embedding(self.num_op_choices, self.op_dim)
        
        self.rnn = nn.LSTM(
            input_size=self.op_dim,
            hidden_size=self.info_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        
        self.out_dim = info_dim

    def forward(self, archs):
        embs = self.op_emb(torch.LongTensor(archs).to(self.op_emb.weight.device))
        out, (h_n, _) = self.rnn(embs)

        if self.use_info:
            y = h_n[0]
        else:
            if self.use_mean:
                y = torch.mean(out, dim=1)
            else:
                y = out[:, -1, :]
        return y


class ViT_LSTMSeqSeparateEmbedder(ArchEmbedder):
    NAME = "vit-lstm-separate"

    def __init__(self, search_space, 
                 info_dim=32,
                 op_dim=32,
                 num_input_emb=1,
                 num_head=4,
                 num_ratio=4,
                 num_depth=3,
                 depth=16,
                 num_layers=1,
                 use_mlp=False,
                 mlp_schedule={},
                 share_emb=True,
                 use_info=False,
                 use_mean=False,
                 schedule_cfg=None):
        super(ViT_LSTMSeqSeparateEmbedder, self).__init__(schedule_cfg)

        self.info_dim = info_dim
        self.op_dim = op_dim
        self.num_input_emb = num_input_emb
        self.num_head = num_head
        self.num_ratio = num_ratio
        self.depth = depth
        self.num_depth = num_depth
        self.use_mlp = use_mlp
        self.mlp_schedule = mlp_schedule
        self.share_emb = share_emb
        self.out_dim = info_dim + op_dim + op_dim
        self.use_mean = use_mean
        self.use_info = use_info
        self.num_layers = num_layers

        if not self.use_mlp:
            self.input_emb = nn.Embedding(self.num_input_emb, op_dim)
            self.depth_emb = nn.Embedding(self.num_depth, op_dim)
        else:
            self.input_emb = construct_MLP(**self.mlp_schedule)
            self.depth_emb = construct_MLP(**self.mlp_schedule)
        self.device_holder = nn.Linear(op_dim, info_dim)
        
        self.qkv_emb = []
        self.mlp_emb = []
        self.dep_emb = []
        
        for i in range(self.depth):
            if (i == 0) or not self.share_emb:
                if not self.use_mlp:
                    self.qkv_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.mlp_emb.append(nn.Embedding(self.num_ratio, op_dim))
                    self.dep_emb.append(nn.Embedding(self.depth, op_dim))
                else:
                    self.qkv_emb.append(construct_MLP(**self.mlp_schedule))
                    self.mlp_emb.append(construct_MLP(**self.mlp_schedule))
                    self.dep_emb.append(construct_MLP(**self.mlp_schedule))
            else:
                self.qkv_emb.append(self.qkv_emb[0])
                self.mlp_emb.append(self.mlp_emb[0])
                self.dep_emb.append(self.dep_emb[0])

        self.qkv_emb = nn.ModuleList(self.qkv_emb)
        self.mlp_emb = nn.ModuleList(self.mlp_emb)
        self.dep_emb = nn.ModuleList(self.dep_emb)

        self.rnn = nn.LSTM(
            input_size=self.op_dim * 3,
            hidden_size=self.info_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
    
    def forward(self, archs):
        archs = self.device_holder.weight.new(archs)
        archs = archs.long() if not self.use_mlp else archs
        embs = []
        for i in range(self.depth):
            dep = np.array([i * 1.0]).repeat(archs.shape[0])
            dep = self.device_holder.weight.new(dep)
            dep = dep.long() if not self.use_mlp else dep

            if not self.use_mlp:
                qkv_emb = self.qkv_emb[i](archs[:, i * 2 + 2])
                mlp_emb = self.mlp_emb[i](archs[:, i * 2 + 3])
                dep_emb = self.dep_emb[i](dep)
            else:
                qkv_emb = self.qkv_emb[i](archs[:, i * 2 + 2].unsqueeze(-1))
                mlp_emb = self.mlp_emb[i](archs[:, i * 2 + 3].unsqueeze(-1))
                dep_emb = self.dep_emb[i](dep.unsqueeze(-1))

            emb = torch.cat((qkv_emb, mlp_emb, dep_emb), dim=1)
            embs.append(emb)

        embs = torch.stack(embs, dim=1)
        out, (h_n, _) = self.rnn(embs)

        if self.use_info:
            embs = h_n[0]
        else:
            if self.use_mean:
                embs = torch.mean(out, dim=1)
            else:
                embs = out[:, -1, :]

        if not self.use_mlp:
            input_emb = self.input_emb(archs[:, 0])
            depth_emb = self.depth_emb(archs[:, 1])
        else:
            input_emb = self.input_emb(archs[:, 0].unsqueeze(-1))
            depth_emb = self.depth_emb(archs[:, 1].unsqueeze(-1))

        embs = torch.cat((input_emb, depth_emb, embs), dim=-1)
        return embs


class GiTEmbedder(ArchEmbedder):
    NAME = "gite"

    def __init__(self, search_space, 
                 info_dim=32,
                 op_dim=32,
                 num_input_emb=1,
                 num_head=4,
                 num_ratio=4,
                 num_depth=3,
                 depth=16,
                 use_mlp=False,
                 mlp_schedule={},
                 share_emb=True,
                 share_emb_tf=True,
                 use_depth_emb=True,
                 use_bn=True,
                 schedule_cfg=None):
        super(GiTEmbedder, self).__init__(schedule_cfg)

        self.info_dim = info_dim
        self.op_dim = op_dim
        self.num_input_emb = num_input_emb
        self.num_head = num_head
        self.num_ratio = num_ratio
        self.depth = depth
        self.num_depth = num_depth
        self.use_mlp = use_mlp
        self.mlp_schedule = mlp_schedule
        self.share_emb = share_emb
        self.share_emb_tf = share_emb_tf
        self.use_depth_emb = use_depth_emb
        self.use_bn = use_bn
        self.out_dim = info_dim * 3 if use_depth_emb else info_dim

        self.input_info = nn.Parameter(torch.Tensor(self.info_dim).normal_())
        if not self.use_mlp:
            self.input_emb = nn.Embedding(self.num_input_emb, op_dim)
            self.depth_emb = nn.Embedding(self.num_depth, op_dim)
        else:
            self.input_emb = construct_MLP(**self.mlp_schedule)
            self.depth_emb = construct_MLP(**self.mlp_schedule)
        self.input_emb_tf = nn.Linear(op_dim, info_dim)
        self.depth_emb_tf = nn.Linear(op_dim, info_dim)

        self.q_emb = []
        self.k_emb = []
        self.v_emb = []
        self.mlp_emb = []
        self.q_emb_tf = []
        self.k_emb_tf = []
        self.v_emb_tf = []
        self.mlp_emb_tf = []

        self.bn = []
        self.input_bn = nn.BatchNorm1d(info_dim)
        
        for i in range(self.depth):
            self.bn.append(nn.BatchNorm1d(info_dim))
            if (i == 0) or not self.share_emb:
                if not self.use_mlp:
                    self.q_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.k_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.v_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.mlp_emb.append(nn.Embedding(self.num_ratio, op_dim))
                else:
                    self.q_emb.append(construct_MLP(**self.mlp_schedule))
                    self.k_emb.append(construct_MLP(**self.mlp_schedule))
                    self.v_emb.append(construct_MLP(**self.mlp_schedule))
                    self.mlp_emb.append(construct_MLP(**self.mlp_schedule))
            else:
                self.q_emb.append(self.q_emb[0])
                self.k_emb.append(self.k_emb[0])
                self.v_emb.append(self.v_emb[0])
                self.mlp_emb.append(self.mlp_emb[0])

            if (i == 0) or not self.share_emb_tf:
                self.q_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.k_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.v_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.mlp_emb_tf.append(nn.Linear(op_dim, info_dim))
            else:
                self.q_emb_tf.append(self.q_emb_tf[0])
                self.k_emb_tf.append(self.k_emb_tf[0])
                self.v_emb_tf.append(self.v_emb_tf[0])
                self.mlp_emb_tf.append(self.mlp_emb_tf[0])

        self.q_emb = nn.ModuleList(self.q_emb)
        self.k_emb = nn.ModuleList(self.k_emb)
        self.v_emb = nn.ModuleList(self.v_emb)
        self.mlp_emb = nn.ModuleList(self.mlp_emb)
        
        self.q_emb_tf = nn.ModuleList(self.q_emb_tf)
        self.k_emb_tf = nn.ModuleList(self.k_emb_tf)
        self.v_emb_tf = nn.ModuleList(self.v_emb_tf)
        self.mlp_emb_tf = nn.ModuleList(self.mlp_emb_tf)

        self.bn = nn.ModuleList(self.bn)

    def forward(self, archs):
        if not self.use_mlp:
            archs = self.input_emb_tf.weight.new(archs).long()
            input_attn = self.input_emb_tf(self.input_emb(archs[:, 0]))
        else:
            archs = self.input_emb_tf.weight.new(archs)
            input_attn = self.input_emb_tf(self.input_emb(archs[:, 0].unsqueeze(-1)))

        embs = torch.sigmoid(input_attn) * self.input_info
        embs = self.input_bn(embs) if self.use_bn else embs

        for i in range(self.depth):
            if not self.use_mlp:
                q_attn = self.q_emb_tf[i](self.q_emb[i](archs[:, i * 2 + 2]))
                k_attn = self.k_emb_tf[i](self.k_emb[i](archs[:, i * 2 + 2]))
                v_attn = self.v_emb_tf[i](self.v_emb[i](archs[:, i * 2 + 2]))
                mlp_attn = self.mlp_emb_tf[i](self.mlp_emb[i](archs[:, i * 2 + 3]))
            else:
                q_attn = self.q_emb_tf[i](self.q_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                k_attn = self.k_emb_tf[i](self.k_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                v_attn = self.v_emb_tf[i](self.v_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                mlp_attn = self.mlp_emb_tf[i](self.mlp_emb[i](archs[:, i * 2 + 3].unsqueeze(-1)))
            
            q_embs = torch.sigmoid(q_attn) * embs
            k_embs = torch.sigmoid(k_attn) * embs
            v_embs = torch.sigmoid(v_attn) * embs

            self_attn = torch.matmul(q_embs.unsqueeze(-1), k_embs.unsqueeze(1))
            self_attn = F.softmax(self_attn, dim=-1)
            
            embs = torch.matmul(self_attn, v_embs.unsqueeze(-1)).squeeze(-1)
            embs = torch.sigmoid(mlp_attn) * embs
            embs = self.bn[i](embs) if self.use_bn else embs

        if not self.use_mlp:
            depth_emb = self.depth_emb_tf(self.depth_emb(archs[:, 1]))
        else:
            depth_emb = self.depth_emb_tf(self.depth_emb(archs[:, 1].unsqueeze(-1)))

        if self.use_depth_emb:
            embs = torch.cat((embs, depth_emb, input_attn), dim=-1)
        embs = F.normalize(embs, 2, dim=0)
        return embs


class Trans_ViT_SimpleSeqEmbedder(ArchEmbedder):
    NAME = "trans-vit-seq"

    def __init__(self, search_space, dim, task_mlp_schedule, schedule_cfg=None):
        super(Trans_ViT_SimpleSeqEmbedder, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.out_dim = dim + task_mlp_schedule["out_dim"]
        self.task_mlp_schedule = task_mlp_schedule
        self.task_mlp = construct_MLP(**self.task_mlp_schedule)
        self._placeholder_tensor = nn.Parameter(torch.zeros(1))

    def forward(self, archs):
        task_emb_len = 2048
        archs = self._placeholder_tensor.new(archs)
        task_emb = self.task_mlp(archs[:, : task_emb_len])
        archs = archs[:, task_emb_len :]
        embs = torch.cat((archs, task_emb), dim=-1)
        return embs


class Trans_ViT_LSTMSeqSeparateEmbedder(ArchEmbedder):
    NAME = "trans-vit-lstm-separate"

    def __init__(self, search_space, 
                 info_dim=32,
                 op_dim=32,
                 num_input_emb=1,
                 num_head=4,
                 num_ratio=4,
                 num_depth=3,
                 depth=16,
                 num_layers=1,
                 use_mlp=False,
                 mlp_schedule={},
                 task_mlp_schedule={},
                 share_emb=True,
                 use_info=False,
                 use_mean=False,
                 schedule_cfg=None):
        super(Trans_ViT_LSTMSeqSeparateEmbedder, self).__init__(schedule_cfg)

        self.info_dim = info_dim
        self.op_dim = op_dim
        self.num_input_emb = num_input_emb
        self.num_head = num_head
        self.num_ratio = num_ratio
        self.depth = depth
        self.num_depth = num_depth
        self.use_mlp = use_mlp
        self.mlp_schedule = mlp_schedule
        self.share_emb = share_emb
        self.out_dim = info_dim + op_dim + op_dim
        self.use_mean = use_mean
        self.use_info = use_info
        self.num_layers = num_layers
        self.task_mlp_schedule = task_mlp_schedule
        self.out_dim = self.out_dim + info_dim

        self.task_mlp = construct_MLP(**self.task_mlp_schedule)

        if not self.use_mlp:
            self.input_emb = nn.Embedding(self.num_input_emb, op_dim)
            self.depth_emb = nn.Embedding(self.num_depth, op_dim)
        else:
            self.input_emb = construct_MLP(**self.mlp_schedule)
            self.depth_emb = construct_MLP(**self.mlp_schedule)
        self.device_holder = nn.Linear(op_dim, info_dim)
        
        self.qkv_emb = []
        self.mlp_emb = []
        self.dep_emb = []
        
        for i in range(self.depth):
            if (i == 0) or not self.share_emb:
                if not self.use_mlp:
                    self.qkv_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.mlp_emb.append(nn.Embedding(self.num_ratio, op_dim))
                    self.dep_emb.append(nn.Embedding(self.depth, op_dim))
                else:
                    self.qkv_emb.append(construct_MLP(**self.mlp_schedule))
                    self.mlp_emb.append(construct_MLP(**self.mlp_schedule))
                    self.dep_emb.append(construct_MLP(**self.mlp_schedule))
            else:
                self.qkv_emb.append(self.qkv_emb[0])
                self.mlp_emb.append(self.mlp_emb[0])
                self.dep_emb.append(self.dep_emb[0])

        self.qkv_emb = nn.ModuleList(self.qkv_emb)
        self.mlp_emb = nn.ModuleList(self.mlp_emb)
        self.dep_emb = nn.ModuleList(self.dep_emb)

        self.rnn = nn.LSTM(
            input_size=self.op_dim * 3,
            hidden_size=self.info_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
    
    def forward(self, archs):
        task_emb_len = 2048
        archs = self.device_holder.weight.new(archs)
        task_emb = self.task_mlp(archs[:, : task_emb_len])
        archs = archs[:, task_emb_len :]
        archs = archs.long() if not self.use_mlp else archs
        embs = []
        for i in range(self.depth):
            dep = np.array([i * 1.0]).repeat(archs.shape[0])
            dep = self.device_holder.weight.new(dep)
            dep = dep.long() if not self.use_mlp else dep

            if not self.use_mlp:
                qkv_emb = self.qkv_emb[i](archs[:, i * 2 + 2])
                mlp_emb = self.mlp_emb[i](archs[:, i * 2 + 3])
                dep_emb = self.dep_emb[i](dep)
            else:
                qkv_emb = self.qkv_emb[i](archs[:, i * 2 + 2].unsqueeze(-1))
                mlp_emb = self.mlp_emb[i](archs[:, i * 2 + 3].unsqueeze(-1))
                dep_emb = self.dep_emb[i](dep.unsqueeze(-1))

            emb = torch.cat((qkv_emb, mlp_emb, dep_emb), dim=1)
            embs.append(emb)

        embs = torch.stack(embs, dim=1)
        out, (h_n, _) = self.rnn(embs)

        if self.use_info:
            embs = h_n[0]
        else:
            if self.use_mean:
                embs = torch.mean(out, dim=1)
            else:
                embs = out[:, -1, :]

        if not self.use_mlp:
            input_emb = self.input_emb(archs[:, 0])
            depth_emb = self.depth_emb(archs[:, 1])
        else:
            input_emb = self.input_emb(archs[:, 0].unsqueeze(-1))
            depth_emb = self.depth_emb(archs[:, 1].unsqueeze(-1))

        embs = torch.cat((input_emb, depth_emb, embs, task_emb), dim=-1)
        return embs


class TransGiTEmbedder(ArchEmbedder):
    NAME = "trans-gite"

    def __init__(self, search_space, 
                 info_dim=32,
                 op_dim=32,
                 num_input_emb=1,
                 num_head=4,
                 num_ratio=4,
                 num_depth=3,
                 depth=16,
                 use_mlp=False,
                 mlp_schedule={},
                 task_mlp_schedule={},
                 share_emb=True,
                 share_emb_tf=True,
                 use_depth_emb=True,
                 use_bn=True,
                 use_task_emb=True,
                 schedule_cfg=None):
        super(TransGiTEmbedder, self).__init__(schedule_cfg)

        self.info_dim = info_dim
        self.op_dim = op_dim
        self.num_input_emb = num_input_emb
        self.num_head = num_head
        self.num_ratio = num_ratio
        self.depth = depth
        self.num_depth = num_depth
        self.use_mlp = use_mlp
        self.mlp_schedule = mlp_schedule
        self.task_mlp_schedule = task_mlp_schedule
        self.share_emb = share_emb
        self.share_emb_tf = share_emb_tf
        self.use_depth_emb = use_depth_emb
        self.use_bn = use_bn
        self.use_task_emb = use_task_emb
        self.out_dim = info_dim * 3 if use_depth_emb else info_dim
        self.out_dim = self.out_dim + info_dim if use_task_emb else self.out_dim

        self.input_info = nn.Parameter(torch.Tensor(self.info_dim).normal_())
        if not self.use_mlp:
            self.input_emb = nn.Embedding(self.num_input_emb, op_dim)
            self.depth_emb = nn.Embedding(self.num_depth, op_dim)
        else:
            self.input_emb = construct_MLP(**self.mlp_schedule)
            self.depth_emb = construct_MLP(**self.mlp_schedule)
        self.input_emb_tf = nn.Linear(op_dim, info_dim)
        self.depth_emb_tf = nn.Linear(op_dim, info_dim)

        self.q_emb = []
        self.k_emb = []
        self.v_emb = []
        self.mlp_emb = []
        self.q_emb_tf = []
        self.k_emb_tf = []
        self.v_emb_tf = []
        self.mlp_emb_tf = []

        self.bn = []
        self.input_bn = nn.BatchNorm1d(info_dim)
        
        for i in range(self.depth):
            self.bn.append(nn.BatchNorm1d(info_dim))
            if (i == 0) or not self.share_emb:
                if not self.use_mlp:
                    self.q_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.k_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.v_emb.append(nn.Embedding(self.num_head, op_dim))
                    self.mlp_emb.append(nn.Embedding(self.num_ratio, op_dim))
                else:
                    self.q_emb.append(construct_MLP(**self.mlp_schedule))
                    self.k_emb.append(construct_MLP(**self.mlp_schedule))
                    self.v_emb.append(construct_MLP(**self.mlp_schedule))
                    self.mlp_emb.append(construct_MLP(**self.mlp_schedule))
            else:
                self.q_emb.append(self.q_emb[0])
                self.k_emb.append(self.k_emb[0])
                self.v_emb.append(self.v_emb[0])
                self.mlp_emb.append(self.mlp_emb[0])

            if (i == 0) or not self.share_emb_tf:
                self.q_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.k_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.v_emb_tf.append(nn.Linear(op_dim, info_dim))
                self.mlp_emb_tf.append(nn.Linear(op_dim, info_dim))
            else:
                self.q_emb_tf.append(self.q_emb_tf[0])
                self.k_emb_tf.append(self.k_emb_tf[0])
                self.v_emb_tf.append(self.v_emb_tf[0])
                self.mlp_emb_tf.append(self.mlp_emb_tf[0])

        self.q_emb = nn.ModuleList(self.q_emb)
        self.k_emb = nn.ModuleList(self.k_emb)
        self.v_emb = nn.ModuleList(self.v_emb)
        self.mlp_emb = nn.ModuleList(self.mlp_emb)
        
        self.q_emb_tf = nn.ModuleList(self.q_emb_tf)
        self.k_emb_tf = nn.ModuleList(self.k_emb_tf)
        self.v_emb_tf = nn.ModuleList(self.v_emb_tf)
        self.mlp_emb_tf = nn.ModuleList(self.mlp_emb_tf)

        self.bn = nn.ModuleList(self.bn)

        self.task_emb = construct_MLP(**self.task_mlp_schedule)
        self.task_emb_tf = nn.Linear(op_dim, info_dim)

    def forward(self, archs):
        task_emb_len = 2048
        if not self.use_mlp:
            archs = self.input_emb_tf.weight.new(archs)
            tasks = archs[:, : task_emb_len]
            archs = archs[:, task_emb_len :].long()
            input_attn = self.input_emb_tf(self.input_emb(archs[:, 0]))
        else:
            archs = self.input_emb_tf.weight.new(archs)
            tasks = archs[:, : task_emb_len]
            archs = archs[:, task_emb_len :]
            input_attn = self.input_emb_tf(self.input_emb(archs[:, 0].unsqueeze(-1)))

        #tasks = F.normalize(tasks, dim=-1)
        input_info = self.task_emb_tf(self.task_emb(tasks))
        embs = torch.sigmoid(input_attn) * self.input_info
        embs = self.input_bn(embs) if self.use_bn else embs

        for i in range(self.depth):
            if not self.use_mlp:
                q_attn = self.q_emb_tf[i](self.q_emb[i](archs[:, i * 2 + 2]))
                k_attn = self.k_emb_tf[i](self.k_emb[i](archs[:, i * 2 + 2]))
                v_attn = self.v_emb_tf[i](self.v_emb[i](archs[:, i * 2 + 2]))
                mlp_attn = self.mlp_emb_tf[i](self.mlp_emb[i](archs[:, i * 2 + 3]))
            else:
                q_attn = self.q_emb_tf[i](self.q_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                k_attn = self.k_emb_tf[i](self.k_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                v_attn = self.v_emb_tf[i](self.v_emb[i](archs[:, i * 2 + 2].unsqueeze(-1)))
                mlp_attn = self.mlp_emb_tf[i](self.mlp_emb[i](archs[:, i * 2 + 3].unsqueeze(-1)))
            
            q_embs = torch.sigmoid(q_attn) * embs
            k_embs = torch.sigmoid(k_attn) * embs
            v_embs = torch.sigmoid(v_attn) * embs

            self_attn = torch.matmul(q_embs.unsqueeze(-1), k_embs.unsqueeze(1))
            self_attn = F.softmax(self_attn, dim=-1)
            
            embs = torch.matmul(self_attn, v_embs.unsqueeze(-1)).squeeze(-1)
            embs = torch.sigmoid(mlp_attn) * embs
            embs = self.bn[i](embs) if self.use_bn else embs

        if not self.use_mlp:
            depth_emb = self.depth_emb_tf(self.depth_emb(archs[:, 1]))
        else:
            depth_emb = self.depth_emb_tf(self.depth_emb(archs[:, 1].unsqueeze(-1)))

        if self.use_depth_emb:
            embs = torch.cat((embs, depth_emb, input_attn), dim=-1)
        if self.use_task_emb:
            embs = torch.cat((embs, input_info), dim=-1)
        embs = F.normalize(embs, 2, dim=0)
        return embs
