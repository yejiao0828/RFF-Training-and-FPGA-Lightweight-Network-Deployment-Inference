import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""

    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)

        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()

        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))
        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size

        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss



import torch
import math


def npair_loss(scores, labels):
    """
    计算N-pair Loss
    :param scores: tensor类型，表示模型输出的分数，shape为(batch_size, num_items)
    :param labels: tensor类型，表示每个样本的真实标签，shape为(batch_size, num_items)
    :return: N-pair Loss
    """
    batch_size, num_items = scores.shape

    # 确保输入的 scores 和 labels 形状正确
    assert labels.shape == (batch_size, num_items), "labels shape must match (batch_size, num_items)"
    assert scores.shape == (batch_size, num_items), "scores shape must match (batch_size, num_items)"

    # 计算相似度矩阵
    sim_mat = torch.matmul(scores, scores.t())

    # 构造mask矩阵
    mask = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()

    # 排除掉相同的样本
    mask = mask.fill_diagonal_(0)

    # 计算N-pair Loss
    neg_sim = sim_mat.masked_fill(mask.bool(), float('-inf'))
    neg_sim = torch.logsumexp(neg_sim, dim=1, keepdim=True)

    # 正样本相似度
    pos_sim = sim_mat.diag().unsqueeze(1)

    # N-pair loss公式
    loss = neg_sim - pos_sim + torch.log(torch.tensor(num_items - 1, dtype=torch.float32))

    return loss.mean()