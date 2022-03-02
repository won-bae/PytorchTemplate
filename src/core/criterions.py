import torch
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F


class CrossEntropyLoss(_Loss):
    def __init__(self, weight=None, gamma=1., temp=1., reduction='mean', eps=1e-6):
        super(_Loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.temp = temp
        self.reduction = reduction
        self.eps = eps

    def forward(self, preds, labels):
        preds = preds / self.temp
        if self.gamma >= 1.:
            loss = F.cross_entropy(
                preds, labels, weight=self.weight, reduction=self.reduction)
        else:
            log_prob = preds - torch.logsumexp(preds, dim=1, keepdim=True)
            log_prob = log_prob * self.gamma
            loss = F.nll_loss(
                log_prob, labels, weight=self.weight, reduction=self.reduction)

        losses = {'loss': loss}
        return losses


class FocalLoss(_Loss):
    def __init__(self, weight=None, alpha=1., gamma=1., reduction='mean'):
        super(_Loss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, labels):
        log_prob = F.log_softmax(preds, dim=-1)
        prob = torch.exp(log_prob)
        loss = F.nll_loss(
            (self.alpha * (1 - prob) ** self.gamma) * log_prob, labels,
            weight=self.weight, reduction = self.reduction)
        losses = {'loss': loss}
        return losses


class CustomCriterion(_Loss):
    def __init__(self):
        super(_Loss, self).__init__()
        self.criterion = CrossEntropyLoss()

    def forward(self, output_dict):
        preds = output_dict['logits']
        labels = output_dict['labels']
        losses = self.criterion(preds=preds, labels=labels)

        return losses
