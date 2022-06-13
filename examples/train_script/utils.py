#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: June 9, 2022
"""
from functools import partialmethod

from torchvision import transforms


def build_train_transform(crop_size, mean=0., std=1.):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


def build_test_transform(resize_size, center_crop_size=None, mean=0., std=1.):
    center_crop_size = center_crop_size or resize_size
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(center_crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return transform


def _partialclass(cls, *args, **kwds):

    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwds)

    return NewCls


def set_seed(seed):
    """
    Seeding everything for PyTorch

    Reference: https://pytorch.org/docs/stable/notes/randomness.html
    """
    import numpy as np
    import torch
    import torch.utils.data

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    torch.utils.data.DataLoader = _partialclass(torch.utils.data.DataLoader, generator=g)


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
