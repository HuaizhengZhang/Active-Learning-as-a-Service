#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: June 9, 2022
"""
import argparse
import logging
import os
import sys
import time
from ast import literal_eval

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from utils import (
    accuracy,
    AvgrageMeter,
    build_train_transform,
    build_test_transform,
    set_seed,
)
from data_reader import DataFrameDataset


def parse_args():
    parser = argparse.ArgumentParser("Train ResNet")
    parser.add_argument('--train_csv_path', type=str, required=True, help='CSV path for training')
    parser.add_argument('--normalize_mean', type=str, default='(0., 0., 0.)', help='Normalization mean')
    parser.add_argument('--normalize_std', type=str, default='(1., 1., 1.)', help='Normalization std')
    # Data pre-processing
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--val_batch_size', type=int, default=64, help='Validation size')
    parser.add_argument('--split_ratio', type=float, default=0.9, help='Split ratio for training set/total')
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(), help='Number of workers for data loading')
    parser.add_argument('--image_resize', type=int, default=224, help='Size of image resizing')
    # Optimizer
    parser.add_argument('--lr', type=float, default=0.025, help='init learning rate')
    parser.add_argument('--eta_min', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
    # Training
    parser.add_argument('--cuda', action='store_true', default=False, help='Enable CUDA')
    parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
    # Logging
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    args.normalize_mean = literal_eval(args.normalize_mean)
    args.normalize_std = literal_eval(args.normalize_std)
    args.save = f'eval-{args.save}-{time.strftime("%Y%m%d-%H%M%S")}'
    os.mkdir(args.save)

    if not torch.cuda.is_available():
        logging.warning('no gpu device available')
        args.cuda = False
    setattr(args, 'device', torch.device('cuda' if args.cuda else 'cpu'))

    return args


def setup_dataset(
        train_csv_path,
        split_ratio,
        batch_size,
        val_batch_size,
        image_resize,
        num_workers=os.cpu_count(),
        mean=0., std=1.
):
    """Setup dataset.
    Use this method to override :code:`setup_dataset` in :code:`medical.train_search`.

    """
    train_transform = build_train_transform(crop_size=image_resize, mean=mean, std=std)
    # test_transform = build_test_transform(resize_size=image_resize, mean=mean, std=std)
    train_set = DataFrameDataset(root=train_csv_path, transform=train_transform)
    train_size = int(split_ratio * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, (train_size, val_size))

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=val_batch_size, num_workers=num_workers, pin_memory=True,
                            shuffle=False)

    return train_loader, val_loader


def setup_modules(args):
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=args.num_classes)
    if args.cuda:
        model = model.to(args.device)

    return model


def setup_optimizers(model: 'torch.nn.Module', lr, momentum, weight_decay, epochs, eta_min=0):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=eta_min)

    return optimizer, scheduler


def train(train_loader, model, criterion, optimizer, report_freq, epoch, device, **kwargs):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.train()

    with logging_redirect_tqdm():
        with tqdm(train_loader, desc=f'train {epoch}') as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                logits = model(inputs)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                prec1, = accuracy(logits, targets, topk=(1,))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

                tbar.set_postfix({'loss': loss.item(), 'top1': prec1.item()})
                if step % report_freq == 0:
                    logging.info('train %03d loss_avg=%e top1_avg=%f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(val_loader, model, criterion, report_freq, epoch, device, **kwargs):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.eval()

    with logging_redirect_tqdm():
        with tqdm(val_loader, desc=f'Val  {epoch}') as tbar:
            for step, (inputs, targets) in enumerate(tbar):
                inputs, targets = inputs.to(device), targets.to(device)

                logits = model(inputs)
                loss = criterion(logits, targets)

                prec1, = accuracy(logits, targets, topk=(1,))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)

                tbar.set_postfix({'loss': loss.item(), 'top1': prec1.item()})
                if step % report_freq == 0:
                    logging.info('valid %03d loss_avg=%e top1_avg=%f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


if __name__ == "__main__":
    args = parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.seed is not None:
        set_seed(args.seed)
    logging.info(args)

    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = setup_dataset(
        train_csv_path=args.train_csv_path,
        split_ratio=args.split_ratio,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        image_resize=args.image_resize,
        mean=args.normalize_mean,
        std=args.normalize_std,
    )
    # set num_classes
    setattr(args, 'num_classes', len(train_loader.dataset.classes))
    print('Number of classes:', args.num_classes)

    model = setup_modules(args)

    optimizer, scheduler = setup_optimizers(
        model, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, epochs=args.epochs,
        eta_min=args.eta_min,
    )

    epoch_best, valid_acc_best = 0, 0
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        train_acc, train_obj = train(train_loader, model, criterion, optimizer, epoch=epoch, **vars(args))
        logging.info('train_acc %f', train_acc)

        valid_acc, valid_obj = infer(test_loader, model, criterion, epoch=epoch, **vars(args))
        logging.info('valid_acc %f', valid_acc)

        if valid_acc > valid_acc_best:
            torch.save(model.state_dict(), os.path.join(args.save, 'weights.pt'))
            valid_acc_best = valid_acc
            epoch_best = epoch

        if epoch - epoch_best > 30:
            print('Accuracy not improving in 30 epochs, early stop.')
            exit(0)
