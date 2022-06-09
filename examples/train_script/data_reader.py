#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Li Yuanming
Email: yuanmingleee@gmail.com
Date: June 9, 2022
"""
import pandas as pd
from PIL.Image import Image
from torchvision.datasets import VisionDataset


class DataFrameDataset(VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root=root, transform=transform, target_transform=target_transform)
        df = pd.read_csv(root)
        image_paths = df.iloc[:, 0].values.tolist()
        labels = df.iloc[:, 1].values.tolist()
        classes, class_to_idx = self._find_classes(labels)
        labels = list(map(class_to_idx, df.iloc[:, 1].values.tolist()))

        self.samples = list(zip(image_paths, labels))
        self.classes = classes
        self.class_to_idx = class_to_idx

    @classmethod
    def _find_classes(cls, labels: list):
        """
        Finds the class labels in a dataset.
        Adapted from :code:`torchvision.datasets.folder.DatasetFolder._find_classes`

        Args:
            labels (list): A list of label classes.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a label number.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = list(set(labels))
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            images and its labels if include_label is True
            images only if include_label is False
        """
        image, label = self.samples[index]
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.samples)
