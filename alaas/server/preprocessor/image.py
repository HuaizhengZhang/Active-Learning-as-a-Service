"""
Data processing and augmentation toolkit
For image data.
@author huangyz0918 (huangyz0918@gmail.com)
@date 20/07/2022
"""

import cv2
import numpy as np
from torchvision import transforms

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def load_image_data_as_np(image_path):
    return np.array([img_transform(cv2.imread(image_path)).numpy()], dtype=np.float32)


def load_images_data_as_np(data_pool):
    img_list = []
    img_md5_list = []
    img_uuid_list = []
    for data in data_pool:
        img_md5_list.append(data[0])
        img_uuid_list.append(data[1])
        img_list.append(img_transform(cv2.imread(data[2])).numpy())
    img_list = np.array(img_list, dtype=np.float32)
    return img_md5_list, img_uuid_list, img_list
