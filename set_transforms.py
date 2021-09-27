from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image

from torch.utils.data import Dataset

import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2


height = 512
width = 512
means = (0.485, 0.456, 0.406)
stds = (0.229, 0.224, 0.225)

data_transforms_default = {
    'train': A.Compose(
        [
            A.SmallestMaxSize(max_size=160),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    'val': A.Compose(
        [
            A.SmallestMaxSize(max_size=160),
            A.CenterCrop(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
}

data_transforms_cube = {
    'train': A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST, p=1),
            A.HorizontalFlip(p=0.5),
            A.IAAAdditiveGaussianNoise(p=0.2),
            A.IAAPerspective(p=0.5),
            A.ShiftScaleRotate(p=0.5, shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=0, value=0, mask_value=0),
            A.OneOf(
                [
                    A.RandomBrightness(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.5,
            ),
            A.OneOf(
                [
                    A.RandomContrast(p=1),
                    A.HueSaturationValue(p=1),
                ],
                p=0.5,
            ),
            A.Normalize(mean=means, std=stds, p=1),
            ToTensorV2(),
        ], p=1
    ),
    'val': A.Compose(
        [
            A.Resize(height, width, cv2.INTER_NEAREST, p=1),
            A.Normalize(mean=means, std=stds, p=1),
            ToTensorV2(),
        ], p=1
    ),
}