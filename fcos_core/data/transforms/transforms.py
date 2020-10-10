# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
import torch.nn.functional as nn_F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        d = -1
        if len(image_size)==4:
            c, w, h, d = image_size
            size_pair = (w, h, d)
        elif len(image_size)==3:
            c, w, h = image_size
            size_pair = (w, h)
        else:
            assert len(image_size)==3, 'image shape should be C, H, W for 2D image or C, H, W, D for 3D image'
        size = self.min_size
        max_size = self.max_size

        if max_size is not None:
            min_original_size = float(min(size_pair))
            max_original_size = float(max(size_pair))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if min(size_pair) == size:
            return size_pair
        ratio = size / min(size_pair)
        return tuple([int(ratio * _) for _ in size_pair])

    def __call__(self, image, target=None):
        size = self.get_size(image.size())
        mode = 'bilinear'
        if len(size) == 3:
            mode = 'trilinear'

        if (size==image.size()[1:]):
            pass
        else:
            image = nn_F.interpolate(image.unsqueeze(0), size=size, mode=mode)[0]
        if isinstance(target, list):
            target = [t.resize(size) for t in target]
        elif target is None:
            return image
        else:
            target = target.resize(size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        if isinstance(image, torch.Tensor):
            return image, target
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target=None):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image
        return image, target
