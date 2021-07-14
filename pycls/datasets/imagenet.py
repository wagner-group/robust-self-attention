#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""ImageNet dataset."""

import os

import cv2
import numpy as np
import pycls.core.logging as logging
import pycls.datasets.transforms as transforms
import torch.utils.data
from pycls.core.config import cfg


logger = logging.get_logger(__name__)

# Constants for lighting normalization on ImageNet (in RGB order)
# https://github.com/facebookarchive/fb.resnet.torch/blob/master/datasets/imagenet.lua
_EIG_VALS = [[0.2175, 0.0188, 0.0045]]
_EIG_VECS = [
    [-0.5675, 0.7192, 0.4009],
    [-0.5808, -0.0045, -0.8140],
    [-0.5836, -0.6948, 0.4203],
]


class ImageNet(torch.utils.data.Dataset):
    """ImageNet dataset."""

    def __init__(self, data_path, split):
        assert os.path.exists(data_path), "Data path '{}' not found".format(data_path)
        logger.info("Constructing ImageNet {}...".format(split))
        self._data_path, self._split = data_path, split
        self._construct_imdb()

    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        split_path = os.path.join(self._data_path, self._split)
        logger.info("{} data path: {}".format(self._split, split_path))
        # Images are stored per class in subdirs (format: n<number>)
        self._class_ids = sorted(filter(lambda d: os.path.isdir(os.path.join(split_path, d)),
                                        os.listdir(split_path)))
        self._class_ids = np.array(self._class_ids)
        np.random.seed(0)
        classes = np.sort(np.random.permutation(len(self._class_ids))[:cfg.MODEL.NUM_CLASSES])
        self._class_ids = self._class_ids[classes]
        # Map ImageNet class ids to contiguous ids
        self._class_id_cont_id = {v: i for i, v in enumerate(self._class_ids)}
        # Construct the image db
        self._imdb = []
        for class_id in self._class_ids:
            cont_id = self._class_id_cont_id[class_id]
            im_dir = os.path.join(split_path, class_id)
            for im_name in os.listdir(im_dir):
                im_path = os.path.join(im_dir, im_name)
                self._imdb.append({"im_path": im_path, "class": cont_id})
        if self._split == 'val' and cfg.TEST.NUM_IMAGES > 0:
            image_ids = np.random.permutation(len(self._imdb))[:cfg.TEST.NUM_IMAGES]
            self._imdb = [self._imdb[i] for i in image_ids]
        logger.info("Number of images: {}".format(len(self._imdb)))
        logger.info("Number of classes: {}".format(len(self._class_ids)))

    def _prepare_im(self, im):
        """Prepares the image for network input (HWC/BGR/int -> CHW/BGR/float)."""
        # Convert HWC/BGR/int to HWC/RGB/float format for applying transforms
        im = im[:, :, ::-1].astype(np.float32) / 255
        # Train and test setups differ
        train_size, test_size = cfg.TRAIN.IM_SIZE, cfg.TEST.IM_SIZE
        if self._split == "train":
            # For training use random_sized_crop, horizontal_flip, augment, lighting
            im = transforms.random_sized_crop(im, train_size)
            im = transforms.horizontal_flip(im, prob=0.5)
            im = transforms.augment(im, cfg.TRAIN.AUGMENT)
            im = transforms.lighting(im, cfg.TRAIN.PCA_STD, _EIG_VALS, _EIG_VECS)
        else:
            # For testing use scale and center crop
            im = transforms.scale_and_center_crop(im, test_size, train_size)
        # For training and testing use color normalization
        im = transforms.color_norm(im, cfg.DATA_LOADER.MEAN, cfg.DATA_LOADER.STD)
        # Convert HWC/RGB/float to CHW/BGR/float format
        if cfg.DATA_LOADER.BGR:
            im = im[:, :, ::-1]
        im = np.ascontiguousarray(im).transpose([2, 0, 1])
        return im

    def __getitem__(self, index):
        # Load the image
        im = cv2.imread(self._imdb[index]["im_path"])
        # Prepare the image for training / testing
        im = self._prepare_im(im)
        # Retrieve the label
        label = self._imdb[index]["class"]
        return im, label

    def __len__(self):
        return len(self._imdb)
