#!/usr/bin/env python3
# coding: utf-8

from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import torch


def load_dataset(data_path, test_path, batch_size = 1, distributed = 0):
    """Load training and validation dataset.

    Parameters
    ----------
    data_path : str
    batch_size : int

    Returns
    -------
    train_dataloader: torch.utils.data.DataLoader
    val_dataloader: torch.utils.data.DataLoader

    """
    transform = transforms.Compose([
        transforms.ToTensor()])
    nusc_train_val = NuscDataset(data_path, transform)
    nusc_test = NuscDataset(test_path, transform)

    train_size = int(0.9 * len(nusc_train_val))
    val_size = len(nusc_train_val) - train_size

    #Gz
    #train_size = 16790
    #val_size = 1868
    print("train-size",train_size)
    print("val-size",val_size)
    train_dataset, val_dataset = random_split(nusc_train_val, [train_size, val_size])
    test_dataset = nusc_test

    train_sampler = None
    val_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=16)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=8)

    return train_dataloader, val_dataloader, test_dataloader


def onehot(data, n=5):
    """Convert label to onehot_vector

    Originally implemented in
    https://github.com/yunlongdong/FCN-pytorch/blob/master/onehot.py

    Parameters
    ----------
    data : numpy.ndarray
        np.ndarray with int stored in label
    n : int, optional
        [description], by default 5

    Returns
    -------
    buf numpy.ndarray
        onehot vector of class

    """
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size) * n + data.ravel()
    buf.ravel()[nmsk] = 1
    return buf


class NuscDataset(Dataset):
    """Nuscenes dataset

    Parameters
    ----------
    data_path : str
        Path of generated dataset.
    transform : torchvision.transforms.Compose, optional
        Currently it only converts a numpy.ndarray to a torch tensor,
        not really needed., by default None

    """

    def __init__(self, data_path, transform=None, mode='train'):
        self.data_path = data_path
        self.transform = transform
        if mode == 'train':
            self.in_feature_paths = list(sorted(Path(self.data_path).glob("in_feature/*.npy")))
            print(self.in_feature_paths)
        else:
            self.test_feature_paths = list(sorted(Path(self.data_path).glob("in_feature/*.npy")))


    def __len__(self):
        return len(self.in_feature_paths)

    def __getitem__(self, idx):
        in_feature_path = self.in_feature_paths[idx]
        out_feature_path = in_feature_path.parent.parent / \
            'out_feature' / in_feature_path.name
        #print(in_feature_path)
        in_feature = np.load(str(in_feature_path))
        in_feature = in_feature.astype(np.float32)
        #print(in_feature_path, in_feature.shape, in_feature.size)
        size = in_feature.shape
        if size[2] != 4:
            print("input feture error")
        out_feature = np.load(str(out_feature_path))
        one_hot_class = onehot(out_feature[..., 4].astype(np.int8), 5)

        out_feature = np.concatenate(
            [out_feature[..., 0:4],
             one_hot_class, out_feature[..., 5:]], 2)

        out_feature = out_feature.astype(np.float32)

        if self.transform:
            in_feature = self.transform(in_feature)
            out_feature = self.transform(out_feature)

        return in_feature, out_feature
