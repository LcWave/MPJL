import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

class WithIndex(Dataset):
    def __init__(self, base): self.base = base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, y, idx

class GenesisSegLoader(object):
    def __init__(self, data_path, win_size, step=1, mode="train"):
        """
        Args:
            data_path (str): Path to dataset folder
            win_size (int): Window size (segment size)
            step (int): Not used here but reserved
            mode (str): 'train', 'val' or 'test'
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load data
        train_data = np.load(data_path + "/Genesis_train.npy")
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(data_path + "/Genesis_test.npy")
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.val = self.test  # Validation use test set

        # Labels
        self.test_labels = np.load(data_path + "/Genesis_test_label.npy")  # 0/1 label

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            segment = self.create_segment(self.train, index)
            return np.float32(segment), np.float32(segment)  # No label needed for training

        elif self.mode == "val":
            segment = self.create_segment(self.val, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

        elif self.mode == "test":
            segment = self.create_segment(self.test, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

    def create_segment(self, data, index):
        start_idx = index - self.win_size // 2
        end_idx = index + self.win_size // 2

        if start_idx < 0:
            segment = np.tile(data[0], (self.win_size, 1))
            segment[-start_idx - 1:] = data[:end_idx + 1]
        elif end_idx >= len(data):
            segment = np.tile(data[-1], (self.win_size, 1))
            segment[:len(data) - start_idx] = data[start_idx:]
        else:
            segment = data[start_idx:end_idx + 1]

        segment = segment[:self.win_size]

        return segment

class GenesisPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "Genesis_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "Genesis_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "Genesis_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class HAIPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "HAI_train.npy"), mmap_mode='r')  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "HAI_test.npy"), mmap_mode='r')    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        # self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        # self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        # self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.train = train_data
        self.test = test_data
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "HAI_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class HAISegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/HAI_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/HAI_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/HAI_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step=1, mode="train"):
        """
        Args:
            data_path (str): Path to dataset folder
            win_size (int): Window size (segment size)
            step (int): Not used here but reserved
            mode (str): 'train', 'val' or 'test'
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load data
        train_data = np.load(data_path + "//MSL_train.npy")
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(data_path + "//MSL_test.npy")
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.val = self.test  # Validation use test set

        # Labels
        self.test_labels = np.load(data_path + "//MSL_test_label.npy")  # 0/1 label

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            segment = self.create_segment(self.train, index)
            return np.float32(segment), np.float32(segment)  # No label needed for training

        elif self.mode == "val":
            segment = self.create_segment(self.val, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

        elif self.mode == "test":
            segment = self.create_segment(self.test, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

    def create_segment(self, data, index):
        start_idx = index - self.win_size // 2
        end_idx = index + self.win_size // 2

        if start_idx < 0:
            segment = np.tile(data[0], (self.win_size, 1))
            segment[-start_idx - 1:] = data[:end_idx + 1]
        elif end_idx >= len(data):
            segment = np.tile(data[-1], (self.win_size, 1))
            segment[:len(data) - start_idx] = data[start_idx:]
        else:
            segment = data[start_idx:end_idx + 1]

        segment = segment[:self.win_size]

        return segment

class MSLPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        # self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "MSL_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "MSL_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        # self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        # self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        # self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        # self.val = self.test

        self.train = train_data
        self.test = test_data
        self.val = self.test  # Validation use test set

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "MSL_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class GECCOSegLoader(object):
    def __init__(self, data_path, win_size, step=1, mode="train"):
        """
        Args:
            data_path (str): Path to dataset folder
            win_size (int): Window size (segment size)
            step (int): Not used here but reserved
            mode (str): 'train', 'val' or 'test'
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load data
        train_data = np.load(data_path + "/GECCO_train.npy")
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(data_path + "/GECCO_test.npy")
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.val = self.test  # Validation use test set

        # Labels
        self.test_labels = np.load(data_path + "/GECCO_test_label.npy")  # 0/1 label

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            segment = self.create_segment(self.train, index)
            return np.float32(segment), np.float32(segment)  # No label needed for training

        elif self.mode == "val":
            segment = self.create_segment(self.val, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

        elif self.mode == "test":
            segment = self.create_segment(self.test, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

    def create_segment(self, data, index):
        start_idx = index - self.win_size // 2
        end_idx = index + self.win_size // 2

        if start_idx < 0:
            segment = np.tile(data[0], (self.win_size, 1))
            segment[-start_idx - 1:] = data[:end_idx + 1]
        elif end_idx >= len(data):
            segment = np.tile(data[-1], (self.win_size, 1))
            segment[:len(data) - start_idx] = data[start_idx:]
        else:
            segment = data[start_idx:end_idx + 1]

        segment = segment[:self.win_size]

        return segment

class GECCOPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "GECCO_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "GECCO_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "GECCO_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class NIPS_TS_SwanSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/NIPS_TS_Swan_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/NIPS_TS_Swan_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/NIPS_TS_Swan_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class PSMPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "PSM_train.npy"), mmap_mode="r")  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "PSM_test.npy"), mmap_mode="r")    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "PSM_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class PUMPPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "PUMP_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "PUMP_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "PUMP_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class SKABSegLoader(object):
    def __init__(self, data_path, win_size, step=1, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SKAB_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SKAB_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SKAB_test_label.npy")

    def __len__(self):
        """
        Returns the number of timestamps, as each timestamp corresponds to one segment.
        """
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]


    def __getitem__(self, index):
        """
                Get the segment for a specific timestamp (index).
                """
        if self.mode == "train":
            segment_data = self.create_segment(self.train, index)
            # label = self.test_labels[index]  # Label for this timestamp
            return np.float32(segment_data), np.float32(segment_data)  # No label for training data

        elif self.mode == "val":
            segment_data = self.create_segment(self.val, index)
            label = self.test_labels[index]
            return np.float32(segment_data), np.float32(label)

        elif self.mode == "test":
            segment_data = self.create_segment(self.test, index)
            label = self.test_labels[index]
            return np.float32(segment_data), np.float32(label)


    def create_segment(self, data, index):
        """
        Create a segment centered around a specific timestamp.
        If the segment is near the beginning or end of the series, it will be padded.
        """
        start_idx = index - self.win_size // 2
        end_idx = index + self.win_size // 2

        # re_start_idx = index - self.re_size // 2
        # re_end_idx = index + self.re_size // 2

        # Handle cases near the beginning
        if start_idx < 0:
            # Pad the segment with the first value
            segment = np.tile(data[0], (self.win_size, 1))
            segment[-start_idx - 1:] = data[:end_idx + 1]  # Fill the rest with available data
        # Handle cases near the end
        elif end_idx >= len(data):
            # Pad the segment with the last value
            segment = np.tile(data[-1], (self.win_size, 1))
            segment[:len(data) - start_idx] = data[start_idx:]  # Fill the rest with available data
        else:
            # Normal case: no padding
            segment = data[start_idx:end_idx + 1]

        segment = segment[:self.win_size]

        return segment

class SKABPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "SKAB_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "SKAB_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "SKAB_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class SMAPPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "SMAP_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "SMAP_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "SMAP_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SMAP_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "SMD_train.npy"), mmap_mode="r")  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "SMD_test.npy"), mmap_mode="r")    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "SWaT_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class SWaTPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "SWaT_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "SWaT_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "SWaT_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class SWaTSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/SWaT_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SWaT_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/SWaT_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class WaDiPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "WaDi_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "WaDi_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "WaDi_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class WaDiSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/WaDi_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/WaDi_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/WaDi_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class CompoundSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Compound_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/Compound_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Compound_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class OnlyphySegLoader(object):
    def __init__(self, data_path, win_size, step=1, mode="train"):
        """
        Args:
            data_path (str): Path to dataset folder
            win_size (int): Window size (segment size)
            step (int): Not used here but reserved
            mode (str): 'train', 'val' or 'test'
        """
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load data
        train_data = np.load(data_path + "/Onlyphy_train.npy")
        train_data = np.nan_to_num(train_data)
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)

        test_data = np.load(data_path + "/Onlyphy_test.npy")
        test_data = np.nan_to_num(test_data)
        test_data = self.scaler.transform(test_data)

        self.train = train_data
        self.test = test_data
        self.val = self.test  # Validation use test set

        # Labels
        self.test_labels = np.load(data_path + "/Onlyphy_test_label.npy")  # 0/1 label

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode == "val":
            return self.val.shape[0]
        elif self.mode == "test":
            return self.test.shape[0]
        else:
            return self.test.shape[0]

    def __getitem__(self, index):
        if self.mode == "train":
            segment = self.create_segment(self.train, index)
            return np.float32(segment), np.float32(segment)  # No label needed for training

        elif self.mode == "val":
            segment = self.create_segment(self.val, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

        elif self.mode == "test":
            segment = self.create_segment(self.test, index)
            label = self.test_labels[index]
            return np.float32(segment), np.float32(label)

    def create_segment(self, data, index):
        start_idx = index - self.win_size // 2
        end_idx = index + self.win_size // 2

        if start_idx < 0:
            segment = np.tile(data[0], (self.win_size, 1))
            segment[-start_idx - 1:] = data[:end_idx + 1]
        elif end_idx >= len(data):
            segment = np.tile(data[-1], (self.win_size, 1))
            segment[:len(data) - start_idx] = data[start_idx:]
        else:
            segment = data[start_idx:end_idx + 1]

        segment = segment[:self.win_size]

        return segment

class OnlyphyPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train"):
        """
        Args:
            data_path (str): root directory of the dataset
            win_size (int): window size of the patches
            mode (str): 'train', 'val', or 'test'
        """
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        self.scaler = StandardScaler()

        # directory of patches
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_data = np.load(os.path.join(patch_dir, "Onlyphy_train.npy")).astype(np.float32)  # [T, C, H, W]
        test_data = np.load(os.path.join(patch_dir, "Onlyphy_test.npy")).astype(np.float32)    # [T, C, H, W]

        train_data = np.nan_to_num(train_data)
        test_data = np.nan_to_num(test_data)

        # standardization on flattened data
        self.scaler.fit(train_data.reshape(train_data.shape[0], -1))
        self.train = self.scaler.transform(train_data.reshape(train_data.shape[0], -1)).reshape(train_data.shape)
        self.test = self.scaler.transform(test_data.reshape(test_data.shape[0], -1)).reshape(test_data.shape)
        self.val = self.test

        if mode in ["val", "test"]:
            self.test_labels = np.load(os.path.join(data_path, "Onlyphy_test_label.npy")).astype(np.float32)

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0]
        elif self.mode in ["val", "test"]:
            return self.test.shape[0]

    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch  # No label needed for training
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

class Bearing3SegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(data_path + "/Bearing3_train.npy")
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/Bearing3_test.npy")
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(data_path + "/Bearing3_test_label.npy")

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class GenericPatchDataset(object):
    def __init__(self, data_path, win_size, image_size, mode="train", dataset_name="SYN_S1"):
        self.mode = mode
        self.win_size = win_size
        self.image_size = image_size
        patch_dir = os.path.join(data_path, 'patches', f'win{win_size}_img{image_size}')
        train_path = os.path.join(patch_dir, f"{dataset_name}_train.npy")
        test_path = os.path.join(patch_dir, f"{dataset_name}_test.npy")
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            raise FileNotFoundError(f"Patch files not found under {patch_dir}. Run preprocess first.")
        train = np.load(train_path).astype(np.float32)
        test = np.load(test_path ).astype(np.float32)
        train = np.nan_to_num(train)
        test = np.nan_to_num(test)
        self.train = train
        self.test = test
        self.val = self.test
        if mode in ["val","test"]:
            self.test_labels = np.load(os.path.join(data_path, f"{dataset_name}_test_label.npy")).astype(np.float32)


    def __len__(self):
        return self.train.shape[0] if self.mode=="train" else self.test.shape[0]


    def __getitem__(self, idx):
        if self.mode == "train":
            patch = torch.tensor(self.train[idx], dtype=torch.float32)
            return patch, patch
        else:
            patch = torch.tensor(self.test[idx], dtype=torch.float32)
            label = torch.tensor(self.test_labels[idx], dtype=torch.float32)
            return patch, label

def get_loader_segment(index, data_path, batch_size, win_size=100, image_size=224, step=100, mode='train', dataset='KDD'):
    if (dataset == 'Genesis'):
        dataset = GenesisPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'HAI'):
        dataset = HAIPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'MSL'):
        dataset = MSLPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'GECCO'):
        dataset = GECCOPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'NIPS_TS_Swan'):
        dataset = NIPS_TS_SwanSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'PSM'):
        dataset = PSMPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'PUMP'):
        dataset = PUMPPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'SKAB'):
        dataset = SKABPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'SMAP'):
        dataset = SMAPPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'SMD'):
        dataset = SMDPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'SWaT'):
        dataset = SWaTPatchDataset(data_path, win_size, image_size, mode)
    elif (dataset == 'WaDi'):
        dataset = WaDiPatchDataset(data_path, win_size, image_size, mode)
    # elif (dataset == 'Compound'):
    #     dataset = CompoundSegLoader(data_path, win_size, 1, mode)
    elif (dataset == 'Onlyphy'):
        dataset = OnlyphyPatchDataset(data_path, win_size, image_size, mode)
    # elif (dataset == 'Bearing3'):
    #     dataset = Bearing3SegLoader(data_path, win_size, 1, mode)
    # elif dataset.startswith('SYN_'):
    #     ds_name = os.path.basename(os.path.normpath(data_path))
    #     dataset = GenericPatchDataset(data_path, win_size, image_size, mode, dataset_name=ds_name)

    dataset = WithIndex(dataset)

    shuffle = False
    if mode == 'train':
        shuffle = True
    g = torch.Generator()
    g.manual_seed(42)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0,
                             drop_last=False,
                             generator=g)
    return data_loader
