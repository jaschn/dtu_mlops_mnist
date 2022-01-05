import os

import torch
from torch.utils.data import DataLoader, Dataset


class CorruptMNISTset(Dataset):
    def __init__(self, base_path, type="train"):
        if type == "train":
            path = os.path.join(base_path, "train")
        elif type == "test":
            path = os.path.join(base_path, "test")
        else:
            assert "wrong option"

        self.images = torch.load(os.path.join(path, "images.pt"))
        self.labels = torch.load(os.path.join(path, "labels.pt"))

        assert len(self.images) == len(
            self.labels
        ), "image count does not match label count"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def mnist(path, type="train"):
    set = CorruptMNISTset(path, type=type)
    loader = DataLoader(set, batch_size=64, shuffle=True)
    return loader
