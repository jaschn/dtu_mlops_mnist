import hydra
from src.models.model import MyAwesomeModel
import torch.nn as nn
import torch.optim as optim
import math
import torchvision
from torch.utils.data import DataLoader
import torch

def test_training():
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose(config_name="training_conf.yaml")

    input_filepath = cfg.training.input_filepath
    lr = cfg.training.lr
    epochs = 5

    model = MyAwesomeModel(cfg.model)
    critirion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set = torchvision.datasets.MNIST("data",transform=torchvision.transforms.ToTensor(), download=True, train=True)
    train_loader = DataLoader(train_set, batch_size=64)
    losses = []
    for _ in range(epochs):
        running_loss = 0
        for images, labels in train_loader:
            out = model(images)
            loss = critirion(out, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_item = loss.item()
            assert not math.isnan(loss_item), "Loss is NAN"
            running_loss += loss_item
        losses.append(running_loss)

    assert abs(losses[-1]) < abs(losses[0]), "Loss not converging to 0"