from src.data.dataset import CorruptMNISTset
from hydra import initialize, compose
import pytest
import torchvision

def test_dataset_train():
    with initialize(config_path="../config"):
        cfg = compose(config_name="training_conf.yaml")
        
    dataset = torchvision.datasets.MNIST("data", download=True, train=True)
    assert len(dataset) == 60000, "dataset length wrong"

def test_dataset_test():
    with initialize(config_path="../config"):
        cfg = compose(config_name="training_conf.yaml")

    dataset_test = torchvision.datasets.MNIST("data", download=True, train=False)
    assert len(dataset_test) == 10000, "testset length wrong"