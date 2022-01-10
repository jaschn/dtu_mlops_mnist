import pytest
import os
from src.data.dataset import mnist
import hydra

@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
@pytest.mark.parametrize("type, cnt", [("train", 40000), ("test", 5000)])
def test_dataset_train(type, cnt):
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose(config_name="training_conf.yaml")
    dataloader = mnist(cfg.training.input_filepath, type=type)
    assert len(dataloader.dataset) == cnt, "dataset length wrong"
