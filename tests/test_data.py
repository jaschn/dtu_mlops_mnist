import os

import hydra
import pytest
import torch

from src.data.dataset import mnist


@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
@pytest.mark.parametrize("type, cnt", [("train", 40000), ("test", 5000)])
def test_dataset_train(type, cnt):
    with hydra.initialize(config_path="../config"):
        cfg = hydra.compose(config_name="training_conf.yaml")
    dataloader = mnist(cfg.training.input_filepath, type=type)
    assert len(dataloader.dataset) == cnt, "dataset length wrong"
    all_labels = []
    for images, lables in dataloader:
        assert images.shape[1:] == (1, 28, 28)
        all_labels.append(lables)

    all_labels = torch.cat(all_labels, dim=0)
    for x in range(10):
        assert x in all_labels, f"No data point with label: {x}"
