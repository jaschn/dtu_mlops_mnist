import torch
from src.models.model import MyAwesomeModel
from hydra import initialize, compose
import pytest
import re

def test_model():
    with initialize(config_path="../config"):
        cfg = compose(config_name="training_conf.yaml")

    model = MyAwesomeModel(cfg.model)
    testdata = torch.Tensor(64, 1, 28, 28)
    out = model(testdata)
    assert out.shape == (64, 10), "output dimension wrong"

def test_model_checks_dimension():
    with initialize(config_path="../config"):
        cfg = compose(config_name="training_conf.yaml")

    model = MyAwesomeModel(cfg.model)
    testdata = torch.Tensor(64, 2, 28, 28)
    with pytest.raises(ValueError, match=re.escape('Expected each sample to have shape [batch_size, 1, 28, 28]')):
        out = model(testdata)

def test_model_checks_number_dimension():
    with initialize(config_path="../config"):
        cfg = compose(config_name="training_conf.yaml")

    model = MyAwesomeModel(cfg.model)
    testdata = torch.Tensor(64, 2, 28)
    with pytest.raises(ValueError, match=re.escape('Expected input to a 4D tensor')):
        out = model(testdata)