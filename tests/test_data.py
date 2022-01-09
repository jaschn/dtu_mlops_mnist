import torchvision
import pytest

@pytest.mark.parametrize("type, cnt", [(True, 60000), (False, 10000)])
def test_dataset_train(type, cnt):
    dataset = torchvision.datasets.MNIST("data", download=True, train=type)
    assert len(dataset) == cnt, "dataset length wrong"