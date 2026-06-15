# tests/test_inference.py

from src.training.dataset_loader import load_datasets


def test_dataset_loader():

    assert callable(load_datasets)