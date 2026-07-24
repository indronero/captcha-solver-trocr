# tests/test_inference.py

from src.inference.predictor import predict


def test_predict_exists():

    assert callable(predict)