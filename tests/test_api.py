from unittest.mock import patch

from fastapi.testclient import TestClient

from src.api.app import app


class DummyModel:

    def to(self, device):
        return self

    def eval(self):
        return self


class DummyProcessor:
    pass


@patch(
    "src.api.app.login",
    return_value=None
)
@patch(
    "src.api.app.TrOCRProcessor.from_pretrained",
    return_value=DummyProcessor()
)
@patch(
    "src.api.app.VisionEncoderDecoderModel.from_pretrained",
    return_value=DummyModel()
)
def test_health(
    mock_model,
    mock_processor,
    mock_login
):

    with TestClient(app) as client:

        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"
    assert body["model_loaded"] is True

    mock_model.assert_called_once()
    mock_processor.assert_called_once()