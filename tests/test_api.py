from fastapi.testclient import TestClient

from src.api.app import app


def test_health():

    with TestClient(app) as client:

        response = client.get("/health")

        assert response.status_code == 200