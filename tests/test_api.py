
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app

client = TestClient(app)


def create_test_image():
    image = Image.new("RGB", (160, 160), color=(120, 120, 120))
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    return image_bytes


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_valid_image_prediction():
    image = create_test_image()

    response = client.post(
        "/predict",
        files={"file": ("test_image.png", image, "image/png")}
    )

    assert response.status_code == 200

    result = response.json()

    assert result["success"] is True
    assert result["predicted_class"] in ["Bengin", "Malignant", "Normal"]
    assert 0 <= result["confidence"] <= 100
    assert set(result["probabilities"].keys()) == {
        "Bengin", "Malignant", "Normal"
    }
    assert "disclaimer" in result


def test_invalid_file_type():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 400
