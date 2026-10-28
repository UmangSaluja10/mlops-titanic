from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

SAMPLE = {"Pclass": 1, "Sex": "female", "Age": 29, "SibSp": 0,
          "Parch": 0, "Fare": 80.0, "Embarked": "S"}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["survived"] in (0, 1)
    assert 0 <= body["probability"] <= 1


def test_bad_input():
    r = client.post("/predict", json={"Pclass": 1})
    assert r.status_code == 422