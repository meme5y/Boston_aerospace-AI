"""Tests/Test_api.py — Testes das rotas API"""
def test_ping(client):
    r = client.get("/api/ping")
    assert r.status_code == 200
    assert r.json["ok"] is True

def test_register_login(client):
    r = client.post("/api/register", json={"email":"u@test.com","password":"pass","name":"U"})
    assert r.status_code == 201
    r = client.post("/api/login", json={"email":"u@test.com","password":"pass"})
    assert r.json["ok"] is True

def test_predict_unauth(client):
    r = client.post("/api/predict", json={"sensors":[1]*18,"cycle":150})
    assert r.status_code == 401

def test_predict_auth(auth_client):
    sensors = [490, 600, 1500, 1300, 10, 14, 300, 2100, 8500,
               42, 300, 2200, 8100, 360, 2100, 92, 25, 14]
    r = auth_client.post("/api/predict", json={"sensors": sensors, "cycle": 150})
    assert r.status_code == 200
    assert "rul" in r.json
