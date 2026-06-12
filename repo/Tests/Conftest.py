"""Tests/Conftest.py — Fixtures partilhadas"""
import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c

@pytest.fixture
def auth_client(client):
    client.post("/api/register", json={"email":"test@test.com","password":"test","name":"Test"})
    client.post("/api/login",    json={"email":"test@test.com","password":"test"})
    return client
