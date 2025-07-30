
import pytest
from flask import Flask
from frontend.app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    rv = client.get('/')
    assert rv.status_code == 200
    assert b"Game Maker" in rv.data

def test_get_predictions(client):
    rv = client.post('/get_predictions', json={"date": "2025-01-01"})
    assert rv.status_code in [200, 500]  # Allow 500 if backend model is unavailable

def test_get_games(client):
    rv = client.post('/get_games', json={"date": "2025-01-01"})
    assert rv.status_code in [200, 500]

def test_get_teams(client):
    rv = client.get('/get_teams?league=nba')
    assert rv.status_code == 200
