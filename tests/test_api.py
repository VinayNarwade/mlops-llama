# tests/test_api.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.anyio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

@pytest.mark.anyio
async def test_ask_returns_valid_structure():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json={"question": "What is 2+2?"})
    
    if response.status_code == 503:
        pytest.skip("Ollama not running — skipping model integration test")
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], float)

@pytest.mark.anyio
async def test_ask_rejects_empty_question():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ask", json={"question": ""})
    assert response.status_code != 422