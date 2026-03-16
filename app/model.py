import httpx, time

OLLAMA_URL = "http://ollama:11434"
MODEL_NAME = "llama3.2"

async def ask_llama(question: str, system_prompt: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        "stream": False
    }
    start = time.monotonic()
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        response.raise_for_status()
    latency_ms = (time.monotonic() - start) * 1000
    content = response.json()["message"]["content"]
    return {"answer": content, "model": MODEL_NAME, "latency_ms": round(latency_ms, 2)}