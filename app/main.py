import logging
from fastapi import FastAPI, HTTPException
from app.schemas import QuestionRequest, AnswerResponse
from app.model import ask_llama

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama 3.2 MLOps API", version="1.0.0")

metrics = {"total": 0, "success": 0, "failed": 0, "latencies": []}

@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3.2"}

@app.get("/metrics")
def get_metrics():
    lat = metrics["latencies"]
    return {
        "total_requests": metrics["total"],
        "error_rate_pct": round(metrics["failed"] / max(metrics["total"], 1) * 100, 2),
        "avg_latency_ms": round(sum(lat) / len(lat), 2) if lat else 0,
        "p95_latency_ms": round(sorted(lat)[int(len(lat) * 0.95)], 2) if lat else 0,
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    metrics["total"] += 1
    logger.info(f"Question: {request.question[:80]}")
    try:
        result = await ask_llama(request.question, request.system_prompt)
        metrics["success"] += 1
        metrics["latencies"].append(result["latency_ms"])
        return AnswerResponse(question=request.question, **result)
    except Exception as e:
        metrics["failed"] += 1
        logger.error(f"Failed: {e}")
        raise HTTPException(status_code=503, detail="Model unavailable")