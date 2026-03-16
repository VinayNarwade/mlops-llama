# app/main.py
import logging
import time
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from app.schemas import QuestionRequest, AnswerResponse
from app.model import ask_llama

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Llama 3.2 MLOps API", version="1.0.0")

# In-memory metrics store (in production you'd use Prometheus)
metrics = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "latencies_ms": [],
}

@app.get("/health")
def health_check():
    return {"status": "ok", "model": "llama3.2"}

@app.get("/metrics")
def get_metrics():
    """Expose runtime stats — latency, error rate, throughput."""
    latencies = metrics["latencies_ms"]
    return {
        "total_requests":      metrics["total_requests"],
        "successful_requests": metrics["successful_requests"],
        "failed_requests":     metrics["failed_requests"],
        "error_rate_pct":      round(
            metrics["failed_requests"] / max(metrics["total_requests"], 1) * 100, 2
        ),
        "avg_latency_ms":  round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "p95_latency_ms":  round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if latencies else 0,
        "min_latency_ms":  round(min(latencies), 2) if latencies else 0,
        "max_latency_ms":  round(max(latencies), 2) if latencies else 0,
    }

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    metrics["total_requests"] += 1
    logger.info(f"Question received: {request.question[:80]}")
    
    try:
        result = await ask_llama(request.question, request.system_prompt)
        metrics["successful_requests"] += 1
        metrics["latencies_ms"].append(result["latency_ms"])
        logger.info(f"Answered in {result['latency_ms']}ms")
        return AnswerResponse(question=request.question, **result)
    
    except Exception as e:
        metrics["failed_requests"] += 1
        logger.error(f"Model call failed: {e}")
        raise HTTPException(status_code=503, detail="Model unavailable")