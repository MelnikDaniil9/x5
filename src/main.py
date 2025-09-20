"""
server.py
FastAPI-сервер для асинхронного инференса NER модели.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from .inference import run_inference_on_text

# =====================
# 🔹 FastAPI
# =====================
app = FastAPI(title="NER Inference API")

class PredictRequest(BaseModel):
    input: str


@app.post("/api/predict")
async def predict(request: PredictRequest) -> List[Dict]:
    """
    Принимает JSON {"input": "..."} и возвращает список сущностей.
    """
    entities = run_inference_on_text(request.input)

    # Приводим к нужному формату
    response = [
        {"start_index": s, "end_index": e, "entity": lbl}
        for s, e, lbl in entities
    ]
    return response
