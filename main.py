import logging
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json
import os

app = FastAPI()

class InputText(BaseModel):
    text: str

class InputData(BaseModel):
    values: List[InputText]

class ResponseData(BaseModel):
    recordId: int
    data: dict
model = SentenceTransformer(os.path.join(os.getcwd(), 'all-MiniLM-L6-v2'))
@app.get("/custom_embedding", response_model=List[ResponseData])
def custom_embedding(text: Optional[str] = None):
    logging.info('Python HTTP trigger function processed a request.')
    input_text = []
    if text:
        input_text.append(text)
    else:
        return {"error": "No input text found"}

    embeddings = model.encode(input_text)
    response = [ResponseData(recordId=i, data={"vector": embedding.tolist()}) for i, embedding in enumerate(embeddings)]

    return response
