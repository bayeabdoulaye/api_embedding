import logging
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import json
import os

app = FastAPI()

class InputText(BaseModel):
    recordId: str
    data: dict

class InputData(BaseModel):
    values: List[InputText]

model = SentenceTransformer(os.path.join(os.getcwd(), 'all-MiniLM-L6-v2'))

@app.post("/custom_embedding")
async def custom_embedding(input_data: InputData):
    logging.info('Python HTTP trigger function processed a request.')
    print(input_data)
    input_text = [value.data['text'] for value in input_data.values]

    if not input_text:
        return {"error": "No input text found"}

    embeddings = model.encode(input_text)
    response = { "values": [ { "recordId": int(value.recordId), "data": { "vector": embedding.tolist() } } for value, embedding in zip(input_data.values, embeddings)]}

    return response