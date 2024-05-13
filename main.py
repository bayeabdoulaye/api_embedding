import logging
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

app = FastAPI()

class InputText(BaseModel):
    recordId: str
    data: dict

class InputData(BaseModel):
    values: List[InputText]

class TextSplitInput(BaseModel):
    recordId: str
    text: str
    mode: str

class VectorData(BaseModel):
    vector: List[float]
class ResponseItem(BaseModel):
    recordId: int
    data: VectorData

class ResponseData(BaseModel):
    values: List[ResponseItem]

class ChunkOutput(BaseModel):
    recordId: str
    data: dict
class TextSplitInput(BaseModel):
    values: List[dict]

class TextSplitOutput(BaseModel):
    recordId: str
    data: dict

class TextItemOutput(BaseModel):
    textItems: List[str]

class TextSplitResponse(BaseModel):
    values: List[TextSplitOutput]

model = 'all-MiniLM-L6-v2'
model = SentenceTransformer(model)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,chunk_overlap = 40)

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

@app.post("/text_split")
async def text_split(text_split: TextSplitInput):
    try:
        response_values = []
        for item in text_split.values:
            record_id = item["recordId"]
            text = item["data"]["text"]

            doc_chunks = text_splitter.create_documents([text])
            chunks = [doc.dict()["page_content"] for doc in doc_chunks]
            text_items = []
            for chunk in chunks:
                text_items.append(chunk)

            response_values.append({
                        "recordId": record_id,
                        "data": {"textItems": text_items}
                    })

        response = {"values": response_values}
        return JSONResponse(status_code=200, content=response)
    except Exception as e:
        error_message = f'Exception => {e}'
        print(error_message)
        return JSONResponse(
                status_code=400,
                content=error_message
            )

        
