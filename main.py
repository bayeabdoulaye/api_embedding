import logging
from typing import List, Optional
from fastapi import FastAPI
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

class ChunkOutput(BaseModel):
    recordId: str
    data: dict

model = SentenceTransformer(os.path.join(os.getcwd(), 'all-MiniLM-L6-v2'))
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,chunk_overlap = 20)
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
async def text_split(input_data: TextSplitInput):
    if input_data.mode == "sentence":
        doc_text = text_splitter.create_documents([input_data.text])
        chunks = [doc.dict()["page_content"] for doc in doc_text]
        response = { "recordId": input_data.recordId, "data": { "chunks": chunks } }
        return response