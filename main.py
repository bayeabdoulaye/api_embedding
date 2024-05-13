import logging
from typing import List, Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
#from sentence_transformers import SentenceTransformer
#from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
import os

app = FastAPI()

'''class InputText(BaseModel):
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,chunk_overlap = 40)'''

@app.post("/custom_embedding")
async def custom_embedding():
    

    return 'ok'

@app.get("/text_split")
async def text_split():

    return 'merci'

        
