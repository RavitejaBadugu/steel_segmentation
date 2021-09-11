from fastapi import FastAPI,File
from utils import get_prediction
import numpy as np
app=FastAPI()

@app.get('/')
def get_welcome():
    return 'welcome to the page'

@app.post('/segmentation')
def get_segmentation(file: bytes=File(...)):
    response=get_prediction(file)
    return response

