"""
Endpoint defintion for the system's API.
"""
from io import BytesIO

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi import HTTPException
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np


from digits_recognition.api.mlflow_model_setup import mlflow_model_setup
from digits_recognition.api.inference import (
    compute_predictions,
    compute_probabilities,
    annotate_image_with_predictions
)


model, device = mlflow_model_setup()

model.eval()
torch.no_grad()

app = FastAPI()
app.mount("/static", StaticFiles(directory="./digits_recognition/api/static"), name="static")


@app.get("/")
async def root_redirect():
    """
    Returns the main web page to the user.
    """
    return RedirectResponse(url="/static/form.html")


async def _decode_image(file):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    image_data = await file.read()
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    return image


@app.post("/annotations")
async def annotate_image(file: UploadFile = File(...)):
    """
    Endpoint that given an image returns that same image annotated with predictions.
    """
    image = await _decode_image(file)

    annotated_image = annotate_image_with_predictions(model, device, image)

    _, img_encoded = cv2.imencode('.png', annotated_image)
    img_io = BytesIO(img_encoded.tobytes())
    img_io.seek(0)

    return StreamingResponse(img_io, media_type="image/png")


@app.post("/predictions")
async def predict_labels(file: UploadFile = File(...)):
    """
    Endpoint that given an image returns a list of pairs [coordinates, label]
    """
    image = await _decode_image(file)
    preds = compute_predictions(model, device, image)

    return {
        'predictions': preds
    }


@app.post("/probabilities")
async def predict_probs(file: UploadFile = File(...)):
    """
    Endpoint that given an image returns a list of pairs [coordinates, probabilities]
    """
    image = await _decode_image(file)
    probs = compute_probabilities(model, device, image)

    return {
        'probabilities': probs
    }
