from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from pydantic import BaseModel
from ultralytics import YOLO





# Creating FastAPI instance
app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Welcome to my FastAPI phoenix-ag-app!'}
model = YOLO('./model_weights/best.pt') 


class ImageUpload(BaseModel):
    file: UploadFile

# Create an endpoint to receive image data and make prediction
@app.post('/predict')
async def predict(image: UploadFile = File(...)):

    contents = await image.read()
    # img = Image.open(io.BytesIO(contents)).convert('RGB')
    # 'https://ultralytics.com/images/bus.jpg'

    results = model('https://ultralytics.com/images/bus.jpg')


    return {'class': "results"}

