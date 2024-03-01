# from fastapi import FastAPI
# import uvicorn
# from sklearn.datasets import load_iris
# from sklearn.naive_bayes import GaussianNB
# from pydantic import BaseModel

# # Creating FastAPI instance
# app = FastAPI()

# # Creating class to define the request body
# # and the type hints of each attribute
# class request_body(BaseModel):
# 	sepal_length : float
# 	sepal_width : float
# 	petal_length : float
# 	petal_width : float

# # Loading Iris Dataset
# iris = load_iris()

# # Getting our Features and Targets
# X = iris.data
# Y = iris.target

# # Creating and Fitting our Model
# clf = GaussianNB()
# clf.fit(X,Y)

# # Creating an Endpoint to receive the data
# # to make prediction on.
# @app.post('/predict')
# def predict(data : request_body):
# 	# Making the data in a form suitable for prediction
# 	test_data = [[
# 			data.sepal_length, 
# 			data.sepal_width, 
# 			data.petal_length, 
# 			data.petal_width
# 	]]
	
# 	# Predicting the Class
# 	class_idx = clf.predict(test_data)[0]
	
# 	# Return the Result
# 	return { 'class' : iris.target_names[class_idx]}



#   Using cached mpmath-1.3.0-py3-none-any.whl (536 kB)
# Installing collected packages: mpmath, typing-extensions, sympy, networkx, fsspec, filelock, torch, pillow, numpy, torchvision, torchaudio
#   WARNING: The script isympy is installed in '/Users/timi/Library/Python/3.9/bin' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#   WARNING: The scripts convert-caffe2-to-onnx, convert-onnx-to-caffe2 and torchrun are installed in '/Users/timi/Library/Python/3.9/bin' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
#   WARNING: The script f2py is installed in '/Users/timi/Library/Python/3.9/bin' which is not on PATH.
#   Consider adding this directory to PATH or, if you prefer to suppress this warning, use --no-warn-script-location.
# Successfully installed filelock-3.13.1 fsspec-2024.2.0 mpmath-1.3.0 networkx-3.2.1 numpy-1.26.4 pillow-10.2.0 sympy-1.12 torch-2.2.1 torchaudio-2.2.1 torchvision-0.17.1 typing-extensions-4.10.0
# WARNING: You are using pip version 21.2.4; however, version 24.0 is available.
# You should consider upgrading via the '/Applications/Xcode.app/Contents/Developer/usr/bin/python3 -m pip install --upgrade pip' command.





from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import models
from pydantic import BaseModel

# Creating FastAPI instance
app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Welcome to my FastAPI phoenix-ag-app!'}

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Define transformation to preprocess the image
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define class to handle image upload
class ImageUpload(BaseModel):
    file: UploadFile

# Create an endpoint to receive image data and make prediction
@app.post('/predict')
async def predict(image: UploadFile = File(...)):
    # Read image file
    contents = await image.read()
    # Open image using PIL
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    # Preprocess image
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)  # Add batch dimension
    # Make prediction
    with torch.no_grad():
        outputs = model(img)
    # Get predicted class label
    _, predicted_idx = torch.max(outputs, 1)
    # Load labels mapping
    with open("imagenet_classes.txt") as f:
        classes = [line.strip() for line in f.readlines()]
    # Return the result
    return {'class': classes[predicted_idx.item()]}
