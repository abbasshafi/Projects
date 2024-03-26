# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# import uvicorn
# from io import BytesIO
# from PIL import Image

# app=FastAPI()

# @app.get('/')
# def welcome():
#     return "Welcome to the dashboard of Medical app!"

# def read_file_as_image(data) -> np.ndarray:
#     image= np.array(Image,open(BytesIO(data)))
#     return image

# @app.post('/predict')
# async def predict(
#     file: UploadFile=File(...)
# ):
#     bytes= await file.read()
#     return "File Uploaded!"


from fastapi import FastAPI, File, UploadFile
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

@app.get('/')
def welcome():
    return "Welcome to the dashboard of the Medical app!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post('/predict')
async def predict(
    file: UploadFile = File(...)
):
    bytes = await file.read()
    image_array = read_file_as_image(bytes)
   
    return ("Shape of the image array:", image_array.shape)
    # return {"status": "File Uploaded and Converted to NumPy Array"} 
