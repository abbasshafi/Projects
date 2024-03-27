from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL_PATH = "C:/Users/HP/Documents/Github Repo's/Projects/Chest X-ray Classification with Convolutional Neural Networks (CNNs)/Pneumonia_cnn_model.h5"
MODEL = tf.keras.models.load_model(MODEL_PATH)

# @app.get('/')
# def welcome():
#     return "Welcome to the dashboard of Medical app!"

# def read_file_as_image(data) -> np.ndarray:
#     image= np.array(Image,open(BytesIO(data)))
#     image_batch=np.expand_dims(image,0)
#     prediction= MODEL.predict(image_batch)
#     predictied_class=np.argmax(prediction[0])
#     confidence=np.max(prediction[0])
#     return {
#         'class':predicted_class,
#         'confidence':float(confidence)
#     }


# @app.post('/predict')
# async def predict(
#     file: UploadFile=File(...)
# ):
#     bytes= await file.read()
#     return "File Uploaded!"


# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# MODEL= tf.keras.models.load_model("../my_model.keras")
# Class_names=["Pneumonia","Normal"]

# @app.get('/')
# def welcome():
#     return "Welcome to the dashboard of the Medical app!"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post('/predict')
# async def predict(
#     file: UploadFile = File(...)
# ):
#     bytes = await file.read()
#     image_array = read_file_as_image(bytes)
   
#     return ("Shape of the image array:", image_array.shape)
#     # return {"status": "File Uploaded and Converted to NumPy Array"} 






# 



# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from io import BytesIO
# import keras


# app = FastAPI()

# # Load the Keras model
# # MODEL_PATH = "C:/Users/HP/Documents/Github Repo's/Projects/Chest X-ray Classification with Convolutional Neural Networks (CNNs)/my_model.keras"
# # MODEL = tf.keras.models.load_model(MODEL_PATH)

# Class_names = ["Pneumonia", "Normal"]

# # Define a function to preprocess the image
# def preprocess_image(image):
#     # Resize the image to match the input shape of the model (assuming it's 150x150)
#     image = image.resize((150, 150))
#     # Convert image to numpy array and normalize pixel values
#     image_array = np.array(image) / 255.0
#     # Expand dimensions to create a batch of size 1
#     image_batch = np.expand_dims(image_array, axis=0)
#     return image_batch

# @app.post('/predict')
# async def predict(file: UploadFile = File(...)):
#     # Read the uploaded file as an image
#     img = Image.open(BytesIO(await file.read()))
#     # Preprocess the image
#     img_batch = preprocess_image(img)
#     # Make predictions using the model
#     predictions = MODEL.predict(img_batch)
#     # Get the predicted class index and confidence
#     predicted_class_index = np.argmax(predictions[0])
#     confidence = np.max(predictions[0])
#     # Get the class label
#     predicted_class = Class_names[predicted_class_index]
#     return {
#         'class': predicted_class,
#         'confidence': float(confidence)
#     }


from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO

app = FastAPI()

# Load the Keras model
# MODEL_PATH = "C:/Users/HP/Documents/Github Repo's/Projects/Chest X-ray Classification with Convolutional Neural Networks (CNNs)/my_model.keras"
# MODEL = tf.keras.models.load_model(MODEL_PATH)

Class_names = ["Pneumonia", "Normal"]

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input shape of the model (assuming it's 150x150)
    image = image.resize((150, 150))
    # Convert image to numpy array and normalize pixel values
    image_array = np.array(image) / 255.0
    # Expand dimensions to create a batch of size 1
    image_batch = np.expand_dims(image_array, axis=0)
    return image_batch

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    # Read the uploaded file as an image
    img = Image.open(BytesIO(await file.read()))
    # Preprocess the image
    img_batch = preprocess_image(img)
    # Make predictions using the model
    predictions = MODEL.predict(img_batch)
    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    # Get the class label
    predicted_class = Class_names[predicted_class_index]
    
    # Modify the return statement to print the predicted class
    if predicted_class == "Normal":
        result = "Normal"
    else:
        result = "Pneumonia"
        
    return {
        'result': result,
        'confidence': float(confidence)
    }

