from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL_PATH = "C:/Users/HP/Documents/Github Repo's/Projects/Chest X-ray Classification with Convolutional Neural Networks (CNNs)/my_model.keras"
MODEL = tf.keras.models.load_model(MODEL_PATH)
CLASS_NAMES=['Pneumonia','Normal']

@app.get('/')
def welcome():
    return "Welcome to the dashboard of Medical app!"

def read_file_as_image(data) -> np.ndarray:
    image= Image.open(BytesIO(data))   #take an image converts it into binary format
    image_array=np.array(image)
    return image
    # {
    #     'class':predicted_class,
    #     'confidence':float(confidence)
    # }

@app.post('/predict')
async def predict(
    file: UploadFile=File(...)
):
    
    image_array=read_file_as_image(await file.read())
    resized_image = tf.image.resize(image_array, (150, 150))
    image_batch = np.expand_dims(resized_image, axis=0)
    # image_batch=np.expand_dims(image_array,0)
    predictions=MODEL.predict(image_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])] 
    confidence=np.max(predictions[0])
    print(predictions.tolist()) 
 
# int(np.argmax(predictions[0]))
    #                  {
    #     'class':predicted_class,
    #     'confidence': float(confidence)
    # } 
    # bytes= await file.read()
    # return "File Uploaded!"


# # Define the input shape based on your model's requirements
# IMG_WIDTH = ...  # Width of the input images used during training
# IMG_HEIGHT = ...  # Height of the input images used during training

# @app.post('/predict')
# async def predict(
#     file: UploadFile = File(...)
# ):
#     try:
#         # Read the uploaded image file and convert it to a NumPy array
#         image = read_file_as_image(await file.read())
        
#         # Resize the image to match the expected input shape of the model
#         resized_image = tf.image.resize(image, (150, 150))
        
#         # Normalize the pixel values (if necessary) to match the preprocessing applied during training
#         # Add your normalization steps here
        
#         # Expand the dimensions of the image array to match the expected input shape of the model
#         image_batch = np.expand_dims(resized_image, axis=0)
        
#         # Perform prediction using the loaded model
#         predictions = MODEL.predict(image_batch)
        
#         # Decode the predictions and prepare the response
#         predicted_class_index = np.argmax(predictions)
#         predicted_class = CLASS_NAMES[predicted_class_index]
#         confidence = float(predictions[0][predicted_class_index])
        
#         # Return the prediction result as a JSON response
#         return {"class": predicted_class, "confidence": confidence}
#     except Exception as e:
#         # If an error occurs during prediction, return an error response
#         return {"error": str(e)}



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


# from fastapi import FastAPI, File, UploadFile
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from io import BytesIO

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
    
#     # Modify the return statement to print the predicted class
#     if predicted_class == "Normal":
#         result = "Normal"
#     else:
#         result = "Pneumonia"
        
#     return {
#         'result': result,
#         'confidence': float(confidence)
#     }

