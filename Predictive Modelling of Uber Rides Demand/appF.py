# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle

# app=FastAPI()
# with open("taxi.pkl","rb") as file:
#     classifier=pickle.load(file)

# class uber(BaseModel):
#     Population: int
#     Monthlyincome: int  
#     Averageparkingpermonth:int  
#     Numberofweeklyriders:int

# app.post('/predict')
# def predict(data: uber):
#     data=data.model_dump()
#     print(data)    

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Open the pickled model file in read-binary mode
with open("taxi.pkl", "rb") as file:
    # Load the pickled model from the file
    classifier = pickle.load(file)

class Uber(BaseModel):
    Population: int
    Monthlyincome: int  
    Averageparkingpermonth: int  
    Numberofweeklyriders: int

@app.post('/predict')
def predict(data: Uber):
    data=data.model_dump()
    Population=data['Population']
    Monthlyincome=data['Monthlyincome']
    Numberofweeklyriders=data['Numberofweeklyriders']
    
    return (classifier.predict([[data.Population, data.Monthlyincome, data.Averageparkingpermonth, data.Numberofweeklyriders]]))
    return {"prediction": prediction}
