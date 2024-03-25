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

# from fastapi import FastAPI
# from pydantic import BaseModel
# import pickle

# app = FastAPI()

# # Open the pickled model file in read-binary mode
# with open("taxi.pkl", "rb") as file:
#     # Load the pickled model from the file
#     classifier = pickle.load(file)

# class Uber(BaseModel):
#     Population: int
#     Monthlyincome: int  
#     Averageparkingpermonth: int  
#     Priceperweek: int

# @app.get('/')
# def welcome():
#     return "Welcome to Uber Dashboard for predicting weekly uber rides."

# @app.post('/predict')
# def predict(data: Uber):
#     data=data.model_dump()
#     Population=data['Population']
#     Monthlyincome=data['Monthlyincome']
#     Priceperweek=data['Priceperweek']
    
#     Prediction= classifier.predict([[data.Priceperweek, data.Population, data.Monthlyincome, data.Averageparkingpermonth]])
#     print("Number of Rides available are: ", Prediction )


# # 5. Run the API with uvicorn
# #    Will run on http://127.0.0.1:8000
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)


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
    Priceperweek: int

@app.get('/')
def welcome():
    return "Welcome to Uber Dashboard for predicting weekly uber rides."

@app.post('/predict')
def predict(data: Uber):
    Population = data.Population
    Monthlyincome = data.Monthlyincome
    Averageparkingpermonth = data.Averageparkingpermonth
    Priceperweek = data.Priceperweek
    
    Prediction = classifier.predict([[Priceperweek, Population, Monthlyincome, Averageparkingpermonth]])
    return {"Number of Rides available": int(Prediction[0])}
