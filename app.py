from fastapi import FastAPI, Form, Request
import numpy as np
import pickle
import uvicorn
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Create the FastAPI app instance
app = FastAPI()
templates = Jinja2Templates(directory="C:/Users/shopinverse/Documents/FastAPI Project/BankNote/templates")

# Configure the StaticFiles to serve CSS and other static files
app.mount("/static", StaticFiles(directory="C:/Users/shopinverse/Documents/FastAPI Project/BankNote/static"), name="static")

# Load the model
file_path = 'C:/Users/Shopinverse/Documents/FastAPI Project/BankNote/classifier.pkl'
with open(file_path, 'rb') as file:
    clf = pickle.load(file)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("prediction_form.html", {"request": request, "prediction": ""})

@app.post('/predict')
def predict(
    request: Request,
    variance: float = Form(...),
    skewness: float = Form(...),
    curtosis: float = Form(...),
    entropy: float = Form(...),
):
    # Extract data from the request
    input_data = np.array([
        [variance, skewness, curtosis, entropy]
    ])

    prediction = clf.predict(input_data)
    
    if prediction[0] > 0.5:
        result = "This Bank note is FAKE!"
    else:
        result = "This Bank note is GENUINE!"
    
    return templates.TemplateResponse("prediction_form.html", {"request": request, "prediction": result})

# Run the API with uvicorn
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)
