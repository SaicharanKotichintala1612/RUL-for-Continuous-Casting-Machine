from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib
import io

# Load the trained model
loaded_model = joblib.load('decision_tree_model.pkl')

# Function to preprocess new data
def preprocess_data(new_data, train_data_columns):
    # Convert categorical variables into numerical using one-hot encoding
    new_data_encoded = pd.get_dummies(new_data)
    # Align the columns of new_data_encoded with the columns of the training data
    new_data_processed = new_data_encoded.reindex(columns=train_data_columns, fill_value=0)
    return new_data_processed

# Function to make predictions
def predict_rul(new_data, train_data_columns):
    # Preprocess the new data
    new_data_processed = preprocess_data(new_data, train_data_columns)
    # Make predictions using the loaded model
    predictions = loaded_model.predict(new_data_processed)
    return predictions

# Define the FastAPI app
app = FastAPI()

# Load the initial training data to get the column names
initial_data = pd.read_csv("ccm_rul_dataset.csv")
initial_data.dropna(subset=['RUL'], inplace=True)
initial_data_encoded = pd.get_dummies(initial_data)
X_train_columns = initial_data_encoded.drop(columns=['RUL']).columns

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:8000",
    # Add more origins as needed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the endpoint for file upload and prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file
        content = await file.read()
        new_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
       
        # Make predictions
        predictions = predict_rul(new_data, X_train_columns)
       
        # Convert predictions to time format (if needed)
        time_unit = "hours"
        predictions_in_time = [pred / 24 if time_unit == "hours" else pred for pred in predictions]
       
        # Return predictions as JSON response
        result = {f"sample_{i+1}": pred_time for i, pred_time in enumerate(predictions_in_time)}
        return JSONResponse(content=result)
   
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

# Run the app with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)