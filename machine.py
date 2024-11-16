import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib 

# Load the dataset
machine = pd.read_csv("/Users/saicharankotichintala/Documents/Projects/RUL-for-Continuous-Casting-Machine-main/ccm_rul_dataset.csv")

# Drop rows with NaN values in the target variable 'RUL'
machine.dropna(subset=['RUL'], inplace=True)

# Convert categorical variables into numerical using one-hot encoding
machine_encoded = pd.get_dummies(machine)

# Split the dataset into features (X) and target variable (y)
X = machine_encoded.drop(columns=['RUL'])
y = machine_encoded['RUL']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree model
decision_tree_model = DecisionTreeRegressor(random_state=42)
decision_tree_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(decision_tree_model, 'decision_tree_model.pkl')

# Load the saved model
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

# Example usage:
# Load new data
new_data = pd.read_csv("/Users/saicharankotichintala/Documents/Projects/RUL-for-Continuous-Casting-Machine-main/ccm_rul_dataset.csv")
# Make predictions
predictions = predict_rul(new_data, X_train.columns)
print(predictions)

# Print predictions
for i, pred in enumerate(predictions):
    print(f"Prediction for sample {i + 1}: {pred}")
    
# Define the time unit used in the dataset (e.g., hours)
time_unit = "hours"

# Convert the numerical predictions to time format
predictions_in_time = [pred / 24 if time_unit == "hours" else pred for pred in predictions]

# Print predictions in time format
for i, pred_time in enumerate(predictions_in_time):
    print(f"Prediction for sample {i + 1}: {pred_time} days")

