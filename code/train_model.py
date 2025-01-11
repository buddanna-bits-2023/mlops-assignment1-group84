# import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load IRIS dataset and extract features
data = load_iris()
x = pd.DataFrame(data=data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=['target'])

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Check the dataset dimensions
x.shape

# Train the RandomForestClassifier model with the training dataset
model = RandomForestClassifier()
model.fit(x_train, y_train.values.ravel())

# Test the model with the test data
model.predict(x_test)

# Save the model as joblib (serialized)
model_path = os.path.join('model', 'model.joblib')
joblib.dump(model, model_path)
