# import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

# Load IRIS dataset and extract features
data = load_iris()
X = data['data']
y = data['target']

# Split the data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the dataset dimensions
X.shape

# Experiment Name
experiment_name = "MLflow Tracking"
mlflow.set_experiment(experiment_name)

# Model configurations to test
configs = [
    {"n_estimators": 10, "max_depth": 5, "random_state": 42},
    {"n_estimators": 50, "max_depth": 10, "random_state": 48},
    {"n_estimators": 100, "max_depth": None, "random_state": 40},
]

for config in configs:
    run_name_str = 'rf-default_'+str(config["n_estimators"])
    with mlflow.start_run(run_name=run_name_str):
        # Train the RandomForestClassifier model with the training dataset
        model = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"], random_state=config["random_state"])
        model.fit(x_train, y_train)

        # Test the model with the test data
        y_pred = model.predict(x_test)

        # Calculate Accuracy, MSE & R2 score metrics
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test,y_pred)
        r2 =r2_score(y_test,y_pred)
    
         # Log parameters and metrics
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("n_estimators", config["n_estimators"])
        mlflow.log_param("random_state", config["random_state"])

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2 score", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, 'rf-default')

        model_path = os.path.join('mlruns', 'models', run_name_str +'.joblib')
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        
        print(f"Logged run with params={config} and accuracy={accuracy:.4f}")
