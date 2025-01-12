# import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Define parameter grid
param_grid = {
    "n_estimators": [10, 50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
}

# Train the RandomForestClassifier model with the training dataset
model = RandomForestClassifier(random_state=42)
# Perform Grid Search
grid_search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", verbose=1)
grid_search.fit(x_train, y_train.values.ravel())

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# Save the model as joblib (serialized)
model_path = os.path.join('model', 'hyper_tuned_model.joblib')
joblib.dump(grid_search, model_path)