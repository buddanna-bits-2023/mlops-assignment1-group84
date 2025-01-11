#import the required libraries
import joblib
import os

#Load the model using joblib
model_path = os.path.join('model', 'model.joblib')
my_model = joblib.load(model_path)

#Test model loaded or not
def test_model_load():
    assert my_model is not None, "Failed to load the saved model."

#Test the prediction for two values
def test_predict():
    assert my_model.predict([[6.8,2.8,4.8,1.4]]) == 1