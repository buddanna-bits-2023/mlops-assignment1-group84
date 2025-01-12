#import required libraries
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

#Create Flask API instance
app = Flask(__name__)

#Load the best performance ML model from the joblib file
model_path = os.path.join('model', 'hyper_tuned_model.joblib')
model = joblib.load(model_path)

#Default route on launching the Flask API
@app.route('/', methods=['GET'])
def default():
    return 'Flask API running successfully.'

#API POST method/route for prediction of given inputs
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(np.array(data['input']).reshape(1,-1))
    return jsonify({'prediction': prediction.tolist()})

#Starting the Flask API
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')