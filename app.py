import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import joblib

with open(f'etr_model.pkl', 'rb') as f:
    model = joblib.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    
    features = [float(x) for x in request.form.values()]
    data = [np.array(features)]
    prediction = model.predict(data)

    output = round(prediction[0], 6)

    return render_template('home.html', prediction_text = "Motor Temperature = [ {} ]".format(output))

if __name__ == '__main__':
    app.run(debug=True)
