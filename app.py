# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the Random Forest Classifier model
filename = 'heart-disease-prediction-knn-model.pkl'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = request.form.get('sex')
        cp = request.form.get('cp')
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = request.form.get('fbs')
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = request.form.get('exang')
        oldpeak = float(request.form['oldpeak'])
        slope = request.form.get('slope')
        ca = int(request.form['ca'])
        thal = request.form.get('thal')

        sex = 1 if sex.lower() == 'male' else 0
        fbs = 1 if fbs.lower() == 'true' else 0
        exang = 1 if exang.lower() == 'true' else 0
        slope = int(slope)
        thal = int(thal)

        data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        my_prediction = model.predict(data)

        return render_template('result.html', prediction=my_prediction)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if request.method == 'POST':
        data = request.get_json()

        # Extract data from the JSON payload
        age = int(data['age'])
        sex = 1 if data['sex'].lower() == 'male' else 0
        cp = int(data['cp'])
        trestbps = int(data['trestbps'])
        chol = int(data['chol'])
        fbs = 1 if data['fbs'].lower() == 'true' else 0
        restecg = int(data['restecg'])
        thalach = int(data['thalach'])
        exang = 1 if data['exang'].lower() == 'true' else 0
        oldpeak = float(data['oldpeak'])
        slope = int(data['slope'])
        ca = int(data['ca'])
        thal = int(data['thal'])

        data_array = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(data_array)

        # Return the prediction as JSON
        return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=False)
