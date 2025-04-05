from flask import Flask, request, jsonify, render_template
import joblib
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load models
svm_model = joblib.load("svm_model.pkl")
rf_model = joblib.load("rf_model.pkl")
log_reg_model = joblib.load("log_reg_model.pkl")
cnn_model = load_model("cnn_model.h5")

# Preprocessing function
def preprocess_image(file):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64)) / 255.0
    return img

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['image']
    model_name = request.form['model']
    img = preprocess_image(file)
    
    if model_name == 'svm':
        prediction = svm_model.predict([img.flatten()])  # SVM expects 1D input
    elif model_name == 'rf':
        prediction = rf_model.predict([img.flatten()])  # Random Forest expects 1D input
    elif model_name == 'log_reg':
        prediction = log_reg_model.predict([img.flatten()])  # Logistic Regression expects 1D input
    elif model_name == 'cnn':
        img = img.reshape(1, 64, 64, 3)  # Reshape for CNN input
        prediction = cnn_model.predict(img)  # CNN output is a probability distribution
        prediction = np.argmax(prediction, axis=1)  # Convert to class index
    else:
        return jsonify({"error": "Invalid model selected"}), 400

    
    result = "Dog" if int(prediction[0]) == 1 else "Cat"
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
