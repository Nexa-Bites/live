from flask import Flask, request, jsonify
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from deepface import DeepFace


app = Flask(__name__)

# Loading trained model
MODEL_PATH = "liveness_model.h5"
model = load_model(MODEL_PATH)

# Loading scaler used

import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Check if the scaler file exists
if os.path.exists("features.pkl"):
    with open("features.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Ensure it's a valid StandardScaler object
    if not isinstance(scaler, StandardScaler):
        print("Warning: features.pkl is not a valid StandardScaler object. Resetting scaler.")
        scaler = StandardScaler()
        scaler.fit(np.random.rand(10, 128))  # Dummy fit
else:
    print("Warning: features.pkl not found. Creating a new one.")
    scaler = StandardScaler()
    scaler.fit(np.random.rand(10, 128))  # Dummy data to fit the scaler

    # Save the new scaler for future use
    with open("features.pkl", "wb") as f:
        pickle.dump(scaler, f)




def preprocess_image(image_path):
    try:
        # Extract DeepFace embedding
        embedding = DeepFace.represent(image_path, model_name='Facenet', enforce_detection=False)[0]['embedding']
        embedding = np.array(embedding).reshape(1, -1)  # Reshape for model input
        embedding_scaled = scaler.transform(embedding)
        return embedding_scaled
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")  # Print actual error
        return None


#  API route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    image_path = "temp.jpg"
    file.save(image_path)

    # Preprocessing image
    embedding = preprocess_image(image_path)
    if embedding is None:
        return jsonify({"error": "Error processing image"}), 500

    # Predicting using model
    prediction = model.predict(embedding)[0][0]
    label = "Real" if prediction < 0.5 else "Spoof"

    return jsonify({"prediction": label, "confidence": float(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Evaluation script started...")

